"""Recherche de passages : dense, lexical, hybride, + reranking optionnel.

Toutes les strategies renvoient la meme structure `Context`, pour que les
metriques de retrieval soient calculables sans savoir comment le passage a
ete trouve. C'est ce qui permet de comparer dense et lexical sur le meme
recall@k.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg

from ..config import PipelineConfig
from ..llm import LLMClient
from ..settings import settings


@dataclass
class Context:
    chunk_id: int
    document_id: int
    document_external_id: str
    text: str
    score: float
    rank: int
    # Provenance : 'dense' | 'lexical' | 'rrf' | 'rerank' | 'cross_encoder'. Utile pour
    # diagnostiquer un mode hybride qui n'apporte rien.
    source: str = "dense"

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_external_id": self.document_external_id,
            "text": self.text,
            "score": round(self.score, 6),
            "rank": self.rank,
            "source": self.source,
        }


class RerankFailure(RuntimeError):
    """Le reranker n'a produit aucun score exploitable.

    Levee plutot qu'avalee : un reranking qui echoue en silence produit
    exactement les memes chiffres qu'un reranking inutile, et on conclut
    « ca n'apporte rien » alors que ca n'a jamais tourne.
    """


@dataclass
class RetrievalResult:
    contexts: list[Context]
    elapsed_ms: int
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Strategies elementaires
# ---------------------------------------------------------------------


def _dense(
    conn: psycopg.Connection, index_id: int, embedding: list[float], k: int
) -> list[Context]:
    # Parcours exact : pas d'index ANN sur `chunks.embedding` (cf. schema.sql).
    # L'operateur <=> de pgvector est une distance cosinus, donc plus petit
    # = plus proche ; on renvoie une similarite pour rester homogene avec
    # les autres scorers, ou plus grand = meilleur.
    rows = conn.execute(
        """
        SELECT c.id, c.document_id, d.external_id, c.text,
               1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.index_id = %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (embedding, index_id, embedding, k),
    ).fetchall()
    return [
        Context(r["id"], r["document_id"], r["external_id"], r["text"], float(r["score"]), i + 1, "dense")
        for i, r in enumerate(rows)
    ]


def _lexical(conn: psycopg.Connection, index_id: int, query: str, k: int) -> list[Context]:
    # La requete est convertie en OU de ses lexemes, pas en ET.
    #
    # plainto_tsquery() assemble les termes avec AND : sur les questions de
    # MultiHop-RAG, longues de 30 a 60 mots, AUCUN passage ne contient tous
    # les termes et la recherche renvoie systematiquement zero resultat.
    # Le symptome est silencieux — pas d'erreur, juste un retrieval vide et
    # une abstention sur chaque question.
    #
    # tsvector_to_array(to_tsvector(...)) donne les lexemes normalises
    # (racinises, mots-outils retires) qu'on rejoint par ' | '. ts_rank_cd
    # classe ensuite par densite de couverture, donc les passages qui
    # contiennent le plus de termes de la question ressortent en tete.
    rows = conn.execute(
        """
        WITH lexemes AS (
            SELECT unnest(tsvector_to_array(to_tsvector('english', %s))) AS lexeme
        ), q AS (
            -- quote_literal sur chaque lexeme : certains contiennent une
            -- apostrophe ou un tiret et feraient echouer to_tsquery bruts.
            -- NULLIF renvoie une tsquery vide plutot qu'une erreur quand la
            -- question ne contient que des mots-outils.
            SELECT to_tsquery(
                'english',
                NULLIF(string_agg(quote_literal(lexeme), ' | '), '')
            ) AS tsq
            FROM lexemes
        )
        SELECT c.id, c.document_id, d.external_id, c.text,
               ts_rank_cd(c.tsv, q.tsq) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN q
        WHERE c.index_id = %s AND c.tsv @@ q.tsq
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, index_id, k),
    ).fetchall()
    return [
        Context(r["id"], r["document_id"], r["external_id"], r["text"], float(r["score"]), i + 1, "lexical")
        for i, r in enumerate(rows)
    ]


def _cap_per_document(candidates: list[Context], max_per_doc: int) -> list[Context]:
    """Limite le nombre de chunks conserves par document, en gardant les
    mieux classes. L'ordre relatif des survivants est preserve."""
    counts: dict[str, int] = {}
    kept: list[Context] = []
    for ctx in candidates:
        doc = ctx.document_external_id
        if counts.get(doc, 0) >= max_per_doc:
            continue
        counts[doc] = counts.get(doc, 0) + 1
        kept.append(ctx)
    return kept


def _rrf(runs: list[list[Context]], k_const: int, top_k: int) -> list[Context]:
    """Reciprocal Rank Fusion.

    Choisi plutot qu'une somme ponderee de scores parce que les scores
    dense (cosinus, ~0-1) et lexical (ts_rank_cd, non borne) ne sont pas
    sur la meme echelle : les additionner reviendrait a laisser l'echelle
    decider du poids. RRF ne regarde que les rangs, donc rien a calibrer.
    """
    fused: dict[int, float] = {}
    by_id: dict[int, Context] = {}
    for run in runs:
        for ctx in run:
            fused[ctx.chunk_id] = fused.get(ctx.chunk_id, 0.0) + 1.0 / (k_const + ctx.rank)
            by_id.setdefault(ctx.chunk_id, ctx)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out: list[Context] = []
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        base = by_id[chunk_id]
        out.append(
            Context(base.chunk_id, base.document_id, base.document_external_id,
                    base.text, score, rank, "rrf")
        )
    return out


# ---------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------

_RERANK_PROMPT = """You score how useful a passage is for answering a question.

Question: {question}

Passage:
{passage}

Reply with a single integer from 0 to 3:
0 = irrelevant
1 = related topic but does not help answer
2 = partially answers
3 = directly answers

Score:"""


async def _llm_rerank(
    llm: LLMClient, cfg: PipelineConfig, question: str, candidates: list[Context]
) -> list[Context]:
    """Rerank pointwise par le LLM.

    Pointwise et non listwise : c'est le seul schema ou le cout croit
    lineairement avec fetch_k et ou l'ordre de presentation ne biaise pas
    le resultat (le biais de position des juges LLM est documente). Note
    que le reranker consomme fetch_k appels LLM par question — c'est
    souvent le poste de cout dominant d'une campagne.
    """
    prompts = [
        _RERANK_PROMPT.format(question=question, passage=c.text[:2000]) for c in candidates
    ]
    completions = await llm.complete_many(
        prompts,
        model=cfg.models.judge,
        role="reranker",
        temperature=0.0,
        max_tokens=4,
        # thinking=False est OBLIGATOIRE ici, pas une optimisation.
        # Avec le raisonnement actif, les 4 tokens de budget partent
        # entierement dans la reflexion et le modele renvoie une chaine
        # VIDE. Tous les candidats recoivent alors le score 0, le tri est
        # stable, et le reranking devient un no-op silencieux qui coute
        # fetch_k appels LLM par question sans rien changer au classement.
        thinking=False,
    )

    scored: list[tuple[float, Context]] = []
    n_unparsed = 0
    for ctx, comp in zip(candidates, completions, strict=True):
        digits = [ch for ch in comp.text if ch.isdigit()] if comp else []
        if not digits:
            # Echec du reranker : on retombe sur le rang d'origine plutot
            # que d'ejecter le passage, pour ne pas confondre "mauvais
            # passage" et "juge en echec". Le score negatif place ces
            # passages apres tous les passages notes, en preservant leur
            # ordre relatif.
            n_unparsed += 1
            scored.append((-1.0 / ctx.rank, ctx))
            continue
        scored.append((float(digits[0]), ctx))

    if n_unparsed == len(candidates):
        # Aucun verdict exploitable : le reranking n'a rien reordonne. Le
        # signaler evite de conclure « le reranking n'apporte rien » alors
        # qu'il n'a simplement pas fonctionne.
        raise RerankFailure(
            f"le reranker n'a produit aucun score exploitable sur "
            f"{len(candidates)} passages (modele {cfg.models.judge})"
        )

    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        Context(c.chunk_id, c.document_id, c.document_external_id, c.text, s, i + 1, "rerank")
        for i, (s, c) in enumerate(scored)
    ]


# Cache de telechargement des poids. Chemin relatif au repertoire de travail :
# la racine du depot en local, /app dans l'image, ou data/ est monte dans les
# deux cas. N'affecte aucun score, donc hors PipelineConfig.
CROSS_ENCODER_CACHE = Path("data/cache/rerankers")

_cross_encoders: dict[str, Any] = {}
# Une seule inference a la fois : le runtime parallelise deja chaque appel sur
# les coeurs disponibles. Laisser les workers de campagne l'appeler en
# concurrence sursouscrirait le CPU, et sur un processeur hybride cette
# sursouscription divise le debit au lieu de le multiplier.
_cross_encoder_lock = threading.Lock()


def _load_cross_encoder(spec: str) -> Any:
    """Charge (une fois par processus) un reranker de la bibliotheque rerankers."""
    if spec in _cross_encoders:
        return _cross_encoders[spec]
    backend, _, name = spec.partition(":")
    try:
        from rerankers import Reranker
    except ImportError as exc:  # pragma: no cover - depend de l'extra installe
        raise RerankFailure(
            "rerank=cross_encoder demande l'extra rerank : uv sync --extra rerank"
        ) from exc
    CROSS_ENCODER_CACHE.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"model_type": backend, "verbose": 0}
    if backend == "flashrank":
        kwargs["cache_dir"] = str(CROSS_ENCODER_CACHE)
    ranker = Reranker(name, **kwargs)
    _limit_onnx_threads(ranker, settings.rerank_threads)
    _cross_encoders[spec] = ranker
    return ranker


def _limit_onnx_threads(ranker: Any, n_threads: int) -> None:
    """Recree la session ONNX avec un nombre de threads borne.

    FlashRank ouvre sa session avec les options par defaut, sans moyen de les
    passer : on la reconstruit sur le meme fichier de poids. Les autres
    backends n'ont pas d'attribut `session` et sont laisses tels quels.
    """
    inner = getattr(ranker, "model", None)
    session = getattr(inner, "session", None)
    model_path = getattr(session, "_model_path", None)
    if not model_path:
        return
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = n_threads
    options.inter_op_num_threads = 1
    inner.session = ort.InferenceSession(model_path, sess_options=options)


def _cross_encoder_scores(spec: str, question: str, passages: list[str]) -> list[float]:
    ranker = _load_cross_encoder(spec)
    with _cross_encoder_lock:
        ranked = ranker.rank(query=question, docs=passages, doc_ids=list(range(len(passages))))
    scores = [float("nan")] * len(passages)
    for result in ranked.results:
        scores[result.document.doc_id] = float(result.score)
    return scores


def order_by_scores(candidates: list[Context], scores: list[float], source: str) -> list[Context]:
    """Reordonne par score decroissant ; a egalite, le rang d'origine departage.

    Le departage par rang garde le tri deterministe : sans lui, deux passages
    de meme score pourraient s'inverser d'un run a l'autre et deplacer un
    recall@k sans que rien n'ait change.
    """
    if len(scores) != len(candidates) or any(s != s for s in scores):  # NaN
        raise RerankFailure(
            f"le cross-encoder a renvoye {len(scores)} scores pour {len(candidates)} passages"
        )
    ordered = sorted(zip(scores, candidates, strict=True), key=lambda t: (-t[0], t[1].rank))
    return [
        Context(c.chunk_id, c.document_id, c.document_external_id, c.text, s, i + 1, source)
        for i, (s, c) in enumerate(ordered)
    ]


async def _cross_encoder_rerank(
    cfg: PipelineConfig, question: str, candidates: list[Context]
) -> list[Context]:
    """Rerank par cross-encoder : un seul appel pour tout le vivier.

    La question brute est utilisee, pas la requete reecrite : le reranker juge
    la pertinence pour ce que l'utilisateur a demande.
    """
    spec = cfg.retrieval.rerank_model
    assert spec is not None  # garanti par RetrievalConfig
    scores = await asyncio.to_thread(
        _cross_encoder_scores, spec, question, [c.text for c in candidates]
    )
    return order_by_scores(candidates, scores, "cross_encoder")


# ---------------------------------------------------------------------
# Reecriture de requete
# ---------------------------------------------------------------------

_REWRITE_PROMPT = """Rewrite the question below as a concise search query.
Keep every proper noun, date and number. Drop conversational filler.
Reply with the query only, no explanation.

Question: {question}
Query:"""

_DECOMPOSE_PROMPT = """Break the question below into at most {n} short, standalone search
queries. Each query must target ONE fact or ONE article. Keep every proper
noun, news source name and date from the question.
Reply with the queries only, one per line, no numbering, no explanation.

Question: {question}
Queries:"""

_HYDE_PROMPT = """Write a short news passage (3 to 4 sentences) that would directly answer
the question below, in the style of a press article. Invent plausible
details if needed. Reply with the passage only.

Question: {question}
Passage:"""

_CLARIFY_PROMPT = """A user asked a search assistant the question below. These are the
articles the first search found (source, date, title):
{candidates}

If the question clearly identifies which articles are needed, reply exactly: NONE
Otherwise, ask the user ONE short clarifying question that would help pick the
right articles (for example which source, date or event they mean).
Reply with NONE or with the question only.

Question: {question}
Reply:"""

# Une clarification sans utilisateur pour y repondre serait un no-op
# silencieux : la reponse est fournie par l'appelant (humain en production,
# simule en evaluation, cf. runner/simulated_user.py).
UserReply = Callable[[str], Awaitable[str]]

_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def parse_sub_queries(text: str, max_n: int) -> list[str]:
    """Extrait les sous-requetes d'une reponse du LLM, une par ligne.

    Tolere puces et numerotation, que les petits modeles ajoutent malgre la
    consigne, et ecarte les lignes vides et les doublons.
    """
    out: list[str] = []
    for line in text.splitlines():
        query = _LIST_MARKER.sub("", line).strip().strip('"').strip()
        if query and query.lower() not in {q.lower() for q in out}:
            out.append(query)
        if len(out) >= max_n:
            break
    return out


async def _ask(
    llm: LLMClient, cfg: PipelineConfig, prompt: str, *, role: str, max_tokens: int
) -> str:
    """Appel court au generateur pour une technique de requete.

    thinking=False est OBLIGATOIRE : avec le raisonnement actif, un petit
    budget part entierement dans la reflexion et le texte revient vide. La
    technique retombe alors sur la question d'origine et devient un no-op
    silencieux — elle couterait un appel LLM sans rien changer.
    """
    comp = await llm.complete(
        prompt,
        model=cfg.models.generator,
        role=role,
        temperature=0.0,
        max_tokens=max_tokens,
        thinking=False,
    )
    return (comp.text or "").strip()


async def rewrite_query(llm: LLMClient, cfg: PipelineConfig, question: str) -> str | None:
    """Requete reecrite, ou None si la reecriture a echoue (repli trace)."""
    try:
        text = await _ask(llm, cfg, _REWRITE_PROMPT.format(question=question),
                          role="rewriter", max_tokens=64)
        return text or None
    except Exception:  # noqa: BLE001 - optimisation, jamais un point de panne
        return None


async def decompose_query(llm: LLMClient, cfg: PipelineConfig, question: str) -> list[str]:
    try:
        text = await _ask(
            llm, cfg,
            _DECOMPOSE_PROMPT.format(n=cfg.retrieval.max_sub_queries, question=question),
            role="decomposer", max_tokens=160,
        )
    except Exception:  # noqa: BLE001
        return []
    return parse_sub_queries(text, cfg.retrieval.max_sub_queries)


def candidate_lines(contexts: list[Context], limit: int = 8) -> str:
    """Resume des candidats par leur en-tete : source, date, titre.

    L'en-tete contextuel est lu dans le texte du passage (« Title: … »),
    pour ne pas dependre d'une requete de plus en base.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for ctx in contexts:
        if ctx.document_external_id in seen:
            continue
        seen.add(ctx.document_external_id)
        fields = dict(
            line.split(": ", 1) for line in ctx.text.splitlines()[:4] if ": " in line
        )
        date = fields.get("Published", "")[:10]
        lines.append(f"- {fields.get('Source', '?')}, {date}: {fields.get('Title', '?')}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def wants_clarification(reply: str) -> bool:
    """Le modele demande-t-il une precision ? NONE, vide ou sans « ? » : non."""
    text = reply.strip()
    return bool(text) and not text.upper().startswith("NONE") and "?" in text


async def clarifying_question(
    llm: LLMClient, cfg: PipelineConfig, question: str, candidates: list[Context]
) -> str | None:
    try:
        text = await _ask(
            llm, cfg,
            _CLARIFY_PROMPT.format(candidates=candidate_lines(candidates), question=question),
            role="clarifier", max_tokens=60,
        )
    except Exception:  # noqa: BLE001
        return None
    return text if wants_clarification(text) else None


async def hypothetical_document(llm: LLMClient, cfg: PipelineConfig, question: str) -> str | None:
    try:
        text = await _ask(llm, cfg, _HYDE_PROMPT.format(question=question),
                          role="hyde", max_tokens=200)
        return text or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------


async def _search(
    conn: psycopg.Connection,
    llm: LLMClient,
    cfg: PipelineConfig,
    *,
    index_id: int,
    text: str,
    k: int,
    dense_text: str | None = None,
    debug: dict[str, Any] | None = None,
) -> list[Context]:
    """Une recherche elementaire selon le mode configure.

    dense_text remplace le texte vectorise (HyDE) : c'est alors un passage,
    pas une question, et il recoit le prefixe DOCUMENT de l'embedder. La
    recherche lexicale garde toujours `text`.
    """
    rr = cfg.retrieval
    embedding: list[float] | None = None
    if rr.mode in ("dense", "hybrid"):
        to_embed = (
            f"{cfg.chunking.document_prefix}{dense_text}" if dense_text
            else f"{rr.query_prefix}{text}"
        )
        embedding = await llm.embed_one(to_embed, model=cfg.models.embedder)

    match rr.mode:
        case "dense":
            return _dense(conn, index_id, embedding, k)  # type: ignore[arg-type]
        case "lexical":
            return _lexical(conn, index_id, text, k)
        case "hybrid":
            dense_hits = _dense(conn, index_id, embedding, k)  # type: ignore[arg-type]
            lexical_hits = _lexical(conn, index_id, text, k)
            if debug is not None:
                debug["n_dense"] = debug.get("n_dense", 0) + len(dense_hits)
                debug["n_lexical"] = debug.get("n_lexical", 0) + len(lexical_hits)
            return _rrf([dense_hits, lexical_hits], rr.rrf_k, k)
        case _:  # pragma: no cover
            raise ValueError(f"mode de retrieval inconnu : {rr.mode}")


async def retrieve(
    conn: psycopg.Connection,
    llm: LLMClient,
    cfg: PipelineConfig,
    *,
    index_id: int,
    question: str,
    user: UserReply | None = None,
) -> RetrievalResult:
    started = time.perf_counter()
    debug: dict[str, Any] = {}
    rr = cfg.retrieval
    if rr.clarify and user is None:
        raise ValueError(
            "clarify=true exige un utilisateur pour repondre (reel, ou simule en evaluation)"
        )

    search_text = question
    if rr.query_rewrite:
        rewritten = await rewrite_query(llm, cfg, question)
        # Le repli est trace : une reecriture qui echoue toujours imiterait
        # une reecriture inutile.
        debug["rewrite_fallback"] = rewritten is None
        search_text = rewritten or question
        debug["rewritten_query"] = search_text

    dense_text: str | None = None
    if rr.hyde:
        dense_text = await hypothetical_document(llm, cfg, question)
        debug["hyde_fallback"] = dense_text is None
        if dense_text:
            debug["hyde_passage"] = dense_text[:500]

    # On remonte un vivier plus large que top_k des qu'une etape doit
    # ecarter des candidats : reranking ou plafond par document. Sans
    # vivier, ecarter un chunk reduirait simplement le nombre de passages
    # au lieu de le remplacer.
    needs_pool = rr.rerank != "none" or rr.max_per_document is not None
    pool_k = rr.fetch_k if needs_pool else rr.top_k

    candidates = await _search(
        conn, llm, cfg, index_id=index_id, text=search_text, k=pool_k,
        dense_text=dense_text, debug=debug,
    )

    if rr.clarify and candidates:
        asked = await clarifying_question(llm, cfg, question, candidates)
        debug["clarify_asked"] = asked is not None
        if asked is not None:
            reply = await user(asked)  # type: ignore[misc]  # garanti plus haut
            debug["clarify_question"] = asked
            debug["clarify_reply"] = reply
            if reply:
                # La question enrichie de la reponse est cherchee a nouveau ;
                # les deux classements sont fusionnes, pour qu'une reponse
                # hors sujet degrade sans detruire.
                second = await _search(
                    conn, llm, cfg, index_id=index_id, text=f"{question} {reply}", k=pool_k
                )
                candidates = _rrf([candidates, second], rr.rrf_k, pool_k)

    if rr.query_decompose:
        sub_queries = await decompose_query(llm, cfg, question)
        debug["sub_queries"] = sub_queries
        debug["decompose_fallback"] = not sub_queries
        runs = [candidates]
        for sub in sub_queries:
            runs.append(await _search(conn, llm, cfg, index_id=index_id, text=sub, k=pool_k))
        if len(runs) > 1:
            # La question d'origine reste une des listes fusionnees : une
            # decomposition ratee degrade au lieu de detruire le classement.
            candidates = _rrf(runs, rr.rrf_k, pool_k)

    if rr.rerank == "llm" and candidates:
        candidates = await _llm_rerank(llm, cfg, question, candidates)
    elif rr.rerank == "cross_encoder" and candidates:
        rerank_started = time.perf_counter()
        candidates = await _cross_encoder_rerank(cfg, question, candidates)
        debug["rerank_ms"] = int((time.perf_counter() - rerank_started) * 1000)

    if rr.max_per_document is not None:
        before = len(candidates)
        candidates = _cap_per_document(candidates, rr.max_per_document)
        debug["dropped_by_document_cap"] = before - len(candidates)
        debug["distinct_documents"] = len({c.document_external_id for c in candidates[: rr.top_k]})

    contexts = candidates[: rr.top_k]

    if rr.similarity_threshold is not None and rr.mode == "dense":
        # Seuil applique apres coup : on veut pouvoir compter combien de
        # passages il elimine, ce qui est le mecanisme du negative rejection.
        kept = [c for c in contexts if c.score >= rr.similarity_threshold]
        debug["dropped_by_threshold"] = len(contexts) - len(kept)
        contexts = kept

    for i, ctx in enumerate(contexts, start=1):
        ctx.rank = i

    return RetrievalResult(
        contexts=contexts,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        debug=debug,
    )

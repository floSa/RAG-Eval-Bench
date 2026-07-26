"""Recherche de passages : dense, lexical, hybride, + reranking optionnel.

Toutes les strategies renvoient la meme structure `Context`, pour que les
metriques de retrieval soient calculables sans savoir comment le passage a
ete trouve. C'est ce qui permet de comparer dense et lexical sur le meme
recall@k.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from ..config import PipelineConfig
from ..llm import LLMClient


@dataclass
class Context:
    chunk_id: int
    document_id: int
    document_external_id: str
    text: str
    score: float
    rank: int
    # Provenance : 'dense' | 'lexical' | 'rrf' | 'rerank'. Utile pour
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
    rows = conn.execute(
        """
        SELECT c.id, c.document_id, d.external_id, c.text,
               ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.index_id = %s
          AND c.tsv @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, index_id, query, k),
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
    )

    scored: list[tuple[float, Context]] = []
    for ctx, comp in zip(candidates, completions):
        if comp is None:
            # Echec du reranker : on retombe sur le rang d'origine plutot
            # que d'ejecter le passage, pour ne pas confondre "mauvais
            # passage" et "juge en echec".
            scored.append((-1.0 / ctx.rank, ctx))
            continue
        digits = [ch for ch in comp.text if ch.isdigit()]
        scored.append((float(digits[0]) if digits else 0.0, ctx))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        Context(c.chunk_id, c.document_id, c.document_external_id, c.text, s, i + 1, "rerank")
        for i, (s, c) in enumerate(scored)
    ]


# ---------------------------------------------------------------------
# Reecriture de requete
# ---------------------------------------------------------------------

_REWRITE_PROMPT = """Rewrite the question below as a concise search query.
Keep every proper noun, date and number. Drop conversational filler.
Reply with the query only, no explanation.

Question: {question}
Query:"""


async def rewrite_query(llm: LLMClient, cfg: PipelineConfig, question: str) -> str:
    try:
        comp = await llm.complete(
            _REWRITE_PROMPT.format(question=question),
            model=cfg.models.generator,
            role="rewriter",
            temperature=0.0,
            max_tokens=64,
        )
        return comp.text or question
    except Exception:
        return question  # la reecriture est une optimisation, jamais un point de panne


# ---------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------


async def retrieve(
    conn: psycopg.Connection,
    llm: LLMClient,
    cfg: PipelineConfig,
    *,
    index_id: int,
    question: str,
) -> RetrievalResult:
    started = time.perf_counter()
    debug: dict[str, Any] = {}

    search_text = question
    if cfg.retrieval.query_rewrite:
        search_text = await rewrite_query(llm, cfg, question)
        debug["rewritten_query"] = search_text

    rr = cfg.retrieval
    # On remonte un vivier plus large que top_k des qu'une etape doit
    # ecarter des candidats : reranking ou plafond par document. Sans
    # vivier, ecarter un chunk reduirait simplement le nombre de passages
    # au lieu de le remplacer.
    needs_pool = rr.rerank != "none" or rr.max_per_document is not None
    pool_k = rr.fetch_k if needs_pool else rr.top_k

    needs_embedding = rr.mode in ("dense", "hybrid")
    embedding: list[float] | None = None
    if needs_embedding:
        embedding = await llm.embed_one(
            f"{rr.query_prefix}{search_text}", model=cfg.models.embedder
        )

    match rr.mode:
        case "dense":
            candidates = _dense(conn, index_id, embedding, pool_k)  # type: ignore[arg-type]
        case "lexical":
            candidates = _lexical(conn, index_id, search_text, pool_k)
        case "hybrid":
            dense_hits = _dense(conn, index_id, embedding, pool_k)  # type: ignore[arg-type]
            lexical_hits = _lexical(conn, index_id, search_text, pool_k)
            debug["n_dense"] = len(dense_hits)
            debug["n_lexical"] = len(lexical_hits)
            candidates = _rrf([dense_hits, lexical_hits], rr.rrf_k, pool_k)
        case _:  # pragma: no cover
            raise ValueError(f"mode de retrieval inconnu : {rr.mode}")

    if rr.rerank == "llm" and candidates:
        candidates = await _llm_rerank(llm, cfg, question, candidates)

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

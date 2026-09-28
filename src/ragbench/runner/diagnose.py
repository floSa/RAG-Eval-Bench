"""Diagnostics de retrieval, sans generation.

`recall_curve` repond a une question prealable a tout le catalogue de
techniques : le plafond du systeme est-il la QUALITE du retrieval, ou
simplement le NOMBRE de passages donnes au generateur ?

Mesure : on remonte une seule fois les max(ks) meilleurs passages par
question, puis on lit chaque prefixe de longueur k. Si la couverture factuelle
monte vite avec k, les bons passages existent mais arrivent trop bas : il
faut agrandir top_k ou reordonner (reranking). Si elle stagne, les bons
passages ne sont pas remontes du tout : il faut changer la recherche elle-meme
(decomposition, hybride, embeddings), et aucun reranker n'y changera rien.

k compte des PASSAGES, pas des documents : c'est l'unite de top_k, donc de ce
que le generateur recoit reellement. native.ir compte en documents
dedupliques ; les deux lectures se completent.

`compare_curves` apparie deux courbes question par question : c'est le
verdict des techniques de retrieval (reranking, hybride, embeddings) sans
payer une campagne de generation. Il ne dit rien de la reponse finale, seulement
de ce que le generateur aurait eu sous les yeux.

Aucun appel LLM hors embedding de la question : le diagnostic coute un index
et quelques secondes par question, pas une campagne.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .. import db
from ..config import PipelineConfig
from ..evaluators.native_ir import _gold_documents, _retrieved_documents
from ..evaluators.native_nuggets import lexical_support
from ..llm import LLMClient
from ..rag.ingest import ensure_index
from ..rag.retrieve import retrieve
from ..settings import Settings
from ..settings import settings as default_settings
from ..stats import Comparison, Interval, bootstrap_ci, holm, paired_bootstrap
from .campaign import load_questions

DEFAULT_KS: tuple[int, ...] = (5, 10, 20, 30, 50)

CURVE_METRICS: tuple[str, ...] = (
    "doc_recall",
    "hit_rate",
    "nugget_recall",
    "nugget_full_coverage",
)


def question_curve(
    contexts: list[dict[str, Any]],
    gold_evidence: list[dict[str, Any]],
    ks: Iterable[int],
    *,
    nugget_threshold: float = 0.6,
) -> dict[int, dict[str, float | None]]:
    """Metriques d'une question pour chaque prefixe de k passages.

    Les metriques de pepites valent None quand la question ne porte aucun
    fait de reference : indefini, pas nul — le compter comme 0 ferait
    baisser la courbe pour une raison qui n'a rien a voir avec le retrieval.
    """
    gold_docs = _gold_documents({"gold_evidence": gold_evidence})
    facts = [ev["fact"] for ev in gold_evidence if ev.get("fact")]

    out: dict[int, dict[str, float | None]] = {}
    for k in ks:
        prefix = contexts[:k]
        docs = _retrieved_documents({"contexts": prefix})
        found = [d for d in docs if d in gold_docs]
        row: dict[str, float | None] = {
            "doc_recall": len(found) / len(gold_docs) if gold_docs else None,
            "hit_rate": (1.0 if found else 0.0) if gold_docs else None,
            "nugget_recall": None,
            "nugget_full_coverage": None,
        }
        if facts:
            passages = [c["text"] for c in prefix]
            supported = [lexical_support(f, passages, nugget_threshold)[0] for f in facts]
            recall = sum(supported) / len(facts)
            row["nugget_recall"] = recall
            row["nugget_full_coverage"] = 1.0 if recall >= 0.999 else 0.0
        out[k] = row
    return out


def pool_config(cfg: PipelineConfig, max_k: int) -> tuple[PipelineConfig, list[str]]:
    """Config de retrieval qui remonte max_k passages, et ce qu'elle approxime.

    Lire un prefixe de longueur k n'equivaut exactement a un run top_k=k que
    si le classement ne depend pas de la taille du vivier. C'est vrai pour le
    dense et le lexical, et le plafond par document preserve les prefixes.
    Ca ne l'est plus pour RRF ni pour le reranking : les notes renvoyees le
    disent, pour que la courbe ne soit pas lue comme une serie de runs.
    """
    rr = cfg.retrieval
    notes: list[str] = []
    fetch_k = rr.fetch_k
    if rr.max_per_document is not None and rr.rerank == "none":
        # Le plafond ecarte des passages : il faut un vivier plus large pour
        # en garder encore max_k apres filtrage.
        fetch_k = max(fetch_k, 4 * max_k)
        # Mesure sur la baseline (fetch_k=20, plafond 2) : 6 questions sur 200
        # recoivent moins de top_k passages en campagne, faute de vivier.
        notes.append(
            f"plafond par document : vivier porte a {fetch_k} ; en campagne, fetch_k="
            f"{rr.fetch_k} peut s'epuiser et donner moins de top_k passages — la "
            "courbe peut etre legerement optimiste"
        )
    elif rr.rerank != "none":
        fetch_k = max(fetch_k, max_k)
        notes.append(
            f"rerank={rr.rerank} : vivier porte a {fetch_k}, le reranker note plus de "
            "candidats qu'en campagne — la courbe n'est qu'indicative"
        )
    if rr.mode == "hybrid":
        notes.append(
            "mode hybride : RRF fusionne des listes de longueur max(ks), le prefixe k "
            "peut differer legerement d'un run top_k=k"
        )
    if rr.similarity_threshold is not None:
        notes.append(
            f"seuil de similarite {rr.similarity_threshold} actif : certains prefixes "
            "comptent moins de k passages"
        )
    tuned = cfg.model_copy(
        update={"retrieval": rr.model_copy(update={"top_k": max_k, "fetch_k": fetch_k})}
    )
    return tuned, notes


@dataclass
class CurveReport:
    config_name: str
    config_hash: str
    index_hash: str
    ks: list[int]
    n_questions: int
    n_failed: int
    # metric -> k -> intervalle bootstrap
    curve: dict[str, dict[int, Interval]]
    notes: list[str] = field(default_factory=list)
    # question_id -> k -> metrique -> valeur. Garde pour l'appariement ;
    # hors du JSON, qui ne publie que la courbe.
    per_question: dict[int, dict[int, dict[str, float | None]]] = field(
        default_factory=dict, repr=False
    )
    # Latence de retrieval par question, reranking compris : le cout d'une
    # technique fait partie de son verdict.
    retrieval_ms: list[int] = field(default_factory=list, repr=False)

    @property
    def median_retrieval_ms(self) -> float | None:
        if not self.retrieval_ms:
            return None
        ordered = sorted(self.retrieval_ms)
        mid = len(ordered) // 2
        return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "config_hash": self.config_hash,
            "index_hash": self.index_hash,
            "ks": self.ks,
            "n_questions": self.n_questions,
            "n_failed": self.n_failed,
            "median_retrieval_ms": self.median_retrieval_ms,
            "notes": self.notes,
            "curve": {
                metric: {
                    str(k): {"mean": iv.mean, "ci_low": iv.low, "ci_high": iv.high, "n": iv.n}
                    for k, iv in by_k.items()
                }
                for metric, by_k in self.curve.items()
            },
        }


def aggregate(
    per_question: Iterable[dict[int, dict[str, float | None]]], ks: list[int]
) -> dict[str, dict[int, Interval]]:
    per_question = list(per_question)
    curve: dict[str, dict[int, Interval]] = {m: {} for m in CURVE_METRICS}
    for metric in CURVE_METRICS:
        for k in ks:
            values = [q[k][metric] for q in per_question if q[k][metric] is not None]
            curve[metric][k] = bootstrap_ci(values)  # type: ignore[arg-type]
    return curve


async def recall_curve(
    cfg: PipelineConfig,
    *,
    dataset_name: str,
    ks: Iterable[int] = DEFAULT_KS,
    split: str = "eval",
    limit: int | None = None,
    nugget_threshold: float = 0.6,
    settings: Settings | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_index_progress: Callable[[int, int, int], None] | None = None,
) -> CurveReport:
    settings = settings or default_settings
    ks = sorted({int(k) for k in ks if int(k) > 0})
    if not ks:
        raise ValueError("ks doit contenir au moins un entier positif")
    tuned, notes = pool_config(cfg, ks[-1])

    async with LLMClient(settings) as llm:
        with db.connect(settings) as conn:
            dataset = db.get_dataset(conn, dataset_name)
            if dataset is None:
                raise ValueError(
                    f"dataset '{dataset_name}' absent — "
                    f"lancer `ragbench dataset load {dataset_name}`"
                )
            index = await ensure_index(
                conn, llm, cfg, dataset_id=dataset["id"], progress=on_index_progress
            )
            questions = [
                q
                for q in load_questions(conn, dataset["id"], split=split, limit=limit)
                if q.get("gold_evidence")
            ]
            conn.commit()

        queue: asyncio.Queue = asyncio.Queue()
        for q in questions:
            queue.put_nowait(q)
        per_question: dict[int, dict[int, dict[str, float | None]]] = {}
        latencies: list[int] = []
        failed = 0
        done = 0
        lock = asyncio.Lock()

        # Une connexion par worker, comme en campagne : une connexion psycopg
        # n'est pas sure en usage entrelace.
        async def worker() -> None:
            nonlocal failed, done
            with db.connect(settings) as wconn:
                while True:
                    try:
                        q = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    try:
                        result = await retrieve(
                            wconn, llm, tuned, index_id=index.index_id, question=q["question"]
                        )
                        row = question_curve(
                            [c.as_dict() for c in result.contexts],
                            q["gold_evidence"],
                            ks,
                            nugget_threshold=nugget_threshold,
                        )
                        async with lock:
                            per_question[q["id"]] = row
                            latencies.append(result.elapsed_ms)
                    except Exception:  # noqa: BLE001
                        # Comptee, jamais avalee : un diagnostic calcule sur
                        # une partie des questions doit le dire.
                        async with lock:
                            failed += 1
                    async with lock:
                        done += 1
                        if on_progress:
                            on_progress(done, len(questions))

        n_workers = max(1, min(settings.llm_concurrency, len(questions)))
        await asyncio.gather(*(worker() for _ in range(n_workers)))

    return CurveReport(
        config_name=cfg.name,
        config_hash=cfg.hash(),
        index_hash=cfg.index_hash(),
        ks=ks,
        n_questions=len(per_question),
        n_failed=failed,
        curve=aggregate(per_question.values(), ks),
        notes=notes,
        per_question=per_question,
        retrieval_ms=latencies,
    )


@dataclass
class CurveComparison:
    metric: str
    k: int
    comparison: Comparison
    # p-value ajustee par Holm sur toute la famille de tests comparee.
    p_holm: float

    @property
    def verdict(self) -> str:
        c = self.comparison
        if c.significant and self.p_holm < 0.05:
            return "B meilleur" if c.delta > 0 else "B moins bon"
        if c.significant:
            return "a la limite (ne survit pas a Holm)"
        return "non concluant"


def compare_curves(
    a: CurveReport,
    b: CurveReport,
    *,
    metrics: Iterable[str] = ("doc_recall", "nugget_full_coverage"),
    ks: Iterable[int] | None = None,
) -> list[CurveComparison]:
    """Test apparie par question entre deux courbes, corrige par Holm.

    Seules les questions presentes et definies dans les deux courbes sont
    appariees : une question en echec d'un cote n'est pas comptee comme un
    zero, ce qui fabriquerait un ecart.
    """
    ks = sorted(set(ks if ks is not None else a.ks) & set(a.ks) & set(b.ks))
    common = sorted(set(a.per_question) & set(b.per_question))
    raw: list[tuple[str, int, Comparison]] = []
    for metric in metrics:
        for k in ks:
            pairs = [
                (a.per_question[q][k][metric], b.per_question[q][k][metric])
                for q in common
                if a.per_question[q][k][metric] is not None
                and b.per_question[q][k][metric] is not None
            ]
            if not pairs:
                continue
            xs, ys = zip(*pairs, strict=True)
            raw.append((metric, k, paired_bootstrap(list(xs), list(ys))))
    adjusted = holm([c.p_value for _, _, c in raw])
    return [
        CurveComparison(metric, k, comp, p)
        for (metric, k, comp), p in zip(raw, adjusted, strict=True)
    ]

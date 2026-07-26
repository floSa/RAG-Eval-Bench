"""Metriques de retrieval avec verite terrain — sans aucun appel LLM.

Pourquoi commencer par la : ce sont les seules metriques du banc qui soient
DETERMINISTES et GRATUITES. Elles ne dependent d'aucun juge, donc d'aucun
biais de juge, et elles se recalculent instantanement. Toute campagne
devrait commencer par elles : si le recall est a 0.3, aucun reglage de
prompt ne sauvera la generation, et il est inutile de depenser du GPU a
mesurer la fidelite.

Granularite : le DOCUMENT, pas le chunk. La verite terrain de MultiHop-RAG
et de HotpotQA est au niveau du document ; compter un recall sur les chunks
supposerait de savoir quel chunk porte le fait, information qu'on n'a pas.
Les chunks retrouves sont donc dedupliques par document avant calcul, en
conservant le meilleur rang.
"""

from __future__ import annotations

import math
from typing import Any

from .base import EvalContext, Score, register


def _retrieved_documents(prediction: dict[str, Any]) -> list[str]:
    """Documents distincts, dans l'ordre de leur meilleur rang."""
    seen: set[str] = set()
    ordered: list[str] = []
    for ctx in prediction.get("contexts") or []:
        doc = ctx.get("document_external_id")
        if doc and doc not in seen:
            seen.add(doc)
            ordered.append(doc)
    return ordered


def _gold_documents(prediction: dict[str, Any]) -> set[str]:
    return {
        ev["document_external_id"]
        for ev in prediction.get("gold_evidence") or []
        if ev.get("document_external_id")
    }


def _ndcg(retrieved: list[str], gold: set[str], k: int) -> float:
    """nDCG avec pertinence binaire.

    L'IDCG est celui du classement parfait REALISABLE : min(|gold|, k)
    documents pertinents en tete. Utiliser |gold| sans borne penaliserait
    mecaniquement toute config dont top_k < nombre d'evidences, ce qui
    melangerait un choix de configuration avec une qualite de classement.
    """
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@register
class NativeIR:
    name = "native.ir"
    requires = ("gold_evidence",)

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        ks: list[int] = ctx.options.get("ks", [1, 3, 5, 10])
        scored = ctx.answerable()
        if not scored:
            return []

        max_retrieved = max((len(_retrieved_documents(p)) for p in scored), default=0)
        # Un recall@10 sur un top_k=5 vaut exactement le recall@5 : le
        # publier ferait croire a une mesure a 10 qui n'existe pas.
        ks = [k for k in ks if k <= max_retrieved] or [max_retrieved or 1]

        out: list[Score] = []
        aggregates: dict[str, list[float]] = {}

        def _record(metric: str, value: float, qid: int, detail: dict | None = None) -> None:
            out.append(Score(metric=metric, value=value, question_id=qid, detail=detail or {}))
            aggregates.setdefault(metric, []).append(value)

        for pred in scored:
            qid = pred["question_id"]
            retrieved = _retrieved_documents(pred)
            gold = _gold_documents(pred)
            hits = [doc for doc in retrieved if doc in gold]

            for k in ks:
                top = retrieved[:k]
                found = [d for d in top if d in gold]
                _record(f"recall@{k}", len(found) / len(gold), qid)
                _record(f"precision@{k}", len(found) / k, qid)
                _record(f"hit_rate@{k}", 1.0 if found else 0.0, qid)
                _record(f"ndcg@{k}", _ndcg(retrieved, gold, k), qid)

            first = next((i for i, d in enumerate(retrieved, start=1) if d in gold), None)
            _record("mrr", 1.0 / first if first else 0.0, qid)

            # Le detail sert au drill-down : quels documents gold ont ete
            # manques, et par quoi ils ont ete remplaces.
            out.append(
                Score(
                    metric="retrieval_debug",
                    value=None,
                    question_id=qid,
                    detail={
                        "gold": sorted(gold),
                        "retrieved": retrieved,
                        "hit": sorted(set(hits)),
                        "missed": sorted(gold - set(retrieved)),
                    },
                )
            )

        # Agregats de run : la moyenne des scores par question, plus le
        # decompte des questions ecartees. Publier ce decompte evite de
        # lire "recall 0.71" en croyant qu'il porte sur tout le jeu.
        for metric, values in aggregates.items():
            out.append(
                Score(
                    metric=f"mean_{metric}",
                    value=sum(values) / len(values),
                    question_id=None,
                    detail={"n": len(values)},
                )
            )
        out.append(
            Score(
                metric="n_scored",
                value=float(len(scored)),
                question_id=None,
                detail={
                    "total_predictions": len(ctx.predictions),
                    "excluded_unanswerable": len(ctx.unanswerable()),
                    "note": "les questions sans evidence gold sont exclues des "
                            "metriques de retrieval (recall indefini, pas nul)",
                },
            )
        )
        return out

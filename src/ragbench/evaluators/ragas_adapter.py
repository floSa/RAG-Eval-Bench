"""Adaptateur Ragas.

Ragas est la reference de fait de l'evaluation RAG : c'est lui qui a impose
le quatuor faithfulness / answer relevancy / context precision / context
recall. Il apporte ce que les metriques natives ne savent pas produire — la
FIDELITE de la reponse au contexte, decomposee en claims atomiques.

Il apporte aussi ses limites, et le banc existe pour les mesurer plutot que
pour les subir :

1. Ragas est REFERENCE-FREE sur faithfulness et answer relevancy : aucune
   verite terrain, un juge LLM decide. La litterature critique releve une
   correlation avec le jugement humain de l'ordre de 0,55 (moyenne
   harmonique) — insuffisant pour traiter ces scores comme une verite.
   D'ou l'ecran d'annotation et le calcul du kappa cote banc.

2. Avec un petit modele local comme juge, une partie des jugements ressort
   NON PARSABLE et Ragas renvoie NaN. Un NaN silencieusement moyenne est un
   biais : les questions difficiles echouent plus souvent a etre notees, et
   les ignorer surestime le score. On compte donc explicitement la
   COUVERTURE (part des questions effectivement notees) et on la publie a
   cote de chaque metrique.

3. Le cout est reel : faithfulness decompose la reponse en claims puis
   verifie chacune. Compter plusieurs appels LLM par question et par
   metrique. D'ou l'option `max_samples`.

DEUX PIEGES D'INTEGRATION, decouverts en branchant reellement l'outil :

  - Ragas depend de langchain-community, et importe au chargement
    `langchain_community.chat_models.vertexai`, supprime a partir de la
    version 0.4 de langchain-community. Sans le pin `langchain-community<0.4`
    de l'extra, `import ragas` echoue — sur du code Google Vertex dont on
    n'a aucun usage ici.
  - Les noms de colonnes de la sortie ne sont pas ceux des classes de
    metriques : LLMContextPrecisionWithReference produit la colonne
    `llm_context_precision_with_reference`. On lit donc `metric.name`
    plutot que de coder les noms en dur, sinon une metrique disparait en
    silence a la prochaine version.

On passe par `llm_factory` avec un client OpenAI ordinaire, pas par
LangchainLLMWrapper (deprecie en 0.4) : moins d'intermediaires, et la
meme URL OpenAI-compatible que le reste du banc.

Installation :  uv sync --extra ragas
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from openai import OpenAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402

from ..settings import settings as default_settings  # noqa: E402
from .base import EvalContext, Score, register  # noqa: E402

# Ragas 0.4 deplace ses metriques vers ragas.metrics.collections tout en
# gardant l'ancien chemin fonctionnel. On reste sur le chemin classique,
# verifie, et on tait l'avertissement plutot que de suivre une API qu'on
# n'a pas testee.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (  # noqa: E402
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )


def _build_metrics(names: list[str] | None) -> list[Any]:
    """Les quatre metriques canoniques.

    context_precision et context_recall sont les variantes AVEC reference :
    on dispose de la reponse gold, autant s'en servir plutot que de laisser
    le juge inventer sa propre cible.
    """
    catalogue = {
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
    }
    if not names:
        return [cls() for cls in catalogue.values()]
    unknown = set(names) - set(catalogue)
    if unknown:
        raise ValueError(
            f"metriques ragas inconnues : {sorted(unknown)} (disponibles : {sorted(catalogue)})"
        )
    return [catalogue[name]() for name in names]


@register
class RagasEvaluator:
    name = "ragas"
    requires = ("judge",)

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        cfg = ctx.config
        settings = default_settings

        client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
        )
        judge = llm_factory(cfg.models.judge, provider="openai", client=client)
        embeddings = RagasOpenAIEmbeddings(client=client, model=cfg.models.embedder)

        metrics = _build_metrics(ctx.options.get("ragas_metrics"))

        # Les predictions vides ou en erreur sont ecartees : les soumettre au
        # juge produirait un score de fidelite sur une chaine vide, c'est-a-dire
        # un chiffre sans referent.
        usable = [
            p for p in ctx.predictions
            if (p.get("answer") or "").strip() and not p.get("error")
        ]
        limit = ctx.options.get("max_samples")
        if limit:
            usable = usable[: int(limit)]

        if not usable:
            return [
                Score(
                    "coverage", 0.0, None,
                    {"note": "aucune prediction exploitable (reponses vides ou en erreur)"},
                )
            ]

        samples = [
            SingleTurnSample(
                user_input=p["question"],
                retrieved_contexts=[c["text"] for c in (p.get("contexts") or [])],
                response=p["answer"],
                reference=p.get("gold_answer") or "",
            )
            for p in usable
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=metrics,
                llm=judge,
                embeddings=embeddings,
                raise_exceptions=False,  # un jugement non parsable donne NaN, pas un crash
                show_progress=False,
            )
        frame = result.to_pandas()

        out: list[Score] = []
        for metric in metrics:
            column = metric.name
            if column not in frame.columns:
                out.append(
                    Score(
                        column, None, None,
                        {"error": f"colonne absente de la sortie ragas : {list(frame.columns)}"},
                    )
                )
                continue

            valid: list[float] = []
            for prediction, value in zip(usable, frame[column].tolist()):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    # Jugement non parsable : trace explicitement, pas ignore.
                    out.append(
                        Score(
                            column, None, prediction["question_id"],
                            {"unparsed": True, "judge": cfg.models.judge},
                        )
                    )
                    continue
                valid.append(float(value))
                out.append(Score(column, float(value), prediction["question_id"]))

            out.append(
                Score(
                    f"mean_{column}",
                    sum(valid) / len(valid) if valid else None,
                    None,
                    {
                        "n_scored": len(valid),
                        "n_submitted": len(usable),
                        # La couverture est aussi importante que la moyenne :
                        # une moyenne calculee sur 40 % des questions ne dit
                        # rien du systeme, elle dit surtout que le juge cale
                        # sur les cas difficiles.
                        "coverage": round(len(valid) / len(usable), 4),
                        "judge": cfg.models.judge,
                    },
                )
            )

        out.append(
            Score(
                "coverage",
                len(usable) / len(ctx.predictions),
                None,
                {
                    "n_usable": len(usable),
                    "n_total": len(ctx.predictions),
                    "note": "predictions soumises au juge (les reponses vides sont ecartees)",
                },
            )
        )
        return out

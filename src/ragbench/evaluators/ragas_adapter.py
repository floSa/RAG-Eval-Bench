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

Installation :  uv sync --extra ragas
"""

from __future__ import annotations

import math
from typing import Any

from .base import EvalContext, Score, register

# Import au chargement du module : l'echec est capture par base.try_import,
# qui desactive le plugin en le signalant au lieu de casser le banc.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from ..settings import settings as default_settings  # noqa: E402


def _build_metrics(names: list[str] | None) -> dict[str, Any]:
    """Les quatre metriques canoniques, nommees comme dans la litterature.

    context_precision et context_recall sont les variantes AVEC reference :
    on dispose de la reponse gold, autant s'en servir plutot que de laisser
    le juge inventer sa propre cible.
    """
    catalogue = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }
    if not names:
        return catalogue
    unknown = set(names) - set(catalogue)
    if unknown:
        raise ValueError(
            f"metriques ragas inconnues : {sorted(unknown)} "
            f"(disponibles : {sorted(catalogue)})"
        )
    return {name: catalogue[name] for name in names}


@register
class RagasEvaluator:
    name = "ragas"
    requires = ("judge",)

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        cfg = ctx.config
        settings = default_settings

        judge = LangchainLLMWrapper(
            ChatOpenAI(
                model=cfg.models.judge,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                temperature=0.0,
                timeout=settings.llm_timeout_s,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=cfg.models.embedder,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                check_embedding_ctx_length=False,  # Ollama n'expose pas le tokenizer
            )
        )

        metrics = _build_metrics(ctx.options.get("ragas_metrics"))

        # Les predictions en erreur ou vides sont ecartees : les soumettre au
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

        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=list(metrics.values()),
            llm=judge,
            embeddings=embeddings,
            raise_exceptions=False,  # un jugement non parsable donne NaN, pas un crash
            show_progress=False,
        )
        frame = result.to_pandas()

        out: list[Score] = []
        for metric_name in metrics:
            if metric_name not in frame.columns:
                # Ragas renomme parfois ses colonnes d'une version a l'autre :
                # on le signale au lieu de perdre la metrique en silence.
                out.append(
                    Score(
                        metric_name, None, None,
                        {"error": f"colonne absente de la sortie ragas : {list(frame.columns)}"},
                    )
                )
                continue

            values = frame[metric_name].tolist()
            valid: list[float] = []

            for prediction, value in zip(usable, values):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    # Jugement non parsable : trace explicitement, pas ignore.
                    out.append(
                        Score(
                            metric_name, None, prediction["question_id"],
                            {"unparsed": True, "judge": cfg.models.judge},
                        )
                    )
                    continue
                valid.append(float(value))
                out.append(Score(metric_name, float(value), prediction["question_id"]))

            out.append(
                Score(
                    f"mean_{metric_name}",
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

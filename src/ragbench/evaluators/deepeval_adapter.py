"""Adaptateur DeepEval.

Ce que DeepEval apporte et que les autres n'ont pas : il est concu pour la
NON-REGRESSION. Ses metriques portent un seuil et un verdict pass/fail, et
s'integrent a pytest. C'est ce qui transforme une evaluation ponctuelle en
garde-fou permanent : « le recall ne doit pas descendre sous 0,45 » devient
un test qui casse la CI, pas un chiffre qu'on regarde de temps en temps.

C'est aussi le seul adaptateur du banc dont la valeur ne se lit pas dans un
tableau de scores mais dans un fichier de tests — voir tests/test_regression.py.

Le branchement sur un modele local passe par DeepEvalBaseLLM, le point
d'extension documente. On ne passe PAS par les modeles OpenAI integres :
ils supposent un endpoint qui garantit la sortie structuree, ce qu'Ollama
ne fait pas de maniere fiable avec un petit modele.

Installation :  uv sync --extra deepeval
"""

from __future__ import annotations

import json
import re
from typing import Any

from deepeval import evaluate as deepeval_evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402

from ..settings import settings as default_settings  # noqa: E402
from .base import EvalContext, Score, register  # noqa: E402

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LocalJudge(DeepEvalBaseLLM):
    """Juge local expose a DeepEval via son API OpenAI-compatible.

    Le point sensible est la sortie structuree : DeepEval demande un objet
    Pydantic et un petit modele local produit reguliement du JSON entoure
    de texte, ou du JSON legerement invalide. On extrait donc le premier
    bloc {...} et on tente la validation, plutot que de laisser la
    bibliotheque lever et perdre toute la campagne.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        settings = default_settings
        self._sync = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
        )
        self._async = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
        )
        super().__init__(model)

    def load_model(self) -> Any:
        return self._sync

    def get_model_name(self) -> str:
        return f"local:{self.model}"

    def _coerce(self, text: str, schema: Any) -> Any:
        if schema is None:
            return text
        match = _JSON_BLOCK.search(text)
        payload = match.group(0) if match else text
        try:
            return schema.model_validate(json.loads(payload))
        except Exception:
            # Renvoyer le texte brut laisse DeepEval decider : selon la
            # metrique il retombe sur un defaut ou signale l'echec. C'est
            # preferable a une exception qui interrompt tout le run.
            return text

    def generate(self, prompt: str, schema: Any = None) -> Any:
        response = self._sync.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._coerce(response.choices[0].message.content or "", schema)

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        response = await self._async.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._coerce(response.choices[0].message.content or "", schema)


# Seuils par defaut. Ce ne sont PAS des cibles de qualite absolues : ce sont
# des garde-fous de non-regression, a recalibrer sur la baseline du projet
# une fois qu'elle est etablie. Un seuil pris dans la documentation d'un
# outil ne veut rien dire sur un corpus donne.
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "contextual_precision": 0.5,
    "contextual_recall": 0.5,
}


@register
class DeepEvalEvaluator:
    name = "deepeval"
    requires = ("judge", "gold_answer")

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        judge = LocalJudge(ctx.config.models.judge)
        thresholds = {**DEFAULT_THRESHOLDS, **(ctx.options.get("thresholds") or {})}

        metrics = [
            FaithfulnessMetric(threshold=thresholds["faithfulness"], model=judge),
            AnswerRelevancyMetric(threshold=thresholds["answer_relevancy"], model=judge),
            ContextualPrecisionMetric(threshold=thresholds["contextual_precision"], model=judge),
            ContextualRecallMetric(threshold=thresholds["contextual_recall"], model=judge),
        ]

        usable = [
            p for p in ctx.predictions
            if (p.get("answer") or "").strip() and not p.get("error") and p.get("contexts")
        ]
        limit = ctx.options.get("max_samples")
        if limit:
            usable = usable[: int(limit)]
        if not usable:
            return [Score("coverage", 0.0, None, {"note": "aucune prediction exploitable"})]

        cases = [
            LLMTestCase(
                input=p["question"],
                actual_output=p["answer"],
                expected_output=p.get("gold_answer") or "",
                retrieval_context=[c["text"] for c in p["contexts"]],
            )
            for p in usable
        ]

        results = deepeval_evaluate(
            test_cases=cases,
            metrics=metrics,
            display_config=DisplayConfig(show_indicator=False, print_results=False),
            async_config=AsyncConfig(run_async=True, max_concurrent=default_settings.llm_concurrency),
        )

        out: list[Score] = []
        per_metric: dict[str, list[float]] = {}
        per_metric_pass: dict[str, list[bool]] = {}

        for prediction, case_result in zip(usable, getattr(results, "test_results", results)):
            for metric_data in getattr(case_result, "metrics_data", []) or []:
                key = (metric_data.name or "metric").lower().replace(" ", "_")
                if metric_data.score is None:
                    out.append(Score(key, None, prediction["question_id"], {"unparsed": True}))
                    continue
                value = float(metric_data.score)
                per_metric.setdefault(key, []).append(value)
                per_metric_pass.setdefault(key, []).append(bool(metric_data.success))
                out.append(
                    Score(
                        key,
                        value,
                        prediction["question_id"],
                        # La raison textuelle est ce que DeepEval apporte de
                        # plus utile au drill-down : elle dit POURQUOI le juge
                        # a note ainsi.
                        {"reason": (metric_data.reason or "")[:600], "passed": bool(metric_data.success)},
                    )
                )

        for key, values in per_metric.items():
            passes = per_metric_pass.get(key, [])
            out.append(
                Score(
                    f"mean_{key}",
                    sum(values) / len(values),
                    None,
                    {
                        "n_scored": len(values),
                        "n_submitted": len(usable),
                        "coverage": round(len(values) / len(usable), 4),
                        # Le taux de reussite au seuil est la sortie propre a
                        # DeepEval : c'est lui qui pilote la CI.
                        "pass_rate": round(sum(passes) / len(passes), 4) if passes else None,
                        "threshold": thresholds.get(key),
                        "judge": ctx.config.models.judge,
                    },
                )
            )
        return out

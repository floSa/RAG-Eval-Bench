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

TROIS OBSERVATIONS FAITES EN LE BRANCHANT POUR DE VRAI. Elles valent pour
n'importe quel juge local, pas seulement pour DeepEval.

1. LE JUGE N'EST PAS REPRODUCTIBLE. Deux passes identiques, memes questions,
   temperature 0 : faithfulness 1.000 puis 0.583. Avec une graine fixee, la
   variation tombe a 0.633 / 0.667 — beaucoup mieux, mais pas nulle
   (l'echantillon de mesure ne fait que 5 questions, donc une seule question
   qui bascule deplace la moyenne de 0,2). Consequence pratique : un ecart
   mesure par un juge doit etre plus grand que sa propre variabilite avant
   qu'on en conclue quoi que ce soit, et les metriques DETERMINISTES du banc
   (native.ir, native.answer) restent la reference.

2. UN PETIT JUGE NE SAIT PAS TOUT JUGER. llama3.2:3b produit faithfulness
   et answer_relevancy, mais AUCUN score exploitable pour
   contextual_precision et contextual_recall : leurs schemas JSON sont trop
   profonds pour un modele de 3 milliards de parametres. Ces metriques sont
   publiees avec une couverture de 0 plutot qu'omises — une metrique absente
   se lirait comme une metrique non demandee.

3. LE CHOIX DU JUGE EST UN ARBITRAGE, PAS UN DETAIL. gemma4:e4b juge mieux
   mais raisonne a chaque appel : 12 a 30 s par jugement, soit plus de 2 min
   par question sur les 4 metriques — inutilisable au-dela de quelques
   dizaines de questions. llama3.2:3b juge en ~1 s mais echoue sur la moitie
   des metriques. C'est exactement le genre d'arbitrage que le banc doit
   rendre visible au lieu de le laisser subir.

Installation :  uv sync --extra deepeval
"""

from __future__ import annotations

import json
import re
from typing import Any

from deepeval import evaluate as deepeval_evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig  # noqa: E402
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
from .base import EvalContext, Score, align_by_input, register  # noqa: E402

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
        # Le nom du modele est garde dans model_id et surtout PAS dans
        # self.model : DeepEvalBaseLLM.__init__ affecte
        # `self.model = self.load_model()` et ecraserait la chaine par le
        # client OpenAI. L'erreur qui en decoule est indirecte et peu
        # lisible — « Object of type OpenAI is not JSON serializable » au
        # moment du premier appel, parce que le client part comme nom de
        # modele dans le corps de la requete.
        self.model_id = model
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
        return f"local:{self.model_id}"

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
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # Graine fixee : sans elle, deux passes du MEME juge sur les
            # MEMES questions a temperature 0 donnent des scores differents.
            # Mesure sur 5 questions : faithfulness 1.000 puis 0.583. Un
            # resultat de juge non reproductible ne peut fonder aucune
            # conclusion, et la variabilite se confond avec l'effet qu'on
            # cherche a mesurer.
            seed=0,
        )
        return self._coerce(response.choices[0].message.content or "", schema)

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        response = await self._async.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # Graine fixee : sans elle, deux passes du MEME juge sur les
            # MEMES questions a temperature 0 donnent des scores differents.
            # Mesure sur 5 questions : faithfulness 1.000 puis 0.583. Un
            # resultat de juge non reproductible ne peut fonder aucune
            # conclusion, et la variabilite se confond avec l'effet qu'on
            # cherche a mesurer.
            seed=0,
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

        # ignore_errors=True est indispensable avec un juge local.
        #
        # Les metriques de DeepEval exigent des sorties JSON structurees dont
        # les schemas sont parfois profonds (ContextualPrecision en
        # particulier). Un petit modele local produit reguliement du JSON
        # legerement invalide, et sans cette option DeepEval leve — une seule
        # sortie mal formee sur 200 questions ferait tomber toute
        # l'evaluation.
        #
        # Avec l'option, l'echec devient un score MANQUANT, comptabilise dans
        # la couverture publiee plus bas. C'est la bonne lecture : le juge n'a
        # pas su trancher, ce n'est ni un bon ni un mauvais score, et une
        # couverture qui s'effondre est en soi le signal qu'il faut un juge
        # plus capable.
        # NB : dans DeepEval 4.x, ignore_errors n'est plus un parametre des
        # metriques mais un ErrorConfig passe a evaluate() — voir plus bas.
        common = {"model": judge, "async_mode": True}
        metrics = [
            FaithfulnessMetric(threshold=thresholds["faithfulness"], **common),
            AnswerRelevancyMetric(threshold=thresholds["answer_relevancy"], **common),
            ContextualPrecisionMetric(threshold=thresholds["contextual_precision"], **common),
            ContextualRecallMetric(threshold=thresholds["contextual_recall"], **common),
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
            async_config=AsyncConfig(
                run_async=True, max_concurrent=default_settings.llm_concurrency
            ),
            # ignore_errors=True est indispensable avec un juge local.
            #
            # Les metriques de DeepEval exigent des sorties JSON structurees
            # dont les schemas sont parfois profonds (ContextualPrecision en
            # particulier), et un petit modele local produit reguliement du
            # JSON legerement invalide. Sans cette option, DeepEval leve : une
            # seule sortie mal formee sur 200 questions fait tomber toute
            # l'evaluation.
            #
            # Avec l'option, l'echec devient un score MANQUANT, comptabilise
            # dans la couverture publiee plus bas. C'est la bonne lecture : le
            # juge n'a pas su trancher, ce n'est ni un bon ni un mauvais score,
            # et une couverture qui s'effondre est en soi le signal qu'il faut
            # un juge plus capable.
            error_config=ErrorConfig(ignore_errors=True),
        )

        out: list[Score] = []
        per_metric: dict[str, list[float]] = {}
        per_metric_pass: dict[str, list[bool]] = {}

        # DEEPEVAL NE RENVOIE PAS SES RESULTATS DANS L'ORDRE D'ENTREE.
        # En mode asynchrone, les cas reviennent dans leur ordre de
        # COMPLETION. Un zip positionnel attribuerait donc les scores aux
        # mauvaises questions — silencieusement, avec des moyennes
        # parfaitement plausibles et un drill-down qui montre la reponse
        # d'une question a cote du verdict d'une autre. On reapparie sur le
        # texte de la question.
        aligned, n_unmatched = align_by_input(
            usable,
            list(getattr(results, "test_results", results)),
            key=lambda case: getattr(case, "input", None),
        )

        for prediction, case_result in aligned:
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

        if n_unmatched:
            out.append(
                Score(
                    "unmatched_results",
                    float(n_unmatched),
                    None,
                    {
                        "note": "resultats DeepEval non rattachables a une question "
                                "par leur texte — scores correspondants ignores plutot "
                                "que mal attribues"
                    },
                )
            )

        # Les metriques DEMANDEES mais qui n'ont produit aucune valeur
        # exploitable doivent apparaitre explicitement. Omises, elles se
        # liraient comme non demandees — alors qu'elles ont ete demandees et
        # que le juge a echoue a les produire. Sur llama3.2:3b, c'est le cas
        # de contextual_precision et contextual_recall, dont les schemas JSON
        # sont trop profonds pour un modele de 3 milliards de parametres.
        requested = {
            getattr(m, "__name__", type(m).__name__).lower().replace(" ", "_")
            for m in metrics
        }
        for key in sorted(requested - set(per_metric)):
            out.append(
                Score(
                    f"mean_{key}",
                    None,
                    None,
                    {
                        "n_scored": 0,
                        "n_submitted": len(usable),
                        "coverage": 0.0,
                        "judge": ctx.config.models.judge,
                        "note": "aucun jugement exploitable — le juge n'a pas su "
                                "produire la sortie structuree attendue",
                    },
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

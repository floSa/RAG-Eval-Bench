"""Justesse de la reponse finale — sans juge LLM.

Rendu possible par une particularite de MultiHop-RAG : les reponses gold
sont tres courtes (mediane 3 caracteres). L'exact-match normalise et le F1
par tokens, hérités de SQuAD, y sont donc pertinents, la ou ils seraient
inutilisables sur des reponses redigees. Consequence pratique : on obtient
une mesure de justesse deterministe et gratuite, contre laquelle on pourra
CALIBRER les juges LLM au lieu de leur faire confiance a priori.

Deux precautions qui changent la lecture des chiffres :

1. NIVEAU DE CHANCE. 52 % des questions attendent Yes/no. Une accuracy
   globale de 0.60 peut correspondre a un systeme a peine meilleur que le
   hasard. On publie donc l'accuracy VENTILEE par type de question, plus la
   ligne de base de la classe majoritaire calculee sur le jeu lui-meme —
   la seule reference honnete.

2. ABSTENTION. Une abstention sur une question sans reponse est un succes
   (negative rejection) ; sur une question repondable c'est un echec. Les
   deux sont comptes separement, jamais melanges dans une accuracy unique.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from .base import EvalContext, Score, register

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_YES_NO = {"yes", "no"}


def normalize(text: str | None) -> str:
    """Normalisation SQuAD : minuscules, sans ponctuation, sans articles."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, gold: str) -> bool:
    """Le gold apparait-il dans la reponse produite ?

    Necessaire parce qu'un generateur repond « The individual is Sam
    Bankman-Fried. » la ou le gold vaut « Sam Bankman-Fried » : l'exact
    match le compte faux alors que la reponse est juste. Sur les gold
    Yes/no on exige en revanche l'egalite stricte du premier token, sinon
    « ... it is not yes ... » passerait pour bon.
    """
    norm_pred, norm_gold = normalize(prediction), normalize(gold)
    if not norm_gold:
        return False
    if norm_gold in _YES_NO:
        head = norm_pred.split()
        return bool(head) and head[0] == norm_gold
    return norm_gold in norm_pred


@register
class NativeAnswer:
    name = "native.answer"
    requires = ("gold_answer",)

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        out: list[Score] = []
        by_type: dict[str, list[float]] = {}
        gold_by_type: dict[str, list[str]] = {}

        answerable_ids = {p["question_id"] for p in ctx.answerable()}

        for pred in ctx.predictions:
            qid = pred["question_id"]
            gold = pred.get("gold_answer") or ""
            answer = pred.get("answer") or ""
            qtype = pred.get("question_type") or "unknown"
            is_answerable = qid in answerable_ids

            if is_answerable:
                em = float(normalize(answer) == normalize(gold))
                hit = float(contains_gold(answer, gold))
                f1 = token_f1(answer, gold)

                out.append(Score("em", em, qid))
                out.append(Score("contains", hit, qid, {"gold": gold, "answer": answer[:400]}))
                out.append(Score("token_f1", f1, qid))
                # Une abstention sur une question repondable : le systeme
                # a renonce alors qu'il pouvait repondre.
                out.append(Score("false_abstention", float(pred.get("abstained", False)), qid))

                by_type.setdefault(qtype, []).append(hit)
                gold_by_type.setdefault(qtype, []).append(normalize(gold))
            else:
                # Question sans reponse dans le corpus : le seul
                # comportement correct est de s'abstenir.
                rejected = float(pred.get("abstained", False))
                out.append(
                    Score(
                        "negative_rejection",
                        rejected,
                        qid,
                        {"answer": answer[:400], "expected": "abstention"},
                    )
                )

        # --- agregats de run ------------------------------------------
        for qtype, values in sorted(by_type.items()):
            golds = gold_by_type[qtype]
            majority = Counter(golds).most_common(1)[0]
            # Ligne de base honnete : le score qu'obtiendrait un systeme
            # repondant toujours la classe la plus frequente de ce type.
            baseline = majority[1] / len(golds)
            out.append(
                Score(
                    f"accuracy[{qtype}]",
                    sum(values) / len(values),
                    None,
                    {
                        "n": len(values),
                        "majority_class": majority[0],
                        "majority_baseline": round(baseline, 4),
                        "binary": all(g in _YES_NO for g in golds),
                    },
                )
            )

        answered = [v for values in by_type.values() for v in values]
        if answered:
            out.append(
                Score(
                    "accuracy",
                    sum(answered) / len(answered),
                    None,
                    {
                        "n": len(answered),
                        "note": "toutes questions repondables confondues ; a lire "
                                "avec les accuracy[type] et leur majority_baseline",
                    },
                )
            )

        unanswerable = ctx.unanswerable()
        if unanswerable:
            rate = sum(float(p.get("abstained", False)) for p in unanswerable) / len(unanswerable)
            out.append(
                Score(
                    "negative_rejection_rate",
                    rate,
                    None,
                    {
                        "n": len(unanswerable),
                        "note": "part des questions sans reponse ou le systeme s'est abstenu",
                    },
                )
            )

        return out


def summarize_answers(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Statistiques descriptives utilisees par le tableau de bord."""
    lengths = [len(p.get("answer") or "") for p in predictions]
    lengths.sort()
    return {
        "n": len(predictions),
        "abstained": sum(1 for p in predictions if p.get("abstained")),
        "errors": sum(1 for p in predictions if p.get("error")),
        "median_answer_chars": lengths[len(lengths) // 2] if lengths else 0,
    }

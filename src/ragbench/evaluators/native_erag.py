"""eRAG — evaluer le retrieval par l'utilite reelle de chaque document.

D'apres Salemi & Zamani, SIGIR 2024. L'idee corrige un angle mort des
metriques IR classiques : nDCG et recall mesurent la PERTINENCE jugee par
un humain, alors que ce qui compte dans un RAG est l'UTILITE pour le
modele. Un passage peut etre parfaitement pertinent et inexploitable —
trop dense, formule autrement que la question, contredit par un voisin — et
un passage juge marginal peut suffire.

Methode : on soumet chaque document remonte SEUL au generateur, on note la
reponse obtenue contre la reference, et ce score devient le label
d'utilite du document. On agrege ensuite ces labels comme une metrique IR
ordinaire. La litterature montre que ces labels correlent nettement mieux
avec la performance end-to-end que les jugements de pertinence.

Ce que ca revele concretement, et que native.ir ne voit pas :
  - utilite elevee au rang 5 et faible au rang 1  -> probleme de CLASSEMENT
  - utilite faible partout alors que recall est bon -> les bons documents
    sont la, le generateur ne sait pas s'en servir : c'est un probleme de
    prompt ou de modele, pas de recherche

Cout : top_k appels LLM par question, en plus du run lui-meme. C'est
l'evaluateur le plus cher du banc — d'ou `max_samples`, dont la valeur par
defaut est volontairement basse.

LIMITE MESUREE SUR MULTIHOP-RAG. La methode suppose qu'un document SEUL
puisse permettre de repondre. Sur un corpus multi-hop, c'est faux par
construction : la reponse exige de croiser 2 a 4 articles, et l'utilite
individuelle ressort a 0,000 a tous les rangs. Le chiffre n'est pas un bug,
il dit exactement ce qu'il mesure — mais il ne DISCRIMINE plus entre deux
configurations, donc il n'apprend rien ici.

eRAG garde tout son sens sur du QA a un seul saut (SQuAD, Natural
Questions, ou un corpus de documentation technique ou la reponse tient dans
une page). Pour le multi-hop, `native.nuggets` repond a la meme question —
« les bons passages ont-ils ete remontes ? » — sans supposer qu'un seul
suffise.
"""

from __future__ import annotations

import math
from typing import Any

from .base import EvalContext, Score, register
from .native_answer import contains_gold, token_f1

_PROMPT = """Answer the question using only the passage below.
If the passage does not contain the answer, reply exactly: INSUFFICIENT_CONTEXT

Passage:
{passage}

Question: {question}

Answer:"""


@register
class NativeERAG:
    name = "native.erag"
    requires = ("gold_answer",)

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        # Defaut bas et assume : sur 200 questions a top_k=5, la version
        # complete coute 1 000 generations supplementaires. Mieux vaut une
        # mesure explicite sur 40 questions qu'une campagne abandonnee.
        limit = int(ctx.options.get("max_samples", 40))

        scored = [p for p in ctx.answerable() if p.get("contexts")][:limit]
        if not scored:
            return []

        out: list[Score] = []
        utilities_at_rank: dict[int, list[float]] = {}
        per_question_ndcg: list[float] = []
        per_question_best: list[float] = []

        for pred in scored:
            qid = pred["question_id"]
            gold = pred.get("gold_answer") or ""
            contexts = pred["contexts"]

            prompts = [
                _PROMPT.format(passage=c["text"][:4000], question=pred["question"])
                for c in contexts
            ]
            completions = await ctx.llm.complete_many(
                prompts,
                model=ctx.config.models.generator,
                role="erag",
                temperature=0.0,
                max_tokens=128,
                thinking=False,
            )

            utilities: list[float] = []
            for completion in completions:
                if completion is None:
                    utilities.append(0.0)
                    continue
                answer = completion.text
                # Utilite graduee plutot que binaire : un passage qui permet
                # une reponse partielle vaut mieux qu'un passage inutile, et
                # le nDCG a besoin de cette nuance pour discriminer.
                if contains_gold(answer, gold):
                    utilities.append(1.0)
                else:
                    utilities.append(round(token_f1(answer, gold), 4))

            for rank, utility in enumerate(utilities, start=1):
                utilities_at_rank.setdefault(rank, []).append(utility)

            # nDCG calcule sur les labels d'utilite : mesure si le classement
            # met vraiment les passages EXPLOITABLES devant.
            dcg = sum(u / math.log2(r + 1) for r, u in enumerate(utilities, start=1))
            ideal = sum(
                u / math.log2(r + 1)
                for r, u in enumerate(sorted(utilities, reverse=True), start=1)
            )
            ndcg = dcg / ideal if ideal else 0.0
            per_question_ndcg.append(ndcg)
            per_question_best.append(max(utilities, default=0.0))

            out.append(Score("erag_ndcg", ndcg, qid, {"utilities": utilities}))
            out.append(
                Score(
                    "erag_best_utility",
                    max(utilities, default=0.0),
                    qid,
                    {
                        # Le rang du meilleur passage : s'il est
                        # systematiquement > 1, le classement est a revoir,
                        # pas la recherche.
                        "best_rank": (
                            utilities.index(max(utilities)) + 1 if utilities else None
                        )
                    },
                )
            )

        for rank, values in sorted(utilities_at_rank.items()):
            out.append(
                Score(
                    f"mean_utility@rank{rank}",
                    sum(values) / len(values),
                    None,
                    {"n": len(values)},
                )
            )

        if per_question_ndcg:
            out.append(
                Score(
                    "mean_erag_ndcg",
                    sum(per_question_ndcg) / len(per_question_ndcg),
                    None,
                    {
                        "n": len(per_question_ndcg),
                        "note": "nDCG calcule sur l'utilite reelle des passages, "
                                "pas sur des jugements de pertinence",
                    },
                )
            )
            out.append(
                Score(
                    "mean_erag_best_utility",
                    sum(per_question_best) / len(per_question_best),
                    None,
                    {
                        "n": len(per_question_best),
                        "note": "plafond atteignable : ce que donnerait un reranker "
                                "parfait sur les memes passages",
                    },
                )
            )
        return out


__all__: list[Any] = ["NativeERAG"]

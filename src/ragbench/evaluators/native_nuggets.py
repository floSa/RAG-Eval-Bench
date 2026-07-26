"""Nugget recall — la methode d'evaluation la mieux alignee sur le jugement
humain publiee a ce jour (AutoNuggetizer, TREC RAG 2024 : tau = 0,87 de
correlation de rang avec des juges humains).

Principe : decomposer la reponse attendue en « pepites » d'information
atomiques, puis mesurer combien sont couvertes. Ici on n'a pas besoin de
les faire generer par un LLM — MultiHop-RAG livre deja, pour chaque
question, la liste des FAITS qui justifient la reponse (`evidence_list[].fact`).
C'est une verite terrain gratuite et non biaisee par un juge, ce qui evite
le piege documente du nuggetizer : quand le systeme evalue et le systeme
qui fabrique les pepites partagent le meme LLM, on se note soi-meme.

Application choisie : les pepites sont cherchees dans les PASSAGES
REMONTES, pas dans la reponse. Raison — sur ce corpus la reponse attendue
fait un a trois mots (« Valve », « Yes ») et ne peut structurellement pas
contenir les faits. Mesuree sur le contexte, la couverture repond a une
question que native.ir ne sait pas poser : le bon document a-t-il ete
remonte AVEC le passage qui porte le fait ? Un document juste dont on
retient le mauvais paragraphe compte comme un succes pour recall@k et
comme un echec ici — et c'est l'echec qui explique la reponse fausse.

Deux modes, volontairement tous deux disponibles :
  lexical : recouvrement de tokens, deterministe et gratuit
  judge   : verdict d'un LLM, plus fin mais biaise
Les faire tourner cote a cote sur les memes pepites donne une mesure
directe du desaccord entre une metrique deterministe et un juge — c'est
l'objet meme de ce banc.
"""

from __future__ import annotations

import re

from .base import EvalContext, Score, register

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "that",
    "this", "it", "its", "his", "her", "their", "has", "have", "had", "will",
}

_WORD = re.compile(r"[a-z0-9]+")

_JUDGE_PROMPT = """You check whether a fact is supported by a passage.

Fact:
{fact}

Passage:
{passage}

Is the fact stated or directly entailed by the passage?
Reply with exactly one word: YES or NO.

Answer:"""


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def lexical_support(fact: str, contexts: list[str], threshold: float = 0.6) -> tuple[bool, float]:
    """Un fait est couvert si un passage contient assez de ses mots pleins.

    Le seuil porte sur la part des tokens du FAIT retrouves dans le passage,
    pas l'inverse : un passage long ne doit pas etre avantage simplement
    parce qu'il contient beaucoup de mots.
    """
    fact_tokens = _tokens(fact)
    if not fact_tokens:
        return False, 0.0
    best = max(
        (len(fact_tokens & _tokens(ctx)) / len(fact_tokens) for ctx in contexts),
        default=0.0,
    )
    return best >= threshold, best


@register
class NativeNuggets:
    name = "native.nuggets"
    requires = ("gold_evidence",)

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        mode = ctx.options.get("nugget_mode", "lexical")
        threshold = float(ctx.options.get("nugget_threshold", 0.6))
        limit = ctx.options.get("max_samples")

        scored = ctx.answerable()
        if limit:
            scored = scored[: int(limit)]
        if not scored:
            return []

        out: list[Score] = []
        recalls: list[float] = []

        for pred in scored:
            qid = pred["question_id"]
            facts = [
                ev.get("fact", "")
                for ev in (pred.get("gold_evidence") or [])
                if ev.get("fact")
            ]
            passages = [c["text"] for c in (pred.get("contexts") or [])]

            if not facts:
                continue
            if not passages:
                out.append(Score("nugget_recall", 0.0, qid, {"note": "aucun passage remonte"}))
                recalls.append(0.0)
                continue

            if mode == "judge":
                supported = await self._judge(ctx, facts, passages)
            else:
                supported = [lexical_support(f, passages, threshold)[0] for f in facts]

            recall = sum(supported) / len(facts)
            recalls.append(recall)
            out.append(
                Score(
                    "nugget_recall",
                    recall,
                    qid,
                    {
                        "mode": mode,
                        "n_facts": len(facts),
                        "n_supported": int(sum(supported)),
                        # Les faits manques sont l'information exploitable :
                        # ils disent quel paragraphe le retrieval aurait du
                        # remonter.
                        "missing": [
                            f[:200] for f, ok in zip(facts, supported, strict=True) if not ok
                        ][:5],
                    },
                )
            )

        if recalls:
            out.append(
                Score(
                    "mean_nugget_recall",
                    sum(recalls) / len(recalls),
                    None,
                    {
                        "n": len(recalls),
                        "mode": mode,
                        "note": "part des faits de reference couverts par les passages remontes",
                    },
                )
            )
            # Un recall de pepites tres inferieur au recall documentaire
            # signale que les bons documents sont remontes mais decoupes au
            # mauvais endroit — un probleme de chunking, pas d'embedding.
            out.append(
                Score(
                    "nugget_full_coverage",
                    sum(1 for r in recalls if r >= 0.999) / len(recalls),
                    None,
                    {"note": "part des questions dont TOUS les faits sont couverts"},
                )
            )
        return out

    async def _judge(
        self, ctx: EvalContext, facts: list[str], passages: list[str]
    ) -> list[bool]:
        """Un appel par (fait, passage), jusqu'au premier verdict positif.

        Cout : jusqu'a n_facts x n_passages appels par question. C'est le
        mode le plus cher du banc, d'ou `max_samples`.
        """
        joined = "\n\n".join(p[:1500] for p in passages)
        prompts = [_JUDGE_PROMPT.format(fact=f[:800], passage=joined) for f in facts]
        completions = await ctx.llm.complete_many(
            prompts,
            model=ctx.config.models.judge,
            role="judge",
            temperature=0.0,
            max_tokens=4,
            thinking=False,
        )
        # Un verdict illisible est compte NON couvert : l'alternative
        # surestimerait la couverture des cas justement les plus ambigus.
        return [
            bool(c and c.text.strip().upper().startswith("YES")) for c in completions
        ]


__all__ = ["NativeNuggets", "lexical_support"]

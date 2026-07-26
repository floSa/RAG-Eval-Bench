"""Fidelite au niveau de la claim, et ATTRIBUTION DE L'ERREUR.

Reprend l'idee centrale de RAGChecker (Amazon, NeurIPS 2024 D&B) —
decomposer en assertions atomiques et verifier chacune — mais implementee
directement contre le service LLM local, sans la chaine de dependances de
RAGChecker (RefChecker, modeles spacy, litellm).

L'apport unique par rapport a Ragas : ce n'est pas un score de fidelite, ce
sont TROIS chiffres qui designent le coupable.

    claim_precision   part des affirmations de la reponse qui sont
                      soutenues par les passages remontes.
                      Basse -> le GENERATEUR invente.

    context_utilization  part des affirmations soutenues qui viennent
                      effectivement du contexte plutot que de la memoire
                      du modele.

    claim_recall      part des affirmations de la reponse de reference
                      presentes dans la reponse produite.
                      Basse alors que claim_precision est haute
                      -> le RETRIEVAL n'a pas remonte de quoi repondre.

Un score global de fidelite ne permet pas de choisir entre « changer le
prompt » et « changer le retriever ». Ces trois-la, si.

Limite assumee : l'extraction de claims et la verification passent par le
juge, donc heritent de ses biais. La proportion de sorties non parsables
est publiee a cote de chaque metrique — un score calcule sur 40 % des
questions n'est pas un score.
"""

from __future__ import annotations

import re
from typing import Any

from .base import EvalContext, Score, register

_EXTRACT_PROMPT = """Break the text below into atomic factual claims.
Each claim must be a single standalone statement, understandable without the others.
Resolve pronouns to the entities they refer to.
Output one claim per line, prefixed by "- ". No preamble, no numbering.
If the text states no factual claim, output exactly: NONE

Text:
{text}

Claims:"""

_VERIFY_PROMPT = """You check whether a claim is supported by reference passages.

Claim:
{claim}

Passages:
{passages}

Reply with exactly one word:
SUPPORTED   - the passages state or directly entail the claim
CONTRADICTED - the passages state the opposite
NEUTRAL     - the passages neither support nor contradict the claim

Answer:"""

_LINE = re.compile(r"^\s*[-*•]\s*(.+?)\s*$", re.MULTILINE)


def parse_claims(text: str, *, max_claims: int = 12) -> list[str]:
    """Extrait les claims d'une sortie de LLM.

    Tolerant par construction : un petit modele repond tantot en tirets,
    tantot en lignes nues, parfois avec un preambule. Exiger un format
    strict transformerait une variation de style en score de zero.
    """
    if not text or text.strip().upper().startswith("NONE"):
        return []

    claims = [m.group(1).strip() for m in _LINE.finditer(text)]
    if not claims:
        claims = [
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) > 15 and not line.strip().endswith(":")
        ]
    # Borne dure : une reponse verbeuse produirait des dizaines de claims et
    # ferait exploser le cout de verification sans rien apporter.
    return [c for c in claims if len(c) > 10][:max_claims]


@register
class NativeClaims:
    name = "native.claims"
    requires = ("judge",)

    async def evaluate(self, ctx: EvalContext) -> list[Score]:
        limit = int(ctx.options.get("max_samples", 50))
        model = ctx.config.models.judge

        usable = [
            p for p in ctx.predictions
            if (p.get("answer") or "").strip()
            and not p.get("error")
            and not p.get("abstained")
        ][:limit]

        if not usable:
            return [
                Score(
                    "coverage", 0.0, None,
                    {"note": "aucune reponse exploitable (toutes vides, en erreur ou abstenues)"},
                )
            ]

        # --- 1. extraction des claims, en un seul lot ---------------------
        answer_extractions = await ctx.llm.complete_many(
            [_EXTRACT_PROMPT.format(text=(p["answer"] or "")[:3000]) for p in usable],
            model=model, role="judge", temperature=0.0, max_tokens=512, thinking=False,
        )
        gold_extractions = await ctx.llm.complete_many(
            [_EXTRACT_PROMPT.format(text=(p.get("gold_answer") or "")[:3000]) for p in usable],
            model=model, role="judge", temperature=0.0, max_tokens=512, thinking=False,
        )

        out: list[Score] = []
        precisions: list[float] = []
        recalls: list[float] = []
        utilizations: list[float] = []
        n_unparsed = 0

        for prediction, answer_raw, gold_raw in zip(usable, answer_extractions, gold_extractions, strict=True):
            qid = prediction["question_id"]
            if answer_raw is None:
                n_unparsed += 1
                out.append(Score("claim_precision", None, qid, {"unparsed": True}))
                continue

            answer_claims = parse_claims(answer_raw.text)
            gold_claims = parse_claims(gold_raw.text) if gold_raw else []
            passages = "\n\n".join(
                c["text"][:1200] for c in (prediction.get("contexts") or [])
            )

            if not answer_claims:
                out.append(
                    Score("claim_precision", None, qid, {"note": "aucune claim extraite"})
                )
                continue

            # --- 2. chaque claim de la reponse contre le contexte ---------
            verdicts = await ctx.llm.complete_many(
                [
                    _VERIFY_PROMPT.format(claim=c[:500], passages=passages or "(aucun passage)")
                    for c in answer_claims
                ],
                model=model, role="judge", temperature=0.0, max_tokens=6, thinking=False,
            )
            labels = [
                (v.text.strip().upper().split()[0] if v and v.text.strip() else "UNPARSED")
                for v in verdicts
            ]
            supported = sum(1 for label in labels if label.startswith("SUPPORTED"))
            contradicted = sum(1 for label in labels if label.startswith("CONTRADICTED"))
            judged = sum(1 for label in labels if label != "UNPARSED")

            precision = supported / judged if judged else None
            if precision is not None:
                precisions.append(precision)
            out.append(
                Score(
                    "claim_precision", precision, qid,
                    {
                        "n_claims": len(answer_claims),
                        "supported": supported,
                        "contradicted": contradicted,
                        "unparsed": len(labels) - judged,
                        # Les claims contredites sont le signal le plus fort
                        # d'hallucination : le contexte dit l'inverse.
                        "contradicted_claims": [
                            c[:200] for c, label in zip(answer_claims, labels, strict=True)
                            if label.startswith("CONTRADICTED")
                        ][:3],
                    },
                )
            )

            if judged:
                utilizations.append(supported / len(answer_claims))
                out.append(
                    Score("context_utilization", supported / len(answer_claims), qid)
                )

            # --- 3. les claims de reference sont-elles dans la reponse ? ---
            if gold_claims:
                gold_verdicts = await ctx.llm.complete_many(
                    [
                        _VERIFY_PROMPT.format(
                            claim=c[:500], passages=(prediction["answer"] or "")[:3000]
                        )
                        for c in gold_claims
                    ],
                    model=model, role="judge", temperature=0.0, max_tokens=6, thinking=False,
                )
                covered = sum(
                    1 for v in gold_verdicts
                    if v and v.text.strip().upper().startswith("SUPPORTED")
                )
                recall = covered / len(gold_claims)
                recalls.append(recall)
                out.append(
                    Score(
                        "claim_recall", recall, qid,
                        {"n_gold_claims": len(gold_claims), "covered": covered},
                    )
                )

        # --- agregats et attribution --------------------------------------
        def _mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        mean_precision = _mean(precisions)
        mean_recall = _mean(recalls)

        for name, values in [
            ("claim_precision", precisions),
            ("claim_recall", recalls),
            ("context_utilization", utilizations),
        ]:
            out.append(
                Score(
                    f"mean_{name}", _mean(values), None,
                    {
                        "n_scored": len(values),
                        "n_submitted": len(usable),
                        "coverage": round(len(values) / len(usable), 4) if usable else 0.0,
                        "judge": model,
                    },
                )
            )

        # Le verdict d'attribution, formule en clair : c'est la sortie que le
        # lecteur doit retenir, pas les trois chiffres separes.
        if mean_precision is not None and mean_recall is not None:
            if mean_precision >= 0.7 and mean_recall < 0.5:
                verdict = (
                    "le generateur est fidele mais incomplet : les passages remontes "
                    "ne contiennent pas de quoi repondre -> agir sur le RETRIEVAL"
                )
            elif mean_precision < 0.5:
                verdict = (
                    "une part importante des affirmations n'est pas soutenue par le "
                    "contexte -> agir sur le GENERATEUR (prompt, modele, abstention)"
                )
            else:
                verdict = "pas de defaut dominant : precision et rappel sont du meme ordre"
            out.append(
                Score(
                    "attribution", None, None,
                    {
                        "verdict": verdict,
                        "claim_precision": round(mean_precision, 4),
                        "claim_recall": round(mean_recall, 4),
                    },
                )
            )

        out.append(
            Score(
                "coverage", len(usable) / len(ctx.predictions), None,
                {
                    "n_usable": len(usable),
                    "n_total": len(ctx.predictions),
                    "n_unparsed": n_unparsed,
                    "note": "les reponses vides, en erreur ou abstenues sont ecartees",
                },
            )
        )
        return out


__all__: list[Any] = ["NativeClaims", "parse_claims"]

"""Generation de la reponse a partir des passages remontes.

Les templates sont nommes et versionnes ici plutot qu'ecrits en dur dans le
pipeline : le prompt est une variable experimentale de premier ordre, et
son nom entre dans le hash de la config. Changer un mot d'un template sans
changer son nom rend deux runs faussement comparables — d'ou PROMPT_VERSION.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..config import PipelineConfig
from ..llm import LLMClient
from .retrieve import Context

# A incrementer des qu'un template est modifie. Entre dans le hash via
# GenerationConfig.prompt_template ? Non — d'ou la verification explicite
# faite par le runner, qui journalise cette version dans le run.
PROMPT_VERSION = 1

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

_ABSTAIN_PATTERNS = [
    re.compile(rf"\b{ABSTAIN_TOKEN}\b", re.IGNORECASE),
    re.compile(r"\bi (don'?t|do not) (know|have enough)\b", re.IGNORECASE),
    re.compile(r"\bnot (enough|sufficient) (information|context)\b", re.IGNORECASE),
    re.compile(r"\bcannot be (answered|determined) (from|based on)\b", re.IGNORECASE),
]

_ABSTAIN_CLAUSE = (
    f"\nIf the context does not contain the answer, reply exactly: {ABSTAIN_TOKEN}\n"
)

TEMPLATES: dict[str, str] = {
    # Formulation neutre. Sert de reference : toute autre variante doit
    # justifier son ecart par rapport a celle-ci.
    "default": """Answer the question using only the context below.
{abstain}
Context:
{context}

Question: {question}

Answer:""",
    # Force la citation du passage. Hypothese testable : ameliore la
    # faithfulness au prix de la fluidite, et rend l'attribution verifiable.
    "cited": """Answer the question using only the context below.
Cite the passage number in brackets after each claim, like [2].
{abstain}
Context:
{context}

Question: {question}

Answer:""",
    # Decomposition explicite. Cible les questions multi-hop, ou le modele
    # echoue souvent moins par manque de contexte que par raccourci de
    # raisonnement.
    "decompose": """Answer the question using only the context below.
First list the intermediate facts you need, then give the final answer
after the marker "ANSWER:".
{abstain}
Context:
{context}

Question: {question}

Facts:""",
    # Temoin sans retrieval (closed-book). Config d'ablation indispensable :
    # sans elle, on ne sait pas si le RAG apporte quoi que ce soit par
    # rapport au modele seul. Le contexte est ignore.
    "closed_book": """Answer the question from your own knowledge.
{abstain}
Question: {question}

Answer:""",
}


@dataclass
class GenerationResult:
    answer: str
    abstained: bool
    elapsed_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def format_context(contexts: list[Context], max_chars: int = 12000) -> str:
    """Numerote les passages et borne la taille totale.

    La numerotation sert au template "cited" et au drill-down manuel. La
    borne evite qu'un top_k eleve fasse deborder la fenetre de contexte et
    transforme une comparaison de retrieval en comparaison de troncature.
    """
    parts: list[str] = []
    total = 0
    for i, ctx in enumerate(contexts, start=1):
        block = f"[{i}] {ctx.text.strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def detect_abstention(answer: str) -> bool:
    return any(p.search(answer) for p in _ABSTAIN_PATTERNS)


def build_prompt(cfg: PipelineConfig, question: str, contexts: list[Context]) -> str:
    template = TEMPLATES.get(cfg.generation.prompt_template)
    if template is None:
        raise ValueError(
            f"template inconnu : {cfg.generation.prompt_template} "
            f"(disponibles : {', '.join(sorted(TEMPLATES))})"
        )
    return template.format(
        abstain=_ABSTAIN_CLAUSE if cfg.generation.allow_abstain else "",
        context=format_context(contexts),
        question=question,
    )


async def generate(
    llm: LLMClient, cfg: PipelineConfig, *, question: str, contexts: list[Context]
) -> GenerationResult:
    started = time.perf_counter()

    if not contexts and cfg.generation.prompt_template != "closed_book":
        # Retrieval vide : on n'appelle pas le generateur. Le faire
        # produirait une hallucination pure qui polluerait les metriques de
        # fidelite sans rien apprendre sur le generateur.
        return GenerationResult(
            answer=ABSTAIN_TOKEN,
            abstained=True,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    prompt = build_prompt(cfg, question, contexts)
    try:
        comp = await llm.complete(
            prompt,
            model=cfg.models.generator,
            role="generator",
            temperature=cfg.generation.temperature,
            max_tokens=cfg.generation.max_tokens,
            seed=cfg.seed,
        )
    except Exception as exc:  # noqa: BLE001 - l'erreur est stockee, pas avalee
        return GenerationResult(
            answer="",
            abstained=False,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            error=str(exc),
        )

    answer = comp.text
    if cfg.generation.prompt_template == "decompose" and "ANSWER:" in answer:
        answer = answer.split("ANSWER:", 1)[1].strip()

    return GenerationResult(
        answer=answer,
        abstained=detect_abstention(answer),
        elapsed_ms=comp.latency_ms,
        prompt_tokens=comp.prompt_tokens,
        completion_tokens=comp.completion_tokens,
    )

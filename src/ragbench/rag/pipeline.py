"""Le pipeline RAG complet : question -> passages -> reponse.

Une seule fonction publique, `answer()`, qui est le seul endroit du projet
ou retrieval et generation sont enchaines. Tout le reste (evaluateurs,
runner, UI) travaille sur son resultat, jamais sur ses etapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import psycopg

from ..config import PipelineConfig
from ..llm import LLMClient
from .generate import GenerationResult, generate
from .retrieve import Context, retrieve


@dataclass
class RagAnswer:
    question: str
    contexts: list[Context]
    answer: str
    abstained: bool
    retrieval_ms: int
    generation_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def as_prediction_row(self, question_id: int) -> dict[str, Any]:
        return {
            "question_id": question_id,
            "contexts": [c.as_dict() for c in self.contexts],
            "answer": self.answer,
            "abstained": self.abstained,
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


async def answer(
    conn: psycopg.Connection,
    llm: LLMClient,
    cfg: PipelineConfig,
    *,
    index_id: int,
    question: str,
) -> RagAnswer:
    # Le temoin closed-book saute entierement le retrieval : c'est le point
    # de comparaison qui dit si le RAG apporte quelque chose.
    if cfg.generation.prompt_template == "closed_book":
        gen: GenerationResult = await generate(llm, cfg, question=question, contexts=[])
        return RagAnswer(
            question=question,
            contexts=[],
            answer=gen.answer,
            abstained=gen.abstained,
            retrieval_ms=0,
            generation_ms=gen.elapsed_ms,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            error=gen.error,
        )

    retrieval = await retrieve(conn, llm, cfg, index_id=index_id, question=question)
    gen = await generate(llm, cfg, question=question, contexts=retrieval.contexts)

    return RagAnswer(
        question=question,
        contexts=retrieval.contexts,
        answer=gen.answer,
        abstained=gen.abstained,
        retrieval_ms=retrieval.elapsed_ms,
        generation_ms=gen.elapsed_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        error=gen.error,
        debug=retrieval.debug,
    )


__all__ = ["RagAnswer", "answer", "Context"]

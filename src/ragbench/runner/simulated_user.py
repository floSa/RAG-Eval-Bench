"""Utilisateur simule, pour evaluer la clarification.

La clarification est interactive : le systeme pose une question, un humain
repond. Pour la mesurer sur 200 questions, on remplace l'humain par un LLM qui
joue l'auteur de la question.

Ce que l'utilisateur simule SAIT : quels articles il a en tete (titre, source,
date des documents de la verite terrain) — exactement ce qu'un vrai
utilisateur sait de sa propre intention.
Ce qu'il NE SAIT PAS : la reponse. On ne lui donne ni la reponse de reference
ni les faits, et la consigne lui interdit de repondre a sa propre question.

Limite assumee : un vrai utilisateur ne connait pas toujours le titre exact
de l'article qu'il cherche. Le simule, lui, le connait ; le gain mesure est
une BORNE HAUTE de ce que la clarification apporterait en pratique.

Le modele est le juge, pas le generateur : l'utilisateur simule fait partie
de l'appareil de mesure, qui reste fixe quand les techniques changent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import psycopg

from ..config import PipelineConfig
from ..llm import LLMClient

UserReply = Callable[[str], Awaitable[str]]

_USER_PROMPT = """You asked a search assistant the question below. You know which news
articles you have in mind, but you do not know the answer.

Your question: {question}

The articles you have in mind:
{articles}

The assistant asks you: {clarifying_question}

Reply in ONE short sentence, only with what you know about which articles you
mean (source, date, topic). Never answer your own question.
Reply:"""


def load_document_meta(
    conn: psycopg.Connection, external_ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    ids = sorted(set(external_ids))
    if not ids:
        return {}
    rows = conn.execute(
        """
        SELECT external_id, title, metadata->>'source' AS source,
               metadata->>'published_at' AS published_at
        FROM documents WHERE external_id = ANY(%s)
        """,
        (ids,),
    ).fetchall()
    return {r["external_id"]: r for r in rows}


def describe_articles(gold_evidence: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for doc_id in dict.fromkeys(ev.get("document_external_id") for ev in gold_evidence):
        m = meta.get(doc_id or "")
        if not m:
            continue
        date = (m.get("published_at") or "")[:10]
        lines.append(f"- {m.get('source') or '?'}, {date}: {m.get('title') or '?'}")
    return "\n".join(lines) or "- (no specific article)"


def simulated_user(
    llm: LLMClient,
    cfg: PipelineConfig,
    *,
    question: str,
    gold_evidence: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
) -> UserReply:
    articles = describe_articles(gold_evidence, meta)

    async def reply(clarifying_question: str) -> str:
        comp = await llm.complete(
            _USER_PROMPT.format(
                question=question, articles=articles, clarifying_question=clarifying_question
            ),
            model=cfg.models.judge,
            role="simulated_user",
            temperature=0.0,
            max_tokens=60,
            thinking=False,
        )
        return (comp.text or "").strip()

    return reply

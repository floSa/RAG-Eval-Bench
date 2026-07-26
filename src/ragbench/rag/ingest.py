"""Indexation d'un corpus : decoupage, embedding, stockage.

Un index est identifie par PipelineConfig.index_hash() = f(chunking,
embedder). Toute config qui partage ce hash reutilise l'index existant :
comparer top_k=3 et top_k=10 ne doit pas coder 600 documents deux fois.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from .. import db
from ..config import PipelineConfig
from ..llm import LLMClient
from .chunking import chunk_document


@dataclass
class IndexReport:
    index_id: int
    index_hash: str
    reused: bool
    n_documents: int
    n_chunks: int
    dim: int | None


async def ensure_index(
    conn: psycopg.Connection,
    llm: LLMClient,
    cfg: PipelineConfig,
    *,
    dataset_id: int,
    force: bool = False,
    progress=None,
) -> IndexReport:
    index_hash = cfg.index_hash()
    index_id, populated = db.get_or_create_index(
        conn,
        dataset_id=dataset_id,
        index_hash=index_hash,
        chunking=cfg.chunking.model_dump(),
        embedder=cfg.models.embedder,
    )
    conn.commit()

    if populated and not force:
        row = conn.execute(
            "SELECT n_chunks, dim FROM corpus_indexes WHERE id = %s", (index_id,)
        ).fetchone()
        n_docs = conn.execute(
            "SELECT count(*) AS n FROM documents WHERE dataset_id = %s", (dataset_id,)
        ).fetchone()["n"]
        return IndexReport(index_id, index_hash, True, n_docs, row["n_chunks"], row["dim"])

    if force:
        conn.execute("DELETE FROM chunks WHERE index_id = %s", (index_id,))
        conn.commit()

    documents = conn.execute(
        "SELECT id, external_id, title, body FROM documents WHERE dataset_id = %s ORDER BY id",
        (dataset_id,),
    ).fetchall()

    dim: int | None = None
    total_chunks = 0
    # Traitement par lots de documents : garde la memoire bornee et rend
    # une ingestion interrompue reprenable (les chunks deja ecrits sont
    # commites).
    batch_size = 25

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        pending: list[dict] = []

        for doc in batch:
            # Le titre est prefixe au corps avant decoupage : sur un corpus
            # de presse, il porte souvent l'entite nommee qui rend le
            # passage retrouvable.
            body = f"{doc['title']}\n\n{doc['body']}" if doc["title"] else doc["body"]
            for ordinal, text in enumerate(chunk_document(body, cfg.chunking)):
                pending.append({"document_id": doc["id"], "ordinal": ordinal, "text": text})

        if not pending:
            continue

        vectors = await llm.embed(
            [f"{cfg.chunking.document_prefix}{p['text']}" for p in pending],
            model=cfg.models.embedder,
        )
        if dim is None and vectors:
            dim = len(vectors[0])

        for p, vec in zip(pending, vectors):
            p["embedding"] = vec

        db.insert_chunks(conn, index_id, pending)
        conn.commit()
        total_chunks += len(pending)

        if progress is not None:
            progress(min(start + batch_size, len(documents)), len(documents), total_chunks)

    db.finalize_index(conn, index_id, dim or 0)
    conn.commit()

    return IndexReport(index_id, index_hash, False, len(documents), total_chunks, dim)

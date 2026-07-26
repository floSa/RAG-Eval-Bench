"""Acces Postgres/pgvector : connexion, migration, ecritures du banc.

Volontairement en SQL nu plutot qu'en ORM. Les requetes de retrieval sont
la chose qu'on veut pouvoir lire et modifier sans mediation — c'est la
variable experimentale principale du projet.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from importlib import resources
from typing import Any, Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .settings import Settings, settings as default_settings


@contextmanager
def connect(cfg: Settings | None = None, *, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    cfg = cfg or default_settings
    with psycopg.connect(cfg.dsn, row_factory=dict_row, autocommit=autocommit) as conn:
        register_vector(conn)
        yield conn


def ensure_database(cfg: Settings | None = None) -> bool:
    """Cree la base cible si absente. Renvoie True si elle a ete creee."""
    cfg = cfg or default_settings
    with psycopg.connect(cfg.admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (cfg.db_name,))
            if cur.fetchone():
                return False
            cur.execute(f'CREATE DATABASE "{cfg.db_name}"')
            return True


def migrate(cfg: Settings | None = None) -> None:
    """Applique schema.sql. Idempotent (tout est IF NOT EXISTS / OR REPLACE).

    Connexion volontairement sans register_vector : c'est ce script qui cree
    l'extension, donc le type `vector` n'existe pas encore au moment ou l'on
    se connecte.
    """
    cfg = cfg or default_settings
    sql = resources.files("ragbench").joinpath("schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(cfg.dsn, autocommit=True) as conn:
        conn.execute(sql)


# ---------------------------------------------------------------------
# Ecritures — chaque helper est idempotent sur sa cle naturelle, pour
# qu'une ingestion interrompue puisse etre relancee sans nettoyage.
# ---------------------------------------------------------------------


def upsert_dataset(
    conn: psycopg.Connection,
    *,
    name: str,
    source: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO datasets (name, source, description, metadata)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name) DO UPDATE
            SET source = EXCLUDED.source,
                description = EXCLUDED.description,
                metadata = EXCLUDED.metadata
        RETURNING id
        """,
        (name, source, description, Jsonb(metadata or {})),
    ).fetchone()
    return row["id"]


def get_dataset(conn: psycopg.Connection, name: str) -> dict[str, Any] | None:
    return conn.execute("SELECT * FROM datasets WHERE name = %s", (name,)).fetchone()


def upsert_documents(
    conn: psycopg.Connection, dataset_id: int, docs: list[dict[str, Any]]
) -> dict[str, int]:
    """Insere des documents et renvoie {external_id: document_id}."""
    if not docs:
        return {}
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (dataset_id, external_id, title, body, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (dataset_id, external_id) DO UPDATE
                SET title = EXCLUDED.title,
                    body = EXCLUDED.body,
                    metadata = EXCLUDED.metadata
            """,
            [
                (
                    dataset_id,
                    d["external_id"],
                    d.get("title"),
                    d["body"],
                    Jsonb(d.get("metadata") or {}),
                )
                for d in docs
            ],
        )
    rows = conn.execute(
        "SELECT external_id, id FROM documents WHERE dataset_id = %s", (dataset_id,)
    ).fetchall()
    return {r["external_id"]: r["id"] for r in rows}


def upsert_questions(
    conn: psycopg.Connection, dataset_id: int, questions: list[dict[str, Any]]
) -> int:
    if not questions:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO questions
                (dataset_id, external_id, question, answer, question_type,
                 gold_evidence, metadata, split)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_id, external_id) DO UPDATE
                SET question = EXCLUDED.question,
                    answer = EXCLUDED.answer,
                    question_type = EXCLUDED.question_type,
                    gold_evidence = EXCLUDED.gold_evidence,
                    metadata = EXCLUDED.metadata,
                    split = EXCLUDED.split
            """,
            [
                (
                    dataset_id,
                    q["external_id"],
                    q["question"],
                    q.get("answer"),
                    q.get("question_type"),
                    Jsonb(q.get("gold_evidence") or []),
                    Jsonb(q.get("metadata") or {}),
                    q.get("split", "eval"),
                )
                for q in questions
            ],
        )
    return len(questions)


def get_or_create_index(
    conn: psycopg.Connection,
    *,
    dataset_id: int,
    index_hash: str,
    chunking: dict[str, Any],
    embedder: str,
) -> tuple[int, bool]:
    """Renvoie (index_id, deja_peuple).

    `deja_peuple` evite de reindexer un corpus quand deux configs ne
    different que par un parametre de retrieval.
    """
    existing = conn.execute(
        "SELECT id, n_chunks FROM corpus_indexes WHERE dataset_id = %s AND index_hash = %s",
        (dataset_id, index_hash),
    ).fetchone()
    if existing:
        return existing["id"], existing["n_chunks"] > 0

    row = conn.execute(
        """
        INSERT INTO corpus_indexes (dataset_id, index_hash, chunking, embedder)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (dataset_id, index_hash, Jsonb(chunking), embedder),
    ).fetchone()
    return row["id"], False


def insert_chunks(conn: psycopg.Connection, index_id: int, rows: list[dict[str, Any]]) -> None:
    """rows: [{document_id, ordinal, text, embedding}]"""
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (index_id, document_id, ordinal, text, embedding, tsv)
            VALUES (%s, %s, %s, %s, %s, to_tsvector('english', %s))
            ON CONFLICT (index_id, document_id, ordinal) DO UPDATE
                SET text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    tsv = EXCLUDED.tsv
            """,
            [
                (index_id, r["document_id"], r["ordinal"], r["text"], r["embedding"], r["text"])
                for r in rows
            ],
        )


def finalize_index(conn: psycopg.Connection, index_id: int, dim: int) -> None:
    conn.execute(
        """
        UPDATE corpus_indexes
        SET n_chunks = (SELECT count(*) FROM chunks WHERE index_id = %s), dim = %s
        WHERE id = %s
        """,
        (index_id, dim, index_id),
    )


def upsert_config(
    conn: psycopg.Connection,
    *,
    hash_: str,
    name: str,
    description: str,
    payload: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO configs (hash, name, description, payload)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (hash) DO UPDATE SET name = EXCLUDED.name
        """,
        (hash_, name, description, Jsonb(payload)),
    )


def create_run(
    conn: psycopg.Connection,
    *,
    dataset_id: int,
    config_hash: str,
    index_id: int | None,
    label: str | None,
    git_sha: str | None,
    warnings: list[str],
) -> int:
    row = conn.execute(
        """
        INSERT INTO runs (dataset_id, config_hash, index_id, label, git_sha, warnings)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (dataset_id, config_hash, index_id, label, git_sha, Jsonb(warnings)),
    ).fetchone()
    return row["id"]


def finish_run(
    conn: psycopg.Connection,
    run_id: int,
    *,
    status: str,
    n_questions: int,
    n_failed: int,
    usage: dict[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = %s, n_questions = %s, n_failed = %s, usage = %s, finished_at = now()
        WHERE id = %s
        """,
        (status, n_questions, n_failed, Jsonb(usage), run_id),
    )


def insert_predictions(conn: psycopg.Connection, run_id: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO predictions
                (run_id, question_id, contexts, answer, abstained,
                 retrieval_ms, generation_ms, prompt_tokens, completion_tokens, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, question_id) DO UPDATE
                SET contexts = EXCLUDED.contexts,
                    answer = EXCLUDED.answer,
                    abstained = EXCLUDED.abstained,
                    retrieval_ms = EXCLUDED.retrieval_ms,
                    generation_ms = EXCLUDED.generation_ms,
                    prompt_tokens = EXCLUDED.prompt_tokens,
                    completion_tokens = EXCLUDED.completion_tokens,
                    error = EXCLUDED.error
            """,
            [
                (
                    run_id,
                    r["question_id"],
                    Jsonb(r.get("contexts") or []),
                    r.get("answer"),
                    r.get("abstained", False),
                    r.get("retrieval_ms"),
                    r.get("generation_ms"),
                    r.get("prompt_tokens", 0),
                    r.get("completion_tokens", 0),
                    r.get("error"),
                )
                for r in rows
            ],
        )


def insert_scores(conn: psycopg.Connection, run_id: int, rows: list[dict[str, Any]]) -> None:
    """rows: [{question_id|None, evaluator, metric, value, detail}]"""
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO scores (run_id, question_id, evaluator, metric, value, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, COALESCE(question_id, -1), evaluator, metric) DO UPDATE
                SET value = EXCLUDED.value, detail = EXCLUDED.detail
            """,
            [
                (
                    run_id,
                    r.get("question_id"),
                    r["evaluator"],
                    r["metric"],
                    r.get("value"),
                    Jsonb(r.get("detail") or {}),
                )
                for r in rows
            ],
        )


def load_predictions(conn: psycopg.Connection, run_id: int) -> list[dict[str, Any]]:
    """Predictions jointes a leur question — l'entree de tous les evaluateurs."""
    return conn.execute(
        """
        SELECT p.question_id, p.contexts, p.answer, p.abstained, p.error,
               q.external_id, q.question, q.answer AS gold_answer,
               q.question_type, q.gold_evidence
        FROM predictions p
        JOIN questions q ON q.id = p.question_id
        WHERE p.run_id = %s
        ORDER BY p.question_id
        """,
        (run_id,),
    ).fetchall()


def json_default(obj: Any) -> Any:
    """Serialisation tolerante pour les detail JSONB des evaluateurs."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return json.JSONEncoder().default(obj)

"""Execution d'une campagne : une config x un jeu de questions -> un run.

Choix d'implementation notables :

- UNE CONNEXION POSTGRES PAR WORKER. Une connexion psycopg n'est pas sure
  en usage entrelace : un worker qui attend le LLM pendant qu'un autre
  execute une requete sur la meme connexion finit par corrompre l'etat de
  transaction. Le cout d'une poignee de connexions est negligeable devant
  celui d'un run fausse.

- LES PREDICTIONS SONT ECRITES PAR LOTS, AU FIL DE L'EAU. Une campagne dure
  des dizaines de minutes ; tout garder en memoire pour ecrire a la fin
  transforme n'importe quelle interruption en perte totale.

- LE RUN EST CREE AVANT DE COMMENCER, avec ses avertissements. Un run
  interrompu reste visible avec le statut 'running' : c'est une information,
  pas un dechet a cacher.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psycopg

from .. import db
from ..config import PipelineConfig
from ..llm import LLMClient
from ..rag import pipeline
from ..rag.ingest import ensure_index
from ..settings import Settings
from ..settings import settings as default_settings


@dataclass
class CampaignReport:
    run_id: int
    config_name: str
    config_hash: str
    index_id: int
    n_questions: int
    n_failed: int
    elapsed_s: float
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_questions(
    conn: psycopg.Connection, dataset_id: int, *, split: str = "eval", limit: int | None = None
) -> list[dict[str, Any]]:
    # Tri par external_id et non par id : l'ordre reste identique meme si
    # le dataset est recharge, donc un `limit` designe toujours les memes
    # questions.
    sql = """
        SELECT id, external_id, question, answer, question_type, gold_evidence
        FROM questions
        WHERE dataset_id = %s AND split = %s
        ORDER BY external_id
    """
    params: list[Any] = [dataset_id, split]
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


async def run_campaign(
    cfg: PipelineConfig,
    *,
    dataset_name: str,
    split: str = "eval",
    limit: int | None = None,
    label: str | None = None,
    settings: Settings | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_index_progress: Callable[[int, int, int], None] | None = None,
) -> CampaignReport:
    settings = settings or default_settings
    started = time.perf_counter()

    async with LLMClient(settings) as llm:
        # --- preparation (connexion unique, sequentiel) ------------------
        with db.connect(settings) as conn:
            dataset = db.get_dataset(conn, dataset_name)
            if dataset is None:
                raise ValueError(
                    f"dataset '{dataset_name}' absent — lancer `ragbench dataset load {dataset_name}`"
                )
            dataset_id = dataset["id"]

            index = await ensure_index(
                conn, llm, cfg, dataset_id=dataset_id, progress=on_index_progress
            )

            db.upsert_config(
                conn,
                hash_=cfg.hash(),
                name=cfg.name,
                description=cfg.description,
                payload=cfg.payload(),
            )
            warnings = cfg.warnings()
            run_id = db.create_run(
                conn,
                dataset_id=dataset_id,
                config_hash=cfg.hash(),
                index_id=index.index_id,
                label=label or cfg.name,
                git_sha=git_sha(),
                warnings=warnings,
            )
            questions = load_questions(conn, dataset_id, split=split, limit=limit)
            conn.commit()

        # --- execution (N workers, N connexions) -------------------------
        queue: asyncio.Queue = asyncio.Queue()
        for q in questions:
            queue.put_nowait(q)

        done = 0
        failed = 0
        lock = asyncio.Lock()
        n_workers = max(1, min(settings.llm_concurrency, len(questions)))

        async def worker() -> None:
            nonlocal done, failed
            buffer: list[dict[str, Any]] = []
            with db.connect(settings) as wconn:
                while True:
                    try:
                        question = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    try:
                        result = await pipeline.answer(
                            wconn, llm, cfg,
                            index_id=index.index_id,
                            question=question["question"],
                        )
                        row = result.as_prediction_row(question["id"])
                    except Exception as exc:  # noqa: BLE001
                        # Une question qui echoue ne doit pas emporter la
                        # campagne : elle est stockee avec son erreur et
                        # comptee dans n_failed.
                        row = {
                            "question_id": question["id"],
                            "contexts": [],
                            "answer": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }

                    buffer.append(row)
                    if row.get("error"):
                        async with lock:
                            failed += 1

                    if len(buffer) >= 10:
                        db.insert_predictions(wconn, run_id, buffer)
                        wconn.commit()
                        buffer.clear()

                    async with lock:
                        done += 1
                        if on_progress:
                            on_progress(done, len(questions))

                if buffer:
                    db.insert_predictions(wconn, run_id, buffer)
                    wconn.commit()

        await asyncio.gather(*(worker() for _ in range(n_workers)))

        with db.connect(settings) as conn:
            db.finish_run(
                conn, run_id,
                status="completed",
                n_questions=len(questions),
                n_failed=failed,
                usage=llm.usage.as_dict(),
            )
            conn.commit()

        return CampaignReport(
            run_id=run_id,
            config_name=cfg.name,
            config_hash=cfg.hash(),
            index_id=index.index_id,
            n_questions=len(questions),
            n_failed=failed,
            elapsed_s=time.perf_counter() - started,
            warnings=warnings,
            usage=llm.usage.as_dict(),
        )

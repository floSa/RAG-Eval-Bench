"""Application des evaluateurs a un run deja produit.

Separer l'evaluation de l'execution est un choix de fond, pas de confort :

- on peut rejouer une evaluation sur d'anciens runs quand on ajoute un
  framework, sans reconsommer le GPU pour regenerer les reponses ;
- on peut faire noter les MEMES predictions par plusieurs frameworks, ce
  qui est la seule facon de mesurer leur desaccord ;
- l'ordonnancement par role est respecte : toutes les generations d'abord,
  tous les jugements ensuite. Sur un GPU 16 Go, alterner generateur et juge
  question par question fait swapper les modeles en permanence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .. import db, evaluators
from ..config import PipelineConfig
from ..llm import LLMClient
from ..settings import Settings
from ..settings import settings as default_settings


@dataclass
class EvalReport:
    run_id: int
    evaluator: str
    n_scores: int
    elapsed_s: float
    problems: list[str] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)
    # Metriques demandees dont aucune question n'a pu etre notee.
    empty: list[str] = field(default_factory=list)
    skipped: bool = False


def load_run(conn, run_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT r.*, c.payload, c.name AS config_name, d.name AS dataset_name, d.metadata AS dataset_metadata
        FROM runs r
        JOIN configs c ON c.hash = r.config_hash
        JOIN datasets d ON d.id = r.dataset_id
        WHERE r.id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"run {run_id} introuvable")
    return row


async def evaluate_run(
    run_id: int,
    evaluator_names: list[str],
    *,
    settings: Settings | None = None,
    options: dict[str, Any] | None = None,
) -> list[EvalReport]:
    settings = settings or default_settings
    reports: list[EvalReport] = []

    with db.connect(settings) as conn:
        run = load_run(conn, run_id)
        predictions = db.load_predictions(conn, run_id)

    if not predictions:
        raise ValueError(f"run {run_id} : aucune prediction (campagne interrompue ?)")

    # La config est relue depuis la base, pas depuis le YAML : le fichier a
    # pu changer depuis. Un run doit toujours etre evalue avec la config
    # qui l'a produit.
    cfg = PipelineConfig.model_validate({"name": run["config_name"], **run["payload"]})

    # Seule exception : le JUGE peut etre remplace. C'est la manipulation
    # centrale du projet — faire noter les MEMES predictions par plusieurs
    # juges pour mesurer leur desaccord. Le pipeline evalue, lui, reste
    # exactement celui qui a produit les predictions.
    #
    # Le nom du juge est reporte dans le detail de chaque score, donc deux
    # jugements du meme run par deux juges differents restent distinguables
    # en base sans ambiguite.
    judge_override = options.get("judge") if options else None
    if judge_override:
        cfg = cfg.model_copy(
            update={"models": cfg.models.model_copy(update={"judge": str(judge_override)})}
        )

    async with LLMClient(settings) as llm:
        ctx = evaluators.EvalContext(
            predictions=predictions,
            config=cfg,
            llm=llm,
            dataset_metadata=run.get("dataset_metadata") or {},
            options=options or {},
        )

        for name in evaluator_names:
            started = time.perf_counter()
            evaluator = evaluators.get(name)
            problems = evaluators.check_requirements(evaluator, ctx)

            blocking = [p for p in problems if not p.startswith("[avertissement]")]
            if blocking:
                reports.append(
                    EvalReport(run_id, name, 0, 0.0, problems=problems, skipped=True)
                )
                continue

            scores = await evaluator.evaluate(ctx)

            # Quand un juge est impose, il entre dans le NOM de l'evaluateur
            # stocke. Sans ca, une seconde passe avec un autre juge ecraserait
            # la premiere (la table scores est unique sur run/question/
            # evaluateur/metrique) — et on perdrait precisement l'information
            # qu'on cherchait : leur desaccord.
            stored_name = f"{name}@{judge_override}" if judge_override else name

            with db.connect(settings) as conn:
                db.insert_scores(
                    conn,
                    run_id,
                    [
                        {
                            "question_id": s.question_id,
                            "evaluator": stored_name,
                            "metric": s.metric,
                            "value": s.value,
                            "detail": s.detail,
                        }
                        for s in scores
                    ],
                )
                conn.commit()

            reports.append(
                EvalReport(
                    run_id=run_id,
                    evaluator=stored_name,
                    n_scores=len(scores),
                    elapsed_s=time.perf_counter() - started,
                    problems=problems,
                    aggregates={
                        s.metric: s.value
                        for s in scores
                        if s.question_id is None and s.value is not None
                    },
                    # Metriques dont AUCUNE question n'a pu etre notee. Les
                    # omettre les ferait disparaitre du rapport, et une
                    # metrique absente se lit comme une metrique non demandee
                    # — alors qu'elle a ete demandee et a echoue. C'est
                    # precisement ce qu'il faut voir : le juge n'est pas
                    # capable de produire ce jugement.
                    empty=[
                        s.metric
                        for s in scores
                        if s.question_id is None
                        and s.value is None
                        and s.metric.startswith("mean_")
                    ],
                )
            )

    return reports

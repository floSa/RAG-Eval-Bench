"""Acces aux donnees pour le tableau de bord.

Separe de app.py pour deux raisons : les requetes restent testables sans
Streamlit, et le cache est declare ici une bonne fois plutot que disperse
dans l'interface.

Aucune connexion n'est mise en cache — seulement des RESULTATS. Une
connexion psycopg gardee dans un cache Streamlit est partagee entre les
sessions et finit par etre utilisee de facon entrelacee.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .. import db, runner


@st.cache_data(ttl=30)
def list_runs() -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.label, c.name AS config, r.config_hash, d.name AS dataset,
                   r.status, r.n_questions, r.n_failed, r.started_at, r.finished_at,
                   r.warnings, r.git_sha
            FROM runs r
            JOIN configs c ON c.hash = r.config_hash
            JOIN datasets d ON d.id = r.dataset_id
            ORDER BY r.id DESC
            """
        ).fetchall()
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def run_config(run_id: int) -> dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT c.payload FROM runs r JOIN configs c ON c.hash = r.config_hash WHERE r.id = %s",
            (run_id,),
        ).fetchone()
    return row["payload"] if row else {}


@st.cache_data(ttl=30)
def run_metrics(run_id: int) -> pd.DataFrame:
    """Metriques par question, agregees avec leur intervalle de confiance."""
    summary = runner.summarize(run_id)
    rows = [
        {
            "metrique": name,
            "moyenne": ci.mean,
            "ic_bas": ci.low,
            "ic_haut": ci.high,
            "demi_largeur": ci.half_width,
            "n": ci.n,
        }
        for name, ci in sorted(summary.metrics.items())
    ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def run_aggregates(run_id: int) -> pd.DataFrame:
    summary = runner.summarize(run_id)
    return pd.DataFrame(
        [{"agregat": k, "valeur": v} for k, v in sorted(summary.aggregates.items())]
    )


@st.cache_data(ttl=30)
def comparable_metrics(run_a: int, run_b: int) -> list[str]:
    with db.connect() as conn:
        a = {f"{e}/{m}" for e, m, _ in runner.available_metrics(conn, run_a)}
        b = {f"{e}/{m}" for e, m, _ in runner.available_metrics(conn, run_b)}
    return sorted(a & b)


@st.cache_data(ttl=30)
def comparison_table(run_a: int, run_b: int) -> pd.DataFrame:
    results = runner.compare_all(run_a, run_b)
    rows = []
    for res in results:
        c = res.comparison
        rows.append(
            {
                "metrique": f"{res.evaluator}/{res.metric}",
                "A": c.mean_a,
                "B": c.mean_b,
                "ecart": c.delta,
                "ic_bas": c.ci_low,
                "ic_haut": c.ci_high,
                "p": c.p_value,
                "n": c.n_pairs,
                "B_gagne": c.n_better,
                "A_gagne": c.n_worse,
                "ex_aequo": c.n_tied,
                # Trois etats et non deux : « limite » signale que
                # l'intervalle bootstrap et le test de permutation ne
                # concordent pas. C'est la zone ou l'on decide a tort avec
                # assurance, elle merite son propre libelle.
                "verdict": (
                    "limite" if c.borderline else ("concluant" if c.significant else "non")
                ),
                "concluant": c.significant and not c.borderline,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def predictions(run_id: int) -> pd.DataFrame:
    """Predictions enrichies des scores par question, pour le drill-down."""
    with db.connect() as conn:
        preds = db.load_predictions(conn, run_id)
        scores = conn.execute(
            """
            SELECT question_id, evaluator || '/' || metric AS metric, value
            FROM scores
            WHERE run_id = %s AND question_id IS NOT NULL AND value IS NOT NULL
            """,
            (run_id,),
        ).fetchall()
        debug = conn.execute(
            """
            SELECT question_id, detail FROM scores
            WHERE run_id = %s AND metric = 'retrieval_debug'
            """,
            (run_id,),
        ).fetchall()

    by_question: dict[int, dict[str, Any]] = {}
    for score in scores:
        by_question.setdefault(score["question_id"], {})[score["metric"]] = score["value"]
    debug_by_question = {d["question_id"]: d["detail"] for d in debug}

    rows = []
    for pred in preds:
        row = {
            "question_id": pred["question_id"],
            "external_id": pred["external_id"],
            "type": pred["question_type"],
            "question": pred["question"],
            "gold": pred["gold_answer"],
            "reponse": pred["answer"],
            "abstenu": pred["abstained"],
            "erreur": pred["error"],
            "n_contextes": len(pred["contexts"] or []),
            "sans_reponse": not pred["gold_evidence"],
            "_contexts": pred["contexts"],
            "_gold_evidence": pred["gold_evidence"],
            "_debug": debug_by_question.get(pred["question_id"], {}),
        }
        row.update(by_question.get(pred["question_id"], {}))
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def datasets_overview() -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT d.name, d.description,
                   (SELECT count(*) FROM documents WHERE dataset_id = d.id) AS documents,
                   (SELECT count(*) FROM questions WHERE dataset_id = d.id) AS questions,
                   (SELECT count(*) FROM questions
                     WHERE dataset_id = d.id AND split = 'eval') AS split_eval,
                   (SELECT count(*) FROM questions
                     WHERE dataset_id = d.id AND gold_evidence = '[]'::jsonb) AS sans_reponse,
                   (SELECT count(*) FROM corpus_indexes WHERE dataset_id = d.id) AS index
            FROM datasets d ORDER BY d.name
            """
        ).fetchall()
    return pd.DataFrame(rows)

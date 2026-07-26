"""Lecture et comparaison statistique des runs.

Le point delicat est l'APPARIEMENT. Deux runs peuvent ne pas porter sur
exactement les memes questions : l'un a ete lance avec --limit, l'autre
non ; ou une metrique n'a pas pu etre calculee partout (retrieval exclu sur
les questions sans reponse, jugement non parsable cote Ragas). Comparer
leurs moyennes brutes reviendrait alors a comparer deux jeux differents.

Toutes les fonctions ici travaillent donc sur l'INTERSECTION des
question_id, et renvoient combien de questions ont ete ecartees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .. import db, stats
from ..settings import Settings
from ..settings import settings as default_settings


@dataclass
class MetricSeries:
    run_id: int
    evaluator: str
    metric: str
    by_question: dict[int, float]

    def aligned_with(self, other: MetricSeries) -> tuple[list[float], list[float], list[int]]:
        """Valeurs des deux series sur leurs questions communes, meme ordre."""
        common = sorted(set(self.by_question) & set(other.by_question))
        return (
            [self.by_question[q] for q in common],
            [other.by_question[q] for q in common],
            common,
        )


@dataclass
class RunSummary:
    run_id: int
    label: str
    config_name: str
    config_hash: str
    dataset: str
    status: str
    n_questions: int
    n_failed: int
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    # metrique -> intervalle de confiance bootstrap
    metrics: dict[str, stats.Interval] = field(default_factory=dict)
    # agregats de run (question_id NULL) : accuracy par type, taux, etc.
    aggregates: dict[str, float] = field(default_factory=dict)


def metric_series(
    conn, run_id: int, evaluator: str, metric: str
) -> MetricSeries:
    rows = conn.execute(
        """
        SELECT question_id, value FROM scores
        WHERE run_id = %s AND evaluator = %s AND metric = %s
          AND question_id IS NOT NULL AND value IS NOT NULL
        """,
        (run_id, evaluator, metric),
    ).fetchall()
    return MetricSeries(
        run_id, evaluator, metric, {r["question_id"]: float(r["value"]) for r in rows}
    )


def available_metrics(conn, run_id: int) -> list[tuple[str, str, int]]:
    """(evaluateur, metrique, nombre de questions notees), par question."""
    rows = conn.execute(
        """
        SELECT evaluator, metric, count(*) AS n FROM scores
        WHERE run_id = %s AND question_id IS NOT NULL AND value IS NOT NULL
        GROUP BY evaluator, metric ORDER BY evaluator, metric
        """,
        (run_id,),
    ).fetchall()
    return [(r["evaluator"], r["metric"], r["n"]) for r in rows]


def summarize(run_id: int, *, settings: Settings | None = None) -> RunSummary:
    settings = settings or default_settings
    with db.connect(settings) as conn:
        run = conn.execute(
            """
            SELECT r.*, c.name AS config_name, d.name AS dataset_name
            FROM runs r JOIN configs c ON c.hash = r.config_hash
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.id = %s
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(f"run {run_id} introuvable")

        summary = RunSummary(
            run_id=run_id,
            label=run["label"] or "",
            config_name=run["config_name"],
            config_hash=run["config_hash"],
            dataset=run["dataset_name"],
            status=run["status"],
            n_questions=run["n_questions"],
            n_failed=run["n_failed"],
            warnings=run["warnings"] or [],
            usage=run["usage"] or {},
        )

        for evaluator, metric, _ in available_metrics(conn, run_id):
            series = metric_series(conn, run_id, evaluator, metric)
            if series.by_question:
                summary.metrics[f"{evaluator}/{metric}"] = stats.bootstrap_ci(
                    list(series.by_question.values())
                )

        for row in conn.execute(
            """
            SELECT evaluator, metric, value FROM scores
            WHERE run_id = %s AND question_id IS NULL AND value IS NOT NULL
            ORDER BY evaluator, metric
            """,
            (run_id,),
        ).fetchall():
            summary.aggregates[f"{row['evaluator']}/{row['metric']}"] = float(row["value"])

    return summary


@dataclass
class ComparisonResult:
    metric: str
    evaluator: str
    run_a: int
    run_b: int
    comparison: stats.Comparison
    n_dropped_a: int
    n_dropped_b: int
    mcnemar: tuple[int, int, float] | None = None
    binary: bool = False


def compare(
    run_a: int,
    run_b: int,
    *,
    evaluator: str,
    metric: str,
    settings: Settings | None = None,
) -> ComparisonResult:
    settings = settings or default_settings
    with db.connect(settings) as conn:
        sa = metric_series(conn, run_a, evaluator, metric)
        sb = metric_series(conn, run_b, evaluator, metric)

    if not sa.by_question or not sb.by_question:
        raise ValueError(
            f"metrique '{evaluator}/{metric}' absente du run "
            f"{run_a if not sa.by_question else run_b}"
        )

    values_a, values_b, common = sa.aligned_with(sb)
    if not common:
        raise ValueError(
            f"runs {run_a} et {run_b} n'ont aucune question en commun pour "
            f"'{evaluator}/{metric}'"
        )

    # Metrique binaire : McNemar est alors plus adapte que le bootstrap,
    # parce que seules les questions discordantes portent de l'information.
    binary = all(v in (0.0, 1.0) for v in values_a + values_b)

    return ComparisonResult(
        metric=metric,
        evaluator=evaluator,
        run_a=run_a,
        run_b=run_b,
        comparison=stats.paired_bootstrap(values_a, values_b),
        n_dropped_a=len(sa.by_question) - len(common),
        n_dropped_b=len(sb.by_question) - len(common),
        mcnemar=stats.mcnemar(values_a, values_b) if binary else None,
        binary=binary,
    )


def compare_all(
    run_a: int, run_b: int, *, settings: Settings | None = None
) -> list[ComparisonResult]:
    """Compare toutes les metriques par question communes aux deux runs.

    Avertissement de lecture : enchainer N comparaisons multiplie les
    chances qu'au moins une ressorte « significative » par hasard. Avec 15
    metriques a 5 %, on en attend environ une fausse. L'UI et la CLI
    signalent ce compte plutot que de corriger en silence — une correction
    de Bonferroni sur des metriques fortement correlees entre elles serait
    trop conservatrice.
    """
    settings = settings or default_settings
    with db.connect(settings) as conn:
        metrics_a = {(e, m) for e, m, _ in available_metrics(conn, run_a)}
        metrics_b = {(e, m) for e, m, _ in available_metrics(conn, run_b)}

    out: list[ComparisonResult] = []
    for evaluator, metric in sorted(metrics_a & metrics_b):
        try:
            out.append(compare(run_a, run_b, evaluator=evaluator, metric=metric, settings=settings))
        except ValueError:
            continue
    return out


@dataclass
class Calibration:
    """Accord entre un juge et une reference, sur les memes questions."""

    evaluator: str
    metric: str
    kappa: float
    raw_agreement: float
    n: int
    # Part des questions ou le juge dit « bon » et la reference « mauvais ».
    # Un juge peut avoir un kappa correct et etre systematiquement trop
    # genereux ; les deux chiffres ne disent pas la meme chose.
    false_positive_rate: float
    false_negative_rate: float

    def verdict(self) -> str:
        if math.isnan(self.kappa):
            return "indefini (l'un des deux ne varie pas)"
        if self.kappa < 0.4:
            return "faible — ce juge ne mesure pas la meme chose que la reference"
        if self.kappa < 0.6:
            return "moyen — utilisable en tendance, pas pour trancher"
        if self.kappa < 0.8:
            return "substantiel"
        return "excellent"


REFERENCE_METRIC = ("native.answer", "contains")


def calibrate(
    run_id: int,
    *,
    reference: tuple[str, str] = REFERENCE_METRIC,
    settings: Settings | None = None,
    seed_annotations: bool = True,
) -> list[Calibration]:
    """Confronte chaque juge a une reference DETERMINISTE.

    Pourquoi c'est legitime sans annotation humaine : sur un dataset qui
    fournit les reponses gold — MultiHop-RAG en fournit — la metrique
    `native.answer/contains` n'est pas une opinion, c'est une comparaison a
    la verite terrain. Elle constitue donc une reference recevable pour
    mesurer si un juge LLM mesure bien ce qu'il pretend mesurer.

    CE QUE CA NE REMPLACE PAS. La reference porte sur la JUSTESSE de la
    reponse. Elle valide donc correctement les metriques qui pretendent
    mesurer la justesse (answer_relevancy, claim_recall), mais elle est un
    signal FAIBLE pour la fidelite : une reponse peut etre parfaitement
    fidele au contexte et fausse, ou juste et mal soutenue. Pour la
    fidelite, seule une annotation humaine tranche — d'ou l'onglet
    Annotation, qui reste necessaire.

    Les valeurs de reference sont aussi ecrites dans `annotations` sous
    l'annotateur `ground_truth`, pour qu'elles apparaissent a cote des
    annotations humaines et que les deux soient comparables.
    """
    settings = settings or default_settings

    with db.connect(settings) as conn:
        ref_series = metric_series(conn, run_id, reference[0], reference[1])
        if not ref_series.by_question:
            raise ValueError(
                f"reference '{reference[0]}/{reference[1]}' absente du run {run_id} — "
                f"lancer d'abord `ragbench eval run {run_id} -e native.answer`"
            )

        if seed_annotations:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO annotations (run_id, question_id, metric, value, annotator, notes)
                    VALUES (%s, %s, 'correct', %s, 'ground_truth', %s)
                    ON CONFLICT (run_id, question_id, metric, annotator)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    [
                        (run_id, qid, value, f"derive de {reference[0]}/{reference[1]}")
                        for qid, value in ref_series.by_question.items()
                    ],
                )
            conn.commit()

        candidates = available_metrics(conn, run_id)
        out: list[Calibration] = []

        for evaluator, metric, _ in candidates:
            if (evaluator, metric) == reference:
                continue
            series = metric_series(conn, run_id, evaluator, metric)
            ref_values, judge_values, _ = ref_series.aligned_with(series)
            if len(ref_values) < 5:
                continue
            # Les scores continus sont binarises au seuil de 0,5 : le kappa
            # est defini sur des categories, et la reference est binaire.
            ref_bin = [round(v) for v in ref_values]
            judge_bin = [1 if v >= 0.5 else 0 for v in judge_values]

            agree = sum(1 for a, b in zip(ref_bin, judge_bin, strict=True) if a == b)
            fp = sum(1 for a, b in zip(ref_bin, judge_bin, strict=True) if a == 0 and b == 1)
            fn = sum(1 for a, b in zip(ref_bin, judge_bin, strict=True) if a == 1 and b == 0)

            out.append(
                Calibration(
                    evaluator=evaluator,
                    metric=metric,
                    kappa=stats.cohen_kappa(ref_bin, judge_bin),
                    raw_agreement=agree / len(ref_bin),
                    n=len(ref_bin),
                    false_positive_rate=fp / len(ref_bin),
                    false_negative_rate=fn / len(ref_bin),
                )
            )

    # Les kappa indefinis (NaN) en fin de liste : ils ne portent aucune
    # information et ne doivent pas occuper le haut du tableau.
    out.sort(key=lambda c: (-1e9 if math.isnan(c.kappa) else c.kappa), reverse=True)
    return out


def leaderboard(
    dataset: str,
    *,
    evaluator: str,
    metric: str,
    settings: Settings | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Classement des runs d'un dataset sur une metrique, avec IC.

    Note de lecture : ce classement n'est PAS un test statistique. Deux
    lignes voisines dont les intervalles se chevauchent ne sont pas
    departagees — il faut passer par compare() pour trancher.
    """
    settings = settings or default_settings
    rows: list[dict[str, Any]] = []
    with db.connect(settings) as conn:
        runs = conn.execute(
            """
            SELECT r.id, r.label, c.name AS config_name, r.started_at
            FROM runs r
            JOIN configs c ON c.hash = r.config_hash
            JOIN datasets d ON d.id = r.dataset_id
            WHERE d.name = %s AND r.status = 'completed'
            ORDER BY r.id DESC LIMIT %s
            """,
            (dataset, limit),
        ).fetchall()

        for run in runs:
            series = metric_series(conn, run["id"], evaluator, metric)
            if not series.by_question:
                continue
            ci = stats.bootstrap_ci(list(series.by_question.values()))
            rows.append(
                {
                    "run_id": run["id"],
                    "label": run["label"] or run["config_name"],
                    "config": run["config_name"],
                    "mean": ci.mean,
                    "low": ci.low,
                    "high": ci.high,
                    "n": ci.n,
                }
            )

    rows.sort(key=lambda r: r["mean"], reverse=True)
    return rows

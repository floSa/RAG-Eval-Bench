"""Garde-fous de non-regression sur le dernier run de reference.

C'est ici que l'evaluation cesse d'etre un rapport qu'on lit de temps en
temps pour devenir un test qui casse la CI. La distinction est celle que
DeepEval formalise avec ses seuils, et elle vaut aussi pour les metriques
natives.

Deux principes :

1. LES SEUILS SONT DES PLANCHERS, PAS DES CIBLES. Ils sont calibres sous la
   baseline observee, avec une marge qui absorbe le bruit d'echantillonnage.
   Un seuil place au niveau exact de la baseline se declencherait la moitie
   du temps sans qu'aucune regression n'ait eu lieu.

2. LES TESTS SONT IGNORES, PAS ROUGES, quand la base est vide. Un
   developpeur qui clone le depot ne doit pas voir une suite rouge parce
   qu'il n'a pas encore lance de campagne.

Lancement cible :
    RAGBENCH_BASELINE_RUN=11 uv run pytest tests/test_regression.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.regression


def _baseline_run_id() -> int | None:
    """Run de reference : celui indique par l'environnement, sinon le
    dernier run complet portant des scores."""
    explicit = os.environ.get("RAGBENCH_BASELINE_RUN")
    if explicit:
        return int(explicit)

    try:
        from ragbench import db

        with db.connect() as conn:
            row = conn.execute(
                """
                SELECT r.id FROM runs r
                WHERE r.status = 'completed'
                  AND EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.id)
                ORDER BY r.id DESC LIMIT 1
                """
            ).fetchone()
        return row["id"] if row else None
    except Exception:
        return None


@pytest.fixture(scope="module")
def summary():
    run_id = _baseline_run_id()
    if run_id is None:
        pytest.skip("aucun run evalue en base — lancer une campagne d'abord")
    from ragbench import runner

    return runner.summarize(run_id)


def _metric(summary, name: str):
    if name not in summary.metrics:
        pytest.skip(f"metrique '{name}' absente du run {summary.run_id}")
    return summary.metrics[name]


# --- planchers de qualite ---------------------------------------------
# Calibres sous la baseline mesuree sur MultiHop-RAG (200 questions,
# gemma4:e4b + nomic-embed-text). A relever des que la baseline progresse —
# un plancher qui ne bouge jamais finit par ne plus rien garder.

SEUILS = {
    "native.ir/recall@3": 0.30,
    "native.ir/hit_rate@3": 0.55,
    "native.ir/mrr": 0.45,
}


@pytest.mark.parametrize("metrique,plancher", sorted(SEUILS.items()))
def test_plancher_de_qualite(summary, metrique, plancher):
    ci = _metric(summary, metrique)
    # Le test porte sur la BORNE BASSE de l'intervalle de confiance, pas sur
    # la moyenne : on veut echouer quand la regression est etablie, pas
    # quand le tirage a ete defavorable.
    assert ci.low >= plancher, (
        f"{metrique} : borne basse {ci.low:.4f} sous le plancher {plancher} "
        f"(moyenne {ci.mean:.4f}, n={ci.n}, run {summary.run_id})"
    )


def test_le_run_a_termine(summary):
    assert summary.status == "completed", f"run {summary.run_id} en statut {summary.status}"


def test_taux_d_echec_acceptable(summary):
    if summary.n_questions == 0:
        pytest.skip("run vide")
    taux = summary.n_failed / summary.n_questions
    assert taux <= 0.05, (
        f"{summary.n_failed}/{summary.n_questions} questions en echec ({taux:.1%}). "
        "Au-dela de 5 %, les moyennes portent sur un sous-ensemble non aleatoire."
    )


def test_echantillon_suffisant(summary):
    """Sous ~100 questions, les intervalles sont trop larges pour departager
    deux configurations : la campagne coute du GPU sans pouvoir conclure."""
    if not summary.metrics:
        pytest.skip("run non evalue")
    n_max = max(ci.n for ci in summary.metrics.values())
    assert n_max >= 100, (
        f"{n_max} questions notees au maximum : insuffisant pour conclure "
        "(viser 100 a 300, cf. stats.required_n)"
    )


def test_precision_de_mesure_utilisable(summary):
    """Une metrique dont l'intervalle de confiance est plus large que les
    ecarts qu'on cherche a detecter ne sert a rien. On verifie qu'au moins
    une metrique atteint une precision exploitable."""
    if not summary.metrics:
        pytest.skip("run non evalue")
    meilleure = min(ci.half_width for ci in summary.metrics.values())
    assert meilleure <= 0.15, (
        f"demi-largeur minimale {meilleure:.3f} : aucune metrique n'est assez "
        "precise pour comparer deux configurations"
    )

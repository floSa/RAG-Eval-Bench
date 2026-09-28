"""Ces fonctions decident si un ecart entre deux configs est reel. Une
erreur ici ne produit pas un plantage : elle produit une conclusion fausse
qui a l'air raisonnable. D'ou des tests sur les cas ou la reponse est
connue d'avance."""

import pytest

from ragbench.stats import (
    bootstrap_ci,
    cohen_kappa,
    mcnemar,
    paired_bootstrap,
    required_n,
)


class TestBootstrapCI:
    def test_encadre_la_moyenne(self):
        ci = bootstrap_ci([0.0, 1.0] * 50, n_boot=1000)
        assert ci.low < ci.mean < ci.high
        assert ci.mean == pytest.approx(0.5, abs=0.01)

    def test_valeurs_constantes_intervalle_nul(self):
        ci = bootstrap_ci([1.0] * 30, n_boot=500)
        assert ci.low == ci.high == 1.0

    def test_intervalle_retrecit_avec_n(self):
        petit = bootstrap_ci([0.0, 1.0] * 10, n_boot=2000)
        grand = bootstrap_ci([0.0, 1.0] * 200, n_boot=2000)
        assert grand.half_width < petit.half_width

    def test_deterministe(self):
        vals = [0.3, 0.7, 0.1, 0.9, 0.5]
        assert bootstrap_ci(vals, n_boot=500).low == bootstrap_ci(vals, n_boot=500).low

    def test_liste_vide(self):
        assert bootstrap_ci([]).n == 0


class TestPairedBootstrap:
    def test_amelioration_systematique_detectee(self):
        a = [0.0] * 100
        b = [1.0] * 100
        res = paired_bootstrap(a, b, n_boot=1000)
        assert res.significant
        assert res.delta == pytest.approx(1.0)
        assert res.n_better == 100

    def test_aucune_difference_non_significative(self):
        vals = [0.0, 1.0] * 50
        res = paired_bootstrap(vals, vals, n_boot=1000)
        assert not res.significant
        assert res.delta == 0.0
        assert res.n_tied == 100

    def test_appariement_gagne_en_puissance(self):
        """Deux configs dont TOUTES les questions s'ameliorent de 0,05.
        La variance entre questions est enorme, mais l'ecart est constant :
        seul un test apparie peut le voir."""
        a = [i / 100 for i in range(100)]
        b = [x + 0.05 for x in a]
        res = paired_bootstrap(a, b, n_boot=2000)
        assert res.significant
        assert res.delta == pytest.approx(0.05, abs=1e-9)

    def test_p_value_jamais_nulle(self):
        res = paired_bootstrap([0.0] * 50, [1.0] * 50, n_boot=1000)
        assert res.p_value > 0

    def test_tailles_incoherentes(self):
        with pytest.raises(ValueError, match="non apparies"):
            paired_bootstrap([1.0], [1.0, 2.0])


class TestMcNemar:
    def test_domination_totale(self):
        b_wins, a_wins, p = mcnemar([0.0] * 20, [1.0] * 20)
        assert (b_wins, a_wins) == (20, 0)
        assert p < 0.001

    def test_accord_parfait_non_concluant(self):
        """Aucune discordance : le test ne peut rien dire, et doit le dire
        plutot que de renvoyer un p trompeur."""
        b_wins, a_wins, p = mcnemar([1.0] * 30, [1.0] * 30)
        assert (b_wins, a_wins, p) == (0, 0, 1.0)

    def test_discordances_equilibrees(self):
        a = [1.0] * 5 + [0.0] * 5
        b = [0.0] * 5 + [1.0] * 5
        _, _, p = mcnemar(a, b)
        assert p == pytest.approx(1.0)


class TestCohenKappa:
    def test_accord_parfait_avec_variete(self):
        assert cohen_kappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1]) == pytest.approx(1.0)

    def test_desaccord_total(self):
        assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0

    def test_accord_par_hasard_proche_de_zero(self):
        a = [1, 0] * 50
        b = [1, 1, 0, 0] * 25
        assert abs(cohen_kappa(a, b)) < 0.2

    def test_distribution_degeneree_indefinie(self):
        """Si les deux annotateurs notent tout pareil, kappa n'est pas 1 :
        il est indefini. Renvoyer 1.0 laisserait croire a un juge valide
        alors qu'aucune information n'a ete produite."""
        import math

        assert math.isnan(cohen_kappa([1] * 20, [1] * 20))


class TestRequiredN:
    def test_effet_plus_petit_exige_plus_de_questions(self):
        assert required_n(0.05) > required_n(0.20)

    def test_ordre_de_grandeur_plausible(self):
        """Detecter 10 points d'ecart autour de 50 % doit demander quelques
        centaines de questions — le repere qui justifie n=200."""
        assert 100 < required_n(0.10) < 1000


class TestZoneLimite:
    """L'intervalle vient d'un bootstrap, la p-value d'un test de
    permutation : deux procedures differentes qui peuvent diverger juste au
    seuil. C'est precisement la zone ou l'on prend les mauvaises decisions
    avec assurance, donc elle doit etre nommee."""

    def test_accord_franc_non_limite(self):
        res = paired_bootstrap([0.0] * 100, [1.0] * 100, n_boot=1000)
        assert res.significant
        assert not res.borderline

    def test_absence_d_effet_non_limite(self):
        vals = [0.0, 1.0] * 50
        res = paired_bootstrap(vals, vals, n_boot=1000)
        assert not res.significant
        assert not res.borderline

    def test_desaccord_detecte(self):
        """borderline vaut True des que les deux criteres divergent, quel
        que soit le sens du desaccord."""
        from ragbench.stats import Comparison

        c = Comparison(
            mean_a=0.5, mean_b=0.55, delta=0.05,
            ci_low=0.001, ci_high=0.099,   # exclut zero
            p_value=0.063,                  # mais p > 0.05
            n_pairs=177, n_better=40, n_worse=30, n_tied=107,
        )
        assert c.significant
        assert c.borderline
        assert "LIMITE" in c.verdict()


class TestHolm:
    def test_exemple_de_reference(self):
        """Exemple classique : p tries 0.01, 0.02, 0.03, 0.04 sur 4 tests."""
        from ragbench.stats import holm

        adj = holm([0.04, 0.01, 0.03, 0.02])
        assert [round(x, 4) for x in adj] == [0.06, 0.04, 0.06, 0.06]

    def test_monotone_et_borne(self):
        from ragbench.stats import holm

        adj = holm([0.9, 0.5, 0.001])
        assert max(adj) <= 1.0
        assert adj[2] <= adj[1] <= adj[0]

    def test_un_seul_test_inchange(self):
        from ragbench.stats import holm

        assert holm([0.03]) == [0.03]

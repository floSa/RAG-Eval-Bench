"""Les metriques natives sont la reference contre laquelle les juges LLM
seront calibres. Si elles sont fausses, tout le reste l'est."""

import math

import pytest

from ragbench.evaluators.native_answer import contains_gold, normalize, token_f1
from ragbench.evaluators.native_ir import _ndcg
from ragbench.rag.generate import detect_abstention


class TestNormalisation:
    def test_ponctuation_et_articles(self):
        assert normalize("The Verge, Inc.") == "verge inc"

    def test_none_et_vide(self):
        assert normalize(None) == ""
        assert normalize("   ") == ""


class TestContainsGold:
    def test_gold_court_dans_reponse_redigee(self):
        """Cas dominant : le gold vaut 'Sam Bankman-Fried' et le modele
        repond une phrase entiere. L'exact match echouerait a tort."""
        assert contains_gold("The individual is Sam Bankman-Fried.", "Sam Bankman-Fried")

    def test_yes_no_exige_le_premier_token(self):
        """Sur un gold binaire, la simple presence du mot ne suffit pas :
        « ... the answer is not yes ... » contient 'yes' mais dit non."""
        assert contains_gold("Yes, both articles agree.", "Yes")
        assert not contains_gold("The context does not say yes.", "Yes")

    def test_yes_no_casse_ignoree(self):
        assert contains_gold("no, they differ", "no")

    def test_gold_vide(self):
        assert not contains_gold("anything", "")


class TestTokenF1:
    def test_identique(self):
        assert token_f1("Sam Altman", "Sam Altman") == pytest.approx(1.0)

    def test_disjoint(self):
        assert token_f1("Google", "Valve") == 0.0

    def test_partiel(self):
        assert 0 < token_f1("Sam Bankman", "Sam Bankman-Fried") < 1


class TestAbstention:
    def test_jeton_explicite(self):
        assert detect_abstention("INSUFFICIENT_CONTEXT")

    @pytest.mark.parametrize(
        "texte",
        [
            "The provided context does not contain any information about X.",
            "The context does not mention the article.",
            "There is no information regarding that date.",
            "It is impossible to determine the answer.",
            "I don't know.",
        ],
    )
    def test_abstentions_deguisees(self, texte):
        """Formulations observees sur gemma4:e4b. Les manquer revient a
        compter une abstention comme une reponse fausse, ce qui deplace le
        diagnostic du retrieval vers le generateur."""
        assert detect_abstention(texte)

    def test_vraie_reponse_non_detectee(self):
        assert not detect_abstention("Sam Bankman-Fried was charged with fraud.")
        assert not detect_abstention("Yes, both reports agree on the outcome.")


class TestNDCG:
    def test_classement_parfait(self):
        assert _ndcg(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_aucun_pertinent(self):
        assert _ndcg(["x", "y"], {"a"}, 2) == 0.0

    def test_idcg_borne_par_k(self):
        """Avec 4 documents gold mais k=2, le classement parfait realisable
        n'en contient que 2. Sans cette borne, une config a petit top_k
        serait penalisee pour son top_k, pas pour son classement."""
        assert _ndcg(["a", "b"], {"a", "b", "c", "d"}, 2) == pytest.approx(1.0)

    def test_ordre_compte(self):
        bon = _ndcg(["a", "x"], {"a"}, 2)
        mauvais = _ndcg(["x", "a"], {"a"}, 2)
        assert bon > mauvais
        assert mauvais == pytest.approx(1 / math.log2(3))

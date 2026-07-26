"""Les evaluateurs natifs sont la reference du banc : ce sont eux qui
serviront a calibrer les juges. Les parties deterministes doivent donc
etre testees independamment de tout appel LLM."""

import pytest

from ragbench.evaluators.native_claims import parse_claims
from ragbench.evaluators.native_nuggets import lexical_support


class TestSupportLexical:
    def test_fait_present_mot_pour_mot(self):
        ok, score = lexical_support(
            "Sam Bankman-Fried was the CEO of FTX",
            ["Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, went on trial."],
        )
        assert ok
        assert score > 0.6

    def test_fait_absent(self):
        ok, _ = lexical_support(
            "Valve released the Steam Deck OLED",
            ["Apple announced new M3 chips with Dynamic Caching."],
        )
        assert not ok

    def test_mots_outils_ignores(self):
        """Sans filtrage des mots-outils, n'importe quel passage anglais
        couvrirait n'importe quel fait a 40 %."""
        ok, _ = lexical_support("the of and a in on", ["completely unrelated content here"])
        assert not ok

    def test_normalisation_par_le_fait_pas_le_passage(self):
        """Le score est la part des tokens du FAIT retrouves. Un passage tres
        long ne doit pas etre avantage par sa seule longueur."""
        fait = "Valve improvements"
        court = ["Valve improvements"]
        long = ["Valve improvements " + "filler words about other topics " * 50]
        assert lexical_support(fait, court)[1] == lexical_support(fait, long)[1]

    def test_aucun_passage(self):
        assert lexical_support("some fact here", [])[0] is False


class TestParseClaims:
    def test_liste_a_tirets(self):
        claims = parse_claims("- Google paid 26 billion dollars.\n- The trial started Tuesday.")
        assert len(claims) == 2
        assert claims[0].startswith("Google paid")

    def test_marqueur_none(self):
        assert parse_claims("NONE") == []
        assert parse_claims("") == []

    def test_repli_sur_lignes_nues(self):
        """Un petit modele ne respecte pas toujours le format demande.
        Exiger les tirets transformerait une variation de style en score nul."""
        claims = parse_claims(
            "Google paid 26.3 billion dollars in 2021 to remain the default engine.\n"
            "The class action was filed by a news publisher."
        )
        assert len(claims) == 2

    def test_fragments_courts_ecartes(self):
        assert parse_claims("- ok\n- yes\n- Google paid 26 billion dollars in 2021.") == [
            "Google paid 26 billion dollars in 2021."
        ]

    def test_borne_dure(self):
        texte = "\n".join(f"- Claim number {i} with enough characters." for i in range(40))
        assert len(parse_claims(texte, max_claims=12)) == 12


class TestContexteEvaluation:
    def test_separation_repondables_et_sans_reponse(self):
        """La distinction porte tout le traitement des null_query : un
        recall sur une question sans reponse est indefini, pas nul."""
        from ragbench.config import PipelineConfig
        from ragbench.evaluators.base import EvalContext

        ctx = EvalContext(
            predictions=[
                {"question_id": 1, "gold_evidence": [{"document_external_id": "a"}]},
                {"question_id": 2, "gold_evidence": []},
                {"question_id": 3, "gold_evidence": [{"document_external_id": "b"}]},
            ],
            config=PipelineConfig(name="t"),
            llm=None,  # type: ignore[arg-type]
        )
        assert [p["question_id"] for p in ctx.answerable()] == [1, 3]
        assert [p["question_id"] for p in ctx.unanswerable()] == [2]


def test_registre_expose_les_natifs():
    from ragbench import evaluators

    noms = evaluators.available()
    for attendu in ["native.ir", "native.answer", "native.nuggets", "native.erag", "native.claims"]:
        assert attendu in noms


def test_evaluateur_inconnu_message_utile():
    from ragbench import evaluators

    with pytest.raises(KeyError, match="inconnu"):
        evaluators.get("nexiste.pas")

"""Le plafond par document et l'en-tete contextuel sont les deux
corrections issues du pilote. Ce sont des mecanismes discrets, faciles a
casser silencieusement, et leur defaillance ne se voit pas dans les scores
— elle se traduit juste par un recall qui stagne."""

from ragbench.rag.ingest import build_header
from ragbench.rag.retrieve import Context, _cap_per_document, _rrf


def _ctx(chunk_id: int, doc: str, rank: int, score: float = 0.5) -> Context:
    return Context(chunk_id, hash(doc) % 1000, doc, f"texte {chunk_id}", score, rank)


class TestPlafondParDocument:
    def test_limite_respectee(self):
        candidats = [_ctx(i, "docA", i) for i in range(1, 6)]
        assert len(_cap_per_document(candidats, 2)) == 2

    def test_ordre_preserve(self):
        candidats = [_ctx(1, "A", 1), _ctx(2, "A", 2), _ctx(3, "B", 3), _ctx(4, "A", 4)]
        gardes = _cap_per_document(candidats, 1)
        assert [c.document_external_id for c in gardes] == ["A", "B"]
        assert [c.rank for c in gardes] == [1, 3]

    def test_diversite_documentaire(self):
        """L'objectif reel : sur une question multi-hop exigeant 3 articles,
        le plafond doit laisser passer 3 documents distincts la ou un seul
        article monopolisait le top_k."""
        candidats = [
            _ctx(1, "A", 1), _ctx(2, "A", 2), _ctx(3, "A", 3),
            _ctx(4, "B", 4), _ctx(5, "C", 5),
        ]
        gardes = _cap_per_document(candidats, 1)[:3]
        assert {c.document_external_id for c in gardes} == {"A", "B", "C"}

    def test_sans_doublon_rien_ne_change(self):
        candidats = [_ctx(1, "A", 1), _ctx(2, "B", 2)]
        assert _cap_per_document(candidats, 2) == candidats


class TestRRF:
    def test_chunk_present_dans_les_deux_listes_remonte(self):
        """Tout l'interet de RRF : un chunk trouve par le dense ET par le
        lexical passe devant un chunk trouve par un seul, meme si ses rangs
        individuels sont moins bons.

        La fusion se fait sur chunk_id, pas sur le document : les deux
        moteurs interrogent la meme table `chunks`, donc un meme passage
        porte le meme identifiant des deux cotes. La diversite documentaire
        est traitee separement, par max_per_document.
        """
        dense = [_ctx(1, "A", 1), _ctx(2, "B", 2)]
        lexical = [_ctx(3, "C", 1), _ctx(2, "B", 2)]
        fusion = _rrf([dense, lexical], k_const=60, top_k=3)
        assert fusion[0].chunk_id == 2
        assert fusion[0].document_external_id == "B"

    def test_scores_non_comparables_sans_influence(self):
        """RRF ne regarde que les rangs : un scorer a grande echelle ne doit
        pas ecraser l'autre. C'est la raison du choix de RRF plutot qu'une
        somme ponderee."""
        dense = [_ctx(1, "A", 1, score=0.99)]
        lexical = [_ctx(2, "B", 1, score=1200.0)]
        fusion = _rrf([dense, lexical], k_const=60, top_k=2)
        assert fusion[0].score == fusion[1].score


class TestEnTeteContextuel:
    def test_champs_assembles(self):
        doc = {
            "title": "Le proces FTX",
            "metadata": {"source": "The Verge", "published_at": "2023-09-28"},
        }
        header = build_header(doc, ("title", "source", "published_at"))
        assert "Title: Le proces FTX" in header
        assert "Source: The Verge" in header
        assert "Published: 2023-09-28" in header

    def test_champs_absents_ignores(self):
        doc = {"title": "T", "metadata": {}}
        assert build_header(doc, ("title", "source")) == "Title: T"

    def test_liste_vide(self):
        doc = {"title": "T", "metadata": {"source": "S"}}
        assert build_header(doc, ()) == ""


class TestCrossEncoder:
    def test_tri_par_score_et_rangs_renumerotes(self):
        from ragbench.rag.retrieve import order_by_scores

        candidats = [_ctx(1, "A", 1), _ctx(2, "B", 2), _ctx(3, "C", 3)]
        ordonnes = order_by_scores(candidats, [0.1, 0.9, 0.5], "cross_encoder")
        assert [c.chunk_id for c in ordonnes] == [2, 3, 1]
        assert [c.rank for c in ordonnes] == [1, 2, 3]
        assert {c.source for c in ordonnes} == {"cross_encoder"}

    def test_egalite_departagee_par_rang_d_origine(self):
        """Sans departage deterministe, deux runs identiques pourraient
        differer sur un recall@k."""
        from ragbench.rag.retrieve import order_by_scores

        candidats = [_ctx(1, "A", 1), _ctx(2, "B", 2)]
        ordonnes = order_by_scores(candidats, [0.5, 0.5], "cross_encoder")
        assert [c.chunk_id for c in ordonnes] == [1, 2]

    def test_scores_manquants_leve(self):
        """Un reranking qui echoue en silence imiterait un reranking inutile."""
        import pytest

        from ragbench.rag.retrieve import RerankFailure, order_by_scores

        candidats = [_ctx(1, "A", 1), _ctx(2, "B", 2)]
        with pytest.raises(RerankFailure):
            order_by_scores(candidats, [0.5], "cross_encoder")
        with pytest.raises(RerankFailure):
            order_by_scores(candidats, [0.5, float("nan")], "cross_encoder")


class TestSousRequetes:
    def test_puces_et_numeros_retires(self):
        """Les petits modeles numerotent malgre la consigne."""
        from ragbench.rag.retrieve import parse_sub_queries

        text = "1. Will Lutz field goal Sporting News\n- Connor Bedard goalies NYT\n\n"
        assert parse_sub_queries(text, 3) == [
            "Will Lutz field goal Sporting News", "Connor Bedard goalies NYT",
        ]

    def test_plafond_et_doublons(self):
        from ragbench.rag.retrieve import parse_sub_queries

        text = "a\nA\nb\nc\nd"
        assert parse_sub_queries(text, 3) == ["a", "b", "c"]

    def test_reponse_vide(self):
        from ragbench.rag.retrieve import parse_sub_queries

        assert parse_sub_queries("", 3) == []


class TestClarification:
    def test_none_et_reponse_sans_question(self):
        from ragbench.rag.retrieve import wants_clarification

        assert not wants_clarification("NONE")
        assert not wants_clarification("none.")
        assert not wants_clarification("")
        # Sans point d'interrogation, ce n'est pas une question posee.
        assert not wants_clarification("The question is clear.")
        assert wants_clarification("Which Sporting News article do you mean, the NFL or NHL one?")

    def test_candidats_lus_dans_l_en_tete(self):
        from ragbench.rag.retrieve import Context, candidate_lines

        text = "Title: Bedard's debut\nSource: Sporting News\nPublished: 2023-10-11T00:00\n\ncorps"
        ctxs = [
            Context(1, 1, "a", text, 0.9, 1),
            Context(2, 1, "a", text, 0.8, 2),  # meme document : une seule ligne
        ]
        assert candidate_lines(ctxs) == "- Sporting News, 2023-10-11: Bedard's debut"

    def test_sans_utilisateur_refuse(self):
        """Une clarification sans personne pour repondre serait un no-op."""
        import asyncio

        import pytest

        from ragbench.config import PipelineConfig, RetrievalConfig
        from ragbench.rag.retrieve import retrieve

        cfg = PipelineConfig(name="c", retrieval=RetrievalConfig(clarify=True))
        with pytest.raises(ValueError, match="utilisateur"):
            asyncio.run(retrieve(None, None, cfg, index_id=1, question="q"))


def test_utilisateur_simule_ne_voit_que_les_articles():
    """Il connait source, date et titre — jamais la reponse ni les faits."""
    from ragbench.runner.simulated_user import describe_articles

    gold = [
        {"document_external_id": "u1", "fact": "Will Lutz buries his fourth field goal"},
        {"document_external_id": "u1", "fact": "autre fait"},
    ]
    meta = {"u1": {"title": "Vikings vs. Broncos", "source": "Sporting News",
                   "published_at": "2023-11-19T23:00:04+00:00"}}
    text = describe_articles(gold, meta)
    assert text == "- Sporting News, 2023-11-19: Vikings vs. Broncos"
    assert "Lutz" not in text

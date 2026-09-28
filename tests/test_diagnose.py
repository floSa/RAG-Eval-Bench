"""La courbe de couverture tranche entre deux diagnostics opposes : bons
passages remontes trop bas, ou jamais remontes. Une erreur de prefixe ou de
deduplication inverserait la conclusion sans rien faire echouer."""

from ragbench.config import PipelineConfig, RetrievalConfig
from ragbench.runner.diagnose import aggregate, pool_config, question_curve


def _ctx(doc: str, text: str) -> dict:
    return {"document_external_id": doc, "text": text}


GOLD = [
    {"document_external_id": "A", "fact": "the merger closed in march"},
    {"document_external_id": "B", "fact": "revenue doubled after launch"},
]


class TestCourbeParQuestion:
    def test_prefixe_respecte(self):
        """Le document B n'arrive qu'en 3e position : absent a k=2, present a k=3."""
        contexts = [
            _ctx("A", "the merger closed in march"),
            _ctx("A", "other passage"),
            _ctx("B", "revenue doubled after launch"),
        ]
        curve = question_curve(contexts, GOLD, [2, 3])
        assert curve[2]["doc_recall"] == 0.5
        assert curve[2]["nugget_full_coverage"] == 0.0
        assert curve[3]["doc_recall"] == 1.0
        assert curve[3]["nugget_full_coverage"] == 1.0

    def test_documents_dedupliques(self):
        """Deux passages du meme document gold ne comptent qu'une fois."""
        contexts = [_ctx("A", "x"), _ctx("A", "y")]
        curve = question_curve(contexts, GOLD, [2])
        assert curve[2]["doc_recall"] == 0.5
        assert curve[2]["hit_rate"] == 1.0

    def test_bon_document_sans_le_fait(self):
        """Le bon document remonte, mais pas le passage qui porte le fait :
        c'est exactement l'ecart entre recall documentaire et pepites."""
        contexts = [_ctx("A", "unrelated paragraph"), _ctx("B", "revenue doubled after launch")]
        curve = question_curve(contexts, GOLD, [2])
        assert curve[2]["doc_recall"] == 1.0
        assert curve[2]["nugget_recall"] == 0.5
        assert curve[2]["nugget_full_coverage"] == 0.0

    def test_sans_fait_pepites_indefinies(self):
        gold = [{"document_external_id": "A"}]
        curve = question_curve([_ctx("A", "x")], gold, [1])
        assert curve[1]["doc_recall"] == 1.0
        assert curve[1]["nugget_recall"] is None
        assert curve[1]["nugget_full_coverage"] is None

    def test_k_superieur_au_nombre_de_passages(self):
        curve = question_curve([_ctx("A", "the merger closed in march")], GOLD, [5])
        assert curve[5]["doc_recall"] == 0.5


class TestAgregation:
    def test_valeurs_indefinies_ecartees(self):
        """Une question sans fait ne doit pas tirer la couverture vers 0."""
        full = {"doc_recall": 1.0, "hit_rate": 1.0, "nugget_recall": 1.0,
                "nugget_full_coverage": 1.0}
        sans_fait = {"doc_recall": 0.0, "hit_rate": 0.0, "nugget_recall": None,
                     "nugget_full_coverage": None}
        rows = [{5: full}, {5: sans_fait}]
        curve = aggregate(rows, [5])
        assert curve["doc_recall"][5].mean == 0.5
        assert curve["doc_recall"][5].n == 2
        assert curve["nugget_full_coverage"][5].mean == 1.0
        assert curve["nugget_full_coverage"][5].n == 1


def _cfg(**retrieval) -> PipelineConfig:
    return PipelineConfig(name="t", retrieval=RetrievalConfig(**retrieval))


class TestConfigDeVivier:
    def test_top_k_porte_au_maximum(self):
        tuned, notes = pool_config(_cfg(mode="dense", top_k=5), 50)
        assert tuned.retrieval.top_k == 50
        assert notes == []

    def test_plafond_par_document_elargit_le_vivier(self):
        tuned, _ = pool_config(_cfg(top_k=5, fetch_k=20, max_per_document=2), 50)
        assert tuned.retrieval.fetch_k >= 4 * 50

    def test_approximations_signalees(self):
        _, notes = pool_config(_cfg(mode="hybrid", rerank="llm", fetch_k=20), 50)
        assert len(notes) == 2

    def test_config_d_origine_intacte(self):
        cfg = _cfg(top_k=5)
        pool_config(cfg, 50)
        assert cfg.retrieval.top_k == 5


def _report(per_question, ks=(5,)):
    from ragbench.runner.diagnose import CurveReport

    return CurveReport(
        config_name="x", config_hash="h", index_hash="i", ks=list(ks),
        n_questions=len(per_question), n_failed=0,
        curve=aggregate(per_question.values(), list(ks)), per_question=per_question,
    )


def _row(recall, coverage=None):
    return {5: {"doc_recall": recall, "hit_rate": None, "nugget_recall": None,
                "nugget_full_coverage": coverage}}


class TestComparaison:
    def test_appariement_sur_questions_communes(self):
        """Une question absente d'un cote n'est pas comptee comme un zero."""
        from ragbench.runner.diagnose import compare_curves

        a = _report({1: _row(0.0), 2: _row(0.0), 3: _row(1.0)})
        b = _report({1: _row(1.0), 2: _row(1.0)})
        rows = compare_curves(a, b, metrics=["doc_recall"])
        assert rows[0].comparison.n_pairs == 2
        assert rows[0].comparison.delta == 1.0

    def test_valeurs_indefinies_ecartees(self):
        from ragbench.runner.diagnose import compare_curves

        a = _report({1: _row(0.5, None), 2: _row(0.5, 1.0)})
        b = _report({1: _row(0.5, 1.0), 2: _row(0.5, 0.0)})
        rows = {r.metric: r for r in compare_curves(a, b)}
        assert rows["nugget_full_coverage"].comparison.n_pairs == 1

    def test_ecart_net_et_verdict(self):
        from ragbench.runner.diagnose import compare_curves

        a = _report({i: _row(0.0) for i in range(60)})
        b = _report({i: _row(1.0 if i % 4 else 0.0) for i in range(60)})
        (row,) = compare_curves(a, b, metrics=["doc_recall"])
        assert row.verdict == "B meilleur"

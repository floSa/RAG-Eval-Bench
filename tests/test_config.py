"""Le hash de config est l'identite d'un run : s'il est instable ou trop
sensible, tout le suivi longitudinal est faux. D'ou ces tests."""

from ragbench.config import ChunkingConfig, PipelineConfig, RetrievalConfig


def test_hash_ignore_le_nom_et_la_description():
    """Renommer une config ne doit pas casser la comparabilite avec ses
    runs passes."""
    a = PipelineConfig(name="a", description="premiere")
    b = PipelineConfig(name="b", description="seconde")
    assert a.hash() == b.hash()


def test_hash_change_avec_un_parametre_de_pipeline():
    a = PipelineConfig(name="a")
    b = PipelineConfig(name="a", retrieval=RetrievalConfig(top_k=10))
    assert a.hash() != b.hash()


def test_index_hash_ignore_les_parametres_de_retrieval():
    """C'est ce qui evite de reindexer 600 documents pour changer un top_k."""
    a = PipelineConfig(name="a", retrieval=RetrievalConfig(top_k=5))
    b = PipelineConfig(name="a", retrieval=RetrievalConfig(top_k=50, mode="hybrid"))
    assert a.index_hash() == b.index_hash()
    assert a.hash() != b.hash()


def test_index_hash_change_avec_le_chunking():
    a = PipelineConfig(name="a", chunking=ChunkingConfig(chunk_size=1000))
    b = PipelineConfig(name="a", chunking=ChunkingConfig(chunk_size=500))
    assert a.index_hash() != b.index_hash()


def test_warning_juge_egal_generateur():
    cfg = PipelineConfig(name="a")  # defauts : judge == generator
    assert any("auto-preference" in w for w in cfg.warnings())


def test_warning_temperature_non_nulle():
    from ragbench.config import GenerationConfig

    cfg = PipelineConfig(name="a", generation=GenerationConfig(temperature=0.7))
    assert any("temperature" in w for w in cfg.warnings())


def test_matrice_fusionne_en_profondeur(tmp_path):
    """Une variante ne declare que ce qu'elle change ; le reste vient de la
    base, y compris a l'interieur d'un sous-objet."""
    path = tmp_path / "exp.yml"
    path.write_text(
        "base:\n"
        "  name: base\n"
        "  retrieval: {top_k: 5, mode: dense}\n"
        "variants:\n"
        "  - name: v1\n"
        "  - name: v2\n"
        "    retrieval: {top_k: 10}\n",
        encoding="utf-8",
    )
    configs = PipelineConfig.matrix_from_yaml(path)
    assert [c.name for c in configs] == ["v1", "v2"]
    assert configs[0].retrieval.top_k == 5
    assert configs[1].retrieval.top_k == 10
    # mode n'est pas redeclare par v2 : il doit survivre a la fusion.
    assert configs[1].retrieval.mode == "dense"


# Hashes des configs versionnees, releves avant l'ajout du premier champ du
# catalogue de techniques. S'ils bougent, tous les runs historiques deviennent
# incomparables aux nouveaux : c'est une regression, pas une mise a jour.
HASHES_HISTORIQUES = {
    "configs/baseline.yml": "bb2b20210128",
    "configs/recommended.yml": "b4a78b819a03",
}


def test_hashes_historiques_stables():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path, expected in HASHES_HISTORIQUES.items():
        assert PipelineConfig.from_yaml(root / path).hash() == expected, path


def test_champ_ajoute_neutre_hors_du_hash():
    """Un champ du catalogue a sa valeur neutre ne change pas l'identite."""
    from ragbench.config import NEUTRAL_ADDITIONS

    payload = PipelineConfig(name="a").payload()
    for section, key in NEUTRAL_ADDITIONS:
        assert key not in payload[section]


def test_cross_encoder_entre_dans_le_hash():
    a = PipelineConfig(name="a")
    b = PipelineConfig(
        name="a",
        retrieval=RetrievalConfig(rerank="cross_encoder", rerank_model="flashrank:m"),
    )
    c = PipelineConfig(
        name="a",
        retrieval=RetrievalConfig(rerank="cross_encoder", rerank_model="flashrank:autre"),
    )
    assert len({a.hash(), b.hash(), c.hash()}) == 3


def test_rerank_model_incoherent_refuse():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetrievalConfig(rerank="cross_encoder")
    with pytest.raises(ValidationError):
        RetrievalConfig(rerank="none", rerank_model="flashrank:m")
    with pytest.raises(ValidationError):
        RetrievalConfig(rerank="cross_encoder", rerank_model="sans-backend")


def test_techniques_de_requete_coherentes():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetrievalConfig(max_sub_queries=5)  # sans query_decompose : sans effet
    with pytest.raises(ValidationError):
        RetrievalConfig(mode="lexical", hyde=True)  # HyDE n'agit que sur le dense
    assert RetrievalConfig(query_decompose=True, max_sub_queries=5).max_sub_queries == 5


def test_techniques_de_requete_dans_le_hash():
    base = PipelineConfig(name="a")
    decompose = PipelineConfig(name="a", retrieval=RetrievalConfig(query_decompose=True))
    hyde = PipelineConfig(name="a", retrieval=RetrievalConfig(hyde=True))
    assert len({base.hash(), decompose.hash(), hyde.hash()}) == 3

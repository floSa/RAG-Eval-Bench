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

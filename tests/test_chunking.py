"""Le chunking est la variable experimentale la plus en amont : une erreur
ici (perte de texte, chunk vide, taille non respectee) se propage a toutes
les metriques sans etre visible dans les scores."""

import pytest

from ragbench.config import ChunkingConfig
from ragbench.rag.chunking import chunk_document

TEXTE = (
    "Premier paragraphe qui pose le contexte general du document. "
    "Il contient deux phrases distinctes.\n\n"
    "Deuxieme paragraphe, plus long, qui developpe une idee secondaire "
    "avec plusieurs propositions enchainees et un peu de remplissage pour "
    "depasser la taille cible du decoupage.\n\n"
    "Troisieme paragraphe, court."
)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "sentence", "document"])
def test_aucun_chunk_vide(strategy):
    cfg = ChunkingConfig(strategy=strategy, chunk_size=100, chunk_overlap=20)
    chunks = chunk_document(TEXTE, cfg)
    assert chunks
    assert all(c.strip() for c in chunks)


def test_document_ne_decoupe_pas():
    chunks = chunk_document(TEXTE, ChunkingConfig(strategy="document"))
    assert len(chunks) == 1


def test_fixed_respecte_la_taille():
    cfg = ChunkingConfig(strategy="fixed", chunk_size=50, chunk_overlap=0)
    assert all(len(c) <= 50 for c in chunk_document(TEXTE, cfg))


def test_sentence_ne_coupe_pas_une_phrase():
    """Une phrase coupee en deux rend une claim invérifiable : c'est
    exactement le genre de defaut qui fait chuter la faithfulness sans
    qu'on comprenne pourquoi."""
    cfg = ChunkingConfig(strategy="sentence", chunk_size=120, chunk_overlap=0)
    chunks = chunk_document(TEXTE, cfg)
    # Chaque chunk se termine sur une ponctuation forte (le dernier peut
    # ne pas en avoir si le texte n'en a pas).
    assert all(c.rstrip()[-1] in ".!?" for c in chunks[:-1])


def test_recursive_ne_perd_pas_de_contenu():
    """Sans overlap, la concatenation des chunks doit contenir tous les
    mots du texte d'origine."""
    cfg = ChunkingConfig(strategy="recursive", chunk_size=120, chunk_overlap=0)
    chunks = chunk_document(TEXTE, cfg)
    recompose = " ".join(chunks)
    for mot in ["Premier", "Deuxieme", "Troisieme", "remplissage"]:
        assert mot in recompose


def test_overlap_reporte_la_fin_du_chunk_precedent():
    cfg = ChunkingConfig(strategy="recursive", chunk_size=120, chunk_overlap=20)
    chunks = chunk_document(TEXTE, cfg)
    assert len(chunks) > 1
    assert chunks[1].startswith(chunks[0][-20:])


def test_texte_vide():
    assert chunk_document("   ", ChunkingConfig()) == []

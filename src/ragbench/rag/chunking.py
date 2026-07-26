"""Decoupage des documents en passages.

Unite : le CARACTERE, pas le token. Choix assume — un decoupage en tokens
serait lie au tokenizer d'un modele donne, ce qui rendrait deux runs sur
deux generateurs differents non comparables au niveau du corpus. Le
caractere est neutre et reproductible. Ordre de grandeur utile : ~4
caracteres par token en anglais.

Aucune dependance externe : LangChain et consorts sont evitables ici, et le
decoupage est precisement ce qu'on veut pouvoir modifier sans mediation.
"""

from __future__ import annotations

import re

from ..config import ChunkingConfig

# Separateurs par ordre de preference : on coupe d'abord aux frontieres
# fortes (paragraphes), et on ne descend au caractere qu'en dernier recours.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_document(text: str, cfg: ChunkingConfig) -> list[str]:
    text = text.strip()
    if not text:
        return []

    match cfg.strategy:
        case "document":
            return [text]
        case "fixed":
            return _fixed(text, cfg.chunk_size, cfg.chunk_overlap)
        case "recursive":
            return _recursive(text, cfg.chunk_size, cfg.chunk_overlap)
        case "sentence":
            return _sentence(text, cfg.chunk_size, cfg.chunk_overlap)
        case _:  # pragma: no cover - garde-fou, la config est validee en amont
            raise ValueError(f"strategie de chunking inconnue : {cfg.strategy}")


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    """Fenetre glissante aveugle. Sert de temoin : c'est la strategie la
    plus naive, et savoir de combien les autres la battent est un resultat
    en soi."""
    if size <= 0:
        raise ValueError("chunk_size doit etre > 0")
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def _recursive(text: str, size: int, overlap: int, _depth: int = 0) -> list[str]:
    """Descente dans _SEPARATORS jusqu'a obtenir des morceaux qui tiennent."""
    if len(text) <= size:
        return [text] if text.strip() else []

    separator = ""
    for sep in _SEPARATORS[_depth:]:
        if sep == "" or sep in text:
            separator = sep
            break

    if separator == "":
        return _fixed(text, size, overlap)

    parts = text.split(separator)
    chunks: list[str] = []
    buffer = ""

    for part in parts:
        candidate = f"{buffer}{separator}{part}" if buffer else part
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        # Un fragment isole plus grand que la taille cible : on redescend
        # d'un cran dans la liste des separateurs.
        if len(part) > size:
            depth = _SEPARATORS.index(separator) + 1
            chunks.extend(_recursive(part, size, overlap, depth))
            buffer = ""
        else:
            buffer = part

    if buffer.strip():
        chunks.append(buffer)

    return _apply_overlap([c for c in chunks if c.strip()], overlap)


def _sentence(text: str, size: int, overlap: int) -> list[str]:
    """Groupe des phrases entieres sans jamais en couper une.

    Interet pour l'evaluation : c'est la strategie qui preserve le mieux la
    verifiabilite d'une claim, donc celle qui devrait maximiser la
    faithfulness a retrieval egal. Hypothese testable par le banc.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip()
        if len(candidate) <= size or not buffer:
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return _apply_overlap(chunks, overlap)


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefixe chaque chunk par la fin du precedent.

    Le recouvrement se fait en post-traitement plutot que pendant le
    decoupage : ca garde les strategies lisibles et rend l'overlap
    comparable entre elles.
    """
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    # strict=False est VOULU ici : chunks[1:] a deliberement un element de
    # moins, on apparie chaque chunk avec son predecesseur.
    for previous, current in zip(chunks, chunks[1:], strict=False):
        tail = previous[-overlap:]
        out.append(f"{tail}{current}" if tail.strip() else current)
    return out

"""Interface commune des jeux de donnees.

Un loader transforme une source quelconque (fichiers HuggingFace, dossier
de documents maison, export d'un wiki) en deux listes normalisees :
documents et questions. C'est le seul endroit qui connait le format
d'origine ; tout le reste du banc travaille sur cette forme.

Brancher un corpus a soi revient donc a ecrire une fonction qui renvoie un
LoadedDataset — rien d'autre dans le projet n'a a changer.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class LoadedDataset:
    name: str
    source: str
    description: str
    # [{external_id, title, body, metadata}]
    documents: list[dict[str, Any]]
    # [{external_id, question, answer, question_type, gold_evidence, metadata}]
    #
    # gold_evidence : [{document_external_id, fact}]. Une liste VIDE a un
    # sens fort — la question est reputee sans reponse dans le corpus, et
    # sert a mesurer l'abstention. Ne pas confondre avec "evidence non
    # annotee" : un dataset sans verite terrain de retrieval doit le
    # declarer via metadata["has_gold_evidence"] = False.
    questions: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        types: dict[str, int] = defaultdict(int)
        for q in self.questions:
            types[q.get("question_type") or "unknown"] += 1
        unanswerable = sum(1 for q in self.questions if not q.get("gold_evidence"))
        return {
            "documents": len(self.documents),
            "questions": len(self.questions),
            "types": dict(types),
            "unanswerable": unanswerable,
        }


class Loader(Protocol):
    def __call__(self, raw_dir: Path, **kwargs: Any) -> LoadedDataset: ...


_REGISTRY: dict[str, Loader] = {}


def register(name: str) -> Callable[[Loader], Loader]:
    def _decorate(fn: Loader) -> Loader:
        _REGISTRY[name] = fn
        return fn

    return _decorate


def get_loader(name: str) -> Loader:
    if name not in _REGISTRY:
        raise KeyError(f"dataset inconnu : {name} (disponibles : {', '.join(sorted(_REGISTRY))})")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def stratified_sample(
    questions: list[dict[str, Any]], n: int, *, seed: int = 42
) -> list[dict[str, Any]]:
    """Sous-echantillon stratifie par question_type, deterministe.

    Pourquoi stratifier : les types de questions n'ont pas la meme
    difficulte (une comparaison oui/non n'a rien a voir avec une inference
    multi-hop). Un tirage uniforme ferait varier la composition entre deux
    campagnes, et une variation de score refleterait le tirage plutot que
    la config.

    Pourquoi deterministe : toutes les configs d'une matrice doivent voir
    exactement les memes questions, sinon elles ne sont pas comparables.
    """
    if n >= len(questions):
        return list(questions)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_type[q.get("question_type") or "unknown"].append(q)

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    total = len(questions)

    for qtype in sorted(by_type):
        bucket = sorted(by_type[qtype], key=lambda q: q["external_id"])
        rng.shuffle(bucket)
        # Repartition proportionnelle, avec au moins 1 question par type
        # present : perdre entierement une categorie rendrait l'analyse par
        # type impossible.
        quota = max(1, round(n * len(bucket) / total))
        out.extend(bucket[:quota])

    # L'arrondi par strate peut deborder ou manquer la cible : on ajuste
    # sur l'ensemble, toujours de facon deterministe.
    out.sort(key=lambda q: q["external_id"])
    if len(out) > n:
        rng.shuffle(out)
        out = out[:n]
        out.sort(key=lambda q: q["external_id"])
    return out


def read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))

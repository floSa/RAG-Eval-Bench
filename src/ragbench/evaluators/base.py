"""Interface commune des evaluateurs.

Un evaluateur prend les predictions d'un run et renvoie des Score. C'est
tout. Ragas, DeepEval, RAGChecker et les metriques maison se plient au meme
contrat, ce qui permet de stocker leurs sorties dans la MEME table `scores`
avec une colonne `evaluator` — et donc de mesurer leur desaccord sur la
meme metrique et la meme question. C'est l'objet principal du projet.

Deux niveaux de score coexistent :
  - question_id renseigne : score par question, agregeable, indispensable
    au bootstrap et aux tests apparies ;
  - question_id a None : agregat de run que la moyenne ne saurait pas
    reconstituer (accuracy ventilee par type, taux d'abstention...).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import PipelineConfig
from ..llm import LLMClient


@dataclass
class Score:
    metric: str
    value: float | None
    question_id: int | None = None
    # Trace du jugement. Sans elle un score de 0.62 est ininterpretable et
    # le drill-down sur les echecs est impossible.
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalContext:
    """Tout ce dont un evaluateur peut avoir besoin, et rien de plus.

    Les predictions viennent de db.load_predictions() : chaque ligne porte
    contexts, answer, gold_answer, gold_evidence, question_type.
    """

    predictions: list[dict[str, Any]]
    config: PipelineConfig
    llm: LLMClient
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    # Passe tel quel depuis la CLI : permet a un evaluateur d'exposer un
    # reglage (ex. k pour les metriques de retrieval) sans polluer la
    # PipelineConfig, qui ne decrit que le pipeline evalue.
    options: dict[str, Any] = field(default_factory=dict)

    def answerable(self) -> list[dict[str, Any]]:
        """Predictions dont la question a au moins une evidence gold.

        Les questions sans reponse (null_query) faussent toute metrique de
        retrieval : leur recall est indefini, pas nul. Elles sont evaluees
        a part, sur l'abstention.
        """
        return [p for p in self.predictions if p.get("gold_evidence")]

    def unanswerable(self) -> list[dict[str, Any]]:
        return [p for p in self.predictions if not p.get("gold_evidence")]


def align_by_input(
    predictions: list[dict[str, Any]],
    results: list[Any],
    *,
    key: Callable[[Any], str | None],
) -> tuple[list[tuple[dict[str, Any], Any]], int]:
    """Rattache des resultats de framework a leurs predictions, PAR LE TEXTE.

    A utiliser systematiquement plutot qu'un zip positionnel. Verifie sur
    DeepEval : en mode asynchrone il renvoie les cas dans leur ordre de
    COMPLETION, pas d'entree — le verdict du cas 1 revient attache au cas 0.
    Un zip attribue alors chaque score a la mauvaise question.

    C'est le pire type de defaut pour un banc d'evaluation. Il ne provoque
    aucune erreur, les moyennes agregees restent EXACTEMENT les memes (ce
    sont les memes valeurs, permutees), et seul le drill-down revele une
    reponse affichee a cote du verdict d'une autre question. Autrement dit,
    il ne se voit que si on regarde precisement la ou personne ne regarde.

    Renvoie (paires appariees, nombre de resultats orphelins). Les
    orphelins sont comptes et ecartes, jamais rattaches au petit bonheur.
    """
    by_question = {p["question"]: p for p in predictions}
    aligned: list[tuple[dict[str, Any], Any]] = []
    orphans = 0
    for result in results:
        prediction = by_question.get(key(result))
        if prediction is None:
            orphans += 1
            continue
        aligned.append((prediction, result))
    return aligned, orphans


class Evaluator(Protocol):
    name: str
    # 'gold_evidence' | 'gold_answer' | 'judge' — verifie avant lancement,
    # pour ne pas decouvrir apres 40 minutes de GPU qu'il manquait la
    # verite terrain.
    requires: tuple[str, ...]

    async def evaluate(self, ctx: EvalContext) -> list[Score]: ...


# ---------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}
# Plugins dont l'import a echoue : nom -> raison. Un framework absent est
# desactive, jamais fatal. Leurs dependances sont des extras qui entrent
# regulierement en conflit entre elles ; exiger qu'elles soient toutes
# installees rendrait le banc ininstallable.
_UNAVAILABLE: dict[str, str] = {}


def register(cls: type) -> type:
    _REGISTRY[cls.name] = cls
    return cls


def try_import(module: str, plugin_name: str) -> None:
    """Charge un module de plugin, et note l'echec sans le propager."""
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - on veut la raison exacte
        _UNAVAILABLE[plugin_name] = f"{type(exc).__name__}: {exc}"


def get(name: str):
    if name in _REGISTRY:
        return _REGISTRY[name]()
    if name in _UNAVAILABLE:
        raise KeyError(
            f"evaluateur '{name}' indisponible — {_UNAVAILABLE[name]}\n"
            f"  installer l'extra correspondant : uv sync --extra {name.split('.')[0]}"
        )
    raise KeyError(f"evaluateur inconnu : {name} (disponibles : {', '.join(available())})")


def available() -> list[str]:
    return sorted(_REGISTRY)


def unavailable() -> dict[str, str]:
    return dict(_UNAVAILABLE)


def check_requirements(evaluator: Any, ctx: EvalContext) -> list[str]:
    """Verifie les prerequis AVANT de consommer du GPU."""
    problems: list[str] = []
    requires = getattr(evaluator, "requires", ())

    if "gold_evidence" in requires and not ctx.answerable():
        problems.append("aucune question ne porte d'evidence gold")
    if "gold_answer" in requires and not any(p.get("gold_answer") for p in ctx.predictions):
        problems.append("aucune reponse de reference dans le jeu de questions")
    if "judge" in requires and ctx.config.models.judge == ctx.config.models.generator:
        # Avertissement et non blocage : on veut pouvoir produire une
        # baseline volontairement biaisee, a condition que ce soit trace.
        problems.append(
            f"[avertissement] juge == generateur ({ctx.config.models.judge}), "
            "scores optimistes"
        )
    return problems

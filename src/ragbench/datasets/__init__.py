"""Chargement des corpus et jeux de questions.

Ajouter un dataset = ecrire une fonction decoree @register qui renvoie un
LoadedDataset, et l'importer ici. Rien d'autre dans le banc ne change.
"""

# Les imports servent a peupler le registre — l'ordre fixe l'ordre
# d'affichage dans `ragbench dataset list`.
from . import (
    hotpotqa,  # noqa: F401
    multihop_rag,  # noqa: F401
)
from .base import LoadedDataset, available, get_loader, register, stratified_sample

__all__ = ["LoadedDataset", "available", "get_loader", "register", "stratified_sample"]

"""Les plugins de mesure.

Deux familles, et la distinction compte :

  native.*   metriques maison, deterministes, sans appel LLM. Gratuites,
             reproductibles, et non biaisees par un juge. Ce sont elles qui
             servent de reference pour calibrer les autres.

  <framework>  adaptateurs vers Ragas, DeepEval, RAGChecker, TruLens... Ils
             apportent des metriques que le deterministe ne sait pas
             produire (fidelite, pertinence), au prix d'un juge LLM et donc
             de ses biais.

Les adaptateurs sont importes en tolerance de panne : leurs dependances
sont des extras dont les contraintes de version entrent souvent en conflit.
Un framework absent apparait dans `ragbench eval list` avec sa raison, il
ne casse pas le banc.
"""

from .base import (  # noqa: F401
    EvalContext,
    Evaluator,
    Score,
    available,
    check_requirements,
    get,
    register,
    try_import,
    unavailable,
)

# --- natifs : aucune dependance externe, toujours disponibles -----------
from . import native_ir  # noqa: F401
from . import native_answer  # noqa: F401
from . import native_nuggets  # noqa: F401
from . import native_erag  # noqa: F401
from . import native_claims  # noqa: F401

# --- adaptateurs : optionnels -------------------------------------------
try_import("ragbench.evaluators.ragas_adapter", "ragas")
try_import("ragbench.evaluators.deepeval_adapter", "deepeval")

__all__ = [
    "EvalContext",
    "Evaluator",
    "Score",
    "available",
    "check_requirements",
    "get",
    "register",
    "unavailable",
]

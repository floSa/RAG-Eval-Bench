"""Statistiques de comparaison de runs.

Raison d'etre : sur 200 questions, un ecart de 2 points entre deux configs
n'est pas un resultat, c'est du bruit. Un banc qui affiche des moyennes
nues invite a conclure de travers, et c'est exactement l'erreur que ce
projet doit rendre impossible.

Trois outils, chacun pour une question distincte :

  bootstrap_ci      « quelle est la precision de cette mesure ? »
  paired_bootstrap  « B est-il meilleur que A ? »  <- le plus important
  mcnemar           « B est-il meilleur que A, sur une metrique binaire ? »

Le caractere APPARIE est le point central. Deux configs sont evaluees sur
les MEMES questions ; comparer leurs intervalles de confiance separes
ignore cette information et perd enormement de puissance — deux
intervalles qui se chevauchent peuvent parfaitement recouvrir une
difference systematique. On compare donc la distribution des ECARTS
question par question.

Aucune dependance : `statistics` et `random` de la bibliotheque standard
suffisent, et le bootstrap evite d'avoir a supposer une loi normale sur des
scores bornes a [0, 1] souvent bimodaux.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean


@dataclass
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.low:.4f}, {self.high:.4f}] (n={self.n})"

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


@dataclass
class Comparison:
    """Resultat d'une comparaison appariee entre deux runs."""

    mean_a: float
    mean_b: float
    delta: float          # b - a
    ci_low: float
    ci_high: float
    p_value: float
    n_pairs: int
    n_better: int         # questions ou b > a
    n_worse: int
    n_tied: int

    @property
    def significant(self) -> bool:
        """Vrai si l'intervalle de confiance de l'ecart exclut zero.

        Critere unique et explicite. Note qu'il ne dit rien de l'ampleur :
        un ecart significatif de 0,01 reste sans interet pratique.
        """
        return (self.ci_low > 0) or (self.ci_high < 0)

    def verdict(self) -> str:
        if not self.significant:
            return (
                f"ecart non concluant ({self.delta:+.4f}, IC95 "
                f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]) — n={self.n_pairs} "
                f"insuffisant pour trancher"
            )
        direction = "meilleur" if self.delta > 0 else "moins bon"
        return (
            f"B {direction} de {abs(self.delta):.4f} "
            f"(IC95 [{self.ci_low:+.4f}, {self.ci_high:+.4f}], p={self.p_value:.4f})"
        )


def bootstrap_ci(
    values: list[float], *, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42
) -> Interval:
    """Intervalle de confiance par percentiles du bootstrap.

    Bootstrap plutot que t de Student : les scores d'evaluation sont bornes
    a [0, 1] et souvent bimodaux (0 ou 1 sur une metrique binaire). Un
    intervalle normal deborderait alors hors de [0, 1] et serait mal
    calibre.
    """
    if not values:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if len(values) == 1:
        return Interval(values[0], values[0], values[0], 1)

    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return Interval(fmean(values), lo, hi, n)


def paired_bootstrap(
    a: list[float], b: list[float], *, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42
) -> Comparison:
    """Compare deux runs sur les memes questions, dans le meme ordre.

    On bootstrappe la moyenne des ECARTS (b_i - a_i), pas les deux
    moyennes separement : la variance entre questions — enorme, certaines
    questions sont dures pour tout le monde — s'annule dans l'ecart. C'est
    ce qui rend detectable une amelioration de quelques points.

    p_value : test de permutation bilateral. Sous l'hypothese nulle, le
    signe de chaque ecart est arbitraire ; on tire des signes au hasard et
    on compte combien de fois on obtient un ecart moyen au moins aussi
    extreme que l'observe.
    """
    if len(a) != len(b):
        raise ValueError(f"echantillons non apparies : {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("aucune paire a comparer")

    deltas = [bi - ai for ai, bi in zip(a, b)]
    observed = fmean(deltas)
    rng = random.Random(seed)
    n = len(deltas)

    boot = sorted(
        fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)
    )
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]

    rng_perm = random.Random(seed + 1)
    extreme = sum(
        1
        for _ in range(n_boot)
        if abs(fmean(d if rng_perm.random() < 0.5 else -d for d in deltas)) >= abs(observed)
    )
    # +1 au numerateur et au denominateur : evite p=0, qui laisserait croire
    # a une certitude que n observations ne peuvent pas donner.
    p_value = (extreme + 1) / (n_boot + 1)

    return Comparison(
        mean_a=fmean(a),
        mean_b=fmean(b),
        delta=observed,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        n_pairs=n,
        n_better=sum(1 for d in deltas if d > 0),
        n_worse=sum(1 for d in deltas if d < 0),
        n_tied=sum(1 for d in deltas if d == 0),
    )


def mcnemar(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """Test de McNemar pour metriques binaires (exact match, hit_rate...).

    Renvoie (b_gagne, a_gagne, p_value). Seules les questions ou les deux
    systemes DIFFERENT portent de l'information : les accords, majoritaires,
    ne disent rien sur laquelle des deux configs est meilleure.

    p exact binomial bilateral, calcule directement — pas d'approximation
    du chi2, qui est mauvaise quand les discordances sont peu nombreuses,
    ce qui est le cas courant a n=200.
    """
    if len(a) != len(b):
        raise ValueError(f"echantillons non apparies : {len(a)} vs {len(b)}")

    b_wins = sum(1 for ai, bi in zip(a, b) if bi > ai)
    a_wins = sum(1 for ai, bi in zip(a, b) if ai > bi)
    n = b_wins + a_wins
    if n == 0:
        return 0, 0, 1.0

    k = min(b_wins, a_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return b_wins, a_wins, min(1.0, 2 * tail)


def cohen_kappa(rater_a: list[float], rater_b: list[float]) -> float:
    """Accord entre deux annotateurs, corrige du hasard.

    C'est LA mesure qui valide un juge LLM. Un juge dont le kappa contre
    l'humain est faible ne produit pas une metrique, il produit une
    opinion — et l'ecart entre les deux ne se voit pas dans la moyenne des
    scores, qui peut etre parfaitement plausible.

    Reperes usuels : < 0,4 faible ; 0,4-0,6 moyen ; 0,6-0,8 substantiel ;
    > 0,8 excellent. Les valeurs sont arrondies a l'entier, donc a n'utiliser
    que sur des echelles discretes (binaire ou ordinale courte).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"listes de tailles differentes : {len(rater_a)} vs {len(rater_b)}")
    if not rater_a:
        return float("nan")

    a = [round(x) for x in rater_a]
    b = [round(x) for x in rater_b]
    n = len(a)

    observed = sum(1 for x, y in zip(a, b) if x == y) / n

    categories = set(a) | set(b)
    expected = sum(
        (a.count(c) / n) * (b.count(c) / n) for c in categories
    )
    if expected == 1.0:
        # Accord total ET distribution degeneree (tout le monde note pareil) :
        # kappa est indefini. Renvoyer 1.0 masquerait le fait qu'aucune
        # information n'a ete produite.
        return float("nan")
    return (observed - expected) / (1 - expected)


def required_n(effect: float, *, baseline: float = 0.5, power: float = 0.8) -> int:
    """Taille d'echantillon approximative pour detecter un ecart donne.

    Approximation normale pour une proportion appariee, volontairement
    grossiere : elle sert a repondre a « 200 questions suffisent-elles pour
    voir 5 points d'ecart ? » avant de lancer 3 heures de GPU, pas a
    publier un plan d'experience.
    """
    if effect <= 0:
        raise ValueError("l'effet doit etre > 0")
    z_alpha, z_beta = 1.96, 0.84 if power <= 0.8 else 1.28
    variance = baseline * (1 - baseline)
    return math.ceil(2 * variance * ((z_alpha + z_beta) / effect) ** 2)

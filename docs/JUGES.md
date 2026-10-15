# La fiabilité des juges

Quand il n'existe pas de bonne réponse à comparer — « cette réponse est-elle fidèle aux
sources ? » — un second modèle note. C'est le **juge**.

**Un juge non calibré transforme une mesure en opinion, sans prévenir.** Cette page dit
quoi vérifier et comment, sur ses propres données.

Cette branche (`outil`) ne contient aucun résultat de corpus. Les mesures faites sur le
corpus public sont sur la branche `main`.

Vocabulaire : [GLOSSAIRE.md](GLOSSAIRE.md). Métriques et statistique :
[METHODOLOGIE.md](METHODOLOGIE.md).

---

## 1. Trois défauts à connaître

- **Un juge LLM n'est pas reproductible.** Deux passes identiques peuvent donner des
  scores différents, même à température 0.
  - Fixer une graine réduit la variation sans l'annuler.
  - Un écart mesuré par un juge doit dépasser la variabilité de ce juge avant toute
    conclusion.
- **Un petit juge ne sait pas tout juger.** Certaines métriques exigent des sorties JSON
  profondes qu'un petit modèle ne produit pas.
  - Le banc publie ces métriques avec une **couverture de 0**, plutôt que de les omettre.
  - Une métrique absente se lirait comme une métrique non demandée.
- **Un modèle plus récent ne juge pas forcément mieux.** Un juge qui valide tout, ou qui
  conteste tout, est dégénéré.
  - Un accord élevé ne prouve rien si le juge répond toujours la même chose.
  - Seul le κ tranche, car il est corrigé du hasard.

---

## 2. Calibrer un juge

Principe : comparer le juge à une référence déterministe sur les mêmes réponses.

```bash
ragbench eval run <run_id> -e native.claims -o judge=<modele> -o max_samples=60
ragbench calibrate <run_id>
```

- La référence par défaut est `native.answer/contains`. Elle exige des **réponses de
  référence** dans le jeu de données.
- `-o judge=<modele>` fait rejuger un run existant. Les scores sont stockés sous
  `évaluateur@juge`.
- Pour comparer plusieurs juges, les faire noter **les mêmes réponses**.

Lecture du κ de Cohen :

| κ | Lecture |
|---|---|
| < 0,4 | Faible : ne pas utiliser |
| 0,4 – 0,6 | Moyen : tendance seulement |
| 0,6 – 0,8 | Substantiel : utilisable avec prudence |
| > 0,8 | Excellent : le juge fait foi |

- Les colonnes **Généreux** et **Sévère** donnent la direction des erreurs.
- Un juge à κ proche de 0 n'est pas meilleur que le hasard, quelle que soit sa taille.
- L'accord brut seul est trompeur : si 80 % des réponses sont fausses, un juge qui répond
  toujours « fausse » atteint 80 % d'accord et un κ de 0.
- Avec peu d'items (moins de 30 à 50), le κ est instable. L'intervalle compte autant que
  la valeur.

---

## 3. Choisir la taille d'un juge

La taille ne garantit pas la qualité. Elle se mesure.

- Faire noter **les mêmes réponses** par plusieurs juges de tailles différentes.
- Calculer le κ de chacun contre la référence.
- Retenir le **plus petit juge dont le κ atteint le seuil visé**, par exemple 0,6.
- Au-delà de ce seuil, un juge plus grand coûte du temps sans apport démontré.
- Choisir un juge d'une **autre famille** que le générateur : un modèle qui se note
  lui-même est biaisé.
- Garder le même juge pendant toute une comparaison de configurations.

**Option : un grand modèle comme étalon.**

- Faire juger un échantillon par un grand modèle de fondation, puis calculer le κ des
  petits juges contre lui.
- Cela dit si un petit modèle local suffit, ou si seul un gros modèle atteint la qualité
  visée.
- Cette phase implique un **appel sortant**. Elle ne fait pas partie du pipeline
  reproductible.
- **Ne pas la faire si les données sont sensibles.**

---

## 4. La référence a elle aussi un plafond

`contains` vérifie qu'une chaîne est incluse dans une autre. Une réponse correcte donnée
sous forme abrégée est comptée fausse (par exemple « Bankman-Fried » contre
« Sam Bankman-Fried »).

- Les valeurs de `contains` sont des **bornes basses** de la justesse réelle.
- Les **comparaisons** restent valides : le biais s'applique aux deux configurations et
  s'annule dans l'écart apparié.
- Lire « la justesse vaut au moins X », pas « la justesse vaut X ».
- Des réponses de référence **courtes** réduisent ce défaut.

---

## 5. Fidélité et justesse

La référence porte sur la **justesse**. Elle ne valide pas un juge de **fidélité**.

- Une réponse peut être **fidèle aux passages et fausse**, quand le mauvais passage a
  été remonté.
- Une réponse peut être **juste et mal soutenue** par les passages.
- Publier toujours les deux mesures, jamais l'une à la place de l'autre.

Pour valider un juge de fidélité, une annotation humaine est nécessaire :

1. Tirer au hasard une trentaine de réponses non abstenues.
2. Les juger à la main, avec les passages remontés **intégraux** sous les yeux. Critère :
   chaque affirmation est-elle soutenue par les passages ?
3. Saisir les verdicts dans l'onglet **Annotation** du tableau de bord.
4. Lire le κ entre l'humain et le juge.

Le script [annotation_fidelite.py](../scripts/annotation_fidelite.py) donne le patron.

Une réponse fausse mais fidèle indique un problème de **recherche**, pas de génération :
avant de changer de modèle, vérifier `nugget_full_coverage`.

---

## 6. À retenir

- `ragbench calibrate` **n'est pas optionnel** : un juge à κ < 0,4 sur *votre* domaine
  ne doit pas servir à trancher, quelles que soient ses performances ailleurs.
- La calibration par `contains` couvre la justesse. La fidélité demande un humain.
- La marche à suivre complète : [BRANCHER-SES-DONNEES.md](BRANCHER-SES-DONNEES.md).
- Prérequis : [PREREQUIS.md](PREREQUIS.md).

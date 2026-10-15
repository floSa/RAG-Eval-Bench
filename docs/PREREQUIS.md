# Prérequis

Ce qu'il faut réunir avant d'évaluer un RAG avec ce banc. Une page, à lire avant tout le
reste.

- La démarche complète, pas à pas : [BRANCHER-SES-DONNEES.md](BRANCHER-SES-DONNEES.md).
- Le sens des termes : [GLOSSAIRE.md](GLOSSAIRE.md).

---

## 1. Ce que le banc évalue

- Le banc évalue **son propre pipeline** : découpage, indexation, recherche, génération.
- Il **ne se branche pas** sur un RAG existant pour noter ses réponses.
- Pour évaluer un autre RAG, il faut le **reproduire** sous forme de configuration
  (`configs/*.yml`).
- L'import des réponses d'un RAG externe n'existe pas encore. Il est prévu.

---

## 2. Les données à fournir

| Élément | Statut | Ce qu'il permet de mesurer | Sans lui |
|---|---|---|---|
| **Documents** (identifiant, titre, texte, métadonnées) | Obligatoire | Tout | Rien ne tourne |
| **Questions** | Obligatoire | Tout | Rien ne tourne |
| **Réponse de référence courte** par question | Fortement recommandé | `contains`, `em`, `token_f1` et calibration des juges | Il ne reste que des juges LLM non calibrables |
| **Passage attendu** (`gold_evidence`) | Recommandé | `recall@k`, `nDCG@k`, `mrr` : l'échec vient-il de la recherche ou de la génération ? | Les deux causes restent indiscernables |
| **Questions sans réponse** (20 à 30 %) | Recommandé | Le rejet correct : savoir dire « je ne sais pas » | L'abstention n'est jamais mesurée |
| **Type de question** (`question_type`) | Optionnel | Échantillon stratifié, analyse par type | Tout est regroupé |

Précisions :

- Une liste `gold_evidence` **vide** signifie « question sans réponse dans le corpus ».
- Un corpus sans vérité terrain de recherche doit le déclarer :
  `metadata["has_gold_evidence"] = False`.
- Les réponses de référence doivent rester **courtes**. `contains` compte faux une
  réponse correcte donnée sous forme abrégée.
- Le format exact est décrit dans [BRANCHER-SES-DONNEES.md](BRANCHER-SES-DONNEES.md#1-charger-vos-documents-et-vos-questions).

---

## 3. Les questions

- **Volume** : 50 à 100 questions suffisent pour départager deux configurations.
- **Origine, par ordre de qualité** :
  - de vraies questions d'utilisateurs, avec la réponse attendue ;
  - des questions écrites par un expert métier ;
  - des questions générées par un LLM, relues par un humain.
- **Biais à corriger** : des questions écrites à partir des documents ont forcément une
  réponse dans le corpus. Ajouter volontairement 20 à 30 % de questions sans réponse.
- **Un écart trop petit ne se tranche pas** avec peu de questions. L'onglet *Runs* du
  tableau de bord affiche la précision du run : un écart plus petit n'est pas mesurable.

---

## 4. L'annotation humaine

Elle est limitée à un échantillon.

- **Niveau 0 — aucune annotation** : seuls des juges LLM sont disponibles. Les chiffres
  sont des opinions, pas des mesures.
- **Niveau 1 — réponses de référence** : les métriques déterministes et la calibration
  des juges par `ragbench calibrate` deviennent possibles.
- **Niveau 2 — passages attendus** : le diagnostic recherche contre génération devient
  possible.
- **Niveau 3 — 30 jugements humains** tirés au hasard, saisis dans l'onglet
  *Annotation* : le κ de Cohen entre l'humain et le juge valide la fidélité.

Lecture du κ :

- **> 0,8** : le juge fait foi.
- **0,6 à 0,8** : utilisable, avec prudence.
- **< 0,6** : à ne pas utiliser pour trancher.

---

## 5. Tester une fonctionnalité

Le but du banc : savoir si un ajout améliore un RAG. Cela impose :

- **Des variantes** : la fonctionnalité doit pouvoir être activée et désactivée par
  configuration, tout le reste étant identique.
- **Les mêmes questions** pour chaque variante.
- **Un seul changement à la fois**, sinon l'écart n'est pas attribuable.
- **Un juge fixe** pendant toute la comparaison. Le changer rend les scores de fidélité
  incomparables.
- **Le test apparié** : `ragbench compare <A> <B>`. Deux moyennes côte à côte ne
  prouvent rien.

---

## 6. L'infrastructure

- **Python 3.12** et **uv**.
- **PostgreSQL 17 avec pgvector**, fourni par `docker compose up -d db`.
- **Un serveur de modèles à API OpenAI-compatible** (Ollama ou vLLM), avec :
  - un modèle de génération ;
  - un modèle d'embedding ;
  - un modèle juge, de préférence d'une autre famille que le générateur.
- **Un GPU** n'est pas obligatoire, mais une campagne sur CPU est très lente.
- **Aucun appel sortant** : les documents ne quittent pas la machine.

Vérification avant toute campagne :

```bash
uv run ragbench doctor
```

---

## 7. Réglages pour un corpus non anglophone

- `to_tsvector('english', …)` dans [rag/retrieve.py](../src/ragbench/rag/retrieve.py) :
  remplacer par `'french'` pour du français.
- `chunk_size` : réduire pour une documentation très structurée.
- `header_fields` : mettre ses propres champs (version, service, référence). Ce réglage a
  apporté +0.102 de justesse sur le corpus de référence.

---

## 8. Liste de contrôle avant de lancer

- [ ] Les documents ont un identifiant stable.
- [ ] 50 questions au minimum, avec une réponse de référence courte.
- [ ] 20 à 30 % de questions sans réponse, avec `gold_evidence` vide.
- [ ] Les passages attendus sont renseignés, ou `has_gold_evidence = False` est déclaré.
- [ ] La fonctionnalité à tester s'active par configuration.
- [ ] `ragbench doctor` ne signale aucune erreur.
- [ ] Un expert est disponible pour relire 30 jugements.

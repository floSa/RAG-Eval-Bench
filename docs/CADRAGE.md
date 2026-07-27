# Cadrage — ragbench

Le POURQUOI. Le COMMENT est dans [ARCHITECTURE.md](ARCHITECTURE.md), la méthode de
mesure dans [METHODOLOGIE.md](METHODOLOGIE.md).

## 1. Pitch

Un banc d'expérimentation qui compare des configurations RAG entre elles, entièrement
on-premise, en produisant des chiffres sur lesquels on a le droit de s'appuyer.

1. **Exécuter** une matrice de configurations sur un jeu de questions fixe, avec
   traçabilité complète (hash de configuration, version du code, consommation de tokens).
2. **Mesurer** les trois étages séparément — retrieval, génération, bout en bout — avec
   sept évaluateurs, dont cinq déterministes et deux frameworks externes.
3. **Trancher** statistiquement : test apparié et intervalle de confiance sur chaque
   comparaison, jamais deux moyennes mises côte à côte.

Cas d'usage type : « j'hésite entre recherche dense et hybride ». Le banc répond
« hybride, +0.098 de recall@3, p<0.001 » — et si l'écart n'est pas concluant, il le dit.

---

## 2. Objectifs & périmètre

**Dans le périmètre**

- Comparaison de configurations RAG sur un corpus fixe, avec vérité terrain.
- Attribution de la faute : le banc doit dire si un échec vient du retrieval ou du
  générateur, pas seulement qu'il y a échec.
- Branchement de plusieurs frameworks d'évaluation sur les **mêmes** prédictions, pour
  mesurer leur désaccord.
- Fonctionnement 100 % local : aucun appel à un service externe, aucune donnée qui sort.

**Hors périmètre**

- Servir un RAG en production. Le pipeline de [src/ragbench/rag/](../src/ragbench/rag/)
  existe pour être mesuré, pas pour être exploité.
- L'optimisation automatique d'hyperparamètres. Le banc mesure, l'humain décide.
- Le multi-tenant, l'authentification, la haute disponibilité.
- L'entraînement ou le fine-tuning de modèles.

---

## 3. Contraintes (fermes)

| Contrainte | Détail |
|---|---|
| Déploiement | On-premise strict. Aucun appel sortant vers un fournisseur de LLM. |
| Matériel | Un seul GPU **NVIDIA RTX 4060 Ti, 16 Go**. Contraint la taille des modèles et le parallélisme. |
| Service LLM | Mutualisé avec les autres projets via le dépôt `llm-service` (Ollama central). Le banc ne lance pas son propre serveur d'inférence. |
| Licences | Open-source uniquement. Voir le tableau du [README](../README.md#licences--composants). |
| Python | **3.12**, géré par `uv`. Source de vérité : `pyproject.toml` + `uv.lock`. |

---

## 4. Hypothèses

Ces hypothèses ne sont pas lisibles dans le code. Elles conditionnent l'interprétation
des résultats.

- **La vérité terrain du dataset fait autorité.** Les `evidence_list` de MultiHop-RAG
  sont traitées comme correctes et complètes. Une evidence manquante dans le dataset
  ferait apparaître un faux échec de retrieval. Ce qui la remettrait en cause : un
  audit manuel montrant des questions répondables par des documents non listés.

- **Un échantillon de 200 questions suffit pour départager deux configurations.**
  Vérifié empiriquement : les intervalles de confiance obtenus (± 0,05 sur `recall@3`)
  permettent de conclure sur des écarts de 6 points ou plus. En dessous, il faut
  agrandir l'échantillon — les 2 356 questions restantes sont conservées en base sous
  le split `pool` exactement pour ça.

- **Le corpus est du texte anglais journalistique.** Le découpage, la recherche
  lexicale (`to_tsvector('english', …)`) et les seuils de recouvrement sont réglés pour
  ça. Un corpus français ou du code demanderait de revoir la configuration de recherche
  plein-texte.

- **La granularité de la vérité terrain est le DOCUMENT, pas le passage.** Les métriques
  de retrieval dédupliquent donc les chunks par document. `native.nuggets` compense
  partiellement en vérifiant la présence des faits dans les passages.

- **Le générateur ignore ce qu'il n'a pas dans son contexte.** Hypothèse fausse en
  pratique : le corpus est de l'actualité 2023 que `gemma4:e4b` connaît partiellement.
  C'est précisément pourquoi la configuration témoin `closed-book` existe — sans elle,
  on attribuerait au retrieval des réponses venues de la mémoire du modèle.

---

## 5. Stack technique

| Brique | Choix | Licence |
|---|---|---|
| Langage | Python 3.12 | PSF |
| Paquets | uv + `pyproject.toml` / `uv.lock` | MIT / Apache-2.0 |
| Base de données | PostgreSQL 17 + pgvector | PostgreSQL License / MIT |
| Accès base | psycopg 3.3 | **LGPL-3.0-only** |
| Inférence | Ollama (dépôt `llm-service`), API OpenAI-compatible | MIT |
| CLI | typer + rich | MIT |
| Interface | Streamlit 1.60 | Apache-2.0 |
| Frameworks d'évaluation | Ragas 0.4, DeepEval 4.1 | Apache-2.0 |
| Corpus | MultiHop-RAG, HotpotQA | ODC-BY / CC BY-SA 4.0 |

Tableau complet dans le [README](../README.md#licences--composants).

---

## 6. Décisions

### Décisions figées

- **Tout passe par une API OpenAI-compatible** plutôt que par les API natives de chaque
  moteur, **parce que** Ollama et vLLM l'exposent tous deux et que *tous* les frameworks
  d'évaluation acceptent un `base_url` custom. Un seul point d'entrée suffit donc pour
  brancher n'importe quel framework sur n'importe quel moteur.
  *Limite* : l'endpoint OpenAI-compatible d'Ollama n'expose pas le réglage du
  raisonnement. Une échappatoire documentée vers `/api/chat` existe pour ce seul cas —
  voir [ARCHITECTURE.md](ARCHITECTURE.md#6-décisions-darchitecture).

- **Deux identités distinctes pour une configuration** : `hash()` couvre toute la
  configuration, `index_hash()` seulement le couple (découpage, embedder). **Parce que**
  comparer `top_k=5` et `top_k=10` ne doit pas réindexer 609 documents. Mesuré : une
  matrice de 8 configurations ne construit que 2 index.

- **Le corpus principal est MultiHop-RAG** plutôt qu'un corpus maison, **parce qu'**il
  porte les passages de référence de chaque question — les métriques de retrieval sont
  donc calculables sans juge LLM — et qu'il contient 301 questions délibérément sans
  réponse, ce qui permet de mesurer le *negative rejection*.
  *Limite* : ses réponses gold font une à trois mots, ce qui rend l'exact-match
  utilisable mais fausse la lecture d'une accuracy globale (52 % de questions binaires).

- **Métriques déterministes d'abord, juges LLM ensuite.** **Parce que** les
  déterministes sont gratuites, reproductibles et non biaisées, et qu'elles servent de
  référence pour calibrer les juges. Mesuré : les juges LLM ne sont **pas**
  reproductibles (voir [METHODOLOGIE.md](METHODOLOGIE.md#5-la-fiabilité-des-juges)).

- **Aucun index ANN sur les embeddings** (ni ivfflat ni hnsw), parcours exact,
  **parce qu'**un index approximatif introduit une perte de recall qui se confondrait
  avec la qualité du modèle d'embedding : on mesurerait l'index, pas le retrieval.
  *Limite* : ne tient que sur un corpus de banc d'essai (~24 000 chunks aujourd'hui).

- **Le service LLM central est le mode par défaut**, pas une option. **Parce qu'**une
  campagne consomme des milliers d'appels et qu'un second Ollama à côté du central
  saturerait les 16 Go de VRAM, faussant toute mesure de latence.

- **Le juge par défaut reste `llama3.2:3b`** malgré son âge, **parce que** c'est le seul
  des quatre candidats testés dont l'accord avec la vérité terrain dépasse le hasard.
  Trois modèles nettement plus récents ont été mesurés sur les mêmes 60 réponses du
  run 24 :

  | Juge | Publié | κ `claim_precision` | κ `claim_recall` | Trop généreux | Trop sévère | n |
  |---|---|---|---|---|---|---|
  | `llama3.2:3b` | ~1 an | +0.048 | **+0.458** | 0.200 | 0.229 | 48–50 |
  | `granite4.1:3b` | 2 mois | +0.191 | indéfini | 0.250 | **0.000** | 24 |
  | `nemotron-3-nano:4b` | 4 mois | **+0.000** | indéfini | 0.318 | **0.000** | 22 |
  | `qwen3.5:4b` | 4 mois | — | — | — | — | **0** |

  Lecture : une colonne « trop sévère » à 0.000 avec un `claim_recall` constant à 1.0
  signe un juge qui valide sans contester — κ indéfini parce que sa sortie ne varie pas.
  `nemotron-3-nano:4b` est donné état de l'art en suivi d'instructions (IFEval, IFBench)
  par NVIDIA ; cette compétence ne se transfère pas au jugement critique. `qwen3.5:4b`
  n'a produit aucun jugement exploitable. `llama3.2:3b` produit en outre deux fois plus
  de jugements exploitables (48 contre 22).

  *Limite* : n = 22 à 50, ces κ sont bruités. L'échec de `qwen3.5:4b` et le κ nul de
  `nemotron-3-nano:4b` sont nets ; l'écart entre `granite4.1:3b` et `llama3.2:3b` ne
  l'est pas. Reproductible via `ragbench eval run 24 -e native.claims -o judge=<modele>`
  puis `ragbench calibrate 24`.

  **Ce que ça invalide** : l'hypothèse — intuitive et fausse — qu'un modèle plus récent
  fait un meilleur juge. La date de publication n'a aucun pouvoir prédictif ici.

  **Ce que ça ne dit pas** : qu'un juge sévère aurait raison. Contester systématiquement
  est aussi dégénéré que valider systématiquement, et le κ le sanctionne pareil — un
  juge aléatoire obtient κ ≈ 0 quel que soit son penchant. Le +0.458 de `llama3.2:3b`
  affirme seulement que son accord avec la vérité terrain dépasse le hasard : « mieux
  que rien », pas « fiable ». Il reste sous le repère de 0,6.

### À trancher

- **Faut-il un profil vLLM ?** Le continuous batching donnerait un net gain de débit sur
  une charge d'évaluation. Reco par défaut : attendre qu'une campagne dépasse l'heure
  avant d'ajouter cette complexité. Contrainte connue : vLLM épingle un seul modèle en
  VRAM, sans swap, et exigerait des poids quantifiés sur 16 Go.

- **Faut-il un vérificateur NLI dédié** (HHEM-2.1-open, MiniCheck) pour la fidélité,
  plutôt qu'un juge génératif ? Reco par défaut : oui, à évaluer — HHEM tourne même en
  CPU. Non implémenté à ce jour.

---

## 7. Roadmap

Les phases 0 à 3 sont réalisées. La suite n'est pas commencée.

0. **Socle** — configuration hashée, schéma Postgres, client LLM, pipeline paramétrable.
1. **Dataset et référence** — chargement MultiHop-RAG, évaluateurs déterministes,
   première campagne chiffrée.
2. **Comparaison** — couche statistique, tableau de bord, tests de non-régression.
3. **Multi-frameworks** — évaluateurs à base de juge, adaptateurs Ragas et DeepEval,
   calibration contre la vérité terrain.
4. **Non commencé** — observabilité (traces OpenTelemetry), profil vLLM, jeu de
   robustesse (bruit injecté, contexte contradictoire), intégration continue.

---

## 8. Stratégie de tests

Trois niveaux, avec un objectif de preuve distinct.

| Niveau | Commande | Ce qu'on prouve |
|---|---|---|
| Unitaire | `uv run pytest -m "not regression"` | Les briques déterministes sont justes : découpage sans perte, hash stable, nDCG borné, appariement des résultats de frameworks. **84 tests.** |
| Non-régression | `uv run pytest tests/test_regression.py` | La qualité mesurée n'a pas baissé. Les assertions portent sur la **borne basse** de l'intervalle de confiance, pour échouer sur une régression établie et non sur un tirage défavorable. **7 tests.** |
| Bout en bout | `ragbench doctor` puis une campagne | La chaîne complète répond : Postgres, pgvector, modèles provisionnés, conteneurs sur `llm-net`. |

Les tests unitaires portent prioritairement sur ce qui **échoue en silence** : un
appariement positionnel qui permute des scores ne change aucune moyenne agrégée, un
découpage qui perd du texte ne lève aucune erreur. C'est là que les tests paient.

---

## 9. Références

État de l'art sur lequel le banc s'appuie :

| Travail | Apport repris |
|---|---|
| [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang & Yang, 2024) | Corpus principal, passages de référence, questions sans réponse |
| [HotpotQA](https://hotpotqa.github.io/) | Second corpus, pour valider que la couche dataset est pluggable |
| AutoNuggetizer, TREC RAG 2024 ([arXiv 2504.15068](https://arxiv.org/pdf/2504.15068)) | Nugget recall — τ = 0,87 de corrélation avec le jugement humain |
| eRAG (Salemi & Zamani, SIGIR 2024) | Utilité réelle d'un passage plutôt que sa pertinence jugée |
| RAGChecker (Amazon, NeurIPS 2024 D&B) | Décomposition en claims et attribution de l'erreur |
| [Ragas](https://docs.ragas.io/) | Les quatre métriques canoniques |
| [DeepEval](https://deepeval.com/) | Seuils pass/fail, donc non-régression en intégration continue |
| CALM (biais des juges LLM) | Justifie la contrainte juge ≠ générateur |
| [Biais de sycophantie des juges](https://pacific.ai/detecting-and-evaluating-sycophancy-bias-an-analysis-of-llm-and-ai-solutions/) · [arXiv 2510.12462](https://arxiv.org/pdf/2510.12462) | Les petits modèles ouverts notent haut sans fonder leur jugement (6–22 %) — explique le κ nul de `nemotron-3-nano:4b` |
| [Label Your Data — LLM as a Judge](https://labelyourdata.com/articles/llm-as-a-judge) | Protocole de validation d'un juge : κ juge↔référence, repère κ > 0,6 |
| [DeepEval — LLM-as-a-judge](https://deepeval.com/blog/llm-as-a-judge) | Confirme qu'aucun framework ne prescrit de modèle juge : le choix revient à la mesure |

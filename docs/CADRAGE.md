# Cadrage — ragbench

Le POURQUOI. Le COMMENT est dans [ARCHITECTURE.md](ARCHITECTURE.md), la méthode de
mesure dans [METHODOLOGIE.md](METHODOLOGIE.md).

## 1. Pitch

Une boîte à outils on-premise pour **évaluer et améliorer un RAG**. Les techniques
d'amélioration sont des briques qu'on active ou non ; un outillage d'évaluation fixe et
calibré les mesure. Pour chaque technique, le banc dit **ce qu'elle rapporte, ce qu'elle
coûte, et sur quels documents**.

Question type : « sur mes documents, le reranking vaut-il le coup ? ». Réponse type :
« sur le corpus de référence, non : −0.001, p = 0.96, pour 5× le coût. Sur vos contrats,
tendance à +0.08 sur les questions à documents proches, à confirmer. »

### Deux modes

| Mode | Corpus | Vérité terrain | Ce qu'on obtient |
|---|---|---|---|
| **Bench** | Corpus de référence (MultiHop-RAG, HotpotQA) | Fournie avec le corpus | Un **verdict** : gain, intervalle de confiance, test apparié |
| **Aperçu** | N'importe quelle source : PDF, docx, HTML, tableaux, exports | Questions générées, échantillon relu à la main | Une **tendance**, étiquetée comme telle |

Le bench tranche, l'aperçu oriente. Un aperçu désigne les techniques qui méritent
d'être confirmées ; il ne les déclare pas gagnantes.

### Deux familles de briques, à ne pas mélanger

| Famille | Rôle | Exemples |
|---|---|---|
| **Outils d'évaluation** — la règle | Mesurer | recall@k, nuggets, juges LLM calibrés, Ragas, DeepEval |
| **Techniques d'amélioration** — ce qu'on mesure | Changer les réponses | découpage, recherche hybride, reranking, reformulation, prompts, vérification dans le pipeline, graphe de connaissance |

**La règle reste fixe, seules les techniques bougent.** Sinon on ne sait plus si un score
a changé à cause de la technique ou de la règle. Un juge LLM est donc un instrument, pas
une amélioration ; un LLM qui vérifie les passages *dans* le pipeline (famille CRAG,
self-RAG) est une technique, et se mesure comme le reranking — avec un juge
d'évaluation **différent** de ce vérificateur, sans quoi le modèle se note lui-même.

### Trois exigences, héritées du socle

1. **Attribuer la faute** : mesurer séparément retrieval, génération et bout en bout.
2. **Trancher statistiquement** : test apparié et intervalle de confiance sur chaque
   comparaison, jamais deux moyennes mises côte à côte.
3. **Se méfier des juges** : les calibrer contre la vérité terrain et publier leur κ.

---

## 2. Objectifs & périmètre

**Dans le périmètre**

- Un **catalogue de techniques** activables par configuration, chacune entrant dans le
  hash : recherche des passages, découpage, génération, vérification dans le pipeline,
  graphe de connaissance.
- Un **outillage d'évaluation** fixe : métriques déterministes, juges calibrés,
  frameworks externes branchés sur les **mêmes** prédictions.
- Le **mode bench** sur corpus de référence avec vérité terrain.
- Le **mode aperçu** sur une source quelconque : ingestion multi-format, analyse du
  corpus, génération d'un jeu de test avec **questions pièges** (documents proches,
  champs similaires), relecture humaine d'un échantillon.
- Un **rapport de recommandation** par technique : gain avec intervalle de confiance,
  coût (latence, tokens, temps d'indexation), verdict — *à faire*, *inutile*, *non
  concluant* — ventilé par type de question et type de corpus.
- Attribution de la faute : dire si un échec vient du retrieval ou du générateur.
- Fonctionnement 100 % local : aucun appel à un service externe, aucune donnée qui sort.

**Hors périmètre**

- Servir un RAG en production. Le pipeline de [src/ragbench/rag/](../src/ragbench/rag/)
  existe pour être mesuré, pas pour être exploité.
- L'optimisation automatique d'hyperparamètres. Le banc recommande, l'humain décide.
- Présenter un aperçu comme un verdict.
- Le multi-tenant, l'authentification, la haute disponibilité.
- L'entraînement ou le fine-tuning de modèles.

### Ce que le mode aperçu ne peut pas dire

Ces limites sont structurelles ; le rapport doit les afficher, pas les taire.

- **Des questions générées sont biaisées.** Écrites à partir d'un passage, elles en
  reprennent le vocabulaire — ce qui avantage la recherche lexicale — et sont plus
  faciles que des questions réelles. Elles ne contiennent aucun piège entre documents
  proches, sauf à les générer exprès.
- **Un petit corpus donne des intervalles larges.** Avec 50 questions, beaucoup de
  verdicts seront *non concluant*. C'est la bonne réponse, pas un échec de l'outil.
- **La recommandation ne vaut que pour les questions testées.** Si les questions réelles
  ne ressemblent pas au jeu de test, la conclusion ne se transfère pas. Un historique de
  vraies questions, même court, vaut mieux que mille questions générées.
- **Sans relecture humaine, les métriques de génération sont des opinions.** Aucun juge
  local n'atteint κ = 0,6 (voir [JUGES.md](JUGES.md)) ; la relecture d'un échantillon est
  ce qui permet de calibrer le juge sur le nouveau corpus.

### Stratégie de campagne

Le nombre de combinaisons explose vite, et une campagne se compte en heures sur CPU.
D'où l'ordre : **chaque technique seule contre la baseline**, puis **seulement les
gagnantes combinées entre elles**. Les comparaisons multiples sont corrigées, sans quoi
une technique sur vingt « gagnerait » par hasard.

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

  **Partiellement infirmée pour les réponses** (pas pour les evidences). Un audit manuel
  de 53 réponses a montré que `contains` compte faux une réponse correcte donnée sous
  forme abrégée — `Bankman-Fried` pour un gold `Sam Bankman-Fried`. Les valeurs de
  `contains` sont donc des **bornes basses** : 0.340 mesuré contre 0.358 réel sur cet
  échantillon. Les comparaisons appariées restent valides, le biais s'annulant dans
  l'écart. Détail dans [JUGES.md](JUGES.md#4-la-référence-elle-même-a-un-plafond).

- **Un échantillon de 200 questions suffit pour départager deux configurations.**
  Vérifié empiriquement : les intervalles de confiance obtenus (± 0,05 sur `recall@3`)
  permettent de conclure sur des écarts de 6 points ou plus. En dessous, il faut
  agrandir l'échantillon — les 2 356 questions restantes sont conservées en base sous
  le split `pool` exactement pour ça.

- **Le corpus est du texte anglais journalistique.** Le découpage, la recherche
  lexicale (`to_tsvector('english', …)`) et les seuils de recouvrement sont réglés pour
  ça. Un corpus français ou du code demanderait de revoir la configuration de recherche
  plein-texte.
  Vrai pour le mode bench ; le mode aperçu devra rendre la configuration plein-texte
  paramétrable par langue.

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
  reproductibles (voir [JUGES.md](JUGES.md)).

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

  **Le plafond n'est pas la tâche, c'est la taille.** Les mêmes 53 réponses jugées à la
  main par Claude Opus 5 donnent **κ = +0.958**. Les questions étaient donc jugeables ;
  ce sont les petits modèles qui ne les jugent pas. L'écart 0.958 / 0.458 chiffre le prix
  de la contrainte on-premise sur ce poste précis. Ça ne fait pas d'un juge externe une
  option : il viole la contrainte, et sert d'étalon mesuré une fois sur corpus public.

### À trancher

- **Faut-il un profil vLLM ?** Le continuous batching donnerait un net gain de débit sur
  une charge d'évaluation. Reco par défaut : attendre qu'une campagne dépasse l'heure
  avant d'ajouter cette complexité. Contrainte connue : vLLM épingle un seul modèle en
  VRAM, sans swap, et exigerait des poids quantifiés sur 16 Go.

- **Réimplémenter ou emprunter ?** [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
  implémente déjà de nombreuses méthodes RAG, et sa réimplémentation commune fournit
  les seules comparaisons équitables publiées (Self-RAG y déçoit, IRCoT et Search-R1
  y gagnent). Reco par défaut : reprendre les **algorithmes** et les briques sous
  licence permissive listées dans [ETAT-DE-L-ART.md](ETAT-DE-L-ART.md#5-briques-récupérables),
  n'écrire soi-même que ce qui doit entrer dans le hash de configuration. La valeur
  propre du banc est ailleurs : vos documents, le 100 % local, la rigueur statistique,
  le français.

- **Le graphe de connaissance est-il testable sur ce poste ?** Sa construction extrait
  les entités de tout le corpus par LLM, ce qui se compte en jours sur CPU. Reco par
  défaut : **HippoRAG 2** plutôt que GraphRAG ou LightRAG — seul graphe à un coût par
  requête proche du RAG (~1 000 tokens contre 100 000 à 331 000), et seul gain net sur
  le raisonnement complexe dans GraphRAG-Bench. Le mesurer d'abord sur un sous-corpus,
  réserver la campagne complète à la machine GPU. Risque : ses gains publiés supposent
  un extracteur de 70B ; ~7B est le minimum pratique. Son coût d'indexation fait partie
  du résultat.

- **Faut-il un vérificateur NLI dédié** pour la fidélité, plutôt qu'un juge génératif ?
  Reco par défaut : **oui**. Les vérificateurs de 0,1 à 0,8 B (HHEM-2.1-Open,
  MiniCheck-Flan-T5-L, FactCG-DeBERTa-L) approchent les juges de 7–8 B sur
  LLM-AggreFact et tournent sur CPU ; LettuceDetect a des variantes EuroBERT qui
  couvrent le français. Les calibrer par κ sur le même jeu que les juges actuels.
  Non implémenté à ce jour.

- **Faut-il encore exiger κ ≥ 0,6 pour utiliser un juge ?** Reco par défaut : non,
  **corriger son biais** plutôt que l'écarter. Avec l'échantillon relu à la main, la
  *prediction-powered inference* (ARES) ou la correction par sensibilité et spécificité
  (arXiv 2511.21140) donnent des intervalles de confiance honnêtes ; ~100 questions
  annotées suffisent. C'est ce qui rend le mode aperçu praticable avec des juges
  locaux.

- **Combien de générations par question ?** Le bruit de génération domine souvent celui
  du choix des questions (arXiv 2512.21326). Reco par défaut : k = 3 générations pour
  les comparaisons serrées, k = 1 pour le criblage.

---

## 7. Roadmap

Les phases 0 à 3 constituent le socle de mesure et sont réalisées. La suite n'est pas
commencée.

0. **Socle** — configuration hashée, schéma Postgres, client LLM, pipeline paramétrable.
1. **Dataset et référence** — chargement MultiHop-RAG, évaluateurs déterministes,
   première campagne chiffrée.
2. **Comparaison** — couche statistique, tableau de bord, tests de non-régression.
3. **Multi-frameworks** — évaluateurs à base de juge, adaptateurs Ragas et DeepEval,
   calibration contre la vérité terrain.
4. **Catalogue de techniques** — chacune derrière un champ de configuration, donc dans
   le hash. Ordre fixé par l'[état de l'art](ETAT-DE-L-ART.md) et nos constats :
   1. *Prérequis, coût nul* : courbe de recall de k = 5 à 50 sur les evidences.
      Si la couverture monte vite, le plafond est le nombre de passages, pas la
      qualité du retrieval.
   2. Décomposition de requête avec retrieval hybride par sous-question.
   3. Reranker cross-encoder sur un top-50 hybride (Qwen3-Reranker-0.6B,
      jina-reranker-v3, bge-reranker-v2-m3, via `rerankers`).
   4. Retrieval itératif type IRCoT.
   5. Filtrage par métadonnées et désambiguïsation.
   6. Embeddings récents (Qwen3-Embedding-0.6B, EmbeddingGemma, BGE-M3).
   7. Graphe de connaissance : HippoRAG 2.
   8. *Priorité basse, gains non répliqués* : semantic chunking, Self-RAG, HyDE sur
      corpus lexical, long contexte.
5. **Évaluation renforcée** — vérificateurs NLI de fidélité, correction du biais des
   juges (PPI), protocole TREC RAG 2025 (*strict vital recall*, support par phrase),
   abstention sur quatre types de contexte (supportif, dégradé, absent, **trompeur**),
   correction de Holm, k générations par question.
6. **Mode aperçu** — ingestion par Docling (MinerU en option pour les PDF difficiles),
   analyse du corpus (langue, quasi-doublons, part de tableaux), génération du jeu de
   test par DeepEval Synthesizer encapsulé : distribution de styles fixée, rejet des
   questions à fort recouvrement lexical, pièges entre documents voisins, questions
   sans réponse. Vérité terrain par les prompts nuggetizer et UMBRELA. Relecture d'un
   échantillon stratifié depuis le tableau de bord. Recherche plein-texte paramétrable
   par langue.
7. **Rapport de recommandation** — verdict par technique, gain et coût, ventilation par
   type de question (taxonomie EnterpriseRAG-Bench) et de corpus, avertissements du
   mode aperçu affichés.
8. **Plus tard** — observabilité (traces OpenTelemetry), profil vLLM, jeu de robustesse
   (bruit injecté, contexte contradictoire), intégration continue.

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

La veille complète, à jour au 28/09/2026 — techniques, évaluation, outillage, briques
récupérables et licences — est dans [ETAT-DE-L-ART.md](ETAT-DE-L-ART.md). Ci-dessous,
les travaux sur lesquels le socle actuel s'appuie :

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

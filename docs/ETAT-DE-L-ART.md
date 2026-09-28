# État de l'art — ragbench

Veille arrêtée au **28/09/2026**. Ce document dit ce qui existe, ce que les mesures
**indépendantes** en disent, et ce que le banc en reprend. Les décisions qui en découlent
sont dans [CADRAGE.md](CADRAGE.md#6-décisions), la roadmap dans
[CADRAGE.md](CADRAGE.md#7-roadmap).

Règle de lecture : un chiffre publié par l'auteur d'une méthode n'est pas une mesure
indépendante. Les deux sont distingués, et ce qui n'a pas été revérifié dans la source
est marqué *(non revérifié)*.

---

## En résumé

| Constat | Conséquence pour le banc |
|---|---|
| Les techniques « sophistiquées » gagnent rarement en réplication indépendante : semantic chunking, Self-RAG, GraphRAG global | Le banc a une raison d'être : ce qui marche dépend du corpus, il faut le mesurer |
| Un reranker faible **dégrade** un premier étage fort | Notre « reranking LLM sans gain » n'est pas anormal ; un vrai cross-encoder reste à tester |
| Les meilleurs gains multi-hop viennent de la **décomposition** et du **retrieval itératif** | Priorités du catalogue |
| Les **quasi-doublons** effondrent le retrieval (66 % → 5–28 %) et le filtrage par métadonnées le relève fortement | Les questions pièges et le filtrage sont centraux, pas accessoires |
| Les **vérificateurs NLI de 0,1–0,8 B** approchent les juges de 7–8 B en fidélité | Remplacer le juge génératif de fidélité, sur CPU |
| Un juge imparfait peut être **corrigé** plutôt qu'écarté (PPI, sensibilité/spécificité) | Le seuil κ ≥ 0,6 n'est plus une condition d'usage |
| Les **questions synthétiques** sont plus longues et plus faciles que les vraies ; la config gagnante en synthétique baisse sur requêtes réelles | Le mode aperçu doit débiaiser ses jeux de test, et le dire |
| **Aucun benchmark RAG texte en français sur documents d'entreprise** | Place à prendre |

---

## 1. Techniques d'amélioration

### 1.1 Retrieval

- **Rerankers cross-encoder.** Fiche Qwen3-Reranker (juin 2025, mesure du fournisseur),
  top-100 de Qwen3-Embedding-0.6B en MTEB-R : retrieval seul 61.8, **bge-reranker-v2-m3
  57.0** (dégrade), Qwen3-Reranker-0.6B 65.8, 4B 69.8. jina-reranker-v3 (0.6B) annonce
  61.94 sur BEIR contre 61.44 pour mxbai-rerank-large-v2, tous deux auto-évalués.
  Pas de support Ollama natif : passer par `transformers`, ONNX ou llama.cpp.
- **Embeddings.** Qwen3-Embedding (0.6B / 4B / 8B, 8B n°1 MTEB multilingue en juin 2025),
  EmbeddingGemma-300M (sept. 2025), BGE-M3 (dense + sparse + multi-vecteur, référence
  multilingue). Pas de mesure indépendante vérifiée sur le français.
- **Late interaction.** GTE-ModernColBERT 54.2 BEIR contre 53.0 pour jina-colbert-v2
  (PyLate, 2025). Faisable sur CPU, index lourd.
- **HyDE et expansion de requête.** Gain quand vocabulaire question et documents
  divergent, régression quand ils sont alignés (arXiv 2507.16754). MultiHop-RAG est
  lexical (notre lexical > dense le confirme) : gain attendu faible.
- **Décomposition de requête** sur MultiHop-RAG (arXiv 2507.00355, juil. 2025) :
  +36.7 % MRR@10, +11.6 % F1 contre RAG standard.

### 1.2 Découpage

- **Semantic chunking** : gains incohérents, ne justifient pas le coût — *Is Semantic
  Chunking Worth the Computational Cost?*, NAACL Findings 2025 (arXiv 2410.13070).
- **Étude à grande échelle** (Caspari et al., CIKM'26, arXiv 2608.16586) : huit méthodes
  jusqu'à 10 M de documents ; les méthodes complexes apportent rarement des gains
  consistants, le découpage fixe par tokens reste un bon défaut. Le contextuel indexe
  0.26 à 5.26 documents par seconde.
- **Taxonomie** (arXiv 2602.16974, fév. 2026) : le découpage structurel (paragraphes) est
  le meilleur ; les propositions seules sont nettement en dessous ; la contextualisation
  les relève de +23 à +27 % mais dégrade la recherche intra-document (−4 à −53 %).
- **Late chunking** : +1.5 à +1.9 point BEIR (mesure de l'auteur, Jina). Exige les
  embeddings par token, qu'Ollama n'expose pas.
- **En-tête contextuel** : notre +0.10 est cohérent avec Anthropic (−35 % d'échecs en
  embeddings, −49 % avec BM25 contextuel, sept. 2024, mesure interne).

### 1.3 Graphes de connaissance

GraphRAG-Bench (ICLR 2026, arXiv 2506.05690), accuracy romans / médical :

| Tâche | RAG + rerank | HippoRAG 2 | MS-GraphRAG | LightRAG |
|---|---|---|---|---|
| Faits simples | 60.9 / 64.7 | 60.1 / 66.3 | 49.3 / 38.6 | — |
| Raisonnement complexe | 42.9 / 58.6 | 53.4 / 62.0 | — | 49.1 / 61.3 |

Tokens par requête : ~954 (RAG), ~1 020 (HippoRAG 2), ~100 000 (LightRAG), ~331 000
(GraphRAG global).

- *RAG vs. GraphRAG* (Han et al., arXiv 2502.11371) sur MultiHop-RAG : GraphRAG local
  69.0 % contre 67.0 % ; seules 65.8 % des entités-réponses de HotpotQA sont dans le
  graphe construit ; les verdicts de juges LLM s'inversent selon l'ordre de présentation.
- HippoRAG 2 : gains nets en multi-hop, mais mesurés avec un extracteur Llama-3.3-70B.
- En local, ~7B est le minimum pratique pour l'extraction (arXiv 2605.20815).

**Lecture** : le graphe ne paie que sur le raisonnement complexe, et HippoRAG 2 est le
seul à un coût proche du RAG. Le risque est l'extraction par un modèle ≤ 8B.

### 1.4 RAG correctif et agentique

Réimplémentation commune FlashRAG (Llama3-8B, e5, 5 documents), exact match :

| Méthode | HotpotQA | 2Wiki | TriviaQA |
|---|---|---|---|
| RAG standard | 35.3 | 21.0 | 58.9 |
| Self-RAG | 29.6 | 25.1 | 38.2 |
| Adaptive-RAG | 39.1 | 28.4 | — |
| IRCoT (itératif) | 41.5 | 32.4 | — |
| Search-R1 (RL) | 54.5 | 42.6 | — |

Self-RAG déçoit en réplication. CRAG : +7 à +36.6 % annoncés par ses auteurs, aucune
réplication indépendante trouvée. Search-R1 publie des checkpoints 3B et 7B.

### 1.5 Documents difficiles

- **Quasi-doublons** : RARE / RedQA (ACL 2026, arXiv 2604.19047) — un bon retriever passe
  de 66.4 % de PerfRecall@10 (4 sauts, Wikipédia) à **5–28 %** sur corpus redondants
  (finance, droit, brevets). *Vector search dilution* (arXiv 2606.11350) : accuracy de
  75 % à moins de 40 % quand le corpus passe de 54 à 1 128 documents.
- **Filtrage par métadonnées** : AMAQA (2025) — de 0.27 à 0.76 d'accuracy pour des LLM
  ouverts.
- **Tableaux** : T²-RAGBench (EACL 2026) — l'hybride BM25 + dense est le meilleur.
- **PDF visuels** : ViDoRe V3 (arXiv 2601.08620, janv. 2026, 6 langues dont le français) —
  les retrievers visuels battent le texte (59.8 contre 51.0 nDCG@10). ColQwen2.5-3B tient
  sur 16 Go.
- **OCR ≠ RAG** : InduOCRBench (ACL 2026 Industry, arXiv 2605.00911) — 82.9 % de précision
  OCR ne donnent que 53.0 % de précision RAG. Un parseur s'évalue par son effet final.

### 1.6 Long contexte contre RAG

Pas de gagnant universel (LaRA, ICML 2025). Ajouter des passages fait monter puis baisser
la qualité à cause des passages proches mais faux (ICLR 2025, arXiv 2410.05983).
Hors de portée sur CPU avec des modèles ≤ 8B.

---

## 2. Évaluation

### 2.1 Protocoles

- **TREC RAG 2025** (arXiv 2603.09891, mars 2026) : requêtes longues découpées en
  sous-narratifs, *strict vital recall*, taux de support par phrase. Référence à suivre
  après AutoNuggetizer 2024.
- **The Great Nugget Recall** (SIGIR 2025) : les classements de systèmes par nuggets
  automatiques sont fortement corrélés aux classements manuels — au niveau du **système**,
  pas de la réponse individuelle.
- **Accord sur la citation** (arXiv 2504.15205, SIGIR 2025) : GPT-4o et un humain
  concordent parfaitement dans 56 % des cas, 72 % quand l'humain corrige le LLM. Donne un
  plafond réaliste pour situer nos κ.
- **Abstention** : UAEval4RAG (ACL 2025) — six catégories de questions sans réponse.
  GRAB-RAG (arXiv 2608.22228, auteur unique, non relu) — des modèles de 3.8 à 8B
  s'abstiennent bien sans contexte, mais répondent encore à 41.6 % des questions à
  **contexte trompeur**.

### 2.2 Vérificateurs de fidélité et juges

| Modèle | Taille, licence | Mesure publiée | Local |
|---|---|---|---|
| HHEM-2.1-Open | 0.1B, Apache-2.0 | 76.55 % AggreFact-SOTA | CPU, ~1.5 s / 2k tokens ; anglais |
| MiniCheck-Flan-T5-L | 0.8B | 75.0 % LLM-AggreFact | CPU ; licence à vérifier |
| FactCG-DeBERTa-L | 0.4B | 75.6 % LLM-AggreFact | CPU ; licence à vérifier |
| LettuceDetect | 17M–610M, MIT | F1 0.53–0.57 (sous-ensemble RAGTruth) | CPU ; **variantes EuroBERT avec le français** |
| Granite Guardian 4.1 | 8B, Apache-2.0 | BAcc 0.760 | Ollama ; anglais seulement |
| Selene-1-Mini | 8B, Apache-2.0 | Au-dessus de Prometheus 2 et Flow-Judge | GGUF ; français |
| Bespoke-MiniCheck-7B, Lynx-8B | 7–8B, **CC BY-NC** | 77.4 % / 82.9 % | Usage non commercial |

**Fiabilité des juges** : même Gemini-2.5-Pro et GPT-4 sont incohérents sur ~25 % des cas
difficiles (Sage, arXiv 2512.16041). Les juges spécialisés, les panels et les grilles
explicites améliorent la cohérence.

### 2.3 Statistique

- Sous quelques centaines d'items, les intervalles fondés sur le théorème central limite
  sous-estiment l'incertitude (Bowyer et al., ICML 2025, arXiv 2503.01747). Le banc
  utilise déjà un bootstrap apparié et un test de permutation : conforme.
- Le **bruit de génération** domine souvent le bruit dû au choix des questions : moyenner
  k générations par question aide davantage qu'ajouter des questions (arXiv 2512.21326).
- **Corriger un juge imparfait** par sa sensibilité et sa spécificité mesurées
  (arXiv 2511.21140, ICML 2026), ou par *prediction-powered inference* : ~100 requêtes
  annotées suffisent (arXiv 2601.18777). ARES en fournit une implémentation.
- **Comparaisons multiples** : Holm ou Benjamini-Hochberg restent la pratique ; aucune
  publication propre aux LLM.

### 2.4 Frameworks

| Framework | État au 28/09/2026 | Pour le banc |
|---|---|---|
| Ragas | 0.4.3 (13/01/2026), dépôt migré vers `vibrantlabsai/ragas`, activité en baisse | Garder l'adaptateur figé en 0.4.x |
| DeepEval | 4.2.x, Apache-2.0, très actif | Montée depuis 4.1 à tester |
| Opik | Apache-2.0, auto-hébergeable | Option d'observabilité |
| Phoenix | **Elastic License 2.0**, non OSI | Écarté |
| promptfoo | Racheté par OpenAI (mars 2026) | Risque de dépendance |

---

## 3. Outillage du mode aperçu

### 3.1 Génération de jeux de test et biais

- **Biais mesurés** : requêtes synthétiques de 15.7 mots contre 6.8 pour les vraies ; la
  configuration gagnante en synthétique baisse sur requêtes réelles et coûte jusqu'à 8×
  plus de latence (Kucia & Gawlik, arXiv 2609.14579, sept. 2026). Les retrievers
  neuronaux favorisent le texte généré par LLM (arXiv 2310.20501).
- **Générateurs** : DeepEval Synthesizer (Apache-2.0, évolutions multi-contexte et
  comparatives, très actif) ; Ragas (graphe de connaissances, personas, activité en
  baisse) ; YourBench (Apache-2.0, contrôle par citation vérifiée) ; DataMorgana
  (distribution configurable de profils et de styles, non open-source *(à confirmer)*) ;
  générateurs Giskard `distracting`, `double`, `oos`, `situational` (tag `v2.19.2`,
  supprimés de la v3).
- **Parades retenues** : distribution de styles fixée à l'avance ; rejet des questions à
  fort recouvrement lexical avec leur passage ; questions multi-documents ; pièges entre
  documents voisins (paires MinHash ou embeddings, réponse différente selon le document) ;
  questions sans réponse ; relecture humaine d'un échantillon stratifié.
- **Vérité terrain sans evidence_list** : les prompts `nuggetizer` (création de nuggets)
  et `UMBRELA` (pertinence de passage, échelle TREC) de castorini, Apache-2.0, validés en
  TREC.

### 3.2 Parseurs

| Parseur | Licence | CPU | Remarque |
|---|---|---|---|
| **Docling** | MIT | Oui | docx, pptx, xlsx, HTML natifs ; tableaux via TableFormer |
| MinerU | Apache-2.0 + clauses (attribution, seuils commerciaux) | Pipeline oui | Meilleur sur PDF difficiles, surtout sur GPU |
| Marker | Code Apache-2.0, **poids OpenRAIL-M modifiée** | Oui, sans OCR | Chiffres publiés par l'éditeur |
| olmOCR | Apache-2.0 | **Non** (VLM 7B) | — |
| Unstructured | Apache-2.0 | Oui | Pas de mesure indépendante récente |

Aucun comparatif n'est neutre, et les scores OCR prédisent mal le résultat RAG
(InduOCRBench) : le parseur est une brique à évaluer comme les autres.

### 3.3 Benchmarks utiles

| Benchmark | Intérêt | Licence |
|---|---|---|
| EnterpriseRAG-Bench (mai 2026) | 500 questions en 10 catégories, dont *Conflicting Info* et *Info Not Found* ; quasi-doublons volontaires | MIT, anglais |
| RARE / RedQA (ACL 2026) | Corpus redondants, multi-sauts | À vérifier |
| ViDoRe V3 | PDF visuels, deux corpus français | À vérifier |
| Open RAG Bench (Vectara) | 1 000 PDF arXiv, 600 négatifs difficiles | Aucune licence : lire, ne pas copier |
| CRAG (Meta) | 4 409 questions, 5 domaines | CC BY-NC 4.0 |

---

## 4. Boîtes à outils comparables et positionnement

| Outil | Nature | État |
|---|---|---|
| FlashRAG | Boîte à outils de recherche, dizaines de méthodes, MIT | Actif |
| UltraRAG | Framework low-code sur MCP, Apache-2.0 | Très actif |
| AutoRAG | Pivot vers un « agent bibliothécaire » ; l'optimiseur de pipeline est en maintenance | — |
| RAGChecker, RAGLAB | Métriques par claims / reproduction d'algorithmes | Inactifs depuis 2024 |
| BERGEN (NAVER) | Bench multilingue | **CC BY-NC-SA** |
| RAGFlow | Moteur RAG de production | Pas un outil de mesure |

**Positionnement** : aucun ne réunit le 100 % local sur modèles ≤ 8B, les documents de
l'utilisateur avec une vérité terrain générée **puis relue**, des pièges ciblés entre
quasi-doublons, un verdict gain/coût technique par technique, et le français. C'est la
place du banc — pas « un FlashRAG de plus ».

---

## 5. Briques récupérables

Licences vérifiées par l'API GitHub le 28/09/2026.

| Dépôt | Licence | Ce qu'on reprend | Phase |
|---|---|---|---|
| AnswerDotAI/rerankers | Apache-2.0 | API unique pour plusieurs rerankers (cross-encoder, FlashRank, ColBERT) | 4 |
| PrithivirajDamodaran/FlashRank | Apache-2.0 | Rerankers ONNX sur CPU | 4 |
| FlagOpen/FlagEmbedding | MIT | Poids bge-reranker | 4 |
| castorini/nuggetizer, castorini/umbrela | Apache-2.0 | Prompts de création de nuggets et de jugement de pertinence | 5 |
| vectara/open-rag-eval | Apache-2.0 | Assemblage UMBRELA + nuggets, métriques citation et non-réponse | 5 |
| stanford-futuredata/ARES | Apache-2.0 | Formule des intervalles PPI (`ares/rag_scoring.py`) | 5, 6 |
| KRLabsOrg/LettuceDetect | MIT | Détecteur d'hallucination par segment, CPU | Évaluation |
| Giskard-AI/giskard-oss @ v2.19.2 | Apache-2.0 | Générateurs de questions `distracting`, `double`, `oos` | 5 |
| onyx-dot-app/EnterpriseRAG-Bench | MIT | Taxonomie de 10 catégories de questions | 5, 6 |
| brandonstarxel/chunking_evaluation | MIT | Évaluation d'un découpage au token (IoU, précision, rappel) | 4 |
| AmenRa/ranx | MIT | Fusions autres que RRF (`ranx/fusion/`) ; ses tests statistiques doublonnent les nôtres | 4 |
| HKUDS/LightRAG | MIT | Prompts d'extraction d'entités, implémentation Postgres | 4 |

**À ne pas copier** : NirDiamant/RAG_Techniques (licence non commerciale),
HuskyInSalt/CRAG, texttron/hyde, langchain-ai/rag-from-scratch, le code MultiHop-RAG
(sans licence). Archivés : IntelLabs/fastRAG, microsoft/PIKE-RAG. Figés depuis 2024 :
RAPTOR, Self-RAG, late-chunking — reprendre l'algorithme, pas le code.

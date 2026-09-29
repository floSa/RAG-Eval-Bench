# Méthodologie d'évaluation

Ce document décrit **ce que le banc mesure et pourquoi on peut le croire**. Le cadrage
est dans [CADRAGE.md](CADRAGE.md), la mise en œuvre dans [ARCHITECTURE.md](ARCHITECTURE.md),
les résultats sur les juges dans [JUGES.md](JUGES.md).

**Si le vocabulaire ne vous est pas familier, commencez par le
[GLOSSAIRE.md](GLOSSAIRE.md)** — chaque terme y est expliqué en français simple, sans
formule.

Tous les chiffres cités proviennent de campagnes réelles sur MultiHop-RAG, échantillon
stratifié de **200 questions**, générateur `gemma4:e4b`, embedder `nomic-embed-text`.

---

## Les résultats en un coup d'œil

Ce que le banc a établi, avec le test qui le soutient. Le détail suit.

| Question posée | Réponse mesurée | Verdict |
|---|---|---|
| Recherche dense ou hybride ? | **Hybride**, +0.098 de `recall@3` | p < 0.001 |
| La recherche par mots exacts sert-elle encore ? | **Oui, elle bat le dense** à elle seule, +0.062 | p = 0.010 |
| Le reranking par LLM vaut-il son coût ? | **Non**, gain nul pour 5× le prix | p = 0.959 |
| Faut-il rappeler la source dans chaque passage ? | **Oui**, +0.102 de justesse | p < 0.001 |
| Quel prompt ? | **`decompose`**, +0.096, sans sacrifier l'abstention | p = 0.001 |
| Config par défaut contre config optimisée | **La justesse double**, 0.169 → 0.339 | p < 0.001 |
| Le RAG bat-il le modèle seul, sans documents ? | **On ne peut pas conclure** | p = 0.425 |
| Qu'est-ce qui limite vraiment le système ? | **La recherche** : dans 96 % des cas, tous les faits nécessaires ne sont pas remontés | `nugget_full_coverage` = 0.0395 |
| Un reranker cross-encoder aide-t-il ? *(GPU, 29/09)* | **Selon le modèle** : Qwen3-Reranker-0.6B +0.192 d'articles à k = 5, MiniLM −0.063 | p Holm < 0.001 / 0.026 |
| Réécrire, décomposer la requête, HyDE ? *(GPU, 29/09)* | **Non** : décomposition −0.071, HyDE −0.046, réécriture −0.040 | p Holm 0.003 / 0.040 / 0.107 |
| Meilleur générateur de moins de 4 Go ? *(GPU, 29/09)* | **`qwen3:4b-instruct`**, justesse 0.503 contre 0.294 pour `qwen3.5:4b` | p < 0.001 |
| Peut-on faire confiance à un juge local de 3–4 B ? | **Non**, aucun n'atteint le seuil publié | κ max 0.458 pour un seuil de 0,6 |

Le dernier point est le plus contre-intuitif et il est traité à part, dans
[JUGES.md](JUGES.md).

---

## 1. Les trois étages

Une note globale ne se corrige pas. Le banc mesure donc les étages séparément, pour
pouvoir désigner le responsable d'un échec.

| Étage | Question posée | Évaluateurs |
|---|---|---|
| Retrieval | A-t-on remonté les bons passages ? | `native.ir`, `native.nuggets`, `native.erag` |
| Génération | La réponse est-elle fidèle aux passages remontés ? | `native.claims`, `ragas`, `deepeval` |
| Bout en bout | La réponse répond-elle à la question ? | `native.answer` |
| Coût | À quel prix ? | Consommation ventilée par rôle de modèle, dans `runs.usage` |

Second axe, orthogonal : **reference-based** (il faut une vérité terrain — coûteux,
fiable) contre **reference-free** (un juge note sans vérité terrain — bon marché,
dérive). Les évaluateurs `native.*` sont reference-based ou déterministes ; `ragas` et
`deepeval` sont majoritairement reference-free.

---

## 2. Le catalogue d'évaluateurs

| Évaluateur | Apport propre | Déterministe | Coût par question |
|---|---|---|---|
| `native.ir` | recall@k, precision@k, hit_rate@k, nDCG@k, MRR au niveau document | ✅ | nul |
| `native.answer` | exact match, containment, F1 par tokens, *negative rejection*, accuracy ventilée par type | ✅ | nul |
| `native.nuggets` | couverture des faits de référence par les passages remontés (mode lexical) | ✅ | nul |
| `native.erag` | utilité réelle de chaque passage pris isolément | ❌ | `top_k` appels |
| `native.claims` | attribution de l'erreur : générateur ou retrieval ? | ❌ | plusieurs appels |
| `ragas` | les quatre métriques canoniques, avec taux de couverture | ❌ | ~32 s par métrique |
| `deepeval` | seuils pass/fail, donc non-régression en intégration continue | ❌ | ~34 s pour 4 métriques |

Trois précautions de calcul valent d'être connues, parce qu'elles changent les chiffres :

- **Les questions sans réponse sont exclues des métriques de retrieval.** Leur recall
  est *indéfini*, pas nul. Les inclure ferait chuter artificiellement toutes les
  moyennes. Elles sont évaluées à part, sur l'abstention.
- **L'IDCG du nDCG est borné par k.** Le classement parfait *réalisable* contient
  `min(|gold|, k)` documents pertinents. Sans cette borne, une configuration à petit
  `top_k` serait pénalisée pour son `top_k` et non pour son classement.
- **La granularité est le document.** Les chunks sont dédupliqués par document avant
  calcul, en conservant le meilleur rang, parce que la vérité terrain est
  document-level.

---

## 3. La discipline statistique

Sur 200 questions, deux points d'écart sont du bruit. Trois outils, chacun pour une
question distincte, dans [stats.py](../src/ragbench/stats.py) :

| Outil | Question | Méthode |
|---|---|---|
| `bootstrap_ci` | Quelle est la précision de cette mesure ? | Percentiles du bootstrap, 5 000 tirages |
| `paired_bootstrap` | B est-il meilleur que A ? | Bootstrap sur la moyenne des **écarts** + test de permutation |
| `mcnemar` | Idem, sur une métrique binaire | Test binomial exact bilatéral |
| `cohen_kappa` | Le juge dit-il la même chose que la référence ? | Accord corrigé du hasard |

**L'appariement est le point central.** Deux configurations voient les *mêmes* questions.
La variance entre questions — énorme, certaines sont dures pour tout le monde —
s'annule dans l'écart. Comparer deux intervalles de confiance séparés perdrait cette
information : deux intervalles qui se chevauchent peuvent parfaitement recouvrir une
différence systématique.

**Le bootstrap plutôt qu'un test de Student**, parce que les scores d'évaluation sont
bornés à [0, 1] et souvent bimodaux (0 ou 1 sur une métrique binaire) : un intervalle
normal déborderait hors de [0, 1] et serait mal calibré.

Deux garde-fous exposés par l'outil lui-même :

- **La zone « limite ».** L'intervalle vient d'un bootstrap, la p-value d'un test de
  permutation. Ces deux procédures peuvent diverger juste au seuil. Quand c'est le cas,
  `ragbench compare` affiche **limite** et non **oui** — c'est précisément dans cette
  zone qu'on tranche à tort avec assurance.
- **Le compte de faux positifs.** Enchaîner 19 comparaisons à 5 % en produit environ une
  fausse par hasard. Le nombre attendu est affiché sous chaque tableau, plutôt que
  corrigé en silence : une correction de Bonferroni sur des métriques fortement
  corrélées entre elles serait trop conservatrice.

---

## 4. Résultats de référence

Campagne sur 200 questions. Les écarts sont donnés par test apparié.

### Retrieval

| Configuration | recall@3 | hit_rate@3 | MRR | nugget_recall | contains |
|---|---|---|---|---|---|
| `closed-book` (témoin sans retrieval) | — | — | 0.000 | — | 0.192 |
| `dense-k5` (référence) | 0.351 | 0.644 | 0.545 | 0.244 | 0.158 |
| `no-doc-cap` | 0.327 | 0.627 | 0.537 | — | 0.158 |
| `no-header` | 0.369 | 0.667 | 0.559 | — | 0.056 |
| `lexical-k5` | 0.413 | 0.729 | 0.619 | 0.328 | 0.203 |
| **`hybrid-k5`** | **0.449** | **0.768** | **0.676** | **0.340** | 0.203 |
| `dense-k5-rerank` | 0.448 | 0.763 | 0.656 | 0.303 | 0.181 |

Trois conclusions établies :

- **L'hybride bat le dense** : recall@3 **+0.098**, IC95 [+0.060, …], p < 0.001.
- **Le lexical seul bat le dense** : **+0.062**, IC95 [+0.017, …], p = 0.010. Contre-
  intuitif, mais cohérent avec le corpus : les questions citent des organes de presse et
  des dates, c'est de l'appariement exact, pas de la similarité sémantique.
- **Le reranking LLM n'apporte rien** : −0.001, p = 0.959 — pour **5× le coût**
  (3,8 s/question contre 0,7).

Et un non-résultat qui compte : `dense-k5` n'est **pas démontrablement meilleur** que le
témoin sans retrieval (+0.034, IC95 [−0.034, +0.107], p = 0.425). La formulation honnête
n'est pas « le RAG est moins bon », c'est « on ne peut pas conclure ».

### Génération

| Configuration | contains | exact match | fausses abstentions | rejet correct |
|---|---|---|---|---|
| `abstain-reference` | 0.169 | 0.119 | 0.729 | 1.000 |
| `abstain-off` | 0.243 | 0.040 | 0.644 | **0.783** |
| `cited` | 0.203 | 0.000 | 0.684 | 1.000 |
| **`decompose`** | **0.266** | **0.209** | **0.610** | **1.000** |

- **`decompose` est le meilleur réglage** : +0.096 de justesse (p = 0.001), et il est
  **indiscernable de `abstain-off`** (p = 0.541) tout en conservant un rejet correct de
  1.000. Il obtient donc le gain de `abstain-off` sans en payer le prix.
- **`abstain-off` achète 7 points de justesse contre 22 points de rejet correct.** Le
  banc publie systématiquement les deux : un rejet correct de 1.000 obtenu en refusant
  *tout* ne vaut rien.
- **`cited` illustre un piège de métrique** : `contains` monte, mais `em` tombe à 0.000
  (p < 0.001). Les marqueurs `[2]` polluent le texte — c'est un artefact du prompt, pas
  une baisse de qualité. Lire une seule métrique aurait conduit à la conclusion inverse.

### Le diagnostic de fond

`nugget_full_coverage = 0.0395`. Sur **4 % des questions seulement**, *tous* les faits
nécessaires sont présents dans les passages remontés.

Le système s'abstient sur 73 % des questions répondables — non par excès de prudence,
mais parce qu'il n'a réellement pas de quoi répondre. Quand il répond, il a raison **62 %
du temps** (0.169 / 0.271).

Conséquence sur les priorités : régler le prompt ne servira à rien tant que la couverture
factuelle est à 4 %. C'est exactement le genre d'arbitrage qu'une note globale rend
invisible.

### Campagne de mesure sur poste GPU (29/09/2026)

Sauf mention contraire, les mesures de cette section viennent du poste fixe GPU
(RTX 4060 Ti 16 Go, 64 Go de RAM, Ryzen 5 9600X, Ollama central en CUDA,
`LLM_CONCURRENCY=4`), sur l'échantillon standard de 200 questions (177 répondables).
Les écarts sont des tests appariés (bootstrap), corrigés par Holm au sein de chaque
comparaison.

#### 1. Reproductibilité : l'écart suit le serveur d'inférence

Le run 25 (`baseline.yml`, hash `bb2b20210128`) a tourné en 3 min 40 s sur le poste
GPU (1,1 s par question, contre environ 2 h sur le portable CPU) :

| Métrique | Documenté (GPU, juillet) | Portable CPU (28/09) | GPU, run 25 (29/09) |
|---|---|---|---|
| recall@3 | 0.351 | 0.525 | 0.351 |
| hit_rate@3 | 0.644 | 0.842 | 0.644 |
| MRR | 0.545 | 0.705 | 0.545 |
| nugget_recall | 0.244 | 0.368 | 0.244 |
| nugget_full_coverage | 0.0395 | 0.113 | 0.0395 |
| contains | 0.158 | — | 0.164 |

Le poste GPU retrouve les chiffres de juillet à l'identique : écart nul sur toutes les
métriques de recherche, p = 1.0. Le code d'indexation et de découpage n'a pas changé
depuis juillet. Le portable fait donc tourner le même code, mais sur un autre serveur
d'inférence (Ollama 0.32 natif en CPU, contre Ollama central en CUDA). **C'est là que
se situe l'écart**, et probablement dans les embeddings.

Ce que la mesure ne dit pas : **laquelle des deux versions est la bonne**. Le portable
remonte *mieux* (+0,17 de recall@3), ce qui exclut l'idée d'une simple dégradation due
au CPU. Deux pistes restent à départager : l'index est-il réutilisé tel qu'il a été
construit en juillet, ou reconstruit ? Les embeddings de nomic-embed-text changent-ils
d'une version d'Ollama à l'autre ? Test décisif : sur le poste GPU, embarquer les
mêmes passages avec les deux versions d'Ollama et comparer les cosinus, puis
reconstruire l'index. **La règle reste la même : ne comparer que des runs issus du
même serveur.** Toutes les comparaisons ci-dessous la respectent.

#### 2. Courbe de couverture (`diagnose recall-curve`, config `baseline`)

| Passages (k) | Articles de référence | ≥ 1 bon article | Faits couverts | Tous les faits |
|---|---|---|---|---|
| 5 | 0.400 [0.354, 0.448] | 0.695 [0.627, 0.763] | 0.250 [0.207, 0.292] | 0.045 [0.017, 0.079] |
| 10 | 0.508 [0.461, 0.556] | 0.808 [0.746, 0.864] | 0.317 [0.271, 0.362] | 0.079 [0.040, 0.119] |
| 20 | 0.649 [0.600, 0.698] | 0.864 [0.814, 0.910] | 0.420 [0.371, 0.466] | 0.141 [0.090, 0.192] |
| 30 | 0.719 [0.672, 0.765] | 0.910 [0.864, 0.949] | 0.456 [0.406, 0.504] | 0.158 [0.107, 0.215] |
| 50 | **0.820** [0.778, 0.860] | **0.949** [0.915, 0.977] | **0.530** [0.480, 0.577] | **0.209** [0.153, 0.266] |

Même conclusion que sur le portable, avec des niveaux plus bas. La couverture double
entre 5 et 50 passages, preuve que les bons articles sont dans le vivier mais classés
trop bas : le classement est le premier levier. En revanche, même à 50 passages, les
faits plafonnent à 53 % : une partie du déficit ne tient pas au classement.

#### 3. Techniques côté requête (`configs/experiments/query.yml`)

Chaque technique est comparée à la référence `hybrid-k5`. C'est `qwen3.5:4b` qui
transforme la requête.

| Variante | doc_recall@5 | Écart [IC95] | p Holm | Latence médiane | Replis | Verdict |
|---|---|---|---|---|---|---|
| `hybrid-rewrite` | 0.473 vs 0.512 | −0.040 [−0.074, −0.006] | 0.107 | 2,2 s (vs 0,6) | 0 | à la limite |
| `hybrid-decompose` | 0.441 vs 0.512 | **−0.071** [−0.111, −0.032] | **0.003** | 3,1 s | 0 | **moins bon** |
| `hybrid-hyde` | 0.466 vs 0.512 | **−0.046** [−0.081, −0.010] | **0.040** | 7,7 s | 0 | **moins bon** |
| `hybrid-clarify` | 0.515 vs 0.512 | +0.003 [+0.000, +0.008] | 1.000 | 1,1 s | question posée 6/177 (3 %) | non concluant |

- Aucune transformation de requête n'améliore l'hybride sur ce corpus : décomposition
  et HyDE le dégradent, la réécriture tend dans le même sens. Hypothèses non vérifiées :
  - les questions de MultiHop-RAG portent déjà les entités, les sources et les dates
    exactes, sur lesquelles s'appuie le lexical ;
  - les sous-requêtes et le passage hypothétique les diluent.
- Ce résultat contredit le gain publié pour la décomposition sur ce corpus. Deux
  différences d'implémentation peuvent l'expliquer : la fusion RRF des sous-requêtes
  avec la question d'origine, et un LLM de 4 B.
- La clarification ne se déclenche presque jamais : les questions de ce corpus sont
  explicites. Le résultat ne dit rien des corpus où les questions sont ambiguës.

#### 4. Rerankers (`configs/experiments/rerank.yml`)

**Qwen3-Reranker-0.6B (poste GPU, fp16 CUDA)** : `hybrid-k5` (vivier 20) contre
`hybrid-k5-qwen3-reranker` (vivier 50 reclassé).

| Métrique | k | Hybride seul | + Qwen3-Reranker | Écart [IC95] | p Holm | Verdict |
|---|---|---|---|---|---|---|
| `doc_recall` | 5 | 0.512 | **0.704** | **+0.192** [+0.148, +0.239] | **0.0008** | **meilleur** |
| `doc_recall` | 10 | 0.627 | **0.793** | **+0.166** [+0.129, +0.205] | **0.0008** | **meilleur** |
| `nugget_full_coverage` | 5 | 0.113 | **0.271** | **+0.158** [+0.102, +0.220] | **0.0008** | **meilleur** |
| `nugget_full_coverage` | 10 | 0.141 | **0.311** | **+0.169** [+0.113, +0.232] | **0.0008** | **meilleur** |

C'est le gain le plus net mesuré par le banc : +19 points d'articles à k = 5, et une
couverture complète des faits multipliée par 2,4. Il coûte 4,4 s par question
(médiane) sur GPU pour 50 passages, contre environ 140 s sur le portable CPU.

- **Le gain vient-il du vivier plus large ?** La comparaison change deux choses à la
  fois : le reranker, et le vivier (20 → 50). Mais élargir le vivier seul ne change
  rien (ablation ci-dessous, +0.010, non concluant). Le gain est donc attribuable au
  reclassement.
- **Le classement reste le goulot.** 30 % des articles de référence manquent encore
  au top-5, et 73 % des questions n'ont pas tous leurs faits.

**Cross-encoder MiniLM (portable CPU, 28/09, non refait sur GPU)** :
`ms-marco-MiniLM-L-12-v2` sur 50 candidats. Ses niveaux absolus ne se comparent pas au
tableau précédent (autre serveur, voir § 1).

| Premier étage | Métrique @5 | Sans rerank | Avec | Écart [IC95] | p Holm | Verdict |
|---|---|---|---|---|---|---|
| dense | articles | 0.567 | 0.552 | −0.015 [−0.057, +0.027] | 1.00 | non concluant |
| dense | tous les faits | 0.113 | 0.079 | −0.034 [−0.085, +0.017] | 1.00 | non concluant |
| hybride | articles | 0.619 | 0.556 | **−0.063** [−0.111, −0.016] | 0.026 | **moins bon** |
| hybride | tous les faits | 0.164 | 0.056 | **−0.107** [−0.164, −0.051] | 0.002 | **moins bon** |

**Leçon** : « ajouter un reranker » n'a pas d'effet en soi, tout dépend du modèle.
MiniLM, entraîné sur les passages courts de MS MARCO, dégrade l'hybride. Qwen3-Reranker,
un modèle récent de taille comparable, le fait progresser nettement. Pour que les deux
chiffres se comparent, il faut encore refaire MiniLM sur le poste GPU.

#### 5. Générateurs légers (`configs/experiments/generators.yml`)

Six générateurs de moins de 4 Go, sur le pipeline hybride recommandé (prompt `default`,
température 0, 200 questions, sans reranker) :

| Modèle | Run | Justesse (`contains`) | Rejet correct (23 q.) | Fausses abstentions | s/question | Tokens produits |
|---|---|---|---|---|---|---|
| **`qwen3:4b-instruct`** | 28 | **0.503** [0.429, 0.576] | 1.000 | **0.429** | 1.0 | 11 113 |
| `phi4-mini:3.8b` | 29 | 0.429 [0.356, 0.503] | 0.652 | 0.525 | 0.6 | 3 361 |
| `qwen3.5:2b` | 27 | 0.362 [0.294, 0.429] | 0.783 | 0.446 | 1.1 | 12 476 |
| `ministral-3:3b` | 31 | 0.333 [0.266, 0.401] | 0.957 | 0.576 | 0.7 | 8 618 |
| `qwen3.5:4b` | 26 | 0.294 [0.226, 0.362] | 1.000 | 0.689 | 0.9 | 2 612 |
| `granite4.1:3b` | 30 | 0.226 [0.169, 0.288] | 1.000 | 0.768 | 0.4 | 999 |

- **`qwen3:4b-instruct` passe en tête.** Il gagne +0.209 de justesse sur `qwen3.5:4b`
  (écart apparié, p < 0.001) et rejette correctement toutes les questions sans réponse.
  C'est le premier générateur du banc à dépasser 50 % de justesse. Mais il s'abstient
  encore à tort sur 43 % des questions répondables. Or, avec 5 passages, tous les faits
  ne sont réunis que pour 11 % d'entre elles : une bonne part de ces abstentions est
  honnête.
- **L'écart entre générateurs tient surtout à la prudence.** Classés par justesse, les
  modèles le sont presque à l'inverse par fausses abstentions. `qwen3.5:4b` et
  `granite4.1:3b` refusent de trancher sur plus des deux tiers des questions.
- **`phi4-mini:3.8b`** est le plus rapide des bons modèles. Il répond pourtant à un
  tiers des questions pièges, qui n'ont pas de réponse (8 sur 23 ; l'échantillon est
  petit, l'IC est large).
- **Mesure à faire** : combiner `qwen3:4b-instruct` et Qwen3-Reranker. C'est la
  configuration candidate au remplacement de `recommended`.

#### 6. Ablations (`configs/experiments/ablations.yml`)

- **Plafond de passages par article, 2 contre 4** : 4 fait perdre −0.030 d'articles à
  k = 5 [−0.046, −0.015], p Holm = 0.0008. Les passages supplémentaires d'un même
  article prennent les rangs utiles aux autres sources, alors que le multi-hop en a
  besoin. Le plafond de 2 est donc préférable à 4 ; les autres valeurs n'ont pas été
  testées.
- **Vivier sans reranker, 20 contre 50** : +0.010, p Holm = 0.95, non concluant. Sans
  reclassement, le RRF ne fait pas remonter plus haut les candidats supplémentaires :
  élargir le vivier ne sert qu'avec un reranker.

---

## 5. La fiabilité des juges

C'est le point le plus important et le moins intuitif de l'évaluation de RAG : **un juge
non calibré transforme une mesure en opinion, sans prévenir.** Le sujet a produit assez de
résultats pour mériter son propre document — **[JUGES.md](JUGES.md)**.

L'essentiel en six lignes :

| Constat | Chiffre |
|---|---|
| Un juge LLM n'est **pas reproductible** | 1.000 puis 0.583 sur deux passes identiques |
| Un petit juge **ne sait pas tout juger** | 2 métriques DeepEval sur 4 à couverture nulle |
| Un modèle **plus récent** ne juge pas mieux | κ +0.000 (4 mois) contre +0.458 (1 an) |
| Aucun juge local **n'atteint le seuil** d'utilisabilité | meilleur κ 0.458, seuil publié 0,6 |
| La tâche est pourtant **jugeable** | κ +0.958 par un très grand modèle |
| `contains` a **lui aussi** un plafond | une réponse abrégée correcte est comptée fausse |

Deux conséquences pratiques, à retenir même sans lire le détail :

- **Les valeurs de `contains` publiées par le banc sont des bornes basses.** Les
  *comparaisons* restent valides — le biais frappe identiquement les deux configurations
  et s'annule dans l'écart apparié.
- **`ragbench calibrate` n'est pas optionnel** quand vous branchez vos propres données.
  Un juge à κ < 0,4 sur votre domaine ne doit pas servir à trancher.

---

## 6. Défauts trouvés en exécutant, pas en relisant

Six défauts ont été trouvés en faisant tourner le banc. Aucun n'était visible à la
lecture du code, et cinq sur six étaient **silencieux** — ni erreur, ni exception, juste
des chiffres faux d'apparence plausible. C'est l'argument principal pour un banc plutôt
qu'une revue.

| Défaut | Symptôme observé | Ce qu'on aurait conclu à tort |
|---|---|---|
| `gemma4:e4b` est un modèle à raisonnement | 465 tokens de réflexion pour 20 caractères de réponse ; avec `max_tokens=512`, un tiers des réponses **vides** | « Le générateur échoue sur ces questions » |
| `/v1` ignore `think` en silence | Tokens de réflexion facturés, réglage sans effet | « Désactiver le raisonnement ne change rien » |
| `plainto_tsquery` assemble en ET | Recherche lexicale à **zéro résultat** sur les 200 questions | « Le lexical est inutile sur ce corpus » |
| Reranking sans `thinking=False` | Budget de 4 tokens mangé par la réflexion, chaîne vide, tous les scores à 0, ordre inchangé — pour 4 000 appels LLM | « Le reranking n'apporte rien » |
| DeepEval renvoie hors ordre | En asynchrone, les cas reviennent dans l'ordre de **complétion**. Un `zip` positionnel attribue chaque score à la mauvaise question — et les moyennes agrégées restent **identiques**, ce sont les mêmes valeurs permutées | Rien du tout : le défaut ne se voit que dans le drill-down |
| Chunks sans source ni date | Le générateur répond « le contexte ne contient pas d'article de The Verge » avec le bon article au rang 2 | « Le retrieval a échoué » |

Le correctif de ce dernier point — recopier titre, source et date en en-tête de chaque
chunk — vaut **+0.102 de justesse** (p < 0.001), l'un des plus gros gains de la campagne.

---

## 7. Ce que le banc ne mesure pas

Sections d'honnêteté, à lire avant d'extrapoler.

- **La robustesse.** Aucun jeu avec bruit injecté, contexte contradictoire ou
  information contrefactuelle. Les benchmarks RGB et CRUD-RAG couvrent ce terrain ; ils
  ne sont pas branchés.
- **La fidélité validée à grande échelle.** La calibration automatique ne couvre pas la
  fidélité, et l'annotation manuelle ne porte que sur 13 réponses — voir
  [JUGES.md](JUGES.md#6-annotation-manuelle-de-la-fidélité).
- **La latence en conditions réelles.** Les temps mesurés le sont sous
  `OLLAMA_NUM_PARALLEL=4`, avec un GPU dédié à la campagne. Ils ne préjugent pas d'une
  latence utilisateur en production.
- **La généralisation hors de MultiHop-RAG.** Toutes les conclusions ci-dessus valent
  pour un corpus de presse anglophone avec des questions multi-hop citant leurs sources.
  Le lexical qui bat le dense est très probablement un effet de ce corpus précis.
- **Le multilingue.** La recherche plein-texte est configurée en anglais
  (`to_tsvector('english', …)`).

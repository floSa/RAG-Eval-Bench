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

Toutes les mesures ci-dessous ont été obtenues sur le poste fixe GPU (NVIDIA RTX 4060 Ti 16 Go VRAM, 64 Go RAM, AMD Ryzen 5 9600X, Ollama central mutualisé avec accélération CUDA fp16, `LLM_CONCURRENCY=4`), sur l'échantillon standard de 200 questions (177 répondables) de MultiHop-RAG.

#### 1. Résolution de l'écart de reproductibilité (Baseline run 25)

Le run 25 (configuration `baseline.yml`, hash historique `bb2b20210128`) a été exécuté en **3 min 40 s** (1,1 s/question, contre environ 2 h sur CPU) et évalué :

| Métrique | Documenté initial | Mesuré sur CPU (28/09) | **Mesuré sur GPU (29/09, run 25)** | Écart run 25 vs doc | p |
|---|---|---|---|---|---|
| recall@3 | 0.351 | 0.525 | **0.3508** | +0.0000 | 1.000 |
| hit_rate@3 | 0.644 | 0.842 | **0.6441** | +0.0000 | 1.000 |
| MRR | 0.545 | 0.705 | **0.5448** | +0.0000 | 1.000 |
| nugget_recall | 0.244 | 0.368 | **0.2439** | +0.0000 | 1.000 |
| nugget_full_coverage | 0.0395 | 0.113 | **0.0395** | +0.0000 | 1.000 |
| contains (accuracy) | 0.158 | — | **0.1638** | +0.0056 | 1.000 |

**Verdict sur la reproductibilité** : L'accord avec les chiffres historiques est **total et parfait** (écarts nuls sur toutes les métriques de retrieval, tests appariés non significatifs avec \(p=1.0\)). L'écart observé sur le portable CPU provenait des spécificités d'environnement local (Ollama CPU 0.32 vs Ollama central GPU).

#### 2. Courbe de recall (`diagnose recall-curve`)

Couverture selon le nombre de passages remontés (`diagnose recall-curve configs/baseline.yml --json data/cache/courbe.json`) :

| Passages (k) | Articles de référence | Au moins 1 document | Faits couverts | Tous les faits |
|---|---|---|---|---|
| 5 | 0.400 [0.354, 0.448] | 0.695 [0.627, 0.763] | 0.250 [0.207, 0.292] | 0.045 [0.017, 0.079] |
| 10 | 0.508 [0.461, 0.556] | 0.808 [0.746, 0.864] | 0.317 [0.271, 0.362] | 0.079 [0.040, 0.119] |
| 20 | 0.649 [0.600, 0.698] | 0.864 [0.814, 0.910] | 0.420 [0.371, 0.466] | 0.141 [0.090, 0.192] |
| 30 | 0.719 [0.672, 0.765] | 0.910 [0.864, 0.949] | 0.456 [0.406, 0.504] | 0.158 [0.107, 0.215] |
| 50 | **0.820** [0.778, 0.860] | **0.949** [0.915, 0.977] | **0.530** [0.480, 0.577] | **0.209** [0.153, 0.266] |

À \(k=50\), 95 % des questions ont au moins un bon document dans le vivier. En revanche, tous les faits nécessaires ne sont présents qu'à 21 %, confirmant le classement comme goulot d'étranglement majeur.

#### 3. Techniques côté requête (`configs/experiments/query.yml`)

Comparaisons appariées contre la référence hybride `hybrid-k5` (générateur de requête : `qwen3.5:4b`) :

| Variante | Technique | doc_recall@5 | Écart [IC95] | p Holm | Latence médiane | Replis / taux | Verdict |
|---|---|---|---|---|---|---|---|
| `hybrid-rewrite` | Réécriture concise | 0.473 vs 0.512 | −0.040 [−0.074, −0.006] | 0.1072 | 2 203 ms (vs 590) | 0 fallback | À la limite (non retenu) |
| `hybrid-decompose` | Décomposition 3 sous-requêtes | 0.441 vs 0.512 | **−0.071** [−0.111, −0.032] | **0.0032** | 3 094 ms (vs 591) | 0 fallback | **Moins bon** |
| `hybrid-hyde` | Passage hypothétique (HyDE) | 0.466 vs 0.512 | **−0.046** [−0.081, −0.010] | **0.0396** | 7 661 ms (vs 597) | 0 fallback | **Moins bon** |
| `hybrid-clarify` | Clarification utilisateur simulé | 0.515 vs 0.512 | +0.003 [+0.000, +0.008] | 1.0000 | 1 052 ms (vs 602) | `clarify_asked`: 3 % (6/177) | Non concluant |

**Analyse** :
- La décomposition de requêtes et HyDE **dégradent significativement** le premier étage hybride sur MultiHop-RAG. Les sous-requêtes fragmentent l'intention et diluent les documents pertinents dans le top-5 RRF, tandis que le passage halluciné par HyDE injecte du bruit lexical néfaste sur un corpus factuel journalistique.
- `clarify` ne déclenche une question que dans 3 % des cas et apporte un gain imperceptible.

#### 4. Rerankers : Triomphe du cross-encoder Qwen3-0.6B (`configs/experiments/rerank.yml`)

Comparaison de `hybrid-k5` contre `hybrid-k5-qwen3-reranker` (vivier étendu à `fetch_k=50`, réordonnancement par `Qwen/Qwen3-Reranker-0.6B` en fp16 sur GPU CUDA) :

| Métrique | k | Hybride seul | Avec Qwen3-Reranker | Écart [IC95] | p Holm | Verdict |
|---|---|---|---|---|---|---|
| `doc_recall` | 5 | 0.512 | **0.704** | **+0.192** [+0.148, +0.239] | **0.0008** | **Meilleur** |
| `doc_recall` | 10 | 0.627 | **0.793** | **+0.166** [+0.129, +0.205] | **0.0008** | **Meilleur** |
| `nugget_full_coverage` | 5 | 0.113 | **0.271** | **+0.158** [+0.102, +0.220] | **0.0008** | **Meilleur** |
| `nugget_full_coverage` | 10 | 0.141 | **0.311** | **+0.169** [+0.113, +0.232] | **0.0008** | **Meilleur** |

- **Coût** : Latence médiane de 4,39 s par question (pour 50 passages évalués en batch).
- **Enseignement capital** : Alors que le cross-encoder MiniLM (entraîné sur MS-MARCO) dégradait l'hybride (−0,063), **Qwen3-Reranker-0.6B fait faire un bond de +19,2 points de rappel et fait plus que doubler la couverture complète des faits (11,3 % → 27,1 %)**. C'est le résultat le plus net du projet.

#### 5. Comparaison des générateurs légers (`configs/experiments/generators.yml`)

Campagne complète de 6 générateurs (< 4 Go) sur le pipeline hybride recommandé (prompt `default`, `temperature=0`, 200 questions chacune) :

| Modèle | Run | Justesse (`contains`) | Rejet correct | Fausses abstentions | Débit (s/q) | Tokens complétion |
|---|---|---|---|---|---|---|
| **`qwen3:4b-instruct`** | **28** | **0.503** [0.429, 0.576] | **1.000** | **0.429** | 1.0 s | 11 113 |
| `phi4-mini:3.8b` | 29 | 0.429 [0.356, 0.503] | 0.652 | 0.525 | 0.6 s | 3 361 |
| `qwen3.5:2b` | 27 | 0.362 [0.294, 0.429] | 0.783 | 0.446 | 1.1 s | 12 476 |
| `ministral-3:3b` | 31 | 0.333 [0.266, 0.401] | 0.957 | 0.576 | 0.7 s | 8 618 |
| `qwen3.5:4b` | 26 | 0.294 [0.226, 0.362] | 1.000 | 0.689 | 0.9 s | 2 612 |
| `granite4.1:3b` | 30 | 0.226 [0.169, 0.288] | 1.000 | 0.768 | 0.4 s | 999 |

**Résultats clés** :
- **`qwen3:4b-instruct` est le champion indiscutable** : il franchit pour la première fois la barre des **50 % de justesse** (0.503 vs 0.294 pour qwen3.5:4b, écart apparié **+0.209**, \(p < 0.001\)), avec un rejet négatif parfait (1.000) et le plus bas taux de fausse abstention.
- `phi4-mini:3.8b` est très rapide et précis (0.429) mais échoue sur le rejet négatif (0.652).
- `qwen3.5:4b` souffre d'un excès sévère de prudence (68,9 % de fausses abstentions).

#### 6. Ablations (`configs/experiments/ablations.yml`)

1. **Plafond par document (`max_per_document`) : 2 vs 4 passages** :
   - Relever le plafond à 4 dégrade significativement `doc_recall@5` (**−0.030** [−0.046, −0.015], \(p\) Holm = 0.0008). Un même document monopolise les rangs utiles et évince les autres sources nécessaires au multi-hop. La valeur 2 est donc confirmée comme optimale.
2. **Taille du vivier (`fetch_k`) : 20 vs 50 (sans rerank)** :
   - Écart non concluant (+0.010, \(p\) Holm = 0.9502) : sans second étage de reclassement cross-encoder, élargir le vivier RRF n'a pas d'effet sur le top-5. L'élargissement ne prend tout son sens qu'avec Qwen3-Reranker.

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

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

### Diagnostic de retrieval sans génération (28/09/2026)

Mesures de `ragbench diagnose`, sur le poste CPU (Core Ultra 7, Ollama 0.32 natif),
177 questions répondables de l'échantillon `eval`.

**Les chiffres historiques ne se reproduisent pas.** Même configuration (hash
`bb2b20210128`), même échantillon, métriques recalculées par les mêmes évaluateurs :

| Métrique | Documenté | Mesuré le 28/09 |
|---|---|---|
| recall@3 | 0.351 | 0.525 |
| hit_rate@3 | 0.644 | 0.842 |
| MRR | 0.545 | 0.705 |
| nugget_recall | 0.244 | 0.368 |
| nugget_full_coverage | 0.0395 | 0.113 |

Le hash couvre la configuration, pas le code ni le serveur d'inférence : l'écart vient
de l'un des deux, sans qu'on puisse encore dire lequel. **Conséquence** : ne comparer
entre eux que des runs produits sur le même poste et la même version du code. Le
tableau « Retrieval » ci-dessus reste valable en relatif, pas en absolu.

**Le plafond est le classement, pas la recherche.** Couverture selon le nombre de
passages remontés (`diagnose recall-curve`, config `baseline`) :

| Passages | Articles de référence | Faits couverts | Tous les faits |
|---|---|---|---|
| 5 | 0.567 | 0.368 | 0.113 |
| 10 | 0.710 | 0.455 | 0.181 |
| 20 | 0.871 | 0.553 | 0.237 |
| 50 | **0.956** | 0.630 | 0.322 |

À 50 passages, 96 % des articles de référence sont là : ils existent dans le vivier
mais arrivent trop bas. En revanche, les faits plafonnent à 63 % même quand les bons
articles sont remontés — deux suspects à tester séparément : le plafond de 2 passages
par article, et le support lexical (seuil de 60 % des mots), qui rate les reformulations.

**Un cross-encoder faible dégrade un premier étage fort** (`diagnose compare`,
`configs/experiments/rerank.yml`, `ms-marco-MiniLM-L-12-v2` sur 50 candidats) :

| Premier étage | Métrique @5 | Sans rerank | Avec | Écart [IC95] | p Holm | Verdict |
|---|---|---|---|---|---|---|
| dense | articles | 0.567 | 0.552 | −0.015 [−0.057, +0.027] | 1.00 | non concluant |
| dense | tous les faits | 0.113 | 0.079 | −0.034 [−0.085, +0.017] | 1.00 | non concluant |
| hybride | articles | 0.619 | 0.556 | **−0.063** [−0.111, −0.016] | 0.026 | **moins bon** |
| hybride | tous les faits | 0.164 | 0.056 | **−0.107** [−0.164, −0.051] | 0.002 | **moins bon** |

C'est le résultat que l'état de l'art laissait attendre ([ETAT-DE-L-ART.md](ETAT-DE-L-ART.md#11-retrieval)) :
un reranker entraîné sur des passages courts de MS MARCO ne fait pas mieux qu'un
premier étage hybride, et le dégrade nettement. Hypothèse non vérifiée : il ignore les
contraintes de source et de date portées par l'en-tête contextuel. Le coût est réel :
environ 2 s de CPU par question pour 50 passages (la latence médiane affichée par
`diagnose compare`, 7 à 8 s, inclut l'attente des workers sérialisés sur le modèle).
Reste à tester un reranker récent — Qwen3-Reranker-0.6B, le seul à battre nettement le
retrieval dense dans les mesures publiées.

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

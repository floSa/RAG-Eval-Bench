# Méthodologie d'évaluation

Ce document décrit **ce que le banc mesure et pourquoi on peut le croire**. Le cadrage
est dans [CADRAGE.md](CADRAGE.md), la mise en œuvre dans [ARCHITECTURE.md](ARCHITECTURE.md).

Tous les chiffres cités proviennent de campagnes réelles sur MultiHop-RAG, échantillon
stratifié de **200 questions**, générateur `gemma4:e4b`, embedder `nomic-embed-text`.

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

---

## 5. La fiabilité des juges

C'est le point le plus important, et le moins intuitif.

**Un juge LLM n'est pas reproductible.** Deux passes identiques, mêmes questions,
température 0, sur `llama3.2:3b` :

| Réglage | Passe 1 | Passe 2 |
|---|---|---|
| Sans graine | faithfulness **1.000** | faithfulness **0.583** |
| Avec graine fixée | faithfulness 0.633 | faithfulness 0.667 |

La graine réduit fortement la variation sans l'annuler. L'échantillon de mesure ne fait
que 5 questions — une seule qui bascule déplace la moyenne de 0,2 — mais l'ordre de
grandeur suffit à poser la règle : **un écart mesuré par un juge doit dépasser sa propre
variabilité avant qu'on en conclue quoi que ce soit.**

**Un petit juge ne sait pas tout juger.** `llama3.2:3b` produit `faithfulness` et
`answer_relevancy`, mais **aucun** score exploitable pour `contextual_precision` et
`contextual_recall` : leurs schémas JSON sont trop profonds pour un modèle de
3 milliards de paramètres. Ces métriques sont publiées avec une **couverture de 0**
plutôt qu'omises — une métrique absente se lirait comme une métrique non demandée.

**Le choix du juge est un arbitrage.** `gemma4:e4b` juge mieux mais raisonne à chaque
appel : 12 à 30 secondes par jugement, soit plus de 2 minutes par question sur les
quatre métriques. Une campagne Ragas de 20 questions a été interrompue après 40 minutes.

### Un modèle plus récent ne fait pas un meilleur juge

Hypothèse intuitive, et fausse. Quatre juges ont été mesurés sur les **mêmes** 60
réponses du run 24, dont trois nettement plus récents que `llama3.2:3b` :

| Juge | Publié | κ `claim_precision` | κ `claim_recall` | Trop généreux | Trop sévère | n |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | ~1 an | +0.048 | **+0.458** | 0.200 | 0.229 | 48–50 |
| `granite4.1:3b` | 2 mois | +0.191 | indéfini | 0.250 | **0.000** | 24 |
| `nemotron-3-nano:4b` | 4 mois | **+0.000** | indéfini | 0.318 | **0.000** | 22 |
| `qwen3.5:4b` | 4 mois | — | — | — | — | **0** |

Trois lectures, dans l'ordre de gravité :

1. **`nemotron-3-nano:4b` obtient κ = +0.000** — l'accord du hasard. Sa colonne « trop
   sévère » vaut 0.000 : il ne conteste **jamais**, il valide. Son `claim_recall` est
   constant à 1.0, d'où un κ indéfini — une sortie qui ne varie pas ne peut pas
   s'accorder avec quoi que ce soit. NVIDIA le donne état de l'art en suivi
   d'instructions (IFEval, IFBench) dans sa catégorie ; **cette compétence ne se
   transfère pas au jugement critique**.
2. **`qwen3.5:4b` n'a produit aucun jugement exploitable** sur les 60 questions.
3. **`llama3.2:3b`, le plus ancien, reste le meilleur** : seul κ au-dessus de 0,4. Il
   produit aussi deux fois plus de jugements exploitables (48 contre 22).

**Attention à ne pas inverser le raisonnement.** Un juge qui conteste tout n'a pas plus
raison qu'un juge qui valide tout : les deux sont dégénérés, symétriquement. Que les
erreurs de `llama3.2:3b` aillent dans les deux sens (0.200 généreux / 0.229 sévère) est
un **symptôme** de non-dégénérescence, pas une preuve de justesse — un juge qui
contesterait au hasard serait tout aussi bidirectionnel.

Ce qui tranche, c'est le **κ, et lui seul**, parce qu'il est corrigé du hasard : un juge
aléatoire obtient κ ≈ 0 qu'il soit complaisant, sévère ou équilibré. Le +0.458 signifie
que l'accord avec la vérité terrain dépasse ce que le hasard expliquerait. C'est une
affirmation faible — « mieux que rien », pas « fiable » — et elle reste sous le repère
de 0,6 publié pour un juge utilisable.

Ce résultat rejoint la littérature : les petits modèles ouverts manifestent un
**biais de complaisance** — ils attribuent des scores élevés sans fonder leur jugement
sur les preuves, avec des taux de sycophantie mesurés entre 6 et 22 %
([Pacific AI](https://pacific.ai/detecting-and-evaluating-sycophancy-bias-an-analysis-of-llm-and-ai-solutions/),
[arXiv 2510.12462](https://arxiv.org/pdf/2510.12462)). La même revue note que
**peu de travaux évaluent les modèles ouverts de 3 à 4 milliards de paramètres comme
juges** — cette mesure comble donc un angle mort documenté, elle ne le contredit pas.

Aucun framework ne prescrit de juge. Ragas et DeepEval acceptent tous deux un modèle
local sans en recommander aucun : DeepEval renvoie explicitement au choix de
l'utilisateur ([DeepEval, LLM-as-a-judge](https://deepeval.com/blog/llm-as-a-judge)).
La méthode publiée est précisément celle du banc — échantillonner, annoter, calculer le
κ juge↔référence — avec pour repère **κ > 0,6 acceptable, > 0,8 solide**
([Label Your Data](https://labelyourdata.com/articles/llm-as-a-judge)). Sur ce corpus,
aucun des quatre juges n'atteint 0,6 : c'est la conclusion honnête à retenir.

*Limite* : n = 22 à 50, ces κ sont bruités. L'échec de `qwen3.5:4b` et le κ nul de
`nemotron-3-nano:4b` sont nets ; l'écart entre `granite4.1:3b` et `llama3.2:3b` ne
l'est pas. Reproductible :

```bash
ragbench eval run 24 -e native.claims -o judge=<modele> -o max_samples=60
ragbench calibrate 24
```

### Calibration contre la vérité terrain

Puisque MultiHop-RAG fournit les réponses gold, `native.answer/contains` n'est pas une
opinion mais une comparaison à la vérité terrain. Elle constitue donc une **référence
recevable pour calibrer les juges, sans annotation humaine** :

```bash
ragbench calibrate <run_id>
```

Résultat sur le run `decompose` :

| Métrique jugée | κ de Cohen | Trop généreux | Lecture |
|---|---|---|---|
| `native.answer/em` | +0.845 | 0.000 | Mesure bien la même chose |
| `native.answer/token_f1` | +0.833 | 0.011 | Mesure bien la même chose |
| `claim_recall` | +0.545 | 0.130 | Moyen — utilisable en tendance |
| `claim_precision` | **+0.164** | **0.306** | Faible, et systématiquement indulgent |
| `native.answer/false_abstention` | −0.567 | 0.605 | Anticorrélé, comme attendu |

`claim_precision` **valide la conception du banc** plutôt qu'elle ne l'infirme : une
réponse peut être parfaitement fidèle au contexte **et fausse**, parce que le mauvais
contexte a été remonté. La fidélité n'est pas la justesse, et la calibration le montre
chiffres à l'appui.

Repères d'interprétation du κ : < 0,4 faible · 0,4–0,6 moyen · 0,6–0,8 substantiel ·
> 0,8 excellent. L'accord brut seul est trompeur : sur un jeu où 80 % des réponses sont
fausses, un juge qui répond toujours « fausse » atteint 80 % d'accord et un κ de 0.

### Ce que la calibration ne remplace pas

La référence porte sur la **justesse**. Elle valide correctement les métriques qui
prétendent mesurer la justesse, mais reste un signal **faible pour la fidélité** : une
réponse peut être fidèle et fausse, ou juste et mal soutenue. Pour valider un juge de
fidélité, l'annotation humaine reste nécessaire — d'où l'onglet **Annotation** du
tableau de bord, qui conserve sa raison d'être.

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
- **La fidélité validée par un humain.** Voir plus haut : la calibration contre la
  vérité terrain ne couvre pas la fidélité.
- **La latence en conditions réelles.** Les temps mesurés le sont sous
  `OLLAMA_NUM_PARALLEL=4`, avec un GPU dédié à la campagne. Ils ne préjugent pas d'une
  latence utilisateur en production.
- **La généralisation hors de MultiHop-RAG.** Toutes les conclusions ci-dessus valent
  pour un corpus de presse anglophone avec des questions multi-hop citant leurs sources.
  Le lexical qui bat le dense est très probablement un effet de ce corpus précis.
- **Le multilingue.** La recherche plein-texte est configurée en anglais
  (`to_tsvector('english', …)`).

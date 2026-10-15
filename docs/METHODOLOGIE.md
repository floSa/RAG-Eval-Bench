# Méthodologie d'évaluation

Ce document décrit **ce que le banc mesure et pourquoi on peut le croire**. Le cadrage
est dans [CADRAGE.md](CADRAGE.md), la mise en œuvre dans [ARCHITECTURE.md](ARCHITECTURE.md),
les juges LLM dans [JUGES.md](JUGES.md).

**Si le vocabulaire ne vous est pas familier, commencez par le
[GLOSSAIRE.md](GLOSSAIRE.md)** — chaque terme y est expliqué en français simple, sans
formule.

Cette branche (`outil`) ne contient **aucun résultat de corpus**. Les mesures réalisées
sur le corpus public MultiHop-RAG sont sur la branche `main`.

---

## Les questions que le banc permet de trancher

Chaque question appelle une comparaison **appariée** entre deux configurations qui ne
diffèrent que sur un point.

| Question | Ce qu'on fait varier | Ce qu'on lit |
|---|---|---|
| Recherche dense, lexicale ou hybride ? | `retrieval.mode` | `recall@k`, `nDCG@k` |
| Le reranking vaut-il son coût ? | `retrieval.rerank` | `recall@k` et la latence |
| Faut-il rappeler la source dans chaque passage ? | `chunking.header_fields` | Justesse |
| Quel prompt ? | `generation.prompt_template` | Justesse et abstention |
| Quel modèle générateur ? | `models.generator` | Justesse, fausses abstentions, coût |
| Quel modèle juge ? | `models.judge` | κ contre la référence |
| Le RAG bat-il le modèle seul ? | `closed_book` | Justesse |
| Qu'est-ce qui limite le système ? | Rien : diagnostic | `nugget_full_coverage` |

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

## 4. Lire ses résultats

Ordre de lecture recommandé pour un run.

1. **Vérifier la précision.** L'onglet *Runs* donne la demi-largeur de l'intervalle de
   confiance. Un écart plus petit n'est pas mesurable avec ce nombre de questions.
2. **Lire le retrieval avant la génération.**
   - `nugget_full_coverage` bas : les passages remontés ne contiennent pas tous les
     faits nécessaires. Le générateur ne peut pas y remédier.
   - Dans ce cas, changer de prompt ou de modèle ne sert à rien. Agir sur la recherche.
3. **Lire la justesse avec l'abstention.**
   - Fausses abstentions élevées : le modèle refuse de répondre à des questions
     répondables. C'est de la prudence, pas de l'ignorance.
   - Rejet correct bas : le modèle invente une réponse quand il n'y en a pas.
4. **Ne jamais lire la fidélité seule.** Une réponse peut être fidèle aux passages et
   fausse, parce que le mauvais passage a été remonté.
5. **Comparer avec `ragbench compare`.** Jamais deux moyennes côte à côte.

Lecture du verdict d'une comparaison :

| Affichage | Lecture |
|---|---|
| **oui** | L'intervalle de confiance exclut zéro : l'écart est établi |
| **limite** | Intervalle et test divergent : ne pas trancher |
| **non** | Écart non démontré : ce n'est pas la preuve qu'il n'existe pas |

Précautions :

- **Valeurs absolues** : `contains` est une borne basse de la justesse réelle. Les
  *comparaisons* restent valides, le biais s'annulant dans l'écart apparié.
- **Faux positifs** : le nombre attendu est affiché sous chaque tableau. Plus on
  compare de métriques, plus un écart « concluant » peut être un hasard.
- **Un juge non calibré** ne tranche rien : voir la section suivante.

---

## 5. La fiabilité des juges

C'est le point le plus important et le moins intuitif de l'évaluation de RAG : **un juge
non calibré transforme une mesure en opinion, sans prévenir.** Le sujet a produit assez de
résultats pour mériter son propre document — **[JUGES.md](JUGES.md)**.

L'essentiel :

- Un juge LLM n'est **pas reproductible** d'une passe à l'autre.
- Un petit juge **ne sait pas tout juger** : certaines métriques sortent à couverture nulle.
- Un modèle **plus récent** ne juge pas forcément mieux.
- Seul le **κ de Cohen contre une référence** dit si un juge est utilisable.
- `contains` a **lui aussi** un plafond : une réponse abrégée correcte est comptée fausse.

Deux conséquences pratiques :

- **Les valeurs de `contains` sont des bornes basses.** Les *comparaisons* restent
  valides — le biais frappe identiquement les deux configurations et s'annule dans
  l'écart apparié.
- **`ragbench calibrate` n'est pas optionnel.** Un juge à κ < 0,4 sur votre domaine ne
  doit pas servir à trancher.

---

## 6. Défauts trouvés en exécutant, pas en relisant

Six défauts ont été trouvés en faisant tourner le banc, sur le corpus public. Aucun n'était visible à la
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

---

## 7. Ce que le banc ne mesure pas

Sections d'honnêteté, à lire avant d'extrapoler.

- **La robustesse.** Aucun jeu avec bruit injecté, contexte contradictoire ou
  information contrefactuelle. Les benchmarks RGB et CRUD-RAG couvrent ce terrain ; ils
  ne sont pas branchés.
- **La fidélité validée automatiquement.** La calibration automatique ne couvre pas la
  fidélité : elle demande une annotation humaine — voir
  [JUGES.md](JUGES.md#5-fidélité-et-justesse).
- **La latence en conditions réelles.** Les temps mesurés le sont sous
  `OLLAMA_NUM_PARALLEL=4`, avec un GPU dédié à la campagne. Ils ne préjugent pas d'une
  latence utilisateur en production.
- **La généralisation d'un corpus à l'autre.** Une conclusion vaut pour le corpus sur
  lequel elle a été mesurée. Un résultat obtenu sur un corpus ne se transpose pas : il se
  refait sur le sien.
- **Le multilingue.** La recherche plein-texte est configurée en anglais
  (`to_tsvector('english', …)`).

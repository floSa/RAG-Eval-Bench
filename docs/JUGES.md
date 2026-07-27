# La fiabilité des juges

Quand il n'existe pas de bonne réponse à comparer — « cette réponse est-elle fidèle aux
sources ? » — on fait noter par un second modèle. C'est le **juge**.

Ce document rassemble tout ce que le banc a mesuré sur eux. C'est le sujet le plus
important et le moins intuitif de l'évaluation de RAG : **un juge non calibré transforme
une mesure en opinion, sans prévenir.**

Vocabulaire dans le [GLOSSAIRE.md](GLOSSAIRE.md). Les métriques et la statistique sont
dans [METHODOLOGIE.md](METHODOLOGIE.md).

---

## En résumé

| Constat | Chiffre |
|---|---|
| Un juge LLM n'est **pas reproductible** | 1.000 puis 0.583 sur deux passes identiques |
| Un petit juge **ne sait pas tout juger** | 2 métriques DeepEval sur 4 à couverture nulle |
| Un modèle **plus récent** ne juge pas mieux | κ +0.000 (4 mois) contre +0.458 (1 an) |
| Aucun juge local **n'atteint le seuil** d'utilisabilité | meilleur κ 0.458, seuil publié 0,6 |
| La tâche est pourtant **jugeable** | κ +0.958 par un très grand modèle |
| La référence déterministe a **elle aussi** un plafond | `contains` compte faux une réponse abrégée correcte |

---

## 1. Trois défauts structurels

**Un juge LLM n'est pas reproductible.** Deux passes identiques, mêmes questions,
température 0, sur `llama3.2:3b` :

| Réglage | Passe 1 | Passe 2 |
|---|---|---|
| Sans graine | faithfulness **1.000** | faithfulness **0.583** |
| Avec graine fixée | faithfulness 0.633 | faithfulness 0.667 |

La graine réduit fortement la variation sans l'annuler. L'échantillon ne fait que
5 questions — une seule qui bascule déplace la moyenne de 0,2 — mais l'ordre de grandeur
suffit à poser la règle : **un écart mesuré par un juge doit dépasser sa propre
variabilité avant qu'on en conclue quoi que ce soit.**

**Un petit juge ne sait pas tout juger.** `llama3.2:3b` produit `faithfulness` et
`answer_relevancy`, mais **aucun** score exploitable pour `contextual_precision` et
`contextual_recall` : leurs schémas JSON sont trop profonds pour un modèle de
3 milliards de paramètres. Ces métriques sont publiées avec une **couverture de 0**
plutôt qu'omises — une métrique absente se lirait comme une métrique non demandée.

**Le choix du juge est un arbitrage.** `gemma4:e4b` juge mieux mais raisonne à chaque
appel : 12 à 30 secondes par jugement, soit plus de 2 minutes par question sur les quatre
métriques. Une campagne Ragas de 20 questions a été interrompue après 40 minutes.

---

## 2. Un modèle plus récent ne fait pas un meilleur juge

Hypothèse intuitive, et fausse. Quatre juges ont été mesurés sur les **mêmes** 60 réponses
du run 24, dont trois nettement plus récents que `llama3.2:3b` :

| Juge | Publié | κ `claim_precision` | κ `claim_recall` | Trop généreux | Trop sévère | n |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | ~1 an | +0.048 | **+0.458** | 0.200 | 0.229 | 48–50 |
| `granite4.1:3b` | 2 mois | +0.191 | indéfini | 0.250 | **0.000** | 24 |
| `nemotron-3-nano:4b` | 4 mois | **+0.000** | indéfini | 0.318 | **0.000** | 22 |
| `qwen3.5:4b` | 4 mois | — | — | — | — | **0** |

Trois lectures, dans l'ordre de gravité :

1. **`nemotron-3-nano:4b` obtient κ = +0.000** — l'accord du hasard. Sa colonne « trop
   sévère » vaut 0.000 : il ne conteste **jamais**, il valide. Son `claim_recall` est
   constant à 1.0, d'où un κ indéfini — une sortie qui ne varie pas ne peut pas s'accorder
   avec quoi que ce soit. NVIDIA le donne état de l'art en suivi d'instructions (IFEval,
   IFBench) dans sa catégorie ; **cette compétence ne se transfère pas au jugement
   critique**.
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
affirmation faible — « mieux que rien », pas « fiable » — et elle reste sous le repère de
0,6 publié pour un juge utilisable.

Ce résultat rejoint la littérature : les petits modèles ouverts manifestent un **biais de
complaisance** — ils attribuent des scores élevés sans fonder leur jugement sur les
preuves, avec des taux de sycophantie mesurés entre 6 et 22 %
([Pacific AI](https://pacific.ai/detecting-and-evaluating-sycophancy-bias-an-analysis-of-llm-and-ai-solutions/),
[arXiv 2510.12462](https://arxiv.org/pdf/2510.12462)). La même revue note que **peu de
travaux évaluent les modèles ouverts de 3 à 4 milliards de paramètres comme juges** —
cette mesure comble donc un angle mort documenté, elle ne le contredit pas.

Aucun framework ne prescrit de juge. Ragas et DeepEval acceptent tous deux un modèle local
sans en recommander aucun : DeepEval renvoie explicitement au choix de l'utilisateur
([DeepEval, LLM-as-a-judge](https://deepeval.com/blog/llm-as-a-judge)). La méthode publiée
est précisément celle du banc — échantillonner, annoter, calculer le κ juge↔référence —
avec pour repère **κ > 0,6 acceptable, > 0,8 solide**
([Label Your Data](https://labelyourdata.com/articles/llm-as-a-judge)). Sur ce corpus,
aucun des quatre juges n'atteint 0,6 : c'est la conclusion honnête à retenir.

*Limite* : n = 22 à 50, ces κ sont bruités. L'échec de `qwen3.5:4b` et le κ nul de
`nemotron-3-nano:4b` sont nets ; l'écart entre `granite4.1:3b` et `llama3.2:3b` ne l'est
pas. Reproductible :

```bash
ragbench eval run 24 -e native.claims -o judge=<modele> -o max_samples=60
ragbench calibrate 24
```

---

## 3. Le plafond : ce que donnerait un très grand juge

Les quatre juges locaux plafonnent sous le seuil d'utilisabilité. Restait à savoir si
c'est une limite de la **taille des modèles** ou une limite de la **tâche** — certaines
questions de ce corpus sont peut-être indécidables. Un juge hors catégorie tranche la
question : les 53 mêmes réponses du run 24 ont été jugées à la main par **Claude Opus 5**,
en aveugle (référence non consultée avant verdict), sur le critère de `contains`.

| Juge | κ vs vérité terrain | Trop généreux | Trop sévère | n |
|---|---|---|---|---|
| **Claude Opus 5** (externe) | **+0.958** | 0.019 | 0.000 | 53 |
| `llama3.2:3b` | +0.458 | 0.200 | 0.229 | 48 |
| `granite4.1:3b` | +0.191 | 0.250 | 0.000 | 24 |
| `nemotron-3-nano:4b` | +0.000 | 0.318 | 0.000 | 22 |
| `qwen3.5:4b` | échec | — | — | 0 |

**La tâche est donc jugeable ; ce sont les petits modèles qui ne la jugent pas.** L'écart
0.958 contre 0.458 mesure exactement le prix de la contrainte on-premise sur ce poste.

*Ce que ça ne dit pas* : que le banc devrait appeler un juge externe. Claude Opus 5
**viole la contrainte on-premise** — un jugement par appel sortant, sur un corpus qui
pourrait être confidentiel dans un autre déploiement. C'est un **étalon de calibration**,
mesuré une fois sur un corpus public, pas une brique du pipeline.

*Limites* : n = 53, un seul annotateur, aucune réplication. Un protocole complet ferait
juger plusieurs annotateurs indépendants et mesurerait leur accord **entre eux** avant de
comparer un juge à eux. Les 53 verdicts sont conservés en base sous l'annotateur
`claude-opus-5` et consultables dans l'onglet **Annotation**.

---

## 4. La référence elle-même a un plafond

Le seul désaccord sur 53 est le résultat le plus utile de la mesure, parce qu'il
n'incrimine pas le juge mais **la référence** :

| Question | Réponse gold | Réponse du système | `contains` | Verdict humain |
|---|---|---|---|---|
| qid 278 | `Sam Bankman-Fried` | `Bankman-Fried` | **0** | **1** |

`contains` vérifie mécaniquement qu'une chaîne est incluse dans une autre. Une réponse
correcte donnée sous forme abrégée est comptée fausse. Sur un corpus saturé de noms
propres, ce n'est pas un cas limite.

**Conséquence à retenir** : toutes les valeurs de `contains` publiées par le banc sont des
**bornes basses** de la justesse réelle. Sur cet échantillon l'écart est d'un point
(0.340 mesuré, 0.358 réel).

**Ce que ça n'invalide pas** : les *comparaisons*. Le biais s'applique identiquement aux
deux configurations comparées, donc il se soustrait dans l'écart apparié. « `recommended`
double la justesse de `baseline` » tient ; « la justesse vaut exactement 0.339 » est à
lire comme « au moins 0.339 ». C'est la raison d'être de la discipline statistique
décrite dans [METHODOLOGIE.md](METHODOLOGIE.md#3-la-discipline-statistique) : on publie
des écarts testés, pas des valeurs absolues.

---

## 5. Calibration contre la vérité terrain

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

---

## 6. Annotation manuelle de la fidélité

La calibration automatique porte sur la justesse. La fidélité, elle, demandait une
annotation à la main — c'est la seule chose que la vérité terrain ne pouvait pas fournir.
**13 réponses non abstenues du run 24** ont été jugées avec les passages remontés
**intégraux** sous les yeux, critère : chaque affirmation est-elle soutenue par les
passages ?

**Fidélité = 11/13, soit 0.846.** Les deux échecs et surtout les trois réponses *fausses
mais fidèles* sont plus instructifs que la moyenne :

| Cas | Réponse | Diagnostic |
|---|---|---|
| qid 465 | « Google » là où les critères désignent Apple | **Non fidèle.** Le modèle a retenu l'entité la plus fréquente du contexte, pas celle qui satisfait les critères. Pas une hallucination — une attribution non fondée. |
| qid 655 | « Yes » | **Non fidèle.** Les passages disent explicitement le contraire des deux clauses. |
| qid 600, 720, 864 | réponses **fausses** | **Fidèles.** Le passage nécessaire n'avait pas été remonté ; répondre « No » était fondé sur ce qui avait été fourni. |

**C'est la preuve empirique, sur ce corpus, que fidélité ≠ justesse.** Trois réponses sur
treize sont fausses sans que le générateur soit en cause : il a raisonné correctement sur
un contexte incomplet. Publier la seule fidélité aurait donné 0.846 et masqué un taux de
justesse de 0.339.

**Ce que ça dit du diagnostic** : le générateur n'hallucine quasiment pas sur ce corpus,
il est affamé. Cohérent avec `nugget_full_coverage = 0.0395`. Avant de changer de modèle,
il faut remonter plus de documents pertinents.

*Limites* : n = 13, un seul annotateur. Suffisant pour établir que les deux notions
divergent, insuffisant pour chiffrer un taux de fidélité. Rejouable :
`uv run python scripts/annotation_fidelite.py`.

---

## 7. Ce que la calibration ne remplace pas

La référence porte sur la **justesse**. Elle valide correctement les métriques qui
prétendent mesurer la justesse, mais reste un signal **faible pour la fidélité** : une
réponse peut être fidèle et fausse, ou juste et mal soutenue. Pour valider un juge de
fidélité, l'annotation humaine reste nécessaire — d'où l'onglet **Annotation** du tableau
de bord, qui conserve sa raison d'être.

---

## À retenir si vous branchez vos propres données

`ragbench calibrate` **n'est pas optionnel**. Un juge qui obtient κ < 0,4 sur *votre*
domaine ne doit pas servir à trancher, quelles que soient ses performances ailleurs.
La marche à suivre est dans [BRANCHER-SES-DONNEES.md](BRANCHER-SES-DONNEES.md).

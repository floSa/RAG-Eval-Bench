# Glossaire

Chaque terme du banc, en français simple, sans formule. À lire avant les autres documents
si l'évaluation de RAG ne vous est pas familière.

---

## Les briques du système

**RAG** (*Retrieval-Augmented Generation*) — Au lieu de laisser un modèle répondre de
mémoire, on va d'abord **chercher** des passages dans une base documentaire, puis on les
lui **donne à lire** avant qu'il réponde. Deux étapes, donc deux endroits où ça peut
rater.

**Corpus** — L'ensemble de vos documents.

**Chunk** (passage) — Un document entier est trop gros pour être donné au modèle. On le
découpe en morceaux de quelques centaines de mots : ce sont les chunks. C'est l'unité que
la recherche remonte.

**Embedding** — Une façon de représenter un texte par une liste de nombres, construite
pour que deux textes de sens proche aient des nombres proches. C'est ce qui permet de
chercher « par le sens » plutôt que par les mots exacts.

**Recherche dense** — Chercher par le sens, via les embeddings. Trouve « voiture » quand
on demande « automobile ».

**Recherche lexicale** — Chercher par les mots exacts, comme un moteur de recherche
classique. Trouve les noms propres, les références, les dates — là où le sens ne suffit
pas.

**Recherche hybride** — Les deux à la fois, avec fusion des deux classements. C'est la
configuration recommandée du banc : elle vaut +0.098 de `recall@3` sur le corpus de démo.

**Reranking** — Après avoir remonté 30 passages, demander à un modèle de les reclasser du
plus au moins pertinent avant d'en garder 5. Sur ce corpus, **ça n'apporte rien** pour
5 fois le coût.

**Générateur** — Le modèle qui rédige la réponse finale à partir des passages fournis.
Ici `gemma4:e4b`.

**Juge** — Un **second** modèle, chargé de noter la réponse du premier quand il n'existe
pas de corrigé à comparer. Il doit être d'une autre famille que le générateur, sinon il
note ses propres copies et se surnote. Tout ce qu'on a mesuré à leur sujet est dans
[JUGES.md](JUGES.md).

---

## Ce qu'on mesure

### Sur la recherche

**`recall@k`** — Parmi les documents qu'il *fallait* trouver, quelle proportion se trouve
dans les k passages remontés. `recall@3 = 0.46` : on en récupère un peu moins de la
moitié dans le top 3.

**`precision@k`** — L'inverse : parmi les k passages remontés, quelle proportion était
réellement utile.

**`hit_rate@k`** — Question binaire : y a-t-il **au moins un** bon document dans le top k ?
Plus indulgent que `recall@k`.

**`MRR`** (*Mean Reciprocal Rank*) — À quelle position arrive le premier bon document. 1.0
= toujours en première position, 0.5 = en moyenne en deuxième.

**`nDCG@k`** — Comme `recall@k`, mais récompense le fait de placer les bons documents
**en haut** du classement plutôt qu'en bas.

**`nugget_recall`** — Un « nugget » est un **fait** nécessaire pour répondre. Cette mesure
regarde quelle proportion de ces faits est réellement présente dans les passages remontés
— pas juste le bon document, le bon *fait*.

**`nugget_full_coverage`** — Beaucoup plus exigeant : la proportion de questions pour
lesquelles **tous** les faits nécessaires sont présents. Sur le corpus de démo, **4 %**.
Autrement dit, dans 96 % des cas on ne donne pas au modèle de quoi répondre. C'est le
principal facteur limitant du système, loin devant le choix du modèle.

### Sur la réponse

**`contains`** — La réponse de référence apparaît-elle dans la réponse produite ? Simple
et mécanique — donc faillible : « Bankman-Fried » ne contient pas « Sam Bankman-Fried » et
sera compté faux alors que c'est la même personne. Les valeurs publiées sont donc des
**minimums**.

**`em`** (*exact match*) — La réponse est-elle **exactement** la réponse attendue, mot
pour mot. Très strict.

**`token_f1`** — Un entre-deux : quelle proportion de mots est partagée entre la réponse
produite et la réponse attendue.

**Justesse** (*correctness*) — La réponse est-elle **vraie** ?

**Fidélité** (*faithfulness*) — La réponse est-elle **soutenue par les passages fournis** ?

> **Ces deux-là ne sont pas la même chose, et c'est le piège numéro un.** Une réponse peut
> être irréprochablement fidèle aux passages *et fausse*, parce qu'on lui a remonté les
> mauvais passages. Mesuré sur le banc : 3 réponses fausses sur 13 étaient parfaitement
> fidèles. Ne publiez jamais l'une pour l'autre.

**Abstention** — Le modèle répond « je ne sais pas ». Ce n'est pas un échec en soi.

**Rejet correct** (*negative rejection*) — Sur les questions **réellement sans réponse**
dans le corpus, le modèle s'abstient-il ? On veut 1.000.

**Fausse abstention** — Sur les questions qui **avaient** une réponse, le modèle
s'est-il abstenu à tort ?

> Ces deux-là se publient **toujours ensemble**. Un modèle qui refuse de répondre à tout
> obtient un rejet correct parfait de 1.000 — et ne sert à rien.

---

## Comment on décide qu'un écart est réel

**Test apparié** — On fait passer les **mêmes** questions aux deux configurations
comparées, et on regarde l'écart question par question. Certaines questions sont dures
pour tout le monde ; cette difficulté commune s'annule dans l'écart. C'est bien plus
sensible que de comparer deux moyennes.

**Intervalle de confiance (IC95)** — La fourchette dans laquelle la vraie valeur se
trouve avec 95 % de chances. « +0.17, IC95 [+0.10, …] » : le gain est d'au moins 10
points, même dans l'hypothèse défavorable.

**p-value** — La probabilité d'observer un tel écart **si les deux configurations étaient
en réalité équivalentes**. En dessous de 0,05, on considère l'écart établi. Ce n'est pas
la probabilité d'avoir raison.

**Bootstrap** — Une façon d'obtenir un intervalle de confiance en retirant au sort dans
ses propres données des milliers de fois. Ne suppose rien sur la forme des données, ce
qui compte ici parce que les scores sont souvent des 0 et des 1.

**« Limite »** — Le verdict que rend `ragbench compare` quand l'intervalle de confiance et
le test de permutation ne disent pas la même chose. C'est un aveu d'incertitude, pas un
« oui » timide.

**κ de Cohen** (kappa) — Le taux d'accord entre deux juges (ou entre un juge et la vérité
terrain), **corrigé du hasard**. Indispensable : sur un jeu où 80 % des réponses sont
fausses, un juge qui répond toujours « fausse » obtient 80 % d'accord brut — et un κ de 0,
parce qu'il n'apporte aucune information.
>
> Repères : **< 0,4** faible · **0,4–0,6** moyen · **0,6–0,8** substantiel ·
> **> 0,8** excellent. Le seuil publié pour un juge sur lequel on s'appuie est **0,6**.

**κ « indéfini »** — Le juge a donné la même note à tout. Une sortie qui ne varie jamais ne
peut s'accorder avec rien : c'est le signe d'un juge dégénéré, pas d'un juge parfait.

**Vérité terrain** (*ground truth*) — Les bonnes réponses, établies à l'avance et faisant
autorité. Sans elle, le banc produit des chiffres mais pas des mesures.

---

## Les outils branchés

**Ragas** — Bibliothèque d'évaluation de RAG, quatre métriques de référence
(`faithfulness`, `answer_relevancy`, `contextual_precision`, `contextual_recall`).

**DeepEval** — Autre bibliothèque, orientée seuils réussite/échec — donc utilisable en
intégration continue pour détecter une régression.

**Couverture** — La proportion de questions pour lesquelles une métrique a effectivement
pu être calculée. Une métrique à couverture 0 signifie que le juge n'a rien produit
d'exploitable. Le banc la publie toujours, plutôt que de masquer la métrique.

**Déterministe** — Se dit d'une métrique calculée par un programme, sans modèle : gratuite,
reproductible à l'identique, non biaisée. À privilégier partout où c'est possible.

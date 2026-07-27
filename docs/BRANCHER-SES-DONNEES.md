# Brancher ses propres données

Ce banc est livré avec MultiHop-RAG en démonstration. Ce document explique comment le
faire tourner sur **vos** documents et **vos** questions.

Il y a deux marches, et la seconde est celle qu'on sous-estime :

1. **Charger vos documents** — une fonction Python d'une vingtaine de lignes.
2. **Construire une vérité terrain** — sans elle, les chiffres produits ne sont pas
   vérifiables. C'est le vrai travail, et il ne peut pas être automatisé entièrement.

---

## 1. Charger vos documents et vos questions

Tout le banc travaille sur une structure unique, `LoadedDataset`. Brancher un corpus
revient à écrire une fonction qui la renvoie — voir
[src/ragbench/datasets/base.py](../src/ragbench/datasets/base.py).

Créez `src/ragbench/datasets/mon_corpus.py` :

```python
from pathlib import Path
from typing import Any

from .base import LoadedDataset, register


@register("mon-corpus")
def load(raw_dir: Path, **_: Any) -> LoadedDataset:
    documents = [
        {
            "external_id": "DOC-001",       # identifiant stable, il sert de clé
            "title": "Procédure de sauvegarde",
            "body": "Le texte intégral du document…",
            "metadata": {"service": "infra", "version": "3.2"},
        },
        # …
    ]

    questions = [
        {
            "external_id": "Q-001",
            "question": "Quelle est la fréquence des sauvegardes incrémentales ?",
            "answer": "toutes les 4 heures",   # la réponse de référence — voir §2
            "question_type": "factuelle",      # sert à stratifier l'échantillon
            "gold_evidence": [
                {"document_external_id": "DOC-001", "fact": "sauvegarde incrémentale toutes les 4 h"},
            ],
        },
        # …
    ]

    return LoadedDataset(
        name="mon-corpus",
        source="interne",
        description="Documentation technique interne",
        documents=documents,
        questions=questions,
        metadata={"has_gold_evidence": True},
    )
```

Puis déclarez-le dans [src/ragbench/datasets/\_\_init\_\_.py](../src/ragbench/datasets/__init__.py) :

```python
from . import hotpotqa, mon_corpus, multihop_rag  # noqa: F401
```

C'est tout. Rien d'autre dans le projet ne change — `hotpotqa` existe précisément pour
prouver que cette abstraction tient sur un format très différent.

```bash
uv run ragbench dataset load mon-corpus --sample 200
uv run ragbench run configs/recommended.yml --dataset mon-corpus
```

### Les trois champs qui décident de ce que vous pourrez mesurer

| Champ | Si vous le renseignez | Si vous l'omettez |
|---|---|---|
| `answer` | `contains`, `em`, `token_f1` et la **calibration des juges** deviennent possibles | Il ne reste que des juges LLM, non calibrables |
| `gold_evidence` | `recall@k`, `nDCG@k`, `mrr` — vous saurez si l'échec vient du retrieval | Impossible de distinguer un échec de recherche d'un échec de génération |
| `question_type` | Échantillon stratifié, analyse par type | Tout est mis dans un même sac |

**`gold_evidence` vide a un sens fort** : la question est réputée **sans réponse** dans
le corpus. Ces questions mesurent la capacité à dire « je ne sais pas » (*negative
rejection*). Si votre dataset n'a pas de vérité terrain de retrieval du tout, déclarez-le
explicitement — `metadata["has_gold_evidence"] = False` — plutôt que de laisser des
listes vides qui seraient interprétées comme « sans réponse ».

---

## 2. Construire la vérité terrain — l'étape qu'on ne peut pas sauter

**C'est ici que tout se joue.** MultiHop-RAG fournit les réponses gold ; vos documents
techniques, non. Sans elles, la référence déterministe disparaît et il ne reste que des
juges LLM — dont ce banc a mesuré qu'aucun modèle local de 3–4 milliards de paramètres
n'est fiable (voir [METHODOLOGIE.md](METHODOLOGIE.md#un-modèle-plus-récent-ne-fait-pas-un-meilleur-juge)).

Brancher ses données sans vérité terrain produit des chiffres, pas des mesures.

### Combien de questions annoter ?

**50 à 100 suffisent** pour départager deux configurations. Mesuré sur ce banc : 53
annotations ont donné un κ de +0.958 et un intervalle de confiance exploitable. Ne visez
pas 1 000 questions au premier tour — vous n'en aurez pas besoin, et vous abandonnerez.

Les 2 356 questions non échantillonnées de MultiHop-RAG restent en base sous le split
`pool` exactement pour ce cas : agrandir l'échantillon quand un écart est trop serré
pour conclure.

### Trois façons de s'y prendre, par ordre de qualité

**a. Les vraies questions de vos utilisateurs, avec la réponse attendue.**
De loin la meilleure. Même 30 questions issues d'un support réel valent mieux que 300
questions inventées.

**b. Un expert métier rédige les questions à partir des documents.**
Correct, mais attention au biais : des questions écrites *depuis* les documents sont
mécaniquement des questions auxquelles le corpus répond. Vos scores seront meilleurs que
la réalité. Corrigez en ajoutant délibérément **20 à 30 % de questions sans réponse dans
le corpus** — sinon vous ne mesurerez jamais l'abstention.

**c. Un LLM génère les questions, un humain valide.**
Le plus rapide, le moins fiable. Acceptable pour amorcer, à condition qu'un humain relise
tout. Le biais du (b) s'applique en pire.

### Le raccourci qui marche : annoter par échantillon

Faire relire 300 questions à un expert est irréaliste. Le protocole efficace :

1. Un modèle (ou un prestataire) annote **tout**.
2. Votre expert relit **30 items tirés au hasard**.
3. On calcule le κ de Cohen entre les deux avec le code du banc.

**κ > 0.8** : l'annotation automatique fait foi, vous avez validé 300 questions en en
relisant 30. **κ < 0.6** : on ne garde rien, et les points de désaccord montrent
exactement où votre domaine est ambigu — c'est souvent l'information la plus utile de
l'exercice.

Repères : < 0,4 faible · 0,4–0,6 moyen · 0,6–0,8 substantiel · > 0,8 excellent.

Les deux scripts de [scripts/](../scripts/) donnent le patron exact :
[calibration_juge_externe.py](../scripts/calibration_juge_externe.py) pour la justesse,
[annotation_fidelite.py](../scripts/annotation_fidelite.py) pour la fidélité.

---

## 3. Les réglages à revoir pour un corpus non anglophone

La configuration livrée est réglée pour de l'actualité en **anglais**. Trois points à
reprendre :

| Réglage | Où | Pourquoi |
|---|---|---|
| `to_tsvector('english', …)` | [rag/retrieve.py](../src/ragbench/rag/retrieve.py) | La recherche lexicale ne lemmatise pas le français avec le dictionnaire anglais. Remplacer par `'french'`. |
| `chunk_size: 1000` | [configs/recommended.yml](../configs/recommended.yml) | Réglé pour des articles de presse. Une documentation technique très structurée demande souvent des chunks plus petits. |
| `header_fields` | idem | `[title, source, published_at]` correspond à de la presse. Mettez vos propres champs : version, service, référence. Ce réglage vaut +0.102 de justesse, ne le négligez pas. |

Le modèle d'embedding `nomic-embed-text` est multilingue, il n'a pas besoin d'être
changé pour du français.

---

## 4. La marche à suivre, de bout en bout

```bash
# 1. Le service LLM et la base répondent
uv run ragbench doctor

# 2. Charger votre corpus (échantillon stratifié de 200 questions)
uv run ragbench dataset load mon-corpus --sample 200

# 3. Une première campagne avec la configuration recommandée
uv run ragbench run configs/recommended.yml --dataset mon-corpus

# 4. Les métriques déterministes — gratuites, reproductibles
uv run ragbench eval run <run_id>

# 5. Vérifier que les juges disent la même chose que la vérité terrain
uv run ragbench calibrate <run_id>

# 6. Comparer deux configurations, avec test statistique
uv run ragbench compare <run_A> <run_B>

# 7. Le tableau de bord
uv run streamlit run src/ragbench/ui/app.py
```

**L'étape 5 n'est pas optionnelle.** Elle vous dit si les juges LLM sont utilisables sur
*votre* domaine. Un juge qui obtient κ < 0,4 chez vous ne doit pas servir à trancher,
quelles que soient ses performances sur MultiHop-RAG.

---

## 5. Trois pièges mesurés sur ce banc

**Ne comparez jamais deux moyennes à l'œil.** Utilisez `ragbench compare`, qui produit un
intervalle de confiance et un test apparié. Un écart de 2 points sur 200 questions n'est
pas un écart.

**Le facteur limitant est presque toujours le retrieval, pas le modèle.** Sur
MultiHop-RAG, dans 96 % des cas les passages remontés ne contiennent pas tous les faits
nécessaires — le générateur s'abstient à raison. Une annotation manuelle de 13 réponses
l'a confirmé : 3 réponses **fausses** étaient parfaitement **fidèles** aux passages qu'on
leur avait donnés. Avant de changer de modèle, regardez `nugget_full_coverage`.

**Fidélité n'est pas justesse.** Une réponse peut être irréprochable vis-à-vis des
passages fournis et fausse, parce que le mauvais passage a été remonté. Publiez toujours
les deux, jamais l'une pour l'autre.

---

## 6. Et si mes données sont confidentielles ?

Le banc est **100 % on-premise** : Postgres en local, inférence via le service Ollama
central, aucun appel sortant. Vos documents ne quittent pas votre machine.

Une seule exception, et elle est signalée comme telle dans
[METHODOLOGIE.md](METHODOLOGIE.md#le-plafond--ce-que-donnerait-un-très-grand-juge) : la
mesure d'étalonnage par un très grand modèle externe. Elle a été faite **une fois, sur un
corpus public**, pour chiffrer le plafond atteignable. Ne la rejouez pas sur des données
sensibles.

"""HotpotQA, reglage "distractor" — https://hotpotqa.github.io/

Present pour une raison methodologique, pas pour la performance : il valide
que la couche dataset est reellement pluggable. Son format est tres
different de MultiHop-RAG (corpus local a chaque question, evidence au
niveau de la PHRASE et non du document), et pourtant il se projette sur la
meme structure LoadedDataset. C'est le test que l'abstraction tient.

Difference a garder en tete pour l'interpretation : ici le corpus global
est l'union des paragraphes de toutes les questions, donc chaque question
est noyee dans les distracteurs des autres. La tache est mecaniquement plus
dure que le HotpotQA d'origine, ou le retrieval ne choisit qu'entre les 10
paragraphes de sa propre question. Ne pas comparer les chiffres obtenus ici
a ceux de la litterature.

Telechargement :
    curl -sSL -o data/raw/hotpot_dev_distractor_v1.json \\
      http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import LoadedDataset, read_json, register

DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


@register("hotpotqa")
def load(raw_dir: Path, filename: str = "hotpot_dev_distractor_v1.json", **_: Any) -> LoadedDataset:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"fichier absent : {path}\n  curl -sSL -o {path} {DEV_URL}"
        )

    raw = read_json(path)

    # Le corpus est l'union dedupliquee des paragraphes, indexee par titre
    # (l'identifiant utilise par supporting_facts).
    bodies: dict[str, str] = {}
    for item in raw:
        for title, sentences in item.get("context", []):
            bodies.setdefault(title, "".join(sentences).strip())

    documents = [
        {"external_id": title, "title": title, "body": body, "metadata": {}}
        for title, body in sorted(bodies.items())
        if body
    ]
    known = {d["external_id"] for d in documents}

    questions: list[dict[str, Any]] = []
    for item in raw:
        # supporting_facts donne (titre, index de phrase). On remonte au
        # document : notre unite de verite terrain est le document, parce
        # que c'est la granularite commune a tous les datasets du banc.
        # L'index de phrase est conserve en metadata pour un usage futur.
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for title, sent_id in item.get("supporting_facts", []):
            if title in known and title not in seen:
                seen.add(title)
                evidence.append(
                    {
                        "document_external_id": title,
                        "fact": "",
                        "sentence_ids": [sent_id],
                    }
                )
            elif title in seen:
                for ev in evidence:
                    if ev["document_external_id"] == title:
                        ev["sentence_ids"].append(sent_id)

        questions.append(
            {
                "external_id": item["_id"],
                "question": item["question"],
                "answer": item.get("answer"),
                "question_type": item.get("type"),  # bridge | comparison
                "gold_evidence": evidence,
                "metadata": {"level": item.get("level")},
            }
        )

    return LoadedDataset(
        name="hotpotqa",
        source="hotpotqa",
        description=(
            "HotpotQA distractor (dev) reprojete en corpus global : "
            "les distracteurs de toutes les questions sont melanges."
        ),
        documents=documents,
        questions=questions,
        metadata={
            "has_gold_evidence": True,
            "note": "corpus global, plus dur que le HotpotQA d'origine",
            "url": "https://hotpotqa.github.io/",
        },
    )

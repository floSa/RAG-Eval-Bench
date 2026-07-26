"""MultiHop-RAG (Tang & Yang, 2024) — https://huggingface.co/datasets/yixuantt/MultiHopRAG

Choisi comme corpus principal du banc pour trois raisons :

1. Il est CONSTRUIT pour l'evaluation RAG, pas detourne d'une tache de QA :
   chaque question porte sa liste de passages de reference (`evidence_list`),
   donc on peut calculer recall@k et nDCG@k avec une vraie verite terrain,
   sans passer par un juge LLM.
2. Il contient 301 `null_query` DELIBEREMENT sans reponse dans le corpus.
   C'est rare et precieux : ca permet de mesurer le negative rejection
   (savoir dire "je ne sais pas"), qui est le comportement le plus
   discriminant en production et le moins souvent evalue.
3. Sa taille (609 articles, ~5 000 chunks) tient sur un GPU 16 Go sans
   compromis, ce qui garde une campagne dans l'heure plutot que la journee.

Piege a connaitre — les reponses gold sont TRES courtes (mediane 3
caracteres) et 52 % sont des Yes/no. Deux consequences :
  - l'exact-match est utilisable, ce qui evite un juge LLM couteux ;
  - mais le niveau de chance sur les comparaisons binaires est de 50 %,
    donc une accuracy globale se lit uniquement ventilee par type de
    question. C'est fait dans evaluators/native_answer.py.

Telechargement :
    curl -sSL -o data/raw/corpus.json \\
      https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json
    curl -sSL -o data/raw/MultiHopRAG.json \\
      https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import LoadedDataset, read_json, register

CORPUS_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json"
QUERIES_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"


@register("multihop-rag")
def load(raw_dir: Path, **_: Any) -> LoadedDataset:
    corpus_path = raw_dir / "corpus.json"
    queries_path = raw_dir / "MultiHopRAG.json"

    missing = [p.name for p in (corpus_path, queries_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"fichiers absents de {raw_dir} : {', '.join(missing)}\n"
            f"  curl -sSL -o {corpus_path} {CORPUS_URL}\n"
            f"  curl -sSL -o {queries_path} {QUERIES_URL}"
        )

    raw_docs = read_json(corpus_path)
    raw_queries = read_json(queries_path)

    # L'URL sert d'identifiant : elle est unique sur les 609 articles et
    # c'est elle qui relie une evidence a son document. Le titre serait
    # unique aussi ici, mais il est plus fragile (apostrophes typographiques,
    # troncatures selon les sources).
    documents = [
        {
            "external_id": d["url"],
            "title": d["title"],
            "body": d["body"],
            "metadata": {
                "author": d.get("author"),
                "source": d.get("source"),
                "category": d.get("category"),
                "published_at": d.get("published_at"),
            },
        }
        for d in raw_docs
    ]

    known_urls = {d["external_id"] for d in documents}

    questions: list[dict[str, Any]] = []
    orphan_evidence = 0

    for i, q in enumerate(raw_queries):
        evidence = []
        for ev in q.get("evidence_list") or []:
            url = ev.get("url")
            if url not in known_urls:
                # Une evidence qui pointe hors corpus rendrait le recall
                # mecaniquement inatteignable : on la compte et on la
                # signale plutot que de la laisser fausser la mesure.
                orphan_evidence += 1
                continue
            evidence.append({"document_external_id": url, "fact": ev.get("fact", "")})

        questions.append(
            {
                # Index stable : le dataset n'a pas d'identifiant propre, et
                # l'echantillonnage doit rester reproductible d'une
                # execution a l'autre.
                "external_id": f"mhr-{i:05d}",
                "question": q["query"],
                "answer": q.get("answer"),
                "question_type": q.get("question_type"),
                "gold_evidence": evidence,
                "metadata": {"n_evidence": len(evidence)},
            }
        )

    return LoadedDataset(
        name="multihop-rag",
        source="multihop-rag",
        description=(
            "MultiHop-RAG (Tang & Yang 2024) : 609 articles de presse, 2556 questions "
            "multi-hop avec passages de reference, dont 301 sans reponse dans le corpus."
        ),
        documents=documents,
        questions=questions,
        metadata={
            "has_gold_evidence": True,
            "orphan_evidence": orphan_evidence,
            "unanswerable_marker": "Insufficient information.",
            "url": "https://huggingface.co/datasets/yixuantt/MultiHopRAG",
        },
    )

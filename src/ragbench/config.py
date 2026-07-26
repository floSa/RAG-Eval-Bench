"""Configuration d'un pipeline RAG.

Principe : *tout* ce qui peut changer un resultat est declare ici, et rien
d'autre. Le hash d'une config est son identite ; deux runs qui partagent un
hash sont comparables, deux runs qui n'en partagent pas ne le sont pas. C'est
ce qui rend le suivi longitudinal possible.

Consequence pratique : ne jamais lire un parametre de pipeline depuis
os.environ ailleurs que dans cette classe. Un parametre hors config est un
parametre invisible dans les resultats.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChunkingConfig(_Frozen):
    # chunk_size et chunk_overlap sont en CARACTERES (cf. rag/chunking.py) :
    # une unite neutre vis-a-vis du tokenizer, donc comparable entre
    # generateurs. Compter ~4 caracteres par token en anglais.
    strategy: Literal["fixed", "recursive", "sentence", "document"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Prefixe applique aux passages avant embedding. nomic-embed-text attend
    # "search_document: " cote corpus et "search_query: " cote question ;
    # l'oublier coute plusieurs points de recall.
    document_prefix: str = "search_document: "


class RetrievalConfig(_Frozen):
    # "lexical" = recherche plein-texte Postgres (ts_rank_cd), pas BM25 au
    # sens strict — l'appeler bm25 serait un abus de langage. Sert de
    # temoin lexical face au dense, et de composante du mode hybride.
    mode: Literal["dense", "lexical", "hybrid"] = "dense"
    top_k: int = 5
    # Nombre de candidats remontes avant reranking (ignore si rerank is None).
    fetch_k: int = 20
    # Constante du Reciprocal Rank Fusion pour le mode hybride.
    rrf_k: int = 60
    rerank: Literal["none", "llm"] = "none"
    similarity_threshold: float | None = None
    query_prefix: str = "search_query: "
    # Reecriture de la question avant recherche (le pipeline d'origine le
    # faisait en dur ; c'est devenu une variable experimentale).
    query_rewrite: bool = False


class GenerationConfig(_Frozen):
    prompt_template: str = "default"
    temperature: float = 0.0
    max_tokens: int = 512
    # Consigne d'abstention. Sans elle, impossible de mesurer le negative
    # rejection (savoir dire "je ne sais pas").
    allow_abstain: bool = True


class ModelConfig(_Frozen):
    """Les 4 roles de modeles. Volontairement independants.

    Contrainte methodologique : judge != generator. Un LLM qui juge ses
    propres sorties surnote systematiquement (biais d'auto-preference,
    documente par le framework CALM). Verifie par PipelineConfig.warnings().
    """

    generator: str = "gemma4:e4b"
    embedder: str = "nomic-embed-text:latest"
    judge: str = "gemma4:e4b"
    verifier: str | None = None


class PipelineConfig(_Frozen):
    name: str
    description: str = ""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    seed: int = 42

    def payload(self) -> dict[str, Any]:
        """Contenu hashable : tout sauf le nom et la description, qui sont
        des etiquettes humaines et ne doivent pas changer l'identite."""
        d = self.model_dump()
        d.pop("name", None)
        d.pop("description", None)
        return d

    def hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def index_hash(self) -> str:
        """Identite de l'index de corpus.

        Seuls le chunking et l'embedder determinent les vecteurs stockes.
        Deux configs qui ne different que par top_k partagent donc le meme
        index : on ne reindexe pas 600 documents pour changer un k.
        """
        payload = {
            "chunking": self.chunking.model_dump(),
            "embedder": self.models.embedder,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def warnings(self) -> list[str]:
        """Problemes methodologiques detectables statiquement.

        Renvoyes plutot que leves : on veut pouvoir lancer une baseline
        volontairement biaisee, a condition que le biais soit trace dans les
        resultats.
        """
        out: list[str] = []
        if self.models.judge == self.models.generator:
            out.append(
                f"judge == generator ({self.models.judge}) : biais d'auto-preference, "
                "les scores de qualite seront optimistes"
            )
        if self.generation.temperature != 0:
            out.append(
                f"temperature={self.generation.temperature} : resultats non reproductibles, "
                "les ecarts entre configs seront confondus avec le bruit d'echantillonnage"
            )
        if self.retrieval.rerank != "none" and self.retrieval.fetch_k <= self.retrieval.top_k:
            out.append(
                f"fetch_k={self.retrieval.fetch_k} <= top_k={self.retrieval.top_k} : "
                "le reranker n'a rien a reordonner"
            )
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def matrix_from_yaml(cls, path: str | Path) -> list[PipelineConfig]:
        """Charge un fichier d'experience : une base + une liste de variantes.

        Format attendu :

            base:
              name: baseline
              retrieval: {top_k: 5}
            variants:
              - name: top_k_10
                retrieval: {top_k: 10}

        Chaque variante est fusionnee (merge profond) sur la base, ce qui
        evite de reecrire la config entiere pour changer un parametre.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        base = data.get("base", {})
        variants = data.get("variants") or [{}]
        return [cls.model_validate(_deep_merge(base, v)) for v in variants]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out

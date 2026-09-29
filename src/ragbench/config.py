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
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChunkingConfig(_Frozen):
    # chunk_size et chunk_overlap sont en CARACTERES (cf. rag/chunking.py) :
    # une unite neutre vis-a-vis du tokenizer, donc comparable entre
    # generateurs. Compter ~4 caracteres par token en anglais.
    strategy: Literal["fixed", "recursive", "sentence", "document"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # En-tete contextuel recopie sur CHAQUE chunk (« contextual chunking »).
    #
    # Ce n'est pas un detail de confort. Sur MultiHop-RAG, les questions
    # referencent explicitement l'organe de presse et la date (« as reported
    # by The Verge », « published on November 1, 2023 »). Sans ces champs
    # dans le texte du chunk, le generateur ne peut PAS verifier la
    # contrainte et repond « le contexte ne contient pas d'article de The
    # Verge » — alors meme que le bon article a ete remonte au rang 2.
    # Observe sur le pilote : c'est la cause principale des abstentions, pas
    # la qualite du retrieval.
    #
    # Champs resolus depuis documents.title et documents.metadata. Liste vide
    # = aucun en-tete (temoin utile : il mesure exactement ce que l'en-tete
    # apporte).
    header_fields: tuple[str, ...] = ("title", "source", "published_at")
    # "chunk"    : en-tete sur chaque chunk (defaut, plus couteux en tokens)
    # "document" : en-tete une seule fois avant decoupage — seul le premier
    #              chunk en herite, ce qui etait le comportement initial et
    #              se revele insuffisant.
    header_scope: Literal["chunk", "document"] = "chunk"
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
    # "llm"           : le juge note chaque candidat (pointwise, fetch_k appels).
    # "cross_encoder" : reranker dedie, modele designe par rerank_model.
    rerank: Literal["none", "llm", "cross_encoder"] = "none"
    # Modele du cross-encoder, sous la forme "<backend>:<modele>" de la
    # bibliotheque rerankers, par exemple "flashrank:ms-marco-MiniLM-L-12-v2".
    # Le backend fait partie de l'identite : le meme poids servi en ONNX ou
    # en PyTorch ne donne pas exactement les memes scores.
    rerank_model: str | None = None
    # Plafond de chunks conserves par document source.
    #
    # Sans plafond, un article tres proche de la question occupe tout le
    # top_k avec ses propres passages. C'est fatal en multi-hop, ou la
    # reponse exige 2 a 4 documents DIFFERENTS : le recall plafonne alors
    # que la precision semble bonne. L'effet s'aggrave avec les en-tetes
    # contextuels, qui rendent les chunks d'un meme document plus
    # semblables entre eux.
    # Necessite fetch_k > top_k pour avoir de quoi remplacer les chunks
    # ecartes.
    max_per_document: int | None = None
    similarity_threshold: float | None = None
    query_prefix: str = "search_query: "
    # Reecriture de la question avant recherche (le pipeline d'origine le
    # faisait en dur ; c'est devenu une variable experimentale).
    query_rewrite: bool = False
    # Decomposition de la question en sous-requetes, chacune cherchee
    # separement, les classements fusionnes par RRF avec celui de la question
    # d'origine. Cible le multi-hop : une question qui exige 3 articles
    # ressemble rarement a chacun d'eux pris isolement.
    query_decompose: bool = False
    max_sub_queries: int = 3
    # HyDE : le generateur redige un passage hypothetique qui repondrait a la
    # question, et c'est CE passage qui est vectorise pour la recherche dense
    # (la recherche lexicale garde la question). Aide quand vocabulaire de la
    # question et des documents divergent ; peut nuire quand ils sont alignes.
    hyde: bool = False

    @model_validator(mode="after")
    def _requete_coherente(self) -> RetrievalConfig:
        if self.max_sub_queries != 3 and not self.query_decompose:
            raise ValueError("max_sub_queries n'a d'effet qu'avec query_decompose=true")
        if self.max_sub_queries < 1:
            raise ValueError("max_sub_queries doit etre >= 1")
        if self.hyde and self.mode == "lexical":
            raise ValueError("hyde n'agit que sur la recherche dense : sans effet en mode lexical")
        return self

    @model_validator(mode="after")
    def _rerank_coherent(self) -> RetrievalConfig:
        # Erreur et non avertissement : un rerank_model sans effet changerait
        # le hash sans changer le pipeline, et deux runs identiques
        # paraitraient differents.
        if self.rerank == "cross_encoder" and not self.rerank_model:
            raise ValueError("rerank=cross_encoder exige rerank_model")
        if self.rerank_model and self.rerank != "cross_encoder":
            raise ValueError(
                f"rerank_model n'a d'effet qu'avec rerank=cross_encoder (rerank={self.rerank})"
            )
        if self.rerank_model and ":" not in self.rerank_model:
            raise ValueError(
                f"rerank_model attendu sous la forme <backend>:<modele>, recu {self.rerank_model!r}"
            )
        return self


class GenerationConfig(_Frozen):
    prompt_template: str = "default"
    temperature: float = 0.0
    # 512 suffit sans raisonnement, pas avec. Mesure faite sur gemma4:e4b :
    # une reponse de 20 caracteres consomme ~465 tokens de generation quand
    # le raisonnement est actif — le budget part entierement dans la
    # reflexion et la reponse visible ressort VIDE. C'est un piege
    # silencieux : ni erreur, ni abstention, juste une chaine vide.
    max_tokens: int = 1024
    # Consigne d'abstention. Sans elle, impossible de mesurer le negative
    # rejection (savoir dire "je ne sais pas").
    allow_abstain: bool = True
    # Raisonnement explicite du modele (« thinking »).
    #   None  = on laisse le modele decider
    #   False = desactive
    #   True  = force
    # Variable experimentale a part entiere : sur gemma4:e4b, le desactiver
    # divise les tokens generes par ~70 (499 -> 7 sur un prompt RAG type).
    # Reste a savoir ce que ca coute en justesse — c'est precisement ce que
    # le banc doit mesurer, pas ce qu'on doit supposer.
    #
    # Limite technique : Ollama n'honore ce reglage que sur son endpoint
    # NATIF /api/chat. Son endpoint OpenAI-compatible ignore le champ
    # silencieusement tout en facturant les tokens de reflexion. Cf.
    # llm.LLMClient._complete_ollama_native.
    thinking: bool | None = None


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


# Champs ajoutes apres les premiers runs, avec leur valeur neutre. Ils
# n'entrent dans le hash que s'ils s'en ecartent. Sans cette regle, ajouter une
# technique au catalogue changerait le hash de TOUTES les configs existantes,
# et plus aucun run historique ne serait comparable a un nouveau. Tout champ
# ajoute desormais doit etre declare ici, avec la valeur qui reproduit le
# comportement d'avant son ajout.
NEUTRAL_ADDITIONS: dict[tuple[str, str], Any] = {
    ("retrieval", "rerank_model"): None,
    ("retrieval", "query_decompose"): False,
    ("retrieval", "max_sub_queries"): 3,
    ("retrieval", "hyde"): False,
}


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
        for (section, key), neutral in NEUTRAL_ADDITIONS.items():
            if d.get(section, {}).get(key, neutral) == neutral:
                d[section].pop(key, None)
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

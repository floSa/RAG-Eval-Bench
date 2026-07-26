"""Configuration d'infrastructure — deliberement separee de PipelineConfig.

Distinction a tenir : ce qui est ici ne change *pas* un resultat
d'evaluation (adresse de la base, URL du serveur d'inference, niveau de
concurrence). Ce qui change un resultat vit dans config.PipelineConfig et
entre dans le hash. Melanger les deux rend les runs incomparables.

Le niveau de concurrence est la seule zone grise : il n'affecte pas les
scores, mais il affecte la duree — donc les metriques de latence, qui ne
doivent jamais etre comparees entre deux runs de concurrence differente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- Base de donnees -------------------------------------------------
    db_host: str = field(default_factory=lambda: _env("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(_env("DB_PORT", "5432")))
    db_user: str = field(default_factory=lambda: _env("DB_USER", "postgres"))
    db_password: str = field(default_factory=lambda: _env("DB_PASSWORD", "postgres"))
    db_name: str = field(default_factory=lambda: _env("DB_NAME", "ragbench"))

    # --- Serveur d'inference --------------------------------------------
    # Une seule URL, OpenAI-compatible. Ollama expose /v1 nativement, vLLM
    # aussi : basculer de l'un a l'autre ne demande que de changer cette
    # variable, et devient donc une variable experimentale mesurable.
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "ollama"))

    # Nombre d'appels LLM simultanes. A aligner sur OLLAMA_NUM_PARALLEL cote
    # llm-service : demander plus ne fait que remplir la file d'attente.
    llm_concurrency: int = field(default_factory=lambda: int(_env("LLM_CONCURRENCY", "4")))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "180")))
    llm_max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "3")))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def admin_dsn(self) -> str:
        """DSN sur la base `postgres`, pour creer la base cible."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/postgres"
        )


settings = Settings()

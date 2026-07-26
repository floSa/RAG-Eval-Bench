"""Interface en ligne de commande du banc.

Toute operation longue passe par ici plutot que par l'UI Streamlit :
une campagne d'evaluation dure des heures, elle ne doit pas dependre
d'un onglet de navigateur ouvert.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .config import PipelineConfig
from .llm import LLMClient, probe
from .settings import settings

app = typer.Typer(help="Banc d'evaluation RAG on-premise", no_args_is_help=True)
db_app = typer.Typer(help="Base de donnees", no_args_is_help=True)
config_app = typer.Typer(help="Configurations de pipeline", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(config_app, name="config")

console = Console()


@app.command()
def doctor() -> None:
    """Verifie que Postgres et le serveur d'inference repondent.

    A lancer avant toute campagne : decouvrir a la question 400 qu'un
    modele manque coute une heure de GPU.
    """
    table = Table("Composant", "Etat", "Detail")

    try:
        with db.connect() as conn:
            version = conn.execute("SELECT version()").fetchone()["version"]
            has_vector = conn.execute(
                "SELECT 1 AS ok FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        table.add_row("postgres", "[green]ok[/]", version.split(",")[0])
        table.add_row(
            "pgvector",
            "[green]ok[/]" if has_vector else "[red]absent[/]",
            "extension vector" + ("" if has_vector else " — lancer `ragbench db init`"),
        )
    except Exception as exc:  # noqa: BLE001
        table.add_row("postgres", "[red]ko[/]", f"{settings.dsn} — {exc}")

    result = asyncio.run(probe())
    if result["ok"]:
        models = result["models"]
        table.add_row("inference", "[green]ok[/]", f"{result['base_url']}")
        table.add_row("modeles", "[green]%d[/]" % len(models), ", ".join(models))
    else:
        table.add_row("inference", "[red]ko[/]", f"{result['base_url']} — {result['error']}")

    console.print(table)


@db_app.command("init")
def db_init() -> None:
    """Cree la base si besoin et applique le schema (idempotent)."""
    created = db.ensure_database()
    console.print(
        f"base [bold]{settings.db_name}[/] : "
        + ("[green]creee[/]" if created else "[dim]deja presente[/]")
    )
    db.migrate()
    console.print("schema [green]applique[/]")


@db_app.command("stats")
def db_stats() -> None:
    """Compte le contenu du banc, table par table."""
    tables = [
        "datasets", "documents", "corpus_indexes", "chunks",
        "questions", "configs", "runs", "predictions", "scores", "annotations",
    ]
    table = Table("Table", "Lignes")
    with db.connect() as conn:
        for name in tables:
            n = conn.execute(f"SELECT count(*) AS n FROM {name}").fetchone()["n"]
            table.add_row(name, str(n))
    console.print(table)


@config_app.command("show")
def config_show(path: str) -> None:
    """Affiche une config, son hash, son index_hash et ses avertissements."""
    cfg = PipelineConfig.from_yaml(path)
    console.print(f"[bold]{cfg.name}[/]  hash=[cyan]{cfg.hash()}[/]  index=[cyan]{cfg.index_hash()}[/]")
    console.print_json(json.dumps(cfg.payload()))
    for warning in cfg.warnings():
        console.print(f"[yellow]! {warning}[/]")


@config_app.command("matrix")
def config_matrix(path: str) -> None:
    """Developpe un fichier d'experience en liste de configs."""
    configs = PipelineConfig.matrix_from_yaml(path)
    table = Table("Nom", "hash", "index_hash", "Avertissements")
    for cfg in configs:
        table.add_row(cfg.name, cfg.hash(), cfg.index_hash(), str(len(cfg.warnings())))
    console.print(table)
    # Les index partages sont l'information utile : ils disent combien
    # d'indexations couteuses la matrice va reellement declencher.
    distinct = {c.index_hash() for c in configs}
    console.print(f"{len(configs)} configs, [bold]{len(distinct)}[/] index de corpus a construire")


@app.command()
def embed(text: str, model: str = "nomic-embed-text:latest") -> None:
    """Calcule un embedding — sert a verifier la dimension d'un modele."""

    async def _run() -> None:
        async with LLMClient() as llm:
            vec = await llm.embed_one(text, model=model)
            console.print(f"modele=[cyan]{model}[/] dim=[bold]{len(vec)}[/]")
            console.print(f"[dim]{vec[:8]}...[/]")

    asyncio.run(_run())


if __name__ == "__main__":
    app()

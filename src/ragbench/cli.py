"""Interface en ligne de commande du banc.

Toute operation longue passe par ici plutot que par l'UI Streamlit :
une campagne d'evaluation dure des heures, elle ne doit pas dependre
d'un onglet de navigateur ouvert.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import datasets as ds
from . import db, evaluators, runner
from .config import PipelineConfig
from .llm import LLMClient, probe
from .settings import settings

app = typer.Typer(help="Banc d'evaluation RAG on-premise", no_args_is_help=True)
db_app = typer.Typer(help="Base de donnees", no_args_is_help=True)
config_app = typer.Typer(help="Configurations de pipeline", no_args_is_help=True)
dataset_app = typer.Typer(help="Corpus et jeux de questions", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluateurs", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")
app.add_typer(eval_app, name="eval")

console = Console()
RAW_DIR = Path("data/raw")


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


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------


@dataset_app.command("list")
def dataset_list() -> None:
    """Loaders disponibles et datasets deja charges en base."""
    table = Table("Loader", "En base", "Documents", "Questions", "Sans reponse")
    with db.connect() as conn:
        for name in ds.available():
            row = db.get_dataset(conn, name)
            if row is None:
                table.add_row(name, "[dim]non[/]", "-", "-", "-")
                continue
            counts = conn.execute(
                """
                SELECT (SELECT count(*) FROM documents WHERE dataset_id = %s) AS docs,
                       (SELECT count(*) FROM questions WHERE dataset_id = %s) AS qs,
                       (SELECT count(*) FROM questions
                         WHERE dataset_id = %s AND gold_evidence = '[]'::jsonb) AS nulls
                """,
                (row["id"], row["id"], row["id"]),
            ).fetchone()
            table.add_row(
                name, "[green]oui[/]", str(counts["docs"]), str(counts["qs"]), str(counts["nulls"])
            )
    console.print(table)


@dataset_app.command("load")
def dataset_load(
    name: str,
    sample: int = typer.Option(
        0,
        help="Taille de l'echantillon d'evaluation (0 = tout). Stratifie par type "
             "de question et deterministe : toutes les configs voient les memes questions.",
    ),
    seed: int = typer.Option(42, help="Graine de l'echantillonnage."),
    raw_dir: Path = typer.Option(RAW_DIR, help="Dossier des fichiers bruts telecharges."),
) -> None:
    """Charge un corpus et son jeu de questions en base."""
    loaded = ds.get_loader(name)(raw_dir)
    summary = loaded.summary()
    console.print(f"[bold]{loaded.name}[/] — {loaded.description}")
    console.print(f"  {summary['documents']} documents, {summary['questions']} questions")
    console.print(f"  types : {summary['types']}")
    console.print(f"  sans reponse dans le corpus : {summary['unanswerable']}")
    if loaded.metadata.get("orphan_evidence"):
        console.print(
            f"  [yellow]{loaded.metadata['orphan_evidence']} evidence(s) hors corpus, ignorees[/]"
        )

    questions = loaded.questions
    if sample:
        kept = ds.stratified_sample(questions, sample, seed=seed)
        kept_ids = {q["external_id"] for q in kept}
        # Les questions non retenues restent en base avec split='pool' :
        # elles servent a agrandir l'echantillon plus tard sans recharger,
        # et a verifier a posteriori que l'echantillon est representatif.
        for q in questions:
            q["split"] = "eval" if q["external_id"] in kept_ids else "pool"
        console.print(f"  [cyan]echantillon eval : {len(kept)} questions (seed={seed})[/]")

    with db.connect() as conn:
        dataset_id = db.upsert_dataset(
            conn,
            name=loaded.name,
            source=loaded.source,
            description=loaded.description,
            metadata=loaded.metadata | {"summary": summary},
        )
        db.upsert_documents(conn, dataset_id, loaded.documents)
        db.upsert_questions(conn, dataset_id, questions)
        conn.commit()
    console.print("[green]charge[/]")


# ---------------------------------------------------------------------
# Campagnes
# ---------------------------------------------------------------------


@app.command("index")
def index_build(
    config: str,
    dataset: str = typer.Option(..., help="Nom du dataset en base."),
    force: bool = typer.Option(False, help="Reindexe meme si l'index existe deja."),
) -> None:
    """Construit l'index de corpus d'une config (chunking + embeddings)."""
    from .rag.ingest import ensure_index

    cfg = PipelineConfig.from_yaml(config)

    async def _build(llm: LLMClient, conn) -> object:
        row = db.get_dataset(conn, dataset)
        if row is None:
            raise typer.BadParameter(f"dataset '{dataset}' absent")

        with Progress(
            TextColumn("[bold]indexation"), BarColumn(),
            TextColumn("{task.completed}/{task.total} docs"),
            TextColumn("{task.fields[chunks]} chunks"), TimeElapsedColumn(),
            console=console,
        ) as bar:
            task = bar.add_task("idx", total=None, chunks=0)

            def _progress(done: int, total: int, chunks: int) -> None:
                bar.update(task, completed=done, total=total, chunks=chunks)

            return await ensure_index(
                conn, llm, cfg, dataset_id=row["id"], force=force, progress=_progress
            )

    async def _run() -> None:
        # db.connect() est un gestionnaire de contexte synchrone : il ne peut
        # pas etre imbrique dans un `async with` sur la meme ligne.
        async with LLMClient() as llm:
            with db.connect() as conn:
                report = await _build(llm, conn)

        state = "[dim]reutilise[/]" if report.reused else "[green]construit[/]"
        console.print(
            f"index {state} — hash=[cyan]{report.index_hash}[/] "
            f"{report.n_chunks} chunks / {report.n_documents} documents, dim={report.dim}"
        )

    asyncio.run(_run())


@app.command("run")
def run_cmd(
    config: str,
    dataset: str = typer.Option(..., help="Nom du dataset en base."),
    split: str = typer.Option("eval", help="Split de questions a utiliser."),
    limit: int = typer.Option(0, help="Limite le nombre de questions (0 = tout le split)."),
    label: str = typer.Option("", help="Etiquette lisible du run."),
    matrix: bool = typer.Option(False, help="Traite le fichier comme une matrice base/variants."),
) -> None:
    """Execute une campagne : config(s) x jeu de questions -> run(s)."""
    configs = (
        PipelineConfig.matrix_from_yaml(config) if matrix else [PipelineConfig.from_yaml(config)]
    )

    async def _run() -> None:
        for cfg in configs:
            for warning in cfg.warnings():
                console.print(f"[yellow]! {cfg.name} : {warning}[/]")

            with Progress(
                TextColumn(f"[bold]{cfg.name}"), BarColumn(),
                TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                console=console,
            ) as bar:
                task = bar.add_task("run", total=None)

                def _progress(done: int, total: int) -> None:
                    bar.update(task, completed=done, total=total)

                def _index_progress(done: int, total: int, chunks: int) -> None:
                    bar.update(task, completed=done, total=total)

                report = await runner.run_campaign(
                    cfg,
                    dataset_name=dataset,
                    split=split,
                    limit=limit or None,
                    label=label or None,
                    on_progress=_progress,
                    on_index_progress=_index_progress,
                )

            console.print(
                f"  run [bold cyan]{report.run_id}[/] — {report.n_questions} questions, "
                f"{report.n_failed} echecs, {report.elapsed_s:.0f}s "
                f"({report.elapsed_s / max(1, report.n_questions):.1f}s/question)"
            )
            console.print(f"  [dim]tokens : {report.usage}[/]")

    asyncio.run(_run())


@app.command("runs")
def runs_list(limit: int = 20) -> None:
    """Derniers runs."""
    table = Table("id", "label", "config", "hash", "dataset", "n", "echecs", "statut", "date")
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.label, c.name AS config_name, r.config_hash, d.name AS dataset,
                   r.n_questions, r.n_failed, r.status, r.started_at
            FROM runs r
            JOIN configs c ON c.hash = r.config_hash
            JOIN datasets d ON d.id = r.dataset_id
            ORDER BY r.id DESC LIMIT %s
            """,
            (limit,),
        ).fetchall()
    for r in rows:
        table.add_row(
            str(r["id"]), r["label"] or "", r["config_name"], r["config_hash"][:8], r["dataset"],
            str(r["n_questions"]), str(r["n_failed"]), r["status"],
            r["started_at"].strftime("%d/%m %H:%M"),
        )
    console.print(table)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


@eval_app.command("list")
def eval_list() -> None:
    """Evaluateurs disponibles, et ceux qui ne le sont pas avec la raison."""
    table = Table("Evaluateur", "Etat", "Prerequis / raison")
    for name in evaluators.available():
        cls = evaluators.get(name)
        table.add_row(name, "[green]ok[/]", ", ".join(cls.requires) or "-")
    for name, reason in evaluators.unavailable().items():
        table.add_row(name, "[red]indisponible[/]", reason[:90])
    console.print(table)


@eval_app.command("run")
def eval_run(
    run_id: int,
    evaluator: list[str] = typer.Option(
        None, "--evaluator", "-e", help="Evaluateur a appliquer (repetable). Defaut : tous les natifs."
    ),
) -> None:
    """Applique des evaluateurs a un run existant."""
    names = evaluator or [n for n in evaluators.available() if n.startswith("native.")]

    async def _run() -> None:
        reports = await runner.evaluate_run(run_id, names)
        for rep in reports:
            if rep.skipped:
                console.print(f"[yellow]{rep.evaluator} : ignore — {'; '.join(rep.problems)}[/]")
                continue
            console.print(
                f"[bold]{rep.evaluator}[/] — {rep.n_scores} scores en {rep.elapsed_s:.1f}s"
            )
            for problem in rep.problems:
                console.print(f"  [yellow]{problem}[/]")
            for metric, value in sorted(rep.aggregates.items()):
                # escape() indispensable : des noms comme accuracy[inference_query]
                # seraient interpretes par Rich comme des balises de style et
                # disparaitraient de l'affichage.
                console.print(f"  {escape(metric):<40} {value:.4f}")

    asyncio.run(_run())


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

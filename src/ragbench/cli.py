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

                # task est lie par defaut plutot que capture : la boucle sur
                # `configs` en cree un nouveau a chaque tour, et une capture
                # tardive ferait pointer les callbacks du run N sur la barre
                # du run N+1.
                def _progress(done: int, total: int, _task=task) -> None:
                    bar.update(_task, completed=done, total=total)

                def _index_progress(done: int, total: int, chunks: int, _task=task) -> None:
                    bar.update(_task, completed=done, total=total)

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
        None, "--evaluator", "-e",
        help="Evaluateur a appliquer (repetable). Defaut : les natifs gratuits.",
    ),
    option: list[str] = typer.Option(
        None, "--option", "-o",
        help="Reglage d'evaluateur, format cle=valeur (repetable). "
             "Ex. -o max_samples=40 -o nugget_mode=judge",
    ),
) -> None:
    """Applique des evaluateurs a un run existant.

    Par defaut, seuls les evaluateurs DETERMINISTES sont lances : ils sont
    gratuits et reproductibles. Les evaluateurs a base de juge (native.claims,
    native.erag, ragas, deepeval) doivent etre demandes explicitement, parce
    qu'ils coutent plusieurs appels LLM par question.
    """
    names = evaluator or ["native.ir", "native.answer", "native.nuggets"]

    options: dict[str, object] = {}
    for item in option or []:
        key, _, value = item.partition("=")
        if not value:
            raise typer.BadParameter(f"option mal formee : {item} (attendu cle=valeur)")
        # Conversion souple : un max_samples doit arriver en entier cote
        # evaluateur, pas en chaine.
        try:
            options[key] = int(value)
        except ValueError:
            try:
                options[key] = float(value)
            except ValueError:
                options[key] = value

    async def _run() -> None:
        reports = await runner.evaluate_run(run_id, names, options=options)
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


# ---------------------------------------------------------------------
# Comparaison
# ---------------------------------------------------------------------


@app.command("show")
def show_run(run_id: int) -> None:
    """Detail d'un run : metriques avec intervalles de confiance."""
    summary = runner.summarize(run_id)
    console.print(
        f"[bold]run {summary.run_id}[/] — {summary.label} "
        f"(config [cyan]{summary.config_name}[/] {summary.config_hash[:8]}, "
        f"dataset {summary.dataset}, {summary.n_questions} questions, "
        f"{summary.n_failed} echecs)"
    )
    for warning in summary.warnings:
        console.print(f"[yellow]! {warning}[/]")

    if summary.metrics:
        table = Table("Metrique", "Moyenne", "IC 95 %", "n")
        for name, ci in sorted(summary.metrics.items()):
            table.add_row(
                escape(name), f"{ci.mean:.4f}", f"[{ci.low:.4f}, {ci.high:.4f}]", str(ci.n)
            )
        console.print(table)

    if summary.aggregates:
        table = Table("Agregat de run", "Valeur")
        for name, value in sorted(summary.aggregates.items()):
            table.add_row(escape(name), f"{value:.4f}")
        console.print(table)

    if summary.usage:
        console.print(f"[dim]tokens : {summary.usage}[/]")


@app.command("compare")
def compare_runs(
    run_a: int,
    run_b: int,
    metric: str = typer.Option("", help="Une seule metrique, format evaluateur/metrique."),
) -> None:
    """Compare deux runs par test apparie sur les memes questions."""
    if metric:
        evaluator, _, name = metric.partition("/")
        results = [runner.compare(run_a, run_b, evaluator=evaluator, metric=name)]
    else:
        results = runner.compare_all(run_a, run_b)

    if not results:
        console.print("[yellow]aucune metrique commune aux deux runs[/]")
        return

    table = Table("Metrique", "A", "B", "Ecart", "IC 95 % de l'ecart", "p", "n", "Verdict")
    n_significant = 0
    n_borderline = 0
    for res in results:
        c = res.comparison
        if c.borderline:
            n_borderline += 1
            verdict = "[yellow]limite[/]"
        elif c.significant:
            n_significant += 1
            verdict = "[green]oui[/]"
        else:
            verdict = "[dim]non[/]"
        table.add_row(
            escape(f"{res.evaluator}/{res.metric}"),
            f"{c.mean_a:.4f}",
            f"{c.mean_b:.4f}",
            f"{c.delta:+.4f}",
            f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
            f"{c.p_value:.3f}",
            str(c.n_pairs),
            verdict,
        )
    console.print(table)
    console.print(
        f"[dim]{n_significant}/{len(results)} ecarts concluants. "
        f"Avec {len(results)} comparaisons a 5 %, environ "
        f"{len(results) * 0.05:.1f} faux positif(s) sont attendus par hasard.[/]"
    )
    if n_borderline:
        console.print(
            f"[yellow]{n_borderline} ecart(s) « limite » : l'intervalle de confiance "
            f"et la p-value ne concordent pas. A reproduire sur un echantillon plus "
            f"grand avant d'en tirer une decision.[/]"
        )

    dropped = max((r.n_dropped_a + r.n_dropped_b) for r in results)
    if dropped:
        console.print(
            f"[yellow]{dropped} question(s) ecartee(s) au maximum : "
            f"notees dans un run et pas dans l'autre[/]"
        )


@app.command("report")
def report_cmd(
    runs: list[int] = typer.Argument(None, help="Identifiants de runs. Defaut : les 10 derniers."),
    metrics: str = typer.Option(
        "native.ir/recall@3,native.ir/hit_rate@3,native.ir/mrr,"
        "native.nuggets/nugget_recall,native.answer/contains,native.answer/false_abstention",
        help="Metriques a afficher, separees par des virgules.",
    ),
) -> None:
    """Tableau comparatif de plusieurs runs, une ligne par run.

    Volontairement sans test statistique : c'est une vue d'ensemble pour
    reperer ou regarder, pas pour conclure. Deux valeurs proches ne sont
    pas departagees ici — passer par `ragbench compare`.
    """
    wanted = [m.strip() for m in metrics.split(",") if m.strip()]

    with db.connect() as conn:
        if runs:
            ids = list(runs)
        else:
            ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM runs WHERE status = 'completed' ORDER BY id DESC LIMIT 10"
                ).fetchall()
            ][::-1]
        names = {
            r["id"]: r["name"]
            for r in conn.execute(
                "SELECT r.id, c.name FROM runs r JOIN configs c ON c.hash = r.config_hash "
                "WHERE r.id = ANY(%s)",
                (ids,),
            ).fetchall()
        }

    table = Table("run", "config", *[escape(m.split("/")[-1]) for m in wanted])
    for run_id in ids:
        summary = runner.summarize(run_id)
        cells = []
        for metric in wanted:
            ci = summary.metrics.get(metric)
            cells.append(f"{ci.mean:.3f}" if ci else "[dim]—[/]")
        table.add_row(str(run_id), names.get(run_id, "?"), *cells)
    console.print(table)
    console.print(
        "[dim]Vue d'ensemble sans test statistique : deux valeurs proches ne sont "
        "pas departagees. Utiliser `ragbench compare A B`.[/]"
    )


@app.command("leaderboard")
def leaderboard_cmd(
    dataset: str = typer.Option("multihop-rag"),
    metric: str = typer.Option("native.answer/contains", help="evaluateur/metrique"),
    limit: int = 20,
) -> None:
    """Classement des runs d'un dataset sur une metrique."""
    evaluator, _, name = metric.partition("/")
    rows = runner.leaderboard(dataset, evaluator=evaluator, metric=name, limit=limit)
    if not rows:
        console.print(f"[yellow]aucun run note sur {escape(metric)}[/]")
        return

    table = Table("#", "run", "config", "Moyenne", "IC 95 %", "n")
    for rank, row in enumerate(rows, start=1):
        table.add_row(
            str(rank), str(row["run_id"]), row["config"],
            f"{row['mean']:.4f}", f"[{row['low']:.4f}, {row['high']:.4f}]", str(row["n"]),
        )
    console.print(table)
    console.print(
        "[dim]Classement, pas test statistique : deux lignes dont les IC se "
        "chevauchent ne sont pas departagees. Utiliser `ragbench compare`.[/]"
    )


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

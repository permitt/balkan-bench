"""``balkanbench leaderboard`` subcommand group."""

from __future__ import annotations

from pathlib import Path

import typer

from balkanbench.cli._paths import schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.leaderboard.export import ExportError, write_leaderboard_export

leaderboard_app = typer.Typer(
    name="leaderboard",
    help="Leaderboard export + submission utilities.",
    no_args_is_help=True,
    add_completion=False,
)

# Ranked task list and primary metrics for SuperGLUE v0.1. Pulled dynamically
# from the benchmark + task YAMLs so this stays truthful as the contribution
# flow adds new tasks.
_DEFAULT_SEEDS = 5


def _red(t: str) -> str:
    return typer.style(t, fg=typer.colors.RED, bold=True)


def _green(t: str) -> str:
    return typer.style(t, fg=typer.colors.GREEN, bold=True)


@leaderboard_app.command("export")
def export_cmd(
    benchmark: str = typer.Option(..., "--benchmark", help="Benchmark identifier."),
    language: str = typer.Option(..., "--language", help="BCMS language code."),
    results_dir: Path = typer.Option(
        ...,
        "--results-dir",
        help="Directory containing {benchmark}-{language}/ subtree of result artifacts.",
    ),
    out: Path = typer.Option(..., "--out", help="Path to write benchmark_results.json."),
    benchmark_version: str = typer.Option(
        "0.1.0", "--benchmark-version", help="Benchmark version recorded in the export."
    ),
) -> None:
    """Assemble `benchmark_results.json` from on-disk official artifacts."""
    try:
        ranked_tasks, primary_metrics = _collect_ranked_tasks(benchmark, language)
    except FileNotFoundError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    target_root = results_dir / f"{benchmark}-{language}"

    try:
        write_leaderboard_export(
            benchmark=benchmark,
            language=language,
            results_root=target_root,
            ranked_tasks=ranked_tasks,
            task_primary_metrics=primary_metrics,
            benchmark_version=benchmark_version,
            out_path=out,
            seeds=_DEFAULT_SEEDS,
        )
    except ExportError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    typer.echo(_green(f"Wrote leaderboard export to {out}"))


def _collect_ranked_tasks(benchmark: str, language: str) -> tuple[list[str], dict[str, str]]:
    """Walk the benchmark's task YAMLs, return (ranked_tasks, primary_metric map)."""
    import os

    configs_dir = Path(
        os.environ.get("BALKANBENCH_CONFIGS_DIR") or Path(__file__).resolve().parents[3] / "configs"
    )
    tasks_dir = configs_dir / "benchmarks" / benchmark / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"no tasks directory at {tasks_dir}")

    ranked: list[str] = []
    primary_map: dict[str, str] = {}
    for task_yaml in sorted(tasks_dir.glob("*.yaml")):
        cfg = load_yaml_with_schema(task_yaml, schemas_root() / "task_spec.json")
        if cfg.get("status") != "ranked":
            continue
        if language not in cfg["languages"].get("ranked", []):
            continue
        task = cfg["task"]
        ranked.append(task)
        primary_map[task] = cfg["metrics"]["task_score"]

    if not ranked:
        raise FileNotFoundError(f"no ranked tasks for {benchmark}/{language} under {tasks_dir}")
    return ranked, primary_map

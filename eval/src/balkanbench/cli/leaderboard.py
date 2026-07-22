"""``balkanbench leaderboard`` subcommand group."""

from __future__ import annotations

from pathlib import Path

import typer

from balkanbench.cli._paths import configs_root, schemas_root
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
    benchmark_version: str | None = typer.Option(
        None,
        "--benchmark-version",
        help="Benchmark version recorded in the export. Defaults to the "
        "`version` declared in configs/benchmarks/{benchmark}/benchmark.yaml.",
    ),
    access: str | None = typer.Option(
        None,
        "--access",
        help=(
            "Filter to one access mode (open_weights/api) and stamp it into the "
            "export. Omit for the legacy, unfiltered export."
        ),
    ),
) -> None:
    """Assemble `benchmark_results.json` from on-disk official artifacts."""
    if access is not None and access not in {"open_weights", "api"}:
        typer.echo(_red(f"--access must be one of open_weights, api (got {access!r})"))
        raise typer.Exit(code=1)

    try:
        ranked_tasks, primary_metrics = _collect_ranked_tasks(benchmark, language, access=access)
        resolved_version = benchmark_version or _default_benchmark_version(benchmark)
    except FileNotFoundError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    # Both access modes for a given benchmark/language share one results tree
    # (see write_generative_result_artifact); the split happens in
    # assemble_leaderboard via `access`, not via a different input directory.
    target_root = results_dir / f"{benchmark}-{language}"

    try:
        write_leaderboard_export(
            benchmark=benchmark,
            language=language,
            results_root=target_root,
            ranked_tasks=ranked_tasks,
            task_primary_metrics=primary_metrics,
            benchmark_version=resolved_version,
            out_path=out,
            seeds=_DEFAULT_SEEDS,
            access=access,
        )
    except ExportError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    typer.echo(_green(f"Wrote leaderboard export to {out}"))


def _default_benchmark_version(benchmark: str) -> str:
    """Read ``version`` from ``configs/benchmarks/{benchmark}/benchmark.yaml``.

    Used when ``--benchmark-version`` is omitted, so the export stays truthful
    to the benchmark's declared manifest version instead of a stale hardcoded
    default.
    """
    manifest = configs_root() / "benchmarks" / benchmark / "benchmark.yaml"
    cfg = load_yaml_with_schema(manifest, schemas_root() / "benchmark_spec.json")
    version = cfg["version"]
    assert isinstance(version, str)
    return version


def _collect_ranked_tasks(
    benchmark: str, language: str, *, access: str | None = None
) -> tuple[list[str], dict[str, str]]:
    """Walk the benchmark's task YAMLs, return (ranked_tasks, primary_metric map).

    The primary-metric column differs per board: an ``access="api"`` export
    reads ``api_protocol.metrics.task_score`` (the metric produced by the
    API-reformulated protocol, e.g. ``acc`` for a multiple-choice task scored
    generatively) rather than ``metrics.task_score`` (the open-weights
    loglikelihood metric, e.g. ``acc_norm``). Benchmarks without an
    ``api_protocol`` block (e.g. superglue) fall back to ``metrics.task_score``
    regardless of ``access``.
    """
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
        api_metrics = cfg.get("api_protocol", {}).get("metrics") if access == "api" else None
        primary_map[task] = (
            api_metrics["task_score"] if api_metrics else cfg["metrics"]["task_score"]
        )

    if not ranked:
        raise FileNotFoundError(f"no ranked tasks for {benchmark}/{language} under {tasks_dir}")
    return ranked, primary_map

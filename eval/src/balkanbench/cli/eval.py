"""``balkanbench eval`` CLI: run train + validation for N seeds, emit artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from datasets import load_dataset

from balkanbench.cli._paths import resolve_model_config, resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.evaluation import aggregate_seed_results, run_multiseed
from balkanbench.provenance import collect_provenance
from balkanbench.scoring.artifact import write_result_artifact


def _red(text: str) -> str:
    return typer.style(text, fg=typer.colors.RED, bold=True)


def _green(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN, bold=True)


def eval_cmd(
    model: str = typer.Option(..., "--model", help="Model config name (e.g. 'bertic')."),
    benchmark: str = typer.Option(..., "--benchmark", help="Benchmark identifier."),
    task: str = typer.Option(..., "--task", help="Task identifier within the benchmark."),
    language: str = typer.Option(..., "--language", help="BCMS language code."),
    seeds: list[int] = typer.Option(
        None,
        "--seeds",
        help="Seeds to run. Repeatable. Defaults to model YAML's seeds list.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for results. Artifact lands at "
        "{out}/{benchmark}-{language}/{model}/result.json.",
    ),
    benchmark_version: str = typer.Option(
        "0.1.0",
        "--benchmark-version",
        help="Benchmark version recorded in the artifact.",
    ),
    dataset_revision: str = typer.Option(
        "v0.1.0-data",
        "--dataset-revision",
        help="Dataset revision pinned on the result artifact.",
    ),
    run_type: str = typer.Option(
        "official",
        "--run-type",
        help="'official' (rankable) or 'experimental' (not rankable).",
    ),
) -> None:
    """Train on train, evaluate on validation, write a result artifact."""
    try:
        task_cfg = load_yaml_with_schema(
            resolve_task_config(benchmark, task), schemas_root() / "task_spec.json"
        )
        model_cfg = load_yaml_with_schema(
            resolve_model_config(model), schemas_root() / "model_spec.json"
        )
    except FileNotFoundError as exc:
        typer.echo(_red(f"config not found: {exc}"))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    chosen_seeds: list[int] = list(seeds) if seeds else list(model_cfg.get("seeds") or [])
    if not chosen_seeds:
        typer.echo(_red("no seeds provided on CLI and the model config does not declare any"))
        raise typer.Exit(code=1)

    typer.echo(
        _green(
            f"Loading dataset {task_cfg['dataset']['public_repo']}:{task_cfg['dataset']['config']}"
        )
    )
    datasets = load_dataset(
        task_cfg["dataset"]["public_repo"],
        task_cfg["dataset"]["config"],
        revision=dataset_revision,
    )

    typer.echo(
        _green(f"Running {model} on {benchmark}.{task}.{language} over seeds {chosen_seeds}")
    )
    seed_results = run_multiseed(
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        language=language,
        datasets=datasets,
        seeds=chosen_seeds,
        output_dir=out / "work" / f"{benchmark}-{language}" / model,
    )

    aggregate = aggregate_seed_results(seed_results)
    provenance = collect_provenance()

    hp_search: dict[str, Any] = {
        "tool": "optuna",
        "sampler": "TPESampler",
        "sampler_seed": 42,
        "num_trials": 0,
        "search_space_id": "none-v0.1",
        "early_stopping_policy": (
            f"patience={task_cfg['training'].get('early_stopping_patience', 0)} "
            f"on {task_cfg['training']['metric_for_best_model']}"
        ),
    }

    artifact_path = write_result_artifact(
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        language=language,
        seed_results=seed_results,
        aggregate=aggregate,
        provenance=provenance,
        dataset_revision=dataset_revision,
        benchmark_version=benchmark_version,
        hp_search=hp_search,
        out_dir=out,
        run_type=run_type,
    )

    typer.echo(_green(f"Artifact: {artifact_path}"))
    for name, value in aggregate.mean.items():
        stdev = aggregate.stdev.get(name, 0.0)
        typer.echo(f"  {name}: {value:.4f} ± {stdev:.4f}  (n={len(seed_results)} seeds)")

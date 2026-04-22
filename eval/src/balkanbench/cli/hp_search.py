"""``balkanbench hp-search`` CLI: Optuna TPE search on train -> validation."""

from __future__ import annotations

from pathlib import Path

import typer
from datasets import load_dataset

from balkanbench.cli._paths import resolve_model_config, resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.hp_search import HPSearchError, run_hp_search


def _red(t: str) -> str:
    return typer.style(t, fg=typer.colors.RED, bold=True)


def _green(t: str) -> str:
    return typer.style(t, fg=typer.colors.GREEN, bold=True)


def hp_search_cmd(
    model: str = typer.Option(..., "--model"),
    benchmark: str = typer.Option(..., "--benchmark"),
    task: str = typer.Option(..., "--task"),
    language: str = typer.Option(..., "--language"),
    n_trials: int = typer.Option(20, "--n-trials", help="Number of Optuna trials."),
    sampler_seed: int = typer.Option(42, "--sampler-seed"),
    seed_for_trials: int = typer.Option(
        42,
        "--seed-for-trials",
        help="Seed used for each trial's training run.",
    ),
    out: Path = typer.Option(..., "--out", help="Sweep output directory."),
    dataset_revision: str = typer.Option("v0.1.0-data", "--dataset-revision"),
) -> None:
    """Run Optuna HP search on train -> validation; write the winning config."""
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

    datasets = load_dataset(
        task_cfg["dataset"]["public_repo"],
        task_cfg["dataset"]["config"],
        revision=dataset_revision,
    )

    try:
        result = run_hp_search(
            task_cfg=task_cfg,
            model_cfg=model_cfg,
            language=language,
            datasets=datasets,
            n_trials=n_trials,
            sampler_seed=sampler_seed,
            out_dir=out,
            seed_for_trials=seed_for_trials,
        )
    except HPSearchError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    typer.echo(_green(f"Sweep complete: {result.sweep_id}"))
    typer.echo(f"  best_trial: {result.best_trial_number}")
    typer.echo(f"  best_value: {result.best_value:.4f}")
    typer.echo(f"  best_config: {result.best_config_path}")

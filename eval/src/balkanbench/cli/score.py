"""``balkanbench score`` CLI: score a predictions.jsonl against private labels."""

from __future__ import annotations

from pathlib import Path

import typer

from balkanbench.cli._paths import resolve_model_config, resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.scoring.score import ScoreError, score_predictions


def _red(text: str) -> str:
    return typer.style(text, fg=typer.colors.RED, bold=True)


def _green(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN, bold=True)


def score_cmd(
    predictions: Path = typer.Option(
        ..., "--predictions", help="Path to predictions.jsonl produced by balkanbench predict."
    ),
    model: str = typer.Option(..., "--model"),
    benchmark: str = typer.Option(..., "--benchmark"),
    task: str = typer.Option(..., "--task"),
    language: str = typer.Option(..., "--language"),
    out: Path = typer.Option(..., "--out", help="Output directory for the result artifact."),
    benchmark_version: str = typer.Option("0.1.0", "--benchmark-version"),
    dataset_revision: str = typer.Option("v0.1.0-data", "--dataset-revision"),
    run_type: str = typer.Option("official", "--run-type"),
) -> None:
    """Score a predictions.jsonl against the private labels HF repo (requires HF_OFFICIAL_TOKEN)."""
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

    try:
        artifact_path = score_predictions(
            predictions_path=predictions,
            task_cfg=task_cfg,
            model_cfg=model_cfg,
            language=language,
            dataset_revision=dataset_revision,
            benchmark_version=benchmark_version,
            out_dir=out,
            run_type=run_type,
        )
    except ScoreError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    typer.echo(_green(f"Scored artifact written to {artifact_path}"))

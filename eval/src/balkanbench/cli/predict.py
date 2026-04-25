"""``balkanbench predict`` CLI: predict on the public test split, emit JSONL."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from balkanbench.cli._paths import resolve_model_config, resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.data.repo import DatasetRepoError, resolve_dataset_repo, resolve_hf_token
from balkanbench.evaluation import run_single_seed
from balkanbench.provenance import collect_provenance
from balkanbench.scoring.artifact import compute_predictions_hash


def __getattr__(name: str) -> Any:
    # Lazy import of datasets.load_dataset to keep `balkanbench --version` fast.
    if name == "load_dataset":
        import datasets

        return datasets.load_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _red(text: str) -> str:
    return typer.style(text, fg=typer.colors.RED, bold=True)


def _green(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN, bold=True)


def predict_cmd(
    model: str = typer.Option(..., "--model"),
    benchmark: str = typer.Option(..., "--benchmark"),
    task: str = typer.Option(..., "--task"),
    language: str = typer.Option(..., "--language"),
    out: Path = typer.Option(
        ..., "--out", help="Output directory for predictions.jsonl + run_metadata.json."
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for the single prediction run."),
    dataset_revision: str = typer.Option("v0.1.0-data", "--dataset-revision"),
) -> None:
    """Run prediction on the public test split. Does not require private labels."""
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
        repo_id = resolve_dataset_repo(task_cfg, language, prefer="private")
    except DatasetRepoError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc
    token = resolve_hf_token()

    from balkanbench.cli import predict as _self

    datasets = _self.load_dataset(
        repo_id,
        task_cfg["dataset"]["config"],
        revision=dataset_revision,
        token=token,
    )

    # Predict-only path: the public test split has no label column (hidden
    # test labels, see docs/methodology/data_provenance.md) and there is no
    # train split here, so we must explicitly opt out of both training and
    # metric computation. Scoring happens separately via `balkanbench score`
    # with the private labels, in the official scoring environment.
    seed_result = run_single_seed(
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        language=language,
        datasets=datasets,
        seed=seed,
        output_dir=out / "work",
        eval_split="test",
        train=False,
        compute_metrics=False,
    )

    out.mkdir(parents=True, exist_ok=True)
    preds_path = out / "predictions.jsonl"
    with preds_path.open("w") as fh:
        id_field = task_cfg["inputs"]["id_field"]
        ids = datasets["test"][id_field]
        for example_id, prediction in zip(ids, seed_result.predictions, strict=True):
            fh.write(
                json.dumps(
                    {"example_id": example_id, "prediction": prediction},
                    separators=(",", ":"),
                )
                + "\n"
            )

    run_meta = {
        "benchmark": benchmark,
        "task": task,
        "language": language,
        "model": model_cfg["name"],
        "model_id": model_cfg["hf_repo"],
        "dataset_revision": dataset_revision,
        "seed": seed,
        "num_predictions": len(seed_result.predictions),
        "test_predictions_hash": compute_predictions_hash(seed_result.predictions),
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": collect_provenance(),
        "sponsor": "Recrewty",
    }
    (out / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    typer.echo(
        _green(
            f"Wrote {len(seed_result.predictions)} predictions to {preds_path} "
            f"({run_meta['test_predictions_hash']})"
        )
    )

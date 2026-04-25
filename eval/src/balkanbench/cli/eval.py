"""``balkanbench eval`` CLI: run train + validation for N seeds, emit artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from balkanbench.cli._paths import resolve_model_config, resolve_task_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.data.repo import DatasetRepoError, resolve_dataset_repo, resolve_hf_token
from balkanbench.evaluation import aggregate_seed_results, run_multiseed
from balkanbench.provenance import collect_provenance
from balkanbench.scoring.artifact import write_result_artifact


# datasets.load_dataset is lazy to keep `balkanbench --version` fast. Tests
# that monkeypatch ``balkanbench.cli.eval.load_dataset`` still win because the
# monkeypatched attribute is consulted before __getattr__.
def __getattr__(name: str) -> Any:
    if name == "load_dataset":
        import datasets

        return datasets.load_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    eval_split: str | None = typer.Option(
        None,
        "--eval-split",
        help="Dataset split to evaluate on. Defaults to 'validation' for ranked "
        "tasks and 'test' for diagnostic tasks.",
    ),
    no_train: bool = typer.Option(
        False,
        "--no-train",
        help="Skip Trainer.train() and evaluate the loaded checkpoint as-is. "
        "Automatic for tasks with status='diagnostic' (no train split).",
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

    # Diagnostics (AXb / AXg) have only a test split and are evaluated
    # without training. Real use would load a fine-tuned RTE checkpoint
    # first; v0.1 runs the base model and relies on the below-random
    # sanity gate in DiagnosticTask to catch wiring bugs.
    is_diagnostic = task_cfg.get("status") == "diagnostic"
    effective_eval_split = eval_split or ("test" if is_diagnostic else "validation")
    effective_no_train = no_train or is_diagnostic
    if is_diagnostic and not no_train:
        typer.echo(
            typer.style(
                "diagnostic task detected: auto-enabling --no-train and "
                "--eval-split=test. For rigorous AXb/AXg scoring, run with a "
                "fine-tuned RTE checkpoint (v0.2 feature).",
                fg=typer.colors.YELLOW,
            )
        )

    try:
        repo_id = resolve_dataset_repo(task_cfg, language, prefer="private")
    except DatasetRepoError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc
    token = resolve_hf_token()

    typer.echo(_green(f"Loading dataset {repo_id}:{task_cfg['dataset']['config']}"))
    from balkanbench.cli import eval as _self

    datasets = _self.load_dataset(
        repo_id,
        task_cfg["dataset"]["config"],
        revision=dataset_revision,
        token=token,
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
        eval_split=effective_eval_split,
        train=not effective_no_train,
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

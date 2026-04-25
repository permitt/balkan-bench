"""``balkanbench run`` CLI: full pipeline for one (model, benchmark, language).

Loops the ranked tasks declared by the benchmark, runs HP search per task,
promotes the winning HP into a per-run model_cfg copy (the official
``configs/models/official/`` YAML is left untouched), runs a multi-seed
eval on the requested split, and finally writes a leaderboard export.

State lives entirely under ``--out``. A second invocation with the same
``--out`` resumes: tasks whose ``result.json`` already exists are skipped,
and a task whose sweep already completed reuses its winning config
without re-searching. This is enough to recover from a mid-run crash
without redoing the expensive bits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from balkanbench.cli._paths import (
    configs_root,
    resolve_model_config,
    resolve_task_config,
    schemas_root,
)
from balkanbench.config import load_yaml_with_schema
from balkanbench.data.repo import DatasetRepoError, resolve_dataset_repo, resolve_hf_token
from balkanbench.evaluation import aggregate_seed_results, run_multiseed
from balkanbench.hp_search import HPSearchError, run_hp_search
from balkanbench.leaderboard.export import ExportError, write_leaderboard_export
from balkanbench.provenance import collect_provenance
from balkanbench.scoring.artifact import write_result_artifact


def __getattr__(name: str) -> Any:
    if name == "load_dataset":
        import datasets

        return datasets.load_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _red(t: str) -> str:
    return typer.style(t, fg=typer.colors.RED, bold=True)


def _green(t: str) -> str:
    return typer.style(t, fg=typer.colors.GREEN, bold=True)


def _yellow(t: str) -> str:
    return typer.style(t, fg=typer.colors.YELLOW, bold=True)


def run_cmd(
    model: str = typer.Option(..., "--model", help="Model config name (e.g. 'bertic')."),
    benchmark: str = typer.Option(..., "--benchmark", help="Benchmark identifier."),
    language: str = typer.Option(..., "--language", help="BCMS language code."),
    out: Path = typer.Option(..., "--out", help="Run output directory (state + artifacts live here)."),
    tasks: list[str] = typer.Option(
        None,
        "--tasks",
        help="Restrict to these task names. Repeatable. Defaults to every ranked "
        "task that lists this language.",
    ),
    n_trials: int = typer.Option(
        20, "--n-trials", help="Optuna trials per task. Set 0 with --skip-hp-search."
    ),
    sampler_seed: int = typer.Option(42, "--sampler-seed"),
    seed_for_trials: int = typer.Option(42, "--seed-for-trials"),
    seeds: list[int] = typer.Option(
        None,
        "--seeds",
        help="Eval seeds. Repeatable. Defaults to the model YAML's seeds list.",
    ),
    eval_split: str = typer.Option(
        "test", "--eval-split", help="Dataset split to evaluate on. Defaults to test."
    ),
    skip_hp_search: bool = typer.Option(
        False,
        "--skip-hp-search",
        help="Skip the sweep stage and use the model YAML as-is for eval.",
    ),
    benchmark_version: str = typer.Option("0.1.0", "--benchmark-version"),
    dataset_revision: str = typer.Option("v0.1.0-data", "--dataset-revision"),
    run_type: str = typer.Option(
        "official", "--run-type", help="'official' (rankable) or 'experimental'."
    ),
) -> None:
    """End-to-end: HP search -> 5-seed eval on test -> leaderboard export."""
    schemas = schemas_root()
    try:
        model_cfg_base = load_yaml_with_schema(
            resolve_model_config(model), schemas / "model_spec.json"
        )
    except FileNotFoundError as exc:
        typer.echo(_red(f"model config not found: {exc}"))
        raise typer.Exit(code=1) from exc

    task_names = list(tasks) if tasks else _enumerate_ranked_tasks(benchmark, language)
    if not task_names:
        typer.echo(_red(f"no ranked tasks found for {benchmark}/{language}"))
        raise typer.Exit(code=1)

    chosen_seeds: list[int] = list(seeds) if seeds else list(model_cfg_base.get("seeds") or [])
    if not chosen_seeds:
        typer.echo(_red("no seeds on CLI and the model config does not declare any"))
        raise typer.Exit(code=1)

    token = resolve_hf_token()
    out.mkdir(parents=True, exist_ok=True)
    sweeps_dir = out / "sweeps"
    results_dir = out / "results"

    typer.echo(
        _green(
            f"Running {model} on {benchmark}/{language}: tasks={task_names} seeds={chosen_seeds} "
            f"split={eval_split} n_trials={n_trials if not skip_hp_search else 0}"
        )
    )

    for task in task_names:
        try:
            task_cfg = load_yaml_with_schema(
                resolve_task_config(benchmark, task), schemas / "task_spec.json"
            )
        except FileNotFoundError as exc:
            typer.echo(_red(f"task config not found for {task!r}: {exc}"))
            raise typer.Exit(code=1) from exc

        if language not in (task_cfg.get("languages", {}).get("ranked") or []):
            typer.echo(_yellow(f"skip {task}: {language} not in languages.ranked"))
            continue

        artifact_path = results_dir / f"{benchmark}-{language}" / model / task / "result.json"
        if artifact_path.is_file():
            typer.echo(_yellow(f"skip {task}: result already at {artifact_path} (resume)"))
            continue

        try:
            repo_id = resolve_dataset_repo(task_cfg, language, prefer="private")
        except DatasetRepoError as exc:
            typer.echo(_red(str(exc)))
            raise typer.Exit(code=1) from exc

        from balkanbench.cli import run as _self

        typer.echo(_green(f"[{task}] loading {repo_id}:{task_cfg['dataset']['config']}"))
        datasets = _self.load_dataset(
            repo_id,
            task_cfg["dataset"]["config"],
            revision=dataset_revision,
            token=token,
        )

        task_model_cfg, hp_search_meta = _resolve_task_model_cfg(
            base_model_cfg=model_cfg_base,
            task_cfg=task_cfg,
            language=language,
            datasets=datasets,
            sweeps_dir=sweeps_dir / task,
            n_trials=n_trials,
            sampler_seed=sampler_seed,
            seed_for_trials=seed_for_trials,
            dataset_revision=dataset_revision,
            skip_hp_search=skip_hp_search,
        )

        typer.echo(_green(f"[{task}] eval {chosen_seeds} on split={eval_split}"))
        seed_results = run_multiseed(
            model_cfg=task_model_cfg,
            task_cfg=task_cfg,
            language=language,
            datasets=datasets,
            seeds=chosen_seeds,
            output_dir=out / "work" / task,
            eval_split=eval_split,
            train=True,
        )
        aggregate = aggregate_seed_results(seed_results)
        provenance = collect_provenance()

        artifact = write_result_artifact(
            task_cfg=task_cfg,
            model_cfg=task_model_cfg,
            language=language,
            seed_results=seed_results,
            aggregate=aggregate,
            provenance=provenance,
            dataset_revision=dataset_revision,
            benchmark_version=benchmark_version,
            hp_search=hp_search_meta,
            out_dir=results_dir,
            run_type=run_type,
        )
        typer.echo(_green(f"[{task}] artifact: {artifact}"))

    typer.echo(_green("All tasks done. Building leaderboard export."))
    export_path = out / "benchmark_results.json"
    try:
        ranked_tasks, primary_metrics = _collect_ranked_tasks(benchmark, language)
        write_leaderboard_export(
            benchmark=benchmark,
            language=language,
            results_root=results_dir / f"{benchmark}-{language}",
            ranked_tasks=ranked_tasks,
            task_primary_metrics=primary_metrics,
            benchmark_version=benchmark_version,
            out_path=export_path,
            seeds=len(chosen_seeds),
        )
    except (ExportError, FileNotFoundError) as exc:
        typer.echo(_red(f"leaderboard export failed: {exc}"))
        raise typer.Exit(code=1) from exc

    typer.echo(_green(f"Leaderboard: {export_path}"))


def _resolve_task_model_cfg(
    *,
    base_model_cfg: dict[str, Any],
    task_cfg: dict[str, Any],
    language: str,
    datasets: Any,
    sweeps_dir: Path,
    n_trials: int,
    sampler_seed: int,
    seed_for_trials: int,
    dataset_revision: str,
    skip_hp_search: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run HP search (or reuse a prior winner) and return the task-scoped model_cfg.

    Returns ``(model_cfg, hp_search_meta)``. ``model_cfg`` carries the
    winning ``task_overrides`` block; ``hp_search_meta`` is the dict that
    lands on the result artifact's ``hp_search`` field.
    """
    if skip_hp_search:
        meta = _no_search_meta(task_cfg)
        return base_model_cfg, meta

    cached = _load_cached_winner(sweeps_dir)
    if cached is not None:
        cached_cfg, cached_meta = cached
        typer.echo(_yellow(f"reuse cached HP winner from {sweeps_dir}"))
        return cached_cfg, cached_meta

    sweeps_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_hp_search(
            task_cfg=task_cfg,
            model_cfg=base_model_cfg,
            language=language,
            datasets=datasets,
            n_trials=n_trials,
            sampler_seed=sampler_seed,
            out_dir=sweeps_dir,
            seed_for_trials=seed_for_trials,
            dataset_revision=dataset_revision,
        )
    except HPSearchError as exc:
        typer.echo(_red(f"HP search failed for {task_cfg['task']}: {exc}"))
        raise typer.Exit(code=1) from exc

    meta = {
        "tool": "optuna",
        "sampler": "TPESampler",
        "sampler_seed": sampler_seed,
        "num_trials": n_trials,
        "search_space_id": f"default-{task_cfg['task_type']}-v1",
        "early_stopping_policy": (
            f"patience={task_cfg['training'].get('early_stopping_patience', 0)} "
            f"on {task_cfg['training']['metric_for_best_model']}"
        ),
        "best_trial_number": result.best_trial_number,
        "best_value": result.best_value,
        "sweep_id": result.sweep_id,
    }
    return result.best_model_cfg, meta


def _load_cached_winner(sweeps_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """If a prior sweep wrote a ``sweep_summary.json``, rebuild (cfg, meta)."""
    if not sweeps_dir.is_dir():
        return None
    summaries = sorted(sweeps_dir.glob("sweep-*/sweep_summary.json"))
    if not summaries:
        return None
    latest = summaries[-1]
    summary = json.loads(latest.read_text())
    cfg = summary["best_model_cfg"]
    meta = {
        "tool": "optuna",
        "sampler": "TPESampler",
        "sampler_seed": summary["sampler_seed"],
        "num_trials": summary["n_trials"],
        "search_space_id": summary["search_space_id"],
        "early_stopping_policy": summary["early_stopping_policy"],
        "best_trial_number": summary["best_trial_number"],
        "best_value": summary["best_value"],
        "sweep_id": summary["sweep_id"],
    }
    return cfg, meta


def _no_search_meta(task_cfg: dict[str, Any]) -> dict[str, Any]:
    """hp_search field for runs with --skip-hp-search."""
    return {
        "tool": "optuna",
        "sampler": "TPESampler",
        "sampler_seed": 42,
        "num_trials": 0,
        "search_space_id": "skip-hp-search",
        "early_stopping_policy": (
            f"patience={task_cfg['training'].get('early_stopping_patience', 0)} "
            f"on {task_cfg['training']['metric_for_best_model']}"
        ),
    }


def _enumerate_ranked_tasks(benchmark: str, language: str) -> list[str]:
    """Discover ranked tasks for ``language`` by walking the configs dir."""
    tasks_dir = configs_root() / "benchmarks" / benchmark / "tasks"
    if not tasks_dir.is_dir():
        return []
    out: list[str] = []
    for path in sorted(tasks_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("status") != "ranked":
            continue
        if language not in (cfg.get("languages", {}).get("ranked") or []):
            continue
        out.append(cfg["task"])
    return out


def _collect_ranked_tasks(benchmark: str, language: str) -> tuple[list[str], dict[str, str]]:
    tasks_dir = configs_root() / "benchmarks" / benchmark / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"no tasks directory at {tasks_dir}")
    ranked: list[str] = []
    primary_map: dict[str, str] = {}
    for path in sorted(tasks_dir.glob("*.yaml")):
        cfg = load_yaml_with_schema(path, schemas_root() / "task_spec.json")
        if cfg.get("status") != "ranked":
            continue
        if language not in (cfg.get("languages", {}).get("ranked") or []):
            continue
        task = cfg["task"]
        ranked.append(task)
        primary_map[task] = cfg["metrics"]["task_score"]
    if not ranked:
        raise FileNotFoundError(f"no ranked tasks for {benchmark}/{language} under {tasks_dir}")
    return ranked, primary_map

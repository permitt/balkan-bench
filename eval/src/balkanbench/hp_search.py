"""Optuna-backed hyperparameter search driver.

Philosophy: HP search is an adjunct, not a first-class part of the
benchmark. It writes its outputs to a sweep-scoped directory and its
winning config is promoted into ``configs/models/official/`` by a human.
The main library never reads the sweep database at runtime; the sweep
lifecycle stays invisible to end users.

Design rules (from `spec_claude.md` decision log):
- Seeded sampler (``TPESampler(seed=sampler_seed)``) so reruns of a sweep
  are comparable across machines.
- Single-seed validation objective keeps compute bounded; rerank-across-
  seeds is a follow-on step once the search space has converged.
- Search space defaults are declared per task_type in Python (this file)
  and can be overridden by the caller via ``search_space=``. Community
  contributions extend via PR; each task YAML does not carry its own
  search space today.
- ``test`` is never used for tuning, selection, or early stopping. The
  evaluator already enforces ``--eval-split=validation`` for ranked
  tasks; the sweep driver stays inside that contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from balkanbench.evaluation import run_single_seed
from balkanbench.provenance import collect_provenance

try:  # pragma: no cover - import-level only
    import optuna
    from optuna.samplers import TPESampler
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "balkanbench.hp_search requires optuna. Install with: pip install 'balkanbench[ml]'"
    ) from exc


class HPSearchError(RuntimeError):
    """Raised when the sweep cannot proceed."""


@dataclass
class HPSearchResult:
    best_trial_number: int
    best_value: float
    best_model_cfg: dict[str, Any]
    best_config_path: Path
    sweep_id: str


# ---------------------------------------------------------------------
# Default search spaces
# ---------------------------------------------------------------------


_SPACE_BY_TASK_TYPE: dict[str, dict[str, Any]] = {
    "binary_classification": {
        "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 5e-5},
        "num_epochs": {"type": "int", "low": 3, "high": 15},
    },
    "multiclass_classification": {
        "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 5e-5},
        "num_epochs": {"type": "int", "low": 5, "high": 30},
    },
    "multiple_choice": {
        "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 5e-5},
        "num_epochs": {"type": "int", "low": 5, "high": 30},
    },
    "grouped_binary_classification": {
        "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 5e-5},
        "num_epochs": {"type": "int", "low": 3, "high": 10},
    },
    "wsc": {
        "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 5e-5},
        "num_epochs": {"type": "int", "low": 5, "high": 30},
    },
}


def default_search_space_for(task_type: str) -> dict[str, Any]:
    """Return the built-in search space for ``task_type``. Raises on unknown."""
    if task_type not in _SPACE_BY_TASK_TYPE:
        raise HPSearchError(
            f"no default search space for task_type={task_type!r}; "
            f"known families: {sorted(_SPACE_BY_TASK_TYPE)}"
        )
    # Return a shallow copy so callers can freely tweak without polluting the module-level dict.
    return {k: dict(v) for k, v in _SPACE_BY_TASK_TYPE[task_type].items()}


# ---------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------


def _suggest(
    trial: Any, name: str, spec: dict[str, Any]
) -> Any:  # pragma: no cover - dispatches to optuna
    kind = spec["type"]
    if kind == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if kind == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise HPSearchError(f"unknown search space entry type: {kind}")


def run_hp_search(
    *,
    task_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    language: str,
    datasets: Any,
    n_trials: int,
    sampler_seed: int,
    out_dir: Path,
    search_space: dict[str, Any] | None = None,
    seed_for_trials: int = 42,
) -> HPSearchResult:
    """Run an Optuna TPE sweep; return the best trial + write its config."""
    if n_trials < 1:
        raise HPSearchError(f"n_trials must be >= 1; got {n_trials}")

    space = search_space or default_search_space_for(task_cfg["task_type"])
    task_score_metric = task_cfg["metrics"]["task_score"]

    sweep_id = datetime.now(UTC).strftime("sweep-%Y%m%d-%H%M%S")
    sweep_dir = Path(out_dir) / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{sweep_dir / 'study.db'}"

    study = optuna.create_study(
        study_name=sweep_id,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=sampler_seed),
        load_if_exists=False,
    )

    def objective(trial: optuna.Trial) -> float:  # pragma: no cover - hit via study.optimize
        overrides = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        trial_model_cfg = _apply_training_overrides(model_cfg, overrides)
        seed_result = run_single_seed(
            model_cfg=trial_model_cfg,
            task_cfg=task_cfg,
            language=language,
            datasets=datasets,
            seed=seed_for_trials,
            output_dir=sweep_dir / f"trial-{trial.number}",
            eval_split="validation",
        )
        # Optuna maximises; return the task's task_score (which mirrors the
        # primary metric). Fall back to task_score field if the metric is not
        # itself in primary (shouldn't happen with the v0.1 configs).
        value = seed_result.primary.get(task_score_metric, seed_result.task_score)
        return float(value)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    if best.value is None:
        raise HPSearchError(f"sweep {sweep_id}: best trial has no value (all trials failed?)")
    best_value = float(best.value)
    best_model_cfg = _apply_training_overrides(model_cfg, best.params)
    best_config_path = sweep_dir / f"{model_cfg['name']}_best.yaml"
    _write_best_config(
        path=best_config_path,
        model_cfg=best_model_cfg,
        sweep_id=sweep_id,
        sampler_seed=sampler_seed,
        n_trials=n_trials,
        best_value=best_value,
        best_trial_number=int(best.number),
        task_cfg=task_cfg,
    )

    return HPSearchResult(
        best_trial_number=int(best.number),
        best_value=best_value,
        best_model_cfg=best_model_cfg,
        best_config_path=best_config_path,
        sweep_id=sweep_id,
    )


def _apply_training_overrides(
    base_model_cfg: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Return a copy of ``base_model_cfg`` with ``overrides`` merged into training."""
    new_cfg = {k: v for k, v in base_model_cfg.items()}
    new_training = dict(base_model_cfg.get("training") or {})
    new_training.update(overrides)
    new_cfg["training"] = new_training
    return new_cfg


def _write_best_config(
    *,
    path: Path,
    model_cfg: dict[str, Any],
    sweep_id: str,
    sampler_seed: int,
    n_trials: int,
    best_value: float,
    best_trial_number: int,
    task_cfg: dict[str, Any],
) -> None:
    """Write the winning model YAML with a provenance comment block."""
    provenance = collect_provenance()
    header = [
        "# Auto-generated by `balkanbench hp-search`. Do not hand-edit.",
        f"# sweep_id: {sweep_id}",
        f"# best_trial_number: {best_trial_number}",
        f"# best_value: {best_value:.6f}",
        f"# sampler_seed: {sampler_seed}",
        f"# n_trials: {n_trials}",
        f"# benchmark: {task_cfg['benchmark']}",
        f"# task: {task_cfg['task']}",
        f"# task_score_metric: {task_cfg['metrics']['task_score']}",
        f"# package_version: {provenance['package_version']}",
        f"# code_revision: {provenance['code_revision']}",
        f"# torch_version: {provenance['torch_version']}",
        "# promote by copying into configs/models/official/ and removing these comments.",
        "",
    ]
    body = yaml.safe_dump(model_cfg, sort_keys=False)
    path.write_text("\n".join(header) + body)

    # Also write a json study summary for quick inspection.
    summary = {
        "sweep_id": sweep_id,
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "sampler_seed": sampler_seed,
        "n_trials": n_trials,
        "benchmark": task_cfg["benchmark"],
        "task": task_cfg["task"],
        "task_score_metric": task_cfg["metrics"]["task_score"],
        "best_model_cfg": model_cfg,
    }
    (path.parent / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

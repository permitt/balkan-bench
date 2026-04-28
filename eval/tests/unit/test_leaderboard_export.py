"""Tests for leaderboard export assembler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from balkanbench.leaderboard.export import (
    ExportError,
    assemble_leaderboard,
    write_leaderboard_export,
)

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _artifact(
    *,
    task: str,
    model: str,
    model_id: str,
    params: int,
    primary_name: str,
    primary_value: float,
    rankable: bool = True,
) -> dict:
    return {
        "benchmark_name": "balkanbench",
        "benchmark_version": "0.1.0",
        "run_type": "official" if rankable else "experimental",
        "task_id": f"superglue.{task}.sr",
        "language": "sr",
        "model": model,
        "model_id": model_id,
        "model_revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "code_revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "dataset_revision": "v0.1.0-data",
        "image_digest": "sha256:" + "0" * 64,
        "config_hash": "sha256:" + "1" * 64,
        "selection_metric": primary_name,
        "hp_search": {
            "tool": "optuna",
            "sampler": "TPESampler",
            "sampler_seed": 42,
            "num_trials": 0,
            "search_space_id": "none",
        },
        "seeds": [42, 43, 44, 45, 46],
        "seed_results": [
            {"seed": s, "primary": {primary_name: primary_value}, "secondary": {}}
            for s in [42, 43, 44, 45, 46]
        ],
        "aggregate": {
            "mean": {primary_name: primary_value},
            "stdev": {primary_name: 0.01},
        },
        "task_score": primary_value,
        "rankable": rankable,
        "test_predictions_hash": "sha256:" + "2" * 64,
        "sponsor": "Recrewty",
    }


def _scaffold_official_dir(tmp_path: Path, rows: dict[str, dict[str, float]]) -> Path:
    """Layout: tmp/benchmark-lang/{model}/{task}/result.json.

    ``rows`` maps model -> {task: primary_metric_value}.
    """
    root = tmp_path / "superglue-sr"
    for model, task_scores in rows.items():
        for task, value in task_scores.items():
            artifact = _artifact(
                task=task,
                model=model,
                model_id=f"hf/{model}",
                params=110_000_000,
                primary_name=_PRIMARY_METRIC[task],
                primary_value=value,
            )
            task_dir = root / model / task
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "result.json").write_text(json.dumps(artifact))
    return root


_PRIMARY_METRIC = {
    "boolq": "accuracy",
    "cb": "f1_macro",
    "copa": "accuracy",
    "rte": "accuracy",
    "multirc": "f1_a",
    "wsc": "accuracy",
}

_ALL_TASKS = list(_PRIMARY_METRIC)


def test_assemble_leaderboard_all_complete(tmp_path) -> None:
    rows = {
        "bertic": {t: 0.75 + 0.02 * i for i, t in enumerate(_ALL_TASKS)},
        "xlmr": {t: 0.60 + 0.01 * i for i, t in enumerate(_ALL_TASKS)},
    }
    root = _scaffold_official_dir(tmp_path, rows)

    export = assemble_leaderboard(
        benchmark="superglue",
        language="sr",
        results_root=root,
        ranked_tasks=_ALL_TASKS,
        task_primary_metrics=_PRIMARY_METRIC,
        benchmark_version="0.1.0",
    )

    schema = json.loads((SCHEMAS_DIR / "leaderboard_export.json").read_text())
    Draft202012Validator(schema).validate(export)

    # Both rows complete; bertic ranked 1 (higher avg), xlmr ranked 2
    ranked = sorted(export["rows"], key=lambda r: r["rank"])
    assert [r["model"] for r in ranked] == ["bertic", "xlmr"]
    assert ranked[0]["complete"] is True
    assert ranked[0]["tasks_completed"] == 6


def test_partial_row_keeps_rank_null_and_partial_flag(tmp_path) -> None:
    # modernbertic-small is missing the last task (wsc)
    rows = {
        "bertic": {t: 0.8 for t in _ALL_TASKS},  # complete
        "modernbertic-small": {t: 0.7 for t in _ALL_TASKS[:-1]},  # missing wsc
    }
    root = _scaffold_official_dir(tmp_path, rows)

    export = assemble_leaderboard(
        benchmark="superglue",
        language="sr",
        results_root=root,
        ranked_tasks=_ALL_TASKS,
        task_primary_metrics=_PRIMARY_METRIC,
        benchmark_version="0.1.0",
    )
    partial = next(r for r in export["rows"] if r["model"] == "modernbertic-small")
    assert partial["rank"] is None
    assert partial["complete"] is False
    assert partial["tasks_completed"] == 5
    assert partial["partial_flag"] == "(5/6) partial"
    # The missing task must show up as None
    assert partial["results"]["wsc"] is None


def test_assemble_rejects_empty_results_root(tmp_path) -> None:
    with pytest.raises(ExportError):
        assemble_leaderboard(
            benchmark="superglue",
            language="sr",
            results_root=tmp_path / "does-not-exist",
            ranked_tasks=_ALL_TASKS,
            task_primary_metrics=_PRIMARY_METRIC,
            benchmark_version="0.1.0",
        )


def test_assemble_rejects_non_rankable_in_official_path(tmp_path) -> None:
    """If an artifact has rankable=false it must not count toward a rank."""
    root = tmp_path / "superglue-sr" / "experimental_model" / "boolq"
    root.mkdir(parents=True)
    artifact = _artifact(
        task="boolq",
        model="experimental_model",
        model_id="hf/experimental_model",
        params=1,
        primary_name="accuracy",
        primary_value=0.99,
        rankable=False,
    )
    (root / "result.json").write_text(json.dumps(artifact))

    export = assemble_leaderboard(
        benchmark="superglue",
        language="sr",
        results_root=tmp_path / "superglue-sr",
        ranked_tasks=_ALL_TASKS,
        task_primary_metrics=_PRIMARY_METRIC,
        benchmark_version="0.1.0",
    )
    row = next(iter(export["rows"]))
    # rankable=false => treated as partial (no rank) even when it would otherwise be #1
    assert row["rank"] is None


def test_write_leaderboard_export_round_trips(tmp_path) -> None:
    rows = {
        "bertic": {t: 0.78 for t in _ALL_TASKS},
    }
    root = _scaffold_official_dir(tmp_path, rows)
    out = tmp_path / "benchmark_results.json"

    write_leaderboard_export(
        benchmark="superglue",
        language="sr",
        results_root=root,
        ranked_tasks=_ALL_TASKS,
        task_primary_metrics=_PRIMARY_METRIC,
        benchmark_version="0.1.0",
        out_path=out,
    )

    data = json.loads(out.read_text())
    assert data["benchmark"] == "superglue"
    assert data["sponsor"] == "Recrewty"
    assert len(data["rows"]) == 1

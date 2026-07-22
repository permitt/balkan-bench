"""Tests for SLE-track leaderboard exports (access-split boards, Task 14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from balkanbench.cli.leaderboard import _collect_ranked_tasks
from balkanbench.cli.main import app
from balkanbench.evaluation.generative import GenerativeRunResult
from balkanbench.leaderboard.export import ExportError, assemble_leaderboard
from balkanbench.scoring.artifact import write_generative_result_artifact

runner = CliRunner()

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"

_RANKED_TASKS = ["arc_easy", "boolq"]
_OPEN_PRIMARY = {"arc_easy": "acc_norm", "boolq": "acc"}
_API_PRIMARY = {"arc_easy": "acc", "boolq": "acc"}


def _schema() -> dict:
    return json.loads((SCHEMAS_DIR / "leaderboard_export.json").read_text())


def _fake_provenance() -> dict:
    return {
        "code_revision": "d" * 40,
        "image_digest": "sha256:" + "0" * 64,
        "package_version": "0.1.0.dev0",
        "torch_version": "2.11.0",
        "transformers_version": "5.5.4",
        "python_version": "3.11.13",
    }


def _task_cfg(task: str) -> dict:
    open_primary = _OPEN_PRIMARY[task]
    api_primary = _API_PRIMARY[task]
    return {
        "benchmark": "sle",
        "task": task,
        "task_type": "multiple_choice_loglikelihood",
        "metrics": {
            "primary": [open_primary],
            "report": [open_primary],
            "task_score": open_primary,
        },
        "api_protocol": {
            "reformulation": "multiple_choice_generative",
            "metrics": {
                "primary": [api_primary],
                "report": [api_primary],
                "task_score": api_primary,
            },
        },
    }


def _model_cfg(name: str, *, access: str) -> dict:
    cfg: dict = {"name": name, "access": access}
    if access == "api":
        cfg["provider"] = "anthropic"
        cfg["api_model_id"] = f"{name}-api-id"
    else:
        cfg["hf_repo"] = f"permitt/{name}"
    return cfg


def _write_artifact(
    root: Path, *, task: str, model: str, access: str, metric: str, value: float
) -> None:
    write_generative_result_artifact(
        task_cfg=_task_cfg(task),
        model_cfg=_model_cfg(model, access=access),
        language="sr",
        run_result=GenerativeRunResult(
            metrics={metric: value},
            per_example=[{"example_id": "e1", "prediction": 0, "correct": True}],
            protocol="loglikelihood",
            num_fewshot=0,
            unparsed_responses=0,
            api_cost_usd=0.0 if access == "open_weights" else 0.05,
        ),
        task_score_metric=metric,
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=root,
    )


def _seed_tree(tmp_path: Path) -> Path:
    """Layout: tmp/sle-sr/{model}/{task}/result.json - 2 open + 2 api models,
    sharing one results tree (both access modes write to the same
    {benchmark}-{language} directory - see write_generative_result_artifact)."""
    for model, base in (("galton-open-a", 0.80), ("galton-open-b", 0.60)):
        for task in _RANKED_TASKS:
            _write_artifact(
                tmp_path,
                task=task,
                model=model,
                access="open_weights",
                metric=_OPEN_PRIMARY[task],
                value=base,
            )
    for model, base in (("claude-x", 0.70), ("gpt-y", 0.50)):
        for task in _RANKED_TASKS:
            _write_artifact(
                tmp_path,
                task=task,
                model=model,
                access="api",
                metric=_API_PRIMARY[task],
                value=base,
            )
    return tmp_path / "sle-sr"


def _assemble(root: Path, *, access: str) -> dict:
    primary = _OPEN_PRIMARY if access == "open_weights" else _API_PRIMARY
    return assemble_leaderboard(
        benchmark="sle",
        language="sr",
        results_root=root,
        ranked_tasks=_RANKED_TASKS,
        task_primary_metrics=primary,
        benchmark_version="0.1.0",
        access=access,
    )


def test_access_filter_splits_open_and_api_boards(tmp_path) -> None:
    root = _seed_tree(tmp_path)

    open_export = _assemble(root, access="open_weights")
    api_export = _assemble(root, access="api")

    Draft202012Validator(_schema()).validate(open_export)
    Draft202012Validator(_schema()).validate(api_export)

    assert {r["model"] for r in open_export["rows"]} == {"galton-open-a", "galton-open-b"}
    assert {r["model"] for r in api_export["rows"]} == {"claude-x", "gpt-y"}


def test_access_and_protocol_are_stamped(tmp_path) -> None:
    root = _seed_tree(tmp_path)

    open_export = _assemble(root, access="open_weights")
    api_export = _assemble(root, access="api")

    assert open_export["access"] == "open_weights"
    assert api_export["access"] == "api"
    assert open_export["protocol"] == "loglikelihood"
    assert api_export["protocol"] == "loglikelihood"


def test_generative_boards_omit_seeds(tmp_path) -> None:
    root = _seed_tree(tmp_path)
    open_export = _assemble(root, access="open_weights")
    api_export = _assemble(root, access="api")
    assert "seeds" not in open_export
    assert "seeds" not in api_export


def test_api_rows_have_null_params_and_api_display(tmp_path) -> None:
    root = _seed_tree(tmp_path)
    api_export = _assemble(root, access="api")
    assert api_export["rows"]
    for row in api_export["rows"]:
        assert row["params"] is None
        assert row["params_display"] == "API"


def test_open_rows_keep_integer_params(tmp_path) -> None:
    root = _seed_tree(tmp_path)
    open_export = _assemble(root, access="open_weights")
    assert open_export["rows"]
    for row in open_export["rows"]:
        assert isinstance(row["params"], int)


def test_ranks_assigned_independently_per_board(tmp_path) -> None:
    root = _seed_tree(tmp_path)
    open_export = _assemble(root, access="open_weights")
    api_export = _assemble(root, access="api")

    open_ranks = {r["model"]: r["rank"] for r in open_export["rows"]}
    api_ranks = {r["model"]: r["rank"] for r in api_export["rows"]}

    assert open_ranks["galton-open-a"] == 1
    assert open_ranks["galton-open-b"] == 2
    assert api_ranks["claude-x"] == 1
    assert api_ranks["gpt-y"] == 2


def test_conflicting_protocol_across_included_artifacts_raises(tmp_path) -> None:
    root = _seed_tree(tmp_path)
    _write_artifact(
        tmp_path,
        task="arc_easy",
        model="mixed-model",
        access="api",
        metric="acc",
        value=0.4,
    )
    artifact_path = root / "mixed-model" / "arc_easy" / "result.json"
    data = json.loads(artifact_path.read_text())
    data["run_config"]["protocol"] = "generative"
    artifact_path.write_text(json.dumps(data))

    with pytest.raises(ExportError):
        _assemble(root, access="api")


def test_mixed_seeds_presence_across_included_artifacts_raises(tmp_path) -> None:
    root = _seed_tree(tmp_path)
    _write_artifact(
        tmp_path,
        task="arc_easy",
        model="mixed-seeds-model",
        access="api",
        metric="acc",
        value=0.4,
    )
    artifact_path = root / "mixed-seeds-model" / "arc_easy" / "result.json"
    data = json.loads(artifact_path.read_text())
    data["seeds"] = [42]
    artifact_path.write_text(json.dumps(data))

    with pytest.raises(ExportError, match="disagree on presence of 'seeds'"):
        _assemble(root, access="api")


def test_run_config_absent_artifact_excluded_from_api_board(tmp_path) -> None:
    """An artifact with no run_config block defaults to open_weights (back
    compat) and is excluded from the access="api" board - only the api
    artifact's model row is included."""
    root = tmp_path / "sle-sr"
    root.mkdir()

    legacy_dir = root / "legacy-model" / "arc_easy"
    legacy_dir.mkdir(parents=True)
    legacy_artifact = {
        "benchmark_name": "balkanbench",
        "benchmark_version": "0.1.0",
        "run_type": "official",
        "task_id": "sle.arc_easy.sr",
        "language": "sr",
        "model": "legacy-model",
        "model_id": "hf/legacy-model",
        "model_revision": "a" * 40,
        "code_revision": "b" * 40,
        "dataset_revision": "v0.1.0-data",
        "image_digest": "sha256:" + "0" * 64,
        "config_hash": "sha256:" + "1" * 64,
        "selection_metric": "acc",
        "hp_search": {
            "tool": "optuna",
            "sampler": "TPESampler",
            "sampler_seed": 42,
            "num_trials": 0,
            "search_space_id": "none",
        },
        "seeds": [42],
        "seed_results": [{"seed": 42, "primary": {"acc": 0.8}, "secondary": {}}],
        "aggregate": {"mean": {"acc": 0.8}, "stdev": {"acc": 0.0}},
        "task_score": 0.8,
        "rankable": True,
        "test_predictions_hash": "sha256:" + "2" * 64,
        "sponsor": "Recrewty",
        "params": 110_000_000,
    }
    (legacy_dir / "result.json").write_text(json.dumps(legacy_artifact))

    _write_artifact(
        tmp_path,
        task="arc_easy",
        model="claude-x",
        access="api",
        metric="acc",
        value=0.7,
    )

    export = _assemble(root, access="api")
    assert {r["model"] for r in export["rows"]} == {"claude-x"}


def test_access_none_is_legacy_unfiltered_export(tmp_path) -> None:
    """Regression: access=None keeps the pre-Task-14 behavior byte-for-byte -
    no filtering, no protocol/access stamped, seeds always present. Covers the
    encoder (superglue) export path used elsewhere in the suite."""
    root = tmp_path / "sle-sr"
    root.mkdir()
    for model in ("bertic",):
        for task in _RANKED_TASKS:
            d = root / model / task
            d.mkdir(parents=True)
            artifact = {
                "benchmark_name": "balkanbench",
                "benchmark_version": "0.1.0",
                "run_type": "official",
                "task_id": f"superglue.{task}.sr",
                "language": "sr",
                "model": model,
                "model_id": f"hf/{model}",
                "model_revision": "a" * 40,
                "code_revision": "b" * 40,
                "dataset_revision": "v0.1.0-data",
                "image_digest": "sha256:" + "0" * 64,
                "config_hash": "sha256:" + "1" * 64,
                "selection_metric": "acc",
                "hp_search": {
                    "tool": "optuna",
                    "sampler": "TPESampler",
                    "sampler_seed": 42,
                    "num_trials": 0,
                    "search_space_id": "none",
                },
                "seeds": [42],
                "seed_results": [{"seed": 42, "primary": {"acc": 0.8}, "secondary": {}}],
                "aggregate": {"mean": {"acc": 0.8}, "stdev": {"acc": 0.0}},
                "task_score": 0.8,
                "rankable": True,
                "test_predictions_hash": "sha256:" + "2" * 64,
                "sponsor": "Recrewty",
                "params": 110_000_000,
            }
            (d / "result.json").write_text(json.dumps(artifact))

    export = assemble_leaderboard(
        benchmark="superglue",
        language="sr",
        results_root=root,
        ranked_tasks=_RANKED_TASKS,
        task_primary_metrics={t: "acc" for t in _RANKED_TASKS},
        benchmark_version="0.1.0",
    )
    Draft202012Validator(_schema()).validate(export)
    assert "access" not in export
    assert "protocol" not in export
    assert export["seeds"] == 5
    assert export["rows"][0]["params"] == 110_000_000


def test_collect_ranked_tasks_primary_metric_differs_per_access() -> None:
    """arc_easy is acc_norm on the open board (loglikelihood scoring) but acc
    on the api board (the api_protocol reformulation's task_score)."""
    _, open_primary = _collect_ranked_tasks("sle", "sr")
    _, api_primary = _collect_ranked_tasks("sle", "sr", access="api")

    assert open_primary["arc_easy"] == "acc_norm"
    assert api_primary["arc_easy"] == "acc"
    assert open_primary != api_primary


def test_cli_export_with_access_api(tmp_path) -> None:
    """End-to-end: `balkanbench leaderboard export --access api` filters to
    the api board, using the real sle task configs for the primary-metric
    resolution (exercises _collect_ranked_tasks(access="api") end to end)."""
    _seed_tree(tmp_path)
    out_path = tmp_path / "sle-sr-api" / "benchmark_results.json"

    result = runner.invoke(
        app,
        [
            "leaderboard",
            "export",
            "--benchmark",
            "sle",
            "--language",
            "sr",
            "--results-dir",
            str(tmp_path),
            "--out",
            str(out_path),
            "--access",
            "api",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    assert data["access"] == "api"
    assert {r["model"] for r in data["rows"]} == {"claude-x", "gpt-y"}
    for row in data["rows"]:
        assert row["params"] is None
        assert row["params_display"] == "API"
    # arc_easy's api-board primary metric is acc, not acc_norm.
    assert data["task_primary_metrics"]["arc_easy"] == "acc"


def test_cli_export_rejects_bad_access_value(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "leaderboard",
            "export",
            "--benchmark",
            "sle",
            "--language",
            "sr",
            "--results-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.json"),
            "--access",
            "bogus",
        ],
    )
    assert result.exit_code == 1

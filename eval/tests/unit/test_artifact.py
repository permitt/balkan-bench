"""Tests for the result artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from balkanbench.evaluation import Aggregate, SeedResult
from balkanbench.scoring.artifact import (
    compute_config_hash,
    compute_predictions_hash,
    write_result_artifact,
)

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _fake_provenance() -> dict:
    return {
        "code_revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "image_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "package_version": "0.1.0.dev0",
        "torch_version": "2.11.0",
        "transformers_version": "5.5.4",
        "python_version": "3.11.13",
    }


def _fake_seeds(values: list[float]) -> list[SeedResult]:
    out = []
    for i, v in enumerate(values):
        out.append(
            SeedResult(
                seed=42 + i,
                primary={"accuracy": v},
                secondary={},
                task_score=v,
                predictions=[0, 1, 0, 1],
                references=[0, 1, 0, 1],
                group_ids=None,
            )
        )
    return out


def _fake_task_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "boolq",
        "task_type": "binary_classification",
        "metrics": {"primary": ["accuracy"], "report": ["accuracy"], "task_score": "accuracy"},
    }


def _fake_model_cfg() -> dict:
    return {"name": "bertic", "hf_repo": "classla/bcms-bertic"}


def test_compute_predictions_hash_deterministic() -> None:
    h1 = compute_predictions_hash([0, 1, 0], ["a", "b", "c"])
    h2 = compute_predictions_hash([0, 1, 0], ["a", "b", "c"])
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_compute_predictions_hash_changes_when_predictions_change() -> None:
    h1 = compute_predictions_hash([0, 1, 0], ["a", "b", "c"])
    h2 = compute_predictions_hash([1, 1, 0], ["a", "b", "c"])
    assert h1 != h2


def test_compute_config_hash_is_stable() -> None:
    cfg = {"a": 1, "b": {"c": 2}}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash({"b": {"c": 2}, "a": 1})  # same content, different key order
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_write_result_artifact_is_schema_valid(tmp_path) -> None:
    seeds = _fake_seeds([0.78, 0.80, 0.79, 0.81, 0.77])
    agg = Aggregate(
        mean={"accuracy": sum(s.task_score for s in seeds) / len(seeds)},
        stdev={"accuracy": 0.015},
    )

    out_path = write_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=_fake_model_cfg(),
        language="sr",
        seed_results=seeds,
        aggregate=agg,
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        hp_search={
            "tool": "optuna",
            "sampler": "TPESampler",
            "sampler_seed": 42,
            "num_trials": 0,
            "search_space_id": "none",
            "early_stopping_policy": "patience=5 on accuracy",
        },
        out_dir=tmp_path,
    )

    # Path convention: eval/results/official/{benchmark}-{language}/{model}/result.json
    assert out_path.parent.name == "bertic"
    assert out_path.parent.parent.name == "superglue-sr"
    assert out_path.name == "result.json"

    data = json.loads(out_path.read_text())
    schema = json.loads((SCHEMAS_DIR / "result_artifact.json").read_text())
    Draft202012Validator(schema).validate(data)

    assert data["task_id"] == "superglue.boolq.sr"
    assert data["model"] == "bertic"
    assert data["model_id"] == "classla/bcms-bertic"
    assert data["sponsor"] == "Recrewty"
    assert data["rankable"] is True
    assert data["test_predictions_hash"].startswith("sha256:")


def test_write_result_artifact_marks_rankable_false_for_experimental(tmp_path) -> None:
    seeds = _fake_seeds([0.5])
    out_path = write_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg={**_fake_model_cfg(), "tier": "experimental"},
        language="sr",
        seed_results=seeds,
        aggregate=Aggregate(mean={"accuracy": 0.5}, stdev={"accuracy": 0.0}),
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        hp_search={
            "tool": "optuna",
            "sampler": "TPESampler",
            "sampler_seed": 42,
            "num_trials": 0,
            "search_space_id": "none",
        },
        out_dir=tmp_path,
        run_type="experimental",
    )
    data = json.loads(out_path.read_text())
    assert data["run_type"] == "experimental"
    assert data["rankable"] is False


def test_artifact_rejects_empty_seed_results(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_result_artifact(
            task_cfg=_fake_task_cfg(),
            model_cfg=_fake_model_cfg(),
            language="sr",
            seed_results=[],
            aggregate=Aggregate(mean={}, stdev={}),
            provenance=_fake_provenance(),
            dataset_revision="v0.1.0-data",
            benchmark_version="0.1.0",
            hp_search={
                "tool": "optuna",
                "sampler": "TPESampler",
                "sampler_seed": 42,
                "num_trials": 0,
                "search_space_id": "none",
            },
            out_dir=tmp_path,
        )

"""Validate benchmark manifests against `schemas/benchmark_spec.json`."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs" / "benchmarks"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "benchmark_spec.json").read_text())


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_valid_superglue_benchmark_passes() -> None:
    Draft202012Validator(_load_schema()).validate(_load_yaml("superglue_valid.yaml"))


def test_bad_version_fails() -> None:
    errors = list(
        Draft202012Validator(_load_schema()).iter_errors(
            _load_yaml("superglue_invalid_bad_version.yaml")
        )
    )
    assert errors
    assert any("version" in str(e.path) or "pattern" in e.message for e in errors)


def test_missing_maintainers_fails() -> None:
    doc = _load_yaml("superglue_valid.yaml")
    del doc["maintainers"]
    errors = list(Draft202012Validator(_load_schema()).iter_errors(doc))
    assert errors


def test_weighted_mean_accepts_weights() -> None:
    doc = _load_yaml("superglue_valid.yaml")
    doc["aggregation"] = {
        "formula": "weighted_mean",
        "over": "primary_task_scores",
        "weights": {"boolq": 1.0, "cb": 0.5, "copa": 1.0, "rte": 1.0, "multirc": 1.5, "wsc": 1.0},
    }
    Draft202012Validator(_load_schema()).validate(doc)

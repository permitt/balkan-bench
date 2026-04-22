"""Validate task YAMLs against `schemas/task_spec.json`."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs" / "tasks"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "task_spec.json").read_text())


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_valid_boolq_passes() -> None:
    Draft202012Validator(_load_schema()).validate(_load_yaml("boolq_valid.yaml"))


def test_invalid_boolq_missing_metrics_fails() -> None:
    validator = Draft202012Validator(_load_schema())
    errors = list(validator.iter_errors(_load_yaml("boolq_invalid_missing_metrics.yaml")))
    assert errors, "expected at least one schema error"
    assert any("metrics" in (e.message + str(e.path)) for e in errors)

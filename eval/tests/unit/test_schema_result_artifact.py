"""Validate per-run result artifacts against `schemas/result_artifact.json`."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "results"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "result_artifact.json").read_text())


def test_valid_artifact_passes() -> None:
    artifact = json.loads((FIXTURES / "bertic_boolq_sr_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(artifact)


def test_bad_hash_format_fails() -> None:
    artifact = json.loads((FIXTURES / "bertic_boolq_sr_valid.json").read_text())
    artifact["test_predictions_hash"] = "not-a-hash"
    errors = list(Draft202012Validator(_load_schema()).iter_errors(artifact))
    assert errors
    assert any("test_predictions_hash" in str(e.path) or "pattern" in e.message for e in errors)


def test_missing_sponsor_fails() -> None:
    artifact = json.loads((FIXTURES / "bertic_boolq_sr_valid.json").read_text())
    del artifact["sponsor"]
    errors = list(Draft202012Validator(_load_schema()).iter_errors(artifact))
    assert errors

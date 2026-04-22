"""Validate dataset manifests against `schemas/dataset_manifest.json`."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "manifests"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "dataset_manifest.json").read_text())


def test_valid_manifest_passes() -> None:
    manifest = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(manifest)


def test_missing_hidden_test_labels_fails() -> None:
    manifest = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    del manifest["hidden_test_labels"]
    errors = list(Draft202012Validator(_load_schema()).iter_errors(manifest))
    assert errors
    assert any("hidden_test_labels" in e.message for e in errors)

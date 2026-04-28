"""Validate model YAMLs against `schemas/model_spec.json`."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs" / "models"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "model_spec.json").read_text())


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_valid_bertic_passes() -> None:
    Draft202012Validator(_load_schema()).validate(_load_yaml("bertic_valid.yaml"))


def test_missing_hf_repo_fails() -> None:
    errors = list(
        Draft202012Validator(_load_schema()).iter_errors(
            _load_yaml("bertic_invalid_no_hf_repo.yaml")
        )
    )
    assert errors
    assert any("hf_repo" in (e.message + str(e.path)) for e in errors)


def test_bare_task_override_keys_rejected() -> None:
    errors = list(
        Draft202012Validator(_load_schema()).iter_errors(
            _load_yaml("bertic_invalid_bare_override_keys.yaml")
        )
    )
    assert errors, "task_overrides must use {benchmark}.{task} namespaced keys"

"""Validate submission metadata JSON."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "submissions"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "submission_metadata.json").read_text())


def test_valid_submission_passes() -> None:
    submission = json.loads((FIXTURES / "example_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(submission)


def test_bad_identity_provider_fails() -> None:
    submission = json.loads((FIXTURES / "example_valid.json").read_text())
    submission["submitter"]["identity"]["provider"] = "twitter"
    errors = list(Draft202012Validator(_load_schema()).iter_errors(submission))
    assert errors


def test_bad_package_hash_fails() -> None:
    submission = json.loads((FIXTURES / "example_valid.json").read_text())
    submission["predictions_package"]["sha256"] = "deadbeef"
    errors = list(Draft202012Validator(_load_schema()).iter_errors(submission))
    assert errors

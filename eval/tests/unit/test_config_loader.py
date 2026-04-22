"""Tests for the unified YAML + JSON Schema loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from balkanbench.config import ConfigError, load_yaml_with_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = REPO_ROOT / "eval" / "schemas"
FIXTURES = REPO_ROOT / "eval" / "tests" / "fixtures" / "configs"


def test_loads_valid_task_yaml() -> None:
    cfg = load_yaml_with_schema(
        FIXTURES / "tasks" / "boolq_valid.yaml",
        SCHEMAS_DIR / "task_spec.json",
    )
    assert cfg["benchmark"] == "superglue"
    assert cfg["task"] == "boolq"
    assert cfg["metrics"]["task_score"] == "accuracy"


def test_rejects_invalid_task_yaml() -> None:
    with pytest.raises(ConfigError) as exc:
        load_yaml_with_schema(
            FIXTURES / "tasks" / "boolq_invalid_missing_metrics.yaml",
            SCHEMAS_DIR / "task_spec.json",
        )
    assert "metrics" in str(exc.value)


def test_reports_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_yaml_with_schema(
            FIXTURES / "tasks" / "does_not_exist.yaml",
            SCHEMAS_DIR / "task_spec.json",
        )

"""Smoke tests for `balkanbench validate-config` and `validate-data`."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "eval" / "tests" / "fixtures"


def test_validate_config_accepts_valid_task_yaml() -> None:
    result = runner.invoke(
        app,
        [
            "validate-config",
            str(FIXTURES / "configs" / "tasks" / "boolq_valid.yaml"),
            "--schema",
            "task_spec",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_validate_config_rejects_invalid_task_yaml() -> None:
    result = runner.invoke(
        app,
        [
            "validate-config",
            str(FIXTURES / "configs" / "tasks" / "boolq_invalid_missing_metrics.yaml"),
            "--schema",
            "task_spec",
        ],
    )
    assert result.exit_code == 1


def test_validate_data_accepts_valid_manifest() -> None:
    result = runner.invoke(
        app,
        ["validate-data", str(FIXTURES / "manifests" / "superglue_sr_valid.json")],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output

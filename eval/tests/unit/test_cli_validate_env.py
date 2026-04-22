"""Smoke tests for `balkanbench validate-env`."""

from __future__ import annotations

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def test_validate_env_runs() -> None:
    result = runner.invoke(app, ["validate-env"])
    assert result.exit_code == 0, result.output
    assert "python" in result.stdout.lower()


def test_validate_env_reports_hf_token_absence(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_OFFICIAL_TOKEN", raising=False)
    result = runner.invoke(app, ["validate-env"])
    assert result.exit_code == 0
    assert "hf_token" in result.stdout.lower() or "huggingface" in result.stdout.lower()

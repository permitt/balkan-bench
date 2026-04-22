"""Smoke tests for `balkanbench list`."""
from __future__ import annotations

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def test_list_benchmarks_runs_even_with_no_configs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "benchmarks"])
    assert result.exit_code == 0, result.output
    assert "no benchmarks" in result.stdout.lower() or result.stdout.strip() == ""


def test_list_models_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "models"])
    assert result.exit_code == 0


def test_list_tasks_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "tasks"])
    assert result.exit_code == 0


def test_list_languages_returns_sr_for_v01(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "languages"])
    assert result.exit_code == 0
    assert "sr" in result.stdout

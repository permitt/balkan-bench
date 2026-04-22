"""Smoke tests for the typer CLI skeleton."""
from __future__ import annotations

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "0.1.0.dev0" in result.stdout


def test_help_mentions_balkanbench() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "balkanbench" in result.stdout.lower()


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.stdout or "Usage" in result.output

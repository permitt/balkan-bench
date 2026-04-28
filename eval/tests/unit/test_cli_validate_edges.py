"""Edge cases for the validate-* commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "eval" / "tests" / "fixtures"


def test_validate_data_rejects_broken_manifest(tmp_path) -> None:
    broken = tmp_path / "broken_manifest.json"
    broken.write_text('{"benchmark": "superglue"}')
    result = runner.invoke(app, ["validate-data", str(broken)])
    assert result.exit_code == 1
    assert "failed schema" in result.output or "required" in result.output.lower()


def test_validate_env_hard_fails_when_import_missing(monkeypatch) -> None:
    import importlib

    original_import = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "pydantic":
            raise ImportError("simulated")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    result = runner.invoke(app, ["validate-env"])
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_validate_env_shows_present_env_var(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_test")
    result = runner.invoke(app, ["validate-env"])
    assert result.exit_code == 0
    assert "HF_TOKEN" in result.output
    assert "present" in result.output.lower()


def test_validate_config_against_model_spec_fixture() -> None:
    result = runner.invoke(
        app,
        [
            "validate-config",
            str(FIXTURES / "configs" / "models" / "bertic_valid.yaml"),
            "--schema",
            "model_spec",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output

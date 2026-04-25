"""Tests for `balkanbench list` when configs are present."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def _write_yaml(path: Path, content: str = "placeholder: true\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_list_benchmarks_shows_benchmark_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    (tmp_path / "benchmarks" / "superglue").mkdir(parents=True)
    (tmp_path / "benchmarks" / "sle").mkdir(parents=True)

    result = runner.invoke(app, ["list", "benchmarks"])
    assert result.exit_code == 0, result.output
    assert "superglue" in result.stdout
    assert "sle" in result.stdout


def test_list_tasks_namespaces_by_benchmark(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    _write_yaml(tmp_path / "benchmarks" / "superglue" / "tasks" / "boolq.yaml")
    _write_yaml(tmp_path / "benchmarks" / "superglue" / "tasks" / "cb.yaml")

    result = runner.invoke(app, ["list", "tasks"])
    assert result.exit_code == 0, result.output
    assert "superglue.boolq" in result.stdout
    assert "superglue.cb" in result.stdout


def test_list_models_reports_tier(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    _write_yaml(tmp_path / "models" / "official" / "bertic.yaml")
    _write_yaml(tmp_path / "models" / "experimental" / "mystery.yaml")

    result = runner.invoke(app, ["list", "models"])
    assert result.exit_code == 0, result.output
    assert "bertic" in result.stdout
    assert "official" in result.stdout
    assert "mystery" in result.stdout
    assert "experimental" in result.stdout


def test_list_tasks_empty_benchmark_still_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    (tmp_path / "benchmarks" / "superglue").mkdir(parents=True)
    result = runner.invoke(app, ["list", "tasks"])
    assert result.exit_code == 0
    assert "no tasks" in result.stdout.lower()


def test_configs_root_defaults_when_env_unset(monkeypatch) -> None:
    from balkanbench.cli.listcmd import _configs_root

    monkeypatch.delenv("BALKANBENCH_CONFIGS_DIR", raising=False)
    root = _configs_root()
    assert root.name == "configs"


def test_list_languages_discovers_from_task_yamls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    task_yaml = tmp_path / "benchmarks" / "superglue" / "tasks" / "boolq.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text("languages:\n  available: [sr]\n  ranked: [sr]\n  roadmap: [hr, mne]\n")
    result = runner.invoke(app, ["list", "languages"])
    assert result.exit_code == 0, result.output
    assert "sr\tavailable" in result.stdout
    assert "hr\troadmap" in result.stdout
    assert "mne\troadmap" in result.stdout


def test_list_languages_merges_across_benchmarks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    sg = tmp_path / "benchmarks" / "superglue" / "tasks" / "boolq.yaml"
    sg.parent.mkdir(parents=True)
    sg.write_text("languages:\n  available: [sr]\n  ranked: [sr]\n")
    sle = tmp_path / "benchmarks" / "sle" / "tasks" / "arc_challenge.yaml"
    sle.parent.mkdir(parents=True)
    sle.write_text("languages:\n  available: [sr, hr]\n  ranked: [sr]\n")
    result = runner.invoke(app, ["list", "languages"])
    assert result.exit_code == 0, result.output
    assert "sr\tavailable" in result.stdout
    assert "hr\tavailable" in result.stdout

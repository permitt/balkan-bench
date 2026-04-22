"""Smoke tests for `balkanbench publish-dataset`."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict
from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def _tiny_source() -> dict[str, DatasetDict]:
    def mk(prefix: str, n: int, with_label: bool) -> Dataset:
        data: dict[str, list] = {
            "question": [f"q{i}" for i in range(n)],
            "passage": [f"p{i}" for i in range(n)],
        }
        if with_label:
            data["label"] = [i % 2 for i in range(n)]
        return Dataset.from_dict(data)

    boolq = DatasetDict(
        {
            "train": mk("b-train", 4, True),
            "validation": mk("b-val", 2, True),
            "test": mk("b-test", 2, True),
        }
    )
    return {"boolq": boolq}


def test_publish_dataset_dry_run_prints_summary(monkeypatch) -> None:
    source = _tiny_source()

    def fake_load(_: str, config: str, **__: Any) -> DatasetDict:
        return source[config]

    monkeypatch.setattr("balkanbench.data.publish.load_dataset", fake_load)
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")

    result = runner.invoke(
        app,
        [
            "publish-dataset",
            "--source-repo",
            "permitt/superglue",
            "--public-repo",
            "permitt/superglue-serbian",
            "--private-repo",
            "permitt/superglue-private",
            "--language",
            "sr",
            "--license",
            "CC-BY-4.0",
            "--dataset-revision",
            "v0.1.0-data",
            "--config",
            "boolq",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output.lower()
    assert "boolq" in result.output
    assert "Recrewty" in result.output or "CC-BY-4.0" in result.output


def test_publish_dataset_fails_without_token(monkeypatch) -> None:
    source = _tiny_source()

    def fake_load(_: str, config: str, **__: Any) -> DatasetDict:
        return source[config]

    monkeypatch.setattr("balkanbench.data.publish.load_dataset", fake_load)
    monkeypatch.delenv("HF_OFFICIAL_TOKEN", raising=False)

    result = runner.invoke(
        app,
        [
            "publish-dataset",
            "--source-repo",
            "permitt/superglue",
            "--public-repo",
            "permitt/superglue-serbian",
            "--language",
            "sr",
            "--license",
            "CC-BY-4.0",
            "--dataset-revision",
            "v0.1.0-data",
            "--config",
            "boolq",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "HF_OFFICIAL_TOKEN" in result.output

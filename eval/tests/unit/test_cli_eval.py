"""Smoke tests for `balkanbench eval`."""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset, DatasetDict
from typer.testing import CliRunner

from balkanbench.cli.main import app
from balkanbench.evaluation import SeedResult

runner = CliRunner()


def _fake_datasets() -> DatasetDict:
    train = Dataset.from_dict(
        {
            "example_id": ["t0", "t1"],
            "question": ["q0", "q1"],
            "passage": ["p0", "p1"],
            "label": [0, 1],
        }
    )
    val = Dataset.from_dict(
        {
            "example_id": ["v0", "v1"],
            "question": ["qv0", "qv1"],
            "passage": ["pv0", "pv1"],
            "label": [1, 0],
        }
    )
    return DatasetDict({"train": train, "validation": val})


def test_eval_command_writes_result_artifact(tmp_path, monkeypatch) -> None:
    def fake_run_multiseed(**kwargs: Any) -> list[SeedResult]:
        return [
            SeedResult(
                seed=42,
                primary={"accuracy": 0.77},
                secondary={},
                task_score=0.77,
                predictions=[0, 1],
                references=[0, 1],
                group_ids=None,
            ),
            SeedResult(
                seed=43,
                primary={"accuracy": 0.80},
                secondary={},
                task_score=0.80,
                predictions=[1, 1],
                references=[0, 1],
                group_ids=None,
            ),
        ]

    monkeypatch.setattr("balkanbench.cli.eval.run_multiseed", fake_run_multiseed)
    monkeypatch.setattr(
        "balkanbench.cli.eval.load_dataset",
        lambda repo, config, **_: _fake_datasets(),
    )
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "bertic",
            "--benchmark",
            "superglue",
            "--task",
            "boolq",
            "--language",
            "sr",
            "--seeds",
            "42",
            "--seeds",
            "43",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    artifact_path = tmp_path / "superglue-sr" / "bertic" / "result.json"
    assert artifact_path.is_file()
    data = json.loads(artifact_path.read_text())
    assert data["task_id"] == "superglue.boolq.sr"
    assert data["seeds"] == [42, 43]
    assert data["sponsor"] == "Recrewty"


def test_eval_uses_model_config_seeds_when_seeds_flag_omitted(tmp_path, monkeypatch) -> None:
    captured_seeds: list[int] = []

    def fake_run_multiseed(**kwargs: Any) -> list[SeedResult]:
        captured_seeds.extend(kwargs["seeds"])
        return [
            SeedResult(
                seed=s,
                primary={"accuracy": 0.7 + 0.01 * s},
                secondary={},
                task_score=0.7 + 0.01 * s,
                predictions=[0, 1],
                references=[0, 1],
                group_ids=None,
            )
            for s in kwargs["seeds"]
        ]

    monkeypatch.setattr("balkanbench.cli.eval.run_multiseed", fake_run_multiseed)
    monkeypatch.setattr(
        "balkanbench.cli.eval.load_dataset",
        lambda repo, config, **_: _fake_datasets(),
    )
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "bertic",
            "--benchmark",
            "superglue",
            "--task",
            "boolq",
            "--language",
            "sr",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # bertic.yaml declares seeds [42, 43, 44, 45, 46]
    assert captured_seeds == [42, 43, 44, 45, 46]


def test_eval_rejects_unknown_model(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "not_a_real_model",
            "--benchmark",
            "superglue",
            "--task",
            "boolq",
            "--language",
            "sr",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()

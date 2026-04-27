"""Smoke tests for `balkanbench predict`."""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset, DatasetDict
from typer.testing import CliRunner

from balkanbench.cli.main import app
from balkanbench.evaluation import SeedResult

runner = CliRunner()


def _fake_test_dataset() -> DatasetDict:
    test = Dataset.from_dict(
        {
            # Real datasets expose the id under `idx` (matches the published
            # SuperGLUE-BCMS schema and the boolq.yaml id_field).
            "idx": [0, 1, 2],
            "question": ["q0", "q1", "q2"],
            "passage": ["p0", "p1", "p2"],
        }
    )
    return DatasetDict({"test": test})


def test_predict_emits_predictions_jsonl(tmp_path, monkeypatch) -> None:
    captured_kwargs: dict[str, Any] = {}
    captured_repo: dict[str, str] = {}

    def fake_run_single_seed(**kwargs: Any) -> SeedResult:
        captured_kwargs.update(kwargs)
        # Predict always class 1 for 3 test examples
        return SeedResult(
            seed=42,
            primary={},
            secondary={},
            task_score=0.0,
            predictions=[1, 1, 1],
            references=[0, 0, 0],  # placeholder - predict shouldn't use labels
            group_ids=None,
        )

    monkeypatch.setattr("balkanbench.cli.predict.run_single_seed", fake_run_single_seed)
    def fake_load_dataset(repo: str, config: str, **_: Any) -> DatasetDict:
        captured_repo["repo"] = repo
        return _fake_test_dataset()

    monkeypatch.setattr("balkanbench.cli.predict.load_dataset", fake_load_dataset)

    result = runner.invoke(
        app,
        [
            "predict",
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

    preds_path = tmp_path / "predictions.jsonl"
    meta_path = tmp_path / "run_metadata.json"
    assert preds_path.is_file()
    assert meta_path.is_file()

    lines = preds_path.read_text().strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert "example_id" in first
    assert "prediction" in first

    meta = json.loads(meta_path.read_text())
    assert meta["benchmark"] == "superglue"
    assert meta["task"] == "boolq"
    assert meta["language"] == "sr"
    assert meta["num_predictions"] == 3

    # Regression guard for the review finding: predict must opt into the
    # no-train + no-metrics path. Otherwise the evaluator will try to score
    # the unlabeled public test split.
    assert captured_kwargs["compute_metrics"] is False, (
        "predict CLI must pass compute_metrics=False (unlabeled public test split)"
    )
    assert captured_kwargs["train"] is False, (
        "predict CLI must pass train=False (no train split, no training wanted)"
    )
    assert captured_kwargs["eval_split"] == "test"
    assert captured_repo["repo"] == "permitt/superglue-sr"

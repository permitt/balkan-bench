"""Smoke test for `balkanbench throughput`."""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset, DatasetDict
from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def _val(n: int = 32) -> DatasetDict:
    return DatasetDict(
        {
            "validation": Dataset.from_dict(
                {
                    "example_id": [f"v{i}" for i in range(n)],
                    "question": [f"q{i}" for i in range(n)],
                    "passage": [f"p{i}" for i in range(n)],
                    "label": [i % 2 for i in range(n)],
                }
            )
        }
    )


class _FakeEncoder:
    model = object()

    class _Tok:
        def __call__(self, *args, **kwargs):
            return {"input_ids": [1] * 16, "attention_mask": [1] * 16}

    tokenizer = _Tok()
    training_args: dict[str, Any] = {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 1,
        "metric_for_best_model": "accuracy",
    }


def test_throughput_cli_writes_per_task_and_aggregate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "balkanbench.cli.throughput.load_dataset",
        lambda repo, config, **_: _val(32),
    )
    monkeypatch.setattr(
        "balkanbench.cli.throughput.HFEncoder.build",
        lambda model_cfg, task_cfg: _FakeEncoder(),
    )

    # Deterministic per-batch latency: 0.01s for every batch -> 1600 ex/sec @ bs=16.
    # The closure in throughput_cmd passes task_type + num_choices through to the
    # default_predict_fn, so fakes must accept (or ignore) them.
    def fake_predict(model, batch, *, batch_size: int, max_seq_len: int, **_):
        return [0] * batch_size, 0.01

    monkeypatch.setattr("balkanbench.cli.throughput.default_predict_fn", fake_predict)

    result = runner.invoke(
        app,
        [
            "throughput",
            "--model",
            "bertic",
            "--benchmark",
            "superglue",
            "--language",
            "sr",
            "--task",
            "boolq",
            "--hardware",
            "NVIDIA L4 24GB",
            "--precision",
            "fp16",
            "--out",
            str(tmp_path),
            "--warmup-batches",
            "0",
            "--measurement-batches",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    task_path = tmp_path / "superglue-sr" / "bertic" / "throughput" / "boolq.json"
    aggregate_path = tmp_path / "superglue-sr" / "bertic" / "throughput.json"
    assert task_path.is_file()
    assert aggregate_path.is_file()

    task_data = json.loads(task_path.read_text())
    assert task_data["benchmark"] == "superglue"
    assert task_data["task"] == "boolq"
    assert task_data["sponsor"] == "Recrewty"
    assert task_data["throughput_ex_per_sec"] > 0

    agg = json.loads(aggregate_path.read_text())
    assert set(agg["tasks"]) == {"boolq"}
    assert agg["mean_ex_per_sec"] == task_data["throughput_ex_per_sec"]


def test_throughput_cli_sweeps_every_ranked_task_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "balkanbench.cli.throughput.load_dataset",
        lambda repo, config, **_: _val(32),
    )
    monkeypatch.setattr(
        "balkanbench.cli.throughput.HFEncoder.build",
        lambda model_cfg, task_cfg: _FakeEncoder(),
    )

    def fake_predict(model, batch, *, batch_size: int, max_seq_len: int, **_):
        return [0] * batch_size, 0.02

    monkeypatch.setattr("balkanbench.cli.throughput.default_predict_fn", fake_predict)

    result = runner.invoke(
        app,
        [
            "throughput",
            "--model",
            "bertic",
            "--benchmark",
            "superglue",
            "--language",
            "sr",
            "--hardware",
            "NVIDIA L4 24GB",
            "--precision",
            "fp16",
            "--out",
            str(tmp_path),
            "--warmup-batches",
            "0",
            "--measurement-batches",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    throughput_root = tmp_path / "superglue-sr" / "bertic" / "throughput"
    written = sorted(p.stem for p in throughput_root.glob("*.json"))
    # All 6 ranked SuperGLUE tasks should have per-task artifacts
    assert set(written) == {"boolq", "cb", "copa", "rte", "multirc", "wsc"}

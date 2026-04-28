"""Tests for the throughput measurement loop + artifact writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset
from jsonschema import Draft202012Validator

from balkanbench.throughput.measure import (
    ThroughputSample,
    measure_task_throughput,
)
from balkanbench.throughput.writer import (
    write_model_throughput_aggregate,
    write_task_throughput,
)

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


class _FakeTokenizer:
    """Fake tokenizer that returns fixed-length ids."""

    def __call__(self, *args, **kwargs) -> dict:
        return {
            "input_ids": list(range(32)),
            "attention_mask": [1] * 32,
        }


def _boolq_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "boolq",
        "task_type": "binary_classification",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "config": "boolq",
            "per_language": {
                "sr": {
                    "public_repo": "permitt/superglue-sr",
                    "private_repo": "permitt/superglue-sr-private",
                }
            },
            "splits": {"public": ["validation"], "labeled_public": ["validation"]},
        },
        "inputs": {"fields": ["question", "passage"], "id_field": "example_id"},
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "boolq_sr_v1"}},
        "tokenizer": {"max_length": 128, "padding": "longest"},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


def _val_dataset(n: int = 32) -> Dataset:
    return Dataset.from_dict(
        {
            "example_id": [f"v{i}" for i in range(n)],
            "question": [f"q{i}" for i in range(n)],
            "passage": [f"p{i}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )


def _fake_predict_fn_constant(seconds_per_batch: float):
    """Return a predict_fn that reports a fixed wall-clock per batch."""

    def predict_fn(model, batch, *, batch_size: int, max_seq_len: int) -> tuple[list, float]:
        return [0] * batch_size, seconds_per_batch

    return predict_fn


def test_measure_task_throughput_median_of_timings() -> None:
    cfg = _boolq_cfg()
    timings = [0.010, 0.010, 0.020, 0.010, 0.010]  # median = 0.010s per batch

    calls = iter(timings)

    def predict_fn(model, batch, *, batch_size: int, max_seq_len: int) -> tuple[list, float]:
        return [0] * batch_size, next(calls)

    sample = measure_task_throughput(
        model=object(),
        tokenizer=_FakeTokenizer(),
        task_cfg=cfg,
        dataset=_val_dataset(80),
        language="sr",
        hardware="NVIDIA L4 24GB",
        precision="fp16",
        warmup_batches=0,
        measurement_batches=5,
        predict_fn=predict_fn,
    )
    # median batch latency = 0.010s at batch_size=16 -> 1600 ex/sec
    assert sample.batch_size == 16
    assert sample.throughput_ex_per_sec == pytest.approx(1600.0, rel=1e-3)
    assert sample.throughput_tok_per_sec == pytest.approx(1600.0 * sample.max_seq_len, rel=1e-3)
    assert sample.measurement_batches == 5


def test_measure_discards_warmup_batches() -> None:
    cfg = _boolq_cfg()
    # First two batches (warmup) are very slow; measurement batches are fast.
    # If warmup was counted, throughput would be much lower.
    timings = [0.500, 0.500, 0.010, 0.010, 0.010]
    calls = iter(timings)

    def predict_fn(model, batch, *, batch_size: int, max_seq_len: int) -> tuple[list, float]:
        return [0] * batch_size, next(calls)

    sample = measure_task_throughput(
        model=object(),
        tokenizer=_FakeTokenizer(),
        task_cfg=cfg,
        dataset=_val_dataset(80),
        language="sr",
        hardware="NVIDIA L4 24GB",
        precision="fp16",
        warmup_batches=2,
        measurement_batches=3,
        predict_fn=predict_fn,
    )
    # Only the 0.010s batches are counted: median = 0.010s -> 1600 ex/sec
    assert sample.throughput_ex_per_sec == pytest.approx(1600.0, rel=1e-3)


def test_measure_respects_full_pass_cap() -> None:
    """If the dataset has fewer batches than measurement_batches asks for, stop early."""
    cfg = _boolq_cfg()
    cfg["training"]["batch_size"] = 16
    # 16 rows, batch_size 16 -> one full batch
    tiny = _val_dataset(16)
    predict_fn = _fake_predict_fn_constant(0.010)

    sample = measure_task_throughput(
        model=object(),
        tokenizer=_FakeTokenizer(),
        task_cfg=cfg,
        dataset=tiny,
        language="sr",
        hardware="NVIDIA L4 24GB",
        precision="fp16",
        warmup_batches=0,
        measurement_batches=50,
        predict_fn=predict_fn,
    )
    assert sample.measurement_batches == 1


def test_measure_partial_final_batch_does_not_inflate_throughput() -> None:
    """A tail batch shorter than batch_size must be counted by its actual size.

    24 rows at batch_size=16 yields one full batch of 16 and one partial batch
    of 8 examples. If both see the same per-batch latency (0.010s), the full
    batch runs at 1600 ex/s and the partial runs at 800 ex/s; the median
    across per-batch throughputs is 1200 ex/s, not the 1600 ex/s we would
    report if we naively divided the configured batch_size by the median
    latency.
    """
    cfg = _boolq_cfg()
    predict_fn = _fake_predict_fn_constant(0.010)
    sample = measure_task_throughput(
        model=object(),
        tokenizer=_FakeTokenizer(),
        task_cfg=cfg,
        dataset=_val_dataset(24),
        language="sr",
        hardware="NVIDIA L4 24GB",
        precision="fp16",
        warmup_batches=0,
        measurement_batches=50,
        predict_fn=predict_fn,
    )
    assert sample.measurement_batches == 2
    assert sample.throughput_ex_per_sec == pytest.approx(1200.0, rel=1e-3)


def test_write_task_throughput_matches_schema(tmp_path) -> None:
    sample = ThroughputSample(
        batch_size=16,
        max_seq_len=128,
        throughput_ex_per_sec=1600.0,
        throughput_tok_per_sec=1600.0 * 128,
        peak_vram_mib=4820.0,
        hardware="NVIDIA L4 24GB",
        precision="fp16",
        warmup_batches=2,
        measurement_batches=50,
        torch_version="2.11.0",
        driver_version="535.129.03",
    )
    path = write_task_throughput(
        sample=sample,
        out_dir=tmp_path,
        task="boolq",
        model="bertic",
        model_id="classla/bcms-bertic",
        benchmark="superglue",
        language="sr",
    )
    assert path.is_file()
    data = json.loads(path.read_text())
    schema = json.loads((SCHEMAS_DIR / "task_throughput.json").read_text())
    Draft202012Validator(schema).validate(data)
    assert data["task"] == "boolq"
    assert data["sponsor"] == "Recrewty"


def test_write_model_aggregate_computes_mean(tmp_path) -> None:
    samples = [
        (
            "boolq",
            ThroughputSample(
                batch_size=16,
                max_seq_len=128,
                throughput_ex_per_sec=1600.0,
                throughput_tok_per_sec=204800.0,
                peak_vram_mib=4000,
                hardware="NVIDIA L4 24GB",
                precision="fp16",
                warmup_batches=2,
                measurement_batches=50,
                torch_version="2.11.0",
                driver_version="535",
            ),
        ),
        (
            "cb",
            ThroughputSample(
                batch_size=16,
                max_seq_len=128,
                throughput_ex_per_sec=800.0,
                throughput_tok_per_sec=102400.0,
                peak_vram_mib=5000,
                hardware="NVIDIA L4 24GB",
                precision="fp16",
                warmup_batches=2,
                measurement_batches=50,
                torch_version="2.11.0",
                driver_version="535",
            ),
        ),
    ]
    path = write_model_throughput_aggregate(
        samples=samples,
        out_dir=tmp_path,
        model="bertic",
        model_id="classla/bcms-bertic",
        benchmark="superglue",
        language="sr",
        hardware="NVIDIA L4 24GB",
        precision="fp16",
    )
    data = json.loads(path.read_text())
    assert data["mean_ex_per_sec"] == pytest.approx(1200.0)
    assert data["max_peak_vram_mib"] == 5000
    assert set(data["tasks"]) == {"boolq", "cb"}

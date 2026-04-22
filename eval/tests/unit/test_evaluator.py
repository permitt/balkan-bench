"""Tests for the evaluator: aggregate + run_single_seed + run_multiseed."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from balkanbench.evaluation import (
    Aggregate,
    SeedResult,
    aggregate_seed_results,
    run_multiseed,
    run_single_seed,
)

# ---------- aggregation ----------


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_seed_results([])


def test_aggregate_single_seed_stdev_zero() -> None:
    seed = SeedResult(
        seed=42,
        primary={"accuracy": 0.8},
        secondary={},
        task_score=0.8,
        predictions=[0, 1],
        references=[0, 1],
        group_ids=None,
    )
    agg = aggregate_seed_results([seed])
    assert agg.mean == {"accuracy": 0.8}
    assert agg.stdev == {"accuracy": 0.0}


def test_aggregate_multi_seed_mean_and_stdev() -> None:
    seeds = [
        SeedResult(
            seed=42,
            primary={"accuracy": 0.8},
            secondary={},
            task_score=0.8,
            predictions=[],
            references=[],
            group_ids=None,
        ),
        SeedResult(
            seed=43,
            primary={"accuracy": 0.9},
            secondary={},
            task_score=0.9,
            predictions=[],
            references=[],
            group_ids=None,
        ),
        SeedResult(
            seed=44,
            primary={"accuracy": 0.85},
            secondary={},
            task_score=0.85,
            predictions=[],
            references=[],
            group_ids=None,
        ),
    ]
    agg = aggregate_seed_results(seeds)
    assert agg.mean["accuracy"] == pytest.approx(0.85)
    # Sample stdev (N-1) of [0.8, 0.9, 0.85] ~= 0.05
    assert agg.stdev["accuracy"] == pytest.approx(math.sqrt(0.0025), abs=1e-4)


def test_aggregate_multiple_metrics() -> None:
    seeds = [
        SeedResult(
            seed=42,
            primary={"accuracy": 0.8, "f1_macro": 0.78},
            secondary={},
            task_score=0.8,
            predictions=[],
            references=[],
            group_ids=None,
        ),
        SeedResult(
            seed=43,
            primary={"accuracy": 0.9, "f1_macro": 0.88},
            secondary={},
            task_score=0.9,
            predictions=[],
            references=[],
            group_ids=None,
        ),
    ]
    agg = aggregate_seed_results(seeds)
    assert agg.mean["accuracy"] == pytest.approx(0.85)
    assert agg.mean["f1_macro"] == pytest.approx(0.83)


# ---------- run_single_seed with mocked Trainer ----------


def _boolq_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "boolq",
        "task_type": "binary_classification",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "boolq",
            "splits": {
                "public": ["train", "validation"],
                "labeled_public": ["train", "validation"],
            },
        },
        "inputs": {"fields": ["question", "passage"], "id_field": "example_id"},
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "boolq_sr_v1"}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


def _bertic_cfg() -> dict:
    return {
        "name": "bertic",
        "hf_repo": "classla/bcms-bertic",
        "family": "electra",
        "params_hint": "110M",
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "fp16": False,
        },
    }


def _tiny_boolq_datasets() -> DatasetDict:
    train = Dataset.from_dict(
        {
            "example_id": [f"t{i}" for i in range(8)],
            "question": [f"q{i}" for i in range(8)],
            "passage": [f"p{i}" for i in range(8)],
            "label": [i % 2 for i in range(8)],
        }
    )
    val = Dataset.from_dict(
        {
            "example_id": [f"v{i}" for i in range(4)],
            "question": [f"q{i}" for i in range(4)],
            "passage": [f"p{i}" for i in range(4)],
            "label": [1, 0, 1, 0],
        }
    )
    return DatasetDict({"train": train, "validation": val})


class _FakeTrainer:
    """Drop-in replacement for ``transformers.Trainer`` used in unit tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.train_called = False
        self.predict_called = False

    def train(self) -> None:
        self.train_called = True

    def predict(self, dataset: Any) -> Any:
        self.predict_called = True
        n = len(dataset) if hasattr(dataset, "__len__") else 4
        logits = np.tile([[0.2, 0.8]], (n, 1))  # always predicts class 1
        return type("PredictionOutput", (), {"predictions": logits, "label_ids": None})()


def test_run_single_seed_returns_seed_result(monkeypatch) -> None:
    monkeypatch.setattr("balkanbench.evaluation.evaluator.Trainer", _FakeTrainer)
    monkeypatch.setattr(
        "balkanbench.evaluation.evaluator.TrainingArguments",
        lambda **kw: kw,
    )

    class _FakeEncoder:
        model = object()
        tokenizer = type(
            "Tok",
            (),
            {"__call__": lambda self, text, **kw: {"input_ids": [1, 2], "attention_mask": [1, 1]}},
        )()
        training_args = {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        }
        model_cfg = _bertic_cfg()
        task_cfg = _boolq_cfg()

    datasets = _tiny_boolq_datasets()
    monkeypatch.setattr(
        "balkanbench.evaluation.evaluator.HFEncoder.build",
        lambda model_cfg, task_cfg: _FakeEncoder(),
    )

    result = run_single_seed(
        model_cfg=_bertic_cfg(),
        task_cfg=_boolq_cfg(),
        language="sr",
        datasets=datasets,
        seed=42,
        output_dir="/tmp/balkanbench-test",
    )

    assert isinstance(result, SeedResult)
    assert result.seed == 42
    # Labels in validation are [1, 0, 1, 0]; fake always predicts 1 -> accuracy = 2/4 = 0.5
    assert result.primary["accuracy"] == pytest.approx(0.5)
    assert result.task_score == pytest.approx(0.5)
    assert len(result.predictions) == 4


# ---------- run_multiseed ----------


def test_run_multiseed_calls_single_seed_per_seed(monkeypatch) -> None:
    seeds_called: list[int] = []

    def fake_single(
        *,
        model_cfg: dict,
        task_cfg: dict,
        language: str,
        datasets: DatasetDict,
        seed: int,
        output_dir: Any,
        eval_split: str = "validation",
        train: bool = True,
        compute_metrics: bool = True,
    ) -> SeedResult:
        seeds_called.append(seed)
        return SeedResult(
            seed=seed,
            primary={"accuracy": 0.7 + 0.01 * seed},
            secondary={},
            task_score=0.7 + 0.01 * seed,
            predictions=[],
            references=[],
            group_ids=None,
        )

    monkeypatch.setattr("balkanbench.evaluation.evaluator.run_single_seed", fake_single)

    datasets = _tiny_boolq_datasets()
    results = run_multiseed(
        model_cfg=_bertic_cfg(),
        task_cfg=_boolq_cfg(),
        language="sr",
        datasets=datasets,
        seeds=[42, 43, 44, 45, 46],
        output_dir="/tmp/bb",
    )
    assert seeds_called == [42, 43, 44, 45, 46]
    assert len(results) == 5


def test_aggregate_returns_aggregate_dataclass() -> None:
    seeds = [
        SeedResult(
            seed=i,
            primary={"accuracy": 0.8 + 0.05 * (i % 3)},
            secondary={},
            task_score=0.8,
            predictions=[],
            references=[],
            group_ids=None,
        )
        for i in range(5)
    ]
    agg = aggregate_seed_results(seeds)
    assert isinstance(agg, Aggregate)
    assert "accuracy" in agg.mean
    assert "accuracy" in agg.stdev

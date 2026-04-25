"""Regression tests for evaluator fixes (review findings 1, 2, 3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from balkanbench.evaluation import run_single_seed


def _unlabeled_test_boolq() -> DatasetDict:
    """BoolQ-style test split with **no** label column (matches the public release)."""
    test = Dataset.from_dict(
        {
            "example_id": ["e0", "e1", "e2"],
            "question": ["q0", "q1", "q2"],
            "passage": ["p0", "p1", "p2"],
        }
    )
    return DatasetDict({"test": test})


def _axg_test_only_with_side_channel() -> DatasetDict:
    """AXg-style diagnostic: only test split, with is_pro_stereotype column."""
    test = Dataset.from_dict(
        {
            "example_id": ["a0", "a1", "a2", "a3"],
            "premise": ["pre0", "pre1", "pre2", "pre3"],
            "hypothesis": ["hy0", "hy1", "hy2", "hy3"],
            "label": [1, 0, 1, 0],
            "is_pro_stereotype": [True, False, True, False],
        }
    )
    return DatasetDict({"test": test})


def _boolq_cfg(metric_columns: list[str] | None = None) -> dict:
    cfg = {
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
            "splits": {"public": ["train", "test"], "labeled_public": ["train"]},
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
    if metric_columns:
        cfg["inputs"]["metric_columns"] = metric_columns
    return cfg


def _axg_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "axg",
        "task_type": "diagnostic",
        "languages": {"available": ["sr"], "ranked": []},
        "dataset": {
            "config": "axg",
            "per_language": {
                "sr": {
                    "public_repo": "permitt/superglue-sr",
                    "private_repo": "permitt/superglue-sr-private",
                }
            },
            "splits": {"public": ["test"], "labeled_public": ["test"]},
        },
        "inputs": {
            "fields": ["premise", "hypothesis"],
            "id_field": "example_id",
            "metric_columns": ["is_pro_stereotype"],
        },
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy", "gender_parity"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "ax_g_sr_v1"}},
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


class _FakeTrainer:
    def __init__(self, *_: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.trained = False

    def train(self) -> None:
        self.trained = True

    def predict(self, dataset: Any) -> Any:
        n = len(dataset)
        logits = np.tile([[0.2, 0.8]], (n, 1))  # always predicts class 1
        return type("P", (), {"predictions": logits, "label_ids": None})()


def _patch_encoder_and_trainer(monkeypatch) -> None:
    class _Enc:
        model = object()
        tokenizer = type(
            "Tok",
            (),
            {"__call__": lambda self, text, **kw: {"input_ids": [1], "attention_mask": [1]}},
        )()
        training_args = {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        }

    monkeypatch.setattr(
        "balkanbench.evaluation.evaluator.HFEncoder.build",
        lambda model_cfg, task_cfg: _Enc(),
    )
    monkeypatch.setattr("balkanbench.evaluation.evaluator.Trainer", _FakeTrainer)
    monkeypatch.setattr(
        "balkanbench.evaluation.evaluator.TrainingArguments",
        lambda **kw: kw,
    )


# ---------- Finding 1: predict path with unlabeled test split ----------


def test_predict_only_mode_skips_reference_loading(monkeypatch) -> None:
    """With compute_metrics=False the evaluator must not touch the label column."""
    _patch_encoder_and_trainer(monkeypatch)

    datasets = _unlabeled_test_boolq()
    result = run_single_seed(
        model_cfg=_bertic_cfg(),
        task_cfg=_boolq_cfg(),
        language="sr",
        datasets=datasets,
        seed=42,
        output_dir="/tmp/bb-predict-test",
        eval_split="test",
        compute_metrics=False,
    )
    assert len(result.predictions) == 3
    assert result.primary == {}
    assert result.secondary == {}
    assert result.task_score == 0.0
    # References should be empty when we did not compute metrics
    assert result.references == []


# ---------- Finding 2: diagnostic path without a train split ----------


def test_run_without_train_skips_trainer_train(monkeypatch) -> None:
    fake_trainers: list[_FakeTrainer] = []

    class _RecordingTrainer(_FakeTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            fake_trainers.append(self)

    class _Enc:
        model = object()
        tokenizer = type(
            "Tok",
            (),
            {"__call__": lambda self, text, **kw: {"input_ids": [1], "attention_mask": [1]}},
        )()
        training_args = {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        }

    monkeypatch.setattr(
        "balkanbench.evaluation.evaluator.HFEncoder.build",
        lambda model_cfg, task_cfg: _Enc(),
    )
    monkeypatch.setattr("balkanbench.evaluation.evaluator.Trainer", _RecordingTrainer)
    monkeypatch.setattr(
        "balkanbench.evaluation.evaluator.TrainingArguments",
        lambda **kw: kw,
    )

    # AXg-style dataset: only a test split
    datasets = _axg_test_only_with_side_channel()
    result = run_single_seed(
        model_cfg=_bertic_cfg(),
        task_cfg=_axg_cfg(),
        language="sr",
        datasets=datasets,
        seed=42,
        output_dir="/tmp/bb-axg-test",
        eval_split="test",
        train=False,
    )
    assert len(result.predictions) == 4
    assert fake_trainers, "Trainer should still be built even without train()"
    assert all(not t.trained for t in fake_trainers), "Trainer.train() should not have run"


# ---------- Finding 3: side-channel metric columns forwarded to score() ----------


def test_metric_columns_are_forwarded_to_score(monkeypatch) -> None:
    _patch_encoder_and_trainer(monkeypatch)

    datasets = _axg_test_only_with_side_channel()
    # AXg's DiagnosticTask.score runs accuracy + gender_parity; gender_parity
    # requires is_pro_stereotype. This must come through inputs.metric_columns.
    result = run_single_seed(
        model_cfg=_bertic_cfg(),
        task_cfg=_axg_cfg(),
        language="sr",
        datasets=datasets,
        seed=42,
        output_dir="/tmp/bb-axg-score",
        eval_split="test",
        train=False,
    )
    assert "accuracy" in result.primary
    assert "gender_parity" in result.secondary


def test_metric_columns_absent_column_is_clear_error(monkeypatch) -> None:
    """If the config asks for a column that's not in the dataset, fail loud."""
    _patch_encoder_and_trainer(monkeypatch)

    cfg = _axg_cfg()
    cfg["inputs"]["metric_columns"] = ["nonexistent_column"]
    datasets = _axg_test_only_with_side_channel()
    with pytest.raises(KeyError, match="nonexistent_column"):
        run_single_seed(
            model_cfg=_bertic_cfg(),
            task_cfg=cfg,
            language="sr",
            datasets=datasets,
            seed=42,
            output_dir="/tmp/bb-axg-missing",
            eval_split="test",
            train=False,
        )

"""Tests for ClassificationTask covering BoolQ / CB / RTE shapes."""

from __future__ import annotations

import numpy as np
import pytest

from balkanbench.tasks import get_task_class
from balkanbench.tasks.classification import ClassificationTask  # ensures registration


class _FakeTokenizer:
    """Fake HF tokenizer that returns hashable-ish ints for test shape checks."""

    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        truncation: bool = True,
        max_length: int = 64,
        padding: str = "longest",
        return_tensors: str | None = None,
    ) -> dict[str, list[int]]:
        ids = [hash((text, text_pair, i)) % 1000 for i in range(max_length // 4)]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }


def _boolq_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "boolq",
        "task_type": "binary_classification",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "boolq",
            "splits": {"public": ["train"], "labeled_public": ["train"]},
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


def _cb_cfg() -> dict:
    cfg = _boolq_cfg()
    cfg["task"] = "cb"
    cfg["task_type"] = "multiclass_classification"
    cfg["inputs"] = {"fields": ["premise", "hypothesis"], "id_field": "example_id"}
    cfg["metrics"] = {
        "primary": ["f1_macro"],
        "report": ["accuracy", "f1_macro"],
        "task_score": "f1_macro",
    }
    cfg["num_labels"] = 3
    return cfg


def test_registered_under_binary_classification() -> None:
    assert get_task_class("binary_classification") is ClassificationTask


def test_registered_under_multiclass_classification() -> None:
    assert get_task_class("multiclass_classification") is ClassificationTask


def test_preprocess_two_field_task_tokenizes_both_as_pair() -> None:
    task = ClassificationTask(_boolq_cfg(), language="sr")
    example = {
        "example_id": "b-0",
        "question": "Is water wet?",
        "passage": "Water is a liquid.",
        "label": 1,
    }
    out = task.preprocess(example, tokenizer=_FakeTokenizer())
    assert "input_ids" in out
    assert "attention_mask" in out
    assert out["labels"] == 1


def test_preprocess_passthrough_without_label() -> None:
    task = ClassificationTask(_boolq_cfg(), language="sr")
    example = {
        "example_id": "b-0",
        "question": "q",
        "passage": "p",
        # no label (test split)
    }
    out = task.preprocess(example, tokenizer=_FakeTokenizer())
    assert "labels" not in out


def test_preprocess_single_field_task() -> None:
    cfg = _boolq_cfg()
    cfg["task"] = "single_field"
    cfg["inputs"] = {"fields": ["text"], "id_field": "example_id"}
    task = ClassificationTask(cfg, language="sr")
    out = task.preprocess(
        {"example_id": "x", "text": "hello", "label": 0}, tokenizer=_FakeTokenizer()
    )
    assert "input_ids" in out
    assert out["labels"] == 0


def test_decode_takes_argmax() -> None:
    task = ClassificationTask(_cb_cfg(), language="sr")
    logits = np.array([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1], [0.0, 0.4, 0.6]])
    preds = task.decode(logits)
    assert list(preds) == [1, 0, 2]


def test_score_produces_f1_macro_and_accuracy_for_cb() -> None:
    task = ClassificationTask(_cb_cfg(), language="sr")
    bundle = task.score(predictions=[0, 1, 2, 0, 1, 2], references=[0, 1, 2, 0, 1, 2])
    assert bundle["accuracy"] == pytest.approx(1.0)
    assert bundle["f1_macro"] == pytest.approx(1.0)


def test_rejects_unavailable_language() -> None:
    with pytest.raises(ValueError):
        ClassificationTask(_boolq_cfg(), language="hr")


def test_num_labels_inferred_from_config_or_default_two() -> None:
    cfg = _cb_cfg()
    task = ClassificationTask(cfg, language="sr")
    assert task.num_labels == 3

    cfg2 = _boolq_cfg()
    task2 = ClassificationTask(cfg2, language="sr")
    assert task2.num_labels == 2

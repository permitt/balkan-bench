"""Tests for the Task ABC + registry."""

from __future__ import annotations

import pytest

from balkanbench.tasks import TaskNotFoundError, get_task_class, register_task
from balkanbench.tasks.base import Task


def _fake_cfg(task_type: str = "binary_classification") -> dict:
    return {
        "benchmark": "superglue",
        "task": "mock",
        "task_type": task_type,
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "mock_v1"}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "mock",
            "splits": {"public": ["train"], "labeled_public": ["train"]},
        },
        "inputs": {"fields": ["text"], "id_field": "example_id"},
    }


def test_registry_raises_on_unknown() -> None:
    with pytest.raises(TaskNotFoundError):
        get_task_class("does_not_exist")


def test_register_and_retrieve_custom_task() -> None:
    @register_task("mock_type_1")
    class MockTask(Task):
        task_type = "mock_type_1"

        def preprocess(self, example, tokenizer=None):
            return {"input_ids": [1]}

        def decode(self, logits):
            return 0

    assert get_task_class("mock_type_1") is MockTask


def test_register_supports_multiple_types_for_one_class() -> None:
    @register_task("mock_type_2", "mock_type_3")
    class MultiTypeTask(Task):
        task_type = "mock_type_2"  # primary

        def preprocess(self, example, tokenizer=None):
            return {}

        def decode(self, logits):
            return 0

    assert get_task_class("mock_type_2") is MultiTypeTask
    assert get_task_class("mock_type_3") is MultiTypeTask


def test_task_id_is_benchmark_dot_task_dot_language() -> None:
    @register_task("mock_type_4")
    class MockTask(Task):
        task_type = "mock_type_4"

        def preprocess(self, example, tokenizer=None):
            return {}

        def decode(self, logits):
            return 0

    task = MockTask(cfg=_fake_cfg(task_type="mock_type_4"), language="sr")
    assert task.task_id == "superglue.mock.sr"
    assert task.primary_metric_names() == ["accuracy"]


def test_score_uses_report_metrics_not_just_primary() -> None:
    @register_task("mock_type_5")
    class MockTask(Task):
        task_type = "mock_type_5"

        def preprocess(self, example, tokenizer=None):
            return {}

        def decode(self, logits):
            return 0

    cfg = _fake_cfg(task_type="mock_type_5")
    cfg["metrics"] = {
        "primary": ["accuracy"],
        "report": ["accuracy", "f1_macro"],
        "task_score": "accuracy",
    }
    task = MockTask(cfg=cfg, language="sr")
    bundle = task.score(predictions=[0, 1, 1, 0], references=[0, 1, 1, 0])
    assert set(bundle) == {"accuracy", "f1_macro"}
    assert bundle["accuracy"] == 1.0
    assert bundle["f1_macro"] == pytest.approx(1.0)


def test_task_score_extracts_scalar_from_bundle() -> None:
    @register_task("mock_type_6")
    class MockTask(Task):
        task_type = "mock_type_6"

        def preprocess(self, example, tokenizer=None):
            return {}

        def decode(self, logits):
            return 0

    task = MockTask(cfg=_fake_cfg(task_type="mock_type_6"), language="sr")
    assert task.task_score({"accuracy": 0.9, "f1_macro": 0.85}) == 0.9


def test_register_rejects_duplicate() -> None:
    @register_task("mock_type_7")
    class A(Task):
        task_type = "mock_type_7"

        def preprocess(self, example, tokenizer=None):
            return {}

        def decode(self, logits):
            return 0

    with pytest.raises(ValueError, match="already registered"):

        @register_task("mock_type_7")
        class B(Task):
            task_type = "mock_type_7"

            def preprocess(self, example, tokenizer=None):
                return {}

            def decode(self, logits):
                return 0

"""Tests for DiagnosticTask and the below-random sanity gate."""

from __future__ import annotations

import pytest

from balkanbench.tasks import get_task_class
from balkanbench.tasks.diagnostic import (
    DiagnosticBelowRandomError,
    DiagnosticTask,  # noqa: F401 (registers the task type)
)


def _axb_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "axb",
        "task_type": "diagnostic",
        "languages": {"available": ["sr"], "ranked": []},
        "dataset": {
            "config": "axb",
            "per_language": {
                "sr": {
                    "public_repo": "permitt/superglue-sr",
                    "private_repo": "permitt/superglue-sr-private",
                }
            },
            "splits": {
                "public": ["test"],
                "labeled_public": ["test"],
                "labeled_private": [],
            },
        },
        "inputs": {"fields": ["premise", "hypothesis"], "id_field": "example_id"},
        "metrics": {
            "primary": ["matthews_correlation"],
            "report": ["matthews_correlation"],
            "task_score": "matthews_correlation",
        },
        "prompts": {"sr": {"template_id": "axb_sr_v1"}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


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
            "splits": {
                "public": ["test"],
                "labeled_public": ["test"],
                "labeled_private": [],
            },
        },
        "inputs": {"fields": ["premise", "hypothesis"], "id_field": "example_id"},
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy", "gender_parity"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "axg_sr_v1"}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


def test_registered_as_diagnostic() -> None:
    assert get_task_class("diagnostic") is DiagnosticTask


def test_axb_sanity_accepts_chance_level() -> None:
    task = DiagnosticTask(_axb_cfg(), language="sr")
    # 20 examples, 10 correct = chance on balanced binary -> matthews near 0
    preds = [1, 0, 1, 0] * 5
    refs = [1, 0, 1, 0] * 3 + [0, 1, 0, 1] + [1, 0, 1, 0]
    bundle = task.score(predictions=preds, references=refs)
    assert "matthews_correlation" in bundle


def test_axb_sanity_fails_below_random() -> None:
    task = DiagnosticTask(_axb_cfg(), language="sr")
    # 20 examples, predictions are perfectly anti-correlated -> matthews = -1
    preds = [0, 1] * 10
    refs = [1, 0] * 10
    with pytest.raises(DiagnosticBelowRandomError, match="matthews_correlation"):
        task.score(predictions=preds, references=refs)


def test_axg_sanity_fails_below_random_accuracy() -> None:
    task = DiagnosticTask(_axg_cfg(), language="sr")
    # 20 examples, all wrong -> accuracy = 0, well below 0.5 - 3sigma (~0.165)
    preds = [0] * 20
    refs = [1] * 20
    is_pro = [True, False] * 10
    with pytest.raises(DiagnosticBelowRandomError, match="accuracy"):
        task.score(predictions=preds, references=refs, is_pro_stereotype=is_pro)


def test_axg_above_chance_passes() -> None:
    task = DiagnosticTask(_axg_cfg(), language="sr")
    # 20 correct out of 20 -> accuracy 1.0
    preds = [1, 0] * 10
    refs = [1, 0] * 10
    is_pro = [True, False] * 10
    bundle = task.score(predictions=preds, references=refs, is_pro_stereotype=is_pro)
    assert bundle["accuracy"] == pytest.approx(1.0)
    assert bundle["gender_parity"] == pytest.approx(0.0)

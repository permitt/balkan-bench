"""Tests for MultiRCTask: grouped binary classification with EM over groups."""

from __future__ import annotations

import numpy as np
import pytest

from balkanbench.tasks import get_task_class
from balkanbench.tasks.multirc import MultiRCTask  # noqa: F401


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        truncation: bool = True,
        max_length: int = 256,
        padding: str = "longest",
    ) -> dict:
        return {
            "input_ids": list(range(32)),
            "attention_mask": [1] * 32,
        }


def _multirc_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "multirc",
        "task_type": "grouped_binary_classification",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "multirc",
            "splits": {"public": ["train", "test"], "labeled_public": ["train"]},
        },
        "inputs": {
            "fields": ["paragraph", "question", "answer"],
            "id_field": "example_id",
            "group_fields": ["paragraph_id", "question_id"],
        },
        "metrics": {
            "primary": ["f1_a"],
            "report": ["f1_a", "exact_match"],
            "task_score": "f1_a",
        },
        "prompts": {"sr": {"template_id": "multirc_sr_v1"}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "f1_a",
        },
    }


def test_registered_as_grouped_binary_classification() -> None:
    assert get_task_class("grouped_binary_classification") is MultiRCTask


def test_preprocess_carries_group_ids_through_columns() -> None:
    task = MultiRCTask(_multirc_cfg(), language="sr")
    example = {
        "example_id": "m-0",
        "paragraph_id": "p1",
        "question_id": "q1",
        "paragraph": "Paragraph text.",
        "question": "Is X true?",
        "answer": "Yes.",
        "label": 1,
    }
    out = task.preprocess(example, tokenizer=_FakeTokenizer())
    # Tokenized inputs
    assert "input_ids" in out
    # Group metadata preserved so the metric can group predictions by example.
    # Lesson from the legacy audit: no positional state.
    assert out["paragraph_id"] == "p1"
    assert out["question_id"] == "q1"
    assert out["labels"] == 1


def test_decode_takes_argmax_and_returns_binary_class() -> None:
    task = MultiRCTask(_multirc_cfg(), language="sr")
    logits = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
    preds = task.decode(logits)
    assert list(preds) == [0, 1, 0]


# ---------- Grouped metric regression (MANDATORY per spec) ----------


def test_exact_match_one_group_all_correct() -> None:
    task = MultiRCTask(_multirc_cfg(), language="sr")
    # One group of 3 candidates, all binary predictions match references
    bundle = task.score(
        predictions=[1, 0, 1],
        references=[1, 0, 1],
        group_ids=[("p1", "q1"), ("p1", "q1"), ("p1", "q1")],
    )
    assert bundle["f1_a"] == pytest.approx(1.0)
    assert bundle["exact_match"] == pytest.approx(1.0)


def test_exact_match_one_wrong_candidate_zeros_the_group() -> None:
    task = MultiRCTask(_multirc_cfg(), language="sr")
    # One wrong prediction in a group of 3 -> group EM is 0
    bundle = task.score(
        predictions=[1, 0, 0],  # third is wrong
        references=[1, 0, 1],
        group_ids=[("p1", "q1"), ("p1", "q1"), ("p1", "q1")],
    )
    # 2/3 candidates correct -> partial f1_a
    assert bundle["f1_a"] > 0
    assert bundle["f1_a"] < 1.0
    # But group EM is 0 because the (p1, q1) group has at least one miss
    assert bundle["exact_match"] == pytest.approx(0.0)


def test_exact_match_averages_across_groups() -> None:
    task = MultiRCTask(_multirc_cfg(), language="sr")
    # Two groups: first perfect, second has a miss
    bundle = task.score(
        predictions=[1, 0, 0, 1, 1],
        references=[1, 0, 1, 1, 1],
        group_ids=[
            ("p1", "q1"),
            ("p1", "q1"),
            ("p1", "q1"),  # group A: miss on 3rd
            ("p2", "q1"),
            ("p2", "q1"),  # group B: perfect
        ],
    )
    # EM over groups = (0 + 1) / 2 = 0.5
    assert bundle["exact_match"] == pytest.approx(0.5)


def test_exact_match_uses_group_not_position() -> None:
    """Regression guard: the metric must identify groups by the columns, not position.

    If two group_ids identify the same group (paragraph, question), the
    candidates must count together, regardless of where they sit in the
    flat prediction list.
    """
    task = MultiRCTask(_multirc_cfg(), language="sr")
    # Interleaved groups
    bundle = task.score(
        predictions=[1, 1, 0, 0],
        references=[1, 1, 0, 0],
        group_ids=[
            ("p1", "q1"),
            ("p2", "q2"),
            ("p1", "q1"),
            ("p2", "q2"),
        ],
    )
    assert bundle["exact_match"] == pytest.approx(1.0)


def test_rejects_mismatched_group_ids_length() -> None:
    task = MultiRCTask(_multirc_cfg(), language="sr")
    with pytest.raises(ValueError):
        task.score(
            predictions=[1, 0],
            references=[1, 0],
            group_ids=[("p1", "q1")],  # too short
        )

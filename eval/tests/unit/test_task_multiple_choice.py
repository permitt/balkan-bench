"""Tests for MultipleChoiceTask (COPA)."""

from __future__ import annotations

import numpy as np
import pytest

from balkanbench.tasks import get_task_class
from balkanbench.tasks.multiple_choice import MultipleChoiceTask  # noqa: F401


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        truncation: bool = True,
        max_length: int = 64,
        padding: str = "longest",
    ) -> dict:
        return {
            "input_ids": list(range(16)),
            "attention_mask": [1] * 16,
        }


def _copa_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "copa",
        "task_type": "multiple_choice",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "num_choices": 2,
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "copa",
            "splits": {"public": ["train"], "labeled_public": ["train"]},
        },
        "inputs": {
            "fields": ["premise", "choice1", "choice2", "question"],
            "id_field": "example_id",
        },
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy"],
            "task_score": "accuracy",
        },
        "prompts": {
            "sr": {
                "template_id": "copa_sr_v1",
                "cause_prompt": "{premise} Šta je bio uzrok?",
                "effect_prompt": "{premise} Šta se desilo kao rezultat?",
            }
        },
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


def test_registered_as_multiple_choice() -> None:
    assert get_task_class("multiple_choice") is MultipleChoiceTask


def test_preprocess_builds_num_choices_pairs() -> None:
    task = MultipleChoiceTask(_copa_cfg(), language="sr")
    example = {
        "example_id": "c-0",
        "premise": "Pas je lajao.",
        "choice1": "Bio je prestrašen.",
        "choice2": "Bio je gladan.",
        "question": "cause",
        "label": 0,
    }
    out = task.preprocess(example, tokenizer=_FakeTokenizer())
    # Each of input_ids / attention_mask has shape [num_choices, seq_len]
    assert len(out["input_ids"]) == 2
    assert len(out["attention_mask"]) == 2
    assert out["labels"] == 0


def test_preprocess_without_label() -> None:
    task = MultipleChoiceTask(_copa_cfg(), language="sr")
    example = {
        "premise": "p",
        "choice1": "a",
        "choice2": "b",
        "question": "effect",
    }
    out = task.preprocess(example, tokenizer=_FakeTokenizer())
    assert "labels" not in out


def test_preprocess_rejects_unknown_question_type() -> None:
    task = MultipleChoiceTask(_copa_cfg(), language="sr")
    example = {
        "premise": "p",
        "choice1": "a",
        "choice2": "b",
        "question": "reason",  # not cause/effect
    }
    with pytest.raises(ValueError, match="question"):
        task.preprocess(example, tokenizer=_FakeTokenizer())


def test_decode_takes_argmax_over_choices() -> None:
    task = MultipleChoiceTask(_copa_cfg(), language="sr")
    logits = np.array([[0.2, 0.8], [0.9, 0.1]])
    preds = task.decode(logits)
    assert list(preds) == [1, 0]


def test_requires_cause_and_effect_prompts() -> None:
    cfg = _copa_cfg()
    del cfg["prompts"]["sr"]["cause_prompt"]
    task = MultipleChoiceTask(cfg, language="sr")
    with pytest.raises(ValueError, match="cause_prompt"):
        task.preprocess(
            {
                "premise": "p",
                "choice1": "a",
                "choice2": "b",
                "question": "cause",
            },
            tokenizer=_FakeTokenizer(),
        )

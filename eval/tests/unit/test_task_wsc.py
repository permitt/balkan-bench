"""Tests for WSCTask: coreference reformulated as binary classification."""

from __future__ import annotations

import pytest

from balkanbench.tasks import get_task_class
from balkanbench.tasks.wsc import WSCTask  # noqa: F401  (registers the task type)


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        truncation: bool = True,
        max_length: int = 128,
        padding: str = "longest",
    ) -> dict:
        return {
            "input_ids": list(range(min(len(text), max_length))),
            "attention_mask": [1] * min(len(text), max_length),
        }


def _cfg(
    template: str = "U rečenici: {text} da li se '{span2_text}' odnosi na '{span1_text}'?",
) -> dict:
    return {
        "benchmark": "superglue",
        "task": "wsc",
        "task_type": "wsc",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "wsc",
            "splits": {"public": ["train"], "labeled_public": ["train"]},
        },
        "inputs": {
            "fields": ["text", "span1_text", "span2_text"],
            "id_field": "example_id",
        },
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "wsc_sr_v1", "template": template}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


def test_registered_as_wsc() -> None:
    assert get_task_class("wsc") is WSCTask


def test_preprocess_renders_template_from_prompts() -> None:
    task = WSCTask(_cfg(), language="sr")
    out = task.preprocess(
        {
            "example_id": "w-0",
            "text": "Mačka je jurila svoj rep.",
            "span1_text": "Mačka",
            "span2_text": "svoj",
            "label": 1,
        },
        tokenizer=_FakeTokenizer(),
    )
    assert "input_ids" in out
    assert out["labels"] == 1


def test_preprocess_requires_template_in_prompts() -> None:
    cfg = _cfg()
    del cfg["prompts"]["sr"]["template"]
    with pytest.raises(ValueError, match="template"):
        WSCTask(cfg, language="sr").preprocess(
            {
                "text": "t",
                "span1_text": "a",
                "span2_text": "b",
            },
            tokenizer=_FakeTokenizer(),
        )

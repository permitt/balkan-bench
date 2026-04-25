"""MultipleChoiceTask for COPA-style tasks.

Tokenizes ``num_choices`` candidate continuations. Each choice becomes one
``(question, choice)`` sequence pair; the model picks the best one. Uses
``AutoModelForMultipleChoice`` upstream; this class only produces
``[num_choices, seq_len]``-shaped ``input_ids`` / ``attention_mask``.

COPA's ``question`` field is either ``"cause"`` or ``"effect"``; the
per-language prompts YAML provides the two corresponding templates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from balkanbench.tasks import register_task
from balkanbench.tasks.base import Task

_QUESTION_TO_PROMPT_KEY: dict[str, str] = {
    "cause": "cause_prompt",
    "effect": "effect_prompt",
}


@register_task("multiple_choice")
class MultipleChoiceTask(Task):
    task_type = "multiple_choice"

    def __init__(self, cfg: dict[str, Any], language: str) -> None:
        super().__init__(cfg, language)
        self._num_choices: int = int(cfg.get("num_choices", 2))
        if self._num_choices < 2:
            raise ValueError("multiple_choice tasks need num_choices >= 2")
        self._label_field: str = cfg.get("label_field", "label")

    @property
    def num_choices(self) -> int:
        return self._num_choices

    def _prompt_for_question(self, question_type: str) -> str:
        key = _QUESTION_TO_PROMPT_KEY.get(question_type)
        if key is None:
            raise ValueError(
                f"unknown COPA question type {question_type!r}; expected cause or effect"
            )
        prompts = self.cfg.get("prompts", {}).get(self.language, {})
        template = prompts.get(key)
        if not template:
            raise ValueError(
                f"MultipleChoiceTask requires prompts.{self.language}.{key} on the task config"
            )
        return str(template)

    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        if tokenizer is None:
            raise ValueError("MultipleChoiceTask.preprocess requires a tokenizer")
        premise = example["premise"]
        choices = [example[f"choice{i + 1}"] for i in range(self._num_choices)]
        question_type = example["question"]
        rendered_question = self._prompt_for_question(question_type).format(premise=premise)

        tokenizer_cfg = self.cfg.get("tokenizer", {})
        max_length = int(tokenizer_cfg.get("max_length", 256))
        # Force padding to max_length so every example produces input_ids
        # of identical shape (num_choices, max_length). This lets the
        # default Trainer collator stack them straight into a 3D tensor;
        # with `padding=longest` (which inside a single-call tokenize is a
        # no-op anyway), the per-choice sequences would have varying
        # lengths and the collator would refuse to build the batch
        # ("excessive nesting (inputs type list where type int is expected)").
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for choice in choices:
            encoded = tokenizer(
                rendered_question,
                text_pair=choice,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            input_ids.append(list(encoded["input_ids"]))
            attention_mask.append(list(encoded["attention_mask"]))

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self._label_field in example:
            out["labels"] = example[self._label_field]
        return out

    def decode(self, logits: Any) -> Any:
        return np.asarray(logits).argmax(axis=-1)

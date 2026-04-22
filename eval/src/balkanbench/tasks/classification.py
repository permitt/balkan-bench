"""Generic sequence-classification task (BoolQ, CB, RTE)."""

from __future__ import annotations

from typing import Any

import numpy as np

from balkanbench.tasks import register_task
from balkanbench.tasks.base import Task


@register_task("binary_classification", "multiclass_classification")
class ClassificationTask(Task):
    """Single- or paired-sequence classification.

    Dispatches on the number of declared input fields:
    - 1 field: tokenize the single field
    - 2 fields: tokenize the pair (HF pair-encoding, BoolQ / CB / RTE)

    ``num_labels`` comes from the task config if present, else defaults to
    ``2`` for ``binary_classification`` and is **required** for
    ``multiclass_classification``.
    """

    task_type = "binary_classification"

    def __init__(self, cfg: dict[str, Any], language: str) -> None:
        super().__init__(cfg, language)
        fields = cfg["inputs"]["fields"]
        if not 1 <= len(fields) <= 2:
            raise ValueError(f"ClassificationTask supports 1 or 2 input fields; got {fields}")
        if cfg["task_type"] == "multiclass_classification" and "num_labels" not in cfg:
            raise ValueError("multiclass_classification requires 'num_labels' in task config")
        self._fields: list[str] = list(fields)
        self._num_labels: int = int(cfg.get("num_labels", 2))
        self._label_field: str = cfg.get("label_field", "label")

    @property
    def num_labels(self) -> int:
        return self._num_labels

    # ------------------------------------------------------------------

    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        if tokenizer is None:
            raise ValueError("ClassificationTask.preprocess requires a tokenizer")

        tokenizer_cfg = self.cfg.get("tokenizer", {})
        max_length = int(tokenizer_cfg.get("max_length", 256))

        if len(self._fields) == 1:
            encoded = tokenizer(
                example[self._fields[0]],
                truncation=True,
                max_length=max_length,
                padding=tokenizer_cfg.get("padding", "longest"),
            )
        else:
            encoded = tokenizer(
                example[self._fields[0]],
                text_pair=example[self._fields[1]],
                truncation=True,
                max_length=max_length,
                padding=tokenizer_cfg.get("padding", "longest"),
            )

        out = dict(encoded)
        if self._label_field in example:
            out["labels"] = example[self._label_field]
        return out

    def decode(self, logits: Any) -> Any:
        arr = np.asarray(logits)
        return arr.argmax(axis=-1)

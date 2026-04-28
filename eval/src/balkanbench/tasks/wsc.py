"""WSCTask: SuperGLUE WSC reformulated as binary classification.

The task is: given a sentence, a referent span (``span1_text``) and a pronoun
span (``span2_text``), decide whether the pronoun refers to the referent.

We ship this as a binary sequence classification over a per-language
natural-language query. The template lives in the task YAML under
``prompts.{language}.template``; preprocess fills it with example fields.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from balkanbench.tasks import register_task
from balkanbench.tasks.base import Task


@register_task("wsc")
class WSCTask(Task):
    task_type = "wsc"
    num_labels = 2

    def __init__(self, cfg: dict[str, Any], language: str) -> None:
        super().__init__(cfg, language)
        self._label_field: str = cfg.get("label_field", "label")

    def _template(self) -> str:
        prompts = self.cfg.get("prompts", {}).get(self.language, {})
        template = prompts.get("template")
        if not template:
            raise ValueError(
                f"WSCTask requires prompts.{self.language}.template on the task config"
            )
        return str(template)

    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        if tokenizer is None:
            raise ValueError("WSCTask.preprocess requires a tokenizer")
        template = self._template()
        rendered = template.format(**example)
        tokenizer_cfg = self.cfg.get("tokenizer", {})
        encoded = tokenizer(
            rendered,
            truncation=True,
            max_length=int(tokenizer_cfg.get("max_length", 256)),
            padding=tokenizer_cfg.get("padding", "longest"),
        )
        out = dict(encoded)
        if self._label_field in example:
            out["labels"] = example[self._label_field]
        return out

    def decode(self, logits: Any) -> Any:
        return np.asarray(logits).argmax(axis=-1)

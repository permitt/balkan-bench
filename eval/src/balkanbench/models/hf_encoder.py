"""Thin wrapper around ``AutoModelFor*`` constructors.

``HFEncoder`` decides the correct ``AutoModel*`` family from the task's
declared ``task_type``, loads the pretrained model + tokenizer, and merges
model + task training args (with benchmark.task-scoped overrides from the
model YAML taking precedence over task defaults).

No custom ``Trainer`` subclass. The evaluator uses the returned model under
the plain HF ``Trainer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# transformers is lazy-loaded via __getattr__ so `balkanbench --version`,
# --help, and the non-ML subcommands avoid its multi-second import cost.
# Tests that monkeypatch ``balkanbench.models.hf_encoder.AutoModelFor*`` still
# work because monkeypatching sets the name in the module dict, which is
# consulted before __getattr__.
_LAZY = {
    "AutoModelForMultipleChoice": "transformers",
    "AutoModelForSequenceClassification": "transformers",
    "AutoTokenizer": "transformers",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


_CLASSIFICATION_TASK_TYPES: set[str] = {
    "binary_classification",
    "multiclass_classification",
    "grouped_binary_classification",
    "wsc",
    "diagnostic",
}


@dataclass
class HFEncoder:
    """Loaded model + tokenizer + merged training args."""

    model: Any
    tokenizer: Any
    training_args: dict[str, Any]
    model_cfg: dict[str, Any]
    task_cfg: dict[str, Any]

    @classmethod
    def build(
        cls,
        *,
        model_cfg: dict[str, Any],
        task_cfg: dict[str, Any],
    ) -> HFEncoder:
        # Resolve lazy names via the module's __getattr__ so tests that
        # monkeypatch ``balkanbench.models.hf_encoder.AutoModelFor*.from_pretrained``
        # still win. The import inside the function body keeps
        # `balkanbench --version` fast by never pulling transformers at
        # package import time.
        from balkanbench.models import hf_encoder as _self

        task_type = task_cfg["task_type"]
        repo = model_cfg["hf_repo"]
        revision = model_cfg.get("hf_revision")

        tokenizer = _self.AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)

        if task_type in _CLASSIFICATION_TASK_TYPES:
            num_labels = int(task_cfg.get("num_labels", 2))
            model = _self.AutoModelForSequenceClassification.from_pretrained(
                repo,
                revision=revision,
                num_labels=num_labels,
            )
        elif task_type == "multiple_choice":
            model = _self.AutoModelForMultipleChoice.from_pretrained(
                repo,
                revision=revision,
            )
        else:
            raise ValueError(f"unknown task_type {task_type!r}; cannot pick an AutoModel* family")

        training_args = _merge_training_args(model_cfg=model_cfg, task_cfg=task_cfg)

        return cls(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            model_cfg=model_cfg,
            task_cfg=task_cfg,
        )


def _merge_training_args(
    *,
    model_cfg: dict[str, Any],
    task_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Merge task + model training args with task-scoped overrides.

    Precedence (low -> high):
    1. task_cfg["training"]
    2. model_cfg["training"]
    3. model_cfg["task_overrides"]["{benchmark}.{task}"]
    """
    merged: dict[str, Any] = {}
    merged.update(task_cfg.get("training", {}))
    merged.update(model_cfg.get("training", {}))

    override_key = f"{task_cfg['benchmark']}.{task_cfg['task']}"
    overrides = model_cfg.get("task_overrides", {}).get(override_key, {})
    merged.update(overrides)

    return merged

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
    "AutoConfig": "transformers",
    "AutoModel": "transformers",
    "AutoModelForMultipleChoice": "transformers",
    "AutoModelForSequenceClassification": "transformers",
    "AutoTokenizer": "transformers",
}

# Architectures whose built-in *ForMultipleChoice impl assumes a pretrained
# pooler that the published BCMS checkpoints (te-sla/teslaXLM,
# classla/xlm-r-bertic) don't carry. transformers >=5 raises
# `IndexError: tuple index out of range` on `outputs[1]` in that case.
# We swap in a generic CLS-pool MultipleChoice head for these to bypass
# the broken path.
_MULTIPLE_CHOICE_NEEDS_CLS_POOL: set[str] = {"xlm-roberta"}


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
            cfg = _self.AutoConfig.from_pretrained(repo, revision=revision)
            arch = (cfg.model_type or "").lower()
            if arch in _MULTIPLE_CHOICE_NEEDS_CLS_POOL:
                model = _build_cls_pool_multiple_choice(
                    repo=repo, revision=revision, num_choices=int(task_cfg.get("num_choices", 2))
                )
            else:
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


def _build_cls_pool_multiple_choice(
    *,
    repo: str,
    revision: str | None,
    num_choices: int,
) -> Any:
    """Construct a generic CLS-pool MultipleChoice model around AutoModel.

    See ``CLSPoolMultipleChoice`` for what this returns and why. The
    class is exposed via the module-level lazy ``__getattr__``, so we
    fetch it through the module rather than as a bare name (the lazy
    hook only fires on attribute access).
    """
    from transformers import AutoConfig, AutoModel

    from balkanbench.models import hf_encoder as _self

    config = AutoConfig.from_pretrained(repo, revision=revision)
    base = AutoModel.from_pretrained(repo, revision=revision, add_pooling_layer=False)
    return _self.CLSPoolMultipleChoice(
        encoder=base,
        config=config,
        num_choices=num_choices,
    )


def _make_cls_pool_class() -> Any:
    """Build the CLSPoolMultipleChoice nn.Module class lazily.

    Lives behind a lazy factory so importing this module doesn't pull in
    torch / transformers. The class itself is module-level (assigned to
    ``CLSPoolMultipleChoice`` below on first use) so HF Trainer's
    checkpoint save can pickle it - inner classes defined inside a
    function silently break ``torch.save`` and the worker hangs on the
    first save callback.
    """
    from torch import nn
    from transformers.modeling_outputs import MultipleChoiceModelOutput

    class CLSPoolMultipleChoice(nn.Module):
        """Generic CLS-pool MultipleChoice head used in place of
        ``AutoModelForMultipleChoice`` for architectures whose stock impl
        crashes when the checkpoint has no pooler (XLM-R in transformers
        >=5: ``XLMRobertaForMultipleChoice.forward`` does ``outputs[1]``
        which raises IndexError when ``pooler_output`` is None).

        Layout: AutoModel(add_pooling_layer=False) -> CLS token ->
        Dropout -> Linear(hidden, 1) -> reshape to (batch, num_choices) +
        CrossEntropy loss. Output schema matches
        ``MultipleChoiceModelOutput`` so the Trainer consumes it without
        changes.
        """

        def __init__(self, *, encoder: Any, config: Any, num_choices: int) -> None:
            super().__init__()
            self.encoder = encoder
            self.config = config
            self._num_choices = num_choices
            drop_p = float(getattr(config, "hidden_dropout_prob", 0.1))
            self.dropout = nn.Dropout(drop_p)
            self.classifier = nn.Linear(config.hidden_size, 1)

        def forward(  # type: ignore[no-untyped-def]
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            **_kw,
        ):
            n = input_ids.shape[1]

            def _flat(t: Any) -> Any:
                return None if t is None else t.view(-1, t.size(-1))

            outputs = self.encoder(
                input_ids=_flat(input_ids),
                attention_mask=_flat(attention_mask),
                token_type_ids=_flat(token_type_ids),
            )
            pooled = outputs.last_hidden_state[:, 0, :]
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled).view(-1, n)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels.view(-1))
            return MultipleChoiceModelOutput(loss=loss, logits=logits)

    return CLSPoolMultipleChoice


# Hook into the module-level __getattr__ defined above by extending its
# table. We can't add a second `def __getattr__` so we patch the existing
# one to also handle the lazy CLSPoolMultipleChoice export.
_orig_getattr = __getattr__


def __getattr__(name: str) -> Any:  # type: ignore[no-redef]
    if name == "CLSPoolMultipleChoice":
        cls = _make_cls_pool_class()
        # Pickle (used by torch.save inside HF Trainer's checkpoint save)
        # locates a class via __module__.__qualname__. The factory's inner
        # class qualname is `_make_cls_pool_class.<locals>.CLSPoolMultipleChoice`,
        # which pickle can't resolve and silently hangs the worker. Rebrand
        # so it looks module-level.
        cls.__module__ = __name__
        cls.__qualname__ = "CLSPoolMultipleChoice"
        globals()[name] = cls
        return cls
    return _orig_getattr(name)

"""MultiRCTask: grouped binary classification with exact-match over groups.

Each example is a ``(paragraph, question, candidate_answer)`` tuple; the model
decides whether the candidate is a correct answer. Candidates that share a
``(paragraph_id, question_id)`` key belong to the same group, and the
group-level ``exact_match`` metric requires every candidate in the group to
be correctly classified.

Lesson from the legacy repo audit: grouped metric state must never live in
class attributes indexed by position. Here, group identity travels with the
example through explicit ``group_ids`` arguments to ``score``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from balkanbench.metrics import get_metric
from balkanbench.tasks import register_task
from balkanbench.tasks.base import Task


@register_task("grouped_binary_classification")
class MultiRCTask(Task):
    task_type = "grouped_binary_classification"
    num_labels = 2

    def __init__(self, cfg: dict[str, Any], language: str) -> None:
        super().__init__(cfg, language)
        self._fields: list[str] = list(cfg["inputs"]["fields"])
        self._group_fields: list[str] = list(cfg["inputs"].get("group_fields", []))
        if not self._group_fields:
            raise ValueError(
                "MultiRCTask requires 'inputs.group_fields' (e.g. paragraph_id, question_id)"
            )
        self._label_field: str = cfg.get("label_field", "label")

    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        if tokenizer is None:
            raise ValueError("MultiRCTask.preprocess requires a tokenizer")
        tokenizer_cfg = self.cfg.get("tokenizer", {})
        max_length = int(tokenizer_cfg.get("max_length", 512))

        # Compose paragraph + question as context; candidate answer is the pair
        context = f"{example[self._fields[0]]} {example[self._fields[1]]}"
        candidate = example[self._fields[2]]
        encoded = tokenizer(
            context,
            text_pair=candidate,
            truncation=True,
            max_length=max_length,
            padding=tokenizer_cfg.get("padding", "longest"),
        )

        out: dict[str, Any] = dict(encoded)
        # Preserve group identity as dataset columns so the metric can group
        # predictions without relying on class-level positional state.
        for gfield in self._group_fields:
            out[gfield] = example[gfield]
        if self._label_field in example:
            out["labels"] = example[self._label_field]
        return out

    def decode(self, logits: Any) -> Any:
        return np.asarray(logits).argmax(axis=-1)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        predictions: Sequence[Any],
        references: Sequence[Any],
        group_ids: Sequence[Any] | None = None,
        **metric_kwargs: Any,
    ) -> dict[str, float]:
        """Compute ``f1_a`` over candidates + ``exact_match`` over groups."""
        if group_ids is None:
            raise ValueError("MultiRCTask.score requires group_ids")
        preds = list(predictions)
        refs = list(references)
        gids = list(group_ids)
        if not (len(preds) == len(refs) == len(gids)):
            raise ValueError(
                "predictions, references, and group_ids must have equal length; "
                f"got {len(preds)} / {len(refs)} / {len(gids)}"
            )

        bundle: dict[str, float] = {}
        for name in self.cfg["metrics"]["report"]:
            if name == "exact_match":
                bundle[name] = _exact_match_over_groups(preds, refs, gids)
            else:
                fn = get_metric(name)
                bundle[name] = fn(predictions=preds, references=refs, **metric_kwargs)
        return bundle


def _exact_match_over_groups(
    predictions: Sequence[Any],
    references: Sequence[Any],
    group_ids: Sequence[Any],
) -> float:
    """Return the fraction of groups in which every candidate is classified correctly."""
    per_group_correct: dict[Any, list[bool]] = {}
    for pred, ref, gid in zip(predictions, references, group_ids, strict=True):
        key = tuple(gid) if isinstance(gid, list | tuple) else gid
        per_group_correct.setdefault(key, []).append(pred == ref)
    if not per_group_correct:
        return 0.0
    perfect = sum(1 for flags in per_group_correct.values() if all(flags))
    return perfect / len(per_group_correct)

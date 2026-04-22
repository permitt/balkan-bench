"""F1 metrics: macro-F1 and positive-class F1 (MultiRC f1_a)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import f1_score

from balkanbench.metrics._common import validate_pair


def f1_macro(
    predictions: Sequence[Any] | None = None,
    references: Sequence[Any] | None = None,
    **_: Any,
) -> float:
    """Macro-averaged F1 across classes."""
    preds, refs = validate_pair(predictions, references)
    return float(f1_score(refs, preds, average="macro"))


def f1_a(
    predictions: Sequence[Any] | None = None,
    references: Sequence[Any] | None = None,
    *,
    positive_label: int = 1,
    **_: Any,
) -> float:
    """F1 of the positive class.

    For MultiRC, this is the candidate-level binary F1 over ``is_correct``
    predictions. Default positive label is ``1``.
    """
    preds, refs = validate_pair(predictions, references)
    return float(f1_score(refs, preds, pos_label=positive_label))

"""Accuracy metric."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from balkanbench.metrics._common import validate_pair


def accuracy(
    predictions: Sequence[Any] | None = None,
    references: Sequence[Any] | None = None,
    **_: Any,
) -> float:
    """Mean agreement between ``predictions`` and ``references``."""
    preds, refs = validate_pair(predictions, references)
    matches = sum(1 for p, r in zip(preds, refs, strict=True) if p == r)
    return matches / len(preds)

"""Shared validation helpers for metric implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def validate_pair(
    predictions: Sequence[Any] | None,
    references: Sequence[Any] | None,
) -> tuple[list[Any], list[Any]]:
    """Coerce to lists, validate length + non-empty."""
    if predictions is None or references is None:
        raise ValueError("predictions and references are required")
    preds = list(predictions)
    refs = list(references)
    if not preds:
        raise ValueError("predictions must be non-empty")
    if len(preds) != len(refs):
        raise ValueError(f"predictions and references length mismatch: {len(preds)} vs {len(refs)}")
    return preds, refs

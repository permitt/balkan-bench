"""Matthews correlation coefficient (AXb primary metric)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import matthews_corrcoef

from balkanbench.metrics._common import validate_pair


def matthews_correlation(
    predictions: Sequence[Any] | None = None,
    references: Sequence[Any] | None = None,
    **_: Any,
) -> float:
    """Matthews correlation coefficient in ``[-1, 1]``."""
    preds, refs = validate_pair(predictions, references)
    return float(matthews_corrcoef(refs, preds))

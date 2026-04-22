"""DiagnosticTask: AX-b + AX-g test-only evaluations.

Structurally these are binary NLI tasks; we inherit the Classification
preprocess/decode shape via composition. The defining behaviour is the
**below-random sanity gate**: the legacy repo shipped a version where
diagnostic scores came back well below chance because of a label-mapping
bug, and no downstream check caught it. v0.1 refuses to emit a diagnostic
result more than 3sigma below the random baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from balkanbench.metrics import get_metric
from balkanbench.tasks import register_task
from balkanbench.tasks.base import Task

# 3-sigma guard: probability of scoring this far below random purely by chance
# is ~0.3%, so a persistent below-random result almost always signals a bug.
_SIGMA_TOLERANCE = 3.0


class DiagnosticBelowRandomError(RuntimeError):
    """Raised when a diagnostic scores > 3sigma below the random baseline."""


@register_task("diagnostic")
class DiagnosticTask(Task):
    task_type = "diagnostic"
    num_labels = 2

    def __init__(self, cfg: dict[str, Any], language: str) -> None:
        super().__init__(cfg, language)
        self._fields: list[str] = list(cfg["inputs"]["fields"])
        self._label_field: str = cfg.get("label_field", "label")

    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        if tokenizer is None:
            raise ValueError("DiagnosticTask.preprocess requires a tokenizer")
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
        return np.asarray(logits).argmax(axis=-1)

    def score(
        self,
        *,
        predictions: Sequence[Any],
        references: Sequence[Any],
        **metric_kwargs: Any,
    ) -> dict[str, float]:
        bundle: dict[str, float] = {}
        for name in self.cfg["metrics"]["report"]:
            fn = get_metric(name)
            bundle[name] = fn(predictions=predictions, references=references, **metric_kwargs)
        _assert_above_random(bundle, n=len(list(references)))
        return bundle


def _binary_random_std(n: int) -> float:
    """Standard deviation of a sample mean for a 50/50 Bernoulli random predictor."""
    if n < 1:
        return float("inf")
    return math.sqrt(0.25 / n)


def _assert_above_random(bundle: dict[str, float], *, n: int) -> None:
    """Raise ``DiagnosticBelowRandomError`` if any known diagnostic metric is far below chance."""
    std = _binary_random_std(n)
    # accuracy: chance = 0.5, fail if value < 0.5 - 3*std
    if "accuracy" in bundle and bundle["accuracy"] < 0.5 - _SIGMA_TOLERANCE * std:
        raise DiagnosticBelowRandomError(
            f"accuracy={bundle['accuracy']:.3f} is more than "
            f"{_SIGMA_TOLERANCE}sigma below the 0.5 random baseline (n={n}, sigma={std:.3f}); "
            "refusing to emit a diagnostic result, this almost always signals a label-mapping bug"
        )
    # matthews_correlation: chance = 0.0, std of MCC under chance for balanced
    # binary is ~1/sqrt(n). Fail if it is far below 0.
    if "matthews_correlation" in bundle:
        mcc_std = 1.0 / math.sqrt(n) if n > 0 else float("inf")
        if bundle["matthews_correlation"] < -_SIGMA_TOLERANCE * mcc_std:
            raise DiagnosticBelowRandomError(
                f"matthews_correlation={bundle['matthews_correlation']:.3f} is more than "
                f"{_SIGMA_TOLERANCE}sigma below the 0.0 random baseline "
                f"(n={n}, sigma={mcc_std:.3f}); "
                "refusing to emit a diagnostic result, this almost always signals a label-mapping bug"
            )

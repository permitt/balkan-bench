"""Metric registry.

Every metric is a plain callable ``(*, predictions, references, **kwargs) -> float``.
Tasks look metrics up by string name from this registry so there is exactly one
source of truth for the metric implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from balkanbench.metrics.accuracy import accuracy
from balkanbench.metrics.f1 import f1_a, f1_macro
from balkanbench.metrics.gender_parity import gender_parity
from balkanbench.metrics.matthews import matthews_correlation

MetricFn = Callable[..., float]


class MetricNotFoundError(KeyError):
    """Raised when a task references a metric that is not in the registry."""


_REGISTRY: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "f1_macro": f1_macro,
    "f1_a": f1_a,
    "matthews_correlation": matthews_correlation,
    "gender_parity": gender_parity,
}


def get_metric(name: str) -> MetricFn:
    """Return the metric registered under ``name`` or raise ``MetricNotFoundError``."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise MetricNotFoundError(
            f"metric {name!r} is not registered; known metrics: {sorted(_REGISTRY)}"
        ) from exc


def list_metrics() -> list[str]:
    """Return the sorted list of registered metric names."""
    return sorted(_REGISTRY)


def register_metric(name: str, fn: MetricFn) -> None:
    """Register a new metric. Intended for community contributions via plugins."""
    if name in _REGISTRY:
        raise ValueError(f"metric {name!r} already registered")
    _REGISTRY[name] = fn


__all__ = [
    "MetricFn",
    "MetricNotFoundError",
    "get_metric",
    "list_metrics",
    "register_metric",
    # direct imports kept for convenience in typing
    "accuracy",
    "f1_macro",
    "f1_a",
    "matthews_correlation",
    "gender_parity",
]


def _ignore_kwargs(fn: MetricFn) -> MetricFn:
    """Wrap ``fn`` so accidental unused kwargs do not break registry callers."""

    def wrapped(**kwargs: Any) -> float:
        return fn(**kwargs)

    return wrapped

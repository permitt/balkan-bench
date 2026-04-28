"""``Task`` ABC.

Every concrete task owns its preprocessing, prediction decoding, and scoring.
The evaluator and CLI never see task-specific code: they work purely through
this interface.

Design rules (Lessons from the legacy repo audit, see spec_claude.md):
- No class-level positional state. Metadata a metric needs (e.g. MultiRC
  ``group_id``) travels with the example through dataset columns.
- No custom Trainer subclasses. Everything goes through the public HF API.
- Prompts are data (YAML), not Python strings embedded in methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from balkanbench.metrics import get_metric


class Task(ABC):
    """Base class for every BalkanBench task."""

    task_type: ClassVar[str]
    """Matches one value of the ``task_type`` enum in ``schemas/task_spec.json``."""

    def __init__(self, cfg: dict[str, Any], language: str) -> None:
        if language not in cfg["languages"]["available"]:
            raise ValueError(
                f"task {cfg['task']!r} does not declare language {language!r}; "
                f"available: {cfg['languages']['available']}"
            )
        self.cfg = cfg
        self.language = language

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def task_id(self) -> str:
        """Canonical ``{benchmark}.{task}.{language}`` identifier."""
        return f"{self.cfg['benchmark']}.{self.cfg['task']}.{self.language}"

    @property
    def benchmark(self) -> str:
        return str(self.cfg["benchmark"])

    @property
    def task_name(self) -> str:
        return str(self.cfg["task"])

    def primary_metric_names(self) -> list[str]:
        return list(self.cfg["metrics"]["primary"])

    def task_score(self, metric_bundle: dict[str, float]) -> float:
        """Extract the single canonical task score from a metric bundle."""
        name = self.cfg["metrics"]["task_score"]
        return metric_bundle[name]

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------

    @abstractmethod
    def preprocess(self, example: dict[str, Any], tokenizer: Any = None) -> dict[str, Any]:
        """Turn one example from the dataset into model inputs.

        Implementations that need a tokenizer may take it as an argument;
        those that do not (diagnostic, some variants) can ignore it.
        """

    @abstractmethod
    def decode(self, logits: Any) -> Any:
        """Map model output (usually logits) to a predicted label / choice."""

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        predictions: Sequence[Any],
        references: Sequence[Any],
        **metric_kwargs: Any,
    ) -> dict[str, float]:
        """Compute the configured ``report`` metrics.

        Extra keyword arguments are forwarded to every metric (e.g.
        ``group_id`` for MultiRC, ``is_pro_stereotype`` for AXg). Metrics
        that do not need them accept ``**_``.
        """
        bundle: dict[str, float] = {}
        for name in self.cfg["metrics"]["report"]:
            fn = get_metric(name)
            bundle[name] = fn(predictions=predictions, references=references, **metric_kwargs)
        return bundle


__all__ = ["Task"]

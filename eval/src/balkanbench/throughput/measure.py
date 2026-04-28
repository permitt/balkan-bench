"""Inference throughput measurement loop.

Design rules:
- ``predict_fn`` is injected so tests can run without a GPU and without HF.
  The real caller supplies an implementation backed by ``model(**inputs)``.
- Warmup batches are discarded; measurement batches form the sample.
- We report **median** wall-clock per batch (less volatile than mean on a
  noisy shared GPU).
- All sizes come from the task config (``batch_size``, ``tokenizer.max_length``)
  so throughput numbers are directly comparable between models on the same task.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

PredictFn = Callable[..., tuple[Any, float]]
"""``(model, batch, *, batch_size, max_seq_len) -> (predictions, seconds_per_batch)``."""


@dataclass
class ThroughputSample:
    """The measured throughput for one (model, task) pair."""

    batch_size: int
    max_seq_len: int
    throughput_ex_per_sec: float
    throughput_tok_per_sec: float
    peak_vram_mib: float
    hardware: str
    precision: str
    warmup_batches: int
    measurement_batches: int
    torch_version: str
    driver_version: str


def _iter_batches(dataset: Dataset, batch_size: int, max_batches: int) -> list[list[int]]:
    """Return a list of row-index batches, capped at ``max_batches``."""
    batches: list[list[int]] = []
    for start in range(0, len(dataset), batch_size):
        batches.append(list(range(start, min(start + batch_size, len(dataset)))))
        if len(batches) >= max_batches:
            break
    return batches


def _peak_vram_mib() -> float:
    try:  # pragma: no cover - GPU-only branch
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:  # pragma: no cover
        return 0.0


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:  # pragma: no cover
        return "unknown"


def measure_task_throughput(
    *,
    model: Any,
    tokenizer: Any,
    task_cfg: dict[str, Any],
    dataset: Dataset,
    language: str,
    hardware: str,
    precision: str,
    warmup_batches: int = 2,
    measurement_batches: int = 50,
    predict_fn: PredictFn,
    driver_version: str = "unknown",
) -> ThroughputSample:
    """Run warmup + measurement batches; return a ``ThroughputSample``.

    ``predict_fn`` is called once per batch and returns ``(predictions, seconds)``.
    The batch payload is opaque to this function; ``predict_fn`` owns tokenisation
    or does inline indexing. We pass ``batch_size`` and ``max_seq_len`` explicitly
    so tokeniser-free fakes can honour them.
    """
    del language  # reserved for future language-specific prompt adjustments
    del tokenizer  # the real caller tokenises inside predict_fn
    batch_size = int(task_cfg["training"]["batch_size"])
    max_seq_len = int(task_cfg.get("tokenizer", {}).get("max_length", 128))

    total_batches_needed = warmup_batches + measurement_batches
    batches = _iter_batches(dataset, batch_size, total_batches_needed)

    # Warmup
    for idx in range(min(warmup_batches, len(batches))):
        predict_fn(model, batches[idx], batch_size=batch_size, max_seq_len=max_seq_len)

    # Measurement
    measurement_slice = batches[warmup_batches:]
    if not measurement_slice:
        raise ValueError(
            "No measurement batches available; increase dataset size or decrease warmup."
        )

    # Per-batch throughput uses the actual batch length, not the configured
    # batch_size, so a tail batch shorter than batch_size does not inflate the
    # reported ex/s. Take the median across per-batch throughputs.
    per_batch_ex_per_sec: list[float] = []
    for batch in measurement_slice:
        _preds, seconds = predict_fn(model, batch, batch_size=batch_size, max_seq_len=max_seq_len)
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError(f"predict_fn reported non-positive latency: {seconds}")
        actual_size = len(batch)
        per_batch_ex_per_sec.append(actual_size / seconds)

    ex_per_sec = statistics.median(per_batch_ex_per_sec)
    tok_per_sec = ex_per_sec * max_seq_len

    return ThroughputSample(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        throughput_ex_per_sec=ex_per_sec,
        throughput_tok_per_sec=tok_per_sec,
        peak_vram_mib=_peak_vram_mib(),
        hardware=hardware,
        precision=precision,
        warmup_batches=warmup_batches,
        measurement_batches=len(per_batch_ex_per_sec),
        torch_version=_torch_version(),
        driver_version=driver_version,
    )

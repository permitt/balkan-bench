"""Tests for `balkanbench.cli.throughput.default_predict_fn`.

Covers the task-type-aware input shape + the no-silent-fallback contract.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from balkanbench.cli.throughput import default_predict_fn


class _RecordingModel:
    """Fake model that captures the shape of each forward-pass tensor."""

    def __init__(self, raise_on_call: bool = False) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.raise_on_call = raise_on_call

    def eval(self) -> None:  # noqa: D401 - torch.nn.Module stub
        """No-op; matches torch.nn.Module.eval signature."""

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> Any:
        self.calls.append(tuple(input_ids.shape))
        if self.raise_on_call:
            raise RuntimeError("simulated forward failure")
        return type("Out", (), {"logits": torch.zeros(input_ids.shape[0])})


def test_default_predict_fn_uses_2d_for_classification() -> None:
    model = _RecordingModel()
    preds, seconds = default_predict_fn(
        model,
        batch=[0, 1, 2, 3],
        batch_size=4,
        max_seq_len=16,
        task_type="binary_classification",
    )
    # Exactly one call with a 2D tensor of shape (batch_size, max_seq_len)
    assert model.calls == [(4, 16)]
    assert seconds > 0
    assert preds.shape == (4,)


def test_default_predict_fn_uses_3d_for_multiple_choice() -> None:
    model = _RecordingModel()
    default_predict_fn(
        model,
        batch=[0, 1, 2, 3],
        batch_size=4,
        max_seq_len=16,
        task_type="multiple_choice",
        num_choices=2,
    )
    # AutoModelForMultipleChoice expects (batch_size, num_choices, max_seq_len)
    assert model.calls == [(4, 2, 16)]


def test_default_predict_fn_uses_3d_for_multiple_choice_three_way() -> None:
    model = _RecordingModel()
    default_predict_fn(
        model,
        batch=[0, 1],
        batch_size=2,
        max_seq_len=8,
        task_type="multiple_choice",
        num_choices=3,
    )
    assert model.calls == [(2, 3, 8)]


def test_default_predict_fn_does_not_silently_swallow_errors() -> None:
    """Regression guard: a failed forward must raise, never fall back.

    The prior implementation caught every Exception and returned
    ``input_ids.sum()`` timings, producing a "throughput" number that
    wasn't measuring a real forward pass.
    """
    model = _RecordingModel(raise_on_call=True)
    with pytest.raises(RuntimeError, match="simulated forward failure"):
        default_predict_fn(
            model,
            batch=[0, 1],
            batch_size=2,
            max_seq_len=4,
            task_type="binary_classification",
        )


def test_default_predict_fn_classification_defaults() -> None:
    """task_type defaults to binary_classification, num_choices defaults to 2."""
    model = _RecordingModel()
    default_predict_fn(
        model,
        batch=[0],
        batch_size=1,
        max_seq_len=4,
    )
    assert model.calls == [(1, 4)]

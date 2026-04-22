"""Tests for `balkanbench.seed`."""

from __future__ import annotations

import random

import numpy as np

from balkanbench.seed import set_seed


def test_set_seed_sets_python_random() -> None:
    set_seed(42)
    first = [random.randint(0, 1_000_000) for _ in range(4)]
    set_seed(42)
    second = [random.randint(0, 1_000_000) for _ in range(4)]
    assert first == second


def test_set_seed_sets_numpy_random() -> None:
    set_seed(123)
    first = np.random.rand(4).tolist()
    set_seed(123)
    second = np.random.rand(4).tolist()
    assert first == second


def test_set_seed_torch_is_idempotent() -> None:
    import torch

    set_seed(7)
    a = torch.randn(3)
    set_seed(7)
    b = torch.randn(3)
    assert torch.equal(a, b)

"""Unit tests for the metric registry + implementations."""

from __future__ import annotations

import math

import pytest

from balkanbench.metrics import MetricNotFoundError, get_metric, list_metrics
from balkanbench.metrics.accuracy import accuracy
from balkanbench.metrics.f1 import f1_a, f1_macro
from balkanbench.metrics.gender_parity import gender_parity
from balkanbench.metrics.matthews import matthews_correlation

# ---------- registry ----------


def test_registry_resolves_known_metrics() -> None:
    for name in ("accuracy", "f1_macro", "f1_a", "matthews_correlation", "gender_parity"):
        fn = get_metric(name)
        assert callable(fn)


def test_registry_lists_all_known_metrics() -> None:
    names = list_metrics()
    assert set(names) >= {
        "accuracy",
        "f1_macro",
        "f1_a",
        "matthews_correlation",
        "gender_parity",
    }


def test_registry_raises_on_unknown() -> None:
    with pytest.raises(MetricNotFoundError):
        get_metric("does_not_exist")


# ---------- accuracy ----------


def test_accuracy_perfect() -> None:
    assert accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_accuracy_half() -> None:
    assert accuracy([0, 0, 1, 1], [0, 1, 1, 0]) == 0.5


def test_accuracy_empty_raises() -> None:
    with pytest.raises(ValueError):
        accuracy([], [])


def test_accuracy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


# ---------- f1 ----------


def test_f1_macro_balanced_binary() -> None:
    # Predictions and references are identical; f1_macro should be 1.0
    preds = [0, 0, 1, 1, 0, 1]
    refs = [0, 0, 1, 1, 0, 1]
    assert f1_macro(preds, refs) == pytest.approx(1.0)


def test_f1_macro_matches_sklearn_for_three_class() -> None:
    from sklearn.metrics import f1_score as sklearn_f1

    preds = [0, 1, 2, 0, 1, 2, 1, 0]
    refs = [0, 2, 1, 0, 1, 2, 0, 1]
    expected = sklearn_f1(refs, preds, average="macro")
    assert f1_macro(preds, refs) == pytest.approx(expected)


def test_f1_a_is_f1_of_positive_class() -> None:
    from sklearn.metrics import f1_score as sklearn_f1

    preds = [1, 0, 1, 1, 0, 0, 1]
    refs = [1, 0, 0, 1, 1, 0, 1]
    expected = sklearn_f1(refs, preds, pos_label=1)
    assert f1_a(preds, refs) == pytest.approx(expected)


# ---------- matthews ----------


def test_matthews_perfect_positive() -> None:
    preds = [0, 0, 1, 1]
    refs = [0, 0, 1, 1]
    assert matthews_correlation(preds, refs) == pytest.approx(1.0)


def test_matthews_perfect_inverse() -> None:
    preds = [1, 1, 0, 0]
    refs = [0, 0, 1, 1]
    assert matthews_correlation(preds, refs) == pytest.approx(-1.0)


def test_matthews_no_correlation() -> None:
    preds = [0, 0, 0, 0]
    refs = [0, 1, 0, 1]
    # Degenerate case: constant predictor yields zero Matthews via sklearn
    assert matthews_correlation(preds, refs) == pytest.approx(0.0)


# ---------- gender_parity ----------


def test_gender_parity_flat_accuracy_same_across_groups() -> None:
    # Pro-stereotype examples all correct (2/2); anti-stereotype also correct (2/2)
    preds = [1, 1, 0, 0]
    refs = [1, 1, 0, 0]
    is_pro = [True, True, False, False]
    # accuracy(pro) - accuracy(anti) = 1.0 - 1.0 = 0.0 (parity)
    assert gender_parity(preds, refs, is_pro_stereotype=is_pro) == pytest.approx(0.0)


def test_gender_parity_high_pro_accuracy_low_anti() -> None:
    # Pro correct 2/2, anti correct 1/2 -> delta 0.5
    preds = [1, 1, 1, 0]
    refs = [1, 1, 0, 0]
    is_pro = [True, True, False, False]
    assert gender_parity(preds, refs, is_pro_stereotype=is_pro) == pytest.approx(0.5)


def test_gender_parity_empty_group_raises() -> None:
    preds = [1, 1]
    refs = [1, 1]
    is_pro = [True, True]
    with pytest.raises(ValueError, match="anti"):
        gender_parity(preds, refs, is_pro_stereotype=is_pro)


def test_gender_parity_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        gender_parity([1, 1], [1], is_pro_stereotype=[True, False])


# ---------- registry-backed call ----------


def test_registry_accuracy_call() -> None:
    fn = get_metric("accuracy")
    out = fn(predictions=[0, 1], references=[0, 1])
    assert out == 1.0


def test_registry_f1_macro_call() -> None:
    fn = get_metric("f1_macro")
    out = fn(predictions=[0, 1, 2], references=[0, 1, 2])
    assert out == pytest.approx(1.0)


def test_registry_gender_parity_call() -> None:
    fn = get_metric("gender_parity")
    out = fn(
        predictions=[1, 1, 1, 0],
        references=[1, 1, 0, 0],
        is_pro_stereotype=[True, True, False, False],
    )
    assert out == pytest.approx(0.5)


# ---------- nan handling ----------


def test_metrics_do_not_leak_nan() -> None:
    """A perfect predictor on a trivial binary problem should return a finite score."""
    for name in ("accuracy", "f1_macro", "f1_a"):
        fn = get_metric(name)
        value = fn(predictions=[0, 1, 0, 1], references=[0, 1, 0, 1])
        assert math.isfinite(value), f"{name} returned non-finite value"

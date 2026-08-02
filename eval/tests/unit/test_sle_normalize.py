"""Unit tests for `balkanbench.data.sle_normalize` (no network)."""

from __future__ import annotations

import pytest

from balkanbench.data.sle_normalize import normalize_rows


def test_mc_row_passthrough_and_example_id_synthesis() -> None:
    rows = [{"query": "Sta je?", "choices": ["a", "b"], "gold": 0}]
    out = normalize_rows("arc_challenge", "test", rows)
    assert out == [
        {
            "example_id": "arc_challenge-test-0",
            "query": "Sta je?",
            "choices": ["a", "b"],
            "gold": 0,
        }
    ]


def test_mc_row_uses_existing_id_field() -> None:
    rows = [{"id": "Mercury_7175875", "query": "Sta je?", "choices": ["a", "b"], "gold": 1}]
    out = normalize_rows("arc_challenge", "test", rows)
    assert out[0]["example_id"] == "Mercury_7175875"


def test_piqa_keeps_goal_field_not_query() -> None:
    rows = [{"goal": "Kako da...", "choices": ["a", "b"], "gold": 1}]
    out = normalize_rows("piqa", "test", rows)
    assert out == [
        {
            "example_id": "piqa-test-0",
            "goal": "Kako da...",
            "choices": ["a", "b"],
            "gold": 1,
        }
    ]


def test_boolq_idx_maps_to_example_id() -> None:
    rows = [
        {"question": "Da li...", "passage": "Neki tekst.", "idx": 0, "label": 1},
        {"question": "Da li...2", "passage": "Neki tekst 2.", "idx": 1, "label": 0},
    ]
    out = normalize_rows("boolq", "test", rows)
    assert out == [
        {
            "example_id": "boolq-test-0",
            "question": "Da li...",
            "passage": "Neki tekst.",
            "label": 1,
        },
        {
            "example_id": "boolq-test-1",
            "question": "Da li...2",
            "passage": "Neki tekst 2.",
            "label": 0,
        },
    ]


def test_triviaqa_flattens_answer() -> None:
    rows = [
        {
            "question": "Ko je...",
            "answer": {"value": "Nikola Tesla", "aliases": ["Tesla", "N. Tesla"]},
        }
    ]
    out = normalize_rows("triviaqa", "test", rows)
    assert out[0] == {
        "example_id": "triviaqa-test-0",
        "question": "Ko je...",
        "answer_value": "Nikola Tesla",
        "answer_aliases": ["Tesla", "N. Tesla"],
    }


def test_winogrande_passthrough() -> None:
    rows = [
        {
            "sentence": "Ana i Marija su... _",
            "option1": "Ana",
            "option2": "Marija",
            "answer": "2",
        }
    ]
    out = normalize_rows("winogrande", "test", rows)
    assert out == [
        {
            "example_id": "winogrande-test-0",
            "sentence": "Ana i Marija su... _",
            "option1": "Ana",
            "option2": "Marija",
            "answer": "2",
        }
    ]


def test_nq_open_passthrough() -> None:
    rows = [{"question": "Gde je...", "answer": ["Beograd", "Belgrade"]}]
    out = normalize_rows("nq_open", "test", rows)
    assert out == [
        {
            "example_id": "nq_open-test-0",
            "question": "Gde je...",
            "answer": ["Beograd", "Belgrade"],
        }
    ]


def test_unknown_task_raises_value_error() -> None:
    with pytest.raises(ValueError):
        normalize_rows("not_a_real_task", "test", [{"anything": 1}])


def test_unknown_task_raises_even_with_empty_rows() -> None:
    with pytest.raises(ValueError):
        normalize_rows("not_a_real_task", "test", [])


@pytest.mark.parametrize(
    "task,row",
    [
        ("arc_challenge", {"query": "x", "choices": ["a"]}),  # missing gold
        ("boolq", {"question": "x", "passage": "y"}),  # missing idx/label
        ("winogrande", {"sentence": "x", "option1": "a"}),  # missing option2/answer
        ("nq_open", {"question": "x"}),  # missing answer
        ("triviaqa", {"answer": {"value": "x", "aliases": []}}),  # missing question
        ("piqa", {"goal": "x", "gold": 0}),  # missing choices
    ],
)
def test_row_missing_required_field_raises_value_error(task: str, row: dict) -> None:
    with pytest.raises(ValueError):
        normalize_rows(task, "test", [row])

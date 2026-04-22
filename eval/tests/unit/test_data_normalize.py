"""Unit tests for `balkanbench.data.normalize`."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from balkanbench.data.normalize import (
    attach_task_metadata,
    rename_splits,
    strip_label_columns,
)


def _mini_copa_with_dev() -> DatasetDict:
    train = Dataset.from_dict(
        {
            "example_id": ["c-train-0", "c-train-1"],
            "premise": ["Čovjek je ustao.", "Pas je lajao."],
            "choice1": ["a", "b"],
            "choice2": ["c", "d"],
            "question": ["cause", "effect"],
            "label": [0, 1],
        }
    )
    dev = Dataset.from_dict(
        {
            "example_id": ["c-dev-0"],
            "premise": ["Zvono je zazvonilo."],
            "choice1": ["a"],
            "choice2": ["b"],
            "question": ["effect"],
            "label": [1],
        }
    )
    test = Dataset.from_dict(
        {
            "example_id": ["c-test-0"],
            "premise": ["Tekst."],
            "choice1": ["a"],
            "choice2": ["b"],
            "question": ["cause"],
            "label": [0],
        }
    )
    return DatasetDict({"train": train, "dev": dev, "test": test})


def _mini_boolq() -> DatasetDict:
    def make(n: int, prefix: str) -> Dataset:
        return Dataset.from_dict(
            {
                "example_id": [f"{prefix}-{i}" for i in range(n)],
                "question": [f"q{i}" for i in range(n)],
                "passage": [f"p{i}" for i in range(n)],
                "label": [i % 2 for i in range(n)],
            }
        )

    return DatasetDict(
        {"train": make(3, "b-train"), "validation": make(2, "b-val"), "test": make(2, "b-test")}
    )


def test_rename_splits_copa_dev_to_validation() -> None:
    copa = _mini_copa_with_dev()
    out = rename_splits({"dev": "validation"}, copa)
    assert set(out.keys()) == {"train", "validation", "test"}
    assert out["validation"][0]["example_id"] == "c-dev-0"


def test_rename_splits_idempotent_when_target_already_exists() -> None:
    boolq = _mini_boolq()
    out = rename_splits({"dev": "validation"}, boolq)
    # boolq has no "dev" split; mapping is a no-op
    assert set(out.keys()) == {"train", "validation", "test"}


def test_rename_splits_errors_on_collision() -> None:
    copa = _mini_copa_with_dev()
    copa_with_val = DatasetDict({**copa, "validation": copa["dev"]})
    with pytest.raises(ValueError, match="collision"):
        rename_splits({"dev": "validation"}, copa_with_val)


def test_strip_label_columns_removes_label_from_test() -> None:
    boolq = _mini_boolq()
    out = strip_label_columns(boolq, split="test", label_fields=["label"])
    assert "label" not in out["test"].column_names
    assert "label" in out["train"].column_names  # untouched
    assert "label" in out["validation"].column_names  # untouched
    assert out["test"].num_rows == 2


def test_strip_label_columns_no_op_when_label_absent() -> None:
    boolq = _mini_boolq()
    boolq_no_test_label = DatasetDict({**boolq, "test": boolq["test"].remove_columns(["label"])})
    out = strip_label_columns(boolq_no_test_label, split="test", label_fields=["label"])
    assert "label" not in out["test"].column_names


def test_attach_task_metadata_adds_task_id_and_language() -> None:
    boolq = _mini_boolq()
    out = attach_task_metadata(boolq, task_id="superglue.boolq.sr", language="sr")
    for split in out:
        cols = out[split].column_names
        assert "task_id" in cols
        assert "language" in cols
        assert all(v == "superglue.boolq.sr" for v in out[split]["task_id"])
        assert all(v == "sr" for v in out[split]["language"])


def test_attach_task_metadata_fills_example_id_if_missing() -> None:
    ds = Dataset.from_dict({"question": ["a", "b"], "passage": ["x", "y"], "label": [0, 1]})
    dd = DatasetDict({"train": ds})
    out = attach_task_metadata(dd, task_id="superglue.boolq.sr", language="sr")
    assert "example_id" in out["train"].column_names
    assert out["train"][0]["example_id"]


def test_attach_task_metadata_overwrites_stale_task_id() -> None:
    ds = Dataset.from_dict(
        {
            "question": ["a", "b"],
            "passage": ["x", "y"],
            "label": [0, 1],
            "task_id": ["wrong.task.en", "wrong.task.en"],
        }
    )
    dd = DatasetDict({"train": ds})
    out = attach_task_metadata(dd, task_id="superglue.boolq.sr", language="sr")
    assert all(v == "superglue.boolq.sr" for v in out["train"]["task_id"])


def test_attach_task_metadata_overwrites_stale_language() -> None:
    ds = Dataset.from_dict({"question": ["a"], "passage": ["x"], "label": [0], "language": ["en"]})
    dd = DatasetDict({"train": ds})
    out = attach_task_metadata(dd, task_id="superglue.boolq.sr", language="sr")
    assert list(out["train"]["language"]) == ["sr"]


def test_attach_task_metadata_preserves_upstream_example_id() -> None:
    ds = Dataset.from_dict(
        {
            "question": ["a", "b"],
            "passage": ["x", "y"],
            "label": [0, 1],
            "example_id": ["upstream-0", "upstream-1"],
        }
    )
    dd = DatasetDict({"train": ds})
    out = attach_task_metadata(dd, task_id="superglue.boolq.sr", language="sr")
    # example_id is the stable key for matching predictions to private labels.
    # Only fill if missing; never overwrite.
    assert list(out["train"]["example_id"]) == ["upstream-0", "upstream-1"]

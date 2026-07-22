"""Unit tests for `scripts/publish_sle_dataset.py`'s `map_source_filenames`.

The script lives under `scripts/` (not an importable package), so it is
loaded directly from its file path via `importlib`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "publish_sle_dataset.py"

_spec = importlib.util.spec_from_file_location("publish_sle_dataset", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
publish_sle_dataset = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = publish_sle_dataset
_spec.loader.exec_module(publish_sle_dataset)

map_source_filenames = publish_sle_dataset.map_source_filenames


def test_maps_matching_files_to_task_split_keys() -> None:
    filenames = [
        "arc_challenge_test_partial_0_1172_end.jsonl",
        "boolq_test_partial_0_3270_end.jsonl",
        "nq_open_train_partial_0_87925_end.jsonl",
        "README.md",
        "not_a_task_test_partial_0_1_end.jsonl",
    ]

    mapping = map_source_filenames(filenames)

    assert mapping == {
        ("arc_challenge", "test"): "arc_challenge_test_partial_0_1172_end.jsonl",
        ("boolq", "test"): "boolq_test_partial_0_3270_end.jsonl",
        ("nq_open", "train"): "nq_open_train_partial_0_87925_end.jsonl",
    }


def test_duplicate_task_split_mapping_raises_value_error() -> None:
    first = "boolq_test_partial_0_3270_end.jsonl"
    second = "boolq_test_partial_0_3271_end.jsonl"

    with pytest.raises(ValueError) as exc_info:
        map_source_filenames([first, second])

    message = str(exc_info.value)
    assert "boolq" in message
    assert "test" in message
    assert first in message
    assert second in message


def test_missing_required_key_is_reported_by_caller() -> None:
    # `map_source_filenames` itself never raises for a key that is never
    # found - that is the caller's job (see `required_source_keys` /
    # the `missing` check in `main`). Confirm the mapping simply omits it.
    mapping = map_source_filenames(["arc_challenge_test_partial_0_1172_end.jsonl"])

    required = publish_sle_dataset.required_source_keys()
    missing = [key for key in required if key not in mapping]

    assert ("boolq", "test") in missing
    assert ("arc_challenge", "test") not in missing

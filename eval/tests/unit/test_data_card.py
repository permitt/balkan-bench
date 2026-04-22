"""Tests for `balkanbench.data.card`."""

from __future__ import annotations

from balkanbench.data.card import render_dataset_card


def _manifest() -> dict:
    return {
        "benchmark": "superglue",
        "language": "sr",
        "public_repo": "permitt/superglue-serbian",
        "private_repo": "permitt/superglue-private",
        "dataset_revision": "v0.1.0-data",
        "license": "CC-BY-4.0",
        "hidden_test_labels": True,
        "configs": {
            "boolq": {
                "splits": {
                    "train": {"num_rows": 9427, "has_labels": True},
                    "validation": {"num_rows": 3270, "has_labels": True},
                    "test": {"num_rows": 3245, "has_labels": False},
                },
                "fields": ["example_id", "question", "passage", "language", "task_id"],
            },
            "cb": {
                "splits": {
                    "train": {"num_rows": 250, "has_labels": True},
                    "validation": {"num_rows": 56, "has_labels": True},
                    "test": {"num_rows": 250, "has_labels": False},
                },
                "fields": ["example_id", "premise", "hypothesis", "language", "task_id"],
            },
        },
    }


def test_card_mentions_sponsor_recrewty() -> None:
    out = render_dataset_card(_manifest())
    assert "Recrewty" in out


def test_card_flags_hidden_test_labels() -> None:
    out = render_dataset_card(_manifest())
    assert "test labels are hidden" in out.lower()
    assert "balkanbench predict" in out or "balkanbench score" in out


def test_card_lists_every_config_and_split() -> None:
    out = render_dataset_card(_manifest())
    assert "boolq" in out
    assert "cb" in out
    assert "9427" in out  # BoolQ train rows
    assert "56" in out  # CB validation rows


def test_card_exposes_revision_and_license() -> None:
    out = render_dataset_card(_manifest())
    assert "v0.1.0-data" in out
    assert "CC-BY-4.0" in out


def test_card_starts_with_yaml_frontmatter() -> None:
    out = render_dataset_card(_manifest())
    assert out.startswith("---\n")
    assert "license: cc-by-4.0" in out.lower() or "license: CC-BY-4.0" in out
    assert "language:" in out

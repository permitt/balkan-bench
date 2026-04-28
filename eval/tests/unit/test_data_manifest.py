"""Tests for `balkanbench.data.manifest`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from jsonschema import Draft202012Validator

from balkanbench.data.manifest import ManifestError, build_manifest

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _tiny_boolq_configs() -> dict[str, DatasetDict]:
    def split(prefix: str, n: int, with_labels: bool) -> Dataset:
        data: dict[str, list] = {
            "example_id": [f"{prefix}-{i}" for i in range(n)],
            "question": [f"q{i}" for i in range(n)],
            "passage": [f"p{i}" for i in range(n)],
        }
        if with_labels:
            data["label"] = [i % 2 for i in range(n)]
        return Dataset.from_dict(data)

    boolq = DatasetDict(
        {
            "train": split("b-train", 8, True),
            "validation": split("b-val", 4, True),
            "test": split("b-test", 4, False),
        }
    )
    return {"boolq": boolq}


def test_build_manifest_matches_schema() -> None:
    configs = _tiny_boolq_configs()
    manifest = build_manifest(
        benchmark="superglue",
        language="sr",
        public_repo="permitt/superglue-serbian",
        private_repo="permitt/superglue-private",
        configs=configs,
        dataset_revision="v0.1.0-data",
        license="CC-BY-4.0",
        hidden_test_labels=True,
    )
    schema = json.loads((SCHEMAS_DIR / "dataset_manifest.json").read_text())
    Draft202012Validator(schema).validate(manifest)

    assert manifest["benchmark"] == "superglue"
    assert manifest["language"] == "sr"
    assert manifest["hidden_test_labels"] is True
    assert manifest["configs"]["boolq"]["splits"]["train"]["num_rows"] == 8
    assert manifest["configs"]["boolq"]["splits"]["test"]["has_labels"] is False
    assert manifest["configs"]["boolq"]["splits"]["validation"]["has_labels"] is True
    assert "example_id" in manifest["configs"]["boolq"]["fields"]


def test_build_manifest_rejects_empty_configs() -> None:
    with pytest.raises(ManifestError):
        build_manifest(
            benchmark="superglue",
            language="sr",
            public_repo="permitt/superglue-serbian",
            private_repo=None,
            configs={},
            dataset_revision="v0.1.0-data",
            license="CC-BY-4.0",
            hidden_test_labels=True,
        )


def test_build_manifest_detects_hidden_labels_mismatch() -> None:
    configs = _tiny_boolq_configs()
    # Add labels back to test to simulate a misconfiguration
    test_with_label = configs["boolq"]["test"].add_column(
        "label", [0] * configs["boolq"]["test"].num_rows
    )
    configs["boolq"] = DatasetDict({**configs["boolq"], "test": test_with_label})
    with pytest.raises(ManifestError, match="labels"):
        build_manifest(
            benchmark="superglue",
            language="sr",
            public_repo="permitt/superglue-serbian",
            private_repo=None,
            configs=configs,
            dataset_revision="v0.1.0-data",
            license="CC-BY-4.0",
            hidden_test_labels=True,
        )

"""Build and validate a dataset manifest from live ``DatasetDict`` configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from jsonschema import Draft202012Validator

LABEL_FIELDS: tuple[str, ...] = ("label",)
"""Column names treated as gold labels. Extend per-task when a task diverges."""


class ManifestError(ValueError):
    """Raised when a manifest cannot be built or fails schema validation."""


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / "dataset_manifest.json"


def _has_labels(column_names: list[str]) -> bool:
    return any(field in column_names for field in LABEL_FIELDS)


def build_manifest(
    *,
    benchmark: str,
    language: str,
    public_repo: str,
    private_repo: str | None,
    configs: dict[str, DatasetDict],
    dataset_revision: str,
    license: str,
    hidden_test_labels: bool,
) -> dict[str, Any]:
    """Build a schema-valid dataset manifest.

    ``configs`` maps task config name (``boolq``, ``cb``, ...) to the
    ``DatasetDict`` as it will be pushed to the public repo.  Raises
    ``ManifestError`` on empty configs, empty splits, or hidden-label
    misconfigurations.
    """
    if not configs:
        raise ManifestError("at least one config is required")

    manifest_configs: dict[str, Any] = {}
    for name, dataset in configs.items():
        if not dataset:
            raise ManifestError(f"config {name!r} has no splits")
        splits: dict[str, Any] = {}
        for split_name, split in dataset.items():
            splits[split_name] = {
                "num_rows": split.num_rows,
                "has_labels": _has_labels(split.column_names),
            }
        if hidden_test_labels and "test" in splits and splits["test"]["has_labels"]:
            raise ManifestError(
                f"config {name!r}: hidden_test_labels=True but test split still carries labels"
            )
        # Fields list uses whichever split is present first; they should be consistent.
        first_split = next(iter(dataset.values()))
        manifest_configs[name] = {
            "splits": splits,
            "fields": list(first_split.column_names),
        }

    manifest: dict[str, Any] = {
        "benchmark": benchmark,
        "language": language,
        "public_repo": public_repo,
        "dataset_revision": dataset_revision,
        "license": license,
        "hidden_test_labels": hidden_test_labels,
        "configs": manifest_configs,
    }
    if private_repo:
        manifest["private_repo"] = private_repo

    schema = json.loads(_schema_path().read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: list(e.path))
    if errors:
        messages = [
            f"  - {'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors
        ]
        raise ManifestError("manifest failed schema:\n" + "\n".join(messages))

    return manifest

"""End-to-end publish orchestrator.

Downloads each requested config from the source HF repo, normalises splits
(``dev`` → ``validation``), strips test labels, attaches task metadata,
builds a schema-valid manifest, renders a dataset card, and uploads
everything to the public repo. The only side-effecting calls are
``datasets.load_dataset`` and ``huggingface_hub.HfApi`` / ``DatasetDict.push_to_hub``.
Tests monkeypatch those entry points; no network is required for CI.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from balkanbench.data.card import render_dataset_card
from balkanbench.data.manifest import LABEL_FIELDS, ManifestError, build_manifest
from balkanbench.data.normalize import (
    attach_task_metadata,
    rename_splits,
    strip_label_columns,
)

if TYPE_CHECKING:
    from datasets import DatasetDict  # noqa: F401


# Lazy seams: datasets.load_dataset and huggingface_hub.HfApi stay out of
# the module's import path until someone actually runs `publish-dataset`.
# Tests that monkeypatch ``balkanbench.data.publish.load_dataset`` / ``HfApi``
# still win because monkeypatch sets the name in the module dict, which is
# consulted before __getattr__.
_LAZY = {
    "load_dataset": ("datasets", "load_dataset"),
    "HfApi": ("huggingface_hub", "HfApi"),
    "DatasetDict": ("datasets", "DatasetDict"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


SPLIT_RENAMES_PER_CONFIG: dict[str, dict[str, str]] = {
    "copa": {"dev": "validation"},
}


class PublishError(RuntimeError):
    """Raised when the publish flow cannot proceed."""


@dataclass
class PublishReport:
    """Summary of a publish run."""

    manifest: dict[str, Any]
    dataset_card: str
    pushed: bool
    configs: list[str] = field(default_factory=list)
    public_repo: str = ""


def _hf_token_or_raise() -> str:
    token = os.environ.get("HF_OFFICIAL_TOKEN")
    if not token:
        raise PublishError(
            "HF_OFFICIAL_TOKEN is not set. Publishing requires a Hugging Face "
            "access token with write scope on the target repo. "
            "Export HF_OFFICIAL_TOKEN=<token> and retry."
        )
    return token


def _prepare_config(
    source: DatasetDict,
    *,
    config_name: str,
    benchmark: str,
    language: str,
) -> DatasetDict:
    """Rename splits, strip test labels, attach metadata for a single config."""
    rename_map = SPLIT_RENAMES_PER_CONFIG.get(config_name, {})
    ds = rename_splits(rename_map, source) if rename_map else source
    ds = strip_label_columns(ds, split="test", label_fields=list(LABEL_FIELDS))
    ds = attach_task_metadata(
        ds, task_id=f"{benchmark}.{config_name}.{language}", language=language
    )
    return ds


def publish_dataset(
    *,
    source_repo: str,
    public_repo: str,
    private_repo: str | None,
    language: str,
    license: str,
    dataset_revision: str,
    configs_to_publish: list[str],
    benchmark: str = "superglue",
    dry_run: bool = False,
) -> PublishReport:
    """Publish every requested config to ``public_repo``.

    Set ``dry_run=True`` to build and validate the manifest + dataset card
    without creating repos or pushing anything. Returns a ``PublishReport``
    with the manifest dict and card markdown for local inspection.
    """
    token = _hf_token_or_raise()

    prepared: dict[str, DatasetDict] = {}
    for config in configs_to_publish:
        try:
            source = load_dataset(source_repo, config)
        except Exception as exc:
            raise PublishError(
                f"failed to load config {config!r} from {source_repo!r}: {exc}"
            ) from exc
        try:
            prepared[config] = _prepare_config(
                source, config_name=config, benchmark=benchmark, language=language
            )
        except ValueError as exc:
            raise PublishError(f"failed to normalize config {config!r}: {exc}") from exc

    try:
        manifest = build_manifest(
            benchmark=benchmark,
            language=language,
            public_repo=public_repo,
            private_repo=private_repo,
            configs=prepared,
            dataset_revision=dataset_revision,
            license=license,
            hidden_test_labels=True,
        )
    except ManifestError as exc:
        raise PublishError(str(exc)) from exc

    card = render_dataset_card(manifest)

    if dry_run:
        return PublishReport(
            manifest=manifest,
            dataset_card=card,
            pushed=False,
            configs=list(prepared.keys()),
            public_repo=public_repo,
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=public_repo,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    for config_name, dataset in prepared.items():
        dataset.push_to_hub(
            public_repo,
            config_name=config_name,
            revision=dataset_revision,
            token=token,
        )

    card_bytes = io.BytesIO(card.encode("utf-8"))
    api.upload_file(
        path_or_fileobj=card_bytes,
        path_in_repo="README.md",
        repo_id=public_repo,
        repo_type="dataset",
        revision=dataset_revision,
    )
    manifest_bytes = io.BytesIO(json.dumps(manifest, indent=2).encode("utf-8"))
    api.upload_file(
        path_or_fileobj=manifest_bytes,
        path_in_repo="dataset_manifest.json",
        repo_id=public_repo,
        repo_type="dataset",
        revision=dataset_revision,
    )

    return PublishReport(
        manifest=manifest,
        dataset_card=card,
        pushed=True,
        configs=list(prepared.keys()),
        public_repo=public_repo,
    )

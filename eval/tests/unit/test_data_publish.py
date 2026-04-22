"""Tests for `balkanbench.data.publish` orchestrator."""

from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset, DatasetDict

from balkanbench.data.publish import (
    PublishError,
    PublishReport,
    publish_dataset,
)

# ---------- fixtures ----------


def _mini_superglue_source(has_dev_on_copa: bool = True) -> dict[str, DatasetDict]:
    """Build an in-memory fake of what `permitt/superglue` looks like."""

    def boolq_split(n: int, prefix: str) -> Dataset:
        return Dataset.from_dict(
            {
                "question": [f"q{i}" for i in range(n)],
                "passage": [f"p{i}" for i in range(n)],
                "label": [i % 2 for i in range(n)],
            }
        )

    def copa_split(n: int, prefix: str) -> Dataset:
        return Dataset.from_dict(
            {
                "premise": [f"pr{i}" for i in range(n)],
                "choice1": [f"c1_{i}" for i in range(n)],
                "choice2": [f"c2_{i}" for i in range(n)],
                "question": ["cause"] * n,
                "label": [i % 2 for i in range(n)],
            }
        )

    boolq = DatasetDict(
        {
            "train": boolq_split(4, "b-train"),
            "validation": boolq_split(2, "b-val"),
            "test": boolq_split(2, "b-test"),
        }
    )
    val_split_name = "dev" if has_dev_on_copa else "validation"
    copa = DatasetDict(
        {
            "train": copa_split(4, "c-train"),
            val_split_name: copa_split(2, f"c-{val_split_name}"),
            "test": copa_split(2, "c-test"),
        }
    )
    return {"boolq": boolq, "copa": copa}


class _FakeHfApi:
    """Records calls so tests can assert without hitting the network."""

    def __init__(self) -> None:
        self.created_repos: list[dict] = []
        self.uploaded_files: list[dict] = []

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.created_repos.append({"repo_id": repo_id, **kwargs})

    def upload_file(
        self,
        *,
        path_or_fileobj: Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        **kwargs: Any,
    ) -> None:
        content = path_or_fileobj.read() if hasattr(path_or_fileobj, "read") else path_or_fileobj
        if isinstance(content, bytes):
            content = content.decode()
        self.uploaded_files.append(
            {
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "content": content,
            }
        )


@pytest.fixture
def fake_source(monkeypatch):
    source = _mini_superglue_source()

    def fake_load_dataset(repo: str, config: str, **_: Any) -> DatasetDict:
        if repo != "permitt/superglue":
            raise AssertionError(f"unexpected source repo {repo}")
        if config not in source:
            raise ValueError(f"Config {config!r} not in fake source")
        return source[config]

    monkeypatch.setattr("balkanbench.data.publish.load_dataset", fake_load_dataset)
    return source


@pytest.fixture
def fake_api(monkeypatch):
    api = _FakeHfApi()
    monkeypatch.setattr("balkanbench.data.publish.HfApi", lambda token=None: api)
    pushes: list[dict] = []

    def fake_push_to_hub(self: DatasetDict, repo_id: str, **kwargs: Any) -> None:
        pushes.append({"repo_id": repo_id, **kwargs, "splits": dict(self)})

    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push_to_hub, raising=False)
    return api, pushes


# ---------- env guard ----------


def test_publish_requires_token(monkeypatch, fake_source) -> None:
    monkeypatch.delenv("HF_OFFICIAL_TOKEN", raising=False)
    with pytest.raises(PublishError, match="HF_OFFICIAL_TOKEN"):
        publish_dataset(
            source_repo="permitt/superglue",
            public_repo="permitt/superglue-serbian",
            private_repo="permitt/superglue-private",
            language="sr",
            license="CC-BY-4.0",
            dataset_revision="v0.1.0-data",
            configs_to_publish=["boolq", "copa"],
        )


# ---------- dry run ----------


def test_dry_run_returns_manifest_and_card(monkeypatch, fake_source) -> None:
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")
    report = publish_dataset(
        source_repo="permitt/superglue",
        public_repo="permitt/superglue-serbian",
        private_repo="permitt/superglue-private",
        language="sr",
        license="CC-BY-4.0",
        dataset_revision="v0.1.0-data",
        configs_to_publish=["boolq", "copa"],
        dry_run=True,
    )
    assert isinstance(report, PublishReport)
    assert report.manifest["benchmark"] == "superglue"
    assert "boolq" in report.manifest["configs"]
    assert "copa" in report.manifest["configs"]
    # copa dev split must have been renamed to validation
    assert "validation" in report.manifest["configs"]["copa"]["splits"]
    assert "dev" not in report.manifest["configs"]["copa"]["splits"]
    # Hidden test labels held
    assert report.manifest["hidden_test_labels"] is True
    assert report.manifest["configs"]["boolq"]["splits"]["test"]["has_labels"] is False
    # Card present and mentions Recrewty
    assert "Recrewty" in report.dataset_card
    # No HF side effects on dry run
    assert report.pushed is False


# ---------- full run with mocks ----------


def test_full_run_creates_repo_and_pushes_every_config(monkeypatch, fake_source, fake_api) -> None:
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")
    api, pushes = fake_api
    report = publish_dataset(
        source_repo="permitt/superglue",
        public_repo="permitt/superglue-serbian",
        private_repo="permitt/superglue-private",
        language="sr",
        license="CC-BY-4.0",
        dataset_revision="v0.1.0-data",
        configs_to_publish=["boolq", "copa"],
    )
    assert report.pushed is True
    # Repo created
    assert any(r["repo_id"] == "permitt/superglue-serbian" for r in api.created_repos)
    # Exactly one push per config
    pushed_configs = [p.get("config_name") for p in pushes]
    assert set(pushed_configs) == {"boolq", "copa"}
    # Manifest + README uploaded
    uploaded_paths = {f["path_in_repo"] for f in api.uploaded_files}
    assert "README.md" in uploaded_paths
    assert "dataset_manifest.json" in uploaded_paths


def test_full_run_strips_test_labels_before_push(monkeypatch, fake_source, fake_api) -> None:
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")
    api, pushes = fake_api
    publish_dataset(
        source_repo="permitt/superglue",
        public_repo="permitt/superglue-serbian",
        private_repo=None,
        language="sr",
        license="CC-BY-4.0",
        dataset_revision="v0.1.0-data",
        configs_to_publish=["boolq"],
    )
    boolq_push = next(p for p in pushes if p.get("config_name") == "boolq")
    test_split = boolq_push["splits"]["test"]
    assert "label" not in test_split.column_names


def test_full_run_renames_copa_dev_before_push(monkeypatch, fake_source, fake_api) -> None:
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")
    _, pushes = fake_api
    publish_dataset(
        source_repo="permitt/superglue",
        public_repo="permitt/superglue-serbian",
        private_repo=None,
        language="sr",
        license="CC-BY-4.0",
        dataset_revision="v0.1.0-data",
        configs_to_publish=["copa"],
    )
    copa_push = next(p for p in pushes if p.get("config_name") == "copa")
    assert set(copa_push["splits"].keys()) == {"train", "validation", "test"}


def test_idempotent_rename_when_copa_already_has_validation(monkeypatch) -> None:
    """If upstream already renamed dev->validation, publish still works."""

    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")
    source = _mini_superglue_source(has_dev_on_copa=False)

    def fake_load(_: str, config: str, **__: Any) -> DatasetDict:
        return source[config]

    monkeypatch.setattr("balkanbench.data.publish.load_dataset", fake_load)

    report = publish_dataset(
        source_repo="permitt/superglue",
        public_repo="permitt/superglue-serbian",
        private_repo=None,
        language="sr",
        license="CC-BY-4.0",
        dataset_revision="v0.1.0-data",
        configs_to_publish=["copa"],
        dry_run=True,
    )
    assert "validation" in report.manifest["configs"]["copa"]["splits"]

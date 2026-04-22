"""End-to-end integration test: simulated permitt/superglue → publish flow."""

from __future__ import annotations

import json
from typing import Any

import pytest
from datasets import Dataset, DatasetDict

from balkanbench.data.publish import publish_dataset


def _fake_source_three_configs() -> dict[str, DatasetDict]:
    def split(n: int, with_label: bool, prefix: str, fields: dict[str, list]) -> Dataset:
        data: dict[str, list] = dict(fields)
        if with_label:
            data["label"] = [i % 2 for i in range(n)]
        return Dataset.from_dict(data)

    def boolq(n: int, with_label: bool, prefix: str) -> Dataset:
        return split(
            n,
            with_label,
            prefix,
            {
                "question": [f"q{i}" for i in range(n)],
                "passage": [f"p{i}" for i in range(n)],
            },
        )

    def cb(n: int, with_label: bool, prefix: str) -> Dataset:
        return split(
            n,
            with_label,
            prefix,
            {
                "premise": [f"pr{i}" for i in range(n)],
                "hypothesis": [f"hy{i}" for i in range(n)],
            },
        )

    def copa(n: int, with_label: bool, prefix: str) -> Dataset:
        return split(
            n,
            with_label,
            prefix,
            {
                "premise": [f"pr{i}" for i in range(n)],
                "choice1": [f"c1_{i}" for i in range(n)],
                "choice2": [f"c2_{i}" for i in range(n)],
                "question": ["cause"] * n,
            },
        )

    return {
        "boolq": DatasetDict(
            {
                "train": boolq(4, True, "bq-t"),
                "validation": boolq(2, True, "bq-v"),
                "test": boolq(2, True, "bq-te"),  # labels present in source
            }
        ),
        "cb": DatasetDict(
            {
                "train": cb(4, True, "cb-t"),
                "validation": cb(2, True, "cb-v"),
                "test": cb(2, True, "cb-te"),
            }
        ),
        "copa": DatasetDict(
            {
                "train": copa(4, True, "cp-t"),
                "dev": copa(2, True, "cp-dev"),  # upstream misnomer
                "test": copa(2, True, "cp-te"),
            }
        ),
    }


class _RecordingApi:
    """Minimal HfApi stand-in that records every mutating call."""

    def __init__(self) -> None:
        self.created_repos: list[dict[str, Any]] = []
        self.uploaded_files: list[dict[str, Any]] = []

    def create_repo(self, **kwargs: Any) -> None:
        self.created_repos.append(kwargs)

    def upload_file(
        self,
        *,
        path_or_fileobj: Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        **_: Any,
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
def wired_publish(monkeypatch):
    source = _fake_source_three_configs()

    def fake_load(_repo: str, config: str, **_: Any) -> DatasetDict:
        return source[config]

    monkeypatch.setattr("balkanbench.data.publish.load_dataset", fake_load)

    api = _RecordingApi()
    monkeypatch.setattr("balkanbench.data.publish.HfApi", lambda token=None: api)

    pushes: list[dict[str, Any]] = []

    def fake_push(self: DatasetDict, repo_id: str, **kwargs: Any) -> None:
        pushes.append({"repo_id": repo_id, **kwargs, "splits": dict(self)})

    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push, raising=False)
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")
    return api, pushes


def test_publish_superglue_sr_end_to_end(wired_publish) -> None:
    api, pushes = wired_publish

    report = publish_dataset(
        source_repo="permitt/superglue",
        public_repo="permitt/superglue-serbian",
        private_repo="permitt/superglue-private",
        language="sr",
        license="CC-BY-4.0",
        dataset_revision="v0.1.0-data",
        configs_to_publish=["boolq", "cb", "copa"],
    )

    # Report sanity
    assert report.pushed is True
    assert report.public_repo == "permitt/superglue-serbian"
    assert set(report.configs) == {"boolq", "cb", "copa"}

    # Single dataset repo was created
    assert len(api.created_repos) == 1
    created = api.created_repos[0]
    assert created["repo_id"] == "permitt/superglue-serbian"
    assert created["repo_type"] == "dataset"
    assert created.get("private") is False

    # Exactly one push per config, with the right revision
    assert len(pushes) == 3
    for push in pushes:
        assert push["repo_id"] == "permitt/superglue-serbian"
        assert push["revision"] == "v0.1.0-data"

    # COPA was renamed from dev to validation before push
    copa = next(p for p in pushes if p.get("config_name") == "copa")
    assert set(copa["splits"].keys()) == {"train", "validation", "test"}

    # Every config's test split has no label column
    for push in pushes:
        test_cols = push["splits"]["test"].column_names
        assert "label" not in test_cols, f"{push['config_name']}: test still has label"

    # Manifest and README uploaded at the top level
    paths = {f["path_in_repo"]: f["content"] for f in api.uploaded_files}
    assert "README.md" in paths
    assert "dataset_manifest.json" in paths

    # Manifest matches the report
    uploaded_manifest = json.loads(paths["dataset_manifest.json"])
    assert uploaded_manifest == report.manifest
    assert uploaded_manifest["benchmark"] == "superglue"
    assert uploaded_manifest["language"] == "sr"
    assert uploaded_manifest["hidden_test_labels"] is True
    for name in ("boolq", "cb", "copa"):
        assert uploaded_manifest["configs"][name]["splits"]["test"]["has_labels"] is False

    # Dataset card mentions sponsorship
    assert "Recrewty" in paths["README.md"]
    assert "test labels are hidden" in paths["README.md"].lower()

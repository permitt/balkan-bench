"""Tests for `balkanbench.provenance`."""

from __future__ import annotations

from balkanbench.provenance import collect_provenance


def test_provenance_reports_python_and_package_version() -> None:
    prov = collect_provenance()
    assert prov["python_version"].count(".") == 2
    assert prov["package_version"]
    assert prov["benchmark_name"] == "balkanbench"


def test_provenance_includes_image_digest_from_env(monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_IMAGE_DIGEST", "sha256:deadbeef")
    prov = collect_provenance()
    assert prov["image_digest"] == "sha256:deadbeef"


def test_provenance_defaults_image_digest_when_absent(monkeypatch) -> None:
    monkeypatch.delenv("BALKANBENCH_IMAGE_DIGEST", raising=False)
    prov = collect_provenance()
    # Use an explicit sentinel string so downstream schema validators can
    # distinguish "not provided" from a real digest.
    assert prov["image_digest"].startswith("sha256:")


def test_provenance_includes_torch_version() -> None:
    prov = collect_provenance()
    assert "torch_version" in prov
    assert prov["torch_version"]


def test_provenance_records_git_sha_if_repo_available(tmp_path, monkeypatch) -> None:
    # When we're running in the balkan-bench repo git sha is real; when
    # we're in an unrelated cwd it should fall back to 'unknown'.
    prov = collect_provenance()
    assert "code_revision" in prov
    assert prov["code_revision"]

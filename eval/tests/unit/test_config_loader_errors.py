"""Edge cases for the YAML + schema loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from balkanbench.config import ConfigError, load_yaml_with_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = REPO_ROOT / "eval" / "schemas"


def test_reports_missing_schema(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("benchmark: superglue\n")
    with pytest.raises(ConfigError) as exc:
        load_yaml_with_schema(yaml_path, tmp_path / "nope.json")
    assert "schema file not found" in str(exc.value)


def test_reports_malformed_yaml(tmp_path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("key: [unterminated\n")
    with pytest.raises(ConfigError) as exc:
        load_yaml_with_schema(yaml_path, SCHEMAS_DIR / "task_spec.json")
    assert "failed to parse YAML" in str(exc.value)


def test_rejects_non_mapping_root(tmp_path) -> None:
    yaml_path = tmp_path / "list.yaml"
    yaml_path.write_text("- just\n- a\n- list\n")
    # Schema path doesn't matter because the list fails schema first; use task_spec.
    with pytest.raises(ConfigError):
        load_yaml_with_schema(yaml_path, SCHEMAS_DIR / "task_spec.json")

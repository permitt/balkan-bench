"""Shared YAML + JSON Schema loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


class ConfigError(ValueError):
    """Raised when a YAML config fails schema validation or cannot be read."""


def load_yaml_with_schema(yaml_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load ``yaml_path``, validate against ``schema_path``, return the parsed dict.

    Raises ``ConfigError`` on missing files, YAML parse errors, or schema violations.
    """
    if not yaml_path.is_file():
        raise ConfigError(f"config file not found: {yaml_path}")
    if not schema_path.is_file():
        raise ConfigError(f"schema file not found: {schema_path}")

    try:
        data = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML {yaml_path}: {exc}") from exc

    try:
        schema = json.loads(schema_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"failed to parse schema {schema_path}: {exc}") from exc

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        messages = [
            f"  - {'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors
        ]
        raise ConfigError(f"{yaml_path} failed schema {schema_path.name}:\n" + "\n".join(messages))

    if not isinstance(data, dict):
        raise ConfigError(f"top-level YAML must be a mapping, got {type(data).__name__}")

    return data

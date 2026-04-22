"""``validate-*`` subcommands."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import typer

from balkanbench.config import ConfigError, load_yaml_with_schema

REQUIRED_IMPORTS: tuple[str, ...] = ("typer", "pydantic", "jsonschema", "yaml")
OPTIONAL_ENV_VARS: tuple[str, ...] = ("HF_TOKEN", "HF_OFFICIAL_TOKEN")


def _green(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN, bold=True)


def _yellow(text: str) -> str:
    return typer.style(text, fg=typer.colors.YELLOW, bold=True)


def _red(text: str) -> str:
    return typer.style(text, fg=typer.colors.RED, bold=True)


def validate_env() -> None:
    """Check the Python + dependency + secrets environment."""
    ok = True

    py = sys.version_info
    typer.echo(f"python: {py.major}.{py.minor}.{py.micro}")
    if (py.major, py.minor) < (3, 11):
        typer.echo(_red("  required: >=3.11"))
        ok = False

    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
            typer.echo(f"import {name}: {_green('OK')}")
        except ImportError:
            typer.echo(f"import {name}: {_red('MISSING')}")
            ok = False

    for var in OPTIONAL_ENV_VARS:
        if os.environ.get(var):
            typer.echo(f"env {var}: {_green('present')}")
        else:
            typer.echo(f"env {var}: {_yellow('absent (needed for private labels)')}")

    if not ok:
        raise typer.Exit(code=1)


def validate_config(
    path: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    schema: str = typer.Option(
        "task_spec",
        "--schema",
        "-s",
        help="Schema name under eval/schemas/ (without .json).",
    ),
) -> None:
    """Validate a YAML config against a named JSON Schema."""
    schema_path = _resolve_schema_path(schema)
    try:
        load_yaml_with_schema(path, schema_path)
    except ConfigError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc
    typer.echo(_green(f"OK: {path} is a valid {schema}"))


def validate_data(
    manifest: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
) -> None:
    """Validate a dataset manifest JSON against ``dataset_manifest.json``."""
    schema_path = _resolve_schema_path("dataset_manifest")
    try:
        load_yaml_with_schema(manifest, schema_path)
    except ConfigError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc
    typer.echo(_green(f"OK: {manifest} is a valid dataset manifest"))


def _resolve_schema_path(schema_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "schemas" / f"{schema_name}.json"

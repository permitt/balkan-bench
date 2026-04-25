"""``balkanbench gcs-upload``: copy a local results tree to a GCS prefix.

Used as the trailing step of GCP runs: the orchestrator writes to a
local dir inside the container, then this command syncs the tree to GCS
before the worker tears down. Avoids relying on Vertex AI's
``AIP_MODEL_DIR`` (which is a ``gs://`` URI string, not a FUSE mount,
and silently swallows direct ``Path`` writes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer


def __getattr__(name: str) -> Any:
    if name == "storage":
        from google.cloud import storage as _storage

        return _storage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _red(t: str) -> str:
    return typer.style(t, fg=typer.colors.RED, bold=True)


def _green(t: str) -> str:
    return typer.style(t, fg=typer.colors.GREEN, bold=True)


def gcs_upload_cmd(
    src: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    dest: str = typer.Argument(..., help="gs://bucket/prefix to upload to."),
) -> None:
    """Recursively upload ``src`` directory to ``dest`` (a gs:// URI)."""
    parsed = urlparse(dest)
    if parsed.scheme != "gs" or not parsed.netloc:
        typer.echo(_red(f"dest must be a gs://bucket/prefix URI; got {dest!r}"))
        raise typer.Exit(code=1)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")

    from balkanbench.cli import gcs as _self

    client = _self.storage.Client()
    bucket = client.bucket(bucket_name)

    files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        typer.echo(_red(f"no files under {src}"))
        raise typer.Exit(code=1)

    for path in files:
        rel = path.relative_to(src).as_posix()
        blob_path = f"{prefix}/{rel}" if prefix else rel
        bucket.blob(blob_path).upload_from_filename(str(path))
        typer.echo(_green(f"uploaded gs://{bucket_name}/{blob_path}"))

    typer.echo(_green(f"done: {len(files)} file(s) -> gs://{bucket_name}/{prefix}/"))

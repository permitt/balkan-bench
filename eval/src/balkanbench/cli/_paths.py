"""Config-path resolvers used by the eval / predict / score CLIs."""

from __future__ import annotations

import os
from pathlib import Path


def configs_root() -> Path:
    override = os.environ.get("BALKANBENCH_CONFIGS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "configs"


def resolve_task_config(benchmark: str, task: str) -> Path:
    return configs_root() / "benchmarks" / benchmark / "tasks" / f"{task}.yaml"


def resolve_model_config(model: str) -> Path:
    """Look for the model under official/ first, then experimental/."""
    root = configs_root() / "models"
    for tier in ("official", "experimental"):
        candidate = root / tier / f"{model}.yaml"
        if candidate.is_file():
            return candidate
    # Fall back to a flat layout (helpful during local development).
    flat = root / f"{model}.yaml"
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"model config {model!r} not found under {root}; tried official/, experimental/, and flat."
    )


def schemas_root() -> Path:
    return Path(__file__).resolve().parents[3] / "schemas"

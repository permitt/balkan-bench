"""Resolve the HF dataset repo id and auth token for a (task, language) pair.

Task YAMLs declare a ``dataset.per_language`` map with one
``{public_repo, private_repo}`` entry per BCMS language. Runtime callers
(``eval``, ``predict``, ``hp_search``, ``throughput``, ``score``) go through
this module so the lookup, the language-availability error, and the token
resolution all live in one place.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_TOKEN_ENV_VARS: tuple[str, ...] = ("HF_TOKEN", "HF_OFFICIAL_TOKEN")


class DatasetRepoError(RuntimeError):
    """Raised when the dataset repo cannot be resolved for the language."""


def resolve_dataset_repo(task_cfg: dict[str, Any], language: str, *, prefer: str = "public") -> str:
    """Return the HF repo id for ``language`` from ``task_cfg.dataset.per_language``.

    ``prefer`` selects between the public split (train/validation labels +
    public test inputs) and the private split (hidden test labels, gated).
    Runtime callers default to ``"public"``; only trusted official scoring
    flows should request ``"private"`` explicitly.
    """
    if prefer not in ("public", "private"):
        raise ValueError(f"prefer must be 'public' or 'private'; got {prefer!r}")

    per_lang = task_cfg["dataset"].get("per_language") or {}
    entry = per_lang.get(language)
    if entry is None:
        available = sorted(per_lang)
        raise DatasetRepoError(
            f"task {task_cfg.get('task')!r} has no dataset for language "
            f"{language!r}; available: {available}"
        )
    return entry[f"{prefer}_repo"]


def resolve_hf_token(*, required: bool = False) -> str | None:
    """Return the first non-empty token from HF_TOKEN, HF_OFFICIAL_TOKEN.

    The published BCMS SuperGLUE datasets are gated, so callers that load
    them must pass a token. ``HF_TOKEN`` is HF's standard env var; we keep
    ``HF_OFFICIAL_TOKEN`` as a fallback for the official scoring flow that
    historically used it.
    """
    for var in DEFAULT_TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    if required:
        raise DatasetRepoError(
            "No Hugging Face token found in env. Set HF_TOKEN (preferred) "
            "or HF_OFFICIAL_TOKEN before running."
        )
    return None

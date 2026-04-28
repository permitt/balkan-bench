"""Run provenance collector.

Builds the ``{code_revision, python_version, torch_version, ...}`` dict that
ends up in every result artifact so a downstream reader can always trace a
score back to the exact build that produced it.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

from balkanbench import __version__


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=_repo_root(),
        )
    except FileNotFoundError:
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # eval/src/balkanbench/ -> eval/src -> eval -> repo root
    return os.path.normpath(os.path.join(here, "..", "..", ".."))


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:
        return "unknown"


def _transformers_version() -> str:
    try:
        import transformers

        return str(transformers.__version__)
    except ImportError:
        return "unknown"


def _cuda_version() -> str:
    try:
        import torch

        return str(torch.version.cuda) if torch.version.cuda else "cpu"
    except (ImportError, AttributeError):
        return "unknown"


def collect_provenance() -> dict[str, Any]:
    """Return a provenance dict. Fields are all strings so JSON serialisation is straightforward."""
    return {
        "benchmark_name": "balkanbench",
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": _torch_version(),
        "transformers_version": _transformers_version(),
        "cuda_version": _cuda_version(),
        "code_revision": _git_sha(),
        "image_digest": os.environ.get("BALKANBENCH_IMAGE_DIGEST", "sha256:local"),
    }

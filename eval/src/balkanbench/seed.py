"""Seed helper.

Sets the Python / numpy / torch / transformers seeds so downstream runs are
reproducible for the same seed value. No CUDA-specific determinism toggles;
those are left to callers who care about bit-exact reproducibility.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python, numpy, torch, and transformers."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass

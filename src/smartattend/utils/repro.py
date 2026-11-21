"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy (and torch if installed); return a NumPy Generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)

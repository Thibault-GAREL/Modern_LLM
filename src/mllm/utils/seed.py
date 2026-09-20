"""Reproducibility helpers.

``set_determinism`` seeds python, numpy and torch in one call. Full
determinism (``torch.use_deterministic_algorithms``) stays behind a flag
because it disables faster non-deterministic kernels and requires
``CUBLAS_WORKSPACE_CONFIG`` to be set on CUDA.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_determinism(seed: int = 42, *, deterministic_algorithms: bool = False) -> None:
    """Seed every RNG the library touches.

    Args:
        seed: shared seed for python, numpy and torch (CPU and all CUDA devices).
        deterministic_algorithms: if True, force deterministic kernels
            (slower, and some ops raise if no deterministic implementation
            exists). Needed for bitwise-reproducible CUDA runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA devices
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

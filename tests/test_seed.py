"""set_determinism must make torch RNG reproducible (M0)."""

from __future__ import annotations

import torch

from mllm.utils.seed import set_determinism


def test_same_seed_same_tensors():
    set_determinism(123)
    a = torch.randn(4, 4)
    set_determinism(123)
    b = torch.randn(4, 4)
    assert torch.equal(a, b)


def test_different_seed_differs():
    set_determinism(1)
    a = torch.randn(4, 4)
    set_determinism(2)
    b = torch.randn(4, 4)
    assert not torch.equal(a, b)

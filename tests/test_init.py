"""Initialization (M1): scaled residual init and the muP width scaling.

The toy stack below is deliberately not a Transformer: it is the smallest
thing that has the property under test, a deep chain of residual branches
whose output projection writes back into the stream.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mllm.config import InitConfig, ModelConfig, MuPConfig
from mllm.init import (
    init_weights,
    is_residual_projection,
    mark_output_layer,
    mark_residual_projection,
    output_logit_multiplier,
    width_multiplier,
)
from mllm.layers.norm import RMSNorm
from mllm.utils.seed import set_determinism


class ToyBlock(nn.Module):
    """One pre-norm residual branch, shaped like a real FFN sub-block."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, 4 * d_model, bias=False)
        self.down = mark_residual_projection(nn.Linear(4 * d_model, d_model, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(F.gelu(self.up(self.norm(x))))


class ToyStack(nn.Module):
    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ToyBlock(d_model) for _ in range(n_layers))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the activation after every block, for depth profiling."""
        acts = []
        for block in self.blocks:
            x = block(x)
            acts.append(x)
        return acts


def _build(d_model: int, n_layers: int, **kwargs) -> ToyStack:
    cfg = ModelConfig(d_model=d_model, n_layers=n_layers, **kwargs)
    model = ToyStack(d_model, n_layers)
    init_weights(model, cfg)
    return model


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


# ---------------------------------------------------------------------------
# Marking
# ---------------------------------------------------------------------------


def test_residual_projections_are_marked():
    block = ToyBlock(32)
    assert is_residual_projection(block.down)
    assert not is_residual_projection(block.up)


# ---------------------------------------------------------------------------
# Scaled residual init
# ---------------------------------------------------------------------------


def test_scaled_residual_divides_output_projection_std():
    n_layers = 8
    scaled = _build(
        64, n_layers, init=InitConfig(scheme="fixed", std=0.02, scaled_residual=True)
    )
    plain = _build(
        64, n_layers, init=InitConfig(scheme="fixed", std=0.02, scaled_residual=False)
    )
    expected_ratio = math.sqrt(2.0 * n_layers)
    ratio = plain.blocks[0].down.weight.std() / scaled.blocks[0].down.weight.std()
    assert abs(ratio - expected_ratio) / expected_ratio < 0.05
    # the input projection is untouched by the depth correction
    torch.testing.assert_close(
        plain.blocks[0].up.weight.std(), scaled.blocks[0].up.weight.std(), rtol=0.05, atol=0
    )


def test_activations_stay_bounded_with_depth():
    """Without the correction the residual stream grows with depth."""
    n_layers = 32
    x = torch.randn(8, 16, 128)
    init_kwargs = {"scheme": "inv_sqrt_d"}

    scaled = _build(
        128, n_layers, init=InitConfig(**init_kwargs, scaled_residual=True)
    )
    plain = _build(
        128, n_layers, init=InitConfig(**init_kwargs, scaled_residual=False)
    )

    with torch.no_grad():
        scaled_growth = scaled(x)[-1].std().item() / x.std().item()
        plain_growth = plain(x)[-1].std().item() / x.std().item()

    assert scaled_growth < 2.0, f"scaled init still grew {scaled_growth:.2f}x over 32 layers"
    assert plain_growth > 2 * scaled_growth, (
        f"expected the uncorrected stack to blow up, got {plain_growth:.2f}x "
        f"versus {scaled_growth:.2f}x"
    )


# ---------------------------------------------------------------------------
# muP
# ---------------------------------------------------------------------------


def test_width_multiplier_and_output_multiplier():
    off = ModelConfig(d_model=512)
    assert width_multiplier(off) == 1.0
    assert output_logit_multiplier(off) == 1.0

    from mllm.config import AttentionConfig

    on = ModelConfig(
        d_model=512,
        mup=MuPConfig(enabled=True, base_d_model=128),
        attention=AttentionConfig(scale="mup"),
    )
    assert width_multiplier(on) == 4.0
    assert output_logit_multiplier(on) == 0.25


@pytest.mark.parametrize("d_model", [128, 256, 512, 1024])
def test_mup_coord_check_at_init(d_model: int):
    """muP change (a): activation scale must not depend on width.

    This is the init half of the coordinate check. The training half needs
    the muP optimizer groups and lives in bench/coord_check.py.
    """
    from mllm.config import AttentionConfig

    n_layers = 4
    cfg = ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        mup=MuPConfig(enabled=True, base_d_model=128),
        attention=AttentionConfig(scale="mup", n_heads=8, n_kv_heads=2),
        init=InitConfig(scheme="fixed", std=0.02, scaled_residual=True),
    )
    model = ToyStack(d_model, n_layers)
    init_weights(model, cfg)

    x = torch.randn(4, 16, d_model)
    with torch.no_grad():
        branch = model.blocks[0].down(
            F.gelu(model.blocks[0].up(model.blocks[0].norm(x)))
        )
    scale = branch.std().item()

    # reference value computed at the base width with the same seed
    set_determinism(0)
    base_cfg = ModelConfig(
        d_model=128,
        n_layers=n_layers,
        mup=MuPConfig(enabled=True, base_d_model=128),
        attention=AttentionConfig(scale="mup", n_heads=8, n_kv_heads=2),
        init=InitConfig(scheme="fixed", std=0.02, scaled_residual=True),
    )
    base_model = ToyStack(128, n_layers)
    init_weights(base_model, base_cfg)
    xb = torch.randn(4, 16, 128)
    with torch.no_grad():
        base_branch = base_model.blocks[0].down(
            F.gelu(base_model.blocks[0].up(base_model.blocks[0].norm(xb)))
        )
    base_scale = base_branch.std().item()

    ratio = scale / base_scale
    assert 0.8 < ratio < 1.25, (
        f"muP should keep activations width-invariant, got {ratio:.3f}x at "
        f"d_model={d_model} versus the base width"
    )


def test_mup_output_layer_uses_a_different_exponent():
    """Table 8 of Yang et al. (2022): hidden variance 1/fan_in, output 1/fan_in^2.

    So the std is divided by sqrt(mult) for hidden matrices and by mult for the
    output one. Confusing the two leaves a visible spread in the coord check.
    """
    from mllm.config import AttentionConfig

    mult = 4.0
    cfg = ModelConfig(
        d_model=512,
        n_layers=2,
        mup=MuPConfig(enabled=True, base_d_model=128),
        attention=AttentionConfig(scale="mup"),
        init=InitConfig(scheme="fixed", std=0.02, scaled_residual=False),
    )

    hidden = nn.Linear(512, 512, bias=False)
    output = mark_output_layer(nn.Linear(512, 100, bias=False))
    model = nn.ModuleDict({"hidden": hidden, "output": output})
    init_weights(model, cfg)

    torch.testing.assert_close(
        hidden.weight.std().item(), 0.02 / math.sqrt(mult), rtol=0.05, atol=0
    )
    torch.testing.assert_close(output.weight.std().item(), 0.02 / mult, rtol=0.05, atol=0)


def test_without_mup_activations_grow_with_width():
    """The contrast case: standard init makes the scale width-dependent."""
    scales = {}
    for d_model in (128, 1024):
        set_determinism(0)
        cfg = ModelConfig(
            d_model=d_model,
            n_layers=2,
            init=InitConfig(scheme="fixed", std=0.02, scaled_residual=True),
        )
        model = ToyStack(d_model, 2)
        init_weights(model, cfg)
        x = torch.randn(4, 16, d_model)
        with torch.no_grad():
            out = model.blocks[0].down(F.gelu(model.blocks[0].up(model.blocks[0].norm(x))))
        scales[d_model] = out.std().item()

    growth = scales[1024] / scales[128]
    assert growth > 2.0, (
        f"standard init should scale with width (expected ~sqrt(8)=2.83), got {growth:.2f}"
    )

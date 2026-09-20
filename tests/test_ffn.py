"""Feed-forward variants (M4)."""

from __future__ import annotations

import pytest
import torch

from mllm.config import FFNConfig, ModelConfig
from mllm.init import is_residual_projection
from mllm.layers.ffn import MLP, GatedMLP, build_ffn, compute_d_ff, round_to_multiple
from mllm.utils.seed import set_determinism

D_MODEL = 256


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def make(kind: str, **kw) -> tuple[torch.nn.Module, ModelConfig]:
    cfg = ModelConfig(d_model=D_MODEL, n_layers=2, ffn=FFNConfig(kind=kind, **kw))
    return build_ffn(cfg), cfg


# ---------------------------------------------------------------------------
# Width and parameter budget
# ---------------------------------------------------------------------------


def test_round_to_multiple_rounds_up():
    assert round_to_multiple(500, 256) == 512
    assert round_to_multiple(512, 256) == 512
    assert round_to_multiple(513, 256) == 768


def test_gated_defaults_to_eight_thirds_and_mlp_to_four():
    gated = compute_d_ff(FFNConfig(kind="swiglu", multiple_of=1), D_MODEL)
    dense = compute_d_ff(FFNConfig(kind="mlp", multiple_of=1), D_MODEL)
    assert gated == int(8 / 3 * D_MODEL)
    assert dense == 4 * D_MODEL


def test_explicit_d_ff_wins_over_the_multiplier():
    assert compute_d_ff(FFNConfig(kind="swiglu", d_ff=999, mult=8.0), D_MODEL) == 999


def test_swiglu_and_mlp_cost_the_same_within_one_percent():
    """The reason the gated width is 8/3 d and not 4 d.

    MLP has ``2 * d * 4d = 8 d^2`` parameters, SwiGLU has
    ``3 * d * (8/3) d = 8 d^2``. Equal by construction, up to rounding.
    """
    d_model = 1024
    mlp_cfg = ModelConfig(d_model=d_model, ffn=FFNConfig(kind="mlp", multiple_of=1))
    swiglu_cfg = ModelConfig(d_model=d_model, ffn=FFNConfig(kind="swiglu", multiple_of=1))
    n_mlp = sum(p.numel() for p in build_ffn(mlp_cfg).parameters())
    n_swiglu = sum(p.numel() for p in build_ffn(swiglu_cfg).parameters())
    ratio = n_swiglu / n_mlp
    assert abs(ratio - 1.0) < 0.01, f"budgets differ by {abs(ratio - 1) * 100:.2f}%"


def test_multiple_of_keeps_shapes_aligned():
    d_ff = compute_d_ff(FFNConfig(kind="swiglu", multiple_of=256), 1024)
    assert d_ff % 256 == 0


# ---------------------------------------------------------------------------
# Shapes and structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mlp", "swiglu", "geglu", "reglu"])
def test_forward_keeps_shape(kind: str):
    ffn, _ = make(kind)
    x = torch.randn(2, 5, D_MODEL)
    assert ffn(x).shape == x.shape


def test_gated_has_three_matrices_and_mlp_two():
    gated, _ = make("swiglu")
    dense, _ = make("mlp")
    assert {"gate_proj", "up_proj", "down_proj"} <= dict(gated.named_children()).keys()
    assert "gate_proj" not in dict(dense.named_children())


def test_down_projection_is_marked_for_scaled_init():
    for kind in ("mlp", "swiglu"):
        ffn, _ = make(kind)
        assert is_residual_projection(ffn.down_proj)
        assert not is_residual_projection(ffn.up_proj)


def test_build_ffn_dispatch():
    assert isinstance(make("mlp")[0], MLP)
    for kind in ("swiglu", "geglu", "reglu"):
        assert isinstance(make(kind)[0], GatedMLP)


def test_gated_variants_differ_from_each_other():
    x = torch.randn(2, 4, D_MODEL)
    outs = []
    for kind in ("swiglu", "geglu", "reglu"):
        set_determinism(5)
        ffn, _ = make(kind)
        outs.append(ffn(x))
    assert not torch.allclose(outs[0], outs[1], rtol=1e-3, atol=1e-3)
    assert not torch.allclose(outs[1], outs[2], rtol=1e-3, atol=1e-3)


def test_gate_actually_gates():
    """A zeroed gate must kill the output, whatever the value path holds."""
    ffn, _ = make("reglu")
    with torch.no_grad():
        ffn.gate_proj.weight.fill_(0.0)  # relu(0) = 0
    out = ffn(torch.randn(2, 4, D_MODEL))
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_no_bias_by_default():
    ffn, _ = make("swiglu")
    assert ffn.up_proj.bias is None and ffn.down_proj.bias is None

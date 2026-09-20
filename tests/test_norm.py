"""Normalization layers (M1).

Two things are checked here: that the fast path matches the literal
transcription of the paper, and that the fp32 policy actually holds in
reduced precision.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mllm.config import NormConfig
from mllm.layers.norm import (
    DyT,
    LayerNorm,
    NormedResidual,
    QKNorm,
    RMSNorm,
    build_norm,
    layer_norm_reference,
    rms_norm_reference,
)
from mllm.utils.seed import set_determinism


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


# ---------------------------------------------------------------------------
# Equivalence with the reference implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit_offset", [False, True])
def test_rmsnorm_matches_reference(unit_offset: bool):
    x = torch.randn(4, 16, 64) * 3
    norm = RMSNorm(64, eps=1e-5, unit_offset=unit_offset)
    with torch.no_grad():
        norm.weight.normal_(0.0, 0.1)
        if not unit_offset:
            norm.weight += 1.0
    expected = rms_norm_reference(x, norm.weight, 1e-5, unit_offset)
    torch.testing.assert_close(norm(x), expected, rtol=1e-6, atol=1e-6)


def test_layernorm_matches_reference():
    x = torch.randn(4, 16, 64) * 3
    norm = LayerNorm(64, eps=1e-5, bias=True)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
        norm.bias.normal_(0.0, 0.1)
    expected = layer_norm_reference(x, norm.weight, norm.bias, 1e-5)
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-5)


def test_unit_offset_equivalent_at_init():
    """Gemma (1 + w, w=0) and LLaMA (w, w=1) conventions agree at init."""
    x = torch.randn(4, 16, 64)
    llama = RMSNorm(64, unit_offset=False)
    gemma = RMSNorm(64, unit_offset=True)
    torch.testing.assert_close(llama(x), gemma(x))


def test_rmsnorm_has_no_bias_and_no_mean_subtraction():
    """The whole point of RMSNorm versus LayerNorm."""
    norm = RMSNorm(64)
    assert not hasattr(norm, "bias") or norm.bias is None
    x = torch.full((1, 1, 64), 5.0)  # constant input, zero variance
    # LayerNorm would map this to zeros, RMSNorm keeps the direction
    assert norm(x).abs().mean() > 0.9


# ---------------------------------------------------------------------------
# fp32 policy
# ---------------------------------------------------------------------------


def test_rmsnorm_computes_in_fp32_not_input_dtype():
    """Our RMSNorm must stay close to the fp32 result in fp16.

    Reference point: an all-fp16 computation of the same formula is roughly
    an order of magnitude further from the fp32 answer.
    """
    x = (torch.randn(4, 32, 128) * 3).half()
    norm = RMSNorm(128).half()

    ours = norm(x).float()
    xf = x.float()
    fp32 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight.float()
    all_fp16 = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight).float()

    err_ours = (ours - fp32).abs().max().item()
    err_fp16 = (all_fp16 - fp32).abs().max().item()
    assert err_ours < err_fp16, f"fp32 path ({err_ours}) should beat fp16 path ({err_fp16})"


def test_torch_rms_norm_does_not_upcast():
    """Documents why F.rms_norm is not used as a fast path.

    torch 2.5.1 computes rms_norm in the input dtype: in fp16 it matches an
    all-fp16 computation exactly, not the fp32 one. If a future torch fixes
    this, this test fails and the fast path becomes usable.
    """
    if not hasattr(F, "rms_norm"):
        pytest.skip("F.rms_norm not available in this torch build")
    x = (torch.randn(4, 16, 128) * 3).half()
    w = torch.ones(128).half()

    torch_out = F.rms_norm(x, (128,), w, 1e-5).float()
    all_fp16 = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w).float()
    xf = x.float()
    fp32 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()

    err_vs_fp16 = (torch_out - all_fp16).abs().max().item()
    err_vs_fp32 = (torch_out - fp32).abs().max().item()
    assert err_vs_fp16 < err_vs_fp32, (
        "F.rms_norm now upcasts to fp32, it can be adopted as the fast path"
    )


def test_norm_preserves_input_dtype():
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(2, 8, 32, dtype=dtype)
        assert RMSNorm(32).to(dtype)(x).dtype == dtype
        assert LayerNorm(32).to(dtype)(x).dtype == dtype
        assert DyT(32).to(dtype)(x).dtype == dtype


# ---------------------------------------------------------------------------
# QK-Norm and DyT
# ---------------------------------------------------------------------------


def test_qk_norm_normalizes_head_dim_separately():
    q = torch.randn(2, 4, 16, 64) * 5
    k = torch.randn(2, 4, 16, 64) * 0.1
    qn, kn = QKNorm(64)(q, k)
    assert qn.shape == q.shape and kn.shape == k.shape
    # after normalization both have RMS ~1 along head_dim, whatever their input scale
    for t in (qn, kn):
        rms = t.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


def test_qk_norm_gains_are_independent():
    qk = QKNorm(64)
    assert qk.q_norm.weight is not qk.k_norm.weight


def test_dyt_is_bounded_and_odd():
    dyt = DyT(32, alpha_init=0.5)
    x = torch.randn(4, 8, 32) * 100
    out = dyt(x)
    assert out.abs().max() <= 1.0 + 1e-5  # gain=1, bias=0 at init
    torch.testing.assert_close(dyt(-x), -out, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


class _Double(torch.nn.Module):
    def forward(self, x):
        return x * 2.0


class _WithAux(torch.nn.Module):
    def forward(self, x):
        return x * 2.0, "aux"


@pytest.mark.parametrize("placement", ["pre", "post", "sandwich"])
def test_placement_shapes_and_identity(placement: str):
    cfg = NormConfig(kind="rmsnorm", placement=placement)
    block = NormedResidual(cfg, 32, _Double())
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.shape == x.shape


def test_post_norm_normalizes_the_residual_stream():
    """The 2017 placement: the output of the whole block is normalized."""
    block = NormedResidual(NormConfig(placement="post"), 32, _Double())
    out = block(torch.randn(2, 8, 32) * 10)
    rms = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


def test_pre_norm_leaves_the_residual_stream_untouched():
    """Pre-norm computes x + f(norm(x)): the input passes through unscaled.

    Two consequences are checked. The branch contribution does not follow the
    input scale (norm eats it), and the output stays aligned with the input
    however large it gets, which is exactly what post-norm destroys.
    """
    block = NormedResidual(NormConfig(placement="pre"), 32, _Double())

    branch_scales = []
    for scale in (0.1, 10.0):
        x = torch.randn(2, 8, 32) * scale
        branch_scales.append((block(x) - x).abs().mean().item())
    ratio = branch_scales[1] / branch_scales[0]
    assert 0.9 < ratio < 1.1, f"branch should be scale invariant, got {ratio:.3f}"

    x = torch.randn(2, 8, 32) * 100
    corr = torch.corrcoef(torch.stack([block(x).flatten(), x.flatten()]))[0, 1]
    assert corr > 0.99, f"pre-norm should preserve the residual stream, corr={corr:.4f}"


def test_sandwich_adds_a_second_norm():
    pre = NormedResidual(NormConfig(placement="pre"), 32, _Double())
    sandwich = NormedResidual(NormConfig(placement="sandwich"), 32, _Double())
    assert pre.norm_out is None
    assert sandwich.norm_out is not None


def test_sublayer_aux_outputs_pass_through():
    block = NormedResidual(NormConfig(placement="pre"), 32, _WithAux())
    out, aux = block(torch.randn(2, 8, 32))
    assert aux == "aux"
    assert out.shape == (2, 8, 32)


def test_build_norm_dispatch():
    assert isinstance(build_norm(NormConfig(kind="rmsnorm"), 16), RMSNorm)
    assert isinstance(build_norm(NormConfig(kind="layernorm"), 16), LayerNorm)
    assert isinstance(build_norm(NormConfig(kind="dyt"), 16), DyT)

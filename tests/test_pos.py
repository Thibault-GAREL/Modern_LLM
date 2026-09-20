"""Positional schemes (M2).

The headline test is ``test_rope_is_relative``: after rotation the dot product
between a query at m and a key at n must depend only on m - n. Everything else
either checks a fast path against the readable reference, or pins down a
formula that is easy to get subtly wrong.
"""

from __future__ import annotations

import math

import pytest
import torch

from mllm.config import PositionConfig, RopeScalingConfig
from mllm.layers.pos import (
    ALiBi,
    LearnedAbsolute,
    NoPE,
    RoPE,
    Sinusoidal,
    apply_rope,
    apply_rope_reference,
    build_inv_freq,
    build_position,
    convert_rope_style,
    rope_style_permutation,
)
from mllm.utils.seed import set_determinism

HEAD_DIM = 64


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


# ---------------------------------------------------------------------------
# The defining property of RoPE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["half", "interleaved"])
def test_rope_is_relative(style: str):
    """<q_m, k_n> must depend on m and n only through m - n."""
    rope = RoPE(HEAD_DIM, style=style, max_seq_len=256)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def dot(m: int, n: int) -> float:
        qr, _ = rope(q, q, torch.tensor([m]))
        _, kr = rope(k, k, torch.tensor([n]))
        return (qr * kr).sum().item()

    for offset in (0, 1, 5, 37):
        reference = dot(offset, 0)
        for shift in (1, 10, 100):
            torch.testing.assert_close(
                dot(offset + shift, shift), reference, rtol=1e-5, atol=1e-5
            )


def test_rope_dot_product_changes_with_distance():
    """Guard against a no-op implementation passing the relativity test."""
    rope = RoPE(HEAD_DIM, max_seq_len=256)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)
    qr, _ = rope(q, q, torch.tensor([0]))
    dots = []
    for n in (0, 8, 64):
        _, kr = rope(k, k, torch.tensor([n]))
        dots.append((qr * kr).sum().item())
    assert len({round(d, 4) for d in dots}) == 3


# ---------------------------------------------------------------------------
# Fast path against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["half", "interleaved"])
def test_apply_rope_matches_reference(style: str):
    rope = RoPE(HEAD_DIM, style=style, max_seq_len=32)
    x = torch.randn(2, 4, 16, HEAD_DIM)
    cos, sin = rope.get_cos_sin(torch.arange(16))
    fast = apply_rope(x, cos, sin, style)
    slow = apply_rope_reference(x, cos, sin, style)
    torch.testing.assert_close(fast, slow, rtol=1e-6, atol=1e-6)


def test_rope_preserves_norm():
    """A rotation is orthogonal, so it cannot change the norm of a head vector."""
    rope = RoPE(HEAD_DIM, max_seq_len=64)
    q = torch.randn(2, 4, 16, HEAD_DIM)
    qr, _ = rope(q, q, torch.arange(16))
    torch.testing.assert_close(qr.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_rope_position_zero_is_identity():
    rope = RoPE(HEAD_DIM, max_seq_len=8)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    qr, _ = rope(q, q, torch.tensor([0]))
    torch.testing.assert_close(qr, q, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# The two conventions and the conversion between them
# ---------------------------------------------------------------------------


def test_permutation_is_a_bijection():
    perm = rope_style_permutation(HEAD_DIM)
    assert torch.equal(torch.sort(perm).values, torch.arange(HEAD_DIM))


def test_styles_differ_without_conversion():
    """The two conventions are both correct and mutually incompatible."""
    rope_h = RoPE(HEAD_DIM, style="half", max_seq_len=32)
    rope_i = RoPE(HEAD_DIM, style="interleaved", max_seq_len=32)
    x = torch.randn(1, 1, 8, HEAD_DIM)
    out_h, _ = rope_h(x, x, torch.arange(8))
    out_i, _ = rope_i(x, x, torch.arange(8))
    assert not torch.allclose(out_h, out_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(("src", "dst"), [("half", "interleaved"), ("interleaved", "half")])
def test_convert_rope_style_preserves_attention_scores(src: str, dst: str):
    """Converting q and k projections must leave every score unchanged."""
    n_heads, d_model, seq = 4, 128, 12
    hidden = torch.randn(seq, d_model)
    sd = {
        "layers.0.attn.q_proj.weight": torch.randn(n_heads * HEAD_DIM, d_model),
        "layers.0.attn.k_proj.weight": torch.randn(n_heads * HEAD_DIM, d_model),
        "layers.0.attn.v_proj.weight": torch.randn(n_heads * HEAD_DIM, d_model),
    }
    converted = convert_rope_style(sd, src, dst, HEAD_DIM)

    def scores(state: dict, style: str) -> torch.Tensor:
        rope = RoPE(HEAD_DIM, style=style, max_seq_len=seq)
        q = (hidden @ state["layers.0.attn.q_proj.weight"].T).view(seq, n_heads, HEAD_DIM)
        k = (hidden @ state["layers.0.attn.k_proj.weight"].T).view(seq, n_heads, HEAD_DIM)
        q, k = q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None]
        qr, kr = rope(q, k, torch.arange(seq))
        return qr @ kr.transpose(-1, -2)

    torch.testing.assert_close(scores(sd, src), scores(converted, dst), rtol=1e-4, atol=1e-4)


def test_convert_rope_style_leaves_v_untouched():
    sd = {
        "q_proj.weight": torch.randn(HEAD_DIM, 32),
        "v_proj.weight": torch.randn(HEAD_DIM, 32),
    }
    out = convert_rope_style(sd, "half", "interleaved", HEAD_DIM)
    torch.testing.assert_close(out["v_proj.weight"], sd["v_proj.weight"])
    assert not torch.allclose(out["q_proj.weight"], sd["q_proj.weight"])


def test_convert_rope_style_roundtrip():
    sd = {"q_proj.weight": torch.randn(2 * HEAD_DIM, 32)}
    there = convert_rope_style(sd, "half", "interleaved", HEAD_DIM)
    back = convert_rope_style(there, "interleaved", "half", HEAD_DIM)
    torch.testing.assert_close(back["q_proj.weight"], sd["q_proj.weight"])


def test_convert_rope_style_rejects_unknown_style():
    with pytest.raises(ValueError, match="rope style"):
        convert_rope_style({}, "half", "rotary", HEAD_DIM)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def test_cos_sin_cache_is_fp32_and_not_persistent():
    rope = RoPE(HEAD_DIM, max_seq_len=16).half()
    assert rope.cos_cached.dtype == torch.float32, "cos/sin must survive .half()"
    assert "cos_cached" not in rope.state_dict()
    assert "sin_cached" not in rope.state_dict()


def test_rope_preserves_input_dtype():
    rope = RoPE(HEAD_DIM, max_seq_len=16)
    for dtype in (torch.float32, torch.float16):
        q = torch.randn(1, 2, 8, HEAD_DIM, dtype=dtype)
        qr, _ = rope(q, q, torch.arange(8))
        assert qr.dtype == dtype


def test_cache_grows_on_demand():
    rope = RoPE(HEAD_DIM, max_seq_len=16)
    assert rope.cos_cached.shape[0] == 16
    q = torch.randn(1, 1, 100, HEAD_DIM)
    rope(q, q, torch.arange(100))
    assert rope.cos_cached.shape[0] >= 100


def test_rope_accepts_non_contiguous_positions():
    """Needed for incremental decoding, where only one position is processed."""
    rope = RoPE(HEAD_DIM, max_seq_len=64)
    q = torch.randn(1, 1, 3, HEAD_DIM)
    out_gathered, _ = rope(q, q, torch.tensor([5, 17, 42]))
    for i, pos in enumerate([5, 17, 42]):
        single, _ = rope(q[:, :, i : i + 1], q[:, :, i : i + 1], torch.tensor([pos]))
        torch.testing.assert_close(out_gathered[:, :, i : i + 1], single)


# ---------------------------------------------------------------------------
# Context extension
# ---------------------------------------------------------------------------


def test_linear_scaling_divides_frequencies():
    device = torch.device("cpu")
    base, _ = build_inv_freq(HEAD_DIM, 10_000.0, None, device)
    scaled, mscale = build_inv_freq(
        HEAD_DIM, 10_000.0, RopeScalingConfig(kind="linear", factor=4.0), device
    )
    torch.testing.assert_close(scaled, base / 4.0)
    assert mscale == 1.0


def test_ntk_aware_raises_theta():
    device = torch.device("cpu")
    base, _ = build_inv_freq(HEAD_DIM, 10_000.0, None, device)
    ntk, _ = build_inv_freq(
        HEAD_DIM, 10_000.0, RopeScalingConfig(kind="ntk-aware", factor=4.0), device
    )
    # highest frequency is nearly untouched, lowest is compressed the most
    assert ntk[0] / base[0] > 0.99
    assert ntk[-1] / base[-1] < 0.30


def test_dynamic_ntk_is_inert_below_the_training_length():
    device = torch.device("cpu")
    cfg = RopeScalingConfig(kind="dynamic-ntk", factor=4.0, original_max_seq_len=2048)
    base, _ = build_inv_freq(HEAD_DIM, 10_000.0, None, device)
    short, _ = build_inv_freq(HEAD_DIM, 10_000.0, cfg, device, seq_len=1024)
    long, _ = build_inv_freq(HEAD_DIM, 10_000.0, cfg, device, seq_len=8192)
    torch.testing.assert_close(short, base)
    assert long[-1] < base[-1]


def test_yarn_interpolates_only_the_slow_bands():
    """The band split is what separates YaRN from plain interpolation."""
    device = torch.device("cpu")
    cfg = RopeScalingConfig(kind="yarn", factor=4.0, original_max_seq_len=2048)
    base, _ = build_inv_freq(HEAD_DIM, 10_000.0, None, device)
    yarn, _ = build_inv_freq(HEAD_DIM, 10_000.0, cfg, device)

    ratio = yarn / base
    assert ratio[0] > 0.99, "fast rotating dims must keep extrapolating"
    torch.testing.assert_close(ratio[-1].item(), 0.25, rtol=0.02, atol=0.01)
    assert torch.all(ratio[1:] <= ratio[:-1] + 1e-6), "ratio must decrease monotonically"


def test_yarn_temperature_formula():
    """sqrt(1/t) = 0.1 * ln(s) + 1, the factor forgotten one time in two."""
    device = torch.device("cpu")
    for s in (2.0, 4.0, 16.0):
        cfg = RopeScalingConfig(kind="yarn", factor=s, attn_temperature=True)
        _, mscale = build_inv_freq(HEAD_DIM, 10_000.0, cfg, device)
        torch.testing.assert_close(mscale, 0.1 * math.log(s) + 1.0, rtol=1e-9, atol=1e-9)


def test_yarn_temperature_can_be_disabled():
    device = torch.device("cpu")
    cfg = RopeScalingConfig(kind="yarn", factor=4.0, attn_temperature=False)
    _, mscale = build_inv_freq(HEAD_DIM, 10_000.0, cfg, device)
    assert mscale == 1.0


def test_yarn_temperature_reaches_cos_sin():
    """The multiplier must actually be applied, not just computed."""
    plain = RoPE(HEAD_DIM, max_seq_len=16)
    scaled = RoPE(
        HEAD_DIM,
        max_seq_len=16,
        scaling=RopeScalingConfig(kind="yarn", factor=8.0, attn_temperature=True),
    )
    assert scaled.mscale > 1.0
    # at position 0 every cos is 1, so the cache directly exposes the multiplier
    torch.testing.assert_close(
        scaled.cos_cached[0].max().item(), scaled.mscale, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(plain.cos_cached[0].max().item(), 1.0, rtol=1e-6, atol=1e-6)


def test_llama3_ramp_keeps_high_frequencies():
    device = torch.device("cpu")
    cfg = RopeScalingConfig(
        kind="llama3", factor=8.0, original_max_seq_len=8192,
        low_freq_factor=1.0, high_freq_factor=4.0,
    )
    base, _ = build_inv_freq(HEAD_DIM, 500_000.0, None, device)
    out, _ = build_inv_freq(HEAD_DIM, 500_000.0, cfg, device)
    ratio = out / base
    assert ratio[0] > 0.99, "short wavelengths must be left alone"
    assert ratio[-1] < 0.2, "long wavelengths must be interpolated"


def test_scaling_lowers_the_effective_rotation_at_long_range():
    """Extension exists so a position past training stays inside the trained arc."""
    long_pos = 8192
    plain = RoPE(HEAD_DIM, max_seq_len=long_pos + 1)
    yarn = RoPE(
        HEAD_DIM,
        max_seq_len=long_pos + 1,
        scaling=RopeScalingConfig(kind="yarn", factor=4.0, original_max_seq_len=2048),
    )
    # the slowest band is the one that never completed a period during training
    plain_angle = math.atan2(
        plain.sin_cached[long_pos, -1].item(), plain.cos_cached[long_pos, -1].item()
    )
    yarn_angle = math.atan2(
        yarn.sin_cached[long_pos, -1].item(), yarn.cos_cached[long_pos, -1].item()
    )
    assert abs(yarn_angle) < abs(plain_angle)


# ---------------------------------------------------------------------------
# ALiBi, NoPE, absolute schemes
# ---------------------------------------------------------------------------


def test_alibi_slopes_are_geometric():
    slopes = ALiBi(8).slopes
    assert slopes.shape == (8,)
    torch.testing.assert_close(slopes[0].item(), 2.0**-1, rtol=1e-6, atol=1e-6)
    ratios = slopes[1:] / slopes[:-1]
    torch.testing.assert_close(ratios, torch.full((7,), 0.5), rtol=1e-5, atol=1e-6)


def test_alibi_handles_non_power_of_two_heads():
    slopes = ALiBi(12).slopes
    assert slopes.shape == (12,)
    assert torch.all(slopes > 0)


def test_alibi_bias_is_relative_and_penalizes_distance():
    bias = ALiBi(4).attn_bias(6, 6, torch.device("cpu"), torch.float32)
    assert bias.shape == (4, 6, 6)
    # zero on the diagonal, more negative the further back the key is
    torch.testing.assert_close(bias[:, 3, 3], torch.zeros(4))
    assert torch.all(bias[:, 3, 0] < bias[:, 3, 2])
    # depends only on the offset
    torch.testing.assert_close(bias[:, 5, 3], bias[:, 4, 2])


def test_alibi_does_not_touch_q_and_k():
    alibi = ALiBi(4)
    q, k = torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16)
    qo, ko = alibi(q, k, torch.arange(8))
    assert qo is q and ko is k


def test_nope_is_a_no_op_everywhere():
    nope = NoPE()
    q, k = torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16)
    qo, ko = nope(q, k, torch.arange(8))
    assert qo is q and ko is k
    assert nope.attn_bias(8, 8, torch.device("cpu"), torch.float32) is None
    assert nope.input_embedding(torch.arange(8), 16) is None


def test_sinusoidal_matches_the_2017_formula():
    dim, pos = 32, 7
    table = Sinusoidal(dim).input_embedding(torch.tensor([pos]), dim)[0]
    for i in range(dim // 2):
        angle = pos / (10_000.0 ** (2 * i / dim))
        torch.testing.assert_close(table[2 * i].item(), math.sin(angle), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(table[2 * i + 1].item(), math.cos(angle), rtol=1e-5, atol=1e-6)


def test_absolute_schemes_do_not_touch_q_and_k():
    for scheme in (Sinusoidal(32), LearnedAbsolute(32, 64)):
        q, k = torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16)
        qo, ko = scheme(q, k, torch.arange(8))
        assert qo is q and ko is k
        assert scheme.attn_bias(8, 8, torch.device("cpu"), torch.float32) is None


def test_learned_absolute_is_trainable():
    scheme = LearnedAbsolute(32, 64)
    assert scheme.input_embedding(torch.arange(8), 32).requires_grad


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("rope", RoPE),
        ("alibi", ALiBi),
        ("sinusoidal", Sinusoidal),
        ("learned", LearnedAbsolute),
        ("nope", NoPE),
    ],
)
def test_build_position_dispatch(kind: str, expected: type):
    scheme = build_position(
        PositionConfig(kind=kind), head_dim=HEAD_DIM, d_model=256, n_heads=8, max_seq_len=128
    )
    assert isinstance(scheme, expected)


def test_build_position_honours_theta_and_style():
    cfg = PositionConfig(kind="rope", rope_theta=500_000.0, rope_style="interleaved")
    rope = build_position(cfg, head_dim=HEAD_DIM, d_model=256, n_heads=8, max_seq_len=128)
    assert rope.theta == 500_000.0
    assert rope.style == "interleaved"


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even head_dim"):
        RoPE(63)

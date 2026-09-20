"""Precision policy (M5).

The global check: running the whole model in reduced precision must stay close
to fp32, and each of the three fp32-critical components must be measurably
better than the same component computed in the reduced dtype.
"""

from __future__ import annotations

import pytest
import torch

from mllm.config import AttentionConfig, FFNConfig, ModelConfig
from mllm.layers.norm import RMSNorm
from mllm.layers.pos import RoPE
from mllm.model import Transformer
from mllm.utils.numerics import (
    autocast_dtype,
    relative_error,
    resolve_precision,
    supports_bf16,
)
from mllm.utils.seed import set_determinism

VOCAB = 64


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def tiny(**kw) -> ModelConfig:
    base = {
        "d_model": 64,
        "n_layers": 3,
        "vocab_size": VOCAB,
        "max_seq_len": 64,
        "attention": AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
        "ffn": FFNConfig(kind="swiglu", multiple_of=1),
    }
    base.update(kw)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_relative_error_is_scale_invariant():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert relative_error(a, a) == 0.0
    assert relative_error(a * 1.01, a) == pytest.approx(
        relative_error(a * 1010, a * 1000), rel=1e-4
    )


def test_autocast_dtype_mapping():
    assert autocast_dtype("bf16") == torch.bfloat16
    assert autocast_dtype("fp16") == torch.float16
    assert autocast_dtype("fp32") is None


def test_bf16_is_not_claimed_on_cpu():
    assert supports_bf16("cpu") is False
    assert resolve_precision("bf16", "cpu") == "fp16"
    assert resolve_precision("fp32", "cpu") == "fp32"


# ---------------------------------------------------------------------------
# The three fp32-critical components
# ---------------------------------------------------------------------------


def test_rmsnorm_beats_a_reduced_precision_reduction():
    """Component 1: the mean square is a reduction over the feature dim."""
    x = (torch.randn(4, 32, 128) * 3).half()
    norm = RMSNorm(128).half()

    xf = x.float()
    fp32 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight.float()
    naive = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight).float()

    assert relative_error(norm(x).float(), fp32) < relative_error(naive, fp32)


def test_rope_tables_stay_fp32_under_half_precision():
    """Component 2: a small angle accumulated over many positions."""
    rope = RoPE(64, max_seq_len=4096).half()
    assert rope.cos_cached.dtype == torch.float32

    q = torch.randn(1, 2, 1, 64)
    far = torch.tensor([4000])
    rotated, _ = rope(q, q, far)
    # a rotation is orthogonal, so any norm drift is pure numerical error
    assert relative_error(rotated.norm(dim=-1), q.norm(dim=-1)) < 1e-5


def test_logits_are_fp32_whatever_the_model_dtype():
    """Component 3: softmax and logsumexp over the vocabulary."""
    model = Transformer(tiny()).half()
    idx = torch.randint(0, VOCAB, (2, 8))
    logits, _, _ = model(idx)
    assert logits.dtype == torch.float32


# ---------------------------------------------------------------------------
# Whole model
# ---------------------------------------------------------------------------


def test_half_precision_forward_stays_close_to_fp32():
    """The global check the milestone asks for."""
    set_determinism(2)
    model = Transformer(tiny())
    idx = torch.randint(0, VOCAB, (2, 16))

    with torch.no_grad():
        fp32_logits, _, _ = model(idx)
        half_logits, _, _ = model.half()(idx)

    error = relative_error(half_logits, fp32_logits)
    assert error < 0.02, f"fp16 forward drifted {error:.4f} from fp32"


def test_half_precision_loss_stays_close_to_fp32():
    set_determinism(2)
    model = Transformer(tiny())
    idx = torch.randint(0, VOCAB, (2, 17))

    with torch.no_grad():
        _, fp32_loss, _ = model(idx[:, :-1], idx[:, 1:])
        _, half_loss, _ = model.half()(idx[:, :-1], idx[:, 1:])

    assert abs(float(half_loss) - float(fp32_loss)) / float(fp32_loss) < 0.01


@pytest.mark.parametrize("kind", ["mha", "gqa", "mqa"])
def test_per_component_drift_is_bounded(kind: str):
    """Per attention variant, so a regression points at the component."""
    n_kv = {"mha": None, "gqa": 2, "mqa": 1}[kind]
    set_determinism(3)
    model = Transformer(tiny(attention=AttentionConfig(kind=kind, n_heads=4, n_kv_heads=n_kv)))
    idx = torch.randint(0, VOCAB, (2, 12))
    with torch.no_grad():
        ref, _, _ = model(idx)
        got, _, _ = model.half()(idx)
    assert relative_error(got, ref) < 0.02


def test_fp32_model_is_bit_identical_to_itself():
    """Guards against hidden non-determinism in the forward pass."""
    model = Transformer(tiny()).eval()
    idx = torch.randint(0, VOCAB, (2, 12))
    with torch.no_grad():
        assert torch.equal(model(idx)[0], model(idx)[0])

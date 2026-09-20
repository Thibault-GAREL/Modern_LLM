"""MatFormer, the nested FFN (Devvrit et al., 2023, arXiv 2310.07707).

Not to be confused with Matryoshka Representation Learning (Kusupati et al.,
2022, arXiv 2205.13147), which nests *embedding dimensions* for retrieval. Both
are called Matryoshka, they solve different problems, and only this one is a
Transformer architecture change. See docs/taxonomy.md.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from mllm.config import AttentionConfig, FFNConfig, ModelConfig, NormConfig, PositionConfig
from mllm.layers.ffn import GatedMLP, MatFormerMLP, build_ffn
from mllm.model import Transformer
from mllm.utils.seed import set_determinism

D_MODEL, VOCAB = 64, 32
GRANULARITIES = [1.0, 0.5, 0.25]


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def cfg(**kw) -> ModelConfig:
    base = {
        "d_model": D_MODEL,
        "n_layers": 2,
        "vocab_size": VOCAB,
        "max_seq_len": 64,
        "attention": AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
        "position": PositionConfig(kind="rope"),
        "norm": NormConfig(kind="rmsnorm", placement="pre"),
        "ffn": FFNConfig(kind="swiglu", multiple_of=1, mat_granularities=GRANULARITIES),
    }
    base.update(kw)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_granularities_must_include_the_full_model():
    with pytest.raises(ValidationError, match="must include 1.0"):
        FFNConfig(mat_granularities=[0.5, 0.25])


def test_granularities_must_be_sorted_and_distinct():
    with pytest.raises(ValidationError, match="sorted"):
        FFNConfig(mat_granularities=[0.5, 1.0])
    with pytest.raises(ValidationError, match="distinct"):
        FFNConfig(mat_granularities=[1.0, 0.5, 0.5])


def test_granularities_must_be_fractions():
    with pytest.raises(ValidationError, match=r"\(0, 1\]"):
        FFNConfig(mat_granularities=[1.0, 0.0])
    with pytest.raises(ValidationError, match=r"\(0, 1\]"):
        FFNConfig(mat_granularities=[1.5, 1.0])


def test_matformer_needs_a_gated_ffn():
    with pytest.raises(ValueError, match="gated FFN kinds only"):
        build_ffn(cfg(ffn=FFNConfig(kind="mlp", mat_granularities=[1.0, 0.5])))


def test_disabled_by_default():
    assert FFNConfig().mat_granularities is None
    assert isinstance(build_ffn(cfg(ffn=FFNConfig(kind="swiglu"))), GatedMLP)
    assert not isinstance(build_ffn(cfg(ffn=FFNConfig(kind="swiglu"))), MatFormerMLP)


# ---------------------------------------------------------------------------
# The nesting property
# ---------------------------------------------------------------------------


def test_smaller_granularities_are_genuine_sub_matrices():
    """The defining property: a smaller model is a slice, not a mask.

    If the small FFN were implemented by zeroing activations it would still
    carry the full parameter count. Slicing the weights is what makes it a
    deployable smaller model.
    """
    ffn = build_ffn(cfg())
    x = torch.randn(2, 5, D_MODEL)

    half_width = ffn.width_for(0.5)
    manual = torch.nn.functional.linear(
        ffn.activation(
            torch.nn.functional.linear(x, ffn.gate_proj.weight[:half_width])
        )
        * torch.nn.functional.linear(x, ffn.up_proj.weight[:half_width]),
        ffn.down_proj.weight[:, :half_width],
    )
    torch.testing.assert_close(ffn(x, 0.5), manual, rtol=1e-6, atol=1e-6)


def test_full_granularity_matches_the_plain_gated_ffn():
    """At 1.0 a MatFormer must be exactly the FFN it nests inside."""
    set_determinism(4)
    mat = build_ffn(cfg())
    set_determinism(4)
    plain = build_ffn(cfg(ffn=FFNConfig(kind="swiglu", multiple_of=1)))
    plain.load_state_dict(mat.state_dict())

    x = torch.randn(2, 5, D_MODEL)
    torch.testing.assert_close(mat(x, 1.0), plain(x), rtol=1e-6, atol=1e-6)


def test_each_granularity_uses_the_expected_width():
    ffn = build_ffn(cfg())
    assert ffn.widths == [ffn.d_ff, round(ffn.d_ff * 0.5), round(ffn.d_ff * 0.25)]
    assert ffn.width_for(1.0) > ffn.width_for(0.5) > ffn.width_for(0.25)


def test_unknown_granularity_snaps_to_the_nearest_declared_one():
    """Mix'n'Match asks for widths that were never explicitly trained."""
    ffn = build_ffn(cfg())
    assert ffn.width_for(0.45) == ffn.width_for(0.5)
    assert ffn.width_for(0.9) == ffn.width_for(1.0)


def test_granularities_produce_different_outputs():
    ffn = build_ffn(cfg())
    x = torch.randn(2, 5, D_MODEL)
    outs = [ffn(x, g) for g in GRANULARITIES]
    assert not torch.allclose(outs[0], outs[1], rtol=1e-3, atol=1e-3)
    assert not torch.allclose(outs[1], outs[2], rtol=1e-3, atol=1e-3)


def test_parameter_count_is_that_of_the_largest_model_only():
    """One set of weights holds every granularity, which is the whole point."""
    set_determinism(5)
    mat = build_ffn(cfg())
    set_determinism(5)
    plain = build_ffn(cfg(ffn=FFNConfig(kind="swiglu", multiple_of=1)))
    assert sum(p.numel() for p in mat.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )


# ---------------------------------------------------------------------------
# Inside a model
# ---------------------------------------------------------------------------


def test_model_runs_at_every_granularity():
    model = Transformer(cfg())
    idx = torch.randint(0, VOCAB, (2, 8))
    for g in GRANULARITIES:
        logits, _, _ = model(idx, granularity=g)
        assert logits.shape == (2, 8, VOCAB)


def test_model_granularity_changes_the_output():
    model = Transformer(cfg()).eval()
    idx = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        full, _, _ = model(idx, granularity=1.0)
        small, _, _ = model(idx, granularity=0.25)
    assert not torch.allclose(full, small, rtol=1e-3, atol=1e-3)


def test_gradients_reach_only_the_used_slice():
    """A backward at 0.25 must leave the outer rows untouched.

    This is what lets the nested models be trained jointly without the small
    one being dragged around by gradients meant for the large one.
    """
    model = Transformer(cfg())
    idx = torch.randint(0, VOCAB, (2, 8))
    _, loss, _ = model(idx, idx, granularity=0.25)
    loss.backward()

    ffn = model.blocks[0].ffn.sublayer
    used = ffn.width_for(0.25)
    grad = ffn.gate_proj.weight.grad
    assert grad[:used].abs().sum() > 0, "the used slice must receive gradient"
    torch.testing.assert_close(
        grad[used:], torch.zeros_like(grad[used:]), rtol=0, atol=0
    )


def test_joint_training_improves_every_granularity():
    """The claim that matters: one run yields several usable models.

    Trained on all granularities at once, each one must end up better than it
    started. Whether it beats a separately trained model of the same size is a
    question for docs/ablations.md, not for a unit test.
    """
    set_determinism(7)
    model = Transformer(cfg(n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)

    # a learnable task: predict the token two steps back
    def batch():
        block = torch.randint(0, VOCAB, (16, 8), generator=gen)
        x = torch.cat([block, block], dim=1)
        return x[:, :-1], x[:, 1:]

    def losses() -> dict[float, float]:
        model.eval()
        gen_eval = torch.Generator().manual_seed(99)
        blk = torch.randint(0, VOCAB, (8, 8), generator=gen_eval)
        xx = torch.cat([blk, blk], dim=1)
        with torch.no_grad():
            out = {g: float(model(xx[:, :-1], xx[:, 1:], granularity=g)[1])
                   for g in GRANULARITIES}
        model.train()
        return out

    before = losses()
    for _ in range(120):
        x, y = batch()
        # every granularity is optimized jointly, which is the training cost
        total = sum(model(x, y, granularity=g)[1] for g in GRANULARITIES)
        opt.zero_grad(set_to_none=True)
        (total / len(GRANULARITIES)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    after = losses()

    for g in GRANULARITIES:
        assert after[g] < before[g], (
            f"granularity {g} did not improve, {before[g]:.3f} to {after[g]:.3f}"
        )

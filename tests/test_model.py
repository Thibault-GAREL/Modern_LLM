"""Block and Transformer assembly (M5)."""

from __future__ import annotations

import pytest
import torch

from mllm.cache import ModelCache, build_model_cache
from mllm.config import (
    AttentionConfig,
    Config,
    FFNConfig,
    ModelConfig,
    MoEConfig,
    NormConfig,
    PositionConfig,
)
from mllm.model import Block, Transformer
from mllm.utils.seed import set_determinism

VOCAB, SEQ, BATCH = 64, 12, 2


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def tiny(**kw) -> ModelConfig:
    base = {
        "d_model": 64,
        "n_layers": 2,
        "vocab_size": VOCAB,
        "max_seq_len": 64,
        "attention": AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
        "ffn": FFNConfig(kind="swiglu", multiple_of=1),
    }
    base.update(kw)
    return ModelConfig(**base)


def ids(t: int = SEQ) -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, t))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placement", ["pre", "post", "sandwich"])
def test_block_runs_under_every_placement(placement: str):
    cfg = tiny(norm=NormConfig(placement=placement))
    block = Block(cfg, 0)
    x = torch.randn(BATCH, SEQ, cfg.d_model)
    out, aux = block(x, torch.arange(SEQ), Transformer(cfg).pos)
    assert out.shape == x.shape
    assert aux is None, "a dense block has no auxiliary loss"


def test_block_uses_moe_only_after_the_dense_prefix():
    cfg = tiny(n_layers=4, moe=MoEConfig(enabled=True, n_experts=4, top_k=2,
                                         n_shared_experts=1, first_k_dense=2))
    uses = [Block(cfg, i).uses_moe for i in range(4)]
    assert uses == [False, False, True, True]


def test_moe_block_returns_an_aux_loss():
    cfg = tiny(moe=MoEConfig(enabled=True, n_experts=4, top_k=2, first_k_dense=0))
    block = Block(cfg, 0)
    _, aux = block(torch.randn(BATCH, SEQ, cfg.d_model), torch.arange(SEQ),
                   Transformer(cfg).pos)
    assert aux is not None and torch.isfinite(aux)


# ---------------------------------------------------------------------------
# Transformer forward
# ---------------------------------------------------------------------------


def test_forward_shapes_and_loss():
    model = Transformer(tiny())
    idx = ids()
    logits, loss, aux = model(idx, idx)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert loss is not None and torch.isfinite(loss)
    assert aux.ce is not None


def test_forward_without_targets_returns_no_loss():
    model = Transformer(tiny())
    logits, loss, aux = model(ids())
    assert loss is None and aux.ce is None
    assert logits.shape == (BATCH, SEQ, VOCAB)


def test_initial_loss_is_near_uniform_entropy():
    """A correctly initialized model starts at ln(vocab_size).

    Too low means something leaks the answer, too high means the init is off.
    Targets are shifted by one, as in real language modelling.
    """
    model = Transformer(tiny())
    idx = ids(SEQ + 1)
    _, loss, _ = model(idx[:, :-1], idx[:, 1:])
    expected = torch.tensor(float(VOCAB)).log()
    torch.testing.assert_close(loss, expected, rtol=0.05, atol=0.0)


def test_tied_embeddings_start_by_predicting_the_input_token():
    """A real consequence of tying, worth knowing before reading a loss curve.

    With logits computed as ``h @ E^T`` and the residual stream still carrying
    the input embedding, an untrained tied model scores the *input* token
    highest. Evaluated against unshifted targets it therefore looks far better
    than chance, which is a measurement artefact and not learning.
    """
    tied = Transformer(tiny(tie_embeddings=True))
    untied = Transformer(tiny(tie_embeddings=False))
    idx = ids()
    uniform = torch.tensor(float(VOCAB)).log()

    _, tied_loss, _ = tied(idx, idx)
    _, untied_loss, _ = untied(idx, idx)
    assert tied_loss < uniform * 0.9, "tying should shortcut the identity task"
    torch.testing.assert_close(untied_loss, uniform, rtol=0.05, atol=0.0)


def test_causality_future_tokens_cannot_change_the_past():
    """The property the whole causal mask exists for."""
    model = Transformer(tiny()).eval()
    idx = ids()
    with torch.no_grad():
        base, _, _ = model(idx)
        altered = idx.clone()
        altered[:, -1] = (altered[:, -1] + 1) % VOCAB
        changed, _, _ = model(altered)
    torch.testing.assert_close(base[:, :-1], changed[:, :-1], rtol=1e-5, atol=1e-5)


def test_tied_embeddings_share_one_matrix():
    tied = Transformer(tiny(tie_embeddings=True))
    untied = Transformer(tiny(tie_embeddings=False))
    assert tied.lm_head.weight is tied.embed.weight
    assert untied.n_params() > tied.n_params()
    assert untied.n_params() - tied.n_params() == VOCAB * 64


@pytest.mark.parametrize("kind", ["rope", "alibi", "nope", "sinusoidal", "learned"])
def test_every_position_scheme_assembles(kind: str):
    model = Transformer(tiny(position=PositionConfig(kind=kind)))
    logits, _, _ = model(ids())
    assert logits.shape == (BATCH, SEQ, VOCAB)


@pytest.mark.parametrize("kind", ["mha", "mqa", "gqa"])
def test_every_dense_attention_kind_assembles(kind: str):
    n_kv = {"mha": None, "mqa": 1, "gqa": 2}[kind]
    model = Transformer(tiny(attention=AttentionConfig(kind=kind, n_heads=4, n_kv_heads=n_kv)))
    assert model(ids())[0].shape == (BATCH, SEQ, VOCAB)


def test_mla_model_assembles():
    cfg = tiny(
        attention=AttentionConfig(
            kind="mla", n_heads=4, n_kv_heads=None, kv_lora_rank=32, q_lora_rank=None,
            qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        )
    )
    assert Transformer(cfg)(ids())[0].shape == (BATCH, SEQ, VOCAB)


# ---------------------------------------------------------------------------
# Shipped profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["base", "llama_style_150m", "moe_1b_a200m", "mla_long_ctx", "gemma_style"],
)
def test_every_shipped_profile_builds_and_runs(name: str):
    """Every config in configs/ must produce a model that actually runs."""
    from pathlib import Path

    cfg = Config.from_yaml(Path(__file__).parents[1] / "configs" / f"{name}.yaml")
    # shrink to something a CPU test can afford, keeping every structural flag
    cfg.model.d_model = 64
    cfg.model.n_layers = 4
    cfg.model.vocab_size = VOCAB
    cfg.model.attention.n_heads = 4
    if cfg.model.attention.kind != "mla":
        cfg.model.attention.n_kv_heads = min(cfg.model.attention.kv_heads, 2)
    else:
        cfg.model.attention.kv_lora_rank = 32
        cfg.model.attention.qk_nope_head_dim = 16
        cfg.model.attention.qk_rope_head_dim = 16
        cfg.model.attention.v_head_dim = 16
        cfg.model.attention.q_lora_rank = None
    cfg.model.ffn.multiple_of = 1
    if cfg.model.moe.enabled:
        cfg.model.moe.n_experts = 4
        cfg.model.moe.top_k = 2
        cfg.model.moe.n_shared_experts = 1
        cfg.model.moe.d_ff_expert = 32

    model = Transformer(cfg.model, z_loss_coef=cfg.train.z_loss_coef)
    idx = ids()
    logits, loss, _ = model(idx, idx)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------


def test_moe_and_z_loss_are_reported_separately():
    cfg = tiny(n_layers=3, moe=MoEConfig(enabled=True, n_experts=4, top_k=2,
                                         first_k_dense=1))
    model = Transformer(cfg, z_loss_coef=1e-4)
    idx = ids()
    _, loss, aux = model(idx, idx)
    assert aux.moe is not None and aux.z_loss is not None
    torch.testing.assert_close(loss, aux.ce + aux.moe + aux.z_loss, rtol=1e-5, atol=1e-6)
    assert set(aux.as_dict()) >= {"loss/ce", "loss/moe", "loss/z_loss"}


def test_routing_metrics_reach_the_model_level():
    cfg = tiny(n_layers=3, moe=MoEConfig(enabled=True, n_experts=8, top_k=2,
                                         first_k_dense=1))
    model = Transformer(cfg)
    idx = ids()
    model(idx, idx)
    metrics = model.routing_metrics()
    assert len(metrics) == 2, "one per MoE layer"
    assert all(m.load_fraction.shape == (8,) for m in metrics)
    model.balance_step()  # must not raise


def test_mtp_loss_is_included():
    cfg = tiny(mtp_depth=2, tie_embeddings=True)
    model = Transformer(cfg)
    idx = ids()
    _, loss, aux = model(idx, idx)
    assert aux.mtp is not None and aux.mtp > 0
    torch.testing.assert_close(loss, aux.ce + aux.mtp, rtol=1e-5, atol=1e-6)


def test_active_params_below_total_under_moe():
    cfg = tiny(n_layers=4, moe=MoEConfig(enabled=True, n_experts=16, top_k=2,
                                         d_ff_expert=32, first_k_dense=1))
    model = Transformer(cfg)
    assert model.n_active_params() < model.n_params()


# ---------------------------------------------------------------------------
# Cache parity, end to end
# ---------------------------------------------------------------------------


def test_incremental_decoding_matches_full_forward():
    """The test that catches nearly every cache and mask mistake."""
    model = Transformer(tiny()).eval()
    idx = torch.randint(0, VOCAB, (1, SEQ))
    with torch.no_grad():
        full, _, _ = model(idx)
        cache = build_model_cache(model.cfg, max_len=SEQ)
        steps = [model(idx[:, i : i + 1], cache=cache)[0] for i in range(SEQ)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-4, atol=1e-4)


def test_incremental_decoding_with_alternating_windows():
    """Local layers use a ring buffer while global ones stay dense."""
    cfg = tiny(
        n_layers=4,
        attention=AttentionConfig(
            kind="gqa", n_heads=4, n_kv_heads=2, sliding_window=4, global_every=2
        ),
    )
    model = Transformer(cfg).eval()
    idx = torch.randint(0, VOCAB, (1, SEQ))
    cache = build_model_cache(cfg, max_len=SEQ)
    from mllm.cache import KVCache, RingCache

    assert isinstance(cache[0], RingCache) and isinstance(cache[1], KVCache)
    with torch.no_grad():
        full, _, _ = model(idx)
        steps = [model(idx[:, i : i + 1], cache=cache)[0] for i in range(SEQ)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-4, atol=1e-4)


def test_model_cache_reports_total_cost():
    cfg = tiny(n_layers=4)
    cache = build_model_cache(cfg, max_len=32)
    assert isinstance(cache, ModelCache) and len(cache) == 4
    assert cache.bytes_per_token() == 4 * 2 * 2 * 16 * 2  # layers, kv, heads, dim, fp16


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------


def test_gradient_checkpointing_gives_the_same_gradients():
    """Recomputing activations must not change what is learned."""
    set_determinism(4)
    model = Transformer(tiny())
    idx = ids()

    _, loss, _ = model(idx, idx)
    loss.backward()
    plain = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing = True
    model.train()
    _, loss_ckpt, _ = model(idx, idx)
    loss_ckpt.backward()
    checkpointed = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    torch.testing.assert_close(loss, loss_ckpt, rtol=1e-5, atol=1e-6)
    for a, b in zip(plain, checkpointed, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)

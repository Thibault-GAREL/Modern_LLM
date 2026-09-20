"""Attention (M3).

Two tests carry the milestone: GQA with as many KV heads as query heads must
reproduce MHA, and the MLA absorbed path must produce the same logits as the
naive one. The rest pins down the masks and the fallbacks.
"""

from __future__ import annotations

import pytest
import torch

from mllm.cache import KVCache, LatentCache
from mllm.config import AttentionConfig, ModelConfig, PositionConfig
from mllm.layers.attention import Attention, build_attention_mask, repeat_kv
from mllm.layers.pos import build_position
from mllm.utils.seed import set_determinism

D_MODEL, N_HEADS, SEQ, BATCH = 128, 8, 12, 2


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def make(att: AttentionConfig, layer_idx: int = 0, **kw) -> tuple[Attention, torch.nn.Module]:
    cfg = ModelConfig(d_model=D_MODEL, n_layers=2, attention=att, **kw)
    attn = Attention(cfg, layer_idx)
    pos = build_position(
        cfg.position, head_dim=cfg.rope_head_dim, d_model=D_MODEL, n_heads=att.n_heads,
        max_seq_len=256,
    )
    return attn, pos


def run(attn, pos, x=None, **kw) -> torch.Tensor:
    if x is None:
        x = torch.randn(BATCH, SEQ, D_MODEL)
    return attn(x, torch.arange(x.shape[1]), pos, **kw)


# ---------------------------------------------------------------------------
# repeat_kv
# ---------------------------------------------------------------------------


def test_repeat_kv_matches_repeat_interleave():
    x = torch.randn(2, 2, 5, 16)
    torch.testing.assert_close(repeat_kv(x, 3), x.repeat_interleave(3, dim=1))


def test_repeat_kv_is_a_noop_for_one():
    x = torch.randn(2, 4, 5, 16)
    assert repeat_kv(x, 1) is x


def _sdpa_supports_enable_gqa() -> bool:
    """torch added the keyword in 2.5, and a pod may well run 2.4."""
    import torch.nn.functional as F

    try:
        F.scaled_dot_product_attention(
            torch.zeros(1, 2, 2, 4), torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4),
            enable_gqa=True,
        )
    except TypeError:
        return False
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA backend")
@pytest.mark.skipif(
    not _sdpa_supports_enable_gqa(), reason="this torch has no enable_gqa keyword"
)
def test_enable_gqa_still_regresses():
    """Locks in why ``USE_SDPA_ENABLE_GQA`` is off.

    ``enable_gqa=True`` should be strictly better than expanding the KV heads.
    On torch 2.5.1 it falls back to the math backend and materializes the full
    score matrix instead. When a torch release fixes this, the assertion below
    fails and the flag can be turned on.
    """
    import gc

    import torch.nn.functional as F

    b, h, kv, t, d = 4, 8, 2, 1024, 64

    def peak(use_gqa: bool) -> float:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        q = torch.randn(b, h, t, d, device="cuda", dtype=torch.float16)
        k = torch.randn(b, kv, t, d, device="cuda", dtype=torch.float16)
        v = torch.randn(b, kv, t, d, device="cuda", dtype=torch.float16)
        base = torch.cuda.memory_allocated()
        if use_gqa:
            F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            F.scaled_dot_product_attention(
                q, repeat_kv(k, h // kv), repeat_kv(v, h // kv), is_causal=True
            )
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 2**20

    assert peak(use_gqa=True) > 4 * peak(use_gqa=False), (
        "enable_gqa no longer materializes the score matrix, "
        "set USE_SDPA_ENABLE_GQA = True and re-run bench/throughput.py"
    )


# ---------------------------------------------------------------------------
# Head layouts
# ---------------------------------------------------------------------------


def test_gqa_with_full_kv_heads_equals_mha():
    """The headline equivalence: GQA is a generalization, not a different op."""
    x = torch.randn(BATCH, SEQ, D_MODEL)
    set_determinism(1)
    mha, pos = make(AttentionConfig(kind="mha", n_heads=N_HEADS, n_kv_heads=None))
    set_determinism(1)
    gqa, _ = make(AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=N_HEADS))
    gqa.load_state_dict(mha.state_dict())
    torch.testing.assert_close(run(mha, pos, x), run(gqa, pos, x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("kind", "n_kv"), [("mha", None), ("mqa", 1), ("gqa", 2), ("gqa", 4)]
)
def test_head_layouts_run_and_keep_shape(kind: str, n_kv: int | None):
    attn, pos = make(AttentionConfig(kind=kind, n_heads=N_HEADS, n_kv_heads=n_kv))
    assert run(attn, pos).shape == (BATCH, SEQ, D_MODEL)


def test_fewer_kv_heads_shrinks_the_cache():
    """The only reason MQA and GQA exist."""
    mha, _ = make(AttentionConfig(kind="mha", n_heads=N_HEADS, n_kv_heads=None))
    gqa, _ = make(AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2))
    mqa, _ = make(AttentionConfig(kind="mqa", n_heads=N_HEADS, n_kv_heads=1))
    assert mha.bytes_per_token() == 4 * gqa.bytes_per_token()
    assert mha.bytes_per_token() == 8 * mqa.bytes_per_token()


def test_gqa_groups_share_kv():
    """Two query heads in the same group must see identical keys."""
    attn, pos = make(AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2))
    x = torch.randn(1, SEQ, D_MODEL)
    k = attn.k_proj(x).view(1, SEQ, 2, attn.head_dim).transpose(1, 2)
    expanded = repeat_kv(k, 2)
    torch.testing.assert_close(expanded[:, 0], expanded[:, 1])
    assert not torch.allclose(expanded[:, 0], expanded[:, 2])


# ---------------------------------------------------------------------------
# MLA
# ---------------------------------------------------------------------------


def mla_config(q_lora_rank: int | None = None) -> AttentionConfig:
    return AttentionConfig(
        kind="mla", n_heads=4, n_kv_heads=None,
        kv_lora_rank=32, q_lora_rank=q_lora_rank,
        qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        qk_norm=False,
    )


@pytest.mark.parametrize("q_lora_rank", [None, 48])
def test_mla_absorption_matches_naive(q_lora_rank: int | None):
    """Folding W_UK into the queries and W_UV into the output changes nothing."""
    attn, pos = make(mla_config(q_lora_rank))
    attn.eval()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    with torch.no_grad():
        naive = run(attn, pos, x, absorbed=False)
        absorbed = run(attn, pos, x, absorbed=True)
    torch.testing.assert_close(naive, absorbed, rtol=1e-5, atol=1e-5)


def test_mla_absorption_matches_naive_in_fp16():
    """Same parity at reduced precision, with the looser tolerance it needs."""
    attn, pos = make(mla_config())
    attn.eval().half()
    x = torch.randn(BATCH, SEQ, D_MODEL).half()
    with torch.no_grad():
        naive = run(attn, pos, x, absorbed=False)
        absorbed = run(attn, pos, x, absorbed=True)
    torch.testing.assert_close(naive, absorbed, rtol=1e-2, atol=1e-3)


def test_mla_absorbed_shapes():
    attn, _ = make(mla_config())
    w_uk, fused = attn.absorb_weights()
    assert w_uk.shape == (4, 16, 32)
    assert fused.shape == (4, D_MODEL, 32)


def test_absorb_weights_rejects_non_mla():
    attn, _ = make(AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2))
    with pytest.raises(ValueError, match="only defined for MLA"):
        attn.absorb_weights()


def test_mla_scale_covers_the_concatenated_vector():
    attn, _ = make(mla_config())
    expected = 1.0 / (16 + 16) ** 0.5
    torch.testing.assert_close(attn.scale, expected, rtol=1e-9, atol=1e-9)


def test_mla_beats_dense_only_past_a_threshold():
    """MLA is not automatically cheaper, and the crossover is worth knowing.

    A latent cache costs ``kv_lora_rank + qk_rope_head_dim`` per token, with no
    factor of two since one vector stands in for keys and values. A dense cache
    costs ``2 * n_kv_heads * head_dim``. Whichever is smaller wins.
    """
    mla, _ = make(mla_config())
    assert mla.bytes_per_token() == (32 + 16) * 2  # rank + rope, fp16

    def dense_bytes(n_kv: int, kind: str = "gqa") -> int:
        cfg = ModelConfig(
            d_model=D_MODEL, n_layers=1,
            attention=AttentionConfig(kind=kind, n_heads=8, n_kv_heads=n_kv),
        )
        return Attention(cfg).bytes_per_token()

    head_dim = D_MODEL // 8
    assert dense_bytes(1, "mqa") == 2 * 1 * head_dim * 2  # 64 bytes, below MLA
    assert dense_bytes(1, "mqa") < mla.bytes_per_token()
    assert dense_bytes(4) > mla.bytes_per_token()
    assert dense_bytes(None, "mha") > mla.bytes_per_token()


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_plain_causal_returns_none_for_the_fast_path():
    assert build_attention_mask(4, 4, torch.device("cpu")) is None


def test_causal_mask_shape_and_content():
    mask = build_attention_mask(3, 3, torch.device("cpu"), window=99)
    expected = torch.tensor(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    torch.testing.assert_close(mask, expected)


def test_sliding_window_bounds_the_past():
    mask = build_attention_mask(5, 5, torch.device("cpu"), window=2)
    # row 4 sees only positions 3 and 4
    torch.testing.assert_close(
        mask[4], torch.tensor([False, False, False, True, True])
    )


def test_sinks_stay_visible_beyond_the_window():
    mask = build_attention_mask(6, 6, torch.device("cpu"), window=2, sinks=1)
    assert mask[5, 0], "the sink token must stay visible"
    assert not mask[5, 1], "a non sink outside the window must not"
    assert mask[5, 4] and mask[5, 5]


def test_decode_step_mask_uses_the_last_positions():
    """One query against a long cache, the query is the newest position."""
    mask = build_attention_mask(1, 10, torch.device("cpu"), window=3)
    torch.testing.assert_close(mask[0, :7], torch.zeros(7, dtype=torch.bool))
    torch.testing.assert_close(mask[0, 7:], torch.ones(3, dtype=torch.bool))


def test_document_boundaries_block_cross_attention():
    doc_ids = torch.tensor([[0, 0, 1, 1]])
    mask = build_attention_mask(4, 4, torch.device("cpu"), doc_ids=doc_ids)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 3, 2] and mask[0, 0, 3, 3]
    assert not mask[0, 0, 3, 0], "token of document 1 must not see document 0"
    assert not mask[0, 0, 2, 1]


def test_sliding_window_only_on_local_layers():
    att = AttentionConfig(
        kind="gqa", n_heads=N_HEADS, n_kv_heads=2, sliding_window=4, global_every=3
    )
    locals_ = [Attention(ModelConfig(d_model=D_MODEL, n_layers=6, attention=att), i).is_local
               for i in range(6)]
    assert locals_ == [True, True, False, True, True, False]


def test_no_window_means_every_layer_is_global():
    att = AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2)
    attn, _ = make(att, layer_idx=0)
    assert not attn.is_local and attn.window is None


# ---------------------------------------------------------------------------
# Score variants
# ---------------------------------------------------------------------------


def test_softcap_bounds_the_logits_and_warns_once():
    att = AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2, logit_softcap=5.0)
    attn, pos = make(att)
    assert attn.needs_naive_path
    import mllm.layers.attention as mod

    mod._SOFTCAP_WARNED = False
    with pytest.warns(RuntimeWarning, match="softcap"):
        run(attn, pos)
    import warnings as w

    with w.catch_warnings():
        w.simplefilter("error")
        run(attn, pos)  # must not warn a second time


def test_softcap_changes_the_output():
    x = torch.randn(BATCH, SEQ, D_MODEL) * 5
    base = AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2)
    capped = AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2, logit_softcap=0.5)
    set_determinism(3)
    a, pos = make(base)
    set_determinism(3)
    b, _ = make(capped)
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(run(a, pos, x), run(b, pos, x), rtol=1e-3, atol=1e-3)


def test_learned_sink_forces_naive_path_and_has_a_parameter():
    att = AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2, learned_sink=True)
    attn, pos = make(att)
    assert attn.needs_naive_path
    assert attn.sink_logit.shape == (N_HEADS,)
    assert run(attn, pos).shape == (BATCH, SEQ, D_MODEL)


def test_learned_sink_lets_attention_weights_sum_below_one():
    """The point of a sink: a head can decline to attend anywhere."""
    att = AttentionConfig(kind="gqa", n_heads=2, n_kv_heads=1, learned_sink=True)
    attn, pos = make(att)
    with torch.no_grad():
        attn.sink_logit.fill_(20.0)  # dominate every real score
    x = torch.randn(1, 4, D_MODEL)
    out = attn(x, torch.arange(4), pos)
    assert out.abs().max() < 1e-2, "a saturated sink must starve the value mixture"


@pytest.mark.parametrize("after_rope", [False, True])
def test_qk_norm_both_orders_run_and_differ(after_rope: bool):
    att = AttentionConfig(
        kind="gqa", n_heads=N_HEADS, n_kv_heads=2, qk_norm=True,
        qk_norm_after_rope=after_rope,
    )
    attn, pos = make(att)
    assert attn.qk_norm is not None
    assert run(attn, pos).shape == (BATCH, SEQ, D_MODEL)


def _qk_norm_pair() -> tuple[Attention, Attention, torch.nn.Module]:
    def mk(after: bool) -> AttentionConfig:
        return AttentionConfig(
            kind="gqa", n_heads=N_HEADS, n_kv_heads=2, qk_norm=True,
            qk_norm_after_rope=after,
        )

    set_determinism(7)
    before, pos = make(mk(False))
    set_determinism(7)
    after, _ = make(mk(True))
    after.load_state_dict(before.state_dict())
    return before, after, pos


def test_qk_norm_order_is_irrelevant_at_init():
    """RoPE is a rotation, so it preserves the RMS it would be divided by.

    With the gains still at one, normalizing before or after the rotation is
    the same operation. The order only starts to matter once the gains are
    learned and non-uniform, which the next test shows.
    """
    x = torch.randn(BATCH, SEQ, D_MODEL)
    before, after, pos = _qk_norm_pair()
    torch.testing.assert_close(run(before, pos, x), run(after, pos, x), rtol=1e-5, atol=1e-5)


def test_qk_norm_order_matters_once_gains_are_learned():
    """Non-uniform gains do not commute with the rotation."""
    x = torch.randn(BATCH, SEQ, D_MODEL)
    before, after, pos = _qk_norm_pair()
    for model in (before, after):
        with torch.no_grad():
            model.qk_norm.q_norm.weight.copy_(
                torch.linspace(0.5, 1.5, model.head_dim)
            )
            model.qk_norm.k_norm.weight.copy_(
                torch.linspace(1.5, 0.5, model.head_dim)
            )
    assert not torch.allclose(run(before, pos, x), run(after, pos, x), rtol=1e-4, atol=1e-4)


def test_mup_scale_uses_inverse_head_dim():
    att = AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2, scale="mup")
    from mllm.config import MuPConfig

    attn, _ = make(att, mup=MuPConfig(enabled=True, base_d_model=64))
    assert attn.scale == 1.0 / attn.head_dim


# ---------------------------------------------------------------------------
# Positions plugged into attention
# ---------------------------------------------------------------------------


def test_alibi_bias_reaches_attention():
    att = AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2)
    cfg = ModelConfig(
        d_model=D_MODEL, n_layers=1, attention=att, position=PositionConfig(kind="alibi")
    )
    attn = Attention(cfg)
    pos = build_position(
        cfg.position, head_dim=cfg.head_dim, d_model=D_MODEL, n_heads=N_HEADS, max_seq_len=64
    )
    assert run(attn, pos).shape == (BATCH, SEQ, D_MODEL)


def test_nope_runs_without_positional_information():
    cfg = ModelConfig(
        d_model=D_MODEL, n_layers=1,
        attention=AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2),
        position=PositionConfig(kind="nope"),
    )
    attn = Attention(cfg)
    pos = build_position(
        cfg.position, head_dim=cfg.head_dim, d_model=D_MODEL, n_heads=N_HEADS, max_seq_len=64
    )
    assert run(attn, pos).shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# Cache parity at the layer level
# ---------------------------------------------------------------------------


def test_incremental_decoding_matches_full_forward():
    """Token by token with a cache must equal one pass over the whole sequence."""
    attn, pos = make(AttentionConfig(kind="gqa", n_heads=N_HEADS, n_kv_heads=2))
    attn.eval()
    x = torch.randn(1, SEQ, D_MODEL)
    with torch.no_grad():
        full = attn(x, torch.arange(SEQ), pos)

        cache = KVCache(1, 2, attn.head_dim, max_len=SEQ)
        steps = [
            attn(x[:, i : i + 1], torch.tensor([i]), pos, cache=cache)
            for i in range(SEQ)
        ]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-4, atol=1e-4)


def test_mla_incremental_decoding_matches_full_forward():
    attn, pos = make(mla_config())
    attn.eval()
    x = torch.randn(1, SEQ, D_MODEL)
    with torch.no_grad():
        full = attn(x, torch.arange(SEQ), pos)
        cache = LatentCache(1, 32, 16, max_len=SEQ)
        steps = [
            attn(x[:, i : i + 1], torch.tensor([i]), pos, cache=cache)
            for i in range(SEQ)
        ]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-4, atol=1e-4)


def test_mla_absorbed_decoding_matches_naive_full_forward():
    """The path actually used at inference, against the training path."""
    attn, pos = make(mla_config())
    attn.eval()
    x = torch.randn(1, SEQ, D_MODEL)
    with torch.no_grad():
        full = attn(x, torch.arange(SEQ), pos)
        cache = LatentCache(1, 32, 16, max_len=SEQ)
        steps = [
            attn(x[:, i : i + 1], torch.tensor([i]), pos, cache=cache, absorbed=True)
            for i in range(SEQ)
        ]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-4, atol=1e-4)

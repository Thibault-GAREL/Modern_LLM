"""KV caches (M3).

The ring buffer test is the one that matters: a sliding window layer must
produce exactly what a dense cache truncated to the window would, otherwise
the memory saving is bought with wrong results.
"""

from __future__ import annotations

import pytest
import torch

from mllm.cache import KVCache, LatentCache, RingCache, build_cache
from mllm.config import AttentionConfig, ModelConfig
from mllm.utils.seed import set_determinism

HEADS, DIM = 2, 8


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def step(cache, layer: int, n: int, batch: int = 1):
    a = torch.randn(batch, cache.shape_a[0], n, cache.shape_a[1])
    b = torch.randn(batch, cache.shape_b[0], n, cache.shape_b[1])
    return cache.update(layer, a, b), a, b


# ---------------------------------------------------------------------------
# Dense cache
# ---------------------------------------------------------------------------


def test_dense_cache_accumulates():
    cache = KVCache(1, HEADS, DIM, max_len=16)
    chunks = []
    for _ in range(4):
        (k_all, _), a, _ = step(cache, 0, 2)
        chunks.append(a)
    assert k_all.shape == (1, HEADS, 8, DIM)
    torch.testing.assert_close(k_all, torch.cat(chunks, dim=2))
    assert cache.seq_len() == 8


def test_dense_cache_rejects_overflow():
    cache = KVCache(1, HEADS, DIM, max_len=4)
    step(cache, 0, 4)
    with pytest.raises(ValueError, match="overflow"):
        step(cache, 0, 1)


def test_reset_clears_positions():
    cache = KVCache(1, HEADS, DIM, max_len=8)
    step(cache, 0, 3)
    assert cache.seq_len() == 3
    cache.reset()
    assert cache.seq_len() == 0


def test_layers_are_independent():
    cache = KVCache(3, HEADS, DIM, max_len=8)
    step(cache, 0, 2)
    step(cache, 1, 5)
    assert cache.seq_len(0) == 2 and cache.seq_len(1) == 5 and cache.seq_len(2) == 0


# ---------------------------------------------------------------------------
# Ring cache
# ---------------------------------------------------------------------------


def test_ring_cache_matches_dense_truncated_to_window():
    """The defining property of the sliding window cache."""
    window, total = 4, 11
    ring = RingCache(1, HEADS, DIM, window=window)
    dense = KVCache(1, HEADS, DIM, max_len=total)

    for _ in range(total):
        a = torch.randn(1, HEADS, 1, DIM)
        b = torch.randn(1, HEADS, 1, DIM)
        ring_k, ring_v = ring.update(0, a, b)
        dense_k, dense_v = dense.update(0, a, b)
        torch.testing.assert_close(ring_k, dense_k[:, :, -window:])
        torch.testing.assert_close(ring_v, dense_v[:, :, -window:])


def test_ring_cache_returns_oldest_first():
    ring = RingCache(1, HEADS, DIM, window=3)
    written = []
    for i in range(5):
        a = torch.full((1, HEADS, 1, DIM), float(i))
        out, _ = ring.update(0, a, a)
        written.append(i)
    expected = torch.tensor([2.0, 3.0, 4.0])
    torch.testing.assert_close(out[0, 0, :, 0], expected)


def test_ring_cache_handles_a_chunk_larger_than_the_window():
    ring = RingCache(1, HEADS, DIM, window=3)
    a = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1).expand(1, HEADS, 10, DIM)
    out, _ = ring.update(0, a.contiguous(), a.contiguous())
    assert out.shape[2] == 3
    torch.testing.assert_close(out[0, 0, :, 0], torch.tensor([7.0, 8.0, 9.0]))


def test_ring_cache_never_exceeds_the_window():
    ring = RingCache(1, HEADS, DIM, window=4)
    for _ in range(20):
        out, _ = step(ring, 0, 1)[0]
    assert ring.a[0].shape[2] == 4
    assert out.shape[2] == 4


# ---------------------------------------------------------------------------
# Latent cache
# ---------------------------------------------------------------------------


def test_latent_cache_stores_two_different_widths():
    cache = LatentCache(1, kv_lora_rank=32, qk_rope_head_dim=8, max_len=16)
    (c_kv, k_rope), _, _ = step(cache, 0, 3)
    assert c_kv.shape == (1, 1, 3, 32)
    assert k_rope.shape == (1, 1, 3, 8)


# ---------------------------------------------------------------------------
# bytes_per_token, the number that decides between the variants
# ---------------------------------------------------------------------------


def test_bytes_per_token_dense_formula():
    cache = KVCache(12, n_kv_heads=4, head_dim=64, max_len=1024)
    assert cache.bytes_per_token(torch.float16) == 2 * 12 * 4 * 64 * 2


def test_bytes_per_token_latent_has_no_factor_two():
    """One latent vector stands in for both keys and values."""
    cache = LatentCache(12, kv_lora_rank=512, qk_rope_head_dim=64, max_len=1024)
    assert cache.bytes_per_token(torch.float16) == 12 * (512 + 64) * 2


def test_ring_cache_cost_is_bounded_by_the_window_not_the_context():
    """Per token the ring costs the same, but it stops growing after `window`."""
    ring = RingCache(12, 4, 64, window=1024)
    dense = KVCache(12, 4, 64, max_len=131072)
    assert ring.bytes_per_token() == dense.bytes_per_token()
    assert ring.capacity == 1024
    assert dense.capacity == 131072


def test_gqa_against_mla_cost_at_realistic_sizes():
    """The comparison the DeepSeek-V2 paper makes, reproduced in numbers."""
    gqa = KVCache(60, n_kv_heads=8, head_dim=128, max_len=1)
    mla = LatentCache(60, kv_lora_rank=512, qk_rope_head_dim=64, max_len=1)
    assert gqa.bytes_per_token() == 60 * 2 * 8 * 128 * 2
    assert mla.bytes_per_token() == 60 * (512 + 64) * 2
    assert mla.bytes_per_token() < gqa.bytes_per_token() / 3


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_cache_picks_latent_for_mla():
    cfg = ModelConfig(
        d_model=128, n_layers=2,
        attention=AttentionConfig(kind="mla", n_kv_heads=None, kv_lora_rank=32,
                                  qk_rope_head_dim=8),
    )
    assert isinstance(build_cache(cfg, max_len=64), LatentCache)


def test_build_cache_alternates_ring_and_dense():
    """Local layers get a ring buffer, global ones stay dense."""
    cfg = ModelConfig(
        d_model=128, n_layers=6,
        attention=AttentionConfig(
            kind="gqa", n_heads=8, n_kv_heads=2, sliding_window=64, global_every=3
        ),
    )
    kinds = [type(build_cache(cfg, max_len=512, layer_idx=i)) for i in range(6)]
    assert kinds == [RingCache, RingCache, KVCache, RingCache, RingCache, KVCache]


def test_build_cache_is_dense_without_a_window():
    cfg = ModelConfig(
        d_model=128, n_layers=2,
        attention=AttentionConfig(kind="gqa", n_heads=8, n_kv_heads=2),
    )
    assert isinstance(build_cache(cfg, max_len=64), KVCache)

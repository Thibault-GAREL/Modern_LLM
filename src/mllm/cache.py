"""Key-value caches for incremental decoding.

The 2017 paper never discusses the cost of autoregressive decoding. Without a
cache, generating token ``n`` recomputes the whole prefix, so a sequence costs
``O(n²)`` forward passes instead of ``O(n)``.

Three layouts, one interface. ``update(layer_idx, a, b) -> (a_all, b_all)``
takes the new entries and returns everything visible so far.

    KVCache      dense, (B, n_kv_heads, max_len, head_dim) per tensor
    RingCache    sliding window layers, only the last ``window`` entries exist
    LatentCache  MLA, one compressed vector plus one shared rotary key

``bytes_per_token()`` is what makes the variants comparable, and it is the
number that actually decides between GQA and MLA. See ``bench/kv_memory.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Cache(ABC):
    """Common interface. Storage is allocated lazily on the first update."""

    def __init__(
        self,
        n_layers: int,
        shape_a: tuple[int, int],
        shape_b: tuple[int, int],
        max_len: int,
    ) -> None:
        """
        Args:
            shape_a, shape_b: ``(n_heads, dim)`` of the two cached tensors.
                For a dense cache both are ``(n_kv_heads, head_dim)``. For MLA
                they are ``(1, kv_lora_rank)`` and ``(1, qk_rope_head_dim)``.
        """
        self.n_layers = n_layers
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.max_len = max_len
        self.a: list[Tensor | None] = [None] * n_layers
        self.b: list[Tensor | None] = [None] * n_layers
        self.pos: list[int] = [0] * n_layers

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Number of entries physically stored per layer."""

    def _allocate(self, layer_idx: int, batch: int, dtype: torch.dtype, device) -> None:
        ha, da = self.shape_a
        hb, db = self.shape_b
        opts = {"dtype": dtype, "device": device}
        self.a[layer_idx] = torch.zeros(batch, ha, self.capacity, da, **opts)
        self.b[layer_idx] = torch.zeros(batch, hb, self.capacity, db, **opts)

    @abstractmethod
    def update(self, layer_idx: int, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        """Append ``a`` and ``b`` for one layer and return everything visible.

        Args:
            a, b: ``(batch, heads, new_len, dim)``.
        """

    def bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        """Cache cost of one token of context, across all layers."""
        size = torch.finfo(dtype).bits // 8
        ha, da = self.shape_a
        hb, db = self.shape_b
        return self.n_layers * (ha * da + hb * db) * size

    def reset(self) -> None:
        self.pos = [0] * self.n_layers

    def rollback(self, n: int, layer_idx: int = 0) -> None:
        """Drop the last ``n`` entries, for speculative decoding rejections.

        The stored data past the new position is never read again, so only the
        cursor has to move.
        """
        if n < 0 or n > self.pos[layer_idx]:
            raise ValueError(f"cannot roll back {n} of {self.pos[layer_idx]} entries")
        self.pos[layer_idx] -= n

    def seq_len(self, layer_idx: int = 0) -> int:
        """Number of entries currently visible for this layer.

        Bounded by the capacity, so on a ring buffer it stops growing. Use
        :meth:`total_seen` for absolute positions.
        """
        return min(self.pos[layer_idx], self.capacity)

    def total_seen(self, layer_idx: int = 0) -> int:
        """Tokens processed so far, unbounded.

        This is what positional encodings need. Reading ``seq_len`` instead
        freezes RoPE at the window size on a ring buffer, which is silent and
        wrong.
        """
        return self.pos[layer_idx]


class KVCache(Cache):
    """Dense cache. Every past key and value is kept.

    Cost per token, which is the number GQA exists to reduce:
    ``2 * n_layers * n_kv_heads * head_dim * dtype_size``.
    """

    def __init__(
        self, n_layers: int, n_kv_heads: int, head_dim: int, max_len: int
    ) -> None:
        super().__init__(n_layers, (n_kv_heads, head_dim), (n_kv_heads, head_dim), max_len)

    @property
    def capacity(self) -> int:
        return self.max_len

    def update(self, layer_idx: int, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        if self.a[layer_idx] is None:
            self._allocate(layer_idx, a.shape[0], a.dtype, a.device)
        pos, n = self.pos[layer_idx], a.shape[2]
        if pos + n > self.capacity:
            raise ValueError(
                f"KVCache overflow: {pos + n} entries requested, capacity {self.capacity}"
            )
        self.a[layer_idx][:, :, pos : pos + n] = a
        self.b[layer_idx][:, :, pos : pos + n] = b
        self.pos[layer_idx] = pos + n
        end = self.pos[layer_idx]
        return self.a[layer_idx][:, :, :end], self.b[layer_idx][:, :, :end]


class RingCache(Cache):
    """Circular buffer for sliding window layers.

    This is where the memory win of a sliding window actually comes from. The
    mask alone saves compute, but the cache still holds everything unless it is
    bounded like this.

    Entries are returned oldest first, which costs a roll when the buffer has
    wrapped. A production implementation avoids that by keeping the ring order
    and masking on the stored positions instead.
    """

    def __init__(
        self, n_layers: int, n_kv_heads: int, head_dim: int, window: int
    ) -> None:
        super().__init__(n_layers, (n_kv_heads, head_dim), (n_kv_heads, head_dim), window)
        self.window = window

    @property
    def capacity(self) -> int:
        return self.window

    def rollback(self, n: int, layer_idx: int = 0) -> None:
        """Only possible while the buffer has not wrapped.

        Once it has, the entries a rollback would restore have already been
        overwritten. Speculative decoding therefore cannot run on a windowed
        layer past the window, and saying so is better than silently returning
        whatever happens to be in those slots.
        """
        if self.pos[layer_idx] - n >= self.window:
            raise NotImplementedError(
                "RingCache cannot roll back past the window, the entries are gone"
            )
        super().rollback(n, layer_idx)

    def update(self, layer_idx: int, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        if self.a[layer_idx] is None:
            self._allocate(layer_idx, a.shape[0], a.dtype, a.device)
        n = a.shape[2]
        pos = self.pos[layer_idx]

        # Anything older than the last `window` entries can never be read again
        if n > self.window:
            a, b = a[:, :, -self.window :], b[:, :, -self.window :]
            pos += n - self.window
            n = self.window

        slots = (pos + torch.arange(n, device=a.device)) % self.window
        self.a[layer_idx][:, :, slots] = a
        self.b[layer_idx][:, :, slots] = b
        self.pos[layer_idx] = pos + n

        total = self.pos[layer_idx]
        if total < self.window:
            return self.a[layer_idx][:, :, :total], self.b[layer_idx][:, :, :total]
        start = total % self.window
        return (
            torch.roll(self.a[layer_idx], shifts=-start, dims=2),
            torch.roll(self.b[layer_idx], shifts=-start, dims=2),
        )


class LatentCache(Cache):
    """MLA cache: one compressed vector plus one rotary key, per token.

    Cost per token: ``n_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_size``,
    with no factor of two because a single latent vector stands in for both the
    keys and the values. MLA only beats GQA once the number of KV heads is
    large enough that ``2 * n_kv_heads * head_dim`` exceeds
    ``kv_lora_rank + qk_rope_head_dim``.
    """

    def __init__(
        self, n_layers: int, kv_lora_rank: int, qk_rope_head_dim: int, max_len: int
    ) -> None:
        super().__init__(n_layers, (1, kv_lora_rank), (1, qk_rope_head_dim), max_len)
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

    @property
    def capacity(self) -> int:
        return self.max_len

    def update(self, layer_idx: int, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        if self.a[layer_idx] is None:
            self._allocate(layer_idx, a.shape[0], a.dtype, a.device)
        pos, n = self.pos[layer_idx], a.shape[2]
        if pos + n > self.capacity:
            raise ValueError(
                f"LatentCache overflow: {pos + n} entries requested, capacity {self.capacity}"
            )
        self.a[layer_idx][:, :, pos : pos + n] = a
        self.b[layer_idx][:, :, pos : pos + n] = b
        self.pos[layer_idx] = pos + n
        end = self.pos[layer_idx]
        return self.a[layer_idx][:, :, :end], self.b[layer_idx][:, :, :end]


class ModelCache:
    """One cache per layer, addressed by layer index.

    A model alternating local and global attention needs a ring buffer on the
    windowed layers and a dense cache on the others, so a single flat cache
    cannot describe it. This composes them and keeps the ``update`` signature
    the attention module already expects.
    """

    def __init__(self, caches: list[Cache]) -> None:
        self.caches = caches

    def update(self, layer_idx: int, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        return self.caches[layer_idx].update(0, a, b)

    def bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        return sum(c.bytes_per_token(dtype) for c in self.caches)

    def seq_len(self, layer_idx: int = 0) -> int:
        return self.caches[layer_idx].seq_len(0)

    def total_seen(self, layer_idx: int = 0) -> int:
        return self.caches[layer_idx].total_seen(0)

    def rollback(self, n: int) -> None:
        for c in self.caches:
            c.rollback(n, 0)

    def reset(self) -> None:
        for c in self.caches:
            c.reset()

    def __len__(self) -> int:
        return len(self.caches)

    def __getitem__(self, i: int) -> Cache:
        return self.caches[i]


def build_model_cache(cfg, *, max_len: int) -> ModelCache:
    """Build the full per-layer cache for a model config."""
    return ModelCache(
        [build_cache(cfg, max_len=max_len, layer_idx=i) for i in range(cfg.n_layers)]
    )


def build_cache(cfg, *, max_len: int, layer_idx: int = 0) -> Cache:
    """Pick the cache matching one layer of a model config.

    A model alternating local and global attention needs a different cache per
    layer, which is why this takes a layer index.
    """
    from mllm.config import ModelConfig

    assert isinstance(cfg, ModelConfig)
    att = cfg.attention

    if att.kind == "mla":
        return LatentCache(1, att.kv_lora_rank, att.qk_rope_head_dim, max_len)

    is_local = att.sliding_window is not None and not _is_global_layer(att, layer_idx)
    if is_local:
        return RingCache(1, att.kv_heads, cfg.head_dim, att.sliding_window)
    return KVCache(1, att.kv_heads, cfg.head_dim, max_len)


def _is_global_layer(att, layer_idx: int) -> bool:
    """True when this layer keeps full attention despite a sliding window."""
    if att.global_every is None:
        return False
    return (layer_idx + 1) % att.global_every == 0

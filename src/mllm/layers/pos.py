"""Positional information schemes.

The 2017 Transformer adds a fixed sinusoidal signal to the input embedding,
once, before the first layer. Position is therefore absolute and it competes
with content inside the residual stream, where it gets diluted with depth.

Everything since moves the injection point. RoPE (Su et al., 2021, arXiv
2104.09864) rotates q and k at every layer so the dot product depends only on
``m - n``. ALiBi (Press et al., 2021, arXiv 2108.12409) adds a distance
penalty to the scores. NoPE (Kazemnejad et al., 2023, arXiv 2305.19466)
injects nothing and lets the causal mask break the symmetry.

Three hooks cover the three injection points, and a scheme implements only
the ones it needs:

    input_embedding(positions)          added to the token embedding  (2017)
    __call__(q, k, positions)           applied to q and k            (RoPE)
    attn_bias(q_len, kv_len, ...)       added to the scores           (ALiBi)

Numerics: inverse frequencies, cos and sin, and the rotation itself are all
computed in fp32 and cast back at the end. In bf16 the drift is visible after
a few thousand positions.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from mllm.config import PositionConfig, RopeScalingConfig

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


@runtime_checkable
class PositionalScheme(Protocol):
    """Common interface for every positional scheme."""

    def __call__(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Transform q and k in place of nothing. Identity for absolute schemes."""
        ...

    def attn_bias(
        self, q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        """Additive bias of shape (n_heads, q_len, kv_len), or None."""
        ...

    def input_embedding(self, positions: Tensor, dim: int) -> Tensor | None:
        """Vector added to the token embedding, or None."""
        ...


class _NoHooks:
    """Default implementations: every hook is a no-op.

    Defines ``forward`` rather than ``__call__`` because ``nn.Module`` owns
    ``__call__`` and comes first in the MRO, so an override here would simply
    be shadowed.
    """

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def attn_bias(
        self, q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        return None

    def input_embedding(self, positions: Tensor, dim: int) -> Tensor | None:
        return None


# ---------------------------------------------------------------------------
# Reference implementation of the rotation (readable, used by the tests)
# ---------------------------------------------------------------------------


def apply_rope_reference(
    x: Tensor, cos: Tensor, sin: Tensor, style: str
) -> Tensor:
    """Rotate every 2D sub-vector of ``x`` explicitly, one pair at a time.

    Literal transcription of eq. (34) of Su et al. (2021). Deliberately a
    Python loop so there is no doubt about which coordinate is paired with
    which, which is the entire difference between the two conventions.

    Args:
        x: ``(..., seq, head_dim)``
        cos, sin: ``(seq, head_dim // 2)``
        style: ``"interleaved"`` pairs ``(x0, x1), (x2, x3), ...``
            (original paper, GPT-J). ``"half"`` pairs ``(x_i, x_{i + d/2})``
            (GPT-NeoX, LLaMA).
    """
    d = x.shape[-1]
    half = d // 2
    out = torch.empty_like(x)
    for i in range(half):
        if style == "interleaved":
            a, b = 2 * i, 2 * i + 1
        else:
            a, b = i, i + half
        xa, xb = x[..., a], x[..., b]
        c, s = cos[..., i], sin[..., i]
        out[..., a] = xa * c - xb * s
        out[..., b] = xa * s + xb * c
    return out


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, style: str) -> Tensor:
    """Vectorized rotation. ``cos``/``sin`` are ``(..., seq, head_dim // 2)``."""
    orig_dtype = x.dtype
    xf = x.float()
    if style == "interleaved":
        xa, xb = xf[..., 0::2], xf[..., 1::2]
    else:
        xa, xb = xf.chunk(2, dim=-1)

    ra = xa * cos - xb * sin
    rb = xa * sin + xb * cos

    if style == "interleaved":
        out = torch.stack((ra, rb), dim=-1).flatten(-2)
    else:
        out = torch.cat((ra, rb), dim=-1)
    return out.to(orig_dtype)


def rope_style_permutation(head_dim: int) -> Tensor:
    """Row permutation taking a half-split layout to an interleaved one.

    Half-split reads the pair ``(j, j + d/2)`` for frequency ``j``, interleaved
    reads ``(2j, 2j + 1)``. Sending coordinate ``j`` to ``2j`` and coordinate
    ``j + d/2`` to ``2j + 1`` makes the two schemes compute the same rotation.
    """
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    for j in range(half):
        perm[j] = 2 * j
        perm[j + half] = 2 * j + 1
    return perm


def convert_rope_style(
    state_dict: dict[str, Tensor],
    src: str,
    dst: str,
    head_dim: int,
    key_patterns: tuple[str, ...] = ("q_proj", "k_proj"),
) -> dict[str, Tensor]:
    """Convert a checkpoint between the two RoPE conventions.

    Only the rows of the query and key projections move. Since q and k get the
    same permutation and a permutation is orthogonal, their dot product is
    unchanged, so ``v_proj`` and ``o_proj`` are left alone.

    Args:
        state_dict: mapping of parameter name to tensor. Not modified.
        src, dst: ``"interleaved"`` or ``"half"``.
        head_dim: per-head dimension the permutation applies within.
        key_patterns: substrings identifying the projections to permute.

    Returns:
        A new state dict with the affected weights permuted.
    """
    valid = {"interleaved", "half"}
    if src not in valid or dst not in valid:
        raise ValueError(f"rope style must be one of {valid}, got {src!r} and {dst!r}")
    out = dict(state_dict)
    if src == dst:
        return out

    perm = rope_style_permutation(head_dim)
    # perm sends half-split coordinate j to interleaved slot perm[j], so going
    # TO interleaved gathers with its inverse and coming back gathers with perm
    index = torch.argsort(perm) if dst == "interleaved" else perm

    for name, tensor in state_dict.items():
        if not any(pat in name for pat in key_patterns):
            continue
        if tensor.shape[0] % head_dim != 0:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows, not a multiple of head_dim={head_dim}"
            )
        n_heads = tensor.shape[0] // head_dim
        reshaped = tensor.reshape(n_heads, head_dim, *tensor.shape[1:])
        out[name] = reshaped.index_select(1, index).reshape(tensor.shape).contiguous()
    return out


# ---------------------------------------------------------------------------
# Frequency construction and context extension
# ---------------------------------------------------------------------------


def _base_inv_freq(head_dim: int, theta: float, device: torch.device) -> Tensor:
    """``1 / theta^(2i / d)`` for ``i`` in ``[0, d/2)``, in fp32."""
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    return 1.0 / (theta**exponents)


def _yarn_correction_dim(
    n_rotations: float, head_dim: int, theta: float, max_pos: int
) -> float:
    """Dimension index whose wavelength completes ``n_rotations`` over ``max_pos``."""
    return (
        head_dim
        * math.log(max_pos / (n_rotations * 2 * math.pi))
        / (2 * math.log(theta))
    )


def _linear_ramp(low: float, high: float, n: int, device: torch.device) -> Tensor:
    if abs(high - low) < 1e-3:  # guard against a degenerate range
        high = low + 1e-3
    ramp = (torch.arange(n, dtype=torch.float32, device=device) - low) / (high - low)
    return ramp.clamp(0.0, 1.0)


def build_inv_freq(
    head_dim: int,
    theta: float,
    scaling: RopeScalingConfig | None,
    device: torch.device,
    *,
    seq_len: int | None = None,
) -> tuple[Tensor, float]:
    """Inverse frequencies after context extension, plus the attention multiplier.

    Returns ``(inv_freq, mscale)`` where ``mscale`` is 1.0 for every scheme
    except YaRN.
    """
    inv_freq = _base_inv_freq(head_dim, theta, device)
    if scaling is None:
        return inv_freq, 1.0

    s = scaling.factor
    kind = scaling.kind

    if kind == "linear":
        # Position Interpolation (Chen et al., 2023, arXiv 2306.15595):
        # squeeze positions into the trained range, pos -> pos / s
        return inv_freq / s, 1.0

    if kind in ("ntk-aware", "dynamic-ntk"):
        if kind == "dynamic-ntk":
            # recompute the scale from the length actually being processed
            current = seq_len if seq_len is not None else scaling.original_max_seq_len
            s = max(1.0, current / scaling.original_max_seq_len)
            if s == 1.0:
                return inv_freq, 1.0
        # spread the interpolation over the frequency bands by moving theta
        adjusted = theta * s ** (head_dim / (head_dim - 2))
        return _base_inv_freq(head_dim, adjusted, device), 1.0

    if kind == "yarn":
        # NTK-by-parts (Peng et al., 2023, arXiv 2309.00071): interpolate the
        # low frequency bands, extrapolate the high frequency ones, ramp between.
        extrapolation = inv_freq
        interpolation = inv_freq / s
        low = _yarn_correction_dim(
            scaling.beta_fast, head_dim, theta, scaling.original_max_seq_len
        )
        high = _yarn_correction_dim(
            scaling.beta_slow, head_dim, theta, scaling.original_max_seq_len
        )
        low, high = math.floor(low), math.ceil(high)
        # ramp goes 0 -> 1 with the dimension index, and a value of 1 means
        # "keep extrapolating", which is what the fast rotating dims want
        keep = 1.0 - _linear_ramp(low, high, head_dim // 2, device)
        merged = interpolation * (1 - keep) + extrapolation * keep
        # The temperature that separates YaRN from plain NTK-by-parts.
        # sqrt(1/t) = 0.1 * ln(s) + 1, applied to cos and sin, which is
        # equivalent to dividing the attention logits by t.
        mscale = 0.1 * math.log(s) + 1.0 if scaling.attn_temperature else 1.0
        return merged, mscale

    if kind == "llama3":
        # Wavelength-band ramp (arXiv 2407.21783)
        old_ctx = scaling.original_max_seq_len
        low_wavelen = old_ctx / scaling.low_freq_factor
        high_wavelen = old_ctx / scaling.high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        smooth = (old_ctx / wavelen - scaling.low_freq_factor) / (
            scaling.high_freq_factor - scaling.low_freq_factor
        )
        smoothed = (1 - smooth) * inv_freq / s + smooth * inv_freq
        out = torch.where(wavelen > low_wavelen, inv_freq / s, smoothed)
        out = torch.where(wavelen < high_wavelen, inv_freq, out)
        return out, 1.0

    raise ValueError(f"unknown rope scaling kind: {kind}")


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------


class RoPE(_NoHooks, nn.Module):
    """Rotary position embedding, applied to q and k at every layer.

    The property that matters: after rotation, ``<q_m, k_n>`` depends on the
    positions only through ``m - n``. Position becomes relative without any
    term being added to the residual stream.

    cos and sin are cached in a non-persistent buffer (so they never enter a
    checkpoint) and the cache grows on demand when a longer sequence arrives.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        theta: float = 10_000.0,
        style: str = "half",
        scaling: RopeScalingConfig | None = None,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta
        self.style = style
        self.scaling = scaling
        self._cached_len = 0
        self.mscale = 1.0
        # Plain attributes, not buffers. A buffer would be cast by .half() or
        # .bfloat16() along with the rest of the module, which is exactly the
        # fp32 guarantee this class exists to provide. Device changes are
        # handled by rebuilding in _ensure_cache.
        self.cos_cached: Tensor = torch.empty(0)
        self.sin_cached: Tensor = torch.empty(0)
        self._build_cache(max_seq_len, torch.device("cpu"))

    @property
    def is_dynamic(self) -> bool:
        return self.scaling is not None and self.scaling.kind == "dynamic-ntk"

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        inv_freq, mscale = build_inv_freq(
            self.head_dim, self.theta, self.scaling, device, seq_len=seq_len
        )
        self.mscale = mscale
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (seq, head_dim // 2), fp32
        self.cos_cached = (freqs.cos() * mscale).contiguous()
        self.sin_cached = (freqs.sin() * mscale).contiguous()
        self._cached_len = seq_len

    def _ensure_cache(self, needed: int, device: torch.device) -> None:
        stale = (
            needed > self._cached_len
            or self.cos_cached.device != device
            or self.cos_cached.dtype != torch.float32
            # dynamic-ntk changes the frequencies themselves with the length
            or (self.is_dynamic and needed > self.scaling.original_max_seq_len)
        )
        if stale:
            self._build_cache(max(needed, self._cached_len), device)

    def get_cos_sin(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """cos and sin for arbitrary (possibly non-contiguous) positions."""
        self._ensure_cache(int(positions.max().item()) + 1, positions.device)
        flat = positions.reshape(-1)
        cos = self.cos_cached.index_select(0, flat).reshape(*positions.shape, -1)
        sin = self.sin_cached.index_select(0, flat).reshape(*positions.shape, -1)
        return cos, sin

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate q and k.

        Args:
            q, k: ``(batch, n_heads, seq, head_dim)``
            positions: ``(seq,)`` or ``(batch, seq)``
        """
        cos, sin = self.get_cos_sin(positions)
        # broadcast over batch and head dimensions
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        return apply_rope(q, cos, sin, self.style), apply_rope(k, cos, sin, self.style)


class ALiBi(_NoHooks, nn.Module):
    """Attention with Linear Biases (Press et al., 2021, arXiv 2108.12409).

    No embedding and no rotation. A per-head slope multiplies the distance
    between query and key, and the result is subtracted from the score. Head
    ``i`` uses slope ``2^(-8i / n_heads)``, so different heads look back over
    different ranges.
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.register_buffer("slopes", self._build_slopes(n_heads), persistent=False)

    @staticmethod
    def _build_slopes(n_heads: int) -> Tensor:
        def powers_of_two(n: int) -> list[float]:
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            return [start ** (i + 1) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(powers_of_two(n_heads), dtype=torch.float32)
        # Non power of two: take the closest power of two below, then fill in
        # from the next power of two up, as in the reference implementation.
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = powers_of_two(closest)
        extra = powers_of_two(2 * closest)[0::2][: n_heads - closest]
        return torch.tensor(slopes + extra, dtype=torch.float32)

    def attn_bias(
        self, q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device, dtype=torch.float32)
        k_pos = torch.arange(kv_len, device=device, dtype=torch.float32)
        distance = k_pos[None, :] - q_pos[:, None]  # negative in the causal past
        bias = distance[None, :, :] * self.slopes.to(device)[:, None, None]
        return bias.to(dtype)


class Sinusoidal(_NoHooks, nn.Module):
    """The 2017 scheme: a fixed signal added to the input embedding.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d))`` and
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d))``, section 3.5 of Vaswani et al.
    """

    def __init__(self, dim: int, theta: float = 10_000.0, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("table", self._build(dim, theta, max_seq_len), persistent=False)

    @staticmethod
    def _build(dim: int, theta: float, seq_len: int) -> Tensor:
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        inv = _base_inv_freq(dim, theta, torch.device("cpu"))
        angles = pos * inv
        table = torch.zeros(seq_len, dim, dtype=torch.float32)
        table[:, 0::2] = angles.sin()
        table[:, 1::2] = angles.cos()
        return table

    def input_embedding(self, positions: Tensor, dim: int) -> Tensor:
        needed = int(positions.max().item()) + 1
        if needed > self.table.shape[0] or self.table.device != positions.device:
            self.table = self._build(self.dim, self.theta, max(needed, self.table.shape[0])).to(
                positions.device
            )
        return self.table.index_select(0, positions.reshape(-1)).reshape(*positions.shape, -1)


class LearnedAbsolute(_NoHooks, nn.Module):
    """A learned position table added to the input embedding (GPT-2 style)."""

    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)

    def input_embedding(self, positions: Tensor, dim: int) -> Tensor:
        return self.embedding(positions)


class NoPE(_NoHooks, nn.Module):
    """No positional information at all (Kazemnejad et al., 2023).

    The causal mask alone breaks the permutation symmetry, so a decoder can
    still infer position. Reported to generalize to unseen lengths better than
    several explicit schemes.
    """


def build_position(
    cfg: PositionConfig, *, head_dim: int, d_model: int, n_heads: int, max_seq_len: int
) -> nn.Module:
    """Instantiate the scheme selected by the config."""
    if cfg.kind == "rope":
        return RoPE(
            head_dim,
            theta=cfg.rope_theta,
            style=cfg.rope_style,
            scaling=cfg.scaling,
            max_seq_len=max_seq_len,
        )
    if cfg.kind == "alibi":
        return ALiBi(n_heads)
    if cfg.kind == "sinusoidal":
        return Sinusoidal(d_model, cfg.rope_theta, max_seq_len)
    if cfg.kind == "learned":
        return LearnedAbsolute(d_model, max_seq_len)
    if cfg.kind == "nope":
        return NoPE()
    raise ValueError(f"unknown position kind: {cfg.kind}")

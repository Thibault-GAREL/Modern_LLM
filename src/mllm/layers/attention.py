"""Attention: one class, four head layouts, and the mask variants.

MHA is the 2017 layout, one key and one value head per query head. MQA
(Shazeer, 2019, arXiv 1911.02150) collapses them to one, GQA (Ainslie et al.,
2023, arXiv 2305.13245) to a few groups, and MLA (DeepSeek-V2, 2024, arXiv
2405.04434) replaces them with a compressed latent vector.

None of this is about quality. It is about the KV cache, which at long context
becomes larger than the weights, and about decode steps that are limited by
memory bandwidth rather than arithmetic.

The mask side (sliding window, attention sinks, document packing) and the
score side (QK-norm, logit softcapping) are orthogonal to the head layout and
combine with all four.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mllm.config import ModelConfig
from mllm.init import mark_residual_projection
from mllm.layers.norm import QKNorm, RMSNorm

_SOFTCAP_WARNED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Whether to let SDPA broadcast the KV heads itself instead of expanding them.
#
# In principle enable_gqa=True (PyTorch >= 2.5) is strictly better, since it
# skips the expansion. Measured on torch 2.5.1 with B=8, H=8, KV=2, T=2048,
# fp16, it is dramatically worse, because it falls back to the math backend and
# materializes the whole (B, H, T, T) score matrix:
#
#     enable_gqa=True    +3496 MiB   100.0 ms
#     repeat_kv + SDPA     +48 MiB    36.3 ms
#
# So expansion wins by 73x on memory and 2.8x on time. Flip this to True and
# re-run bench/throughput.py once a torch release routes enable_gqa through the
# flash and memory-efficient backends. See test_enable_gqa_still_regresses.
USE_SDPA_ENABLE_GQA = False


def _sdpa_has_enable_gqa() -> bool:
    """torch added the keyword in 2.5. On 2.4 even passing False raises."""
    try:
        F.scaled_dot_product_attention(
            torch.zeros(1, 2, 2, 4),
            torch.zeros(1, 1, 2, 4),
            torch.zeros(1, 1, 2, 4),
            enable_gqa=True,
        )
    except TypeError:
        return False
    return True


SDPA_HAS_ENABLE_GQA = _sdpa_has_enable_gqa()


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand ``(B, n_kv_heads, T, D)`` to ``(B, n_kv_heads * n_rep, T, D)``.

    Uses ``expand`` plus ``reshape`` rather than ``repeat``, so the broadcast
    stays a view instead of copying. See ``USE_SDPA_ENABLE_GQA`` above for why
    this is preferred over letting SDPA broadcast internally.
    """
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    """Plain causal mask, queries being the last ``q_len`` positions."""
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    return k_pos <= q_pos


def build_attention_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    *,
    causal: bool = True,
    window: int | None = None,
    sinks: int = 0,
    doc_ids: Tensor | None = None,
) -> Tensor | None:
    """Combine causal, sliding window, sinks and document boundaries.

    Returns a bool tensor where ``True`` means "attend", broadcastable to
    ``(batch, heads, q_len, kv_len)``, or ``None`` when the result is plain
    causal so the caller can take the fused ``is_causal=True`` path.

    Queries are assumed to be the last ``q_len`` positions of the sequence,
    which is what makes this work for both prefill and incremental decoding.
    """
    plain_causal = causal and window is None and sinks == 0 and doc_ids is None
    if plain_causal:
        # None means "no mask tensor needed", which is only safe in two cases.
        # SDPA's is_causal aligns the triangle top-left, so it is correct only
        # for a square block. With a single query against a filled cache every
        # position is visible, so no mask is needed either. Anything in between
        # (chunked prefill) has to be materialized.
        if q_len == kv_len or q_len == 1:
            return None
        return _causal_mask(q_len, kv_len, device)

    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)

    mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
    if causal:
        mask &= k_pos <= q_pos
    if window is not None:
        mask &= (q_pos - k_pos) < window
    if sinks > 0:
        # StreamingLLM: the first tokens stay visible however far back they are.
        # Softmax has to put its mass somewhere, and evicting these collapses it.
        mask |= (k_pos < sinks) & (k_pos <= q_pos if causal else True)

    if doc_ids is not None:
        # Sequence packing: a token must not attend across a document boundary
        q_doc = doc_ids[:, kv_len - q_len : kv_len].unsqueeze(-1)
        k_doc = doc_ids[:, :kv_len].unsqueeze(-2)
        mask = mask.unsqueeze(0) & (q_doc == k_doc)
        mask = mask.unsqueeze(1)
    return mask


def _warn_softcap_once() -> None:
    global _SOFTCAP_WARNED
    if not _SOFTCAP_WARNED:
        warnings.warn(
            "logit_softcap is incompatible with fused attention kernels, "
            "falling back to the naive path. Gemma 3 dropped softcapping in "
            "favour of QK-norm for exactly this reason.",
            RuntimeWarning,
            stacklevel=3,
        )
        _SOFTCAP_WARNED = True


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """One attention module covering MHA, MQA, GQA and MLA.

    A variant is a config change, never a different class, which is what makes
    an ablation a single edited field.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.att = cfg.attention
        self.layer_idx = layer_idx
        self.kind = self.att.kind
        self.n_heads = self.att.n_heads
        self.d_model = cfg.d_model

        if self.kind == "mla":
            self._init_mla(cfg)
        else:
            self._init_dense(cfg)

        if self.att.learned_sink:
            # one learned logit per head, joining the softmax denominator
            self.sink_logit = nn.Parameter(torch.zeros(self.n_heads))

        self.dropout = self.att.dropout

    # -- construction -------------------------------------------------------

    def _init_dense(self, cfg: ModelConfig) -> None:
        self.head_dim = cfg.head_dim
        self.n_kv_heads = self.att.kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        bias = cfg.bias

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = mark_residual_projection(
            nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=bias)
        )
        self.qk_norm = QKNorm(self.head_dim) if self.att.qk_norm else None

        # muP change (c): 1 / head_dim instead of 1 / sqrt(head_dim)
        self.scale = (
            1.0 / self.head_dim
            if self.att.scale == "mup"
            else 1.0 / math.sqrt(self.head_dim)
        )
        self.v_head_dim = self.head_dim

    def _init_mla(self, cfg: ModelConfig) -> None:
        a = self.att
        self.qk_nope_head_dim = a.qk_nope_head_dim
        self.qk_rope_head_dim = a.qk_rope_head_dim
        self.qk_head_dim = a.qk_nope_head_dim + a.qk_rope_head_dim
        self.v_head_dim = a.v_head_dim
        self.kv_lora_rank = a.kv_lora_rank
        self.q_lora_rank = a.q_lora_rank
        self.head_dim = self.qk_head_dim
        bias = cfg.bias

        if a.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.qk_head_dim, bias=bias)
        else:
            self.q_a_proj = nn.Linear(cfg.d_model, a.q_lora_rank, bias=bias)
            self.q_a_norm = RMSNorm(a.q_lora_rank)
            self.q_b_proj = nn.Linear(
                a.q_lora_rank, self.n_heads * self.qk_head_dim, bias=bias
            )

        # one projection produces the latent and the decoupled rotary key
        self.kv_a_proj = nn.Linear(
            cfg.d_model, a.kv_lora_rank + a.qk_rope_head_dim, bias=bias
        )
        self.kv_a_norm = RMSNorm(a.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            a.kv_lora_rank, self.n_heads * (a.qk_nope_head_dim + a.v_head_dim), bias=False
        )
        self.o_proj = mark_residual_projection(
            nn.Linear(self.n_heads * a.v_head_dim, cfg.d_model, bias=bias)
        )
        self.qk_norm = QKNorm(self.qk_head_dim) if a.qk_norm else None

        # the scale covers the concatenated [nope | rope] vector
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)
        self._absorbed: tuple[Tensor, Tensor] | None = None

    # -- properties ---------------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True when a sliding window applies to this layer."""
        if self.att.sliding_window is None:
            return False
        if self.att.global_every is None:
            return True
        return (self.layer_idx + 1) % self.att.global_every != 0

    @property
    def window(self) -> int | None:
        return self.att.sliding_window if self.is_local else None

    @property
    def needs_naive_path(self) -> bool:
        """Softcapping and learned sinks cannot go through a fused kernel."""
        return self.att.logit_softcap is not None or self.att.learned_sink

    # -- core attention -----------------------------------------------------

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        mask: Tensor | None,
        bias: Tensor | None,
        enable_gqa: bool = False,
    ) -> Tensor:
        if not self.needs_naive_path:
            attn_mask = mask
            if bias is not None:
                # merge the bool mask into the additive bias SDPA expects.
                # Built this way round so the two broadcast against each other
                # instead of requiring bias to already have the mask's shape.
                if mask is not None:
                    blocked = torch.zeros(
                        mask.shape, dtype=bias.dtype, device=bias.device
                    ).masked_fill(~mask, float("-inf"))
                    attn_mask = bias + blocked
                else:
                    attn_mask = bias
            # is_causal only when the block is square, see build_attention_mask
            is_causal = attn_mask is None and q.shape[-2] == k.shape[-2]
            # the keyword only exists from torch 2.5, and passing it as False
            # is still a TypeError on 2.4, so it has to be left out entirely
            extra = {"enable_gqa": enable_gqa} if SDPA_HAS_ENABLE_GQA else {}
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0,
                **extra,
            )
        return self._attend_naive(q, k, v, mask=mask, bias=bias)

    def _attend_naive(
        self, q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor | None, bias: Tensor | None
    ) -> Tensor:
        """Readable reference path, and the only one that supports softcapping."""
        scores = (q @ k.transpose(-2, -1)) * self.scale

        cap = self.att.logit_softcap
        if cap is not None:
            _warn_softcap_once()
            # applied to the raw scores, before masking: tanh(-inf) is -1, not
            # -inf, so capping after the mask would resurrect masked positions
            scores = cap * torch.tanh(scores / cap)

        if bias is not None:
            scores = scores + bias
        if mask is None:
            # the builder returns None for plain causal so SDPA can fuse it,
            # but the naive path has to materialize that case
            mask = _causal_mask(scores.shape[-2], scores.shape[-1], scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        if self.att.learned_sink:
            # An extra logit joins the denominator without owning a value
            # vector, so a head can decline to attend instead of dumping its
            # mass on the first token.
            sink = self.sink_logit.view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
            scores = torch.cat([scores, sink.to(scores.dtype)], dim=-1)
            attn = scores.float().softmax(dim=-1).to(scores.dtype)[..., :-1]
        else:
            attn = scores.float().softmax(dim=-1).to(scores.dtype)

        if self.dropout and self.training:
            attn = F.dropout(attn, p=self.dropout)
        return attn @ v

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        pos_scheme: nn.Module,
        *,
        cache=None,
        doc_ids: Tensor | None = None,
        absorbed: bool = False,
    ) -> Tensor:
        if self.kind == "mla":
            return self._forward_mla(
                x, positions, pos_scheme, cache=cache, doc_ids=doc_ids, absorbed=absorbed
            )
        return self._forward_dense(
            x, positions, pos_scheme, cache=cache, doc_ids=doc_ids
        )

    def _prepare_mask_and_bias(
        self, q_len: int, kv_len: int, device, dtype, pos_scheme, doc_ids
    ) -> tuple[Tensor | None, Tensor | None]:
        mask = build_attention_mask(
            q_len,
            kv_len,
            device,
            window=self.window,
            sinks=self.att.attn_sinks,
            doc_ids=doc_ids,
        )
        bias = pos_scheme.attn_bias(q_len, kv_len, device, dtype)
        return mask, bias

    def _forward_dense(
        self, x: Tensor, positions: Tensor, pos_scheme, *, cache, doc_ids
    ) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None and not self.att.qk_norm_after_rope:
            q, k = self.qk_norm(q, k)
        q, k = pos_scheme(q, k, positions)
        if self.qk_norm is not None and self.att.qk_norm_after_rope:
            q, k = self.qk_norm(q, k)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        kv_len = k.shape[2]
        mask, bias = self._prepare_mask_and_bias(
            t, kv_len, x.device, x.dtype, pos_scheme, doc_ids
        )

        use_gqa = (
            USE_SDPA_ENABLE_GQA and not self.needs_naive_path and self.n_rep > 1
        )
        if not use_gqa:
            k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        out = self._attend(q, k, v, mask=mask, bias=bias, enable_gqa=use_gqa)
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)

    # -- MLA ----------------------------------------------------------------

    def _mla_queries(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, t, _ = x.shape
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.view(b, t, self.n_heads, self.qk_head_dim).transpose(1, 2)
        return q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    def _forward_mla(
        self, x: Tensor, positions: Tensor, pos_scheme, *, cache, doc_ids, absorbed: bool
    ) -> Tensor:
        b, t, _ = x.shape
        q_nope, q_rope = self._mla_queries(x)

        compressed = self.kv_a_proj(x)
        c_kv, k_rope = compressed.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        c_kv = self.kv_a_norm(c_kv)
        # the decoupled rotary key is a single head shared by every query head
        k_rope = k_rope.unsqueeze(1)

        q_rope, k_rope = pos_scheme(q_rope, k_rope, positions)

        if cache is not None:
            c_kv_cached, k_rope = cache.update(
                self.layer_idx, c_kv.unsqueeze(1), k_rope
            )
            c_kv = c_kv_cached.squeeze(1)

        kv_len = c_kv.shape[1]
        mask, bias = self._prepare_mask_and_bias(
            t, kv_len, x.device, x.dtype, pos_scheme, doc_ids
        )
        if absorbed:
            return self._mla_absorbed(q_nope, q_rope, c_kv, k_rope, mask, bias, b, t)
        return self._mla_naive(q_nope, q_rope, c_kv, k_rope, mask, bias, b, t)

    def _mla_naive(
        self, q_nope, q_rope, c_kv, k_rope, mask, bias, b: int, t: int
    ) -> Tensor:
        """Training path: materialize k_nope and v from the latent."""
        kv_len = c_kv.shape[1]
        kv = self.kv_b_proj(c_kv)
        kv = kv.view(b, kv_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, self.n_heads, -1, -1)], dim=-1)

        out = self._attend(q, k, v, mask=mask, bias=bias)
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.v_head_dim)
        return self.o_proj(out)

    def absorb_weights(self) -> tuple[Tensor, Tensor]:
        """Fold ``W_UK`` into the queries and ``W_UV`` into the output.

        ``q_nope · (W_UK c_kv) = (W_UKᵀ q_nope) · c_kv``, so the keys never need
        materializing. Likewise ``W_O (attn · W_UV c_kv)`` folds into a single
        matrix applied to the attended latent.

        Returns ``(w_uk, fused_out)`` shaped ``(n_heads, nope, rank)`` and
        ``(n_heads, d_model, rank)``.
        """
        if self.kind != "mla":
            raise ValueError("absorb_weights is only defined for MLA")
        if self._absorbed is not None:
            return self._absorbed

        # kv_b_proj maps rank -> n_heads * (nope + v_head_dim)
        w = self.kv_b_proj.weight.view(
            self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_uk = w[:, : self.qk_nope_head_dim, :]  # (H, nope, rank)
        w_uv = w[:, self.qk_nope_head_dim :, :]  # (H, v_head_dim, rank)

        # o_proj maps n_heads * v_head_dim -> d_model
        w_o = self.o_proj.weight.view(self.d_model, self.n_heads, self.v_head_dim)
        fused_out = torch.einsum("dhv,hvr->hdr", w_o, w_uv)  # (H, d_model, rank)

        self._absorbed = (w_uk.contiguous(), fused_out.contiguous())
        return self._absorbed

    def _mla_absorbed(
        self, q_nope, q_rope, c_kv, k_rope, mask, bias, b: int, t: int
    ) -> Tensor:
        """Inference path: attend directly against the cached latent."""
        w_uk, fused_out = self.absorb_weights()

        q_latent = torch.einsum("bhtn,hnr->bhtr", q_nope, w_uk.to(q_nope.dtype))
        scores = torch.einsum("bhtr,bsr->bhts", q_latent, c_kv)
        # k_rope carries a single shared head, and einsum does not broadcast it
        scores = scores + torch.einsum("bhtd,bsd->bhts", q_rope, k_rope.squeeze(1))
        scores = scores * self.scale

        cap = self.att.logit_softcap
        if cap is not None:
            _warn_softcap_once()
            scores = cap * torch.tanh(scores / cap)
        if bias is not None:
            scores = scores + bias
        if mask is None:
            mask = _causal_mask(t, c_kv.shape[1], scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = scores.float().softmax(dim=-1).to(scores.dtype)
        latent_out = torch.einsum("bhts,bsr->bhtr", attn, c_kv)
        out = torch.einsum("bhtr,hdr->btd", latent_out, fused_out.to(latent_out.dtype))
        if self.o_proj.bias is not None:
            out = out + self.o_proj.bias
        return out

    def bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        """Cache cost of one token for this layer, for comparing variants."""
        size = torch.finfo(dtype).bits // 8
        if self.kind == "mla":
            return (self.kv_lora_rank + self.qk_rope_head_dim) * size
        return 2 * self.n_kv_heads * self.head_dim * size

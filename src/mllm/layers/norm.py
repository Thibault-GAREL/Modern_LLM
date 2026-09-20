"""Normalization layers and their placement inside a block.

LayerNorm (Ba et al., 2016, arXiv 1607.06450) is what the 2017 Transformer
used, applied *after* each sub-block. Current models use RMSNorm (Zhang and
Sennrich, 2019, arXiv 1910.07467) applied *before*, which drops the mean
subtraction and the bias. QK-Norm (Henry et al., 2020, arXiv 2010.04245)
normalizes q and k inside attention, and DyT (Zhu et al., 2025, arXiv
2503.10622) replaces normalization with a scaled tanh.

Numerics: every statistic here is computed in fp32 and cast back at the end.
This is not optional. ``torch.nn.functional.rms_norm`` is deliberately NOT
used as a fast path because it computes in the input dtype (verified: in
fp16 it matches an all-fp16 computation bit for bit, not the fp32 one), and
that is the single most common source of bf16/fp16 divergence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mllm.config import NormConfig

# ---------------------------------------------------------------------------
# Reference implementations (readable, used only by the equivalence tests)
# ---------------------------------------------------------------------------


def rms_norm_reference(
    x: Tensor, weight: Tensor | None, eps: float, unit_offset: bool = False
) -> Tensor:
    """Literal transcription of RMSNorm, eq. (4) of Zhang and Sennrich (2019).

    Kept slow and explicit on purpose: it divides by the root mean square
    written out as a square root of a mean, instead of the fused rsqrt used
    by the fast path.
    """
    orig_dtype = x.dtype
    x = x.float()
    ms = x.pow(2).sum(dim=-1, keepdim=True) / x.shape[-1]
    out = x / torch.sqrt(ms + eps)
    if weight is not None:
        w = weight.float()
        out = out * (1.0 + w) if unit_offset else out * w
    return out.to(orig_dtype)


def layer_norm_reference(
    x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
) -> Tensor:
    """Literal transcription of LayerNorm (Ba et al., 2016)."""
    orig_dtype = x.dtype
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    out = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root mean square normalization, computed in fp32.

    Args:
        dim: size of the normalized (last) dimension.
        eps: added inside the square root.
        unit_offset: store the gain as ``1 + w`` with ``w`` initialized to
            zero (Gemma convention) instead of ``w`` initialized to one
            (LLaMA convention). The two are mathematically equivalent at
            init, but they put the weight decay and the gradient scale in
            different places, so a checkpoint written for one is wrong for
            the other.
        elementwise_affine: set False to drop the learned gain entirely.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        unit_offset: bool = False,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.unit_offset = unit_offset
        if elementwise_affine:
            init = torch.zeros(dim) if unit_offset else torch.ones(dim)
            self.weight = nn.Parameter(init)
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            w = self.weight.float()
            out = out * (1.0 + w) if self.unit_offset else out * w
        return out.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, unit_offset={self.unit_offset}"


class LayerNorm(nn.Module):
    """LayerNorm of the 2017 Transformer, with an optional bias.

    Modern models drop the bias (and usually LayerNorm itself). It is kept
    switchable so ``configs/base.yaml`` can reproduce the original block.
    """

    def __init__(self, dim: int, eps: float = 1e-5, *, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        out = F.layer_norm(
            x.float(),
            (self.dim,),
            self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        )
        return out.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, bias={self.bias is not None}"


class DyT(nn.Module):
    """Dynamic Tanh (Zhu et al., 2025, arXiv 2503.10622).

    ``DyT(x) = gamma * tanh(alpha * x) + beta`` with a single learned scalar
    ``alpha``. It removes the reduction over the feature dimension entirely,
    so there is no statistic to compute and nothing to synchronize. Reported
    as a drop-in replacement for LayerNorm/RMSNorm, still uncommon in
    released open-weights models.
    """

    def __init__(self, dim: int, *, alpha_init: float = 0.5) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        out = torch.tanh(self.alpha.float() * xf) * self.weight.float() + self.bias.float()
        return out.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class QKNorm(nn.Module):
    """RMSNorm applied to q and k over ``head_dim``, before or after RoPE.

    Stabilizes attention logits when they start to drift (observed above
    roughly a billion parameters, and in Gemma 3 / OLMo 2 at every size).
    Separate gains for q and k, following the released implementations.

    On the order relative to RoPE, one thing is easy to get wrong: a rotation
    is orthogonal, so it preserves the very RMS the normalization divides by.
    While the gains are still uniform the two orders are the *same operation*,
    verified in ``test_qk_norm_order_is_irrelevant_at_init``. They only diverge
    once the gains are learned and non-uniform, because a per-coordinate scale
    does not commute with a rotation that mixes coordinates in pairs.

    Both remain defensible, so it stays a flag: normalizing before RoPE is the
    usual convention, normalizing after bounds exactly what enters the dot
    product. Checkpoints are not interchangeable between the two.
    """

    def __init__(
        self, head_dim: int, eps: float = 1e-5, *, unit_offset: bool = False
    ) -> None:
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps, unit_offset=unit_offset)
        self.k_norm = RMSNorm(head_dim, eps, unit_offset=unit_offset)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.q_norm(q), self.k_norm(k)


def build_norm(cfg: NormConfig, dim: int, *, bias: bool = True) -> nn.Module:
    """Instantiate the normalization layer selected by the config."""
    if cfg.kind == "rmsnorm":
        return RMSNorm(dim, cfg.eps, unit_offset=cfg.unit_offset)
    if cfg.kind == "layernorm":
        return LayerNorm(dim, cfg.eps, bias=bias)
    if cfg.kind == "dyt":
        return DyT(dim, alpha_init=cfg.dyt_alpha_init)
    raise ValueError(f"unknown norm kind: {cfg.kind}")


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


class NormedResidual(nn.Module):
    """Wrap one sub-block (attention or FFN) with the configured placement.

    ============  ==========================================
    placement     computation
    ============  ==========================================
    ``post``      ``norm(x + f(x))``            Vaswani et al., 2017
    ``pre``       ``x + f(norm(x))``            current standard
    ``sandwich``  ``x + norm_out(f(norm_in(x)))``  Gemma 2, Grok
    ============  ==========================================

    Post-norm puts a normalization on the residual highway itself, which is
    why the original Transformer needed a learning rate warmup to train at
    all. Pre-norm leaves the highway untouched, at the cost of a growing
    residual stream (this is what the scaled residual init in ``mllm.init``
    compensates). Sandwich norm keeps the pre-norm highway and also bounds
    each sub-block output.

    A sub-block returning a tuple (attention with aux outputs, MoE with its
    routing losses) is supported: the first element is treated as the
    residual branch and the rest is passed through unchanged.
    """

    def __init__(
        self, cfg: NormConfig, dim: int, sublayer: nn.Module, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.placement = cfg.placement
        self.sublayer = sublayer
        self.norm_in = build_norm(cfg, dim, bias=bias)
        self.norm_out = (
            build_norm(cfg, dim, bias=bias) if cfg.placement == "sandwich" else None
        )

    def forward(self, x: Tensor, *args, **kwargs):
        if self.placement == "post":
            out = self.sublayer(x, *args, **kwargs)
            out, rest = _split(out)
            return _join(self.norm_in(x + out), rest)

        out = self.sublayer(self.norm_in(x), *args, **kwargs)
        out, rest = _split(out)
        if self.norm_out is not None:
            out = self.norm_out(out)
        return _join(x + out, rest)


def _split(out: Tensor | tuple) -> tuple[Tensor, tuple]:
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, ()


def _join(x: Tensor, rest: tuple) -> Tensor | tuple:
    return (x, *rest) if rest else x

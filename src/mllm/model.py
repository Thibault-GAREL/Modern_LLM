"""The decoder block and the full Transformer.

Everything switchable lives in the layer modules. This file only assembles
them, which is the point: swapping GQA for MLA, or a dense feed-forward for a
mixture of experts, changes a config field and nothing here.

Compared with Vaswani et al. (2017): decoder only, no cross-attention, and no
encoder, because for language modelling there is no separate source sequence
and the encoder is dead weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mllm.config import ModelConfig
from mllm.init import init_weights
from mllm.layers.attention import Attention
from mllm.layers.ffn import MatFormerMLP, build_ffn
from mllm.layers.heads import LMHead, MTPHeads
from mllm.layers.moe import MoE
from mllm.layers.norm import NormedResidual, build_norm
from mllm.layers.pos import build_position


@dataclass
class AuxLosses:
    """Everything added to the cross entropy, kept separable for logging.

    Merging these into one scalar is how a diverging router or an exploding
    z-loss becomes invisible.
    """

    ce: Tensor | None = None
    moe: Tensor | None = None
    z_loss: Tensor | None = None
    mtp: Tensor | None = None
    routing: list = field(default_factory=list)

    def total(self) -> Tensor | None:
        terms = [t for t in (self.ce, self.moe, self.z_loss, self.mtp) if t is not None]
        if not terms:
            return None
        return torch.stack(terms).sum()

    def as_dict(self) -> dict[str, float]:
        out = {}
        for name in ("ce", "moe", "z_loss", "mtp"):
            value = getattr(self, name)
            if value is not None:
                out[f"loss/{name}"] = float(value)
        return out


class Block(nn.Module):
    """One decoder layer: attention then feed-forward, each with its norms.

    The norm placement (pre, post, sandwich) is handled by ``NormedResidual``,
    so this class does not branch on it.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = NormedResidual(
            cfg.norm, cfg.d_model, Attention(cfg, layer_idx), bias=cfg.bias
        )
        # DeepSeek keeps the first layers dense: routing is unstable early and
        # the lowest layers learn features every expert would need anyway
        self.uses_moe = cfg.moe.enabled and layer_idx >= cfg.moe.first_k_dense
        sublayer = MoE(cfg) if self.uses_moe else build_ffn(cfg)
        self.is_matformer = isinstance(sublayer, MatFormerMLP)
        self.ffn = NormedResidual(cfg.norm, cfg.d_model, sublayer, bias=cfg.bias)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        pos_scheme: nn.Module,
        *,
        cache=None,
        doc_ids: Tensor | None = None,
        absorbed: bool = False,
        granularity: float = 1.0,
    ) -> tuple[Tensor, Tensor | None]:
        x = self.attn(
            x, positions, pos_scheme, cache=cache, doc_ids=doc_ids, absorbed=absorbed
        )
        # MatFormer layers accept a width, every other sub-block ignores it
        out = self.ffn(x, granularity) if self.is_matformer else self.ffn(x)
        if isinstance(out, tuple):  # MoE returns its auxiliary loss
            return out[0], out[1]
        return out, None

    @property
    def moe(self) -> MoE | None:
        return self.ffn.sublayer if self.uses_moe else None


class Transformer(nn.Module):
    """Embeddings, blocks, final norm, head, and the optional MTP modules."""

    def __init__(self, cfg: ModelConfig, *, z_loss_coef: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.z_loss_coef = z_loss_coef
        self.gradient_checkpointing = False

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = build_position(
            cfg.position,
            head_dim=cfg.rope_head_dim,
            d_model=cfg.d_model,
            n_heads=cfg.attention.n_heads,
            max_seq_len=cfg.max_seq_len,
        )
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout else None
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = build_norm(cfg.norm, cfg.d_model, bias=cfg.bias)
        self.lm_head = LMHead(cfg, self.embed if cfg.tie_embeddings else None)

        self.mtp = None
        if cfg.mtp_depth > 0:
            self.mtp = MTPHeads(
                cfg,
                self.embed,
                self.lm_head,
                lambda i: Block(cfg, cfg.n_layers + i),
            )

        init_weights(self, cfg)

    # -- helpers ------------------------------------------------------------

    def n_params(self, *, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel()
        return total

    def n_active_params(self) -> int:
        """Parameters used for one token, which differs from the total under MoE."""
        total = self.n_params()
        for block in self.blocks:
            if block.moe is not None:
                total -= block.moe.n_total_params() - block.moe.n_active_params()
        return total

    def routing_metrics(self) -> list:
        return [b.moe.last_metrics for b in self.blocks if b.moe is not None]

    def balance_step(self) -> None:
        """Apply the aux-loss-free bias update on every MoE layer."""
        for block in self.blocks:
            if block.moe is not None:
                block.moe.balance_step()

    # -- forward ------------------------------------------------------------

    def trunk(
        self,
        idx: Tensor,
        *,
        cache=None,
        doc_ids: Tensor | None = None,
        positions: Tensor | None = None,
        absorbed: bool = False,
        granularity: float = 1.0,
    ) -> tuple[Tensor, Tensor, AuxLosses]:
        """Everything up to and including the final norm, without the head.

        Exposed because the MTP modules and any speculative drafter need the
        trunk state, and reconstructing it from outside would be guesswork.

        Returns:
            ``(h, positions, aux)``.
        """
        return self._trunk(
            idx, cache=cache, doc_ids=doc_ids, positions=positions,
            absorbed=absorbed, granularity=granularity,
        )

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        *,
        cache=None,
        doc_ids: Tensor | None = None,
        positions: Tensor | None = None,
        absorbed: bool = False,
        granularity: float = 1.0,
    ) -> tuple[Tensor, Tensor | None, AuxLosses]:
        """
        Args:
            idx: token ids, ``(batch, seq)``.
            targets: ids to predict, ``(batch, seq)``. Enables the losses.
            cache: a ``ModelCache`` for incremental decoding.
            doc_ids: document id per token, for packed sequences.
            positions: explicit positions. Defaults to a range continuing the
                cache, which is what makes decoding work without extra care.
            absorbed: use the MLA inference path, where W_UK is folded into the
                queries and W_UV into the output. Ignored by every other kind.

        Returns:
            ``(logits, loss, aux)``. ``loss`` is None without targets.
        """
        h, positions, aux = self._trunk(
            idx, cache=cache, doc_ids=doc_ids, positions=positions,
            absorbed=absorbed, granularity=granularity,
        )
        logits = self.lm_head(h)

        if targets is None:
            return logits, None, aux

        aux.ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        if self.z_loss_coef:
            # keeps logits in a range bf16 can represent, ST-MoE arXiv 2202.08906
            aux.z_loss = self.z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
        if self.mtp is not None:
            _, mtp_loss = self.mtp(h, idx, targets, self.pos, positions=positions)
            aux.mtp = mtp_loss

        return logits, aux.total(), aux

    def _trunk(
        self,
        idx: Tensor,
        *,
        cache=None,
        doc_ids: Tensor | None = None,
        positions: Tensor | None = None,
        absorbed: bool = False,
        granularity: float = 1.0,
    ) -> tuple[Tensor, Tensor, AuxLosses]:
        _, t = idx.shape
        if positions is None:
            # total_seen, not seq_len: on a ring buffer the visible length
            # stops at the window while positions must keep counting
            offset = cache.total_seen() if cache is not None else 0
            positions = torch.arange(offset, offset + t, device=idx.device)

        x = self.embed(idx)
        absolute = self.pos.input_embedding(positions, self.cfg.d_model)
        if absolute is not None:
            x = x + absolute.to(x.dtype)
        if self.drop is not None:
            x = self.drop(x)

        aux = AuxLosses()
        moe_losses = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, moe_aux = checkpoint(
                    block, x, positions, self.pos, use_reentrant=False
                )
            else:
                x, moe_aux = block(
                    x, positions, self.pos, cache=cache, doc_ids=doc_ids,
                    absorbed=absorbed, granularity=granularity,
                )
            if moe_aux is not None:
                moe_losses.append(moe_aux)

        if moe_losses:
            aux.moe = torch.stack(moe_losses).sum()
        aux.routing = self.routing_metrics()
        return self.norm_f(x), positions, aux

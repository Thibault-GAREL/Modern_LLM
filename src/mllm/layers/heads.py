"""Output head and Multi-Token Prediction.

The 2017 model projects to the vocabulary and predicts one next token. Two
things changed. Tying the head to the embedding table saves ``vocab * d_model``
parameters, which on a small model is a large share of the total. And
Multi-Token Prediction (Gloeckle et al., 2024, arXiv 2404.19737, as used by
DeepSeek-V3) predicts several future tokens through extra modules.

MTP earns its place twice. During training it densifies the signal, since each
position supervises ``mtp_depth + 1`` predictions instead of one. At inference
the modules are either dropped entirely, or reused as a draft model for
speculative decoding, which is the cheapest draft available because it shares
the trunk it drafts for. Both uses are supported here.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mllm.config import ModelConfig
from mllm.init import mark_output_layer, output_logit_multiplier
from mllm.layers.norm import RMSNorm


class LMHead(nn.Module):
    """Projection to the vocabulary, optionally sharing the embedding matrix.

    Logits are produced in fp32. This is the third place where the precision
    is not negotiable, next to normalization and the RoPE tables, because a
    softmax over a large vocabulary in bf16 loses the tail outright.
    """

    def __init__(self, cfg: ModelConfig, embedding: nn.Embedding | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.tied = cfg.tie_embeddings and embedding is not None
        self.logit_mult = output_logit_multiplier(cfg)

        if self.tied:
            self.embedding = embedding
            self.proj = None
        else:
            self.proj = mark_output_layer(
                nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            )

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight if self.tied else self.proj.weight

    def forward(self, x: Tensor) -> Tensor:
        logits = F.linear(x.float(), self.weight.float())
        return logits * self.logit_mult if self.logit_mult != 1.0 else logits


class MTPModule(nn.Module):
    """One prediction depth: combine the trunk state with the future token.

    ``h_k = Block(W [norm(h_{k-1}) ; norm(emb(t + k))])``. The two normalized
    halves are concatenated and projected back to ``d_model`` before going
    through one Transformer block, and the shared LM head reads the result.
    """

    def __init__(self, cfg: ModelConfig, block: nn.Module) -> None:
        super().__init__()
        self.norm_hidden = RMSNorm(cfg.d_model, cfg.norm.eps)
        self.norm_embed = RMSNorm(cfg.d_model, cfg.norm.eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.block = block

    def forward(self, h_prev: Tensor, future_emb: Tensor, *args, **kwargs) -> Tensor:
        merged = torch.cat(
            [self.norm_hidden(h_prev), self.norm_embed(future_emb)], dim=-1
        )
        out = self.block(self.proj(merged), *args, **kwargs)
        # a real Block returns (hidden, moe_aux), and an MTP module has no use
        # for a second auxiliary loss on top of its own
        return out[0] if isinstance(out, tuple) else out


class MTPHeads(nn.Module):
    """The stack of MTP modules, sharing the embedding and the LM head.

    ``block_factory`` is a callable returning a fresh Transformer block. It is
    injected rather than imported so this module does not depend on
    ``model.py``, which is built on top of it.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        embedding: nn.Embedding,
        lm_head: LMHead,
        block_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        if cfg.mtp_depth <= 0:
            raise ValueError("MTPHeads requires mtp_depth > 0")
        self.cfg = cfg
        self.depth = cfg.mtp_depth
        self.embedding = embedding
        self.lm_head = lm_head
        self.modules_ = nn.ModuleList(
            MTPModule(cfg, block_factory(i)) for i in range(cfg.mtp_depth)
        )

    def forward(
        self,
        h: Tensor,
        idx: Tensor,
        targets: Tensor | None = None,
        *block_args,
        positions: Tensor | None = None,
    ) -> tuple[list[Tensor], Tensor]:
        """Predict the next ``depth`` tokens beyond the trunk prediction.

        Args:
            h: trunk hidden states, ``(batch, seq, d_model)``.
            idx: input token ids, ``(batch, seq)``.
            targets: token ids to predict, ``(batch, seq)``. When given, the
                loss is the mean cross entropy over depths, scaled by
                ``mtp_lambda``.
            block_args: forwarded to each inner block, typically the positional
                scheme.
            positions: sliced alongside the sequence, since depth ``k`` drops
                the last ``k`` positions.

        Returns:
            ``(logits_per_depth, loss)``. Positions with no ground truth left
            at that depth are excluded rather than padded.
        """
        _, t = idx.shape
        losses = []
        all_logits = []
        h_prev = h

        for k, module in enumerate(self.modules_, start=1):
            if t <= k:
                break
            # depth k predicts token t + k, so it is fed the embedding of the
            # token k steps ahead and only the positions that still have one
            future = self.embedding(idx[:, k:])
            sub_pos = positions[: t - k] if positions is not None else None
            args = (sub_pos, *block_args) if sub_pos is not None else block_args
            h_prev = module(h_prev[:, : t - k], future, *args)
            logits = self.lm_head(h_prev)
            all_logits.append(logits)

            if targets is not None:
                tgt = targets[:, k:]
                losses.append(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(),
                        tgt.reshape(-1),
                    )
                )

        loss = (
            self.cfg.mtp_lambda * torch.stack(losses).mean()
            if losses
            else h.new_zeros(())
        )
        return all_logits, loss

    @torch.no_grad()
    def draft(
        self, h: Tensor, idx: Tensor, *block_args, positions: Tensor | None = None
    ) -> Tensor:
        """Greedy draft of ``depth`` tokens, for speculative decoding.

        Takes the trunk state of the last position and walks the modules
        forward, feeding each its own previous prediction. This is the cheapest
        draft model available, because it shares the trunk it drafts for.

        Returns:
            ``(batch, depth)`` drafted token ids.
        """
        drafted = []
        h_prev = h[:, -1:]
        last = idx[:, -1:]
        for step, module in enumerate(self.modules_):
            sub_pos = (
                positions[-1:] + step + 1 if positions is not None else None
            )
            args = (sub_pos, *block_args) if sub_pos is not None else block_args
            h_prev = module(h_prev, self.embedding(last), *args)
            last = self.lm_head(h_prev).argmax(dim=-1)
            drafted.append(last)
        return torch.cat(drafted, dim=1)

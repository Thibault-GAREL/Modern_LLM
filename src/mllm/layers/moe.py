"""Mixture of Experts: router, dispatch, balancing, and routing metrics.

Sparse MoE (Shazeer et al., 2017, arXiv 1701.06538; Fedus et al., 2021, arXiv
2101.03961) replaces one feed-forward by many and activates a few per token,
so parameters grow while FLOPs per token do not.

DeepSeekMoE (2024, arXiv 2401.06066) adds the two things that distinguish it
from Switch: many small experts instead of a few large ones, which multiplies
the usable combinations, and always-active shared experts, so common knowledge
is stored once instead of being duplicated in every routed expert.

Balancing is the hard part. Without it the router collapses onto a handful of
experts. The classic answer is an auxiliary loss, which is a second objective
fighting the real one. DeepSeek-V3 (arXiv 2412.19437, method in arXiv
2408.15664) instead keeps a per-expert bias updated outside the gradient and
used **only for the top-k selection**, never for the weighting, so the balance
never leaks into what the model actually computes.

Routing metrics are not optional. A router collapse is invisible in the loss
curve, and only shows up in the entropy and the per-expert token share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mllm.config import ModelConfig, MoEConfig
from mllm.layers.ffn import build_ffn, compute_d_ff


@dataclass
class RoutingMetrics:
    """What has to be logged to notice a router going wrong.

    Attributes:
        entropy: mean entropy of the routing distribution, in nats. Maximal at
            ``log(n_experts)``, and collapsing towards zero is the failure.
        load_fraction: share of assignments received by each expert.
        load_cv: coefficient of variation of that share. Zero is perfect
            balance, and it grows as the load concentrates.
        drop_rate: share of assignments discarded for exceeding capacity.
        max_load: share taken by the single busiest expert.
    """

    entropy: float = 0.0
    load_fraction: Tensor = field(default_factory=lambda: torch.empty(0))
    load_cv: float = 0.0
    drop_rate: float = 0.0
    max_load: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "router/entropy": self.entropy,
            "router/load_cv": self.load_cv,
            "router/drop_rate": self.drop_rate,
            "router/max_load": self.max_load,
        }


class Router(nn.Module):
    """Scores every expert for every token and keeps the top k.

    The selection bias of the aux-loss-free mode lives here as a buffer rather
    than a parameter, because it is updated by an explicit rule outside the
    gradient and must never be optimized.
    """

    def __init__(self, d_model: int, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.weight = nn.Parameter(torch.empty(cfg.n_experts, d_model))
        nn.init.normal_(self.weight, std=0.02)
        # selection-only bias, never part of the weighting and never a Parameter
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Route a flat batch of tokens.

        Args:
            x: ``(n_tokens, d_model)``.

        Returns:
            ``(topk_idx, topk_weight, scores, logits)`` where ``topk_idx`` and
            ``topk_weight`` are ``(n_tokens, top_k)`` and ``scores`` is the
            full ``(n_tokens, n_experts)`` distribution used for the losses.
        """
        logits = F.linear(x.float(), self.weight.float())

        if self.cfg.gate == "sigmoid":
            # DeepSeek-V3 uses sigmoid, so experts are scored independently
            # rather than competing inside one softmax
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits, dim=-1)

        selection = scores + self.expert_bias if self.cfg.balance == "aux_loss_free" else scores
        _, topk_idx = torch.topk(selection, self.top_k, dim=-1)
        # weights come from the unbiased scores, which is the whole point
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topk_idx, topk_weight, scores, logits

    @torch.no_grad()
    def update_bias(self, load_fraction: Tensor) -> None:
        """Aux-loss-free balancing step, run once per optimizer step.

        ``b_i -= gamma * sign(load_i - mean_load)``. Overloaded experts become
        slightly harder to select and underloaded ones slightly easier, without
        a single gradient being involved.

        Known limit: the bias changes *which* expert wins, not *how many*. If
        every token carries the same routing scores they all migrate together,
        so the load cycles between experts instead of spreading. Balancing
        relies on the tokens being distinguishable in the first place, see
        ``test_bias_cannot_split_indistinguishable_tokens``.
        """
        if self.cfg.balance != "aux_loss_free":
            return
        error = load_fraction.to(self.expert_bias.device) - load_fraction.mean()
        self.expert_bias -= self.cfg.bias_update_gamma * torch.sign(error)


def router_z_loss(logits: Tensor) -> Tensor:
    """``mean(logsumexp(logits)^2)`` (ST-MoE, arXiv 2202.08906).

    Keeps router logits small so the gate stays in a range where bf16 has
    resolution, and stops one expert running away on magnitude alone.
    """
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


def load_balancing_loss(
    scores: Tensor, topk_idx: Tensor, n_experts: int
) -> Tensor:
    """``N * sum(f_i * P_i)`` (Switch Transformer, arXiv 2101.03961).

    ``f_i`` is the fraction of assignments landing on expert ``i`` and ``P_i``
    the mean routing probability it received. The product is minimized when
    both are uniform. Note it is a second objective competing with the real
    loss, which is exactly what the aux-loss-free mode avoids.
    """
    n_tokens = scores.shape[0]
    counts = torch.zeros(n_experts, device=scores.device, dtype=scores.dtype)
    counts.scatter_add_(
        0, topk_idx.reshape(-1), torch.ones_like(topk_idx.reshape(-1), dtype=scores.dtype)
    )
    f = counts / max(topk_idx.numel(), 1)
    p = scores.mean(dim=0)
    return n_experts * torch.sum(f * p) * (1.0 if n_tokens else 0.0)


class MoE(nn.Module):
    """Fine-grained routed experts plus always-active shared experts."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.moe = cfg.moe
        self.d_model = cfg.d_model
        self.top_k = cfg.moe.top_k
        self.n_experts = cfg.moe.n_experts

        d_ff_expert = cfg.moe.d_ff_expert or compute_d_ff(cfg.ffn, cfg.d_model)
        self.experts = nn.ModuleList(
            build_ffn(cfg, d_ff_expert) for _ in range(cfg.moe.n_experts)
        )
        self.shared_experts = nn.ModuleList(
            build_ffn(cfg, d_ff_expert) for _ in range(cfg.moe.n_shared_experts)
        )
        self.router = Router(cfg.d_model, cfg.moe)
        self.last_metrics = RoutingMetrics()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: ``(batch, seq, d_model)``.

        Returns:
            ``(output, aux_loss)``. The auxiliary loss aggregates the balancing
            term and the router z-loss, and is zero when both are disabled.
        """
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        topk_idx, topk_weight, scores, logits = self.router(flat)
        out, drop_rate, counts = self._dispatch(flat, topk_idx, topk_weight)

        for shared in self.shared_experts:
            out = out + shared(flat)

        aux = self._aux_loss(scores, topk_idx, logits)
        self._record_metrics(scores, counts, drop_rate, topk_idx.numel())
        return out.view(b, t, d), aux

    def _dispatch(
        self, flat: Tensor, topk_idx: Tensor, topk_weight: Tensor
    ) -> tuple[Tensor, float, Tensor]:
        """Group assignments by expert and run each on one contiguous slice.

        Sorting once turns ``n_experts`` passes over every token into one pass
        over every assignment, which is what makes a large expert count
        affordable.
        """
        n_tokens = flat.shape[0]
        out = torch.zeros_like(flat)

        flat_expert = topk_idx.reshape(-1)
        flat_weight = topk_weight.reshape(-1)
        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        token_of = order // self.top_k

        counts = torch.bincount(flat_expert, minlength=self.n_experts)
        boundaries = torch.cumsum(counts, dim=0)

        capacity = None
        dropped = 0
        if self.moe.capacity_factor is not None:
            capacity = int(
                self.moe.capacity_factor * n_tokens * self.top_k / self.n_experts
            )

        start = 0
        for e in range(self.n_experts):
            end = int(boundaries[e].item())
            if end == start:
                continue
            if capacity is not None and end - start > capacity:
                dropped += end - start - capacity
                end_used = start + capacity
            else:
                end_used = end
            idx = token_of[start:end_used]
            weights = flat_weight[order[start:end_used]].unsqueeze(-1)
            out.index_add_(0, idx, self.experts[e](flat[idx]) * weights.to(flat.dtype))
            start = end

        assert sorted_expert.numel() == flat_expert.numel()
        drop_rate = dropped / max(flat_expert.numel(), 1)
        return out, drop_rate, counts

    def _aux_loss(self, scores: Tensor, topk_idx: Tensor, logits: Tensor) -> Tensor:
        aux = logits.new_zeros(())
        if self.moe.balance == "aux_loss":
            aux = aux + self.moe.aux_loss_alpha * load_balancing_loss(
                scores, topk_idx, self.n_experts
            )
        else:
            # a very small sequence-wise term still helps, as a complement to
            # the bias rather than as the mechanism
            aux = aux + self.moe.seq_aux_alpha * load_balancing_loss(
                scores, topk_idx, self.n_experts
            )
        if self.moe.router_z_loss_coef:
            aux = aux + self.moe.router_z_loss_coef * router_z_loss(logits)
        return aux

    @torch.no_grad()
    def _record_metrics(
        self, scores: Tensor, counts: Tensor, drop_rate: float, n_assign: int
    ) -> None:
        probs = scores.float()
        entropy = -(probs.clamp_min(1e-9).log() * probs).sum(dim=-1).mean()
        load = counts.float() / max(n_assign, 1)
        self.last_metrics = RoutingMetrics(
            entropy=float(entropy),
            load_fraction=load,
            load_cv=float(load.std(unbiased=False) / load.mean().clamp_min(1e-9)),
            drop_rate=drop_rate,
            max_load=float(load.max()),
        )

    @torch.no_grad()
    def balance_step(self) -> None:
        """Apply the aux-loss-free bias update from the last forward pass."""
        self.router.update_bias(self.last_metrics.load_fraction)

    def n_active_params(self) -> int:
        """Parameters actually used for one token, versus the total."""
        per_expert = sum(p.numel() for p in self.experts[0].parameters())
        shared = sum(p.numel() for m in self.shared_experts for p in m.parameters())
        return self.top_k * per_expert + shared + self.router.weight.numel()

    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

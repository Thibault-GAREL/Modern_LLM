"""Sampling, incremental decoding, and speculative decoding.

The 2017 paper says nothing about generation cost. Without a KV cache, token
``n`` recomputes the whole prefix, so a sequence costs ``O(n²)`` forward passes
instead of ``O(n)``.

Speculative decoding (Leviathan et al., 2022, arXiv 2211.17192; Chen et al.,
2023, arXiv 2302.01318) goes further. A cheap drafter proposes ``gamma``
tokens, the target model verifies all of them in a **single** pass, and a
rejection sampling rule decides how many to keep. The rule matters more than
the speed: done correctly, the output distribution is *exactly* the target
model's, so this is free latency rather than a quality trade. Done with a
plausible-looking shortcut, the distribution silently shifts and nothing in a
loss curve or a benchmark will tell you.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mllm.cache import ModelCache, build_model_cache


@dataclass
class SamplingConfig:
    """Sampling knobs, applied in the order listed below.

    Order matters: the repetition penalty acts on raw logits, temperature
    rescales them, and the truncation filters then operate on comparable
    probabilities.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    greedy: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.min_p is not None and not 0 <= self.min_p < 1:
            raise ValueError("min_p must be in [0, 1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


# ---------------------------------------------------------------------------
# Logit processing
# ---------------------------------------------------------------------------


def apply_repetition_penalty(logits: Tensor, seen: Tensor, penalty: float) -> Tensor:
    """Divide the logit of every already generated token by ``penalty``.

    Negative logits are multiplied instead, since dividing them would make the
    token *more* likely, which is the classic sign error here.
    """
    if penalty == 1.0:
        return logits
    out = logits.clone()
    for b in range(logits.shape[0]):
        idx = seen[b].unique()
        picked = out[b, idx]
        out[b, idx] = torch.where(picked > 0, picked / penalty, picked * penalty)
    return out


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the ``k`` largest logits, mask the rest."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus sampling: keep the smallest set whose mass reaches ``p``."""
    if p >= 1.0:
        return logits
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
    # shift so the token that crosses the threshold is itself kept
    drop = cumulative - ordered.softmax(dim=-1) >= p
    drop[..., 0] = False
    return logits.masked_fill(drop.scatter(-1, indices, drop), float("-inf"))


def min_p_filter(logits: Tensor, min_p: float) -> Tensor:
    """Keep tokens at least ``min_p`` times as likely as the best one.

    Unlike top-p this adapts to how confident the model is: a peaked
    distribution keeps few tokens, a flat one keeps many.
    """
    if min_p <= 0:
        return logits
    probs = logits.softmax(dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


def process_logits(
    logits: Tensor, cfg: SamplingConfig, seen: Tensor | None = None
) -> Tensor:
    """Apply every sampling transform, returning logits ready for a softmax."""
    if seen is not None and cfg.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, seen, cfg.repetition_penalty)
    if cfg.temperature != 1.0 and cfg.temperature > 0:
        logits = logits / cfg.temperature
    if cfg.top_k is not None:
        logits = top_k_filter(logits, cfg.top_k)
    if cfg.top_p is not None:
        logits = top_p_filter(logits, cfg.top_p)
    if cfg.min_p is not None:
        logits = min_p_filter(logits, cfg.min_p)
    return logits


def sample_from_logits(
    logits: Tensor, cfg: SamplingConfig, generator: torch.Generator | None = None
) -> Tensor:
    """Draw one token per row. Returns ``(batch, 1)``."""
    if cfg.greedy or cfg.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ---------------------------------------------------------------------------
# Plain generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    cfg: SamplingConfig | None = None,
    *,
    cache: ModelCache | None = None,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
) -> Tensor:
    """Prefill then decode incrementally.

    The prompt goes through in one pass, filling the cache, after which every
    step processes a single token. Passing ``cache=None`` allocates one sized
    for the full output.
    """
    cfg = cfg or SamplingConfig()
    model.eval()
    if cache is None:
        cache = build_model_cache(model.cfg, max_len=idx.shape[1] + max_new_tokens)

    out = idx
    logits, _, _ = model(idx, cache=cache)  # prefill
    for _ in range(max_new_tokens):
        step = process_logits(logits[:, -1], cfg, seen=out)
        token = sample_from_logits(step, cfg, generator)
        out = torch.cat([out, token], dim=1)
        if eos_id is not None and (token == eos_id).all():
            break
        logits, _, _ = model(token, cache=cache)
    return out


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------


def residual_distribution(p: Tensor, q: Tensor) -> Tensor:
    """``norm(max(0, p - q))``, the distribution to resample from on rejection.

    This is the whole correctness argument. Rejecting a draft token and simply
    resampling from ``p`` would over-represent whatever the drafter already
    favoured. Subtracting ``q`` first removes exactly the mass the draft
    already accounted for, which is what makes the combined procedure produce
    ``p`` and nothing else.
    """
    residual = (p - q).clamp_min(0)
    total = residual.sum(dim=-1, keepdim=True)
    # a degenerate residual means p and q agree, so fall back to p
    return torch.where(total > 1e-10, residual / total.clamp_min(1e-10), p)


def verify_draft(
    target_probs: Tensor,
    draft_probs: Tensor,
    drafted: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, Tensor]:
    """Rejection sampling over one block of drafted tokens.

    Pure function of the distributions, so its correctness can be tested
    without any model, which is how the distribution equivalence test below
    stays cheap.

    Args:
        target_probs: ``(gamma + 1, vocab)``, the target model's distribution
            at each drafted position plus the bonus position.
        draft_probs: ``(gamma, vocab)``, the drafter's distributions.
        drafted: ``(gamma,)`` proposed token ids.

    Returns:
        ``(n_accepted, next_token)``. ``next_token`` is the resampled token on
        a rejection, or the bonus token drawn from ``target_probs[-1]`` when
        the whole block is accepted.
    """
    gamma = drafted.shape[0]
    for i in range(gamma):
        token = int(drafted[i])
        p_i, q_i = target_probs[i, token], draft_probs[i, token]
        ratio = 1.0 if q_i <= 0 else min(1.0, float(p_i / q_i))
        r = torch.rand((), generator=generator, device=target_probs.device)
        if float(r) >= ratio:
            residual = residual_distribution(target_probs[i], draft_probs[i])
            return i, torch.multinomial(residual, 1, generator=generator)
    bonus = torch.multinomial(target_probs[gamma], 1, generator=generator)
    return gamma, bonus


@dataclass
class SpeculativeStats:
    """Acceptance rate is the number that decides whether this pays off."""

    proposed: int = 0
    accepted: int = 0
    rounds: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "spec/acceptance_rate": self.acceptance_rate,
            "spec/tokens_per_round": self.tokens_per_round,
            "spec/rounds": float(self.rounds),
        }


@torch.no_grad()
def speculative_generate(
    target: nn.Module,
    draft: nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    *,
    gamma: int = 4,
    cfg: SamplingConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, SpeculativeStats]:
    """Draft ``gamma`` tokens, verify them in one target pass, keep a prefix.

    Batch size must be 1. With a batch, each sequence accepts a different
    number of tokens per round, so the caches go ragged and the bookkeeping
    hides the algorithm. Real implementations handle it, this one states the
    limit instead.

    Returns:
        ``(tokens, stats)``.
    """
    if idx.shape[0] != 1:
        raise ValueError("speculative_generate supports batch size 1 only")
    cfg = cfg or SamplingConfig()
    target.eval()
    draft.eval()
    stats = SpeculativeStats()
    out = idx

    while out.shape[1] - idx.shape[1] < max_new_tokens:
        # 1. the drafter runs autoregressively, cheaply
        drafted, draft_probs = [], []
        current = out
        for _ in range(gamma):
            d_logits, _, _ = draft(current)
            step = process_logits(d_logits[:, -1], cfg, seen=current)
            probs = step.softmax(dim=-1)
            token = sample_from_logits(step, cfg, generator)
            drafted.append(token)
            draft_probs.append(probs[0])
            current = torch.cat([current, token], dim=1)

        drafted_ids = torch.cat(drafted, dim=1)[0]
        q = torch.stack(draft_probs)

        # 2. the target verifies every draft position in a single pass
        t_logits, _, _ = target(current)
        window = t_logits[0, -(gamma + 1) :]
        p = torch.stack(
            [process_logits(window[i : i + 1], cfg, seen=out)[0] for i in range(gamma + 1)]
        ).softmax(dim=-1)

        # 3. rejection sampling decides how much of the draft survives
        n_accepted, next_token = verify_draft(p, q, drafted_ids, generator)
        stats.proposed += gamma
        stats.accepted += n_accepted
        stats.rounds += 1

        keep = drafted_ids[:n_accepted].view(1, -1)
        out = torch.cat([out, keep, next_token.view(1, 1)], dim=1)

    return out[:, : idx.shape[1] + max_new_tokens], stats


@torch.no_grad()
def mtp_draft(model: nn.Module, idx: Tensor) -> Tensor:
    """Draft with the model's own MTP heads, the cheapest drafter available.

    It shares the trunk it drafts for, so there is no second model to train,
    to store, or to keep in sync with the target. Returns ``(batch, depth)``
    proposed token ids.
    """
    if model.mtp is None:
        raise ValueError("this model has no MTP heads to draft with")
    h, positions, _ = model.trunk(idx)
    return model.mtp.draft(h, idx, model.pos, positions=positions)

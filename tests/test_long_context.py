"""Length generalization, measured on a trained model (M7).

The last test of the plan, and the only one that needed a training loop to
exist. Everything else about YaRN can be checked on the frequency tables, but
the claim that matters is behavioural.

The task is exact copying: a random block repeated twice, scored only on the
second half, so solving it requires attending back exactly one block length.
It is length independent by construction, which is what makes it a test of
positional generalization rather than of the task itself.

Read ``docs/ablations.md`` before drawing conclusions about which extension
scheme is best from this file. At this scale the published ordering does not
reproduce, and the tests below deliberately assert only what does.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mllm.config import (
    AttentionConfig,
    FFNConfig,
    ModelConfig,
    NormConfig,
    PositionConfig,
    RopeScalingConfig,
)
from mllm.model import Transformer
from mllm.utils.seed import set_determinism

VOCAB = 32
TRAIN_LEN = 32
SERVE_LEN = 128  # four times the training length


def make(scaling: RopeScalingConfig | None = None, theta: float = 10_000.0) -> ModelConfig:
    return ModelConfig(
        d_model=64,
        n_layers=2,
        vocab_size=VOCAB,
        max_seq_len=SERVE_LEN + 8,
        attention=AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
        position=PositionConfig(kind="rope", rope_theta=theta, scaling=scaling),
        ffn=FFNConfig(kind="swiglu", multiple_of=1),
        norm=NormConfig(kind="rmsnorm", placement="pre"),
    )


def copy_batch(batch: int, length: int, gen: torch.Generator):
    """A random block repeated twice. Returns ``(inputs, targets, half)``."""
    half = length // 2
    block = torch.randint(0, VOCAB, (batch, half), generator=gen)
    x = torch.cat([block, block], dim=1)
    return x[:, :-1], x[:, 1:], half


def copy_loss(model: Transformer, x, y, half) -> torch.Tensor:
    """Scored on the second half only, where the answer is determined."""
    logits, _, _ = model(x)
    return F.cross_entropy(
        logits[:, half - 1 :].reshape(-1, VOCAB), y[:, half - 1 :].reshape(-1)
    )


@torch.no_grad()
def perplexity(model: Transformer, length: int, seed: int = 99) -> float:
    gen = torch.Generator().manual_seed(seed)
    return float(copy_loss(model, *copy_batch(8, length, gen)).exp())


@pytest.fixture(scope="module")
def trained() -> Transformer:
    set_determinism(0)
    model = Transformer(make())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(0)
    model.train()
    for _ in range(400):
        loss = copy_loss(model, *copy_batch(32, TRAIN_LEN, gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def test_the_task_is_learned_at_training_length(trained: Transformer):
    """Without this the rest of the file measures nothing."""
    assert perplexity(trained, TRAIN_LEN) < 1.5


def test_plain_rope_collapses_past_the_training_length(trained: Transformer):
    """The problem every context extension scheme exists to solve.

    A model that copies perfectly at 32 tokens does not degrade gracefully at
    128, it collapses. RoPE generalizes poorly past its training length because
    the slowest frequency bands never completed a period during training.
    """
    short = perplexity(trained, TRAIN_LEN)
    long = perplexity(trained, SERVE_LEN)
    assert short < 1.5
    assert long > 100 * short, (
        f"expected a collapse past the training length, "
        f"got {short:.3f} at {TRAIN_LEN} and {long:.1f} at {SERVE_LEN}"
    )


@pytest.mark.parametrize("kind", ["linear", "ntk-aware", "yarn", "llama3"])
def test_every_scheme_loads_into_a_trained_model(trained: Transformer, kind: str):
    """Extension is applied to existing weights, with no architectural change.

    That is the whole point: a model trained short can be served long without
    retraining from scratch.
    """
    scaled = Transformer(
        make(RopeScalingConfig(kind=kind, factor=4.0, original_max_seq_len=TRAIN_LEN))
    ).eval()
    scaled.load_state_dict(trained.state_dict())  # must not raise
    assert perplexity(scaled, SERVE_LEN) > 0


def test_scaling_costs_short_range_resolution(trained: Transformer):
    """The price side, which is why extension is not on by default.

    Squeezing positions into the trained range disturbs the short distances the
    model already handled, so a scheme should only be enabled when the context
    actually requires it.
    """
    scaled = Transformer(
        make(RopeScalingConfig(kind="linear", factor=4.0, original_max_seq_len=TRAIN_LEN))
    ).eval()
    scaled.load_state_dict(trained.state_dict())
    assert perplexity(scaled, TRAIN_LEN) > perplexity(trained, TRAIN_LEN)


def test_no_scheme_rescues_this_task_zero_shot(trained: Transformer):
    """A negative result, asserted so it cannot be quietly forgotten.

    Applying any scaling to frozen weights leaves the perplexity at 128 in the
    same collapsed regime. This matches the YaRN paper, which fine-tunes at the
    target length rather than claiming a zero-shot fix, and it is the reason
    ``docs/ablations.md`` refuses to rank the schemes from this measurement.
    """
    plain = perplexity(trained, SERVE_LEN)
    for kind in ("linear", "ntk-aware", "yarn"):
        scaled = Transformer(
            make(RopeScalingConfig(kind=kind, factor=4.0, original_max_seq_len=TRAIN_LEN))
        ).eval()
        scaled.load_state_dict(trained.state_dict())
        ratio = perplexity(scaled, SERVE_LEN) / plain
        assert 0.1 < ratio < 10.0, (
            f"{kind} moved perplexity by {ratio:.2f}x, which would be a real "
            f"zero-shot effect and should be investigated rather than assumed"
        )

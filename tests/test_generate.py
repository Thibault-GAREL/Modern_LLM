"""Sampling, decoding, and speculative decoding (M6).

The milestone rests on two assertions. Incremental decoding with a cache must
equal a full forward pass, and speculative decoding must produce *exactly* the
target model's distribution, which is checked statistically rather than
assumed.
"""

from __future__ import annotations

import pytest
import torch

from mllm.cache import build_model_cache
from mllm.config import AttentionConfig, FFNConfig, ModelConfig
from mllm.generate import (
    SamplingConfig,
    SpeculativeStats,
    apply_repetition_penalty,
    generate,
    min_p_filter,
    mtp_draft,
    process_logits,
    residual_distribution,
    sample_from_logits,
    speculative_generate,
    top_k_filter,
    top_p_filter,
    verify_draft,
)
from mllm.model import Transformer
from mllm.utils.seed import set_determinism

VOCAB = 32


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def tiny(**kw) -> ModelConfig:
    base = {
        "d_model": 32,
        "n_layers": 2,
        "vocab_size": VOCAB,
        "max_seq_len": 64,
        "attention": AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
        "ffn": FFNConfig(kind="swiglu", multiple_of=1),
    }
    base.update(kw)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Logit filters
# ---------------------------------------------------------------------------


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.randn(2, VOCAB)
    filtered = top_k_filter(logits, 5)
    assert (filtered > float("-inf")).sum(dim=-1).tolist() == [5, 5]


def test_top_k_is_a_noop_when_k_covers_the_vocabulary():
    logits = torch.randn(2, VOCAB)
    torch.testing.assert_close(top_k_filter(logits, VOCAB), logits)


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    # probabilities 0.5, 0.3, 0.2 after softmax of these logits
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    logits = probs.log()
    kept = top_p_filter(logits, 0.6) > float("-inf")
    assert kept.tolist() == [[True, True, False]], "0.5 alone misses 0.6, so 0.3 joins"


def test_top_p_always_keeps_at_least_one_token():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    kept = top_p_filter(logits, 0.01) > float("-inf")
    assert kept.sum() >= 1


def test_min_p_adapts_to_confidence():
    """A peaked distribution keeps fewer tokens than a flat one, unlike top-k."""
    peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    flat = torch.tensor([[1.0, 0.9, 0.8, 0.7]])
    n_peaked = (min_p_filter(peaked, 0.1) > float("-inf")).sum()
    n_flat = (min_p_filter(flat, 0.1) > float("-inf")).sum()
    assert n_peaked < n_flat


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([[2.0, 1.0, -2.0]])
    seen = torch.tensor([[0, 2]])
    out = apply_repetition_penalty(logits, seen, 2.0)
    assert out[0, 0] == pytest.approx(1.0), "a positive logit is divided"
    assert out[0, 1] == pytest.approx(1.0), "an unseen token is untouched"
    assert out[0, 2] == pytest.approx(-4.0), "a negative logit is multiplied"


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    cold = process_logits(logits, SamplingConfig(temperature=0.5)).softmax(-1)
    hot = process_logits(logits, SamplingConfig(temperature=2.0)).softmax(-1)
    assert cold[0, 0] > hot[0, 0]


def test_greedy_picks_the_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    token = sample_from_logits(logits, SamplingConfig(greedy=True))
    assert int(token) == 1


def test_sampling_config_validates_its_ranges():
    with pytest.raises(ValueError, match="temperature"):
        SamplingConfig(temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        SamplingConfig(top_p=1.5)
    with pytest.raises(ValueError, match="min_p"):
        SamplingConfig(min_p=1.0)
    with pytest.raises(ValueError, match="repetition_penalty"):
        SamplingConfig(repetition_penalty=0.0)


# ---------------------------------------------------------------------------
# Generation and cache parity
# ---------------------------------------------------------------------------


def test_generate_returns_the_requested_length():
    model = Transformer(tiny())
    idx = torch.randint(0, VOCAB, (2, 5))
    out = generate(model, idx, 7, SamplingConfig(greedy=True))
    assert out.shape == (2, 12)
    torch.testing.assert_close(out[:, :5], idx)


def test_greedy_generation_is_deterministic():
    model = Transformer(tiny())
    idx = torch.randint(0, VOCAB, (1, 4))
    cfg = SamplingConfig(greedy=True)
    torch.testing.assert_close(generate(model, idx, 6, cfg), generate(model, idx, 6, cfg))


def test_cached_decoding_matches_an_uncached_full_forward():
    """The parity test that catches nearly every cache or mask mistake."""
    model = Transformer(tiny()).eval()
    idx = torch.randint(0, VOCAB, (1, 6))
    cfg = SamplingConfig(greedy=True)

    with_cache = generate(model, idx, 8, cfg)

    # the same sequence produced without any cache, recomputing every prefix
    out = idx
    with torch.no_grad():
        for _ in range(8):
            logits, _, _ = model(out)
            out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], dim=1)

    torch.testing.assert_close(with_cache, out)


def test_generation_logits_match_between_cached_and_uncached():
    """Stronger than token equality: the distributions themselves must agree."""
    model = Transformer(tiny()).eval()
    idx = torch.randint(0, VOCAB, (1, 10))
    with torch.no_grad():
        full, _, _ = model(idx)
        cache = build_model_cache(model.cfg, max_len=16)
        model(idx[:, :-1], cache=cache)
        stepped, _, _ = model(idx[:, -1:], cache=cache)
    torch.testing.assert_close(stepped[:, -1], full[:, -1], rtol=1e-4, atol=1e-4)


def test_eos_stops_generation_early():
    model = Transformer(tiny()).eval()
    idx = torch.randint(0, VOCAB, (1, 4))
    with torch.no_grad():
        logits, _, _ = model(idx)
        forced = int(logits[0, -1].argmax())
    out = generate(model, idx, 10, SamplingConfig(greedy=True), eos_id=forced)
    assert out.shape[1] < 14


# ---------------------------------------------------------------------------
# Rejection sampling, the correctness core
# ---------------------------------------------------------------------------


def test_residual_distribution_is_normalized_and_non_negative():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.1, 0.6, 0.3])
    r = residual_distribution(p, q)
    assert torch.all(r >= 0)
    torch.testing.assert_close(r.sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)
    assert r[1] == 0.0, "mass the drafter over-allocated must be removed entirely"


def test_residual_falls_back_to_p_when_the_drafter_matches():
    p = torch.tensor([0.5, 0.3, 0.2])
    torch.testing.assert_close(residual_distribution(p, p.clone()), p)


def test_verify_draft_accepts_everything_when_the_models_agree():
    gamma, vocab = 3, 5
    p = torch.full((gamma + 1, vocab), 1.0 / vocab)
    q = torch.full((gamma, vocab), 1.0 / vocab)
    drafted = torch.tensor([1, 2, 3])
    accepted = [verify_draft(p, q, drafted)[0] for _ in range(50)]
    assert all(a == gamma for a in accepted), "identical distributions always accept"


def test_verify_draft_rejects_what_the_target_dislikes():
    gamma, vocab = 2, 4
    p = torch.zeros(gamma + 1, vocab)
    p[:, 0] = 1.0  # the target only ever wants token 0
    q = torch.zeros(gamma, vocab)
    q[:, 3] = 1.0  # the drafter only ever proposes token 3
    drafted = torch.tensor([3, 3])
    n_accepted, token = verify_draft(p, q, drafted)
    assert n_accepted == 0
    assert int(token) == 0, "the resample must land on what the target wanted"


def test_speculative_output_distribution_equals_the_target():
    """The critical assertion of this milestone, checked statistically.

    A plausible but wrong implementation (resampling from p on rejection)
    passes every shape and type check and fails only here.
    """
    torch.manual_seed(0)
    vocab, gamma, trials = 6, 3, 30_000

    p_row = torch.tensor([0.35, 0.25, 0.20, 0.10, 0.07, 0.03])
    q_row = torch.tensor([0.05, 0.10, 0.15, 0.25, 0.20, 0.25])  # deliberately unlike p
    p = p_row.repeat(gamma + 1, 1)
    q = q_row.repeat(gamma, 1)

    gen = torch.Generator().manual_seed(1234)
    counts = torch.zeros(vocab)
    for _ in range(trials):
        drafted = torch.multinomial(q_row, gamma, replacement=True, generator=gen)
        n_accepted, next_token = verify_draft(p, q, drafted, gen)
        # the first token emitted by the round is what must follow p
        first = int(drafted[0]) if n_accepted > 0 else int(next_token)
        counts[first] += 1

    empirical = counts / trials
    total_variation = 0.5 * (empirical - p_row).abs().sum()
    assert total_variation < 0.02, (
        f"speculative output deviates from the target by {total_variation:.4f} "
        f"in total variation\nempirical {empirical.tolist()}\ntarget {p_row.tolist()}"
    )


def test_naive_rejection_would_fail_the_same_check():
    """Shows the test has teeth: resampling from p instead of the residual."""
    torch.manual_seed(0)
    vocab, gamma, trials = 6, 3, 30_000
    p_row = torch.tensor([0.35, 0.25, 0.20, 0.10, 0.07, 0.03])
    q_row = torch.tensor([0.05, 0.10, 0.15, 0.25, 0.20, 0.25])

    gen = torch.Generator().manual_seed(1234)
    counts = torch.zeros(vocab)
    for _ in range(trials):
        drafted = torch.multinomial(q_row, gamma, replacement=True, generator=gen)
        token = int(drafted[0])
        ratio = min(1.0, float(p_row[token] / q_row[token]))
        if float(torch.rand((), generator=gen)) < ratio:
            first = token
        else:  # the shortcut: resample from p, forgetting to subtract q
            first = int(torch.multinomial(p_row, 1, generator=gen))
        counts[first] += 1

    total_variation = 0.5 * (counts / trials - p_row).abs().sum()
    assert total_variation > 0.02, "the shortcut should be detectable"


# ---------------------------------------------------------------------------
# End to end speculative decoding
# ---------------------------------------------------------------------------


def test_speculative_generate_produces_the_requested_length():
    set_determinism(1)
    target = Transformer(tiny())
    draft = Transformer(tiny(n_layers=1))
    idx = torch.randint(0, VOCAB, (1, 4))
    out, stats = speculative_generate(target, draft, idx, 10, gamma=3)
    assert out.shape == (1, 14)
    assert stats.rounds > 0
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_identical_draft_and_target_accept_everything():
    """With the same model on both sides every proposal must survive."""
    set_determinism(2)
    model = Transformer(tiny()).eval()
    idx = torch.randint(0, VOCAB, (1, 4))
    _, stats = speculative_generate(model, model, idx, 12, gamma=4)
    assert stats.acceptance_rate > 0.95


def test_a_disagreeing_drafter_lowers_the_acceptance_rate():
    """Acceptance tracks how much the two distributions actually differ.

    Two freshly initialized models are both near uniform, so ``p / q`` is close
    to one and everything is accepted whatever their size. Sharpening the heads
    is what makes them disagree, and only then does the acceptance rate become
    informative. This is worth knowing before reading an acceptance rate off an
    untrained model and concluding the drafter is good.
    """
    n_tokens = 64  # 16 leaves the rate far too noisy to separate the two cases

    set_determinism(3)
    idx = torch.randint(0, VOCAB, (1, 4))
    target = Transformer(tiny(max_seq_len=128)).eval()
    _, same = speculative_generate(
        target, target, idx, n_tokens, gamma=4, generator=torch.Generator().manual_seed(3)
    )

    set_determinism(3)
    sharp_target = Transformer(tiny(max_seq_len=128)).eval()
    set_determinism(103)
    sharp_draft = Transformer(tiny(n_layers=1, max_seq_len=128)).eval()
    with torch.no_grad():  # make both decisive, and decisive about different tokens
        sharp_target.lm_head.weight.mul_(8.0)
        sharp_draft.lm_head.weight.mul_(8.0)
    _, worse = speculative_generate(
        sharp_target, sharp_draft, idx, n_tokens, gamma=4,
        generator=torch.Generator().manual_seed(3),
    )

    assert same.acceptance_rate > 0.95, "a model always agrees with itself"
    assert worse.acceptance_rate < 0.85, (
        f"a disagreeing drafter should be rejected often, got {worse.acceptance_rate:.2f}"
    )


def test_speculative_rejects_a_batch():
    model = Transformer(tiny())
    with pytest.raises(ValueError, match="batch size 1"):
        speculative_generate(model, model, torch.randint(0, VOCAB, (2, 4)), 4)


def test_stats_report_tokens_per_round():
    stats = SpeculativeStats(proposed=12, accepted=9, rounds=3)
    assert stats.acceptance_rate == pytest.approx(0.75)
    assert stats.tokens_per_round == pytest.approx(4.0)
    assert set(stats.as_dict()) == {
        "spec/acceptance_rate", "spec/tokens_per_round", "spec/rounds"
    }


# ---------------------------------------------------------------------------
# MTP as a drafter
# ---------------------------------------------------------------------------


def test_mtp_heads_can_draft():
    model = Transformer(tiny(mtp_depth=3, tie_embeddings=True)).eval()
    idx = torch.randint(0, VOCAB, (1, 6))
    drafted = mtp_draft(model, idx)
    assert drafted.shape == (1, 3)
    assert drafted.min() >= 0 and drafted.max() < VOCAB


def test_mtp_draft_requires_mtp_heads():
    model = Transformer(tiny())
    with pytest.raises(ValueError, match="no MTP heads"):
        mtp_draft(model, torch.randint(0, VOCAB, (1, 4)))


def test_trunk_matches_the_forward_pass():
    """The trunk hook must return exactly what forward feeds the head."""
    model = Transformer(tiny()).eval()
    idx = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        logits, _, _ = model(idx)
        h, _, _ = model.trunk(idx)
    torch.testing.assert_close(model.lm_head(h), logits, rtol=1e-5, atol=1e-6)

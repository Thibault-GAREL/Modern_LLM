"""LM head and Multi-Token Prediction (M4)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mllm.config import AttentionConfig, ModelConfig, MuPConfig
from mllm.init import is_output_layer
from mllm.layers.heads import LMHead, MTPHeads, MTPModule
from mllm.utils.seed import set_determinism

D_MODEL, VOCAB, SEQ, BATCH = 32, 50, 8, 2


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


class DummyBlock(nn.Module):
    """Stand-in for the real Block, which lands in M5."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return x + self.lin(x)


def make_cfg(**kw) -> ModelConfig:
    return ModelConfig(d_model=D_MODEL, n_layers=2, vocab_size=VOCAB, **kw)


# ---------------------------------------------------------------------------
# LM head
# ---------------------------------------------------------------------------


def test_untied_head_shapes_and_marking():
    head = LMHead(make_cfg(tie_embeddings=False))
    assert not head.tied
    assert is_output_layer(head.proj), "muP needs the output layer tagged"
    assert head(torch.randn(BATCH, SEQ, D_MODEL)).shape == (BATCH, SEQ, VOCAB)


def test_tied_head_shares_the_embedding_matrix():
    emb = nn.Embedding(VOCAB, D_MODEL)
    head = LMHead(make_cfg(tie_embeddings=True), emb)
    assert head.tied
    assert head.weight is emb.weight
    assert head(torch.randn(BATCH, SEQ, D_MODEL)).shape == (BATCH, SEQ, VOCAB)


def test_tying_saves_the_whole_output_matrix():
    emb = nn.Embedding(VOCAB, D_MODEL)
    tied = LMHead(make_cfg(tie_embeddings=True), emb)
    untied = LMHead(make_cfg(tie_embeddings=False))
    assert sum(p.numel() for p in tied.parameters()) == VOCAB * D_MODEL  # the embedding
    assert sum(p.numel() for p in untied.parameters()) == VOCAB * D_MODEL
    # tied reuses the table the model already pays for, untied adds a second one
    assert tied.weight is emb.weight and untied.weight is not emb.weight


def test_logits_are_fp32_even_in_half_precision():
    """A softmax over a large vocabulary in bf16 loses the tail."""
    head = LMHead(make_cfg(tie_embeddings=False)).half()
    out = head(torch.randn(BATCH, SEQ, D_MODEL).half())
    assert out.dtype == torch.float32


def test_mup_output_multiplier_is_applied():
    cfg = make_cfg(
        tie_embeddings=False,
        mup=MuPConfig(enabled=True, base_d_model=D_MODEL // 2),
        attention=AttentionConfig(scale="mup"),
    )
    head = LMHead(cfg)
    assert head.logit_mult == 0.5
    x = torch.randn(1, 1, D_MODEL)
    plain = torch.nn.functional.linear(x.float(), head.weight.float())
    torch.testing.assert_close(head(x), plain * 0.5, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# MTP
# ---------------------------------------------------------------------------


def build_mtp(depth: int = 2) -> tuple[MTPHeads, nn.Embedding]:
    cfg = make_cfg(mtp_depth=depth, tie_embeddings=True, mtp_lambda=0.3)
    emb = nn.Embedding(VOCAB, D_MODEL)
    head = LMHead(cfg, emb)
    return MTPHeads(cfg, emb, head, lambda _i: DummyBlock(D_MODEL)), emb


def test_mtp_module_consumes_both_halves():
    cfg = make_cfg(mtp_depth=1)
    module = MTPModule(cfg, DummyBlock(D_MODEL))
    h = torch.randn(BATCH, SEQ, D_MODEL)
    fut = torch.randn(BATCH, SEQ, D_MODEL)
    assert module(h, fut).shape == (BATCH, SEQ, D_MODEL)
    # the future embedding must actually change the result
    assert not torch.allclose(module(h, fut), module(h, torch.zeros_like(fut)))


def test_mtp_requires_a_positive_depth():
    cfg = make_cfg(mtp_depth=0)
    emb = nn.Embedding(VOCAB, D_MODEL)
    with pytest.raises(ValueError, match="mtp_depth"):
        MTPHeads(cfg, emb, LMHead(cfg, emb), lambda _i: DummyBlock(D_MODEL))


def test_mtp_produces_one_logit_set_per_depth():
    mtp, _ = build_mtp(depth=3)
    idx = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits, loss = mtp(torch.randn(BATCH, SEQ, D_MODEL), idx)
    assert len(logits) == 3
    # depth k drops the last k positions, which have no ground truth left
    for k, lg in enumerate(logits, start=1):
        assert lg.shape == (BATCH, SEQ - k, VOCAB)
    assert loss == 0.0, "no targets means no loss"


def test_mtp_loss_is_scaled_by_lambda():
    mtp, _ = build_mtp(depth=2)
    idx = torch.randint(0, VOCAB, (BATCH, SEQ))
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))
    _, loss = mtp(torch.randn(BATCH, SEQ, D_MODEL), idx, targets)
    assert loss > 0 and torch.isfinite(loss)
    assert loss.requires_grad

    mtp.cfg.mtp_lambda = 0.0
    _, zero = mtp(torch.randn(BATCH, SEQ, D_MODEL), idx, targets)
    torch.testing.assert_close(zero, torch.zeros(()), rtol=0, atol=1e-9)


def test_mtp_stops_when_the_sequence_is_shorter_than_the_depth():
    mtp, _ = build_mtp(depth=4)
    idx = torch.randint(0, VOCAB, (BATCH, 3))
    logits, _ = mtp(torch.randn(BATCH, 3, D_MODEL), idx)
    assert len(logits) == 2, "depths without any usable position are skipped"


def test_mtp_shares_the_lm_head_and_embedding():
    """The modules are extra depth, not extra vocabulary projections."""
    mtp, emb = build_mtp(depth=2)
    assert mtp.embedding is emb
    assert mtp.lm_head.weight is emb.weight


def test_mtp_draft_returns_one_token_per_depth():
    """Second use of MTP: a draft model for speculative decoding."""
    mtp, _ = build_mtp(depth=3)
    idx = torch.randint(0, VOCAB, (BATCH, SEQ))
    drafted = mtp.draft(torch.randn(BATCH, SEQ, D_MODEL), idx)
    assert drafted.shape == (BATCH, 3)
    assert drafted.min() >= 0 and drafted.max() < VOCAB


def test_mtp_modules_are_droppable():
    """First use: throw them away at inference and keep the trunk intact."""
    mtp, _ = build_mtp(depth=2)
    trunk_params = {id(p) for p in mtp.lm_head.parameters()}
    mtp_only = [p for p in mtp.modules_.parameters() if id(p) not in trunk_params]
    assert len(mtp_only) > 0
    assert all(id(p) not in trunk_params for p in mtp_only)

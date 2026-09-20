"""Parameter groups and schedules (M5)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mllm.config import AttentionConfig, FFNConfig, ModelConfig, MuPConfig, TrainConfig
from mllm.model import Transformer
from mllm.optim import build_optimizer, build_param_groups, build_scheduler, lr_multiplier
from mllm.utils.seed import set_determinism


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def tiny(**kw) -> ModelConfig:
    base = {
        "d_model": 64,
        "n_layers": 2,
        "vocab_size": 64,
        "max_seq_len": 32,
        "attention": AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
        "ffn": FFNConfig(kind="swiglu", multiple_of=1),
    }
    base.update(kw)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------


def test_no_weight_decay_on_norms_biases_or_embeddings():
    """Decaying a norm gain shrinks the signal that gain exists to rescale."""
    model_cfg = tiny(bias=True, tie_embeddings=False)
    model = Transformer(model_cfg)
    groups = build_param_groups(model, model_cfg, TrainConfig(weight_decay=0.1))

    decayed = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    exempt = {id(p) for g in groups if g["weight_decay"] == 0 for p in g["params"]}

    from mllm.optim import NORM_TYPES

    for module in model.modules():
        if isinstance(module, NORM_TYPES):
            for p in module.parameters(recurse=False):
                assert id(p) in exempt, "a norm gain must not be decayed"
        elif isinstance(module, nn.Embedding):
            assert id(module.weight) in exempt, "an embedding must not be decayed"
        elif isinstance(module, nn.Linear):
            assert id(module.weight) in decayed, "a matrix must be decayed"
            if module.bias is not None:
                assert id(module.bias) in exempt, "a bias must not be decayed"


def test_every_parameter_lands_in_exactly_one_group():
    model_cfg = tiny(bias=True)
    model = Transformer(model_cfg)
    groups = build_param_groups(model, model_cfg, TrainConfig())

    ids_in_groups = [id(p) for g in groups for p in g["params"]]
    assert len(ids_in_groups) == len(set(ids_in_groups)), "a parameter is duplicated"

    unique_model_params = {id(p) for p in model.parameters()}
    assert set(ids_in_groups) == unique_model_params


def test_tied_weights_are_counted_once():
    """Counting a tied matrix twice would double its decay."""
    model_cfg = tiny(tie_embeddings=True)
    model = Transformer(model_cfg)
    groups = build_param_groups(model, model_cfg, TrainConfig())
    flat = [id(p) for g in groups for p in g["params"]]
    assert flat.count(id(model.embed.weight)) == 1


def test_router_weight_is_decayed_like_a_matrix():
    from mllm.config import MoEConfig

    model_cfg = tiny(moe=MoEConfig(enabled=True, n_experts=4, top_k=2, first_k_dense=0))
    model = Transformer(model_cfg)
    groups = build_param_groups(model, model_cfg, TrainConfig(weight_decay=0.1))
    decayed = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    router = model.blocks[0].moe.router
    assert id(router.weight) in decayed


def test_expert_bias_never_reaches_the_optimizer():
    """It is updated by a rule, so an optimizer touching it would fight that."""
    from mllm.config import MoEConfig

    model_cfg = tiny(moe=MoEConfig(enabled=True, n_experts=4, top_k=2, first_k_dense=0))
    model = Transformer(model_cfg)
    groups = build_param_groups(model, model_cfg, TrainConfig())
    flat = {id(p) for g in groups for p in g["params"]}
    assert id(model.blocks[0].moe.router.expert_bias) not in flat


# ---------------------------------------------------------------------------
# muP groups
# ---------------------------------------------------------------------------


def test_mup_scales_hidden_matrix_learning_rates():
    model_cfg = tiny(
        d_model=256,
        mup=MuPConfig(enabled=True, base_d_model=64),
        attention=AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2, scale="mup"),
    )
    train_cfg = TrainConfig(lr=1e-3)
    model = Transformer(model_cfg)
    groups = build_param_groups(model, model_cfg, train_cfg)

    scaled = [g for g in groups if g.get("mup_scaled")]
    assert len(scaled) == 1
    assert scaled[0]["lr"] == pytest.approx(1e-3 / 4.0)

    others = [g for g in groups if not g.get("mup_scaled")]
    assert all(g["lr"] == pytest.approx(1e-3) for g in others)


def test_no_mup_groups_when_disabled():
    model_cfg = tiny()
    groups = build_param_groups(Transformer(model_cfg), model_cfg, TrainConfig())
    assert len(groups) == 2
    assert not any(g.get("mup_scaled") for g in groups)


def test_scheduler_preserves_the_mup_ratio():
    """LambdaLR multiplies each group's own base lr, so the ratio survives."""
    model_cfg = tiny(
        d_model=256,
        mup=MuPConfig(enabled=True, base_d_model=64),
        attention=AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2, scale="mup"),
    )
    train_cfg = TrainConfig(lr=1e-3, warmup_steps=0, max_steps=100)
    opt = build_optimizer(Transformer(model_cfg), model_cfg, train_cfg)
    sched = build_scheduler(opt, train_cfg)
    for _ in range(20):
        opt.step()
        sched.step()
    lrs = {id(g): g["lr"] for g in opt.param_groups}
    scaled = [g["lr"] for g in opt.param_groups if g.get("mup_scaled")][0]
    unscaled = [g["lr"] for g in opt.param_groups if not g.get("mup_scaled")][0]
    assert scaled == pytest.approx(unscaled / 4.0)
    assert len(lrs) == len(opt.param_groups)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_warmup_ramps_from_near_zero_to_one():
    cfg = TrainConfig(warmup_steps=10, max_steps=100, schedule="cosine")
    assert lr_multiplier(0, cfg) == pytest.approx(0.1)
    assert lr_multiplier(9, cfg) == pytest.approx(1.0)


def test_cosine_decays_to_the_floor():
    cfg = TrainConfig(warmup_steps=0, max_steps=100, schedule="cosine", min_lr_ratio=0.1)
    assert lr_multiplier(0, cfg) == pytest.approx(1.0, abs=1e-6)
    assert lr_multiplier(100, cfg) == pytest.approx(0.1, abs=1e-6)
    mid = lr_multiplier(50, cfg)
    assert 0.5 < mid < 0.6


def test_cosine_is_monotonically_decreasing_after_warmup():
    cfg = TrainConfig(warmup_steps=10, max_steps=100, schedule="cosine")
    values = [lr_multiplier(s, cfg) for s in range(10, 101)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:], strict=False))


def test_wsd_holds_a_plateau_then_decays():
    """The property that lets a run be forked mid-training."""
    cfg = TrainConfig(
        warmup_steps=10, max_steps=100, decay_steps=20, schedule="wsd", min_lr_ratio=0.0
    )
    assert lr_multiplier(20, cfg) == pytest.approx(1.0)
    assert lr_multiplier(70, cfg) == pytest.approx(1.0)
    assert lr_multiplier(79, cfg) == pytest.approx(1.0)
    assert lr_multiplier(90, cfg) == pytest.approx(0.5, abs=1e-6)
    assert lr_multiplier(100, cfg) == pytest.approx(0.0, abs=1e-6)


def test_wsd_plateau_is_longer_than_cosine_at_the_same_step():
    common = {"warmup_steps": 10, "max_steps": 100, "min_lr_ratio": 0.1}
    cosine = TrainConfig(schedule="cosine", **common)
    wsd = TrainConfig(schedule="wsd", decay_steps=20, **common)
    assert lr_multiplier(50, wsd) > lr_multiplier(50, cosine)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_optimizer_uses_the_configured_betas():
    model_cfg = tiny()
    opt = build_optimizer(Transformer(model_cfg), model_cfg, TrainConfig(betas=(0.9, 0.95)))
    assert all(g["betas"] == (0.9, 0.95) for g in opt.param_groups)


def test_one_optimizer_step_changes_the_loss():
    model_cfg = tiny()
    train_cfg = TrainConfig(lr=1e-2, warmup_steps=0, max_steps=10)
    model = Transformer(model_cfg)
    opt = build_optimizer(model, model_cfg, train_cfg)

    idx = torch.randint(0, 64, (2, 9))
    _, before, _ = model(idx[:, :-1], idx[:, 1:])
    before.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
    _, after, _ = model(idx[:, :-1], idx[:, 1:])
    assert after < before

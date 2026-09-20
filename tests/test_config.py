"""Validation rules of the config schema (M0).

Every rule listed in the implementation plan gets one accepting and one
rejecting case, so a refactor of config.py cannot silently drop a guard.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mllm.config import (
    AttentionConfig,
    Config,
    ModelConfig,
    MoEConfig,
    PositionConfig,
    RopeScalingConfig,
    TrainConfig,
)


def test_default_config_is_valid():
    cfg = Config()
    assert cfg.model.attention.kind == "gqa"
    assert cfg.model.head_dim == cfg.model.d_model // cfg.model.attention.n_heads


def test_kv_heads_none_means_mha():
    att = AttentionConfig(kind="mha", n_kv_heads=None)
    assert att.kv_heads == att.n_heads


def test_gqa_divisibility():
    with pytest.raises(ValidationError, match="multiple of"):
        AttentionConfig(kind="gqa", n_heads=16, n_kv_heads=5)
    assert AttentionConfig(kind="gqa", n_heads=16, n_kv_heads=4).kv_heads == 4


def test_mqa_requires_one_kv_head():
    with pytest.raises(ValidationError, match="mqa"):
        AttentionConfig(kind="mqa", n_heads=16, n_kv_heads=4)
    assert AttentionConfig(kind="mqa", n_heads=16, n_kv_heads=1).kv_heads == 1


def test_mla_rejects_explicit_kv_heads():
    with pytest.raises(ValidationError, match="mla"):
        AttentionConfig(kind="mla", n_kv_heads=4)
    AttentionConfig(kind="mla", n_kv_heads=None)  # valid


def test_global_every_requires_sliding_window():
    with pytest.raises(ValidationError, match="sliding_window"):
        AttentionConfig(n_heads=16, n_kv_heads=4, global_every=6)
    AttentionConfig(n_heads=16, n_kv_heads=4, sliding_window=1024, global_every=6)


def test_sinks_exclusive_with_learned_sink():
    with pytest.raises(ValidationError, match="learned_sink"):
        AttentionConfig(n_heads=16, n_kv_heads=4, attn_sinks=4, learned_sink=True)


def test_mtp_requires_tied_embeddings():
    with pytest.raises(ValidationError, match="tie_embeddings"):
        ModelConfig(mtp_depth=1, tie_embeddings=False)
    ModelConfig(mtp_depth=1, tie_embeddings=True)


def test_head_dim_divisibility():
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(d_model=500, attention=AttentionConfig(n_heads=16, n_kv_heads=4))
    explicit = ModelConfig(
        d_model=500, attention=AttentionConfig(n_heads=16, n_kv_heads=4, head_dim=32)
    )
    assert explicit.head_dim == 32


def test_rope_scaling_requires_rope():
    with pytest.raises(ValidationError, match="rope"):
        PositionConfig(kind="alibi", scaling=RopeScalingConfig(kind="yarn"))
    PositionConfig(kind="rope", scaling=RopeScalingConfig(kind="yarn"))


def test_moe_topk_bounded():
    with pytest.raises(ValidationError, match="top_k"):
        MoEConfig(n_experts=4, top_k=8)


def test_moe_first_k_dense_bounded():
    with pytest.raises(ValidationError, match="first_k_dense"):
        ModelConfig(n_layers=2, moe=MoEConfig(enabled=True, first_k_dense=2))


def test_mup_requires_mup_scale():
    from mllm.config import MuPConfig

    with pytest.raises(ValidationError, match="mup"):
        ModelConfig(mup=MuPConfig(enabled=True))
    ModelConfig(
        mup=MuPConfig(enabled=True),
        attention=AttentionConfig(scale="mup"),
    )


def test_wsd_requires_decay_steps():
    with pytest.raises(ValidationError, match="decay_steps"):
        TrainConfig(schedule="wsd", decay_steps=None)
    TrainConfig(schedule="wsd", decay_steps=100)


def test_warmup_bounded_by_max_steps():
    with pytest.raises(ValidationError, match="warmup"):
        TrainConfig(warmup_steps=2000, max_steps=1000)


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        AttentionConfig(n_head=8)  # typo: n_head instead of n_heads


def test_yaml_round_trip(tmp_path):
    yaml_text = """
model:
  d_model: 256
  n_layers: 4
  attention:
    kind: gqa
    n_heads: 8
    n_kv_heads: 2
train:
  max_steps: 50
  warmup_steps: 5
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.model.d_model == 256
    assert cfg.model.attention.kv_heads == 2
    assert cfg.train.max_steps == 50

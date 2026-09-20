"""Training loop (M5). Kept to a CPU-sized smoke test."""

from __future__ import annotations

import json

import pytest
import torch

from mllm.config import AttentionConfig, Config, FFNConfig, ModelConfig, TrainConfig
from mllm.train import ByteDataset, apply_overrides, train
from mllm.utils.seed import set_determinism


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def tiny_config(**train_kw) -> Config:
    train_cfg = {
        "max_steps": 12,
        "warmup_steps": 2,
        "seq_len": 16,
        "micro_batch_size": 4,
        "precision": "fp32",
        "log_interval": 4,
        "ckpt_interval": 1000,
        "lr": 3e-3,
    }
    train_cfg.update(train_kw)
    return Config(
        model=ModelConfig(
            d_model=32,
            n_layers=2,
            vocab_size=256,
            max_seq_len=32,
            attention=AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
            ffn=FFNConfig(kind="swiglu", multiple_of=1),
        ),
        train=TrainConfig(**train_cfg),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_byte_dataset_shapes_and_offset():
    ds = ByteDataset(None, seq_len=16, device=torch.device("cpu"))
    x, y = ds.batch(4)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.max() < ByteDataset.VOCAB_SIZE
    # targets are the inputs shifted by one
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_byte_dataset_reads_a_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"modern transformer " * 200)
    ds = ByteDataset(path, seq_len=8, device=torch.device("cpu"))
    assert "corpus.txt" in ds.source
    x, _ = ds.batch(2)
    assert x.shape == (2, 8)


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------


def test_override_sets_a_nested_field():
    cfg = apply_overrides(tiny_config(), ["model.d_model=64", "train.lr=0.001"])
    assert cfg.model.d_model == 64
    assert cfg.train.lr == 0.001


def test_override_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown config field"):
        apply_overrides(tiny_config(), ["model.d_modle=64"])
    with pytest.raises(ValueError, match="unknown config section"):
        apply_overrides(tiny_config(), ["modle.d_model=64"])
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(tiny_config(), ["model.d_model"])


def test_override_is_revalidated_by_the_schema():
    """An override must not be able to build an inconsistent config."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        apply_overrides(tiny_config(), ["model.attention.n_kv_heads=3"])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def metrics_of(model_dir) -> list[dict]:
    """Read the metrics.jsonl RunLogger wrote for this run.

    ``outputs/models/<name>_run-NN_date-.../`` pairs with
    ``outputs/logs/<name>_run-NN/``.
    """
    run_name = model_dir.name.split("_date-")[0]
    log_file = model_dir.parents[1] / "logs" / run_name / "metrics.jsonl"
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]


def test_training_reduces_the_loss():
    model_dir = train(tiny_config(), mlflow=False, model_name="t-loss")
    records = metrics_of(model_dir)
    assert len(records) >= 3
    assert records[-1]["loss/ce"] < records[0]["loss/ce"]


def test_metrics_contain_every_expected_field():
    model_dir = train(tiny_config(), mlflow=False, model_name="t-fields")
    first = metrics_of(model_dir)[0]
    assert {"step", "lr", "grad_norm", "step_time_s", "tokens", "loss/ce"} <= set(first)


def test_gradient_accumulation_matches_a_larger_batch():
    """Accumulating must be equivalent to one bigger batch, not merely similar."""
    from mllm.model import Transformer
    from mllm.optim import build_optimizer

    cfg = tiny_config()
    set_determinism(1)
    model_a = Transformer(cfg.model)
    set_determinism(1)
    model_b = Transformer(cfg.model)

    x = torch.randint(0, 256, (8, 16))
    y = torch.randint(0, 256, (8, 16))

    _, loss, _ = model_a(x, y)
    loss.backward()

    for chunk in range(2):
        sl = slice(chunk * 4, (chunk + 1) * 4)
        _, part, _ = model_b(x[sl], y[sl])
        (part / 2).backward()

    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-4, atol=1e-6)
    build_optimizer(model_a, cfg.model, cfg.train)  # must not raise


def test_checkpoint_round_trips():
    cfg = tiny_config()
    model_dir = train(cfg, mlflow=False, model_name="t-ckpt")
    # map_location so a GPU checkpoint stays loadable on a CPU-only machine
    ckpt = torch.load(model_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == cfg.train.max_steps - 1
    assert {"model", "optimizer", "scheduler", "scaler"} <= set(ckpt)

    restored = Config.model_validate(ckpt["config"])
    from mllm.model import Transformer

    model = Transformer(restored.model)
    model.load_state_dict(ckpt["model"])


def test_config_is_saved_next_to_the_log():
    model_dir = train(tiny_config(), mlflow=False, model_name="t-cfg")
    run_name = model_dir.name.split("_date-")[0]
    saved = json.loads(
        (model_dir.parents[1] / "logs" / run_name / "config.json").read_text()
    )
    assert saved["model"]["d_model"] == 32


def test_wsd_schedule_runs_end_to_end():
    cfg = tiny_config(schedule="wsd", decay_steps=4)
    model_dir = train(cfg, mlflow=False, model_name="t-wsd")
    lrs = [r["lr"] for r in metrics_of(model_dir) if "lr" in r]
    assert lrs[-1] < max(lrs), "the decay phase must lower the learning rate"


def test_activation_checkpointing_runs():
    cfg = tiny_config(activation_checkpointing=True)
    model_dir = train(cfg, mlflow=False, model_name="t-ckptact")
    assert (model_dir / "ckpt.pt").exists()

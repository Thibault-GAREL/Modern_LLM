"""Tokenized datasets and checkpoint resume (bilingual training run).

No network and no tokenizer here: a fake ``.bin`` covers the reader, and the
tokenization itself is exercised on the pod where the corpus lives.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mllm.config import AttentionConfig, Config, FFNConfig, ModelConfig, TrainConfig
from mllm.data import ByteDataset, TokenDataset, build_dataset
from mllm.train import load_checkpoint, save_checkpoint, train
from mllm.utils.seed import set_determinism

VOCAB = 512


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


@pytest.fixture
def corpus(tmp_path):
    """A directory shaped like the output of scripts/prepare_data.py.

    The tokens repeat a fixed block rather than being random. Uniform noise
    sits at ``ln(vocab)`` forever, so a test asserting the loss goes down would
    be asserting something impossible.
    """
    rng = np.random.default_rng(0)
    block = rng.integers(0, VOCAB, size=128, dtype=np.uint16)
    for split, n in (("train", 20_000), ("val", 4_000)):
        tokens = np.tile(block, n // len(block) + 1)[:n]
        tokens.tofile(tmp_path / f"{split}.bin")
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "vocab_size": VOCAB,
                "dtype": "uint16",
                "tokenizer": "fake",
                "mixture": {"en": 0.7, "fr": 0.3},
                "counts": {"train": {"en": 14000, "fr": 6000}},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# TokenDataset
# ---------------------------------------------------------------------------


def test_reads_metadata_and_tokens(corpus):
    ds = TokenDataset(corpus, "train", seq_len=64, device=torch.device("cpu"))
    assert ds.vocab_size == VOCAB
    assert ds.n_tokens == 20_000
    assert "mixture" in ds.source


def test_batch_shapes_and_shift(corpus):
    ds = TokenDataset(corpus, "train", seq_len=32, device=torch.device("cpu"))
    x, y = ds.batch(4)
    assert x.shape == (4, 32) and y.shape == (4, 32)
    assert x.dtype == torch.int64, "ids must be int64 for embedding lookup"
    # targets are the inputs shifted by one, which is the whole training signal
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_tokens_stay_inside_the_vocabulary(corpus):
    ds = TokenDataset(corpus, "train", seq_len=32, device=torch.device("cpu"))
    x, y = ds.batch(16)
    assert int(x.max()) < VOCAB and int(y.max()) < VOCAB
    assert int(x.min()) >= 0


def test_an_explicit_seed_pins_the_validation_batches(corpus):
    """Validation must be identical across runs, or two losses are not comparable."""
    kw = {"seq_len": 32, "device": torch.device("cpu"), "seed": 7}
    a = TokenDataset(corpus, "val", **kw).batch(4)[0]
    b = TokenDataset(corpus, "val", **kw).batch(4)[0]
    torch.testing.assert_close(a, b)


def test_unseeded_datasets_follow_the_global_rng(corpus):
    """Two datasets built in sequence must not replay the same windows.

    An unseeded ``torch.Generator`` carries a fixed default seed, so this
    would silently pass without deriving the seed from the global RNG.
    """
    cpu = torch.device("cpu")
    set_determinism(0)
    first = TokenDataset(corpus, "train", 32, cpu).batch(4)[0]
    second = TokenDataset(corpus, "train", 32, cpu).batch(4)[0]
    assert not torch.equal(first, second)

    # and the whole sequence stays reproducible under a fixed global seed
    set_determinism(0)
    again = TokenDataset(corpus, "train", 32, cpu).batch(4)[0]
    torch.testing.assert_close(first, again)


def test_splits_are_disjoint_files(corpus):
    train = TokenDataset(corpus, "train", seq_len=32, device=torch.device("cpu"))
    val = TokenDataset(corpus, "val", seq_len=32, device=torch.device("cpu"))
    assert train.n_tokens != val.n_tokens
    assert train.path != val.path


def test_missing_corpus_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        TokenDataset(tmp_path, "train", seq_len=32, device=torch.device("cpu"))


def test_sequence_longer_than_the_corpus_is_rejected(corpus):
    with pytest.raises(ValueError, match="too few"):
        TokenDataset(corpus, "val", seq_len=99_999, device=torch.device("cpu"))


def test_build_dataset_dispatch(corpus, tmp_path):
    cpu = torch.device("cpu")
    assert isinstance(build_dataset(corpus, None, 32, cpu), TokenDataset)
    assert isinstance(build_dataset(None, None, 32, cpu), ByteDataset)
    raw = tmp_path / "raw.txt"
    raw.write_bytes(b"modern transformer " * 200)
    assert isinstance(build_dataset(None, raw, 8, cpu), ByteDataset)


# ---------------------------------------------------------------------------
# Training on tokens
# ---------------------------------------------------------------------------


def tiny_config(**train_kw) -> Config:
    train_cfg = {
        "max_steps": 8,
        "warmup_steps": 2,
        "seq_len": 32,
        "micro_batch_size": 4,
        "precision": "fp32",
        "log_interval": 2,
        "eval_interval": 4,
        "ckpt_interval": 4,
        "lr": 3e-3,
    }
    train_cfg.update(train_kw)
    return Config(
        model=ModelConfig(
            d_model=32,
            n_layers=2,
            vocab_size=8,  # deliberately wrong, the corpus must win
            max_seq_len=64,
            attention=AttentionConfig(kind="mqa", n_heads=4, n_kv_heads=1),
            ffn=FFNConfig(kind="swiglu", multiple_of=1),
        ),
        train=TrainConfig(**train_cfg),
    )


def test_the_corpus_sets_the_vocabulary_not_the_config(corpus):
    cfg = tiny_config()
    assert cfg.model.vocab_size == 8
    train(cfg, data_dir=corpus, mlflow=False, model_name="test-vocab")
    assert cfg.model.vocab_size == VOCAB, "a mismatched vocab would crash the embedding"


def metrics_of(model_dir) -> list[dict]:
    """Read the metrics.jsonl RunLogger wrote next to this run's checkpoints.

    ``outputs/models/<name>_run-NN_date-.../`` pairs with
    ``outputs/logs/<name>_run-NN/``.
    """
    run_name = model_dir.name.split("_date-")[0]
    log_file = model_dir.parents[1] / "logs" / run_name / "metrics.jsonl"
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]


def test_training_on_tokens_reduces_the_loss(corpus):
    model_dir = train(
        tiny_config(max_steps=30), data_dir=corpus, mlflow=False, model_name="test-tok"
    )
    records = metrics_of(model_dir)
    assert len(records) >= 2
    losses = [r["loss/ce"] for r in records if "loss/ce" in r]
    assert losses[-1] < losses[0], f"loss went {losses[0]:.3f} to {losses[-1]:.3f}"
    assert (model_dir / "ckpt.pt").exists()


def test_validation_loss_is_logged_and_saves_the_best_model(corpus):
    model_dir = train(
        tiny_config(max_steps=8, eval_interval=4),
        data_dir=corpus,
        mlflow=False,
        model_name="test-val",
    )
    assert (model_dir / "best_model.pt").exists(), "an eval must save the best model"
    assert any("loss/val" in r for r in metrics_of(model_dir))


def test_no_validation_without_a_prepared_corpus(tmp_path):
    """Byte-level ablations have no val split, and must not crash on eval."""
    model_dir = train(
        tiny_config(max_steps=6, eval_interval=2),
        mlflow=False,
        model_name="test-noval",
    )
    assert not (model_dir / "best_model.pt").exists()
    assert not any("loss/val" in r for r in metrics_of(model_dir))


# ---------------------------------------------------------------------------
# Resume, which a 23 hour run depends on
# ---------------------------------------------------------------------------


def test_checkpoint_carries_scheduler_and_scaler(corpus, tmp_path):
    """Restoring weights alone restarts the schedule from zero, silently."""
    from mllm.model import Transformer
    from mllm.optim import build_optimizer, build_scheduler

    cfg = tiny_config()
    cfg.model.vocab_size = VOCAB
    model = Transformer(cfg.model)
    opt = build_optimizer(model, cfg.model, cfg.train)
    sched = build_scheduler(opt, cfg.train)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    for _ in range(5):
        opt.step()
        sched.step()
    lr_before = sched.get_last_lr()[0]

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, sched, scaler, 4, cfg, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "scheduler", "scaler", "step", "config"} <= set(ckpt)

    fresh = Transformer(cfg.model)
    fresh_opt = build_optimizer(fresh, cfg.model, cfg.train)
    fresh_sched = build_scheduler(fresh_opt, cfg.train)
    fresh_scaler = torch.amp.GradScaler("cpu", enabled=False)
    step = load_checkpoint(path, fresh, fresh_opt, fresh_sched, fresh_scaler, "cpu")

    assert step == 5, "training must continue after the saved step, not repeat it"
    assert fresh_sched.get_last_lr()[0] == pytest.approx(lr_before)
    for a, b in zip(model.parameters(), fresh.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_resuming_continues_instead_of_restarting(corpus):
    """The property a stopped pod depends on."""
    first = train(
        tiny_config(max_steps=6, ckpt_interval=3),
        data_dir=corpus,
        mlflow=False,
        model_name="test-resume-a",
    )
    ckpt = first / "ckpt.pt"
    assert ckpt.exists()

    second = train(
        tiny_config(max_steps=12, ckpt_interval=3),
        data_dir=corpus,
        mlflow=False,
        model_name="test-resume-b",
        resume=ckpt,
    )
    restored = torch.load(second / "ckpt.pt", map_location="cpu", weights_only=False)
    assert restored["step"] == 11, "the resumed run must end at the new max_steps"

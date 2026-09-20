"""The bilingual mixing logic, checked without touching the network.

``prepare_data.py`` runs on a rented pod, so a bug in it costs billed time.
Downloads and tokenization are faked here, which leaves exactly the part worth
testing: whether the languages come out at the requested ratio, whether the
budget is respected, and whether shards are deleted as they are consumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

prepare_data = pytest.importorskip("prepare_data")

TOKENS_PER_SHARD = 1000
MARKER = {"en": 1, "fr": 2}


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Replace the hub by shards of constant, per-language token values.

    English shards are all 1s and French all 2s, so the written file can be
    attributed to a language token by token.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    fetched: list[Path] = []

    def fake_list_shards(lang: str) -> list[str]:
        return [f"{lang}/shard_{i:03d}.parquet" for i in range(20)]

    def fake_fetch(lang: str, remote: str, scratch_dir: Path) -> Path:
        path = scratch / f"{lang}_{Path(remote).name}"
        path.write_bytes(b"fake shard")
        fetched.append(path)
        return path

    def fake_tokenize(
        path: Path, tokenizer, budget: int | None = None, batch: int = 1000
    ) -> np.ndarray:
        lang = path.name.split("_")[0]
        n = TOKENS_PER_SHARD if budget is None else min(TOKENS_PER_SHARD, budget)
        return np.full(n, MARKER[lang], dtype=prepare_data.DTYPE)

    monkeypatch.setattr(prepare_data, "list_shards", fake_list_shards)
    monkeypatch.setattr(prepare_data, "fetch", fake_fetch)
    monkeypatch.setattr(prepare_data, "tokenize_shard", fake_tokenize)
    return fetched


def read_tokens(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=prepare_data.DTYPE)


def run(tmp_path, budgets, fake_hub, split="train"):
    return prepare_data.write_split(
        None, budgets, tmp_path, split, tmp_path / "scratch", workers=2
    )


# ---------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("en_ratio", [0.7, 0.5, 0.3])
def test_mixture_lands_near_the_requested_ratio(tmp_path, fake_hub, en_ratio):
    total = 20_000
    budgets = {"en": int(total * en_ratio), "fr": total - int(total * en_ratio)}
    counts = run(tmp_path, budgets, fake_hub)

    actual = counts["en"] / sum(counts.values())
    assert abs(actual - en_ratio) < 0.02, (
        f"asked for {en_ratio:.0%} English, wrote {actual:.1%}"
    )


def test_both_languages_reach_the_file(tmp_path, fake_hub):
    counts = run(tmp_path, {"en": 5_000, "fr": 5_000}, fake_hub)
    tokens = read_tokens(tmp_path / "train.bin")
    assert set(np.unique(tokens)) == {1, 2}
    assert counts["en"] == 5_000 and counts["fr"] == 5_000


def test_languages_alternate_along_the_file(tmp_path, fake_hub):
    """Shard-level interleaving, which random-offset sampling makes sufficient."""
    run(tmp_path, {"en": 10_000, "fr": 10_000}, fake_hub)
    tokens = read_tokens(tmp_path / "train.bin")
    langs = tokens[::TOKENS_PER_SHARD]
    switches = int(np.sum(langs[1:] != langs[:-1]))
    assert switches >= 5, f"only {switches} switches, that is a concatenation"


def test_a_zero_budget_drops_the_language(tmp_path, fake_hub):
    """An English-only run must not emit a single French token."""
    counts = run(tmp_path, {"en": 5_000, "fr": 0}, fake_hub)
    assert counts == {"en": 5_000}
    assert set(np.unique(read_tokens(tmp_path / "train.bin"))) == {1}


def test_the_budget_is_respected_exactly(tmp_path, fake_hub):
    """Shards are truncated, so the file must not overshoot."""
    run(tmp_path, {"en": 7_500, "fr": 2_500}, fake_hub)
    assert len(read_tokens(tmp_path / "train.bin")) == 10_000


def test_running_out_of_shards_warns_instead_of_hanging(tmp_path, fake_hub, capsys):
    # 20 fake shards of 1000 tokens cap each language at 20_000
    counts = run(tmp_path, {"en": 50_000, "fr": 1_000}, fake_hub)
    assert counts["en"] == 20_000
    assert "ran out of shards" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Disk hygiene, which a 20 GB volume depends on
# ---------------------------------------------------------------------------


def test_shards_are_deleted_as_they_are_consumed(tmp_path, fake_hub):
    run(tmp_path, {"en": 5_000, "fr": 5_000}, fake_hub)
    assert len(fake_hub) >= 10, "the test should have fetched several shards"
    survivors = [p for p in fake_hub if p.exists()]
    assert not survivors, f"{len(survivors)} shards left on disk"


def test_a_prefetched_shard_is_cleaned_when_its_language_finishes(tmp_path, fake_hub):
    """English finishing early must not leave its next download on disk.

    On a 20 GB volume a leaked 2 GB shard is 10% of the space, and it survives
    until the pod is destroyed.
    """
    run(tmp_path, {"en": 1_500, "fr": 8_000}, fake_hub)
    survivors = [p for p in fake_hub if p.exists()]
    assert not survivors, f"{len(survivors)} shards leaked after a language finished"


def test_only_the_needed_tokens_are_produced(tmp_path, fake_hub, monkeypatch):
    """A shard holds 714M tokens, so tokenizing all of it to keep 1M is waste."""
    budgets_seen: list[int | None] = []
    original = prepare_data.tokenize_shard

    def spy(path, tokenizer, budget=None, batch=1000):
        budgets_seen.append(budget)
        return original(path, tokenizer, budget, batch)

    monkeypatch.setattr(prepare_data, "tokenize_shard", spy)
    run(tmp_path, {"en": 1_500, "fr": 500}, fake_hub)
    assert budgets_seen, "tokenize_shard should have been called"
    assert all(b is not None for b in budgets_seen), "the budget must be passed down"


# ---------------------------------------------------------------------------
# The output is what TokenDataset expects
# ---------------------------------------------------------------------------


def test_written_file_loads_as_a_dataset(tmp_path, fake_hub):
    """The two halves of the pipeline must actually fit together."""
    import torch

    from mllm.data import TokenDataset

    run(tmp_path, {"en": 7_000, "fr": 3_000}, fake_hub)
    (tmp_path / "meta.json").write_text(
        json.dumps({"vocab_size": 32000, "dtype": "uint16", "mixture": {"en": 0.7}}),
        encoding="utf-8",
    )
    ds = TokenDataset(tmp_path, "train", seq_len=64, device=torch.device("cpu"))
    x, _ = ds.batch(2)
    assert x.shape == (2, 64)
    assert set(int(v) for v in x.unique()) <= {1, 2}


def test_dtype_holds_the_target_vocabulary():
    """uint16 covers CroissantLLM's 32k, and build_tokenizer refuses anything larger."""
    assert np.iinfo(prepare_data.DTYPE).max >= 32_000
    assert np.dtype(prepare_data.DTYPE).itemsize == 2


def test_sources_point_at_the_expected_datasets():
    en, fr = prepare_data.SOURCES["en"], prepare_data.SOURCES["fr"]
    assert "fineweb-edu" in en["repo"] and en["prefix"] == "sample/10BT/"
    assert "fineweb-2" in fr["repo"] and "fra_Latn" in fr["prefix"]

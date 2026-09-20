"""Every YAML profile shipped in configs/ must load and validate (M0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mllm.config import Config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_shipped_config_loads(path: Path):
    cfg = Config.from_yaml(path)
    assert cfg.model.n_layers > 0


def test_all_five_profiles_present():
    names = {p.stem for p in CONFIG_FILES}
    expected = {"base", "llama_style_150m", "moe_1b_a200m", "mla_long_ctx", "gemma_style"}
    assert expected <= names, f"missing profiles: {expected - names}"

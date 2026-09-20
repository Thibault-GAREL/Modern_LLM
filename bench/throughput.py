"""Prefill and decode throughput, and peak memory, per attention variant.

Prefill and decode are two different regimes and mixing them hides everything.
Prefill processes the whole prompt at once and is compute bound, so it is
mostly insensitive to the KV layout. Decode produces one token at a time,
reads the entire cache to do it, and is bound by memory bandwidth. That is why
GQA and MLA speed up decoding without touching the arithmetic.

    python bench/throughput.py
    python bench/throughput.py --d-model 1024 --layers 12 --context 2048
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from tqdm import tqdm

from mllm.cache import build_model_cache
from mllm.config import AttentionConfig, FFNConfig, ModelConfig
from mllm.model import Transformer
from mllm.utils.numerics import autocast_dtype, pick_device, resolve_precision
from mllm.utils.seed import set_determinism


def variants(n_heads: int) -> dict[str, AttentionConfig]:
    return {
        "MHA": AttentionConfig(kind="mha", n_heads=n_heads, n_kv_heads=None),
        "GQA g=4": AttentionConfig(kind="gqa", n_heads=n_heads, n_kv_heads=n_heads // 4),
        "MQA": AttentionConfig(kind="mqa", n_heads=n_heads, n_kv_heads=1),
        "MLA (naive)": AttentionConfig(
            kind="mla", n_heads=n_heads, n_kv_heads=None,
            kv_lora_rank=256, q_lora_rank=None,
            qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        ),
        "MLA (absorbed)": AttentionConfig(
            kind="mla", n_heads=n_heads, n_kv_heads=None,
            kv_lora_rank=256, q_lora_rank=None,
            qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        ),
        "SWA w=256 (GQA g=4)": AttentionConfig(
            kind="gqa", n_heads=n_heads, n_kv_heads=n_heads // 4,
            sliding_window=256, global_every=4,
        ),
    }


def build(att: AttentionConfig, args) -> ModelConfig:
    return ModelConfig(
        d_model=args.d_model,
        n_layers=args.layers,
        vocab_size=args.vocab,
        max_seq_len=args.context + args.decode + 8,
        attention=att,
        ffn=FFNConfig(kind="swiglu", multiple_of=256),
    )


@torch.no_grad()
def measure(
    cfg: ModelConfig, args, device, amp_dtype, *, absorbed: bool = False
) -> dict[str, float]:
    set_determinism(0)
    if device.type == "cuda":
        # release the previous variant before measuring, otherwise its blocks
        # stay counted as allocated and every row after the first is inflated
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model = Transformer(cfg).to(device).eval()
    idx = torch.randint(0, cfg.vocab_size, (args.batch, args.context), device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    ctx = torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    # warmup, so the first kernel launches do not land in the measurement
    with ctx:
        cache = build_model_cache(cfg, max_len=args.context + args.decode + 8)
        model(idx, cache=cache)
        model(idx[:, -1:], cache=cache, absorbed=absorbed)
    sync()

    # prefill
    with ctx:
        cache = build_model_cache(cfg, max_len=args.context + args.decode + 8)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            cache.reset()
            model(idx, cache=cache)
        sync()
        prefill_s = (time.perf_counter() - t0) / args.repeats

    # decode, one token at a time against the filled cache
    with ctx:
        token = idx[:, -1:]
        sync()
        t0 = time.perf_counter()
        for _ in range(args.decode):
            model(token, cache=cache, absorbed=absorbed)
        sync()
        decode_s = (time.perf_counter() - t0) / args.decode

    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    return {
        "prefill_tok_s": args.batch * args.context / prefill_s,
        "decode_tok_s": args.batch / decode_s,
        "decode_ms": decode_s * 1000,
        "peak_mib": peak,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--decode", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    device = pick_device()
    precision = resolve_precision("bf16", device)
    amp_dtype = autocast_dtype(precision)

    print(
        f"\nmodel: {args.layers} layers, d_model {args.d_model}, {args.heads} heads"
        f"\nserving: context {args.context}, batch {args.batch}, "
        f"{args.decode} decode steps, {device} {precision}\n"
    )

    rows = {}
    for name, att in tqdm(variants(args.heads).items(), desc="throughput", unit="variant"):
        # MLA has two paths: the training one materializes k and v from the
        # latent, the inference one folds W_UK and W_UV away instead
        rows[name] = measure(
            build(att, args), args, device, amp_dtype, absorbed="absorbed" in name
        )

    header = (
        f"{'variant':<22} {'prefill tok/s':>14} {'decode tok/s':>13} "
        f"{'ms/token':>9} {'peak MiB':>10} {'vs MHA':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    baseline = rows["MHA"]["decode_tok_s"]
    for name, r in rows.items():
        print(
            f"{name:<22} {r['prefill_tok_s']:>14,.0f} {r['decode_tok_s']:>13,.0f} "
            f"{r['decode_ms']:>9.2f} {r['peak_mib']:>10,.0f} "
            f"{r['decode_tok_s'] / baseline:>7.2f}x"
        )

    print(
        "\nPrefill is compute bound, so the KV layout barely moves it. Decode reads "
        "the whole cache per token, which is where the layout shows up."
    )


if __name__ == "__main__":
    main()

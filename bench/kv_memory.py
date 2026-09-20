"""KV cache cost per token, per attention variant.

This is the table that decides between MHA, MQA, GQA and MLA. None of them
changes quality much, they change how much memory one token of context costs
during generation, and past a certain context that memory is what limits both
the batch size and the decode speed.

    python bench/kv_memory.py
    python bench/kv_memory.py --layers 32 --d-model 4096 --heads 32 --context 131072
"""

from __future__ import annotations

import argparse

import torch

from mllm.cache import KVCache, LatentCache, RingCache


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024 or unit == "TB":
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-rope-head-dim", type=int, default=64)
    args = p.parse_args()

    head_dim = args.d_model // args.heads
    dtype = torch.float16
    L, C, B = args.layers, args.context, args.batch

    rows = [
        ("MHA", KVCache(L, args.heads, head_dim, C), C),
        ("GQA g=4", KVCache(L, args.heads // 4, head_dim, C), C),
        ("GQA g=8", KVCache(L, args.heads // 8, head_dim, C), C),
        ("MQA", KVCache(L, 1, head_dim, C), C),
        (
            f"SWA w={args.window} (GQA g=4)",
            RingCache(L, args.heads // 4, head_dim, args.window),
            min(args.window, C),
        ),
        (
            "MLA",
            LatentCache(L, args.kv_lora_rank, args.qk_rope_head_dim, C),
            C,
        ),
    ]

    print(
        f"\nmodel: {L} layers, d_model {args.d_model}, {args.heads} heads "
        f"(head_dim {head_dim}), fp16"
    )
    print(f"serving: context {C:,}, batch {B}\n")
    header = f"{'variant':<26} {'bytes/token':>13} {'total cache':>13} {'vs MHA':>9}"
    print(header)
    print("-" * len(header))

    baseline = rows[0][1].bytes_per_token(dtype)
    for name, cache, resident in rows:
        per_token = cache.bytes_per_token(dtype)
        total = per_token * resident * B
        print(
            f"{name:<26} {human(per_token):>13} {human(total):>13} "
            f"{baseline / per_token:>8.1f}x"
        )

    print(
        "\nNote: the sliding window row keeps the same cost per token, it simply "
        f"stops storing past {args.window:,} entries, which is where its saving "
        "comes from."
    )
    crossover = (args.kv_lora_rank + args.qk_rope_head_dim) / (2 * head_dim)
    print(
        f"MLA beats a dense cache below {crossover:.1f} KV heads at this head_dim, "
        f"so it is worth the extra code only above that."
    )


if __name__ == "__main__":
    main()

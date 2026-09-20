"""muP coordinate check (Yang et al., 2022, arXiv 2203.03466).

The only meaningful test of a muP implementation. Train the same toy stack at
several widths and record the average absolute activation ("coordinate size")
of every layer at every step. Under muP those curves lie on top of each other,
under standard parametrization they spread apart as width grows.

Run it with:

    python bench/coord_check.py --steps 8 --out docs/assets

Since M5 this runs on the real ``Transformer``, so all four muP changes are
exercised at once: (a) the init, (b) the optimizer groups, (c) the ``1 /
head_dim`` attention scale and (d) the output logit multiplier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from mllm.config import AttentionConfig, FFNConfig, InitConfig, ModelConfig, MuPConfig, TrainConfig
from mllm.init import init_weights
from mllm.model import Transformer
from mllm.optim import build_param_groups
from mllm.utils.seed import set_determinism

WIDTHS = (128, 256, 512, 1024)
BASE_WIDTH = 128


def run_width(
    d_model: int, *, mup: bool, n_layers: int, steps: int, vocab: int, lr: float, seed: int
) -> list[list[float]]:
    """Train the real Transformer at one width, profiling activations per layer."""
    set_determinism(seed)
    n_heads = 8
    cfg = ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab,
        max_seq_len=64,
        mup=MuPConfig(enabled=mup, base_d_model=BASE_WIDTH),
        attention=AttentionConfig(
            kind="gqa",
            n_heads=n_heads,
            n_kv_heads=2,
            scale="mup" if mup else "1/sqrt(d)",
        ),
        ffn=FFNConfig(kind="swiglu", multiple_of=1),
        init=InitConfig(scheme="fixed", std=0.02, scaled_residual=True),
    )
    train_cfg = TrainConfig(lr=lr, max_steps=steps, warmup_steps=0)
    model = Transformer(cfg)
    init_weights(model, cfg)
    opt = torch.optim.AdamW(
        build_param_groups(model, cfg, train_cfg), lr=lr, betas=(0.9, 0.95)
    )

    coords: list[float] = []

    def record(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        coords.append(hidden.detach().abs().mean().item())

    handles = [b.register_forward_hook(record) for b in model.blocks]

    # A learnable task, not random targets. muP predicts width invariance for
    # *correlated* updates, and random labels produce pure gradient noise whose
    # updates cancel, which makes the coordinates shrink as sqrt(width).
    idx = torch.randint(0, vocab, (8, 32))
    targets = (idx + 1) % vocab

    profiles = []
    for _ in range(steps):
        coords.clear()
        _, loss, _ = model(idx, targets)
        profiles.append(list(coords))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for h in handles:
        h.remove()
    return profiles


def plot(results: dict, out_path: Path, steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = plt.cm.viridis([0.1, 0.4, 0.65, 0.9])

    for ax, mup in zip(axes, (False, True), strict=True):
        for color, width in zip(colors, WIDTHS, strict=True):
            final = results[(width, mup)][-1]
            ax.plot(
                range(len(final)),
                final,
                marker="o",
                markersize=4,
                color=color,
                label=f"d_model = {width}",
            )
        ax.set_title(
            f"{'muP' if mup else 'standard parametrization'}"
            f"{'  (curves should overlap)' if mup else '  (curves spread with width)'}"
        )
        ax.set_xlabel("layer index")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coordinate size (mean |activation|)")
    axes[1].legend(frameon=False)
    fig.suptitle(f"muP coordinate check after {steps} AdamW steps", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def spread(results: dict, mup: bool) -> float:
    """Max over layers of (max across widths / min across widths). 1.0 is perfect."""
    profiles = [results[(w, mup)][-1] for w in WIDTHS]
    worst = 0.0
    for layer in zip(*profiles, strict=True):
        worst = max(worst, max(layer) / max(min(layer), 1e-12))
    return worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--vocab", type=int, default=512)
    # 1e-2 pushes this toy into a regime where the linearized muP argument stops
    # holding and the spread triples. See docs/mup_coord_check.md.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = p.parse_args()

    results = {}
    jobs = [(w, m) for m in (False, True) for w in WIDTHS]
    for width, mup in tqdm(jobs, desc="coord check", unit="run"):
        results[(width, mup)] = run_width(
            width,
            mup=mup,
            n_layers=args.layers,
            steps=args.steps,
            vocab=args.vocab,
            lr=args.lr,
            seed=args.seed,
        )

    std_spread = spread(results, mup=False)
    mup_spread = spread(results, mup=True)
    print(f"\nwidth spread after {args.steps} steps (1.0 = perfectly width invariant)")
    print(f"  standard parametrization : {std_spread:6.2f}x")
    print(f"  muP                      : {mup_spread:6.2f}x")
    if mup_spread < std_spread:
        print(f"  muP is {std_spread / mup_spread:.1f}x flatter across widths")
    else:
        print("  WARNING: muP did not flatten the profile, the implementation is wrong")

    plot(results, args.out / "mup_coord_check.png", args.steps)


if __name__ == "__main__":
    main()

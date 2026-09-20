"""Optimizer, parameter groups, and learning rate schedules.

The 2017 recipe was Adam with ``beta2 = 0.98``, plain L2, and an inverse
square root schedule with 4000 warmup steps. What replaced it:

  AdamW with ``betas = (0.9, 0.95)``, decoupling weight decay from the
  gradient (Loshchilov and Hutter, 2017, arXiv 1711.05101).

  Weight decay applied to matrices only. Decaying a norm gain shrinks the
  signal that gain exists to rescale, and decaying an embedding pulls rare
  tokens towards zero purely for being rare. Both are silent quality losses,
  which is why the groups here are explicit and tested.

  Cosine with warmup, or WSD (MiniCPM, 2024, arXiv 2404.06395), whose stable
  plateau means the token budget is not locked in before the run starts.

  Gradient clipping at 1.0, absent from the paper.
"""

from __future__ import annotations

import math

from torch import nn, optim

from mllm.config import ModelConfig, TrainConfig
from mllm.init import width_multiplier
from mllm.layers.norm import DyT, LayerNorm, RMSNorm

NORM_TYPES = (RMSNorm, LayerNorm, DyT, nn.LayerNorm)


def build_param_groups(
    model: nn.Module, model_cfg: ModelConfig, train_cfg: TrainConfig
) -> list[dict]:
    """Split parameters into decay and no-decay groups, and muP groups.

    No decay for biases, normalization gains and embeddings. Under muP the
    hidden matrices additionally get their learning rate divided by the width
    multiplier, which is change (b) of the four.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    def take(param: nn.Parameter | None, bucket: list) -> None:
        # tied weights appear twice, and adding them twice would double their
        # decay and corrupt the optimizer state
        if param is None or id(param) in seen:
            return
        seen.add(id(param))
        bucket.append(param)

    mup_scaled: set[int] = set()
    mult = width_multiplier(model_cfg)

    for module in model.modules():
        if isinstance(module, NORM_TYPES):
            for p in module.parameters(recurse=False):
                take(p, no_decay)
        elif isinstance(module, nn.Embedding):
            take(module.weight, no_decay)
        elif isinstance(module, nn.Linear):
            if model_cfg.mup.enabled and id(module.weight) not in seen:
                mup_scaled.add(id(module.weight))
            take(module.weight, decay)
            take(module.bias, no_decay)
        else:
            for p in module.parameters(recurse=False):
                # anything left with a single dimension is a gain or a bias
                take(p, no_decay if p.dim() <= 1 else decay)

    if not model_cfg.mup.enabled or mult == 1.0:
        return [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # muP change (b): matrices whose fan-in grows with width get lr / mult
    scaled = [p for p in decay if id(p) in mup_scaled]
    unscaled = [p for p in decay if id(p) not in mup_scaled]
    groups = [
        {
            "params": scaled,
            "weight_decay": train_cfg.weight_decay,
            "lr": train_cfg.lr / mult,
            "mup_scaled": True,
        },
        {"params": no_decay, "weight_decay": 0.0, "lr": train_cfg.lr},
    ]
    if unscaled:
        groups.append(
            {
                "params": unscaled,
                "weight_decay": train_cfg.weight_decay,
                "lr": train_cfg.lr,
            }
        )
    return groups


def build_optimizer(
    model: nn.Module, model_cfg: ModelConfig, train_cfg: TrainConfig
) -> optim.AdamW:
    groups = build_param_groups(model, model_cfg, train_cfg)
    return optim.AdamW(
        groups,
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        eps=1e-8,
        weight_decay=train_cfg.weight_decay,
    )


def lr_multiplier(step: int, cfg: TrainConfig) -> float:
    """Schedule value at ``step``, as a multiple of the base learning rate.

    Warmup is linear in both schedules. Cosine then decays over the whole run,
    which fixes the token budget in advance. WSD holds a plateau and only
    decays over the last ``decay_steps``, so a run can be forked or extended
    at any point on that plateau.
    """
    warmup = cfg.warmup_steps
    if warmup and step < warmup:
        return (step + 1) / warmup

    if cfg.schedule == "cosine":
        progress = (step - warmup) / max(cfg.max_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    # WSD
    decay_steps = cfg.decay_steps or 0
    decay_start = cfg.max_steps - decay_steps
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(decay_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 - (1.0 - cfg.min_lr_ratio) * progress


def build_scheduler(optimizer: optim.Optimizer, cfg: TrainConfig) -> optim.lr_scheduler.LambdaLR:
    """LambdaLR applying ``lr_multiplier`` on top of each group's own base lr.

    LambdaLR multiplies each group's ``initial_lr`` independently, so the muP
    groups keep their ``/ mult`` factor throughout the schedule.
    """
    return optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, cfg))

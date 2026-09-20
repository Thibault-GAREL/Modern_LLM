"""Minimal training loop.

Deliberately small. This exists so ablations can be run end to end, not to
compete with a real training stack, and everything it does not do is listed in
the README under "what is not in this repo".

What it does cover, because each one changes results rather than only speed:
mixed precision with fp32 master weights, gradient accumulation, gradient
clipping, the aux-loss-free balancing step, optional activation checkpointing,
and JSONL logging of every loss term separately.

    python -m mllm.train --config configs/best.yaml
    python -m mllm.train --config configs/bilingual_100m.yaml --data-dir data/bilingual
    python -m mllm.train --config configs/best.yaml --resume outputs/.../ckpt.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from mllm.config import Config
from mllm.data import ByteDataset, build_dataset
from mllm.model import Transformer
from mllm.optim import build_optimizer, build_scheduler
from mllm.run_logger import RunLogger
from mllm.utils.numerics import autocast_dtype, pick_device, resolve_precision
from mllm.utils.seed import set_determinism


def train(
    cfg: Config,
    *,
    data_path: Path | None = None,
    data_dir: Path | None = None,
    resume: Path | None = None,
    model_name: str = "modern-llm",
    mlflow: bool = True,
) -> Path:
    """Run the training loop and return the directory holding the checkpoints.

    Args:
        data_dir: a folder prepared by ``scripts/prepare_data.py``. Takes
            precedence over ``data_path`` and sets the vocabulary from the
            tokenizer that produced it.
        data_path: a raw file read as bytes, for ablations.
        resume: a checkpoint to continue from. A long run will be interrupted,
            and on RunPod ``/workspace`` is not persistent, so this is not
            optional comfort.
    """
    train_cfg = cfg.train
    set_determinism(train_cfg.seed)

    device = pick_device()
    precision = resolve_precision(train_cfg.precision, device)
    amp_dtype = autocast_dtype(precision)

    train_data = build_dataset(
        data_dir, data_path, train_cfg.seq_len, device, split="train"
    )
    val_data = (
        build_dataset(data_dir, None, train_cfg.seq_len, device, split="val", seed=1234)
        if data_dir is not None
        else None
    )
    # the corpus decides the vocabulary, never the config
    cfg.model.vocab_size = getattr(train_data, "vocab_size", ByteDataset.VOCAB_SIZE)
    cfg.model.max_seq_len = max(cfg.model.max_seq_len, train_cfg.seq_len)

    model = Transformer(cfg.model, z_loss_coef=train_cfg.z_loss_coef).to(device)
    model.gradient_checkpointing = train_cfg.activation_checkpointing
    optimizer = build_optimizer(model, cfg.model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))

    start_step = 0
    if resume is not None:
        start_step = load_checkpoint(resume, model, optimizer, scheduler, scaler, device)

    if train_cfg.compile:
        model = torch.compile(model)

    with RunLogger(model_name, mlflow=mlflow) as run:
        run.config(
            {
                **cfg.model_dump(),
                "data/source": train_data.source,
                "data/dir": str(data_dir) if data_dir else "",
                "runtime/device": str(device),
                "runtime/precision": precision,
                "runtime/params": model.n_params(),
                "runtime/params_active": model.n_active_params(),
            }
        )
        if precision != train_cfg.precision:
            run.log(
                f"precision: {train_cfg.precision} would be emulated here, "
                f"using {precision} instead"
            )
        run.log(f"data     : {train_data.source}")
        run.log(
            f"params   : {model.n_params():,} total, {model.n_active_params():,} active"
        )
        if resume is not None:
            run.log(f"resumed  : {resume} at step {start_step}")

        run.section("training")
        model.train()
        tokens_per_step = (
            train_cfg.micro_batch_size * train_cfg.grad_accum_steps * train_cfg.seq_len
        )

        for step in run.track(
            range(start_step, train_cfg.max_steps),
            desc="train",
            total=train_cfg.max_steps - start_step,
        ):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}

            for _ in range(train_cfg.grad_accum_steps):
                x, y = train_data.batch(train_cfg.micro_batch_size)
                with torch.autocast(
                    device.type, dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    _, loss, aux = model(x, y)
                scaler.scale(loss / train_cfg.grad_accum_steps).backward()
                for key, value in aux.as_dict().items():
                    totals[key] = (
                        totals.get(key, 0.0) + value / train_cfg.grad_accum_steps
                    )

            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # the aux-loss-free bias moves once per optimizer step, not per micro batch
            model.balance_step()

            metrics = {
                **totals,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "tokens": (step + 1) * tokens_per_step,
                "step_time_s": time.perf_counter() - t0,
            }
            for i, routing in enumerate(model.routing_metrics()):
                metrics.update(
                    {f"{k}/layer{i}": v for k, v in routing.as_dict().items()}
                )
            run.metric(step=step, **metrics)

            if val_data is not None and (step + 1) % train_cfg.eval_interval == 0:
                val_loss = evaluate(model, val_data, train_cfg, device, amp_dtype)
                run.metric(step=step, **{"loss/val": val_loss})
                if run.best("loss/val", val_loss, step):
                    # inference only, so no optimizer state and a third of the size
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, step, cfg,
                        run.model_dir / "best_model.pt", weights_only=True,
                    )

            if train_cfg.ckpt_interval and (step + 1) % train_cfg.ckpt_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, step, cfg,
                    run.model_dir / "ckpt.pt",
                )

        save_checkpoint(
            model, optimizer, scheduler, scaler, train_cfg.max_steps - 1, cfg,
            run.model_dir / "ckpt.pt",
        )
        run.done()
        return Path(run.model_dir)


@torch.no_grad()
def evaluate(model, dataset, train_cfg, device, amp_dtype, batches: int = 20) -> float:
    """Held-out cross entropy, on a fixed seed so two runs stay comparable."""
    model.eval()
    losses = []
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for _ in range(batches):
            x, y = dataset.batch(train_cfg.micro_batch_size)
            _, _, aux = model(x, y)
            losses.append(float(aux.ce))
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    model, optimizer, scheduler, scaler, step, cfg: Config, path: Path,
    *, weights_only: bool = False,
) -> None:
    """Save a checkpoint, either resumable or inference-sized.

    A resumable checkpoint carries the optimizer, the scheduler and the scaler.
    Dropping any of them makes a resumed run restart its learning rate schedule
    from zero, which is silent and wrong.

    ``weights_only=True`` keeps just the model, which is a third of the size.
    AdamW holds two fp32 moments per parameter, so the optimizer state is twice
    the model itself. The best-so-far checkpoint is only ever loaded for
    inference, so paying 1.33 GB for it instead of 444 MB is what filled a
    20 GB volume and killed a run at step 1999.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    payload = {"step": step, "model": inner.state_dict(), "config": cfg.model_dump()}
    if not weights_only:
        payload |= {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
    # write beside the target then rename, so a full disk cannot leave a
    # half-written file where a valid checkpoint used to be
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device) -> int:
    """Restore a run and return the step to continue from."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"]) + 1


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``--set model.moe.n_experts=8`` style overrides.

    Re-validates through the schema afterwards, so an override cannot smuggle
    in an inconsistent config that the YAML files are checked against.
    """
    if not overrides:
        return cfg
    raw = cfg.model_dump()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            if part not in node:
                raise ValueError(f"unknown config section {part!r} in {key!r}")
            node = node[part]
        if leaf not in node:
            raise ValueError(f"unknown config field {leaf!r} in {key!r}")
        node[leaf] = yaml.safe_load(value)  # gives int, float, bool or str
    return Config.model_validate(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="folder prepared by scripts/prepare_data.py (train.bin, val.bin, meta.json)",
    )
    p.add_argument("--data", type=Path, default=None, help="raw file, read as bytes")
    p.add_argument(
        "--resume", type=Path, default=None, help="checkpoint to continue from"
    )
    p.add_argument("--name", type=str, default="modern-llm", help="run name")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config field, e.g. --set model.d_model=128",
    )
    args = p.parse_args()

    cfg = apply_overrides(Config.from_yaml(args.config), args.overrides)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
        cfg.train.warmup_steps = min(cfg.train.warmup_steps, max(args.max_steps // 10, 1))
    if args.seq_len is not None:
        cfg.train.seq_len = args.seq_len
    if args.batch_size is not None:
        cfg.train.micro_batch_size = args.batch_size

    train(
        cfg,
        data_path=args.data,
        data_dir=args.data_dir,
        resume=args.resume,
        model_name=args.name,
        mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()

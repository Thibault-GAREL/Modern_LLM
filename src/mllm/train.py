"""Minimal training loop.

Deliberately small. This exists so ablations can be run end to end, not to
compete with a real training stack, and everything it does not do is listed in
the README under "what is not in this repo".

What it does cover, because each one changes results rather than only speed:
mixed precision with fp32 master weights, gradient accumulation, gradient
clipping, the aux-loss-free balancing step, optional activation checkpointing,
and JSONL logging of every loss term separately.

    python -m mllm.train --config configs/best.yaml
    python -m mllm.train --config configs/mai_v2.yaml --data-dir data/bilingual   # M.A.I v2
    python -m mllm.train --config configs/best.yaml --resume outputs/.../ckpt.pt
    python -m mllm.train ... --cost-per-hour 0.74   # dollars spent, logged per step
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch import nn

from mllm.config import Config
from mllm.data import ByteDataset, build_dataset
from mllm.layer_stats import LayerStats
from mllm.model import Transformer
from mllm.optim import build_optimizer, build_scheduler
from mllm.run_logger import RunLogger
from mllm.utils.numerics import autocast_dtype, pick_device, resolve_precision
from mllm.utils.seed import set_determinism

# Dense bf16 tensor throughput with fp32 accumulation, in TFLOPS, from the vendor
# sheets. Checked in order, so the PCIe H100 is matched before the SXM one.
PEAK_TFLOPS = {
    "H100 PCIe": 756.0,
    "H100": 989.0,
    "A100": 312.0,
    "L40S": 362.0,
    "RTX 6000 Ada": 364.0,
    "A6000": 154.8,
    "A40": 149.7,
    "4090": 165.2,
    "3090": 71.0,
}


def peak_tflops(device: torch.device) -> float | None:
    """Peak of the current GPU, or None when it is not in the table."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    for key, value in PEAK_TFLOPS.items():
        if re.search(rf"\b{re.escape(key)}\b", name):
            return value
    return None


def flops_per_token(model: nn.Module, seq_len: int) -> float:
    """Training FLOPs per token, ``6N`` plus the attention scores (PaLM, appendix B).

    An estimate, not a count: it ignores the norms and the softmax, and counts
    the tied LM head once although the MTP heads reuse it. Good enough to see
    whether the GPU is busy, which is all MFU is for.
    """
    cfg = model.cfg
    return 6 * model.n_params() + 12 * cfg.n_layers * cfg.d_model * seq_len


@dataclass
class ComputeMeter:
    """Wall time and money spent on one model, summed across resumed runs.

    Each run adds its own segment at its own hourly rate, so a run started on
    one pod and finished on a more expensive one is still counted right. The
    totals travel inside the checkpoint, which is what makes them survive a
    stopped pod.
    """

    cost_per_hour: float = 0.0
    seconds: float = 0.0  # spent before this run
    cost_usd: float = 0.0
    _t0: float = field(default=0.0, repr=False)

    def start(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def run_seconds(self) -> float:
        return time.perf_counter() - self._t0 if self._t0 else 0.0

    def totals(self) -> tuple[float, float]:
        """``(seconds, dollars)`` since the very first run of this model."""
        run_s = self.run_seconds
        return self.seconds + run_s, self.cost_usd + run_s / 3600 * self.cost_per_hour

    def state_dict(self) -> dict[str, float]:
        seconds, dollars = self.totals()
        return {"seconds": seconds, "cost_usd": dollars}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.seconds = float(state.get("seconds", 0.0))
        self.cost_usd = float(state.get("cost_usd", 0.0))


def train(
    cfg: Config,
    *,
    data_path: Path | None = None,
    data_dir: Path | None = None,
    resume: Path | None = None,
    model_name: str | None = None,
    mlflow: bool = True,
    cost_per_hour: float = 0.0,
    prior_hours: float = 0.0,
    prior_cost_usd: float = 0.0,
    project_root: Path | None = None,
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
        cost_per_hour: what the machine bills, in dollars. Turns the step
            time into a running cost, logged next to the loss.
        prior_hours, prior_cost_usd: time and money spent before checkpoints
            carried them. Ignored when the resumed checkpoint has its own.
        model_name: the run name. Defaults to the config's ``name``
            (``mai-v2``), or ``modern-llm`` for a profile without one.
        project_root: where ``outputs/`` and ``mlruns/`` go. Defaults to the
            repository, tests point it at a temporary folder.
    """
    train_cfg = cfg.train
    model_name = model_name or cfg.name or "modern-llm"
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

    compute = ComputeMeter(
        cost_per_hour=cost_per_hour, seconds=prior_hours * 3600, cost_usd=prior_cost_usd
    )
    start_step = 0
    if resume is not None:
        start_step = load_checkpoint(
            resume, model, optimizer, scheduler, scaler, device,
            dataset=train_data, compute=compute,
        )

    tokens_per_step = (
        train_cfg.micro_batch_size * train_cfg.grad_accum_steps * train_cfg.seq_len
    )
    flops_step = flops_per_token(model, train_cfg.seq_len) * tokens_per_step
    peak = peak_tflops(device)

    if train_cfg.compile:
        model = torch.compile(model)

    with RunLogger(model_name, project_root=project_root, mlflow=mlflow) as run:
        run.config(
            {
                **cfg.model_dump(),
                "data/source": train_data.source,
                "data/dir": str(data_dir) if data_dir else "",
                "runtime/device": str(device),
                "runtime/precision": precision,
                "runtime/params": model.n_params(),
                "runtime/params_active": model.n_active_params(),
                "runtime/cost_per_hour": cost_per_hour,
                "runtime/peak_tflops": peak or "",
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
        run.log(
            f"compute  : ${cost_per_hour:.2f}/h, "
            + (f"peak {peak:.0f} TFLOPS for the MFU" if peak else "GPU peak unknown, no MFU")
        )
        if resume is not None:
            run.log(f"resumed  : {resume} at step {start_step}")
            if compute.seconds:
                run.log(
                    f"           {compute.seconds / 3600:.2f} h and "
                    f"${compute.cost_usd:.2f} already spent on this model"
                )

        # layer/<module>/<stat> metrics read by visualisation/, forward hooks
        # would force recompiles under torch.compile so only grads and weights then
        layer_stats = LayerStats(
            model, every=train_cfg.layer_stats_interval, activations=not train_cfg.compile
        )
        run.log(layer_stats.describe())

        run.section("training")
        model.train()
        compute.start()

        for step in run.track(
            range(start_step, train_cfg.max_steps),
            desc="train",
            total=train_cfg.max_steps - start_step,
        ):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            layer_stats.start(step)
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
            # raw per-layer gradient norms, so measured before the clipping
            layer_stats.before_optimizer()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # the aux-loss-free bias moves once per optimizer step, not per micro batch
            model.balance_step()

            step_time = time.perf_counter() - t0
            total_s, total_usd = compute.totals()
            metrics = {
                **totals,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "tokens": (step + 1) * tokens_per_step,
                "step_time_s": step_time,
                "throughput/tok_s": tokens_per_step / step_time,
                "compute/hours": total_s / 3600,
                "compute/cost_usd": total_usd,
            }
            if peak:
                metrics["throughput/mfu"] = flops_step / step_time / (peak * 1e12)
            for i, routing in enumerate(model.routing_metrics()):
                metrics.update(
                    {f"{k}/layer{i}": v for k, v in routing.as_dict().items()}
                )
            metrics.update(layer_stats.collect())
            run.metric(step=step, **metrics)

            if val_data is not None and (step + 1) % train_cfg.eval_interval == 0:
                val_loss = evaluate(model, val_data, train_cfg, device, amp_dtype)
                run.metric(step=step, **{"loss/val": val_loss})
                if run.best("loss/val", val_loss, step):
                    # inference only, so no optimizer state and a third of the size
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, step, cfg,
                        run.model_dir / "best_model.pt", weights_only=True,
                        compute=compute,
                    )

            if train_cfg.ckpt_interval and (step + 1) % train_cfg.ckpt_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, step, cfg,
                    run.model_dir / "ckpt.pt", data=train_data, compute=compute,
                )

        save_checkpoint(
            model, optimizer, scheduler, scaler, train_cfg.max_steps - 1, cfg,
            run.model_dir / "ckpt.pt", data=train_data, compute=compute,
        )
        log_compute_summary(run, compute, train_cfg.max_steps - start_step, tokens_per_step,
                            flops_step, peak)
        run.done()
        return Path(run.model_dir)


def evaluate_checkpoint(cfg: Config, ckpt_path: Path, data_dir: Path) -> float:
    """Held-out loss of a saved model, without training a step.

    What a new validation set needs before a run resumes on it: the loss of
    the checkpoint it starts from, measured on the same windows the run will
    report, so the first point of the new curve has something to compare to.
    """
    device = pick_device()
    amp_dtype = autocast_dtype(resolve_precision(cfg.train.precision, device))
    val = build_dataset(data_dir, None, cfg.train.seq_len, device, split="val", seed=1234)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Transformer(Config.model_validate(ckpt["config"]).model).to(device)
    model.load_state_dict(ckpt["model"])
    loss = evaluate(model, val, cfg.train, device, amp_dtype)
    print(f"{ckpt_path.name} at step {ckpt['step']:,}: loss/val {loss:.4f} "
          f"on {cfg.train.eval_batches} x {cfg.train.micro_batch_size} windows of {val.path}")
    return loss


def log_compute_summary(run, compute: ComputeMeter, steps: int, tokens_per_step: int,
                        flops_step: float, peak: float | None) -> None:
    """What this run cost, and what the model has cost since its first step."""
    run_s = compute.run_seconds
    total_s, total_usd = compute.totals()
    run_tokens = steps * tokens_per_step
    run_usd = run_s / 3600 * compute.cost_per_hour
    line = (
        f"compute  : this run {run_s / 3600:.2f} h, {run_tokens / 1e9:.3f}B tokens, "
        f"{run_tokens / max(run_s, 1e-9):,.0f} tok/s"
    )
    if peak and run_s:
        line += f", MFU {steps * flops_step / run_s / (peak * 1e12):.1%}"
    run.log(line)
    if compute.cost_per_hour:
        run.log(
            f"cost     : this run ${run_usd:.2f}, "
            f"${run_usd / max(run_tokens / 1e9, 1e-9):.2f} per billion tokens"
        )
    run.log(f"total    : {total_s / 3600:.2f} h and ${total_usd:.2f} since the first step")


@torch.no_grad()
def evaluate(model, dataset, train_cfg, device, amp_dtype) -> float:
    """Held-out cross entropy, on the same windows at every call.

    The sampler is rewound first. Drawing fresh windows each time put a
    sampling noise of a few hundredths of a nat on every point of the curve,
    the same order as the progress between two evaluations.
    """
    model.eval()
    if hasattr(dataset, "rewind"):
        dataset.rewind()
    losses = []
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for _ in range(train_cfg.eval_batches):
            x, y = dataset.batch(train_cfg.micro_batch_size)
            _, _, aux = model(x, y)
            losses.append(float(aux.ce))
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    model, optimizer, scheduler, scaler, step, cfg: Config, path: Path,
    *, weights_only: bool = False, data=None, compute: ComputeMeter | None = None,
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

    ``data`` stores the training sampler's state. Without it a resumed run
    rebuilds its sampler from the same seed and replays the batches of step 0
    onwards, which turned the second half of a run into a second epoch over
    the first half.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    payload = {"step": step, "model": inner.state_dict(), "config": cfg.model_dump()}
    if compute is not None:
        payload["compute"] = compute.state_dict()
    if not weights_only:
        payload |= {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        if getattr(data, "generator", None) is not None:
            payload["data_rng"] = data.generator.get_state()
    # write beside the target then rename, so a full disk cannot leave a
    # half-written file where a valid checkpoint used to be
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path, model, optimizer, scheduler, scaler, device,
    *, dataset=None, compute: ComputeMeter | None = None,
) -> int:
    """Restore a run and return the step to continue from.

    Args:
        dataset: the training dataset, whose sampler continues where the
            checkpoint left it instead of restarting from its seed.
        compute: receives the time and money already spent on this model.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt["step"]) + 1

    generator = getattr(dataset, "generator", None)
    if generator is not None:
        if "data_rng" in ckpt:
            generator.set_state(ckpt["data_rng"].cpu())
        else:
            # checkpoints older than the sampler state: a seed derived from the
            # step at least draws windows the first run did not draw in order
            generator.manual_seed(dataset.seed + step)
    if compute is not None and "compute" in ckpt:
        compute.load_state_dict(ckpt["compute"])
    return step


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
    p.add_argument(
        "--name", type=str, default=None,
        help="run name, defaults to the config's name (mai-v2) or modern-llm",
    )
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--cost-per-hour",
        type=float,
        default=float(os.environ.get("MLLM_COST_PER_HOUR", 0.0)),
        help="dollars per hour billed for this machine, read `costPerHr` from "
        "`runpodctl get pod` rather than the price list (env MLLM_COST_PER_HOUR)",
    )
    p.add_argument(
        "--prior-hours", type=float, default=0.0,
        help="training hours spent before the checkpoint recorded them",
    )
    p.add_argument(
        "--prior-cost", type=float, default=0.0,
        help="dollars spent before the checkpoint recorded them",
    )
    p.add_argument(
        "--eval-only", type=Path, default=None, metavar="CKPT",
        help="print the validation loss of a checkpoint on --data-dir, then exit",
    )
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

    if args.eval_only is not None:
        if args.data_dir is None:
            raise SystemExit("--eval-only needs --data-dir, the folder holding val.bin")
        evaluate_checkpoint(cfg, args.eval_only, args.data_dir)
        return

    train(
        cfg,
        data_path=args.data,
        data_dir=args.data_dir,
        resume=args.resume,
        model_name=args.name,
        mlflow=not args.no_mlflow,
        cost_per_hour=args.cost_per_hour,
        prior_hours=args.prior_hours,
        prior_cost_usd=args.prior_cost,
    )


if __name__ == "__main__":
    main()

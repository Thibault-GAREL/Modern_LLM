"""Score a base model on the benchmarks, by log-likelihood of each answer.

A base model is never asked to pick a letter. Every choice is appended to the
context and the model's log-probability of those extra tokens is summed. The
best scored choice is its answer. This is the method of lm-evaluation-harness,
and the only one that gives a 100M model a fair test.

Two accuracies per multiple choice task:

  ``acc``, the plain sum, which favours short answers since every token costs
  probability.
  ``acc_norm``, the sum divided by the answer's length in bytes, which removes
  that bias. The headline number for HellaSwag and ARC.

LAMBADA has a single expected word, counted right when greedy decoding
produces exactly its tokens.

    # our checkpoint, named after the model it holds (mai-v1)
    python bench/eval_model.py --ckpt outputs/models/mai-v1/mai-v1.pt
    # a reference model from the Hugging Face Hub, through transformers
    python bench/eval_model.py --hf HuggingFaceTB/SmolLM2-135M
    # a quick look, 200 items per task
    python bench/eval_model.py --ckpt ... --name quick --limit 200

Results land in ``outputs/results/evals/<name>.json``, with the correctness of
every item, so the score on the first 100 items (the ones a person or an LLM
answers through ``bench/eval_llm.py``) can be recomputed without rerunning.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# pyarrow before torch: the other way round, Windows hits an access violation
# while pyarrow loads its DLLs, measured on torch 2.5.1 with transformers 5
import pyarrow.parquet  # noqa: F401, I001
import torch
import torch.nn.functional as F
from eval_tasks import LANGUAGE, REPO, Item, load_task, parse_tasks
from tqdm import tqdm

RESULTS = REPO / "outputs" / "results" / "evals"
SUBSET = 100  # the head every answerer shares, see eval_tasks.load_task


# ---------------------------------------------------------------------------
# Two kinds of model behind one interface
# ---------------------------------------------------------------------------


class OurModel:
    """A checkpoint of this repository, prompted ``</s>`` first like in training."""

    def __init__(self, ckpt: Path, tokenizer: str | None = None) -> None:
        from mllm.inference import DEFAULT_TOKENIZER, Tokenizer, load_model

        self.loaded = load_model(ckpt)
        self.tok = Tokenizer(tokenizer or DEFAULT_TOKENIZER)
        self.device = self.loaded.device
        self.max_len = self.loaded.max_len
        self.prefix = [self.tok.eos_id]
        self.encode = self.tok.encode
        self.describe = self.loaded.describe()
        self.params = self.loaded.params_deployed
        self.name = self.loaded.name  # mai-v1, when the checkpoint carries it

    def logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # right padding only, and attention is causal, so the pads are never seen
        return self.loaded.model(ids)[0]


class HFModel:
    """Any causal LM of the Hugging Face Hub, prompted the way it was trained.

    The document start each model saw differs: GPT-2 and SmolLM2 separate
    documents with their end of text token, CroissantLLM opens them with
    ``<s>``. The tokenizer knows which, so its own special tokens are kept and
    the end of text token is added only when it adds nothing.
    """

    def __init__(self, repo: str, dtype: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo)
        params = sum(p.numel() for p in model.parameters())
        if dtype == "auto":
            # fp32 when it fits, fp16 for the 1B class on a 6 GB card
            dtype = "fp16" if params > 500e6 and self.device.type == "cuda" else "fp32"
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.model = model.to(self.device, dtype=torch_dtype).eval()
        self.max_len = min(getattr(model.config, "max_position_embeddings", 2048), 4096)

        with_specials = self.tok("a")["input_ids"]
        bare = self.tok("a", add_special_tokens=False)["input_ids"]
        added = with_specials[: len(with_specials) - len(bare)]
        self.prefix = added or [self.tok.eos_token_id]
        self.encode = lambda text: self.tok(text, add_special_tokens=False)["input_ids"]
        self.params = params
        self.describe = f"{repo}, {params / 1e6:.0f}M parameters, {dtype}"

    def logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=mask).logits


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def requests_of(model, item: Item) -> list[tuple[list[int], list[int]]]:
    """``(input ids, continuation ids)`` for each choice of one item.

    The continuation is what the full text adds to the context once both are
    tokenized, which is how lm-evaluation-harness splits them: tokenizing the
    choice alone would lose the merge a SentencePiece space makes with it.
    """
    ctx = model.encode(item.context)
    out = []
    for choice in item.choices:
        whole = model.encode(item.context + choice)
        cont = whole[len(ctx):] if whole[: len(ctx)] == ctx else model.encode(choice)
        if not cont:
            cont = model.encode(choice)
        ids = (model.prefix + ctx + cont)[-model.max_len:]
        out.append((ids, cont))
    return out


@torch.no_grad()
def score_batch(model, batch: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
    """Sum of log-probabilities of each continuation, and whether greedy finds it."""
    width = max(len(ids) for ids, _ in batch)
    x = torch.zeros(len(batch), width, dtype=torch.long)
    mask = torch.zeros(len(batch), width, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
    logits = model.logits(x.to(model.device), mask.to(model.device))

    results = []
    for i, (ids, cont) in enumerate(batch):
        # the logit at position t predicts token t + 1
        start = len(ids) - len(cont) - 1
        rows = logits[i, start : start + len(cont)].float()
        target = torch.tensor(cont, device=rows.device)
        logprobs = F.log_softmax(rows, dim=-1)
        total = float(logprobs.gather(1, target[:, None]).sum())
        greedy = bool((rows.argmax(dim=-1) == target).all())
        results.append((total, greedy))
    return results


def evaluate_task(model, items: list[Item], max_batch_tokens: int) -> dict:
    """Correctness of every item, in the seeded order of ``load_task``."""
    flat = []  # (item index, choice index, ids, cont)
    for i, item in enumerate(items):
        for j, (ids, cont) in enumerate(requests_of(model, item)):
            flat.append((i, j, ids, cont))
    # longest first, so the first batch shows straight away whether memory holds
    flat.sort(key=lambda r: -len(r[2]))

    scores: dict[tuple[int, int], tuple[float, bool]] = {}
    pos = 0
    bar = tqdm(total=len(flat), desc=items[0].task, unit="seq", disable=not sys.stderr.isatty())
    while pos < len(flat):
        n = max(1, max_batch_tokens // len(flat[pos][2]))
        chunk = flat[pos : pos + n]
        for (i, j, _, _), result in zip(chunk, score_batch(model, [(r[2], r[3]) for r in chunk]),
                                        strict=True):
            scores[(i, j)] = result
        pos += len(chunk)
        bar.update(len(chunk))
    bar.close()

    acc, acc_norm = [], []
    for i, item in enumerate(items):
        if item.is_cloze:
            ok = scores[(i, 0)][1]
            acc.append(ok)
            acc_norm.append(ok)
            continue
        raw = [scores[(i, j)][0] for j in range(len(item.choices))]
        norm = [s / max(len(c.encode("utf-8")), 1) for s, c in zip(raw, item.choices, strict=True)]
        acc.append(max(range(len(raw)), key=raw.__getitem__) == item.gold)
        acc_norm.append(max(range(len(norm)), key=norm.__getitem__) == item.gold)
    return {"acc": acc, "acc_norm": acc_norm}


def summarize(flags: list[bool]) -> dict:
    n = len(flags)
    p = sum(flags) / n if n else 0.0
    return {"value": p, "stderr": math.sqrt(p * (1 - p) / n) if n else 0.0, "n": n}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="a checkpoint of this repository")
    src.add_argument("--hf", type=str, help="a causal LM on the Hugging Face Hub")
    p.add_argument("--name", type=str, default=None, help="results file name")
    p.add_argument("--tasks", type=str, default="all")
    p.add_argument("--limit", type=int, default=None, help="items per task, all by default")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    args = p.parse_args()

    model = OurModel(args.ckpt) if args.ckpt else HFModel(args.hf, args.dtype)
    default = getattr(model, "name", None) or args.hf or args.ckpt.stem
    name = (args.name or default).replace("/", "__")
    print(f"model   : {model.describe}")
    print(f"device  : {model.device}, prefix ids {model.prefix}")

    report = {
        "model": name,
        "describe": model.describe,
        "params": model.params,
        "date": f"{datetime.now():%Y-%m-%d %H:%M}",
        "device": torch.cuda.get_device_name() if model.device.type == "cuda" else "cpu",
        "limit": args.limit,
        "tasks": {},
    }
    t_all = time.perf_counter()
    for task in parse_tasks(args.tasks):
        items = load_task(task, args.limit)
        t0 = time.perf_counter()
        flags = evaluate_task(model, items, args.max_batch_tokens)
        report["tasks"][task] = {
            "language": LANGUAGE[task],
            "acc": summarize(flags["acc"]),
            "acc_norm": summarize(flags["acc_norm"]),
            f"acc_first{SUBSET}": summarize(flags["acc"][:SUBSET]),
            f"acc_norm_first{SUBSET}": summarize(flags["acc_norm"][:SUBSET]),
            "seconds": round(time.perf_counter() - t0, 1),
            # one character per item, in the seeded order, to recompute any subset
            "correct": "".join("1" if f else "0" for f in flags["acc"]),
            "correct_norm": "".join("1" if f else "0" for f in flags["acc_norm"]),
        }
        r = report["tasks"][task]
        print(
            f"  {task:<17} acc {r['acc']['value']:6.1%}  acc_norm {r['acc_norm']['value']:6.1%}"
            f"  (n={r['acc']['n']}, {r['seconds']:.0f}s)",
            flush=True,
        )
    report["seconds"] = round(time.perf_counter() - t_all, 1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results : {out}  ({report['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()

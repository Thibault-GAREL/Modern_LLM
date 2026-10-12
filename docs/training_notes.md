# Training notes: what worked, what did not, what to fix next

Kept as we go, on the actual runs of M.A.I (Modern AI), the bilingual model
trained with this library. Its first version was called `bilingual-100m` while
it trained, which is why the logs of M.A.I v1 carry that name. Everything here is something that
was measured or hit in practice, not anticipated from a paper.

Hardware: RunPod RTX 4090 24 GB, Ubuntu 22.04, torch 2.4.1+cu124, Python 3.11.

---

## What worked first time

**The architecture ported without a single change.** The config, the model, the
optimizer groups and the caches all ran on the pod exactly as locally. 349 of
350 tests pass there, the one skip being a torch 2.5 feature detected and
skipped rather than assumed.

**The frozen-corpus discipline paid off.** `docs/ablations.md` numbers were
reproducible because the corpus is hashed and cached. Without that the whole
ablation table would have been noise, as it was on the first sweep.

**RunLogger plus MLflow worked out of the box** once the file-store flag was
set, and the log is genuinely readable without a viewer.

**Checkpoint resume carries the scheduler and the scaler**, so a stopped pod
costs at most one checkpoint interval. Verified by a test, not by hope.

---

## What broke, and why

### `enable_gqa` does not exist before torch 2.5

`F.scaled_dot_product_attention(..., enable_gqa=False)` raises `TypeError` on
torch 2.4, so passing the keyword defensively is not enough: it has to be left
out of the call entirely. `pyproject.toml` claimed `torch>=2.4` while the code
required 2.5, and nothing caught it because the dev machine runs 2.5.1.

**Fix**: detect support once at import, splat the keyword conditionally.
**Lesson**: a version floor in `pyproject.toml` is a claim that has to be
tested, not a wish.

### Streaming data prep was 150x too slow

`datasets.load_dataset(streaming=True)` fetches document by document and
sustains about **10k tokens/s**, which is **167 hours** for 6B tokens.

| approach | throughput |
|---|---|
| streaming, document by document | 10k tok/s |
| direct parquet shard download | 25 MiB/s |
| tokenization, one document at a time | 692k tok/s |
| tokenization, batches of 1000 | **3.2M tok/s** |

The tokenizer was never the bottleneck. **Fix**: download shards in parallel,
batch-tokenize, delete each shard once consumed. Measured after the fix:
**1.4M tokens/s end to end**, so the same job in about an hour.

### The OOM killer, twice over

`pq.read_table(...).to_pylist()` loads a 2 GB parquet fully into Python, and
accumulating 714M tokens as small numpy arrays on top of that got the process
killed (`exit=137`). Worse, it tokenized an entire shard to keep 1.4M tokens
from it, doing 500x the useful work.

**Fix**: `iter_batches` plus an early stop at the requested budget. Memory now
scales with the budget, not with the shard.

### A leaked shard per language

When a language reached its budget, its already-downloaded next shard stayed on
disk forever. On a 20 GB volume that is 10% of the space per leak.

### Disk full killed the run at step 1999

The first real launch died 90 minutes in, while writing a checkpoint:
``PytorchStreamWriter failed writing file data/8``, which is what a full disk
looks like through torch. The 20 GB volume held 12 GB of corpus, a 2.1 GB
leftover from a download-speed test nobody cleaned up, and **two 1.33 GB
checkpoints per dated directory**.

Three separate causes, all mine:

- **`best_model.pt` carried the optimizer state it never needs.** AdamW holds
  two fp32 moments per parameter, so the optimizer is twice the model. A
  checkpoint meant purely for inference cost 1.33 GB instead of 444 MB.
- **`torch.save` wrote in place**, so a failure left a truncated file where a
  valid checkpoint used to be. Now it writes to `.tmp` and renames.
- **RunLogger opens a new dated directory at midnight**, so a run longer than a
  day accumulates checkpoint sets. Worth knowing before a multi-day run.

A fourth cause appeared five hours later, from the fix itself: **an atomic write
needs both files at once**. Writing `ckpt.pt.tmp` beside a 1.33 GB `ckpt.pt`
demands 2.66 GB free, not 1.33. The volume was down to 2.8 GB and the next
checkpoint would have failed again. Safety and space pull in opposite
directions here, and the resolution is to keep fewer checkpoints, not to give
up the atomic write.

The run resumed from the intact `best_model.pt` at step 1999 with its
scheduler and scaler restored, which is exactly what the resume test was
written for. Loss at that point was already **4.21 validation**, down from
10.37.

### Smaller things

- `torch.cuda.is_available()` returns `True` with `device_count() == 0` under
  `CUDA_VISIBLE_DEVICES=""`, so models landed on a phantom device and their
  checkpoints could not be reloaded. Now `pick_device()` checks both.
- MLflow 3.15 refuses the `./mlruns` file store without
  `MLFLOW_ALLOW_FILE_STORE=true`. Set in `run_logger.py`, but **the copy in the
  `thibault-logging` skill needs the same fix** or every new project hits it.
- An unseeded `torch.Generator` carries a fixed default seed, so two datasets
  drew identical windows.
- `pip install` on the pod fails against distutils-installed packages. A venv
  with `--system-site-packages` keeps the system torch and avoids it.

---

## Found while preparing the second run

A review of the code before paying for more pod hours. Three of these would
have quietly spoiled the resumed run, and none of them raised an error.

### The validation set was inside the training set

`write_split` listed each language's shards from the start for **both**
splits, so `val.bin` held the first 5M tokens of shard 0 and `train.bin` began
with the very same tokens. Every `loss/val` of the first run, 3.41 included,
was measured partly on text the model trained on.

**Fix**: `split_shards` reserves the last shard of each language for
validation, and `meta.json` records which. A test checks that the two splits
never fetch a common shard. M.A.I v1 has to be scored again on
the clean set (`python -m mllm.train ... --eval-only ckpt.pt`) before its loss
means anything.

### A resumed run replayed the batches of step 0

`TokenDataset` seeds its sampler from the global seed, and the checkpoint did
not store the sampler. So run-02, resumed at step 2000, drew exactly the
windows run-01 had drawn from step 0: 262M tokens seen twice. Resuming at step
10000 would have replayed 1.31B tokens, turning the rest of the run into a
second epoch over the first.

**Fix**: the sampler's state is saved in `ckpt.pt` and restored on resume. An
older checkpoint, like M.A.I v1, gets a seed derived from its step, so
it at least does not replay in order.

### Every evaluation drew new windows

The validation sampler kept advancing, so each point of the `loss/val` curve
was measured on 80 different windows. That sampling noise, a few hundredths of
a nat, is the same size as the progress between two evaluations.

**Fix**: `evaluate` rewinds the sampler first, and `eval_batches` (50 for the
bilingual config) sets how many windows it reads.

### The prompt started with a token the model had never seen

`scripts/generate.py` encoded prompts with the tokenizer's defaults, which
prepend `<s>`. The corpus was tokenized without special tokens and with `</s>`
between documents, so `<s>` never appeared once in 6B tokens.

| first token | mean loss, four EN and FR paragraphs |
|---|---|
| `<s>` (tokenizer default) | 2.775 |
| `</s>` (as in training) | **2.716** |
| none | 2.917 |

**Fix**: `mllm.inference` encodes every prompt `</s>` first, and the terminal
script, the Gradio page and the benchmarks all go through it.

### Smaller things

- **The test suite wrote into the real `outputs/` and `mlruns/`.** About 150
  `t-*` and `test-*` run folders were committed. `train` now takes a
  `project_root`, and the tests point it at a temporary folder.
- **`pyproject.toml` did not declare `mlflow`, `gradio` or `pyarrow`.** Added
  as the `train`, `demo` and `eval` extras.
- **HellaSwag's `ind` and Belebele's article names are not unique ids.** Two
  Belebele articles are both called "Introduction". `load_task` now refuses
  duplicate ids, which is what caught the second case.
- **On Windows, importing `pyarrow` after `torch` crashes** with an access
  violation (torch 2.5.1, transformers 5). `bench/eval_model.py` imports it
  first.

---

## To optimize before the next run

| Priority | Item | Why |
|---|---|---|
| High | **Set `HF_TOKEN` on the pod** | HF warns about unauthenticated requests and rate-limits them. Free speedup on the download phase |
| High | **Progress granularity in prep** | the bar only advances when a whole shard finishes, so it looks frozen for minutes. Update per batch |
| Medium | **Put the venv on the ephemeral disk** | it eats 2.7 GB of the 20 GB persistent volume for something regenerable in two minutes |
| Done | **Measure MFU during training** | logged at every step as `throughput/mfu`, about 27% on the first run |
| Done | **Log the cost** | `--cost-per-hour` logs `compute/cost_usd`, summed across resumes |
| Medium | **`torch.compile`** | left off for the first run to keep failures interpretable. Worth a measured comparison afterwards |
| Low | **Parallel tokenization across processes** | 3.2M tok/s is already far past the download rate, so this only matters if HF gets faster |
| Low | **Larger network volume** | 20 GB leaves 2.6 GB spare once data and checkpoints are in place, which forbids keeping several checkpoints |

---

## Cost so far

The pod actually billed **$0.74/h**, not the $0.34 quoted for a Community Cloud
RTX 4090. Every estimate given during the run was therefore roughly half the
real figure. **Check `runpodctl pod list` for `costPerHr` before quoting a cost**,
rather than trusting a price list.

| Phase | Time | Cost at $0.74/h |
|---|---|---|
| Setup, install, five bug fixes | ~2 h | ~$1.50 |
| Data preparation (6B tokens) | ~1 h | ~$0.75 |
| Training, 10k steps of 44k | ~6 h | ~$4.40 |
| **Total** | **~9 h** | **~$6.70** |

The two hours of debugging were not wasted: every one of those bugs would have
surfaced during the 30 hour run instead, and three of them would have killed it
outright.

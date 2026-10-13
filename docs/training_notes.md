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
| Done | **`torch.compile`** | measured on v2: 37% more tokens per second, see below |
| Low | **Parallel tokenization across processes** | 3.2M tok/s is already far past the download rate, so this only matters if HF gets faster |
| Low | **Larger network volume** | 20 GB leaves 2.6 GB spare once data and checkpoints are in place, which forbids keeping several checkpoints |

---

## Cost of M.A.I v1

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

---

## The M.A.I v2 run, one night on one pod

M.A.I v1 resumed from step 9999 to step 24000 on `configs/mai_v2.yaml`, the
learning rate held at 3e-4 until step 19200 then decayed linearly to 3e-5.
RunPod RTX 4090, Secure Cloud, torch 2.8.0, Python 3.12, from 22:30 to 07:25.

| Phase | Clock | What happened |
|---|---|---|
| Clone, install, tests | 22:31 | 361 of 363 green, see "torch 2.8" below |
| Corpus, 6B tokens | 22:37 to 23:10 | **32 min** at 3.1M tokens/s with 8 download workers, half the first run's hour |
| Reference benchmarks | 22:38 to 22:50 | GPT-2, SmolLM2-135M and CroissantLLM on every item, on the GPU the corpus job left idle |
| v1 on the clean validation set | 23:11 | `loss/val` **3.332**, the baseline every v2 point is compared to |
| Smoke test | 23:11 to 23:14 | 50.9k tokens/s eager, **69.9k compiled**, identical losses |
| Training, 14000 steps | 23:15 to 06:40 | 7.4 h, 68.7k tokens/s, 37% MFU |
| Benchmarks of v1 and v2, samples | 06:40 to 06:46 | every item, 2 min per model |

| step | `loss/val` |
|---|---|
| 9999 (v1) | 3.332 |
| 14499 | 3.251 |
| 17999 | 3.216 |
| 19200, decay starts | |
| 21999 | 3.127 |
| 23999 (v2) | **3.079** |

The plateau bought 0.13 nats in 9200 steps (3.332 to 3.207 at step 18999). The
decay bought another 0.13 in the last 4800, the same gain in half the steps,
which is why WSD runs keep the decay for the end.

### `torch.compile` was worth 37%

Left off during v1 to keep failures readable. On torch 2.8 and Linux it took
the step from 2.57 s to 1.88 s and the MFU from 27% to 38%, with losses equal
to the third decimal over the same 30 steps. It cut the run from 10 h to 7.4 h,
so about $2.40 at this price.

One graph break remains: the rotary embedding's `_ensure_cache` calls
`positions.max().item()`, which leaves the compiled graph at every layer and
hits the recompile limit. Building the cos and sin tables once for
`max_seq_len` at init would remove it, and probably some more step time.

### The gradient norm climbs at the very end

743 of the last 826 steps have a gradient norm above 2, against a median of
0.58 during the plateau, and the last step reads 7.9. Clipping at 1.0 holds it, and
`loss/val` keeps falling to the last evaluation. This rise near the end of a
decayed schedule is a known effect of weight decay acting on weights whose
scale the normalization ignores, while the learning rate goes to zero. For v3,
decaying the weight decay along with the learning rate is the first thing to
try.

### torch 2.8 retired two sentinel tests

The test on the RMSNorm upcast and the test on the `enable_gqa` regression
fail on torch 2.8, as designed: they pin bugs of torch 2.5.1, and both are
fixed. The fused RMSNorm and `enable_gqa=True` can become fast paths once the
tests are made version aware.

### Smaller things

- **The pod was Secure Cloud at $0.89/h**, picked in the UI instead of the
  $0.34/h Community one. Same card, 2.6x the price. `runpodctl pod list`
  shows `costPerHr`, check it before launching.
- **The run crossed midnight**, so `RunLogger` opened a second dated model
  folder. The final weights were in `..._date-2026-10-13`, the one before
  midnight held stale checkpoints.
- **A pod exchanges files with a home connection at about 1 MB/s.** Sending
  the 1.33 GB resumable checkpoint took 12 minutes, bringing it back about 20.
  The weights alone (444 MB) are enough unless a v3 resumes from them.
- **The local hooks refuse permission changes and deletions in a command.**
  Scripts are sent with scp and run as `bash script.sh`.

## Cost of M.A.I v2

| | Time | Cost |
|---|---|---|
| Training compute, counted by the run | 7.42 h | **$6.61**, $3.60 per billion tokens |
| Pod uptime, setup to deletion | ~8.9 h | **$7.89** billed at $0.89/h |
| **M.A.I since its first step**, training only | **14.60 h** | **$11.93** |

The same night on a Community Cloud 4090 would have cost about $3.10.

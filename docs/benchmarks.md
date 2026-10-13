# Benchmarks: what M.A.I can do, against models of its size and a frontier model

Nine tasks, English and French side by side, scored the same way for every
base model and on the same items for every model. Raw results, item by item, are
in `outputs/results/evals/*.json`. This page is regenerated with
`python bench/eval_report.py`.

---

## Results

### Every item, base models

Accuracy in %, `acc_norm` for multiple choice and greedy `acc` for LAMBADA, on
every item of each task (10,042 for HellaSwag EN, 900 for Belebele). One
standard error is under 0.5 point on HellaSwag and about 1.5 on Belebele.

| Model | Params | Tokens | HellaSwag EN / FR | ARC-E | ARC-C EN / FR | Belebele EN / FR | LAMBADA EN / FR | Mean | EN | FR |
|---|---|---|---|---|---|---|---|---|---|---|
| *chance* | | | 25 / 25 | 25 | 25 / 25 | 25 / 25 | ~0 / ~0 | | | |
| GPT-2 | 124M | WebText, English | 31.5 / 28.8 | 40.5 | 22.4 / 22.2 | 28.7 / 28.8 | 33.7 / 8.4 | 27.2 | 31.3 | 22.0 |
| SmolLM2 | 135M | 2T, English | 44.3 / 29.3 | 58.3 | 29.0 / 22.7 | 33.2 / 31.1 | 43.6 / 16.1 | 34.2 | 41.7 | 24.8 |
| CroissantLLM | 1.3B | 3T, EN and FR | 54.1 / 51.3 | 52.2 | 27.7 / 29.9 | 35.8 / 37.2 | 38.3 / 33.1 | **39.9** | 41.6 | **37.9** |
| **M.A.I v1** | 97M | 1.31B, EN and FR | 28.2 / 28.2 | 38.3 | 21.4 / 22.4 | 28.3 / 27.3 | 11.7 / 13.2 | 24.3 | 25.6 | 22.8 |
| **M.A.I v2** | 97M | 3.15B, EN and FR | 29.0 / 29.3 | 39.3 | 22.4 / 24.0 | 28.3 / 28.6 | 16.7 / 15.9 | **25.9** | 27.1 | 24.5 |

**Reading it.**

- **v2 against v1**: better on 8 tasks of 9, flat on Belebele EN. The largest
  gain is LAMBADA EN, +5 points, the task that needs the whole passage.
- **v2 against GPT-2**: ahead in French (24.5 against 22.0 on average, and
  15.9 against 8.4 on LAMBADA FR), behind in English. GPT-2 saw only English,
  M.A.I spends 30% of its tokens on French.
- **v2 against SmolLM2**: the same size trained on 600 times more tokens is 8
  points ahead on average. At 100M parameters, data is the bottleneck, not the
  architecture.
- **CroissantLLM**, with the same tokenizer and the same two languages, shows
  what 12x the parameters and 1000x the tokens buy. It is also the only model
  that reads the French passages of Belebele noticeably above chance.

### The same 100 items, with a frontier model

Claude Opus 5.5 answers lettered questions in text (see below), so it was run on
the first 100 items of each task only. Every model of the table above is scored
on those same 100 items here. One standard error is 4 to 5 points.

| Model | HellaSwag EN | HellaSwag FR | ARC-E EN | ARC-C EN | ARC-C FR | Belebele EN | Belebele FR | LAMBADA EN | LAMBADA FR | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| *chance* | 25% | 25% | 25% | 25% | 25% | 25% | 25% | ~0% | ~0% | |
| GPT-2 | 21% | 24% | 49% | 16% | 25% | 19% | 27% | 33% | 10% | 25% |
| SmolLM2-135M | 41% | 27% | 57% | 23% | 20% | 34% | 32% | 42% | 17% | 33% |
| CroissantLLM 1.3B | 52% | 56% | 50% | 22% | 32% | 35% | 35% | 37% | 30% | 39% |
| M.A.I v1 | 27% | 32% | 35% | 17% | 22% | 20% | 28% | 13% | 11% | 23% |
| M.A.I v2 | 27% | 36% | 43% | 19% | 26% | 26% | 29% | 18% | 14% | 26% |
| **Claude Opus 5.5**, zero-shot | 98% | 95% | 100% | 99% | 100% | 97% | 96% | 97% | 83% | **96%** |

The gap to the frontier model is the whole distance between predicting text
and understanding it.

### Validation of the scorer on GPT-2

Before trusting any number above, the log-likelihood scorer was checked on a
model whose scores are published.

| GPT-2 124M, every item | measured here | commonly reported (lm-evaluation-harness) |
|---|---|---|
| HellaSwag, `acc_norm` | 31.5% | 31.1% |
| ARC-Easy, `acc` | 43.9% | 43.8% |
| LAMBADA EN, `acc` | 33.7% | 32.6% |

All three within one point. The 100 item head gives GPT-2 only 21% on
HellaSwag, a reminder that 100 items rank models apart only when they differ by
10 points or more.

---

## How each model is scored

### Base models, by log-likelihood (`bench/eval_model.py`)

A base model never picks a letter. Each choice is appended to the context and
the log-probabilities of its tokens are summed. The best scored choice is the
answer.

| Metric | Definition | Used for |
|---|---|---|
| `acc` | argmax of the summed log-probability | LAMBADA |
| `acc_norm` | the same sum divided by the answer's length in bytes | HellaSwag, ARC, Belebele |
| greedy match | LAMBADA counts a hit when greedy decoding produces exactly the last word's tokens | LAMBADA |

**The prompt starts with the token each model was trained to see first.** Our
model gets `</s>`, the end of document id that preceded every document in its
corpus. A Hub model keeps its tokenizer's own special tokens (`<s>` for
CroissantLLM), or its end of text token when the tokenizer adds nothing (GPT-2,
SmolLM2).

**Belebele is scored on the answer's text, not its letter.** lm-evaluation-harness
asks for the letter, which no 100M model can do, so every small model lands at
exactly 25% and the task measures nothing.

### Chat models, by their written answer (`bench/eval_llm.py`)

| Step | What happens | What it rules out |
|---|---|---|
| `export` | writes 900 lettered questions to `outputs/results/evals/llm/questions.jsonl`, and the keys to `.cache/evals/keys/keys.jsonl`, printing counts only | the answerer seeing a key |
| answering | the answerer reads the questions file and nothing else: no dataset, no web, no code, no other model | looking answers up |
| `score` | hashes the answers file first (sha256 `41d4fcb8b48876ee...` for Claude Opus 5.5), then compares | editing answers after the score |

The answers are committed next to the questions, so the score can be checked
line by line.

**An upper bound, not a clean measurement.** These datasets have been public
since 2018 to 2023, so a model trained on the web has likely seen some of them.
LAMBADA FR is a machine translation, which explains the 83%: several passages
end on a word no reader could predict from the translated text.

---

## Reproduce it

Every row of the first table took 2 to 5 minutes on an RTX 4090, 11 minutes
for the three reference models together, run while the corpus was being
prepared on the same pod.

```bash
python bench/eval_model.py --ckpt outputs/models/mai-v2/mai-v2.pt     # saved as mai-v2
python bench/eval_model.py --ckpt outputs/models/mai-v1/mai-v1.pt     # saved as mai-v1
python bench/eval_model.py --hf gpt2
python bench/eval_model.py --hf HuggingFaceTB/SmolLM2-135M
python bench/eval_model.py --hf croissantllm/CroissantLLMBase
python bench/eval_report.py --out docs/benchmark_table.md
```

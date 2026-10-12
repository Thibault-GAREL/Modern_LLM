# Benchmarks: what the bilingual model can do, against a frontier model

Nine tasks, English and French side by side, scored the same way for every
base model and on the same items for every model. Raw results, item by item, are
in `outputs/results/evals/*.json`. This page is regenerated with
`python bench/eval_report.py`.

---

## Results so far

The same first 100 items of each task, in the seeded order of
`bench/eval_tasks.py`. One standard error is about 4 to 5 points at this size.

| Model | HellaSwag EN | HellaSwag FR | ARC-E EN | ARC-C EN | ARC-C FR | Belebele EN | Belebele FR | LAMBADA EN | LAMBADA FR | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| *chance* | 25% | 25% | 25% | 25% | 25% | 25% | 25% | ~0% | ~0% | |
| Modern LLM 111M, step 10k (1.31B tokens) | 27% | 32% | 35% | 17% | 22% | 20% | 28% | 13% | 11% | **23%** |
| Claude Opus 5.5, zero-shot | 98% | 95% | 100% | 99% | 100% | 97% | 96% | 97% | 83% | **96%** |

**Reading it.** The 111M model is at chance on everything that needs
reasoning (ARC-Challenge, Belebele) and slightly above it where surface
statistics help (ARC-Easy, HellaSwag FR, LAMBADA). That is the expected place
for a 100M model at 1.3B tokens: GPT-2 small, 124M parameters trained on
WebText, reaches 29% on HellaSwag and 34% on LAMBADA (measured below). The gap to
the frontier model is the whole distance between predicting text and
understanding it.

### Validation of the scorer on GPT-2

Before trusting any number above, the log-likelihood scorer was checked on a
model whose scores are published.

| GPT-2 124M, 1000 items | measured here | commonly reported (lm-evaluation-harness, full set) |
|---|---|---|
| HellaSwag, `acc_norm` | 29.3% | 31.1% |
| ARC-Easy, `acc` | 45.4% | 43.8% |
| LAMBADA EN, `acc` | 34.0% | 32.6% |

Each within about two standard errors (1.5 points at 1000 items, and the
reference uses the full set, not the same 1000 items). The same 100
item head gives GPT-2 only 21% on HellaSwag, a reminder that 100 items rank
models apart only when they differ by 10 points or more.

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

## Still to measure, on the pod

| Run | Why | Cost |
|---|---|---|
| Modern LLM at step 10k, every item | tight error bars, about 1 point instead of 4 | a few minutes |
| GPT-2 124M, SmolLM2-135M, every item | a 2019 and a 2024 model of the same size, English only | ~10 min |
| CroissantLLMBase 1.3B, every item | **the same tokenizer and the same two languages**, 12x the parameters and 2300x the tokens | ~30 min |
| Modern LLM after the resumed run | what the remaining tokens and the WSD decay buy | a few minutes |

```bash
python bench/eval_model.py --ckpt outputs/models/<run>/final_model.pt --name modern-llm-step46k
python bench/eval_model.py --hf gpt2
python bench/eval_model.py --hf HuggingFaceTB/SmolLM2-135M
python bench/eval_model.py --hf croissantllm/CroissantLLMBase
python bench/eval_report.py --out docs/benchmark_table.md
```

---
license: mit
language:
  - en
  - fr
library_name: pytorch
pipeline_tag: text-generation
tags:
  - base-model
  - bilingual
  - from-scratch
  - modern-llm
datasets:
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceFW/fineweb-2
---

# 🧬 M.A.I v2, Modern AI

A 97M parameter base model trained from scratch on English and French, the
second version of the model trained with
[Modern LLM](https://github.com/Thibault-GAREL/Modern_LLM), a library that
rebuilds every deviation from the 2017 Transformer behind a config flag and
measures which ones are worth it.

**A base model.** It continues a text, it does not follow instructions or
answer questions reliably. Try it in the
[Space](https://huggingface.co/spaces/Thibault-GAREL/mai).

## Versions

| Version | Steps | Tokens | `loss/val` | Benchmark mean |
|---|---|---|---|---|
| M.A.I v1 | 10k | 1.31B | 3.332 | 24.3% |
| **M.A.I v2** | 24k | 3.15B | **3.079** | **25.9%** |

v2 is v1 trained further on the same plan, with the learning rate decayed over
the last 4800 steps.

## Benchmarks

Accuracy in %, log-likelihood scoring (`acc_norm`, greedy `acc` for LAMBADA),
on every item of each task.

| Model | Params | Tokens | HellaSwag EN / FR | ARC-E | ARC-C EN / FR | Belebele EN / FR | LAMBADA EN / FR | Mean |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | 124M | WebText | 31.5 / 28.8 | 40.5 | 22.4 / 22.2 | 28.7 / 28.8 | 33.7 / 8.4 | 27.2 |
| SmolLM2 | 135M | 2T | 44.3 / 29.3 | 58.3 | 29.0 / 22.7 | 33.2 / 31.1 | 43.6 / 16.1 | 34.2 |
| CroissantLLM | 1.3B | 3T | 54.1 / 51.3 | 52.2 | 27.7 / 29.9 | 35.8 / 37.2 | 38.3 / 33.1 | 39.9 |
| **M.A.I v2** | 97M | 3.15B | 29.0 / 29.3 | 39.3 | 22.4 / 24.0 | 28.3 / 28.6 | 16.7 / 15.9 | **25.9** |

Ahead of GPT-2 in French, behind it in English, and 8 points behind a model of
the same size trained on 600 times more tokens. Protocol and every item in
[docs/benchmarks.md](https://github.com/Thibault-GAREL/Modern_LLM/blob/main/docs/benchmarks.md).

## Architecture

| | |
|---|---|
| Layers | 12, `d_model` 768 |
| Attention | MQA, 12 query heads and 1 KV head, RoPE (theta 10000) |
| Block | RMSNorm pre-norm, SwiGLU, no bias, tied embeddings |
| Training heads | Multi-Token Prediction, depth 2, dropped at inference (111M trained, 97M deployed) |
| Context | 2048 tokens |
| Tokenizer | [CroissantLLM](https://huggingface.co/croissantllm/CroissantLLMBase), 32k tokens |

This is `configs/best.yaml` of the library, the combination that won its
ablation sweep on both the held-out loss and the deployed size.

## Training

| | |
|---|---|
| Data | 70% FineWeb-Edu `sample-10BT` (English), 30% FineWeb2 `fra_Latn` (French) |
| Schedule | AdamW (0.9, 0.95), weight decay 0.1, WSD at 3e-4, decayed to 3e-5 over steps 19200 to 24000 |
| Batch | 131k tokens per step, bf16 |
| Hardware | one RTX 4090, 14.6 h in total, `torch.compile` for v2 at 68.7k tokens/s |
| Cost | **$11.93** of GPU time since the first step |

## Use

```bash
pip install "modern-llm[demo] @ git+https://github.com/Thibault-GAREL/Modern_LLM.git"
```

```python
import torch
from huggingface_hub import hf_hub_download
from mllm.generate import SamplingConfig, generate
from mllm.inference import Tokenizer, load_model

loaded = load_model(hf_hub_download("Thibault-GAREL/mai-v2", "mai-v2.pt"))
tok = Tokenizer()
ids = torch.tensor([tok.encode_prompt("La capitale de la France est")], device=loaded.device)
cfg = SamplingConfig(temperature=0.9, top_p=0.92, repetition_penalty=1.15)
print(tok.decode(generate(loaded.model, ids, 60, cfg, eos_id=tok.eos_id)[0].tolist()))
```

**Start prompts with `</s>`**, which `encode_prompt` does: it is the token that
preceded every document in training. Keep the repetition penalty around 1.15,
without it the model loops.

## Limitations

A 97M model on 3B tokens writes fluent looking English and French and gets
facts wrong, often with confidence. It has seen web text only, without any
filtering for toxicity beyond what FineWeb applies. Not for any use where a
wrong or harmful sentence matters.

Trained by Thibault GAREL, [GitHub](https://github.com/Thibault-GAREL).

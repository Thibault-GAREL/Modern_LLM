# Taxonomy: what changed, where, and what for

Every modern component replaces a specific piece of the 2017 decoder block. This
document sorts them by **slot** (which part of the original architecture they sit
in) and by **function** (the constraint they were invented to solve).

Reading it the wrong way round is the classic mistake. These techniques are not a
list of upgrades to apply in order, they are answers to constraints. If you do not
have the constraint, the answer is pure complexity.

---

## Function legend

| Tag | Function | The question it answers |
|---|---|---|
| 💾 | **Memory** | how much VRAM does one token of context cost |
| ⚡ | **Compute** | how many FLOPs, or how fast at fixed FLOPs |
| 🎯 | **Quality** | better loss at an equal parameter or FLOP budget |
| 🛡️ | **Stability** | does the training run survive, and without babysitting |
| 📏 | **Long context** | does it still work past the training length |
| 🔧 | **Transfer** | can the settings found on a small model be reused on a big one |

---

## The slots at a glance

| Slot | 2017 | Today | Dominant function |
|---|---|---|---|
| 0. Tokenizer | BPE, 32k to 37k | BBPE, 128k to 256k | ⚡ 🎯 |
| 1. Input embedding | lookup, scaled by `sqrt(d)` | lookup, often tied to the head | 💾 |
| 2. Positions | sinusoidal, added once at the input | RoPE at every layer, plus rescaling | 📏 🎯 |
| 3. Attention heads | MHA, one KV pair per query head | GQA or MLA | 💾 ⚡ |
| 4. Attention scores | plain scaled dot product softmax | QK-norm, softcap, windows, sinks | 🛡️ ⚡ 📏 |
| 5. Add and Norm | post-norm LayerNorm | pre-norm RMSNorm | 🛡️ |
| 6. Feed-forward | ReLU, width `4d`, two matrices | SwiGLU, or sparse MoE | 🎯 |
| 7. Head and objective | linear, softmax, next token | tied head, z-loss, optional MTP | 🛡️ 🎯 |
| 8. Optimization | Adam, inverse square root schedule | AdamW, cosine or WSD, muP | 🛡️ 🔧 |
| 9. Inference | not a concern in the paper | KV cache, speculative decoding | ⚡ 💾 |
| 10. Topology | encoder plus decoder, cross-attention | decoder only | ⚡ |

---

## Slot 0. Tokenizer and vocabulary

*Before the model. The 2017 paper used byte-pair encoding with a shared source and
target vocabulary of 37000 for English to German.*

| Change | Function | Why |
|---|---|---|
| Byte-level BPE | 🛡️ | no out-of-vocabulary token is possible, any byte sequence encodes |
| Vocabulary 32k to 256k | ⚡ 🎯 | a bigger vocabulary means fewer tokens for the same text, so fewer forward passes for the same content. Multilingual and code coverage improve a lot |
| Digit splitting, code-aware pretokenization | 🎯 | arithmetic and indentation stop depending on how the merger happened to group characters |

**The trade-off nobody mentions.** A 256k vocabulary makes the embedding table and
the output softmax huge. On a small model those two can dominate the parameter
count, which is exactly why `tie_embeddings` matters more the smaller you go.

**Not in this repo.** The tokenizer is upstream of the architecture, so `mllm` takes
token ids as input and stops there.

---

## Slot 1. Input embedding

| Change | Function | Config flag |
|---|---|---|
| Tied embeddings | 💾 | `model.tie_embeddings` |
| Dropping the `sqrt(d_model)` input scaling | 🛡️ | implicit, pre-norm makes it redundant |

Tying the input table with the output head saves `vocab_size × d_model` parameters,
which on a 150M model with a 32k vocabulary is around a fifth of the whole model.
Large models often untie them again, because at that scale the saving is marginal
and the two matrices want to learn different things.

---

## Slot 2. Positional information

*2017: a fixed sinusoidal signal added to the embedding once, at the input.*

The core problem with the original scheme is that position is injected as an
**absolute** value into the **residual stream**, where it competes with content and
gets progressively diluted layer after layer.

| Change | Function | Config flag | What it does differently |
|---|---|---|---|
| **RoPE** | 🎯 📏 | `position.kind: rope` | rotates q and k at **every layer**, so the dot product depends only on `m - n`. Position becomes relative, and is never diluted |
| **ALiBi** | 📏 | `position.kind: alibi` | a fixed linear penalty on distance, added to the scores. No parameters, extrapolates naturally |
| **NoPE** | 🎯 | `position.kind: nope` | nothing at all. The causal mask alone breaks the permutation symmetry, and the model infers position |
| **Position Interpolation** | 📏 | `scaling.kind: linear` | squeezes positions into the trained range. Simple, costs some short-range resolution |
| **NTK-aware** | 📏 | `scaling.kind: ntk-aware` | rescales `theta` instead of positions, preserving high frequencies |
| **YaRN** | 📏 | `scaling.kind: yarn` | interpolates per frequency band, plus a temperature on the attention scale. The best quality per unit of fine-tuning |
| **theta 10k to 500k** | 📏 | `position.rope_theta` | slows the lowest frequencies so they still resolve long distances |

**The most useful thing to understand here.** RoPE alone does not extend context. It
generalizes poorly past its training length because the low frequency bands were
never seen through a full period. Everything in the bottom half of that table exists
to patch that one problem after the fact.

---

## Slot 3. Attention head structure

*2017: `h = 8` heads, each with its own K and V projection.*

This slot changed for **one reason only, and it is not quality**. During generation
every past key and value must be kept. The cost per token is

```
GQA bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_size
```

At long context, with a large batch, that cache becomes larger than the weights and
the decode step becomes memory-bound. All the variants below trade a little quality
for a smaller cache.

| Change | Function | Config flag | KV heads | Quality cost |
|---|---|---|---|---|
| **MHA** | reference | `attention.kind: mha` | `n_heads` | none, it is the baseline |
| **MQA** | 💾 ⚡ | `attention.kind: mqa` | 1 | measurable, and training gets less stable |
| **GQA** | 💾 ⚡ | `attention.kind: gqa` | `n_heads / g` | close to none at `g = 4`, which is why everyone uses it |
| **MLA** | 💾 | `attention.kind: mla` | none, a latent vector instead | reported as better than MHA, at the price of roughly three times the code |

**Why a smaller cache also means faster.** Decoding one token reads the entire cache
and does very little arithmetic with it, so the step is limited by memory bandwidth,
not by FLOPs. Dividing the cache by four roughly divides the decode time by four.
This is the clearest case in the whole list where a memory optimization buys speed.

**The MLA catch.** RoPE does not commute with the low-rank compression, so MLA needs
a separate rotary key shared across heads, and the up-projections must be folded
into the query and output matrices at inference. That is the "three times the code".

---

## Slot 4. Attention score computation

*2017: `softmax(QKᵀ / sqrt(d_k)) V`, full causal mask, nothing else.*

| Change | Function | Config flag | The constraint it answers |
|---|---|---|---|
| **QK-Norm** | 🛡️ | `attention.qk_norm` | attention logits drift upward during training until the softmax saturates and gradients die. Normalizing q and k bounds them by construction |
| **Logit softcap** | 🛡️ | `attention.logit_softcap` | same problem, solved by `c · tanh(scores / c)`. Incompatible with fused attention kernels, which is why Gemma 3 dropped it in favour of QK-norm |
| **Sliding window** | ⚡ 📏 | `attention.sliding_window` | attention is quadratic in length. A window of `w` makes it linear |
| **Local and global alternation** | ⚡ 📏 | `attention.global_every` | pure local attention cannot move information across the whole sequence. One global layer every five restores that at a fraction of the cost |
| **Attention sinks** | 🛡️ 📏 | `attention.attn_sinks` | softmax must sum to one, so when nothing is relevant the model dumps attention on the first tokens. Evict those and the model collapses. Keep them, or add a learned sink logit in the denominator |
| **FlashAttention** | ⚡ 💾 | backend, `flash-attn` extra | **not an architecture change**. Identical mathematics, tiled to avoid ever writing the `n × n` score matrix to memory |

**The two things worth remembering here.** First, the memory win of a sliding window
comes from the **cache** becoming a ring buffer of size `w`, not from the mask itself.
A mask alone saves compute and nothing else. Second, FlashAttention belongs in a
different category from everything else in this document, because it changes no
result, only the memory traffic used to get it.

---

## Slot 5. Add and Norm

*2017: `LayerNorm(x + Sublayer(x))`, that is post-norm.*

This slot is the reason the original Transformer needed a 4000 step warmup. With
post-norm, a normalization sits **on the residual path itself**, so the gradient
reaching layer one has been rescaled once per layer above it.

| Change | Function | Config flag | Effect |
|---|---|---|---|
| **Pre-norm** | 🛡️ | `norm.placement: pre` | `x + Sublayer(norm(x))`. The residual path becomes a clean identity from input to output, gradients reach the bottom layers unattenuated, and deep models train without warmup tricks |
| **RMSNorm** | ⚡ 🛡️ | `norm.kind: rmsnorm` | drops the mean subtraction and the bias. Roughly the same quality, one reduction instead of two, fewer parameters |
| **Sandwich norm** | 🛡️ | `norm.placement: sandwich` | keeps the pre-norm identity path and also bounds each sub-block output before it is added |
| **Scaled residual init** | 🛡️ | `init.scaled_residual` | pre-norm has a side effect, the residual stream is a sum of `2 · n_layers` branches so its variance grows with depth. Dividing the output projections by `sqrt(2 · n_layers)` cancels that at init |
| **DyT** | ⚡ | `norm.kind: dyt` | `gamma · tanh(alpha · x) + beta`. No reduction over the feature dimension at all, so nothing to synchronize. Still rare in released models |

**The one that is not optional.** Every statistic in this slot has to be computed in
fp32 and cast back. Verified in this repo: `torch.nn.functional.rms_norm` computes
in the input dtype, and in fp16 it matches an all-fp16 computation bit for bit
rather than the fp32 one. See `tests/test_norm.py::test_torch_rms_norm_does_not_upcast`.

---

## Slot 6. Feed-forward

*2017: `max(0, x W₁ + b₁) W₂ + b₂`, with `d_ff = 4 · d_model`. Two thirds of the
parameters of the block live here.*

Two independent changes happened, and they answer different questions.

### 6a. The activation, a quality change at constant size

| Change | Function | Config flag |
|---|---|---|
| **SwiGLU, GeGLU, ReGLU** | 🎯 | `ffn.kind` |

A gated unit computes `(x W_gate ⊙ σ(x W_up)) W_down`, so three matrices instead of
two. To keep the parameter budget identical, the width drops from `4d` to `8/3 · d`.
Better loss at the same size, and the paper that introduced it says outright that it
offers no explanation for why.

### 6b. Mixture of Experts, a capacity change at constant compute

| Change | Function | Config flag | Purpose |
|---|---|---|---|
| **Sparse MoE** | 🎯 at fixed ⚡ | `moe.enabled` | replace one FFN by `N`, activate `k`. Parameters grow, FLOPs per token do not |
| **Fine-grained experts** | 🎯 | `moe.n_experts` high, `d_ff_expert` small | many small experts give far more usable combinations than a few big ones |
| **Shared experts** | 🎯 | `moe.n_shared_experts` | always active. Common knowledge lives there instead of being duplicated in every routed expert |
| **Auxiliary loss** | 🛡️ | `moe.balance: aux_loss` | without it the router collapses onto a few experts. But it is a second objective fighting the real one |
| **Aux-loss-free balancing** | 🛡️ 🎯 | `moe.balance: aux_loss_free` | a per-expert bias added **only for selection**, updated outside the gradient. Balances the load without polluting the loss |

**What MoE actually costs.** It buys quality per FLOP, and pays in memory, because
every expert must be resident even though only a few run. It also pays in
engineering, because routing is a communication problem as soon as experts live on
different devices. Below a few billion parameters, on a single GPU, it buys nothing.

---

## Slot 7. Output head and training objective

| Change | Function | Config flag | Purpose |
|---|---|---|---|
| **Tied head** | 💾 | `model.tie_embeddings` | reuses the embedding matrix as the output projection |
| **Output z-loss** | 🛡️ | `train.z_loss_coef` | penalizes `logsumexp(logits)²`, keeping logits in a range where bf16 does not lose them |
| **Multi-Token Prediction** | 🎯 ⚡ | `model.mtp_depth` | predicts the next `k` tokens through extra heads. Denser training signal, and the heads double as a draft model for speculative decoding |

---

## Slot 8. Optimization

*Not part of the architecture, but the 2017 recipe changed completely.*

| Change | Function | Config flag | Versus 2017 |
|---|---|---|---|
| AdamW with `betas = (0.9, 0.95)` | 🛡️ | `train.betas` | 2017 used `beta2 = 0.98` and plain L2 |
| Weight decay excluding norms, biases, embeddings | 🎯 | `train.weight_decay` | decaying a norm gain shrinks the signal it is supposed to rescale |
| Warmup plus cosine | 🛡️ | `train.schedule: cosine` | replaces the inverse square root schedule |
| **WSD** | 🔧 | `train.schedule: wsd` | a stable plateau then a short decay, so the token budget is not locked in advance and branches can fork mid-run |
| Gradient clipping at 1.0 | 🛡️ | `train.max_grad_norm` | absent from the paper |
| bf16 with fp32 master weights | ⚡ 🛡️ | `train.precision` | fp32 everywhere in 2017 |
| **muP** | 🔧 | `mup.enabled` | tune the learning rate on a small model and reuse it at scale, instead of re-searching at every width |

---

## Slot 9. Inference machinery

*Absent from the 2017 paper, which never discusses autoregressive decoding cost.*

| Change | Function | Where | Purpose |
|---|---|---|---|
| **KV cache** | ⚡ | `cache.py` | without it, generating token `n` recomputes the whole prefix, so a sequence costs `O(n²)` forward passes instead of `O(n)` |
| **Ring buffer cache** | 💾 | `cache.py` | on sliding window layers, only the last `w` entries can ever be read, so only those are stored |
| **Latent cache** | 💾 | `cache.py` | MLA stores one compressed vector per token per layer instead of all KV heads |
| **Speculative decoding** | ⚡ | `generate.py` | a small model drafts `gamma` tokens, the big one verifies them in a single pass. Correct rejection sampling makes the output distribution **identical** to the target model, so it is free speed, not a quality trade |

---

## Slot 10. Global topology

| Change | Function | Why |
|---|---|---|
| Encoder-decoder to decoder-only | ⚡ 🎯 | the 2017 model was built for translation, where a full source sentence is available. For language modelling there is no separate source, so the encoder and the cross-attention are dead weight. Every parameter now serves generation |

---

## Cross-view: pick by constraint, not by slot

If you are choosing what to put in a model, this is the table to read.

| Your constraint | What to add | What NOT to add |
|---|---|---|
| Nothing in particular | RoPE, RMSNorm pre-norm, SwiGLU, GQA | everything else |
| Training keeps diverging | QK-norm, z-loss, check the fp32 policy first | MoE, which makes it worse |
| Cache does not fit | GQA first, then MLA if that is not enough | MoE, which adds memory instead |
| Context longer than training | YaRN, then sliding window with a ring buffer | MLA, which is orthogonal to this |
| Many parameters at fixed FLOPs | fine-grained MoE, shared experts, aux-loss-free | only if the serving infrastructure exists |
| Generation is too slow | KV cache first, then speculative decoding | architecture changes, which are the wrong lever |
| Hyperparameters cost too much to search | muP, with an actual coordinate check | anything else, muP is the only answer here |

---

## What is deliberately absent

These matter in production and are out of scope here, because they are systems work
rather than architecture.

- Tensor, pipeline and expert parallelism
- Quantization, both post-training and quantization-aware
- Custom kernels beyond what PyTorch ships
- Post-training entirely, so no supervised fine-tuning, no RLHF, no reasoning traces
- Anything about closed models, whose architectures are simply not published

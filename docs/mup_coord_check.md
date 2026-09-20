# muP coordinate check

muP (Yang et al., 2022, arXiv 2203.03466) makes the optimal learning rate
independent of model width, so it can be tuned once on a small model and reused
at scale. A muP implementation is either exactly right or silently useless, and
unit tests on individual formulas do not catch the difference. The coordinate
check is the only test that does.

## The idea

Train the same model at several widths and record the **coordinate size** of
every layer, meaning the average absolute value of its activations, at every
step. Under muP those curves lie on top of each other. Under standard
parametrization they spread apart as width grows, which is exactly why a
learning rate tuned at one width is wrong at another.

## Running it

```bash
python bench/coord_check.py                    # defaults, 30 steps
python bench/coord_check.py --steps 50 --lr 1e-3
```

## Result

![muP coordinate check](assets/mup_coord_check.png)

```
width spread after 30 steps (1.0 = perfectly width invariant)
  standard parametrization :  31.63x
  muP                      :   1.03x
  muP is 30.7x flatter across widths
```

The spread reported is the worst ratio, across layers, between the largest and
smallest coordinate size over the four widths.

## The four changes, and where each one lives

| | Change | Status | Where |
|---|---|---|---|
| (a) | hidden matrices initialized with variance `1 / mult` | ✅ | `mllm/init.py` |
| (b) | hidden matrices given a learning rate `/ mult` | ✅ | `mllm/optim.py`, `build_param_groups` |
| (c) | attention scaled by `1 / head_dim` instead of `1 / sqrt(head_dim)` | ✅ | `mllm/layers/attention.py` |
| (d) | output logits multiplied by `1 / mult` | ✅ | `mllm/layers/heads.py`, `LMHead` |

All four are now exercised together, on the real `Transformer` rather than on a
stand-in.

## What was found while building this

Three things worth recording, because each one cost a debugging round.

**The output layer needs a different exponent from the hidden ones.** Table 8 of
the paper gives hidden weights a variance of `1 / fan_in` and output weights a
variance of `1 / fan_in²`, so the standard deviation is divided by `sqrt(mult)`
in one case and by `mult` in the other. Getting this wrong left a 6.0x spread
that dropped to 3.6x once fixed.

**Init and training must be checked separately.** The spread at step 0 is 1.02x,
so the initialization half is essentially exact. Everything above that appears
during training, which points at the learning rate groups rather than at the
init. Measuring only the final step would have hidden which half was wrong.

**The learning rate has to be small enough for the argument to apply.** At
`lr = 1e-2` the spread is 3.4x, at `lr = 1e-3` it is 1.31x. The muP derivation
describes the linearized dynamics, and a toy model at a large learning rate
leaves that regime immediately. This is not a bug in the implementation, but it
does mean a coordinate check run at an aggressive learning rate proves nothing.

**Moving to the real model improved the result.** On the toy residual MLP the
spread was 1.31x. On the actual `Transformer` it is 1.03x, essentially the
ideal value. The earlier residual was therefore mostly an artefact of the
stand-in rather than a flaw in the implementation, which is a good reminder
that a coordinate check measures the model you actually run.

## Remaining caveat

The task is synthetic (predict the next token id modulo the vocabulary) and the
model is small. A coordinate check confirms the parametrization is coherent
across widths, it does not confirm that a learning rate tuned at the base width
is optimal at scale. That claim needs a real learning rate sweep at two widths,
which belongs with the ablations in M7.

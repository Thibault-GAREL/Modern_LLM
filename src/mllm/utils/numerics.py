"""Precision policy, and the three places where fp32 is not negotiable.

Mixed precision keeps the weights in fp32 and runs the matmuls in bf16 or
fp16. That is safe for the matmuls and unsafe for anything that reduces over
a dimension or exponentiates, because those either accumulate rounding error
or leave the representable range.

Three places in this library always compute in fp32 and cast back:

1. **Normalization.** The mean square of a whole feature vector is a
   reduction. In fp16 the result matches an all-fp16 computation bit for bit
   rather than the fp32 one, which is why ``torch.nn.functional.rms_norm`` is
   not used as a fast path here. See ``mllm.layers.norm``.

2. **The RoPE cos and sin tables.** The angle for position ``m`` in the
   slowest band is tiny, and rounding it in bf16 makes the relative-position
   property drift after a few thousand tokens. See ``mllm.layers.pos``.

3. **Logits and softmax.** A softmax over a large vocabulary in bf16 loses the
   tail outright, and ``logsumexp`` over such logits is exactly the reduction
   bf16 handles worst. See ``mllm.layers.heads``.

Two of these are reductions over a large dimension and one is a small angle
accumulated over many positions. That is the pattern to look for when adding
a component.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor


@contextmanager
def fp32_ctx() -> Iterator[None]:
    """Disable autocast inside the block, forcing fp32 arithmetic.

    Use around a reduction that must not be done in reduced precision. Casting
    the inputs with ``.float()`` is not enough on its own, because autocast
    would cast them straight back for the next operation.
    """
    if torch.is_autocast_enabled():
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    else:
        yield


def relative_error(actual: Tensor, expected: Tensor) -> float:
    """Max relative deviation, normalized by the scale of ``expected``.

    Plain absolute error is useless for comparing precisions, since it depends
    entirely on how large the activations happen to be.
    """
    actual, expected = actual.float(), expected.float()
    scale = expected.abs().max().clamp_min(1e-12)
    return float((actual - expected).abs().max() / scale)


def pick_device() -> torch.device:
    """CUDA when it is genuinely usable, CPU otherwise.

    ``torch.cuda.is_available()`` alone is not enough. Under
    ``CUDA_VISIBLE_DEVICES=""`` it returns True while ``device_count()`` is 0,
    so a model built from it lands on a phantom device and its checkpoint
    cannot be loaded back anywhere.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_dtype(precision: str) -> torch.dtype | None:
    """Map a config precision string to the autocast dtype, None for fp32."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def supports_bf16(device: torch.device | str = "cuda") -> bool:
    """True only when bf16 runs natively, not through emulation.

    A GTX 1660 Ti reports ``is_bf16_supported() == True`` while having compute
    capability 7.5, where bf16 is emulated and slower than fp16. Trusting the
    plain check is how a training run silently becomes several times slower.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch without the keyword
        return torch.cuda.get_device_capability(device)[0] >= 8


def resolve_precision(precision: str, device: torch.device | str = "cuda") -> str:
    """Fall back from bf16 to fp16 when bf16 would only be emulated."""
    if precision == "bf16" and not supports_bf16(device):
        return "fp16"
    return precision

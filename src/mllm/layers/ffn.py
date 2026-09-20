"""Feed-forward networks, gated and ungated.

The 2017 block uses ``max(0, x W1 + b1) W2 + b2`` with ``d_ff = 4 * d_model``,
two matrices and a ReLU. Two thirds of the parameters of a block live here.

Gated variants (Shazeer, 2020, arXiv 2002.05202) split the input into a value
path and a gate path, so three matrices instead of two. To keep the parameter
budget identical the width drops from ``4d`` to ``8/3 d``, since ``3 * 8/3 =
8 = 2 * 4``. The paper reports better loss at equal size and says outright
that it offers no explanation for why.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from mllm.config import FFNConfig, ModelConfig
from mllm.init import mark_residual_projection

GATED_KINDS = ("swiglu", "geglu", "reglu")

# The gate activation is the only difference between the three gated variants
_GATE_ACTIVATIONS = {
    "swiglu": F.silu,  # Swish, the LLaMA and Mistral default
    "geglu": F.gelu,  # Gemma
    "reglu": F.relu,
}


def round_to_multiple(value: float, multiple: int) -> int:
    """Round up to the next multiple, keeping matmul shapes hardware friendly."""
    return multiple * ((int(value) + multiple - 1) // multiple)


def compute_d_ff(cfg: FFNConfig, d_model: int) -> int:
    """Hidden width of the feed-forward, from the config.

    An explicit ``d_ff`` wins. Otherwise the multiplier defaults to 4 for the
    ungated MLP and to 8/3 for the gated variants, which is what makes the two
    families cost the same.
    """
    if cfg.d_ff is not None:
        return cfg.d_ff
    mult = cfg.mult
    if mult is None:
        mult = 8.0 / 3.0 if cfg.kind in GATED_KINDS else 4.0
    return round_to_multiple(mult * d_model, cfg.multiple_of)


class MLP(nn.Module):
    """The 2017 feed-forward: one activation between two matrices."""

    def __init__(self, cfg: ModelConfig, d_ff: int | None = None) -> None:
        super().__init__()
        self.d_ff = d_ff if d_ff is not None else compute_d_ff(cfg.ffn, cfg.d_model)
        self.activation = F.relu if cfg.ffn.activation == "relu" else F.gelu
        self.up_proj = nn.Linear(cfg.d_model, self.d_ff, bias=cfg.bias)
        self.down_proj = mark_residual_projection(
            nn.Linear(self.d_ff, cfg.d_model, bias=cfg.bias)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.up_proj(x)))


class GatedMLP(nn.Module):
    """SwiGLU, GeGLU or ReGLU.

    ``down(act(gate(x)) * up(x))``. The gate learns which coordinates of the
    value path to let through, which a single activation cannot express.
    """

    def __init__(self, cfg: ModelConfig, d_ff: int | None = None) -> None:
        super().__init__()
        kind = cfg.ffn.kind
        if kind not in _GATE_ACTIVATIONS:
            raise ValueError(f"{kind} is not a gated FFN kind")
        self.kind = kind
        self.activation = _GATE_ACTIVATIONS[kind]
        self.d_ff = d_ff if d_ff is not None else compute_d_ff(cfg.ffn, cfg.d_model)
        self.gate_proj = nn.Linear(cfg.d_model, self.d_ff, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.d_model, self.d_ff, bias=cfg.bias)
        self.down_proj = mark_residual_projection(
            nn.Linear(self.d_ff, cfg.d_model, bias=cfg.bias)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class MatFormerMLP(GatedMLP):
    """Nested gated FFN (Devvrit et al., 2023, arXiv 2310.07707).

    The weights of a smaller FFN are literally the first rows of the larger
    one, so a single set of parameters contains several models at once. Gemma
    3n ships this as its E2B and E4B variants, extracted from one training run.

    What it buys is not quality, it is **one run instead of several**. Training
    a 5M and a 3M model separately costs two runs and gives two checkpoints
    that share nothing. Training a MatFormer gives both, plus every granularity
    in between through Mix'n'Match across layers.

    What it costs is a forward pass per granularity during training, since all
    of them are optimized jointly. That is the honest trade, and it is why the
    ablation reports the full-size loss against a plain FFN trained alone.
    """

    def __init__(self, cfg: ModelConfig, d_ff: int | None = None) -> None:
        super().__init__(cfg, d_ff)
        self.granularities = list(cfg.ffn.mat_granularities or [1.0])
        # resolved once, so a granularity always maps to the same slice
        self.widths = [max(1, int(round(self.d_ff * g))) for g in self.granularities]

    def width_for(self, granularity: float) -> int:
        """Hidden width used by a granularity, matched to the nearest declared one."""
        closest = min(self.granularities, key=lambda g: abs(g - granularity))
        return self.widths[self.granularities.index(closest)]

    def forward(self, x: Tensor, granularity: float = 1.0) -> Tensor:
        d = self.width_for(granularity)
        if d == self.d_ff:
            return super().forward(x)
        # slicing the weights rather than the activations is what makes the
        # smaller model a genuine sub-model instead of a masked one
        gate = F.linear(x, self.gate_proj.weight[:d])
        up = F.linear(x, self.up_proj.weight[:d])
        return F.linear(self.activation(gate) * up, self.down_proj.weight[:, :d])

    def extra_repr(self) -> str:
        return f"d_ff={self.d_ff}, granularities={self.granularities}"


def build_ffn(cfg: ModelConfig, d_ff: int | None = None) -> nn.Module:
    """Instantiate the feed-forward selected by the config."""
    if cfg.ffn.mat_granularities is not None:
        if cfg.ffn.kind == "mlp":
            raise ValueError("MatFormer is implemented for the gated FFN kinds only")
        return MatFormerMLP(cfg, d_ff)
    if cfg.ffn.kind == "mlp":
        return MLP(cfg, d_ff)
    return GatedMLP(cfg, d_ff)

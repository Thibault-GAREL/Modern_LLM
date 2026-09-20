"""Weight initialization: standard, scaled residual, and muP.

The 2017 paper says nothing about initialization. Three conventions have
since become standard and each is a flag here.

**Scaled residual init** (GPT-2, Megatron): the projection that writes back
into the residual stream (``attn.o_proj``, ``ffn.down_proj``) is divided by
``sqrt(2 * n_layers)``. With pre-norm the residual stream is a sum of
``2 * n_layers`` branches, so without this its variance grows linearly with
depth and the last layers see activations the first layers never saw.

**muP** (Yang et al., 2022, arXiv 2203.03466) makes the optimal learning
rate independent of width, so it can be tuned on a small model and reused on
a large one. Four changes are needed and only three live here:

  (a) hidden matrices initialized with variance ``1 / mult``  (this file)
  (b) hidden matrices given a learning rate ``/ mult``        (``optim.py``, M5)
  (c) attention scaled by ``1 / head_dim`` not ``1 / sqrt(head_dim)`` (``attention.py``, M3)
  (d) output logits multiplied by ``1 / mult``                (this file, applied in M5)

where ``mult = d_model / base_d_model``. The only valid check that all four
are right is a coordinate check, see ``bench/coord_check.py``.

Layers are tagged by attribute rather than by name, so this module does not
need to know how attention or the FFN name their submodules.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mllm.config import ModelConfig

RESIDUAL_PROJ_ATTR = "_mt_residual_proj"
OUTPUT_LAYER_ATTR = "_mt_output_layer"


def mark_residual_projection(module: nn.Module) -> nn.Module:
    """Tag a layer as writing back into the residual stream.

    Call this on ``attn.o_proj`` and ``ffn.down_proj`` at construction time.
    Returns the module so it can be used inline.
    """
    setattr(module, RESIDUAL_PROJ_ATTR, True)
    return module


def mark_output_layer(module: nn.Module) -> nn.Module:
    """Tag a layer as the model output (the LM head)."""
    setattr(module, OUTPUT_LAYER_ATTR, True)
    return module


def is_residual_projection(module: nn.Module) -> bool:
    return bool(getattr(module, RESIDUAL_PROJ_ATTR, False))


def is_output_layer(module: nn.Module) -> bool:
    return bool(getattr(module, OUTPUT_LAYER_ATTR, False))


def width_multiplier(cfg: ModelConfig) -> float:
    """``d_model / base_d_model``, the muP width ratio (1.0 when muP is off)."""
    if not cfg.mup.enabled:
        return 1.0
    return cfg.d_model / cfg.mup.base_d_model


def output_logit_multiplier(cfg: ModelConfig) -> float:
    """muP change (d): the factor applied to the logits before the loss."""
    if not cfg.mup.enabled:
        return 1.0
    return 1.0 / width_multiplier(cfg)


def base_std(cfg: ModelConfig) -> float:
    """Standard deviation before any depth or width correction."""
    if cfg.init.scheme == "inv_sqrt_d":
        return 1.0 / math.sqrt(cfg.d_model)
    return cfg.init.std


def init_weights(model: nn.Module, cfg: ModelConfig) -> None:
    """Initialize every parameter of ``model`` in place.

    Order of the corrections on a linear layer:

    1. start from ``base_std(cfg)``
    2. if muP is on and the layer is hidden, divide by ``sqrt(mult)``
       (variance ``1 / mult``)
    3. if it is a residual projection, divide by ``sqrt(2 * n_layers)``
    4. biases to zero, norm gains left at their constructed value
    """
    std = base_std(cfg)
    mult = width_multiplier(cfg)
    depth_factor = math.sqrt(2.0 * cfg.n_layers) if cfg.init.scaled_residual else 1.0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            layer_std = std
            if cfg.mup.enabled:
                # Table 8 of Yang et al. (2022), Adam column:
                #   hidden weights  variance 1 / fan_in   so std / sqrt(mult)
                #   output weights  variance 1 / fan_in^2 so std / mult
                layer_std /= mult if is_output_layer(module) else math.sqrt(mult)
            if is_residual_projection(module):
                layer_std /= depth_factor
            nn.init.normal_(module.weight, mean=0.0, std=layer_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings keep a width-independent scale under muP
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

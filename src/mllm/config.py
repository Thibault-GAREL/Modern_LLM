"""Configuration schema for modern-llm.

Nested Pydantic v2 models validating every architectural flag. Each config
class maps to one component family (attention, positions, FFN/MoE, norms,
init, training) so ablations flip one field at a time against
``configs/base.yaml``, the vanilla Transformer of Vaswani et al. (2017,
arXiv 1706.03762). ``extra="forbid"`` turns YAML typos into immediate
errors instead of silently ignored keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base class for all configs: unknown keys are rejected (YAML typo guard)."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Positions (M2)
# ---------------------------------------------------------------------------


class RopeScalingConfig(StrictModel):
    """Context extension applied on top of RoPE.

    Kinds: ``linear`` is Position Interpolation (Chen et al., 2023, arXiv
    2306.15595), ``ntk-aware`` and ``dynamic-ntk`` rescale theta, ``yarn``
    (Peng et al., 2023, arXiv 2309.00071) interpolates per frequency band and
    adds a temperature factor sqrt(1/t) = 0.1*ln(s) + 1 on the attention
    scale, ``llama3`` uses a wavelength-band ramp (arXiv 2407.21783).
    """

    kind: Literal["linear", "ntk-aware", "dynamic-ntk", "yarn", "llama3"] = "yarn"
    factor: float = Field(4.0, gt=1.0)
    original_max_seq_len: int = Field(4096, gt=0)
    # yarn only
    beta_fast: float = Field(32.0, gt=0)
    beta_slow: float = Field(1.0, gt=0)
    attn_temperature: bool = True  # the factor that distinguishes YaRN from NTK-by-parts
    # llama3 only
    low_freq_factor: float = Field(1.0, gt=0)
    high_freq_factor: float = Field(4.0, gt=0)


class PositionConfig(StrictModel):
    """Positional information scheme.

    ``sinusoidal`` and ``learned`` reproduce 2017-era absolute embeddings.
    ``rope`` (Su et al., 2021, arXiv 2104.09864) rotates q/k pairs, ``alibi``
    (Press et al., 2021, arXiv 2108.12409) adds linear biases to the scores,
    ``nope`` keeps the causal mask alone (arXiv 2305.19466).
    """

    kind: Literal["sinusoidal", "learned", "rope", "alibi", "nope"] = "rope"
    rope_theta: float = Field(10_000.0, gt=0)  # 500_000 for long-context models
    rope_style: Literal["interleaved", "half"] = "half"  # half = GPT-NeoX / LLaMA convention
    scaling: RopeScalingConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> PositionConfig:
        if self.scaling is not None and self.kind != "rope":
            raise ValueError("position.scaling requires position.kind == 'rope'")
        return self


# ---------------------------------------------------------------------------
# Attention (M3)
# ---------------------------------------------------------------------------


class AttentionConfig(StrictModel):
    """Attention family and its variants.

    MQA is Shazeer (2019, arXiv 1911.02150), GQA is Ainslie et al. (2023,
    arXiv 2305.13245), MLA is DeepSeek-V2 (2024, arXiv 2405.04434). Sliding
    window with periodic global layers follows Gemma-style alternation,
    attention sinks follow StreamingLLM (Xiao et al., 2023, arXiv 2309.17453),
    logit softcapping follows Gemma 2 (arXiv 2408.00118).
    """

    kind: Literal["mha", "mqa", "gqa", "mla"] = "gqa"
    n_heads: int = Field(16, gt=0)
    n_kv_heads: int | None = Field(4, gt=0)  # None => n_heads (MHA), 1 => MQA
    head_dim: int | None = Field(None, gt=0)  # None => d_model // n_heads
    # MLA only
    kv_lora_rank: int = Field(512, gt=0)
    q_lora_rank: int | None = Field(1536, gt=0)  # None => full-rank Q projection
    qk_rope_head_dim: int = Field(64, gt=0)
    qk_nope_head_dim: int = Field(128, gt=0)
    v_head_dim: int = Field(128, gt=0)
    # variants
    qk_norm: bool = False
    qk_norm_after_rope: bool = False  # usual convention is norm then RoPE
    logit_softcap: float | None = Field(None, gt=0)  # e.g. 50.0 (Gemma 2)
    sliding_window: int | None = Field(None, gt=0)
    global_every: int | None = Field(None, gt=0)  # 1 global layer every N layers
    attn_sinks: int = Field(0, ge=0)  # n first tokens always visible (StreamingLLM)
    learned_sink: bool = False  # sink as an extra learned logit in the softmax
    scale: Literal["1/sqrt(d)", "mup"] = "1/sqrt(d)"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)

    @property
    def kv_heads(self) -> int:
        """Effective number of KV heads (``n_kv_heads=None`` means MHA)."""
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @model_validator(mode="after")
    def _check(self) -> AttentionConfig:
        if self.kind == "mla":
            if self.n_kv_heads is not None:
                raise ValueError(
                    "attention.kind == 'mla' is incompatible with an explicit "
                    "n_kv_heads: the latent cache replaces KV heads entirely"
                )
        else:
            if self.n_heads % self.kv_heads != 0:
                raise ValueError(
                    f"n_heads ({self.n_heads}) must be a multiple of "
                    f"n_kv_heads ({self.kv_heads})"
                )
            if self.kind == "mqa" and self.kv_heads != 1:
                raise ValueError("attention.kind == 'mqa' requires n_kv_heads == 1")
            if self.kind == "mha" and self.kv_heads != self.n_heads:
                raise ValueError(
                    "attention.kind == 'mha' requires n_kv_heads == n_heads or None"
                )
        if self.global_every is not None and self.sliding_window is None:
            raise ValueError("attention.global_every requires attention.sliding_window")
        if self.learned_sink and self.attn_sinks > 0:
            raise ValueError(
                "choose either attn_sinks (first visible tokens) or learned_sink, not both"
            )
        return self


# ---------------------------------------------------------------------------
# FFN and MoE (M4)
# ---------------------------------------------------------------------------


class FFNConfig(StrictModel):
    """Feed-forward family.

    ``mlp`` is the 2017 two-matrix block. Gated variants (SwiGLU, GeGLU,
    ReGLU) follow Shazeer (2020, arXiv 2002.05202) with three matrices and
    ``d_ff = mult * d_model`` rounded up to ``multiple_of`` (mult defaults to
    8/3 for gated kinds to keep the parameter budget of a 4d MLP).
    """

    kind: Literal["mlp", "swiglu", "geglu", "reglu"] = "swiglu"
    activation: Literal["relu", "gelu"] = "gelu"  # mlp only
    d_ff: int | None = Field(None, gt=0)  # None => computed from mult and multiple_of
    mult: float | None = Field(None, gt=0)  # None => 4.0 (mlp) or 8/3 (gated)
    multiple_of: int = Field(256, gt=0)

    # MatFormer (Devvrit et al., 2023, arXiv 2310.07707), the nested FFN used by
    # Gemma 3n. Each granularity is a fraction of d_ff whose weights are a
    # sub-matrix of the next size up, so one training run yields several
    # deployable models. None disables it entirely.
    mat_granularities: list[float] | None = None

    @model_validator(mode="after")
    def _check_granularities(self) -> FFNConfig:
        if self.mat_granularities is None:
            return self
        g = self.mat_granularities
        if not g:
            raise ValueError("mat_granularities must be None or a non-empty list")
        if any(not 0 < x <= 1 for x in g):
            raise ValueError(f"mat_granularities must lie in (0, 1], got {g}")
        if 1.0 not in g:
            raise ValueError(
                "mat_granularities must include 1.0, the full model the others nest inside"
            )
        if sorted(g, reverse=True) != g:
            raise ValueError(f"mat_granularities must be sorted from largest, got {g}")
        if len(set(g)) != len(g):
            raise ValueError(f"mat_granularities must be distinct, got {g}")
        return self


class MoEConfig(StrictModel):
    """Mixture of Experts.

    Fine-grained experts plus always-active shared experts follow DeepSeekMoE
    (2024, arXiv 2401.06066). Balancing is either the classic auxiliary loss
    (Switch, Fedus et al., 2021, arXiv 2101.03961) or the aux-loss-free
    per-expert selection bias of DeepSeek-V3 (arXiv 2412.19437, method in
    arXiv 2408.15664). Router z-loss follows ST-MoE (arXiv 2202.08906).
    """

    enabled: bool = False
    n_experts: int = Field(64, gt=0)
    top_k: int = Field(6, gt=0)
    n_shared_experts: int = Field(2, ge=0)
    d_ff_expert: int | None = Field(None, gt=0)
    first_k_dense: int = Field(1, ge=0)  # keep the first layers dense (DeepSeek convention)
    gate: Literal["softmax", "sigmoid"] = "softmax"
    balance: Literal["aux_loss", "aux_loss_free"] = "aux_loss_free"
    aux_loss_alpha: float = Field(0.01, ge=0)  # aux_loss mode
    seq_aux_alpha: float = Field(1e-4, ge=0)  # tiny sequence-wise complement (aux_loss_free)
    router_z_loss_coef: float = Field(1e-3, ge=0)
    bias_update_gamma: float = Field(1e-3, gt=0)  # aux_loss_free bias update speed
    capacity_factor: float | None = Field(None, gt=0)  # None => no token dropping

    @model_validator(mode="after")
    def _check(self) -> MoEConfig:
        if self.top_k > self.n_experts:
            raise ValueError(
                f"moe.top_k ({self.top_k}) cannot exceed moe.n_experts ({self.n_experts})"
            )
        return self


# ---------------------------------------------------------------------------
# Normalization and initialization (M1)
# ---------------------------------------------------------------------------


class NormConfig(StrictModel):
    """Normalization layers and their placement in the block.

    ``layernorm`` is the 2017 reference, ``rmsnorm`` (Zhang and Sennrich,
    2019, arXiv 1910.07467) is the current standard and is always computed in
    fp32, ``dyt`` is Dynamic Tanh (Zhu et al., 2025, arXiv 2503.10622).
    Placement: ``post`` (2017), ``pre`` (current standard), ``sandwich``
    (norm before and after each sub-block, Gemma 2 style).
    """

    kind: Literal["layernorm", "rmsnorm", "dyt"] = "rmsnorm"
    placement: Literal["post", "pre", "sandwich"] = "pre"
    eps: float = Field(1e-5, gt=0)
    unit_offset: bool = False  # weight stored as (1 + w), Gemma convention
    dyt_alpha_init: float = Field(0.5, gt=0)


class MuPConfig(StrictModel):
    """muP hyperparameter transfer (Yang et al., 2022, arXiv 2203.03466)."""

    enabled: bool = False
    base_d_model: int = Field(256, gt=0)


class InitConfig(StrictModel):
    """Weight initialization.

    ``scaled_residual`` divides the output projections (attn.o_proj and
    ffn.down_proj) by sqrt(2 * n_layers) so the residual variance stays O(1)
    with depth (GPT-2 / Megatron practice).
    """

    scheme: Literal["fixed", "inv_sqrt_d"] = "fixed"
    std: float = Field(0.02, gt=0)  # fixed scheme only
    scaled_residual: bool = True


# ---------------------------------------------------------------------------
# Training (M5)
# ---------------------------------------------------------------------------


class TrainConfig(StrictModel):
    """Optimizer, schedule and runtime options for the minimal training loop.

    AdamW betas (0.9, 0.95) and weight decay 0.1 excluded from biases, norm
    gains and embeddings are the current open-weights consensus. The output
    z-loss (1e-4 * logsumexp(logits)^2) follows ST-MoE (arXiv 2202.08906).
    """

    # optimizer (AdamW)
    lr: float = Field(3e-4, gt=0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(0.1, ge=0)
    max_grad_norm: float = Field(1.0, gt=0)
    z_loss_coef: float = Field(1e-4, ge=0)
    # schedule
    schedule: Literal["cosine", "wsd"] = "cosine"
    warmup_steps: int = Field(100, ge=0)
    decay_steps: int | None = Field(None, gt=0)  # wsd only
    min_lr_ratio: float = Field(0.1, ge=0, le=1)
    max_steps: int = Field(1000, gt=0)
    # batch
    micro_batch_size: int = Field(8, gt=0)
    grad_accum_steps: int = Field(1, gt=0)
    seq_len: int = Field(256, gt=0)
    # runtime
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"  # fp16 fallback on pre-Ampere GPUs
    compile: bool = False  # torch.compile needs Triton, mostly unavailable on Windows
    activation_checkpointing: bool = False
    seed: int = 42
    log_interval: int = Field(10, gt=0)
    eval_interval: int = Field(500, gt=0)  # held-out loss, needs a val split
    ckpt_interval: int = Field(500, gt=0)
    out_dir: str = "outputs"

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError(f"train.betas must be in [0, 1), got {self.betas}")
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"train.warmup_steps ({self.warmup_steps}) cannot exceed "
                f"max_steps ({self.max_steps})"
            )
        if self.schedule == "wsd" and self.decay_steps is None:
            raise ValueError("train.schedule == 'wsd' requires train.decay_steps")
        return self


# ---------------------------------------------------------------------------
# Model root
# ---------------------------------------------------------------------------


class ModelConfig(StrictModel):
    """Full architecture description assembled from the component configs."""

    d_model: int = Field(512, gt=0)
    n_layers: int = Field(8, gt=0)
    vocab_size: int = Field(32_000, gt=0)
    max_seq_len: int = Field(1024, gt=0)
    bias: bool = False  # 2017 used biases everywhere, modern models drop them
    dropout: float = Field(0.0, ge=0, lt=1)
    tie_embeddings: bool = True
    # Multi-Token Prediction (Gloeckle et al., 2024, arXiv 2404.19737; DeepSeek-V3)
    mtp_depth: int = Field(0, ge=0)
    mtp_lambda: float = Field(0.3, ge=0)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    ffn: FFNConfig = Field(default_factory=FFNConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    norm: NormConfig = Field(default_factory=NormConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    mup: MuPConfig = Field(default_factory=MuPConfig)

    @property
    def head_dim(self) -> int:
        """Per-head dim for q/k/v (non-MLA kinds)."""
        if self.attention.head_dim is not None:
            return self.attention.head_dim
        return self.d_model // self.attention.n_heads

    @property
    def rope_head_dim(self) -> int:
        """Dimension the rotation actually applies to.

        MLA splits each head into a non-rotated and a rotated part, and only
        the second one goes through RoPE, so the cos/sin tables must be built
        for ``qk_rope_head_dim`` rather than for the full head.
        """
        if self.attention.kind == "mla":
            return self.attention.qk_rope_head_dim
        return self.head_dim

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.attention.kind != "mla" and self.attention.head_dim is None:
            if self.d_model % self.attention.n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads "
                    f"({self.attention.n_heads}) when attention.head_dim is not set"
                )
        if self.mtp_depth > 0 and not self.tie_embeddings:
            raise ValueError(
                "mtp_depth > 0 requires tie_embeddings: MTP modules share the LM head"
            )
        if self.moe.enabled and self.moe.first_k_dense >= self.n_layers:
            raise ValueError(
                f"moe.first_k_dense ({self.moe.first_k_dense}) must be < "
                f"n_layers ({self.n_layers})"
            )
        if self.mup.enabled and self.attention.scale != "mup":
            raise ValueError(
                "mup.enabled requires attention.scale == 'mup' (1/head_dim scaling)"
            )
        return self


class Config(StrictModel):
    """Root object loaded from one YAML profile in ``configs/``."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


def load_config(path: str | Path) -> Config:
    """Convenience wrapper around :meth:`Config.from_yaml`."""
    return Config.from_yaml(path)

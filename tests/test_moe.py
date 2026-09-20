"""Mixture of Experts (M4).

Two tests carry the milestone. A degenerate MoE with one expert must equal the
dense feed-forward it replaces, and the aux-loss-free bias must actually drive
an imbalanced router back towards uniform load.
"""

from __future__ import annotations

import pytest
import torch

from mllm.config import FFNConfig, ModelConfig, MoEConfig
from mllm.layers.ffn import build_ffn
from mllm.layers.moe import MoE, Router, load_balancing_loss, router_z_loss
from mllm.utils.seed import set_determinism

D_MODEL = 64


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def make(**moe_kw) -> MoE:
    cfg = ModelConfig(
        d_model=D_MODEL,
        n_layers=2,
        ffn=FFNConfig(kind="swiglu", multiple_of=1),
        moe=MoEConfig(enabled=True, **moe_kw),
    )
    return MoE(cfg)


# ---------------------------------------------------------------------------
# The degenerate case
# ---------------------------------------------------------------------------


def test_single_expert_moe_equals_the_dense_ffn():
    """One expert, top 1, no shared expert: routing becomes the identity."""
    cfg = ModelConfig(
        d_model=D_MODEL,
        n_layers=2,
        ffn=FFNConfig(kind="swiglu", multiple_of=1),
        moe=MoEConfig(enabled=True, n_experts=1, top_k=1, n_shared_experts=0),
    )
    set_determinism(3)
    moe = MoE(cfg)
    set_determinism(3)
    dense = build_ffn(cfg)
    moe.experts[0].load_state_dict(dense.state_dict())

    x = torch.randn(2, 6, D_MODEL)
    out, _ = moe(x)
    torch.testing.assert_close(out, dense(x), rtol=1e-5, atol=1e-5)


def test_shared_expert_is_always_added():
    moe = make(n_experts=4, top_k=1, n_shared_experts=1)
    x = torch.randn(1, 3, D_MODEL)
    out, _ = moe(x)
    shared_only = moe.shared_experts[0](x.reshape(-1, D_MODEL)).view_as(x)
    assert not torch.allclose(out, shared_only)
    # removing the routed part must leave exactly the shared contribution
    with torch.no_grad():
        for e in moe.experts:
            e.down_proj.weight.fill_(0.0)
    out_zeroed, _ = moe(x)
    torch.testing.assert_close(out_zeroed, shared_only, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["softmax", "sigmoid"])
def test_router_weights_are_normalized(gate: str):
    router = Router(D_MODEL, MoEConfig(n_experts=8, top_k=3, gate=gate))
    _, weights, _, _ = router(torch.randn(10, D_MODEL))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(10), rtol=1e-5, atol=1e-5)


def test_router_selects_top_k_distinct_experts():
    router = Router(D_MODEL, MoEConfig(n_experts=8, top_k=3))
    idx, _, _, _ = router(torch.randn(20, D_MODEL))
    assert idx.shape == (20, 3)
    for row in idx:
        assert len(set(row.tolist())) == 3


def test_selection_bias_moves_selection_but_not_weights():
    """The distinguishing property of aux-loss-free balancing."""
    cfg = MoEConfig(n_experts=4, top_k=1, balance="aux_loss_free")
    router = Router(D_MODEL, cfg)
    x = torch.randn(16, D_MODEL)

    idx_before, w_before, scores_before, _ = router(x)
    with torch.no_grad():
        router.expert_bias[3] = 10.0
    idx_after, w_after, scores_after, _ = router(x)

    assert (idx_after == 3).all(), "a large bias must capture every token"
    # the underlying scores, hence the weights, are untouched by the bias
    torch.testing.assert_close(scores_before, scores_after)
    torch.testing.assert_close(w_after, torch.ones_like(w_after))


def test_bias_is_a_buffer_not_a_parameter():
    """It is updated by a rule, so it must never receive a gradient."""
    router = Router(D_MODEL, MoEConfig(n_experts=4, top_k=1))
    names = {n for n, _ in router.named_parameters()}
    assert "expert_bias" not in names
    assert "expert_bias" in dict(router.named_buffers())


def test_bias_update_pushes_overloaded_experts_down():
    cfg = MoEConfig(n_experts=4, top_k=1, balance="aux_loss_free", bias_update_gamma=0.1)
    router = Router(D_MODEL, cfg)
    load = torch.tensor([0.7, 0.1, 0.1, 0.1])
    router.update_bias(load)
    assert router.expert_bias[0] < 0, "the overloaded expert must become harder to pick"
    assert (router.expert_bias[1:] > 0).all()


def test_bias_update_is_inert_in_aux_loss_mode():
    router = Router(D_MODEL, MoEConfig(n_experts=4, top_k=1, balance="aux_loss"))
    router.update_bias(torch.tensor([0.7, 0.1, 0.1, 0.1]))
    torch.testing.assert_close(router.expert_bias, torch.zeros(4))


# ---------------------------------------------------------------------------
# Balancing converges
# ---------------------------------------------------------------------------


def test_aux_loss_free_drives_load_towards_uniform():
    """The bias rule alone must rebalance a router skewed towards one expert."""
    moe = make(n_experts=8, top_k=1, n_shared_experts=0, bias_update_gamma=0.05)
    with torch.no_grad():
        moe.router.weight[0] *= 20  # expert 0 outscores the rest for most tokens

    x = torch.randn(256, 1, D_MODEL)
    moe(x)
    cv_before = moe.last_metrics.load_cv

    for _ in range(200):
        moe(x)
        moe.balance_step()
    cv_after = moe.last_metrics.load_cv

    assert cv_before > 0.9, f"the setup should start imbalanced, cv={cv_before:.2f}"
    assert cv_after < 0.4, f"the bias should have rebalanced it, cv={cv_after:.2f}"
    assert cv_after < cv_before / 2


def test_bias_cannot_split_indistinguishable_tokens():
    """A real limit of the mechanism, worth knowing before trusting it.

    The bias shifts *which* expert wins, not *how many* win. When every token
    carries the same routing scores they migrate as one block, so the load
    cycles between experts and never spreads.
    """
    moe = make(n_experts=8, top_k=1, n_shared_experts=0, bias_update_gamma=0.05)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 10.0
    x = torch.rand(64, 1, D_MODEL)  # all positive, so expert 0 saturates

    moe(x)
    assert moe.last_metrics.max_load == 1.0
    for _ in range(100):
        moe(x)
        moe.balance_step()
    assert moe.last_metrics.max_load == 1.0, (
        "with identical tokens the load moves but never splits"
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_load_balancing_loss_is_minimal_when_uniform():
    n_experts = 4
    uniform_scores = torch.full((16, n_experts), 0.25)
    uniform_idx = torch.arange(16).remainder(n_experts).unsqueeze(1)
    balanced = load_balancing_loss(uniform_scores, uniform_idx, n_experts)

    skewed_scores = torch.zeros(16, n_experts)
    skewed_scores[:, 0] = 1.0
    skewed_idx = torch.zeros(16, 1, dtype=torch.long)
    collapsed = load_balancing_loss(skewed_scores, skewed_idx, n_experts)

    assert collapsed > balanced
    torch.testing.assert_close(balanced, torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_router_z_loss_penalizes_large_logits():
    small = router_z_loss(torch.randn(8, 4) * 0.1)
    large = router_z_loss(torch.randn(8, 4) * 10.0)
    assert large > small


def test_aux_loss_is_returned_and_finite():
    moe = make(n_experts=8, top_k=2, n_shared_experts=1)
    _, aux = moe(torch.randn(2, 8, D_MODEL))
    assert aux.numel() == 1 and torch.isfinite(aux)
    assert aux.requires_grad


# ---------------------------------------------------------------------------
# Metrics, without which a collapse goes unnoticed
# ---------------------------------------------------------------------------


def test_metrics_are_recorded():
    moe = make(n_experts=8, top_k=2)
    moe(torch.randn(4, 16, D_MODEL))
    m = moe.last_metrics
    assert 0.0 <= m.entropy <= torch.tensor(8.0).log().item() + 1e-5
    assert m.load_fraction.shape == (8,)
    torch.testing.assert_close(m.load_fraction.sum(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    assert set(m.as_dict()) == {
        "router/entropy", "router/load_cv", "router/drop_rate", "router/max_load"
    }


def test_entropy_collapses_with_the_router():
    """The metric that reveals a collapse the loss curve hides."""
    moe = make(n_experts=8, top_k=1)
    x = torch.rand(64, 1, D_MODEL)
    moe(x)
    healthy = moe.last_metrics.entropy
    assert healthy > 1.5, "a fresh router should spread its mass"

    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 10.0  # positive inputs, so expert 0 saturates
    moe(x)
    assert moe.last_metrics.entropy < 0.01
    assert moe.last_metrics.max_load == 1.0


# ---------------------------------------------------------------------------
# Capacity and dropping
# ---------------------------------------------------------------------------


def test_capacity_factor_drops_tokens_and_reports_it():
    moe = make(n_experts=4, top_k=1, n_shared_experts=0, capacity_factor=0.5)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 10.0  # everything goes to one expert
    moe(torch.randn(64, 1, D_MODEL))
    assert moe.last_metrics.drop_rate > 0.5


def test_no_drops_without_a_capacity_factor():
    moe = make(n_experts=4, top_k=1, capacity_factor=None)
    moe(torch.randn(64, 1, D_MODEL))
    assert moe.last_metrics.drop_rate == 0.0


# ---------------------------------------------------------------------------
# The point of MoE
# ---------------------------------------------------------------------------


def test_active_params_are_far_below_total():
    """Capacity grows while the compute per token does not."""
    moe = make(n_experts=32, top_k=2, n_shared_experts=1, d_ff_expert=32)
    assert moe.n_active_params() < moe.n_total_params() / 5

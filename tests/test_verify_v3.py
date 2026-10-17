"""Verifier v3: in-distribution ablation and a risk order fixed from the training fold."""

import numpy as np
import pytest

from crta.regime import REGIMES
from crta.verify import (
    audit_many_v3,
    empirical_banks,
    reference_weights,
    regime_order,
    with_crisis_prob,
    with_regime_prob,
)

# (mean, sd) of the next return per regime: calm has the best mean / variance, bear the worst.
MOMENTS = {"bull": (0.002, 0.01), "bear": (-0.001, 0.01), "calm": (0.001, 0.005), "crisis": (-0.003, 0.03)}


def _training(rows_per_regime=60):
    probs, nxt = [], []
    for name, (mean, sd) in MOMENTS.items():
        one_hot = np.eye(len(REGIMES))[REGIMES.index(name)]
        for k in range(rows_per_regime):
            probs.append(one_hot)
            nxt.append(mean + (sd if k % 2 == 0 else -sd))
    return np.array(probs), np.array(nxt)


def test_order_ranks_regimes_by_mean_over_variance():
    probs, nxt = _training()
    order = regime_order(probs, nxt)
    assert order["ranked"] == ["calm", "bull", "crisis", "bear"]
    assert order["least_defensive"] == "calm"
    assert order["most_defensive"] == "bear"
    assert order["rank"]["calm"] == 0 and order["rank"]["bear"] == 3
    assert order["mv_position"]["bull"] == pytest.approx(0.002 / 0.01**2)


def test_a_regime_with_too_little_weight_is_left_unranked():
    probs, nxt = _training()
    keep = ~((probs[:, REGIMES.index("crisis")] == 1.0) & (np.arange(len(probs)) % 60 >= 4))
    order = regime_order(probs[keep], nxt[keep], min_weight=30.0)
    assert order["rank"]["crisis"] is None
    assert "crisis" not in order["ranked"]


def test_with_regime_prob_generalises_the_crisis_sweep():
    row = np.array([0.1, 0.2, 0.6, 0.1])
    assert np.allclose(with_regime_prob(row, "crisis", 0.3), with_crisis_prob(row, 0.3))
    out = with_regime_prob(row, "bear", 0.5)
    assert out[REGIMES.index("bear")] == pytest.approx(0.5)
    assert out.sum() == pytest.approx(1.0)


def test_reference_weights_run_from_plus_one_to_minus_one_along_the_order():
    probs, nxt = _training()
    w = reference_weights(regime_order(probs, nxt))
    assert w[REGIMES.index("calm")] == pytest.approx(1.0)
    assert w[REGIMES.index("bear")] == pytest.approx(-1.0)
    assert w[REGIMES.index("calm")] > w[REGIMES.index("bull")] > w[REGIMES.index("crisis")]


def _audit(policy):
    probs, nxt = _training()
    order = regime_order(probs, nxt)
    banks = empirical_banks(probs)
    baseline = probs.mean(axis=0)
    return audit_many_v3(policy, probs, banks, order, baseline)


def test_a_policy_faithful_to_the_order_passes():
    probs, nxt = _training()
    w = reference_weights(regime_order(probs, nxt))
    result = _audit(lambda batch: np.clip(np.asarray(batch) @ w, -1.0, 1.0))
    assert result["passed"].mean() > 0.95


def test_ignoring_and_inverted_policies_fail():
    probs, nxt = _training()
    w = reference_weights(regime_order(probs, nxt))
    ignore = _audit(lambda batch: np.full(len(batch), 0.5))
    inverted = _audit(lambda batch: np.clip(np.asarray(batch) @ -w, -1.0, 1.0))
    assert ignore["passed"].mean() == 0.0
    assert inverted["passed"].mean() == 0.0


def test_ablation_uses_the_baseline_vector_not_zeros():
    seen = []

    def policy(batch):
        seen.append(np.asarray(batch))
        return np.zeros(len(batch))

    probs, nxt = _training()
    baseline = probs.mean(axis=0)
    audit_many_v3(policy, probs, empirical_banks(probs), regime_order(probs, nxt), baseline)
    rows = np.concatenate(seen)
    assert not np.any(np.all(rows == 0.0, axis=1))
    assert np.any(np.all(np.isclose(rows, baseline), axis=1))


def test_bars_in_an_unranked_regime_are_untestable_for_the_counterfactual():
    probs, nxt = _training()
    keep = ~((probs[:, REGIMES.index("crisis")] == 1.0) & (np.arange(len(probs)) % 60 >= 4))
    order = regime_order(probs[keep], nxt[keep], min_weight=30.0)
    w = reference_weights(order)
    result = audit_many_v3(
        lambda batch: np.clip(np.asarray(batch) @ w, -1.0, 1.0), probs, empirical_banks(probs), order, probs.mean(axis=0)
    )
    crisis_bars = probs[:, REGIMES.index("crisis")] == 1.0
    assert not result["testable"][crisis_bars].any()
    assert result["testable"][~crisis_bars].all()

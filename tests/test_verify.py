import numpy as np
import pytest

from crta.regime import REGIMES
from crta.verify import (
    audit_bar,
    audit_many,
    ignore_action,
    inverted_action,
    probe_action,
    with_crisis_prob,
)


BULLISH = np.array([0.8, 0.05, 0.1, 0.05])


def test_probe_passes():
    result = audit_bar(probe_action, BULLISH)
    assert result["passed"]
    assert np.all(result["curve"][1:] <= result["curve"][:-1] + 1e-8)


def test_ignore_fails_ablation():
    result = audit_bar(ignore_action, BULLISH)
    assert not result["ablation"]
    assert not result["passed"]


def test_inverted_fails_counterfactual():
    result = audit_bar(inverted_action, BULLISH)
    assert not result["counterfactual"]


def test_audit_many_matches_one_bar():
    rows = np.stack([BULLISH, np.array([0.05, 0.1, 0.05, 0.8])])

    def predict(batch):
        return np.array([probe_action(row) for row in batch])

    many = audit_many(predict, rows)
    one = audit_bar(probe_action, rows[1])
    assert bool(many["passed"][1]) is one["passed"]
    assert many["ablation_delta"][1] == pytest.approx(one["ablation_delta"])
    assert np.allclose(many["curve"][1], one["curve"])


def test_crisis_sweep_sums_to_one():
    base = np.array([0.5, 0.2, 0.2, 0.1])
    low = with_crisis_prob(base, 0.0)
    high = with_crisis_prob(base, 1.0)
    assert low.sum() == pytest.approx(1.0)
    assert high.sum() == pytest.approx(1.0)
    assert low[REGIMES.index("crisis")] == pytest.approx(0.0)
    assert high[REGIMES.index("crisis")] == pytest.approx(1.0)

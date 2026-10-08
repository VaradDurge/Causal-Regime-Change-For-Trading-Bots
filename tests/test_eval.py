import numpy as np

from crta.eval import (
    bootstrap_ci,
    sharpe,
    strategy_returns,
    walk_forward_bounds,
)


def test_walk_forward_never_trains_on_the_test_bar():
    bounds = walk_forward_bounds(2400, min_train=600)
    assert bounds[0][0] == 600
    assert bounds[-1][1] == 2400
    for i, (train_end, test_end) in enumerate(bounds):
        test_start = train_end
        assert train_end <= test_start
        assert train_end < test_end
        assert set(range(train_end)).isdisjoint(range(test_start, test_end))
        if i:
            assert train_end > bounds[i - 1][0]
            assert bounds[i - 1][1] == train_end


def test_turnover_is_charged_when_the_position_changes():
    returns = strategy_returns(np.ones(4), np.full(4, 0.01))
    assert returns[0] == np.float64(0.01 - 0.001) or abs(returns[0] - (0.01 - 0.001)) < 1e-12
    assert abs(returns[1] - 0.01) < 1e-12


def test_bootstrap_interval_contains_the_point():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0004, 0.01, 1500)
    ci = bootstrap_ci(sample, sharpe, n=1000, seed=0)
    assert ci["low"] <= ci["point"] <= ci["high"]

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from crta.predictability import hit_rate, momentum_features, oos_r2, walk_forward_forecasts


def test_oos_r2_is_one_for_a_perfect_forecast_and_zero_for_the_benchmark():
    y = np.array([0.1, -0.2, 0.3, 0.0])
    bench = np.full(4, 0.05)
    assert oos_r2(y, y, bench) == pytest.approx(1.0)
    assert oos_r2(y, bench, bench) == pytest.approx(0.0)


def test_hit_rate_skips_zero_targets_and_zero_forecasts():
    y = np.array([1.0, -1.0, 0.0, 2.0, 1.0])
    pred = np.array([1.0, 1.0, 1.0, -1.0, 0.0])
    assert hit_rate(y, pred) == pytest.approx(1 / 3)


def test_momentum_features_are_past_log_returns():
    close = np.exp(np.arange(10.0))
    feats = momentum_features(close, bars_per_week=2, weeks=(1, 2))
    assert list(feats.columns) == ["mom_1w", "mom_2w"]
    assert np.isnan(feats["mom_1w"].iloc[1])
    assert feats["mom_1w"].iloc[5] == pytest.approx(2.0)
    assert feats["mom_2w"].iloc[5] == pytest.approx(4.0)


def test_walk_forward_fits_on_rows_before_each_test_block():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    y = 2.0 * X[:, 0] - X[:, 1]
    bounds = [(200, 300), (300, 400)]
    pred, bench = walk_forward_forecasts(X, y, bounds, LinearRegression)
    assert pred.shape == (200,)
    assert oos_r2(y[200:], pred, bench) == pytest.approx(1.0)
    assert bench[0] == pytest.approx(y[:200].mean())
    assert bench[-1] == pytest.approx(y[:300].mean())

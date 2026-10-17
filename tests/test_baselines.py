import numpy as np
import pytest

from crta.baselines import baseline_positions, momentum_position, regime_rule_position


def test_momentum_is_long_only_when_the_close_beats_its_lookback():
    close = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])
    assert momentum_position(close, 2).tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_momentum_at_t_ignores_later_closes():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    t = 150
    other = close.copy()
    other[t + 1 :] = other[t + 1 :][::-1] * 2.0
    assert np.array_equal(momentum_position(close, 84)[: t + 1], momentum_position(other, 84)[: t + 1])


def test_momentum_needs_a_positive_lookback():
    with pytest.raises(ValueError):
        momentum_position(np.ones(5), 0)


def test_regime_rule_is_long_in_bull_and_calm_and_flat_in_bear_and_crisis():
    probs = np.eye(4)  # bull, bear, calm, crisis
    assert regime_rule_position(probs).tolist() == [1.0, 0.0, 1.0, 0.0]
    assert regime_rule_position(np.array([[0.25, 0.25, 0.25, 0.25]])).tolist() == [0.5]


def test_baseline_positions_cover_buy_hold_momentum_and_the_regime_rule():
    close = np.linspace(100, 200, 400)
    probs = np.tile([0.0, 0.0, 1.0, 0.0], (400, 1))
    out = baseline_positions(close, probs, bars_per_week=42)
    assert set(out) == {"buy_hold", "regime_rule", "momentum_2w", "momentum_4w", "momentum_8w"}
    assert out["buy_hold"].tolist() == [1.0] * 400
    assert out["momentum_8w"][:336].sum() == 0.0
    assert out["momentum_8w"][336:].min() == 1.0

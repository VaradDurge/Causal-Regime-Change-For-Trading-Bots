import numpy as np
import pandas as pd
import pytest

from crta.agent import TradeEnv
from crta.data import FEATURE_COLS
from crta.eval import TURNOVER_COST
from crta.regime import REGIMES
from crta.verify import audit_many


def _frame(n=40, vol=0.02, ret=0.001):
    df = pd.DataFrame(np.zeros((n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
    df["vol_20"] = vol
    for name in REGIMES:
        df[name] = 0.25
    df["next_ret"] = ret
    df["close"] = 100.0
    df["timestamp"] = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return df


def test_mean_variance_reward_subtracts_the_volatility_penalty():
    frame = _frame()
    env = TradeEnv(frame, True, reward_mode="mean_variance", risk_aversion=2.0)
    env.reset(seed=0)
    _, reward, *_ = env.step(np.array([0.5], dtype=np.float32))
    expected = 0.5 * 0.001 - TURNOVER_COST * 0.5 - 2.0 * 0.25 * 0.02**2
    assert np.isclose(reward, expected)


def test_mean_variance_has_an_interior_optimum_where_raw_picks_a_rail():
    vol, ret, lam = 0.02, 0.0004, 2.0
    grid = np.linspace(-1.0, 1.0, 201)
    raw = grid * ret
    mv = grid * ret - lam * grid**2 * vol**2
    assert grid[np.argmax(raw)] == 1.0
    best = grid[np.argmax(mv)]
    assert np.isclose(best, ret / (2 * lam * vol**2), atol=0.01)
    assert -1.0 < best < 1.0


def test_mean_variance_needs_positive_risk_aversion():
    with pytest.raises(ValueError):
        TradeEnv(_frame(), True, reward_mode="mean_variance", risk_aversion=0.0)


def test_risk_aversion_is_ignored_by_the_default_reward():
    frame = _frame()
    plain = TradeEnv(frame, True)
    other = TradeEnv(frame, True, risk_aversion=5.0)
    plain.reset(seed=0)
    other.reset(seed=0)
    action = np.array([0.7], dtype=np.float32)
    assert plain.step(action)[1] == other.step(action)[1]


def test_bound_aware_audit_keeps_passed_and_drops_impossible_bars():
    probs = np.tile(np.array([0.0, 0.0, 1.0, 0.0]), (6, 1))

    def pinned(batch):
        return np.ones(len(batch))

    plain = audit_many(pinned, probs)
    aware = audit_many(pinned, probs, bound_aware=True)
    assert np.array_equal(plain["passed"], aware["passed"])
    assert aware["cf_untestable"].all()
    assert aware["n_testable"] == 0
    assert "cf_untestable" not in plain

"""Daily long/flat task: decision bars, features, folds and the risk-aware reward."""

import numpy as np
import pandas as pd
import pytest

from crta.daily import (
    DAILY_FEATURES,
    PositionEnv,
    Scaler,
    daily_bounds,
    daily_frame,
    execute_position,
)
from crta.regime import REGIMES


def _frame4h(days=120, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 6
    ts = pd.date_range("2021-01-01 00:00", periods=n, freq="4h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    frame = pd.DataFrame({"timestamp": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    probs = rng.dirichlet(np.ones(4), size=n)
    return frame, probs


def test_daily_frame_takes_the_daily_close_and_its_regime_row():
    frame, probs = _frame4h()
    daily = daily_frame(frame, probs)
    assert (daily["timestamp"].dt.hour == 20).all()
    row = daily.iloc[10]
    src = int(row["row4h"])
    assert frame["close"].iloc[src] == pytest.approx(row["close"])
    assert np.allclose(daily.loc[daily.index[10], list(REGIMES)].to_numpy(dtype=float), probs[src])
    assert set(DAILY_FEATURES) <= set(daily.columns)


def test_daily_returns_chain_daily_closes():
    frame, probs = _frame4h()
    daily = daily_frame(frame, probs)
    closes = daily["close"].to_numpy()
    assert np.allclose(daily["log_ret"].to_numpy()[1:], np.log(closes[1:] / closes[:-1]))
    assert np.allclose(daily["next_ret"].to_numpy()[:-1], daily["log_ret"].to_numpy()[1:])


def test_daily_features_at_t_ignore_later_bars():
    frame, probs = _frame4h()
    daily = daily_frame(frame, probs)
    cut = int(daily["row4h"].iloc[30])
    other = frame.copy()
    other.loc[cut + 1 :, ["open", "high", "low", "close"]] *= 1.5
    daily_other = daily_frame(other, probs)
    cols = list(DAILY_FEATURES)
    upto = daily["row4h"] <= cut
    assert np.allclose(daily.loc[upto, cols].to_numpy(), daily_other.loc[upto, cols].to_numpy())


def test_daily_bounds_follow_the_four_hour_calendar():
    frame, probs = _frame4h()
    daily = daily_frame(frame, probs)
    # Cuts after the 56-day momentum warm-up, as the real folds are.
    bounds4h = [(400, 500), (500, 600), (600, len(frame))]
    bounds = daily_bounds(daily["timestamp"], frame["timestamp"], bounds4h)
    assert len(bounds) == 3
    assert bounds[-1][1] == len(daily)
    for (start, end), (s4, _e4) in zip(bounds, bounds4h):
        assert start > 0
        assert daily["timestamp"].iloc[start] >= frame["timestamp"].iloc[s4]
        assert daily["timestamp"].iloc[start - 1] < frame["timestamp"].iloc[s4]
        assert start < end


def test_heads_map_raw_actions_into_zero_one():
    assert execute_position("clip01", 1.7) == 1.0
    assert execute_position("clip01", -0.2) == 0.0
    assert execute_position("tanh01", 0.0) == pytest.approx(0.5)
    assert execute_position("tanh01", 50.0) == pytest.approx(1.0)
    assert execute_position("beta", 0.3) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        execute_position("discrete5", 0.0)


def _toy_daily():
    frame, probs = _frame4h()
    return daily_frame(frame, probs)


def test_reward_is_return_minus_cost_minus_variance_penalty():
    daily = _toy_daily()
    scaler = Scaler.fit(daily[list(DAILY_FEATURES)].to_numpy())
    env = PositionEnv(daily, True, "beta", risk_aversion=2.0, reward_scale=1.0, scaler=scaler)
    env.reset(options={"position": 0.2})
    r = float(daily["next_ret"].iloc[0])
    sigma = float(daily["vol_20"].iloc[0])
    _obs, reward, _done, _trunc, _info = env.step(np.array([0.6], dtype=np.float32))
    a = float(np.float32(0.6))  # the action arrives as float32
    want = a * r - 0.001 * (a - 0.2) - 2.0 * a * a * sigma**2
    assert reward == pytest.approx(want)


def test_observation_is_scaled_features_position_and_probabilities():
    daily = _toy_daily()
    feats = daily[list(DAILY_FEATURES)].to_numpy()
    scaler = Scaler.fit(feats)
    env = PositionEnv(daily, True, "clip01", risk_aversion=1.0, scaler=scaler)
    obs, _ = env.reset(options={"position": 0.5})
    k = len(DAILY_FEATURES)
    assert obs.shape == (k + 1 + 4,)
    assert np.allclose(obs[:k], (feats[0] - scaler.mean) / scaler.std, atol=1e-5)
    assert obs[k] == pytest.approx(0.5)
    blind = PositionEnv(daily, False, "clip01", risk_aversion=1.0, scaler=scaler)
    assert blind.reset()[0].shape == (k + 1,)


def test_scaler_floors_flat_columns():
    s = Scaler.fit(np.column_stack([np.arange(5.0), np.ones(5)]))
    assert s.std[1] == 1.0

"""PPO trains every head end to end, and a saved monotone twin reloads with its constraint."""

import numpy as np
import pandas as pd
import pytest
import torch as th
from stable_baselines3 import PPO

from crta.daily import DAILY_FEATURES, Scaler, audit_predictor, daily_frame, run_positions, train_daily


def _daily(days=200, seed=1):
    rng = np.random.default_rng(seed)
    n = days * 6
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )
    return daily_frame(frame, rng.dirichlet(np.ones(4), size=n))


@pytest.mark.parametrize("head", ["beta", "tanh01", "clip01"])
@pytest.mark.parametrize("monotone", [False, True])
def test_every_head_trains_and_trades_inside_zero_one(head, monotone):
    daily = _daily()
    scaler = Scaler.fit(daily[list(DAILY_FEATURES)].to_numpy())
    model = train_daily(daily, True, head, 2.0, seed=0, timesteps=512, scaler=scaler, monotone=monotone, regime_rank=[1, 3, 0, 2])
    pos = run_positions(model, daily, True, head, scaler)
    assert len(pos) == len(daily)
    assert pos.min() >= 0.0 and pos.max() <= 1.0


def test_audit_predictor_matches_the_rollout_on_the_actual_probabilities():
    daily = _daily()
    scaler = Scaler.fit(daily[list(DAILY_FEATURES)].to_numpy())
    model = train_daily(daily, True, "beta", 2.0, seed=0, timesteps=512, scaler=scaler)
    pos = run_positions(model, daily, True, "beta", scaler)
    before = np.concatenate([[0.0], pos[:-1]])
    predict = audit_predictor(model, daily, before, scaler)
    probs = daily[["bull", "bear", "calm", "crisis"]].to_numpy()
    assert np.allclose(predict(probs), pos, atol=1e-5)


def test_a_saved_monotone_twin_reloads_with_its_rank(tmp_path):
    daily = _daily()
    scaler = Scaler.fit(daily[list(DAILY_FEATURES)].to_numpy())
    model = train_daily(daily, True, "beta", 2.0, seed=0, timesteps=256, scaler=scaler, monotone=True, regime_rank=[1, 3, 0, 2])
    model.save(tmp_path / "twin")
    loaded = PPO.load(tmp_path / "twin", device="cpu")
    assert loaded.policy.monotone and loaded.policy.regime_rank == [1, 3, 0, 2]
    obs = th.randn(16, len(DAILY_FEATURES) + 1 + 4)
    assert th.allclose(loaded.policy.position(obs), model.policy.position(obs), atol=1e-6)

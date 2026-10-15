"""action_mode is opt-in. Omitting it keeps the clipped Gaussian."""

import numpy as np
import pandas as pd

from crta.agent import DISCRETE_LEVELS, TradeEnv, audit_positions, execute_action, load_twin, observation, save_twin, train_ppo
from crta.config import ExperimentConfig
from crta.data import FEATURE_COLS
from crta.regime import REGIMES
from crta.select import choose_action_mode


def _frame(n=80):
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(0.0, 0.01, size=(n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
    for name in REGIMES:
        df[name] = 0.25
    df["next_ret"] = rng.normal(0.0, 0.01, n)
    return df


def test_omitted_action_mode_is_gaussian_clip():
    env = TradeEnv(_frame(), True)
    assert env.action_mode == "gaussian_clip"
    assert tuple(env.action_space.low) == (-1.0,)
    assert tuple(env.action_space.high) == (1.0,)
    assert ExperimentConfig().action_mode == "gaussian_clip"
    assert ExperimentConfig.load("configs/default.yaml").action_mode == "gaussian_clip"
    assert ExperimentConfig.load("configs/final.yaml").action_mode == "gaussian_clip"
    env.reset(seed=0)
    env.step(np.array([2.0], dtype=np.float32))
    assert env.position == 1.0


def test_tanh_and_discrete_map_actions():
    assert execute_action("gaussian_clip", np.array([2.0])) == 1.0
    assert abs(execute_action("tanh_env", np.array([2.0])) - np.tanh(2.0)) < 1e-8
    assert list(DISCRETE_LEVELS) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    env = TradeEnv(_frame(), True, action_mode="tanh_env")
    env.reset(seed=0)
    env.step(np.array([8.0], dtype=np.float32))
    assert abs(env.position - np.tanh(8.0)) < 1e-6
    assert abs(env.position) < 1.0
    discrete = TradeEnv(_frame(), False, action_mode="discrete5")
    discrete.reset(seed=0)
    for index, level in enumerate(DISCRETE_LEVELS):
        discrete.step(np.array([index]))
        assert discrete.position == level


def test_discrete_audit_uses_expectation(tmp_path):
    frame = _frame()
    model = train_ppo(
        frame,
        True,
        timesteps=64,
        window=32,
        action_mode="discrete5",
        threads=1,
        collapse_log_every=32,
        seed=1,
    )
    assert model.crta_meta["action_mode"] == "discrete5"
    loaded, meta = load_twin(save_twin(model, tmp_path / "discrete.zip"))
    assert meta["action_mode"] == "discrete5"
    obs = observation(frame.iloc[0][list(FEATURE_COLS)], frame.iloc[0][list(REGIMES)], 0.0, True)
    expected = float(audit_positions(loaded, obs.reshape(1, -1))[0])
    assert -1.0 <= expected <= 1.0
    tanh_model = train_ppo(
        frame,
        False,
        timesteps=64,
        window=32,
        action_mode="tanh_env",
        threads=1,
        collapse_log_every=32,
        seed=1,
    )
    assert tanh_model.crta_meta["action_mode"] == "tanh_env"
    blind = observation(frame.iloc[0][list(FEATURE_COLS)], frame.iloc[0][list(REGIMES)], 0.0, False)
    pos = float(audit_positions(tanh_model, blind.reshape(1, -1))[0])
    assert abs(pos) <= 1.0


def _tape(mode, asset, ablation, std=0.2, extreme=0.2):
    return {
        "mode": mode,
        "asset": asset,
        "timeframe": "1h",
        "std_c": std,
        "std_b": std,
        "extreme_c": extreme,
        "extreme_b": extreme,
        "ablation": ablation,
        "total_return": 9.0,
    }


def test_action_mode_choice_ignores_returns_and_uses_min_ablation():
    rows = []
    for asset in ("BTC", "ETH"):
        rows.append(_tape("tanh_env", asset, 0.2))
        rows.append(_tape("discrete5", asset, 0.4))
        rows.append(_tape("gaussian_clip", asset, 0.9, extreme=0.99))
    choice = choose_action_mode(rows)
    assert choice["mode"] == "discrete5"
    tied = []
    for asset in ("BTC", "ETH"):
        tied.append(_tape("tanh_env", asset, 0.3, std=0.1))
        tied.append(_tape("discrete5", asset, 0.3, std=0.4))
    assert choose_action_mode(tied)["mode"] == "discrete5"
    still = []
    for asset in ("BTC", "ETH"):
        still.append(_tape("tanh_env", asset, 0.3, std=0.2))
        still.append(_tape("discrete5", asset, 0.3, std=0.2))
    assert choose_action_mode(still)["mode"] == "tanh_env"
    assert choose_action_mode([_tape("tanh_env", "BTC", 0.01)])["mode"] == "none"

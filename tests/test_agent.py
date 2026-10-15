import numpy as np
import pandas as pd

from crta.agent import TradeEnv
from crta.data import FEATURE_COLS
from crta.eval import TURNOVER_COST
from crta.regime import REGIMES


def _frame(n=50):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0.0, 0.01, size=(n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
    for name in REGIMES:
        df[name] = 0.25
    df["next_ret"] = np.linspace(-0.01, 0.01, n)
    return df


def _run(env, seed):
    env.reset(seed=seed)
    start = env._i
    steps = 0
    done = False
    while not done:
        _obs, _r, done, _t, _info = env.step(np.array([0.5], dtype=np.float32))
        steps += 1
    return start, steps


def test_a_window_episode_is_one_stretch_of_the_series():
    env = TradeEnv(_frame(), True, window=10)
    starts = set()
    for seed in range(20):
        start, steps = _run(env, seed)
        assert steps == 10
        assert 0 <= start <= 40
        starts.add(start)
    assert len(starts) > 1


def test_no_window_replays_the_whole_series():
    env = TradeEnv(_frame(), False)
    start, steps = _run(env, 0)
    assert (start, steps) == (0, 50)


def test_reward_scale_multiplies_return_and_cost():
    frame = _frame()
    plain = TradeEnv(frame, True)
    scaled = TradeEnv(frame, True, reward_scale=100.0)
    plain.reset(seed=0)
    scaled.reset(seed=0)
    _o, r1, *_ = plain.step(np.array([1.0], dtype=np.float32))
    _o, r2, *_ = scaled.step(np.array([1.0], dtype=np.float32))
    assert np.isclose(r1, frame["next_ret"].iloc[0] - TURNOVER_COST)
    assert np.isclose(r2, 100.0 * r1)

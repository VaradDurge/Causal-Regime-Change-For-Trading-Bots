import numpy as np
import pandas as pd

from crta.agent import TradeEnv
from crta.data import FEATURE_COLS
from crta.regime import REGIMES


def _frame(n=80):
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(0.0, 0.01, size=(n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
    for name in REGIMES:
        df[name] = 0.05
    df["bull"] = 0.85
    df.loc[n // 2 :, ["bull", "bear", "calm", "crisis"]] = [0.05, 0.05, 0.05, 0.85]
    df["next_ret"] = np.linspace(-0.02, 0.02, n)
    df["close"] = 100 * np.exp(np.cumsum(df["log_ret"].to_numpy()))
    df["timestamp"] = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return df


def _stretch(env, seed):
    env.reset(seed=seed)
    start = env._i
    steps = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(np.array([0.25], dtype=np.float32))
        steps += 1
    return start, steps


def test_balanced_windows_are_one_contiguous_stretch():
    frame = _frame()
    for sampling in ("regime_balanced", "return_balanced"):
        env = TradeEnv(frame, True, window=12, window_sampling=sampling, regime_balance_beta=3.0)
        starts = set()
        for seed in range(12):
            start, steps = _stretch(env, seed)
            assert steps == 12
            assert 0 <= start <= len(frame) - 12
            starts.add(start)
        assert len(starts) > 1


def test_regime_balanced_prefers_bear_and_crisis_windows():
    frame = _frame(120)
    env = TradeEnv(frame, True, window=10, window_sampling="regime_balanced", regime_balance_beta=30.0)
    starts = [_stretch(env, seed)[0] for seed in range(40)]
    assert float(np.mean(starts)) > 60


def test_return_balanced_visits_each_tercile():
    frame = _frame(90)
    env = TradeEnv(frame, True, window=15, window_sampling="return_balanced")
    totals = []
    nxt = frame["next_ret"].to_numpy()
    for seed in range(45):
        start, _ = _stretch(env, seed)
        totals.append(float(nxt[start : start + 15].sum()))
    q1, q2 = np.quantile(
        [nxt[s : s + 15].sum() for s in range(len(frame) - 15 + 1)],
        [1 / 3, 2 / 3],
    )
    buckets = {
        "falling": any(v <= q1 + 1e-12 for v in totals),
        "flat": any(q1 < v <= q2 + 1e-12 for v in totals),
        "rising": any(v > q2 for v in totals),
    }
    assert all(buckets.values())


def test_twins_share_window_draws_including_perturbation():
    frame = _frame()
    kwargs = dict(
        window=16,
        window_sampling="regime_balanced",
        regime_balance_beta=3.0,
        regime_perturb_prob=0.7,
    )
    cond = TradeEnv(frame, True, **kwargs)
    blind = TradeEnv(frame, False, **kwargs)
    for seed in range(8):
        cond.reset(seed=seed)
        blind.reset(seed=seed)
        assert cond._i == blind._i
        assert cond._end == blind._end
        if cond._prob_override is not None:
            assert np.allclose(cond._prob_override, blind._prob_override)


def test_perturbation_keeps_the_real_return_and_uses_an_empirical_row():
    frame = _frame()
    env = TradeEnv(frame, True, window=10, regime_perturb_prob=1.0)
    env.reset(seed=1)
    assert env._prob_override is not None
    bank = frame[list(REGIMES)].to_numpy(dtype=float)
    for row in env._prob_override:
        assert np.any(np.all(np.abs(bank - row) <= 1e-8, axis=1))
    start = env._i
    env.step(np.array([1.0], dtype=np.float32))
    assert env._next[start] == frame["next_ret"].iloc[start]


def test_sign_flip_negates_log_returns_for_both_twins_and_is_off_by_default():
    frame = _frame(70)
    plain = TradeEnv(frame, True, window=20)
    plain.reset(seed=0)
    assert plain._use_flip is False
    kwargs = dict(window=20, sign_flip_prob=1.0, tracker_kwargs={"burn": 30, "stride": 10, "dwell": 4})
    cond = TradeEnv(frame, True, **kwargs)
    blind = TradeEnv(frame, False, **kwargs)
    cond.reset(seed=3)
    blind.reset(seed=3)
    assert cond._use_flip and blind._use_flip
    assert cond._i == blind._i
    start = cond._i
    assert np.isclose(cond._next[start], -frame["next_ret"].iloc[start])
    assert np.isclose(blind._next[start], -frame["next_ret"].iloc[start])
    # The conditioned twin sees regimes recomputed on the flipped series, not the original row.
    assert cond._probs.shape == (len(frame), 4)
    assert np.allclose(cond._probs.sum(axis=1), 1.0)

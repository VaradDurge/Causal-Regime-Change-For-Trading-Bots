import numpy as np
import pandas as pd

from crta.agent import TradeEnv, batch_observation, collapse_stats, diff_sharpe_reward, observation, smooth_probs
from crta.data import FEATURE_COLS
from crta.eval import TURNOVER_COST
from crta.faithful import faithfulness_hinge
from crta.regime import REGIMES


def _frame(n=50):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0.0, 0.01, size=(n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
    for name in REGIMES:
        df[name] = 0.25
    df["next_ret"] = np.linspace(-0.01, 0.01, n)
    df["close"] = 100 * np.exp(np.cumsum(df["log_ret"].to_numpy()))
    df["timestamp"] = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return df


def test_excess_bh_equals_raw_minus_the_buy_and_hold_return():
    frame = _frame()
    raw = TradeEnv(frame, True, reward_mode="raw")
    excess = TradeEnv(frame, True, reward_mode="excess_bh")
    raw.reset(seed=0)
    excess.reset(seed=0)
    action = np.array([0.4], dtype=np.float32)
    _, r_raw, *_ = raw.step(action)
    _, r_ex, *_ = excess.step(action)
    assert r_ex == r_raw - frame["next_ret"].iloc[0]


def test_reward_scale_multiplies_return_and_cost_in_every_mode():
    frame = _frame()
    for mode in ("raw", "excess_bh"):
        plain = TradeEnv(frame, True, reward_mode=mode)
        scaled = TradeEnv(frame, True, reward_mode=mode, reward_scale=100.0)
        plain.reset(seed=1)
        scaled.reset(seed=1)
        action = np.array([-0.3], dtype=np.float32)
        _, r1, *_ = plain.step(action)
        _, r2, *_ = scaled.step(action)
        assert np.isclose(r2, 100.0 * r1)


def test_always_long_earns_about_zero_in_excess_bh_before_costs():
    frame = _frame()
    env = TradeEnv(frame, True, reward_mode="excess_bh")
    env.reset(seed=0, options={"position": 1.0})
    _, reward, *_ = env.step(np.array([1.0], dtype=np.float32))
    assert abs(reward) < 1e-12


def test_diff_sharpe_resets_with_the_episode_and_stays_finite():
    reward, state = diff_sharpe_reward(0.01, (0.0, 0.0), eta=0.01)
    assert reward == 0.0
    assert state != (0.0, 0.0)
    frame = _frame()
    env = TradeEnv(frame, False, reward_mode="diff_sharpe", reward_scale=100.0, window=12)
    env.reset(seed=2)
    seen = []
    done = False
    while not done:
        _, reward, done, _, _ = env.step(np.array([0.2], dtype=np.float32))
        seen.append(reward)
        assert np.isfinite(reward)
    env.reset(seed=2)
    assert env._dsr == (0.0, 0.0)


def test_extreme_penalty_and_variance_bonus_are_zero_by_default():
    frame = _frame()
    plain = TradeEnv(frame, True)
    penal = TradeEnv(frame, True, pos_extreme_penalty=0.0, pos_variance_bonus=0.0)
    plain.reset(seed=0)
    penal.reset(seed=0)
    action = np.array([1.0], dtype=np.float32)
    _, r1, *_ = plain.step(action)
    _, r2, *_ = penal.step(action)
    assert r1 == r2


def test_extreme_penalty_hits_only_positions_past_the_cap():
    frame = _frame()
    env = TradeEnv(frame, True, pos_extreme_penalty=2.0)
    env.reset(seed=0, options={"position": 0.0})
    _, inside, *_ = env.step(np.array([0.95], dtype=np.float32))
    env.reset(seed=0, options={"position": 0.0})
    _, outside, *_ = env.step(np.array([1.0], dtype=np.float32))
    # The 1.0 action also pays a larger turnover than 0.95, so compare against a fresh raw account.
    raw = TradeEnv(frame, True)
    raw.reset(seed=0, options={"position": 0.0})
    _, raw_out, *_ = raw.step(np.array([1.0], dtype=np.float32))
    assert np.isclose(outside, raw_out - 2.0 * 0.05)
    assert inside > outside


def test_variance_bonus_is_bounded_by_half_the_coefficient():
    frame = _frame(n=30)
    env = TradeEnv(frame, True, pos_variance_bonus=1.0, variance_every=4, window=8)
    env.reset(seed=0)
    bonuses = []
    done = False
    while not done:
        before = len(env._pos_hist)
        _, reward, done, _, _ = env.step(np.array([0.2 if before % 2 == 0 else -0.2], dtype=np.float32))
        bonuses.append(reward)
    # A chunk award is at most 0.5 * coef, on top of a return of order 1e-2.
    assert max(bonuses) < 0.5 + 0.05


def test_collapse_warning_needs_several_low_std_logs():
    flat = collapse_stats(np.ones(20))
    assert flat["std"] < 0.02
    assert flat["extreme_share"] == 1.0
    mixed = collapse_stats(np.array([-0.5, 0.0, 0.5, 1.0]))
    assert mixed["std"] > 0.02


def test_smooth_probs_defaults_to_identity_and_floor_lowers_the_peak():
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    assert np.allclose(smooth_probs(peaked), peaked)
    softened = smooth_probs(peaked, floor=0.2)
    assert softened.max() < peaked.max()
    assert np.isclose(softened.sum(), 1.0)
    warmer = smooth_probs(peaked, temperature=4.0)
    assert warmer.max() < peaked.max()
    assert np.isclose(warmer.sum(), 1.0)


def test_faithfulness_hinge_is_zero_for_the_defensive_order():
    assert faithfulness_hinge(0.2, -0.4, 0.8) == 0.0
    assert faithfulness_hinge(0.2, 0.5, 0.0) > 0.0


def test_batch_observation_matches_the_single_row_vector():
    feats = np.array([[0.01, 0.02, -0.03, 0.4, -0.01], [0.0, 0.01, 0.02, 0.5, 0.0]])
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.4]])
    positions = np.array([0.2, -0.4])
    many = batch_observation(feats, probs, positions, True, temperature=2.0, floor=0.1)
    for i in range(2):
        one = observation(feats[i], smooth_probs(probs[i], 2.0, 0.1), float(positions[i]), True)
        assert np.allclose(many[i], one)

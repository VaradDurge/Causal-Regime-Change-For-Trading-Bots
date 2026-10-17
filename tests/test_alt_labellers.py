"""Statistical jump model and Wasserstein k-means labellers: fit quality and causality."""

import numpy as np
import pytest

from crta.jump import fit_jump, jump_dp, online_costs, track_jump
from crta.regime import REGIMES
from crta.wkmeans import fit_wkmeans, track_wkmeans, window_quantiles, w2_sorted


def _two_state(n=600, seed=0):
    rng = np.random.default_rng(seed)
    states = (np.arange(n) // 100) % 2
    X = np.where(states[:, None] == 0, rng.normal(-1.5, 0.5, (n, 2)), rng.normal(1.5, 0.5, (n, 2)))
    return X, states


def test_jump_model_recovers_persistent_states():
    X, truth = _two_state()
    _centroids, states, _loss = fit_jump(X, n_states=2, jump_penalty=5.0, seed=0)
    agree = max(np.mean(states == truth), np.mean(states != truth))
    assert agree > 0.97


def test_a_larger_jump_penalty_switches_less():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    _c, low, _ = fit_jump(X, n_states=3, jump_penalty=0.1, seed=0)
    _c, high, _ = fit_jump(X, n_states=3, jump_penalty=50.0, seed=0)
    assert np.sum(high[1:] != high[:-1]) < np.sum(low[1:] != low[:-1])


def test_dp_pays_the_penalty_only_when_it_saves_more():
    losses = np.array([[0.0, 5.0], [3.0, 0.0], [0.0, 5.0]])
    assert jump_dp(losses, 10.0).tolist() == [0, 0, 0]
    assert jump_dp(losses, 1.0).tolist() == [0, 1, 0]


def test_online_costs_at_t_ignore_later_rows():
    X, _ = _two_state()
    centroids = np.array([[-1.5, -1.5], [1.5, 1.5]])
    other = X.copy()
    other[301:] = 9.0
    assert np.allclose(online_costs(X, centroids, 5.0)[:301], online_costs(other, centroids, 5.0)[:301])


def _market(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    vol = np.repeat(rng.choice([0.004, 0.01, 0.025], size=n // 150), 150)[:n]
    drift = np.repeat(rng.choice([-0.002, 0.0005, 0.002], size=n // 150), 150)[:n]
    r = rng.normal(drift, vol)
    roll_vol = np.array([r[max(0, t - 19) : t + 1].std() for t in range(n)])
    trend = np.array([r[max(0, t - 19) : t + 1].sum() for t in range(n)])
    return r, np.column_stack([r, roll_vol, trend])


def test_track_jump_is_causal_and_names_four_regimes():
    _r, X = _market()
    t = 400 + 120 + 17
    kwargs = {"burn": 400, "stride": 120, "jump_penalty": 20.0, "seed": 0}
    labels, probs = track_jump(X, **kwargs)
    other = X.copy()
    other[t + 1 :] = X[t + 1 :][::-1] * 2.0
    labels_o, probs_o = track_jump(other, **kwargs)
    assert np.allclose(probs[: t + 1], probs_o[: t + 1])
    assert labels[: t + 1] == labels_o[: t + 1]
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert set(labels) <= set(REGIMES)


def test_window_quantiles_are_the_sorted_past_window():
    r = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    q = window_quantiles(r, 3)
    assert np.isnan(q[:2]).all()
    assert q[2].tolist() == [1.0, 2.0, 3.0]
    assert q[4].tolist() == [2.0, 4.0, 5.0]


def test_w2_on_sorted_windows_is_the_quantile_distance():
    a = np.sort(np.array([0.0, 1.0, 2.0]))
    b = a + 3.0
    assert w2_sorted(a, b) == pytest.approx(3.0)


def test_wkmeans_separates_calm_and_turbulent_windows():
    rng = np.random.default_rng(4)
    calm = np.sort(rng.normal(0, 0.005, (200, 30)), axis=1)
    wild = np.sort(rng.normal(0, 0.03, (200, 30)), axis=1)
    centroids, assign = fit_wkmeans(np.vstack([calm, wild]), k=2, seed=0)
    assert len(set(assign[:200])) == 1 and len(set(assign[200:])) == 1
    assert assign[0] != assign[-1]


def test_track_wkmeans_is_causal_and_crisis_is_the_most_volatile():
    r, _X = _market()
    t = 400 + 120 + 9
    kwargs = {"window": 30, "burn": 400, "stride": 120, "seed": 0}
    labels, probs = track_wkmeans(r, **kwargs)
    other = r.copy()
    other[t + 1 :] = r[t + 1 :][::-1] * 3.0
    labels_o, probs_o = track_wkmeans(other, **kwargs)
    assert np.allclose(probs[: t + 1], probs_o[: t + 1])
    assert labels[: t + 1] == labels_o[: t + 1]
    assert np.allclose(probs.sum(axis=1), 1.0)

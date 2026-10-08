import numpy as np
import pandas as pd

import crta.regime as regime
from crta.data import HMM_COLS, model_frame
from crta.regime import (
    REGIMES,
    assign_templates,
    component_params,
    decode,
    enforce_dwell,
    fit_regime,
    lock_states,
    track,
    update_templates,
    w2_diag,
)


def test_dwell_merges_a_short_run():
    labels = ["calm"] * 8 + ["crisis"] * 2 + ["calm"] * 8
    assert enforce_dwell(labels, min_dwell=6) == ["calm"] * 18


def test_dwell_keeps_the_opening_run():
    labels = ["bull"] * 2 + ["bear"] * 8
    assert enforce_dwell(labels, min_dwell=6)[:2] == ["bull", "bull"]


def test_lock_states_sorts_return_then_vol():
    means = np.array(
        [
            [0.02, 0.05, 0.1],
            [0.01, 0.01, 0.0],
            [-0.01, 0.01, 0.0],
            [-0.03, 0.08, -0.2],
        ]
    )
    mapping = lock_states(means)
    assert mapping == {2: "bear", 3: "crisis", 1: "calm", 0: "bull"}


def _runs(labels):
    lengths = []
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        lengths.append(j - i)
        i = j
    return lengths


def test_crash_half_is_not_calm():
    rng = np.random.default_rng(42)
    rets = np.concatenate([rng.normal(0.002, 0.002, 400), rng.normal(-0.02, 0.04, 400)])
    close = 100 * np.exp(np.cumsum(rets))
    raw = pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=len(close), freq="4h", tz="UTC"),
        }
    )
    frame = model_frame(raw)
    fitted = fit_regime(frame[list(HMM_COLS)].to_numpy())
    labels, _probs = decode(fitted, frame[list(HMM_COLS)].to_numpy())
    assert all(n >= 6 for n in _runs(labels)[1:])
    tail = labels[int(len(labels) * 0.65) :]
    assert tail.count("calm") / len(tail) < 0.25
    assert tail.count("bull") / len(tail) < 0.25


def test_w2_diag_is_zero_for_a_copy_and_25_for_a_3_4_shift():
    origin = np.zeros(3)
    scale = np.ones(3)
    assert w2_diag(origin, scale, origin, scale) == 0.0
    # ||(3, 4, 0)||^2 = 25, and the spreads match so they add nothing.
    assert w2_diag(origin, scale, np.array([3.0, 4.0, 0.0]), scale) == 25.0


def test_swapped_components_keep_their_template_names():
    rng = np.random.default_rng(0)
    rets = np.concatenate([rng.normal(0.003, 0.002, 300), rng.normal(-0.02, 0.04, 300)])
    close = 100 * np.exp(np.cumsum(rets))
    raw = pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=len(close), freq="4h", tz="UTC"),
        }
    )
    fitted = fit_regime(model_frame(raw)[list(HMM_COLS)].to_numpy())
    mu, sd = component_params(fitted)
    assert sd.shape == mu.shape
    index = {name: i for i, name in enumerate(REGIMES)}
    template_mu = np.zeros_like(mu)
    template_sd = np.zeros_like(sd)
    for k, name in lock_states(mu).items():
        template_mu[index[name]] = mu[k]
        template_sd[index[name]] = sd[k]
    order = np.array([1, 0, 2, 3])
    assigned = assign_templates(mu[order], sd[order], template_mu, template_sd)
    expected = [index[lock_states(mu)[int(k)]] for k in order]
    assert list(assigned) == expected


def test_smoothing_leaves_an_unmatched_template():
    template_mu = np.array([[0.0, 0.0], [10.0, 10.0]])
    template_sd = np.ones((2, 2))
    mu = np.array([[1.0, 0.0]])
    sd = np.ones((1, 2))
    new_mu, new_sd = update_templates(
        template_mu,
        template_sd,
        mu,
        sd,
        np.ones((4, 1)),
        np.array([0]),
        eta=0.2,
    )
    assert np.allclose(new_mu[0], np.array([0.2, 0.0]))
    assert np.allclose(new_mu[1], template_mu[1])
    assert np.allclose(new_sd, template_sd)


def test_later_bars_do_not_change_a_closed_window():
    rng = np.random.default_rng(1)
    rets = np.concatenate([rng.normal(0.002, 0.004, 160), rng.normal(-0.01, 0.02, 160)])
    close = 100 * np.exp(np.cumsum(rets))
    raw = pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=len(close), freq="4h", tz="UTC"),
        }
    )
    X = model_frame(raw)[list(HMM_COLS)].to_numpy()
    burn, stride = 80, 40
    cut = burn + stride
    short_labels, short_probs = track(X[:cut], burn=burn, stride=stride)
    long_labels, long_probs = track(X, burn=burn, stride=stride)
    assert short_labels == long_labels[:cut]
    assert np.allclose(short_probs, long_probs[:cut])


def test_a_dead_state_still_decodes(monkeypatch):
    real = regime.GaussianHMM.fit

    def fit_then_kill_a_row(self, X, lengths=None):
        real(self, X, lengths)
        self.transmat_ = np.array(self.transmat_, dtype=float, copy=True)
        self.transmat_[0] = 0.0
        return self

    monkeypatch.setattr(regime.GaussianHMM, "fit", fit_then_kill_a_row)
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0, 0.01, 120)
    close = 100 * np.exp(np.cumsum(rets))
    raw = pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=len(close), freq="4h", tz="UTC"),
        }
    )
    X = model_frame(raw)[list(HMM_COLS)].to_numpy()
    labels, probs = track(X, burn=40, stride=20)
    assert len(labels) == len(X)
    assert np.allclose(probs.sum(axis=1), 1.0)

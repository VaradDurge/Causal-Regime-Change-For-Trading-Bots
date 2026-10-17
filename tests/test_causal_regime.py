"""Regime probabilities must be filtered: bar t may not depend on any row after t."""

import numpy as np
import pandas as pd

from crta.data import HMM_COLS, model_frame
from crta.regime import (
    LEGACY_V1,
    _fit_hmm,
    _templates_from_lock,
    _window_labels,
    component_params,
    decode,
    feature_scale,
    fit_regime,
    label_span,
    track,
    track_v2,
)


def _X(n=360, seed=3):
    rng = np.random.default_rng(seed)
    rets = np.concatenate(
        [
            rng.normal(0.003, 0.003, n // 4),
            rng.normal(-0.004, 0.006, n // 4),
            rng.normal(0.0002, 0.002, n // 4),
            rng.normal(-0.01, 0.025, n // 4),
        ]
    )
    close = 100 * np.exp(np.cumsum(rets))
    raw = pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=len(close), freq="4h", tz="UTC"),
        }
    )
    return model_frame(raw)[list(HMM_COLS)].to_numpy()


def _scrambled_after(X, t, seed=9):
    """Same rows up to and including t, unrelated rows after it."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    out[t + 1 :] = X[t + 1 :][rng.permutation(len(X) - t - 1)] * 3.0
    return out


def test_window_probabilities_ignore_later_rows_of_the_same_window():
    X = _X()
    fitted = _fit_hmm(X[:200], 42)
    mu, sd = component_params(fitted)
    _tmu, _tsd, assign = _templates_from_lock(mu, sd)
    window = X[200:242]
    t = 10
    labels, probs, _ = _window_labels(fitted, window, assign, None)
    other_labels, other_probs, _ = _window_labels(fitted, _scrambled_after(window, t), assign, None)
    assert np.allclose(probs[: t + 1], other_probs[: t + 1])
    assert labels[: t + 1] == other_labels[: t + 1]


def test_smoothed_posterior_is_kept_only_to_show_the_leak():
    X = _X()
    fitted = _fit_hmm(X[:200], 42)
    mu, sd = component_params(fitted)
    _tmu, _tsd, assign = _templates_from_lock(mu, sd)
    window = X[200:242]
    t = 10
    _l, probs, _ = _window_labels(fitted, window, assign, None, posterior="smoothed")
    _l, other, _ = _window_labels(fitted, _scrambled_after(window, t), assign, None, posterior="smoothed")
    # Near one-hot rows move by tiny amounts, but any movement means bar t read a later row.
    assert not np.array_equal(probs[: t + 1], other[: t + 1])


def test_history_warms_the_filter_without_using_the_future():
    X = _X()
    fitted = _fit_hmm(X[:200], 42)
    mu, sd = component_params(fitted)
    _tmu, _tsd, assign = _templates_from_lock(mu, sd)
    window = X[200:242]
    _l, cold, _ = _window_labels(fitted, window, assign, None)
    _l, warm, _ = _window_labels(fitted, window, assign, None, history=X[120:200])
    assert cold.shape == warm.shape
    assert np.allclose(warm.sum(axis=1), 1.0)
    t = 5
    _l, warm_other, _ = _window_labels(fitted, _scrambled_after(window, t), assign, None, history=X[120:200])
    assert np.allclose(warm[: t + 1], warm_other[: t + 1])


def test_tracked_probabilities_at_t_ignore_every_later_row():
    X = _X()
    burn, stride = 120, 30
    for t in (burn + 3, burn + stride + 17, len(X) - 20):
        labels, probs = track(X, burn=burn, stride=stride, dwell=6)
        other_labels, other_probs = track(_scrambled_after(X, t), burn=burn, stride=stride, dwell=6)
        assert np.allclose(probs[: t + 1], other_probs[: t + 1]), t
        assert labels[: t + 1] == other_labels[: t + 1], t


def test_tracker_v2_probabilities_at_t_ignore_every_later_row():
    X = _X()
    t = 120 + 30 + 11
    labels, probs = track_v2(X, burn=120, stride=30, dwell=6, death_after=3)
    other_labels, other_probs = track_v2(_scrambled_after(X, t), burn=120, stride=30, dwell=6, death_after=3)
    assert np.allclose(probs[: t + 1], other_probs[: t + 1])
    assert labels[: t + 1] == other_labels[: t + 1]


def test_label_span_ignores_later_rows():
    X = _X()
    t = 230
    _n, probs = label_span(X, 200, len(X))
    _n, other = label_span(_scrambled_after(X, t), 200, len(X))
    assert np.allclose(probs[: t - 200 + 1], other[: t - 200 + 1])


def test_decode_ignores_later_rows():
    X = _X()
    fitted = fit_regime(X[:200])
    t = 260
    _n, probs = decode(fitted, X)
    _n, other = decode(fitted, _scrambled_after(X, t))
    assert np.allclose(probs[: t + 1], other[: t + 1])


def test_feature_scale_is_the_burn_in_std_with_a_floor():
    X = np.column_stack([np.arange(10.0), np.full(10, 3.0)])
    scale = feature_scale(X)
    assert np.isclose(scale[0], np.arange(10.0).std())
    assert scale[1] == 1.0


def test_scaled_matching_does_not_care_about_feature_units():
    X = _X()
    units = np.array([1.0, 1.0, 50.0])
    labels, probs = track_v2(X, burn=120, stride=30, dwell=6, death_after=3)
    again, again_probs = track_v2(X * units, burn=120, stride=30, dwell=6, death_after=3)
    assert labels == again
    assert np.allclose(probs, again_probs, atol=1e-6)


def test_legacy_settings_name_the_published_pipeline():
    assert LEGACY_V1 == {"burn": 252, "posterior": "smoothed", "scale": "raw", "warm": 0}

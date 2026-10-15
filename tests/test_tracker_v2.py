import numpy as np
import pandas as pd

from crta.data import HMM_COLS, model_frame
from crta.regime import (
    REGIMES,
    apply_birth_death,
    assign_matching,
    enforce_dwell_online,
    forward_filter,
    rebirth_source,
    track,
    track_v2,
)


def _series(n=240, seed=1):
    rng = np.random.default_rng(seed)
    rets = np.concatenate(
        [
            rng.normal(0.004, 0.002, n // 4),
            rng.normal(-0.004, 0.004, n // 4),
            rng.normal(0.0002, 0.001, n // 4),
            rng.normal(-0.02, 0.03, n // 4),
        ]
    )
    close = 100 * np.exp(np.cumsum(rets))
    raw = pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=len(close), freq="4h", tz="UTC"),
        }
    )
    return model_frame(raw)


def test_forward_filter_ignores_future_bars():
    class _Hmm:
        startprob_ = np.array([0.5, 0.5])
        transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
        means_ = np.array([[0.0], [2.0]])
        covars_ = np.array([[0.4], [0.4]])

    class _Model:
        hmm = _Hmm()
        mu = np.array([0.0])
        sd = np.array([1.0])

    X = np.array([[0.0], [0.2], [-0.1], [2.2], [2.4], [0.1]])
    base = forward_filter(_Model(), X)
    altered = X.copy()
    altered[3:] = -1.5
    changed = forward_filter(_Model(), altered)
    assert np.allclose(base[:3], changed[:3])
    assert not np.allclose(base[3], changed[3])
    assert np.allclose(base.sum(axis=1), 1.0)


def test_online_dwell_does_not_rewrite_past_labels():
    labels = ["calm"] * 8 + ["crisis"] * 12 + ["bull"] * 3 + ["bear"] * 10
    full = enforce_dwell_online(labels, min_dwell=4, prev="calm")
    for t in range(1, len(labels) + 1):
        assert enforce_dwell_online(labels[:t], min_dwell=4, prev="calm") == full[:t]


def test_a_closed_v2_window_is_not_relabeled():
    X = _series(220)[list(HMM_COLS)].to_numpy()
    burn, stride = 40, 20
    cut = burn + 2 * stride
    short_labels, short_probs = track_v2(X[:cut], burn=burn, stride=stride, dwell=6, death_after=3)
    long_labels, long_probs = track_v2(X, burn=burn, stride=stride, dwell=6, death_after=3)
    assert short_labels == long_labels[:cut]
    assert np.allclose(short_probs, long_probs[:cut])


def test_template_updates_use_only_the_past_and_the_open_window():
    X = _series(160)[list(HMM_COLS)].to_numpy()
    diag = {}
    track_v2(X, burn=40, stride=15, dwell=4, death_after=2, diagnostics=diag)
    refits = [event for event in diag["events"] if event["kind"] == "refit"]
    assert refits
    for event in refits:
        assert event["fit_end"] == event["bar"]
        assert event["window_end"] > event["bar"]
        assert event["fit_end"] <= event["window_end"]


def test_starved_template_is_reborn_and_logged():
    mu = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.02, 0.4, 0.1],
            [-0.01, 0.2, -0.1],
            [-0.04, 0.9, -0.3],
        ]
    )
    sd = np.ones_like(mu) * 0.2
    template_mu = mu.copy()
    template_sd = sd.copy()
    template_mu[0] = np.array([-5.0, -5.0, -5.0])
    assign = np.array([1, 1, 2, 3])
    events = []
    new_mu, _new_sd, empty, reborn = apply_birth_death(
        template_mu,
        template_sd,
        mu,
        sd,
        assign,
        np.zeros(4, dtype=int),
        death_after=1,
        events=events,
        refit_index=7,
        bar=400,
        date="2024-01-01",
    )
    assert reborn
    assert events[0]["kind"] == "death_birth"
    assert events[0]["template"] == "bull"
    assert events[0]["refit_index"] == 7
    assert events[0]["date"] == "2024-01-01"
    assert not np.allclose(new_mu[0], template_mu[0])
    assert empty[0] == 0
    src_mu, _src_sd, how, _comp = rebirth_source(mu, sd, template_mu, template_sd, 0)
    assert how == "farthest"
    assert np.allclose(new_mu[0], src_mu)


def test_hungarian_assignment_is_one_to_one_and_threshold_can_merge():
    mu = np.eye(4)
    sd = np.ones((4, 4))
    assign = assign_matching(mu, sd, mu, sd, matching="hungarian")
    assert sorted(assign.tolist()) == [0, 1, 2, 3]
    historical = [0.01] * 10
    far = mu.copy()
    far[3] = np.array([20.0, 20.0, 20.0, 20.0])
    merged = assign_matching(
        far, sd, mu, sd, matching="hungarian_with_threshold", historical=historical, quantile=0.5
    )
    assert len(set(merged.tolist())) < 4 or int(merged[3]) != 3


def test_diagnostics_do_not_change_v1_labels():
    X = _series(120)[list(HMM_COLS)].to_numpy()
    labels, probs = track(X, burn=40, stride=20, dwell=6)
    diag = {}
    again, again_probs = track(X, burn=40, stride=20, dwell=6, diagnostics=diag)
    assert again == labels
    assert np.allclose(again_probs, probs)
    assert diag["matched_costs"]
    assert len(diag["component_mu"]) == len(diag["matched_costs"])


def test_v1_track_is_unchanged_by_the_v2_entry_point():
    X = _series(120)[list(HMM_COLS)].to_numpy()
    labels, probs = track(X, burn=40, stride=20, dwell=6)
    assert len(labels) == len(X)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert set(labels) <= set(REGIMES)


def test_tracker_v2_bull_share_after_burn_in_on_history():
    path = "paper/results/tf/btc_usdt_1h_history.parquet"
    raw = pd.read_parquet(path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    # Long enough for the default death counter (20 refits) and one window after it.
    frame = model_frame(raw.iloc[:6000].reset_index(drop=True))
    X = frame[list(HMM_COLS)].to_numpy()
    labels, probs = track_v2(X, burn=252, stride=42, dwell=30, death_after=20)
    after = labels[252:]
    share = after.count("bull") / len(after)
    assert share > 0.0, share
    assert np.allclose(probs.sum(axis=1), 1.0)

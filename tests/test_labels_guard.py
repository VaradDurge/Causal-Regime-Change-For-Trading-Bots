"""Template names that follow their moments, and a drawdown input that separates bear rallies."""

import numpy as np
import pandas as pd

from crta.data import HMM_COLS, drawdown, model_frame
from crta.regime import REGIMES, apply_birth_death, lock_states, rename_by_moments, track_v2


def test_drawdown_is_the_distance_below_the_running_high():
    close = pd.Series([1.0, 2.0, 1.5, 3.0, 2.4])
    assert np.allclose(drawdown(close, window=2), [0.0, 0.0, -0.25, 0.0, -0.2])
    assert np.allclose(drawdown(close, window=10), [0.0, 0.0, -0.25, 0.0, -0.2])


def test_drawdown_at_t_ignores_later_closes():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    other = close.copy()
    other.iloc[201:] = other.iloc[201:] * 3.0
    assert np.allclose(drawdown(close, 90)[:201], drawdown(other, 90)[:201])


def test_templates_are_renamed_to_match_their_moments():
    # Rows are bull, bear, calm, crisis. The bear and crisis templates have swapped volatility.
    mu = np.array([[0.02, 0.5, 0.0], [-0.01, 0.9, 0.0], [0.01, 0.2, 0.0], [-0.02, 0.3, 0.0]])
    sd = np.arange(12, dtype=float).reshape(4, 3)
    new_mu, new_sd, perm = rename_by_moments(mu, sd)
    bear, crisis = REGIMES.index("bear"), REGIMES.index("crisis")
    assert np.allclose(new_mu[bear], mu[crisis])
    assert np.allclose(new_mu[crisis], mu[bear])
    assert np.allclose(new_sd[crisis], sd[bear])
    assert np.allclose(new_mu[[0, 2]], mu[[0, 2]])
    assert perm.tolist() == [0, 3, 2, 1]


def test_moment_rebirth_takes_the_component_the_moments_name():
    mu = np.array([[0.03, 0.6, 0.1], [-0.01, 0.2, 0.0], [0.01, 0.1, 0.0], [-0.05, 0.9, -0.2]])
    sd = np.ones_like(mu)
    template_mu = mu.copy()
    template_mu[REGIMES.index("bull")] = np.array([9.0, 9.0, 9.0])
    events = []
    new_mu, _sd, _empty, reborn = apply_birth_death(
        template_mu, sd.copy(), mu, sd, np.array([1, 1, 2, 3]), np.zeros(4, dtype=int), 1, events, 0, 0, source="moment"
    )
    named_bull = [k for k, name in lock_states(mu).items() if name == "bull"][0]
    assert reborn
    assert np.allclose(new_mu[REGIMES.index("bull")], mu[named_bull])
    assert events[0]["source"] == "moment"


def _X(n=420, seed=5, extra=False):
    rng = np.random.default_rng(seed)
    rets = np.concatenate(
        [
            rng.normal(0.004, 0.004, n // 4),
            rng.normal(-0.004, 0.008, n // 4),
            rng.normal(0.0005, 0.002, n // 4),
            rng.normal(-0.012, 0.03, n // 4),
        ]
    )
    raw = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rets)), "timestamp": pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")}
    )
    frame = model_frame(raw)
    cols = list(HMM_COLS) + (["drawdown"] if extra else [])
    return frame[cols].to_numpy()


def test_guarded_templates_keep_their_moment_order_at_every_refit():
    diag = {}
    track_v2(_X(), burn=120, stride=30, dwell=6, death_after=3, semantic_guard=True, diagnostics=diag)
    assert diag["template_mu"]
    for tmu in diag["template_mu"]:
        assert lock_states(tmu) == {g: name for g, name in enumerate(REGIMES)}


def test_guarded_tracker_with_drawdown_input_stays_causal():
    X = _X(extra=True)
    t = 120 + 30 + 13
    rng = np.random.default_rng(9)
    other = X.copy()
    other[t + 1 :] = X[t + 1 :][rng.permutation(len(X) - t - 1)] * 2.0
    kwargs = {"burn": 120, "stride": 30, "dwell": 6, "death_after": 3, "semantic_guard": True}
    labels, probs = track_v2(X, **kwargs)
    other_labels, other_probs = track_v2(other, **kwargs)
    assert np.allclose(probs[: t + 1], other_probs[: t + 1])
    assert labels[: t + 1] == other_labels[: t + 1]

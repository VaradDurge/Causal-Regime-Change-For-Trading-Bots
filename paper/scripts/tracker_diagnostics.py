"""Instrumented replay of crta.regime.track.

Re-runs the causal Wasserstein template tracker step by step, logs every refit
(component parameters, W2 cost matrix, template state) and, on the same refits,
the alternative used before the change: re-naming states by moment sorting on
every refit. Asserts that the replayed labels equal crta.regime.track exactly.

Writes paper/results/tracker.npz and paper/results/tracker.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crta.data import HMM_COLS, RAW_PATH, model_frame
from crta.regime import (
    BURN,
    ETA,
    MIN_DWELL,
    N_STATES,
    REGIMES,
    STRIDE,
    _fit_hmm,
    _templates_from_lock,
    _window_labels,
    assign_templates,
    component_params,
    lock_states,
    track,
    update_templates,
)

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "results"
PERIODS = 6 * 365


def w2_matrix(mu, sd, tmu, tsd):
    return np.sum((mu[:, None, :] - tmu[None]) ** 2 + (sd[:, None, :] - tsd[None]) ** 2, axis=2)


def moment_assign(mu):
    index = {name: i for i, name in enumerate(REGIMES)}
    mapping = lock_states(mu)
    return np.array([index[mapping[k]] for k in range(len(mu))], dtype=int)


def runs(labels):
    out = []
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        out.append(j - i)
        i = j
    return out


def label_stats(labels, frame):
    labels = np.asarray(labels)
    n = len(labels)
    switches = int(np.sum(labels[1:] != labels[:-1]))
    years = n / PERIODS
    lengths = runs(list(labels))
    per = {}
    for name in REGIMES:
        m = labels == name
        per[name] = {
            "share": float(m.mean()),
            "mean_ret_bps": float(frame["log_ret"].to_numpy()[m].mean() * 1e4) if m.any() else None,
            "mean_vol_pct": float(frame["vol_20"].to_numpy()[m].mean() * 100) if m.any() else None,
            "mean_trend_pct": float(frame["trend"].to_numpy()[m].mean() * 100) if m.any() else None,
        }
    return {
        "switches": switches,
        "switches_per_year": switches / years,
        "mean_run_bars": float(np.mean(lengths)),
        "median_run_bars": float(np.median(lengths)),
        "per_regime": per,
    }


def identity_jump(times_mu, times_sd, assigns, probs_list):
    """Mean W2 jump, refit to refit, of the probability-weighted component carrying each name."""
    jumps = {name: [] for name in REGIMES}
    prev = {name: None for name in REGIMES}
    for mu, sd, a, w in zip(times_mu, times_sd, assigns, probs_list):
        for g, name in enumerate(REGIMES):
            members = np.flatnonzero(a == g)
            if len(members) == 0:
                prev[name] = None
                continue
            weight = w[members]
            weight = weight / weight.sum() if weight.sum() > 1e-12 else np.full(len(members), 1 / len(members))
            m = weight @ mu[members]
            s = weight @ sd[members]
            if prev[name] is not None:
                pm, ps = prev[name]
                jumps[name].append(float(np.sum((m - pm) ** 2 + (s - ps) ** 2)))
            prev[name] = (m, s)
    return {k: (float(np.mean(v)) if v else None) for k, v in jumps.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(RAW_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    frame = model_frame(raw)
    X = frame[list(HMM_COLS)].to_numpy(dtype=float)
    n = len(X)
    seed = 42
    t0 = time.time()

    # Replay of track() with logging, plus a moment-sort labelling on the same fits.
    labels_w2: list[str] = []
    labels_ms: list[str] = []
    probs_w2 = np.zeros((n, len(REGIMES)))
    fitted = _fit_hmm(X[:BURN], seed)
    mu, sd = component_params(fitted)
    tmu, tsd, assign = _templates_from_lock(mu, sd)
    wl, wp, proba = _window_labels(fitted, X[:BURN], assign, None)
    labels_w2.extend(wl)
    labels_ms.extend(wl)
    probs_w2[:BURN] = wp

    log_t, log_mu, log_sd, log_assign, log_ms, log_tmu, log_tsd, log_cost, log_w = [], [], [], [], [], [], [], [], []
    t = BURN
    while t < n:
        end = min(t + STRIDE, n)
        fitted = _fit_hmm(X[:t], seed)
        mu, sd = component_params(fitted)
        cost = w2_matrix(mu, sd, tmu, tsd)
        assign = assign_templates(mu, sd, tmu, tsd)
        ms = moment_assign(mu)
        log_t.append(t)
        log_mu.append(mu)
        log_sd.append(sd)
        log_assign.append(assign)
        log_ms.append(ms)
        log_tmu.append(tmu.copy())
        log_tsd.append(tsd.copy())
        log_cost.append(cost)
        wl, wp, proba = _window_labels(fitted, X[t:end], assign, labels_w2[-1])
        log_w.append(proba.sum(axis=0))
        labels_w2.extend(wl)
        probs_w2[t:end] = wp
        wl_ms, _wp_ms, _ = _window_labels(fitted, X[t:end], ms, labels_ms[-1])
        labels_ms.extend(wl_ms)
        tmu, tsd = update_templates(tmu, tsd, mu, sd, proba, assign, ETA)
        t = end
    replay_s = time.time() - t0

    t1 = time.time()
    ref_labels, ref_probs = track(X)
    track_s = time.time() - t1
    assert ref_labels == labels_w2, "replay diverged from crta.regime.track"
    assert np.allclose(ref_probs, probs_w2)

    A = np.array(log_assign)
    M = np.array(log_ms)
    refits = len(A)
    differ_refit = int(np.sum(np.any(A != M, axis=1)))
    differ_comp = float(np.mean(A != M))
    non_injective = int(np.sum([len(set(a.tolist())) < N_STATES for a in A]))
    empty_counts = {name: int(np.sum([g not in set(a.tolist()) for a in A])) for g, name in enumerate(REGIMES)}
    lab_w2 = np.array(labels_w2)
    lab_ms = np.array(labels_ms)
    after = slice(BURN, n)
    agree = float(np.mean(lab_w2[after] == lab_ms[after]))
    matched_cost = np.array([c[np.arange(N_STATES), a].mean() for c, a in zip(log_cost, A)])
    ms_cost = np.array([c[np.arange(N_STATES), m].mean() for c, m in zip(log_cost, M)])

    sub = frame.iloc[after]
    stats = {
        "n_bars": n,
        "burn": BURN,
        "stride": STRIDE,
        "eta": ETA,
        "min_dwell": MIN_DWELL,
        "refits": refits,
        "replay_seconds": replay_s,
        "track_seconds": track_s,
        "refits_where_w2_differs_from_moment_sort": differ_refit,
        "component_disagreement_rate": differ_comp,
        "refits_non_injective": non_injective,
        "refits_template_unused": empty_counts,
        "label_agreement_after_burn": agree,
        "mean_matched_w2_cost": float(matched_cost.mean()),
        "mean_moment_sort_w2_cost": float(ms_cost.mean()),
        "w2": label_stats(lab_w2[after], sub),
        "moment_sort": label_stats(lab_ms[after], sub),
        "identity_jump_w2": identity_jump(log_mu, log_sd, A, log_w),
        "identity_jump_moment_sort": identity_jump(log_mu, log_sd, M, log_w),
        "start": str(frame["timestamp"].iloc[0]),
        "end": str(frame["timestamp"].iloc[-1]),
    }
    (OUT / "tracker.json").write_text(json.dumps(stats, indent=2))
    np.savez(
        OUT / "tracker.npz",
        t=np.array(log_t),
        mu=np.array(log_mu),
        sd=np.array(log_sd),
        assign=A,
        moment=M,
        tmu=np.array(log_tmu),
        tsd=np.array(log_tsd),
        cost=np.array(log_cost),
        weight=np.array(log_w),
        labels_w2=lab_w2,
        labels_ms=lab_ms,
        probs_w2=probs_w2,
        timestamp=frame["timestamp"].astype("int64").to_numpy(),
        close=frame["close"].to_numpy(),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

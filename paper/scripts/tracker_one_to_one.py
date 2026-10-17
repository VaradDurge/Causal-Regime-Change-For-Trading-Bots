"""Diagnostic only (not part of the crta pipeline): the same causal tracker with a
one-to-one (Hungarian) component-to-template assignment instead of independent
nearest-template matching. Used to test whether template starvation comes from
the non-injective matching rule. No PPO training is done on these labels.

Writes paper/results/tracker_one_to_one.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from crta.data import HMM_COLS, RAW_PATH, model_frame
from crta.regime import LEGACY_V1, ETA, REGIMES, STRIDE, _fit_hmm, _templates_from_lock, _window_labels, component_params, update_templates
from tracker_diagnostics import label_stats

# Reproduces the published diagnostic, smoothed probabilities included. Not for new results.
BURN = LEGACY_V1["burn"]
PAPER = Path(__file__).resolve().parents[1]


def main() -> int:
    raw = pd.read_parquet(RAW_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    frame = model_frame(raw)
    X = frame[list(HMM_COLS)].to_numpy(dtype=float)
    n = len(X)
    fitted = _fit_hmm(X[:BURN], 42)
    mu, sd = component_params(fitted)
    tmu, tsd, assign = _templates_from_lock(mu, sd)
    labels, _p, _ = _window_labels(fitted, X[:BURN], assign, None, posterior="smoothed")
    t = BURN
    while t < n:
        end = min(t + STRIDE, n)
        fitted = _fit_hmm(X[:t], 42)
        mu, sd = component_params(fitted)
        cost = np.sum((mu[:, None] - tmu[None]) ** 2 + (sd[:, None] - tsd[None]) ** 2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        assign = np.empty(len(mu), dtype=int)
        assign[rows] = cols
        wl, _wp, proba = _window_labels(fitted, X[t:end], assign, labels[-1], posterior="smoothed")
        labels.extend(wl)
        tmu, tsd = update_templates(tmu, tsd, mu, sd, proba, assign, ETA)
        t = end
    lab = np.array(labels)
    stats = label_stats(lab[BURN:], frame.iloc[BURN:])
    (PAPER / "results" / "tracker_one_to_one.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Label statistics shared by the v1 and v2 trackers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crta.regime import REGIMES


def _runs(labels: list[str]) -> list[int]:
    lengths = []
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        lengths.append(j - i)
        i = j
    return lengths


def identity_jumps(diag: dict) -> dict[str, float | None]:
    """Mean W2 jump of the responsibility-weighted component carrying each name."""
    jumps = {name: [] for name in REGIMES}
    prev = {name: None for name in REGIMES}
    mus = diag.get("component_mu") or []
    sds = diag.get("component_sd") or []
    assigns = diag.get("assign") or []
    weights = diag.get("responsibility") or []
    for mu, sd, assign, weight in zip(mus, sds, assigns, weights):
        weight = np.asarray(weight, dtype=float)
        for g, name in enumerate(REGIMES):
            members = np.flatnonzero(np.asarray(assign) == g)
            if len(members) == 0:
                prev[name] = None
                continue
            w = weight[members]
            w = w / w.sum() if w.sum() > 1e-12 else np.full(len(members), 1.0 / len(members))
            mean = w @ np.asarray(mu)[members]
            scale = w @ np.asarray(sd)[members]
            if prev[name] is not None:
                old_mean, old_scale = prev[name]
                jumps[name].append(float(np.sum((mean - old_mean) ** 2 + (scale - old_scale) ** 2)))
            prev[name] = (mean, scale)
    return {name: (float(np.mean(vals)) if vals else None) for name, vals in jumps.items()}


def label_statistics(labels, frame: pd.DataFrame, periods: float, diag: dict | None = None) -> dict:
    labels = list(labels)
    n = len(labels)
    switches = int(sum(labels[i] != labels[i - 1] for i in range(1, n)))
    years = n / periods if periods else float("nan")
    lengths = _runs(labels)
    per = {}
    ret = frame["log_ret"].to_numpy(dtype=float) if "log_ret" in frame.columns else np.full(n, np.nan)
    vol = frame["vol_20"].to_numpy(dtype=float) if "vol_20" in frame.columns else np.full(n, np.nan)
    arr = np.asarray(labels)
    for name in REGIMES:
        mask = arr == name
        per[name] = {
            "share": float(mask.mean()) if n else None,
            "mean_return": float(ret[mask].mean()) if mask.any() else None,
            "volatility": float(vol[mask].mean()) if mask.any() else None,
        }
    costs = [] if diag is None else diag.get("matched_costs") or []
    jumps = identity_jumps(diag or {})
    finite = [value for value in jumps.values() if value is not None]
    return {
        "n": n,
        "switches_per_year": float(switches / years) if years else None,
        "median_run_bars": float(np.median(lengths)) if lengths else None,
        "mean_matched_w2": float(np.mean(costs)) if costs else None,
        "identity_jump": float(np.mean(finite)) if finite else None,
        "identity_jump_per_regime": jumps,
        "per_regime": per,
    }

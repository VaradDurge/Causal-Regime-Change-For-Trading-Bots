"""Regime label quality against hindsight 20% bull/bear dating, on the frozen 4h frame.

Every labeller is scored on the same bars, the out-of-fold span from bar 1,080 on, so
no labeller is judged on its own burn-in. The labellers isolate one change each:

  v1_published       tracker v1 as published (smoothed, raw units, burn-in 252)
  v2_published       tracker v2 as published (filtered, raw units, burn-in 252, no warm-up)
  v2_scaled          v2 with W2 on burn-in-scaled features, burn-in 252
  v2_burn1080        v2 with raw units and the 1,080-bar burn-in
  v2_fixed           v2 with both, plus the filter warm-up: the labeller the pipeline uses

Matched W2 costs are in each labeller's own units (raw or burn-in-scaled), so they are
comparable only within the same unit.

Usage: .venv/Scripts/python paper/scripts/label_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crta.data import HMM_COLS  # noqa: E402
from crta.dating import hindsight_phases, label_quality  # noqa: E402
from crta.eval import PERIODS  # noqa: E402
from crta.regime import LEGACY_V1, LEGACY_V2, REGIMES, track, track_v2  # noqa: E402
from crta.report import label_statistics  # noqa: E402

FRAME = ROOT / "paper" / "results" / "4h_rerun" / "frame.parquet"
OUT = ROOT / "paper" / "results" / "causal" / "label_quality"
SCORED_FROM = 1080

LABELLERS = {
    "v1_published": (track, dict(LEGACY_V1)),
    "v2_published": (track_v2, dict(LEGACY_V2)),
    "v2_scaled": (track_v2, {"burn": 252, "scale": "burn", "warm": 0}),
    "v2_burn1080": (track_v2, {"burn": 1080, "scale": "raw", "warm": 0}),
    "v2_fixed": (track_v2, {"burn": 1080}),
}


def score(labels, frame: pd.DataFrame, phases: np.ndarray, diag: dict) -> dict:
    scored = slice(SCORED_FROM, len(frame))
    lab = list(labels)[scored]
    return {
        "quality": label_quality(lab, phases[scored]),
        "statistics": label_statistics(lab, frame.iloc[scored], PERIODS, diag),
    }


def main() -> int:
    frame = pd.read_parquet(FRAME)
    X = frame[list(HMM_COLS)].to_numpy(dtype=float)
    phases = hindsight_phases(frame["close"].to_numpy(dtype=float))
    results = {
        "scored_from_bar": SCORED_FROM,
        "n_scored": int(len(frame) - SCORED_FROM),
        "hindsight": {
            "threshold": 0.20,
            "bull_share": float(np.mean(phases[SCORED_FROM:] == "bull")),
            "bear_share": float(np.mean(phases[SCORED_FROM:] == "bear")),
        },
        "labellers": {},
    }
    rows = []
    for name, (fn, kwargs) in LABELLERS.items():
        print(f"labelling {name} {kwargs}", flush=True)
        diag: dict = {}
        labels, _probs = fn(X, diagnostics=diag, **kwargs)
        block = score(labels, frame, phases, diag)
        block["settings"] = kwargs
        results["labellers"][name] = block
        q, s = block["quality"], block["statistics"]
        row = {"labeller": name}
        row.update({f"{g}_share": q["label_share"][g] for g in REGIMES})
        row["bull_labelled_bull"] = q["bull_labelled_bull"]
        row["bear_labelled_bear_or_crisis"] = q["bear_labelled_bear_or_crisis"]
        row["bear_labelled_bull"] = q["bear_labelled_bull"]
        row["switches_per_year"] = s["switches_per_year"]
        row["median_run_bars"] = s["median_run_bars"]
        for g in REGIMES:
            vol = s["per_regime"][g]["volatility"]
            row[f"{g}_vol_pct"] = None if vol is None else vol * 100.0
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_quality.json").write_text(json.dumps(results, indent=2, default=str))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "label_quality.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

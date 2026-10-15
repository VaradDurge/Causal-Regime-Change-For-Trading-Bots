"""Compare v1 and v2 label statistics on the cached timeframe history.

Writes paper/results/v2/tables/label_stats_raw.csv. This reruns both trackers
with diagnostics, so it is separate from the training grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.data import FEATURE_COLS, HMM_COLS, features
from crta.regime import track, track_v2
from crta.report import label_statistics
import timeframe_study as study

OUT = ROOT / "paper" / "results" / "v2" / "tables"
PERIODS = {"1h": 24 * 365, "15m": 4 * 24 * 365}
TAPES = (("BTC", "1h"), ("BTC", "15m"), ("ETH", "1h"), ("ETH", "15m"))


def _flatten(asset, timeframe, version, stats) -> dict:
    row = {
        "asset": asset,
        "timeframe": timeframe,
        "tracker": version,
        "switches_per_year": stats["switches_per_year"],
        "median_run_bars": stats["median_run_bars"],
        "mean_matched_w2": stats["mean_matched_w2"],
        "identity_jump": stats["identity_jump"],
    }
    for name, block in stats["per_regime"].items():
        row[f"{name}_share"] = block["share"]
        row[f"{name}_mean_return"] = block["mean_return"]
        row[f"{name}_volatility"] = block["volatility"]
    return row


def one(asset: str, timeframe: str) -> list[dict]:
    raw = study.candles(asset, timeframe)
    frame = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    X = frame[list(HMM_COLS)].to_numpy(dtype=float)
    cal = dict(study.CAL[timeframe])
    rows = []
    for version, fn, extra in (
        ("v1", track, {}),
        (
            "v2",
            track_v2,
            {"matching": "nearest", "death_after": 20, "cost_quantile": 0.95},
        ),
    ):
        diag: dict = {}
        labels, _probs = fn(X, diagnostics=diag, seed=42, **cal, **extra)
        stats = label_statistics(labels, frame, PERIODS[timeframe], diag)
        after = labels[cal["burn"] :]
        stats["bull_share_after_burn"] = after.count("bull") / len(after) if after else None
        row = _flatten(asset, timeframe, version, stats)
        row["bull_share_after_burn"] = stats["bull_share_after_burn"]
        rows.append(row)
        print(
            f"{asset} {timeframe} {version}: bull after burn {row['bull_share_after_burn']:.4f}",
            flush=True,
        )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    only = sys.argv[1:] or [f"{a}:{tf}" for a, tf in TAPES]
    for item in only:
        asset, timeframe = item.split(":")
        rows.extend(one(asset, timeframe))
        pd.DataFrame(rows).to_csv(OUT / "label_stats_raw.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

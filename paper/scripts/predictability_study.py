"""Can the agent's inputs forecast what it is paid on? Walk-forward on the frozen 4h frame.

Targets, all measured from bar t: the next bar's log return (what the reward pays),
its absolute value (a risk target), and the next day's log return (6 bars). Feature
sets: the agent's five features, those plus 2/4/8-week momentum, and those plus the
filtered regime probabilities of the pipeline's labeller (semantic guard plus the
drawdown input). Models: least squares and gradient-boosted trees.
Each fold fits on rows before its test block (the 4h study's four folds) and scores
out-of-sample R^2 against the training mean, plus the sign hit rate for returns.

Usage: .venv/Scripts/python paper/scripts/predictability_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crta.baselines import BARS_PER_WEEK, MOMENTUM_WEEKS  # noqa: E402
from crta.data import FEATURE_COLS, HMM_COLS_DD, drawdown  # noqa: E402
from crta.eval import walk_forward_bounds  # noqa: E402
from crta.predictability import hit_rate, momentum_features, oos_r2, walk_forward_forecasts  # noqa: E402
from crta.regime import REGIMES, track_v2  # noqa: E402

FRAME = ROOT / "paper" / "results" / "4h_rerun" / "frame.parquet"
OUT = ROOT / "paper" / "results" / "causal" / "predictability"
MIN_TRAIN = 1080
DAY = 6

MODELS = {
    "ols": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "gbt": lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=0
    ),
}


def _targets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    nxt = frame["next_ret"].to_numpy(dtype=float)
    day = pd.Series(nxt).rolling(DAY).sum().shift(-(DAY - 1)).to_numpy()
    return {"next_return": nxt, "next_abs_return": np.abs(nxt), "next_day_return": day}


def _feature_sets(frame: pd.DataFrame, probs: np.ndarray) -> dict[str, np.ndarray]:
    base = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    mom = momentum_features(frame["close"], BARS_PER_WEEK, MOMENTUM_WEEKS).to_numpy(dtype=float)
    return {
        "agent": base,
        "agent_momentum": np.column_stack([base, mom]),
        "agent_regime": np.column_stack([base, probs]),
    }


def _evaluate(X: np.ndarray, y: np.ndarray, bounds, make_model, signed: bool) -> dict:
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    start = int(np.argmax(valid))
    end = int(len(y) - np.argmax(valid[::-1]))
    if not valid[start:end].all():
        raise RuntimeError("gaps inside the usable span")
    shifted = [(max(a, start) - start, min(b, end) - start) for a, b in bounds]
    Xs, ys = X[start:end], y[start:end]
    pred, bench = walk_forward_forecasts(Xs, ys, shifted, make_model)
    test = ys[shifted[0][0] : shifted[-1][1]]
    out = {
        "n_test": int(len(test)),
        "r2_vs_train_mean": oos_r2(test, pred, bench),
        "r2_vs_zero": oos_r2(test, pred, np.zeros(len(test))) if signed else None,
    }
    if signed:
        out["hit_rate"] = hit_rate(test, pred)
        out["up_share"] = float(np.mean(test > 0))
    return out


def main() -> int:
    frame = pd.read_parquet(FRAME)
    bounds = walk_forward_bounds(len(frame), min_train=MIN_TRAIN)
    print("filtered regime probabilities", flush=True)
    frame["drawdown"] = drawdown(frame["close"])
    _labels, probs = track_v2(frame[list(HMM_COLS_DD)].to_numpy(dtype=float), burn=MIN_TRAIN)
    targets = _targets(frame)
    sets = _feature_sets(frame, probs)
    rows = []
    for target, y in targets.items():
        signed = target != "next_abs_return"
        for set_name, X in sets.items():
            for model_name, make_model in MODELS.items():
                res = _evaluate(X, y, bounds, make_model, signed)
                rows.append({"target": target, "features": set_name, "model": model_name, **res})
                print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "predictability.csv", index=False)
    (OUT / "predictability.json").write_text(
        json.dumps({"folds": bounds, "regimes": list(REGIMES), "rows": rows}, indent=2, default=float)
    )
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

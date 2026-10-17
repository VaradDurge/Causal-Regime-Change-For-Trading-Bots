"""Out-of-sample predictability of next-bar targets from the agent's features."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def oos_r2(y, pred, benchmark) -> float:
    """1 - SSE(pred) / SSE(benchmark). Below 0 means the benchmark forecasts better."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    denom = float(np.sum((y - benchmark) ** 2))
    if denom <= 0.0:
        raise ValueError("benchmark has zero error, R^2 is undefined")
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom


def hit_rate(y, pred) -> float | None:
    """Share of bars where the forecast has the target's sign. Zero targets and forecasts are skipped."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    keep = (y != 0.0) & (pred != 0.0)
    if not keep.any():
        return None
    return float(np.mean(np.sign(y[keep]) == np.sign(pred[keep])))


def momentum_features(close, bars_per_week: int, weeks=(2, 4, 8)) -> pd.DataFrame:
    """Log return over the last ``w`` weeks for each ``w``. NaN until the lookback exists."""
    log_close = pd.Series(np.log(np.asarray(close, dtype=float)))
    return pd.DataFrame(
        {f"mom_{w}w": log_close - log_close.shift(int(w) * int(bars_per_week)) for w in weeks}
    )


def walk_forward_forecasts(
    X, y, bounds: list[tuple[int, int]], make_model: Callable[[], object]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on rows before each test block and forecast the block.

    Returns the concatenated forecasts and the training-mean benchmark for the same rows.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    preds, benches = [], []
    for train_end, test_end in bounds:
        model = make_model()
        model.fit(X[:train_end], y[:train_end])
        preds.append(np.asarray(model.predict(X[train_end:test_end]), dtype=float))
        benches.append(np.full(test_end - train_end, float(y[:train_end].mean())))
    return np.concatenate(preds), np.concatenate(benches)

"""Walk-forward splits and return statistics. Train rows never include the test bar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crta.regime import REGIMES
from crta.stats import circular_block_indices

TURNOVER_COST = 0.001
N_FOLDS = 4
MIN_TRAIN = 1080
N_BOOT = 1000
PERIODS = 6 * 365  # 4h bars
SEED = 42


def walk_forward_bounds(n: int, n_folds: int = N_FOLDS, min_train: int = MIN_TRAIN):
    if n - min_train < n_folds:
        raise ValueError(f"need at least {min_train + n_folds} rows, got {n}")
    block = (n - min_train) // n_folds
    bounds = []
    for i in range(n_folds):
        train_end = min_train + i * block
        test_end = n if i == n_folds - 1 else train_end + block
        bounds.append((train_end, test_end))
    return bounds


def strategy_returns(position, next_ret, start_position: float = 0.0) -> np.ndarray:
    pos = np.asarray(position, dtype=float)
    nxt = np.asarray(next_ret, dtype=float)
    prev = np.concatenate([[start_position], pos[:-1]])
    return pos * nxt - TURNOVER_COST * np.abs(pos - prev)


def wealth_curve(returns) -> np.ndarray:
    return np.exp(np.cumsum(np.asarray(returns, dtype=float)))


def max_drawdown(returns) -> float:
    curve = np.concatenate([[1.0], wealth_curve(returns)])
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def _ratio(returns, downside: bool):
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return None
    if downside:
        scale = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    else:
        scale = float(r.std(ddof=1))
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    return float(r.mean() / scale * np.sqrt(PERIODS))


def sharpe(returns):
    return _ratio(returns, downside=False)


def sortino(returns):
    return _ratio(returns, downside=True)


def bootstrap_ci(returns, fn, n: int = N_BOOT, seed: int = SEED, block: int | None = None) -> dict:
    """95% percentile interval from a circular block bootstrap.

    Bars are serially dependent, so resampling single bars understates the spread. The
    default block is ceil(sqrt(len(returns))); ``block=1`` is the i.i.d. bootstrap.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    point = fn(r)
    size = max(1, int(np.ceil(np.sqrt(len(r))))) if block is None else max(1, int(block))
    stats = []
    for idx in circular_block_indices(len(r), size, n, rng):
        val = fn(r[idx])
        if val is not None and np.isfinite(val):
            stats.append(val)
    if not stats:
        return {"point": None if point is None else float(point), "low": None, "high": None, "block": size}
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "point": None if point is None else float(point),
        "low": float(lo),
        "high": float(hi),
        "block": size,
    }


def performance(returns) -> dict:
    r = np.asarray(returns, dtype=float)
    terminal = float(np.exp(r.sum()) - 1.0)
    return {
        "sharpe": bootstrap_ci(r, sharpe),
        "sortino": bootstrap_ci(r, sortino),
        "max_drawdown": max_drawdown(r),
        "total_return": terminal,
    }


def by_regime(returns, labels) -> dict:
    r = np.asarray(returns, dtype=float)
    labels = np.asarray(labels)
    out = {}
    for name in REGIMES:
        mask = labels == name
        chunk = r[mask]
        out[name] = {
            "n": int(mask.sum()),
            "sharpe": None if mask.sum() < 2 else sharpe(chunk),
            "mean_return": None if mask.sum() == 0 else float(chunk.mean()),
        }
    return out


def crash_index(df: pd.DataFrame) -> int:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    window = (ts >= "2022-05-01") & (ts <= "2022-11-30")
    pool = df.loc[window] if window.any() else df
    for name in ("crisis", "bear"):
        if "regime" in pool.columns:
            hit = pool.loc[pool["regime"] == name]
            if len(hit):
                return int(hit["close"].idxmin())
    return int(pool["close"].idxmin())

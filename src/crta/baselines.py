"""Fixed-rule baselines that a learned agent has to beat. Position at t uses data up to t."""

from __future__ import annotations

import numpy as np

from crta.regime import REGIMES

# 4h bars: 6 a day, 42 a week.
BARS_PER_WEEK = 42
MOMENTUM_WEEKS = (2, 4, 8)


def momentum_position(close, lookback: int) -> np.ndarray:
    """Long when the close is above its value ``lookback`` bars ago, flat otherwise."""
    if int(lookback) < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    lookback = int(lookback)
    close = np.asarray(close, dtype=float)
    position = np.zeros(len(close))
    if len(close) > lookback:
        position[lookback:] = (close[lookback:] > close[:-lookback]).astype(float)
    return position


def regime_rule_position(probs) -> np.ndarray:
    """Hold 1 - p(bear) - p(crisis): long in bull and calm, flat in bear and crisis."""
    probs = np.asarray(probs, dtype=float)
    bear = REGIMES.index("bear")
    crisis = REGIMES.index("crisis")
    return np.clip(1.0 - probs[:, bear] - probs[:, crisis], 0.0, 1.0)


def baseline_positions(close, probs, bars_per_week: int = BARS_PER_WEEK) -> dict[str, np.ndarray]:
    """Buy-and-hold, long/flat momentum over 2, 4 and 8 weeks, and the regime rule.

    Pass the full close history so the first test bars still have a lookback.
    """
    close = np.asarray(close, dtype=float)
    out = {
        "buy_hold": np.ones(len(close)),
        "regime_rule": regime_rule_position(probs),
    }
    for weeks in MOMENTUM_WEEKS:
        out[f"momentum_{weeks}w"] = momentum_position(close, weeks * int(bars_per_week))
    return out

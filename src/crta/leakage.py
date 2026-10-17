"""Helpers for the lag test: a signal that looks ahead loses its edge when delayed."""

from __future__ import annotations

import numpy as np


def lag_rows(values, bars: int = 1) -> np.ndarray:
    """Row t gets row t - bars. The first rows repeat row 0, the earliest value known."""
    if int(bars) < 0:
        raise ValueError(f"lag must be non-negative, got {bars}")
    values = np.asarray(values, dtype=float)
    out = values.copy()
    bars = int(bars)
    if bars == 0 or len(values) == 0:
        return out
    out[bars:] = values[:-bars]
    out[:bars] = values[0]
    return out

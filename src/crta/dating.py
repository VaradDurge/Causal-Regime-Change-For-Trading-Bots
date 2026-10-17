"""Hindsight bull/bear dating, the yardstick for regime labels.

A bull phase ends at a peak once the close falls ``threshold`` below it, and a bear
phase ends at a trough once the close rises ``threshold`` above it (the 20% rule of
Pagan and Sossounov, 2003). The dating uses the whole series, so it is for scoring
labels only and never reaches an agent.
"""

from __future__ import annotations

import numpy as np

from crta.regime import REGIMES

THRESHOLD = 0.20
PHASES = ("bull", "bear")


def _turning_points(close: np.ndarray, threshold: float) -> list[tuple[int, str]]:
    turns: list[tuple[int, str]] = []
    hi = lo = 0
    state = None
    for i in range(1, len(close)):
        rising = state in (None, "bull")
        falling = state in (None, "bear")
        if rising and close[i] > close[hi]:
            hi = i
        if falling and close[i] < close[lo]:
            lo = i
        if rising and close[i] <= close[hi] * (1.0 - threshold):
            # Every close since the peak stayed above this one, so it is the low so far.
            turns.append((hi, "peak"))
            state, lo = "bear", i
        elif falling and close[i] >= close[lo] * (1.0 + threshold):
            turns.append((lo, "trough"))
            state, hi = "bull", i
    return turns


def hindsight_phases(close, threshold: float = THRESHOLD) -> np.ndarray:
    """``bull`` or ``bear`` per bar. A peak closes its bull phase and a trough its bear phase.

    The stretch after the last turning point keeps the phase that turn started. A series
    that never moves by ``threshold`` is ``undated``.
    """
    close = np.asarray(close, dtype=float)
    phases = np.full(len(close), "undated", dtype=object)
    turns = _turning_points(close, threshold)
    if not turns:
        return phases
    start = 0
    for index, kind in turns:
        phases[start : index + 1] = "bull" if kind == "peak" else "bear"
        start = index + 1
    phases[start:] = "bear" if turns[-1][1] == "peak" else "bull"
    return phases


def _share(labels: np.ndarray, mask: np.ndarray, names: tuple[str, ...]) -> float | None:
    if not mask.any():
        return None
    return float(np.isin(labels[mask], names).mean())


def label_quality(labels, phases) -> dict:
    """How the regime labels line up with hindsight phases."""
    labels = np.asarray(labels, dtype=object)
    phases = np.asarray(phases, dtype=object)
    if len(labels) != len(phases):
        raise ValueError(f"{len(labels)} labels for {len(phases)} phases")
    bull = phases == "bull"
    bear = phases == "bear"
    n = len(labels)
    return {
        "n": int(n),
        "bull_bars": int(bull.sum()),
        "bear_bars": int(bear.sum()),
        "label_share": {name: (float(np.mean(labels == name)) if n else None) for name in REGIMES},
        "bull_labelled_bull": _share(labels, bull, ("bull",)),
        "bull_labelled_calm": _share(labels, bull, ("calm",)),
        "bear_labelled_bear_or_crisis": _share(labels, bear, ("bear", "crisis")),
        "bear_labelled_calm": _share(labels, bear, ("calm",)),
        "bear_labelled_bull": _share(labels, bear, ("bull",)),
        "confusion": {
            phase: {name: int(((phases == phase) & (labels == name)).sum()) for name in REGIMES} for phase in PHASES
        },
    }

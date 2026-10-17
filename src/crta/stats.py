"""Resampling and multiple-testing statistics for strategy returns.

The deflated Sharpe ratio follows Bailey and Lopez de Prado (2014): the probability that
the true Sharpe ratio beats the largest one expected from ``n_trials`` unskilled trials.
All Sharpe ratios here are per period, not annualised.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def circular_block_indices(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """Resample indices as runs of ``block`` consecutive bars that wrap around the end."""
    block = max(1, int(block))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n


def per_period_sharpe(returns) -> float:
    r = np.asarray(returns, dtype=float)
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd) if sd > 1e-12 else 0.0


def probabilistic_sharpe(returns, benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), allowing for skew and fat tails (benchmark per period)."""
    r = np.asarray(returns, dtype=float)
    sr = per_period_sharpe(r)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 0.0:
        raise ValueError("Sharpe ratio variance estimate is not positive")
    return float(norm.cdf((sr - benchmark) * np.sqrt(len(r) - 1) / np.sqrt(denom)))


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` unskilled trials with this variance."""
    n_trials = int(n_trials)
    if n_trials <= 1:
        return 0.0
    z = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(
        1.0 - 1.0 / (n_trials * np.e)
    )
    return float(np.sqrt(max(sharpe_variance, 0.0)) * z)


def deflated_sharpe(returns, trial_sharpes, n_trials: int | None = None) -> dict:
    """Deflated Sharpe ratio of ``returns`` given the per-period Sharpe ratios of all trials.

    ``n_trials`` defaults to the number of trial Sharpe ratios. Pass a larger count when
    more configurations were screened than were kept.
    """
    trials = np.asarray([s for s in trial_sharpes if s is not None and np.isfinite(s)], dtype=float)
    count = len(trials) if n_trials is None else int(n_trials)
    variance = float(trials.var(ddof=1)) if len(trials) > 1 else 0.0
    sr0 = expected_max_sharpe(variance, count)
    return {
        "dsr": probabilistic_sharpe(returns, sr0),
        "sr": per_period_sharpe(returns),
        "sr0": sr0,
        "n_trials": count,
        "trial_sharpe_variance": variance,
    }


def seed_summary(values) -> dict:
    """Median and interquartile range across seeds, with mean and standard deviation beside them."""
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(clean) == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "iqr": None, "mean": None, "std": None}
    q25, median, q75 = np.percentile(clean, [25, 50, 75])
    return {
        "n": int(len(clean)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
    }

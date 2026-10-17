"""Wasserstein k-means on rolling return distributions (Horvath, Issa and Muguruza, 2021).

Each bar is described by the empirical distribution of the last ``window`` log returns.
For two samples of equal size the 2-Wasserstein distance is the root mean squared gap
between their sorted values, and the barycentre of a cluster is the average of its
sorted vectors, so Wasserstein k-means is k-means on sorted windows. A window ending at
t uses returns up to t only.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

from crta.regime import BURN, REGIMES, lock_states

WK_WINDOW = 42  # one week of 4h returns
WK_STRIDE = 180
N_CLUSTERS = len(REGIMES)


def window_quantiles(returns, window: int) -> np.ndarray:
    """Row t is the sorted window of returns ending at t; rows before the first full window are NaN."""
    r = np.asarray(returns, dtype=float)
    out = np.full((len(r), int(window)), np.nan)
    if len(r) >= window:
        out[window - 1 :] = np.sort(sliding_window_view(r, int(window)), axis=1)
    return out


def w2_sorted(a, b) -> float:
    """2-Wasserstein distance between two equal-size samples given as sorted vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_wkmeans(Q, k: int = N_CLUSTERS, seed: int = 0, n_init: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Centroids (W2 barycentres) and assignments of sorted windows."""
    model = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=int(seed)).fit(np.asarray(Q, dtype=float))
    return model.cluster_centers_, model.labels_


def _mapping(centroids: np.ndarray) -> dict[int, str]:
    moments = np.column_stack([centroids.mean(axis=1), centroids.std(axis=1)])
    return lock_states(moments)


def _probs(Q: np.ndarray, centroids: np.ndarray, mapping: dict[int, str], tau: float) -> np.ndarray:
    d2 = ((Q[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    rel = d2 - d2.min(axis=1, keepdims=True)
    weights = np.exp(-rel / tau)
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((len(Q), len(REGIMES)))
    for comp, name in mapping.items():
        out[:, REGIMES.index(name)] += weights[:, comp]
    return out


def _fit_on(Q: np.ndarray, end: int, window: int, sample_every: int, seed: int):
    rows = np.arange(window - 1, end, int(sample_every))
    centroids, _ = fit_wkmeans(Q[rows], seed=seed)
    d2 = ((Q[rows][:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2).min(axis=1)
    tau = float(np.median(d2)) if np.median(d2) > 0 else 1e-12
    return centroids, _mapping(centroids), tau


def track_wkmeans(
    returns,
    window: int = WK_WINDOW,
    burn: int = BURN,
    stride: int = WK_STRIDE,
    seed: int = 0,
    sample_every: int = 1,
    temperature: float | None = None,
) -> tuple[list[str], np.ndarray]:
    """Walk-forward Wasserstein k-means labels and soft probabilities.

    Every ``stride`` bars the clusters are refitted on windows that end before the refit
    and named by moment sorting of their barycentres. Probabilities are softmax(-W2^2 /
    tau), tau the median nearest W2^2 on the fitting windows unless ``temperature`` is
    set. Bars before the first full window get uniform probabilities and the label calm.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    window = int(window)
    burn = int(min(max(burn, window + 4 * N_CLUSTERS), n))
    Q = window_quantiles(r, window)
    labels: list[str] = ["calm"] * (window - 1)
    probs = np.full((n, len(REGIMES)), 1.0 / len(REGIMES))

    centroids, mapping, tau = _fit_on(Q, burn, window, sample_every, seed)
    span = slice(window - 1, burn)
    probs[span] = _probs(Q[span], centroids, mapping, temperature or tau)
    labels.extend(REGIMES[int(np.argmax(row))] for row in probs[span])

    t = burn
    while t < n:
        end = min(t + int(stride), n)
        centroids, mapping, tau = _fit_on(Q, t, window, sample_every, seed)
        probs[t:end] = _probs(Q[t:end], centroids, mapping, temperature or tau)
        labels.extend(REGIMES[int(np.argmax(row))] for row in probs[t:end])
        t = end
    return labels, probs

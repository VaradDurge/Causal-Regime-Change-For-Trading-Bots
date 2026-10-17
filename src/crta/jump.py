"""Statistical jump model (Nystrup et al., 2020; Shu, Yu and Mulvey, 2024).

States are centroids in feature space. Fitting alternates two steps until the state
path stops changing: the path that minimises squared distance to its centroids plus
``jump_penalty`` per switch (dynamic programming), then each centroid as the mean of
its bars. A larger penalty gives longer, more persistent regimes.

Inference is online: the forward recursion V_t(k) = loss_t(k) + min(V_{t-1}(k),
min_j V_{t-1}(j) + penalty) uses rows up to t only, and the state at t is argmin V_t.
"""

from __future__ import annotations

import numpy as np

from crta.regime import BURN, REGIMES, feature_scale, lock_states

JUMP_STRIDE = 180  # refit monthly on 4h bars
JUMP_WARM = 252  # rows of earlier history the online recursion runs over first
N_JUMP_STATES = len(REGIMES)


def _losses(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)


def jump_dp(losses: np.ndarray, jump_penalty: float) -> np.ndarray:
    """State path minimising total loss plus ``jump_penalty`` per switch. Uses the whole sequence."""
    losses = np.asarray(losses, dtype=float)
    n, k = losses.shape
    value = losses[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    stay = np.arange(k)
    for t in range(1, n):
        j = int(np.argmin(value))
        switch = value[j] + jump_penalty
        take = switch < value
        back[t] = np.where(take, j, stay)
        value = losses[t] + np.where(take, switch, value)
    states = np.empty(n, dtype=np.int64)
    states[-1] = int(np.argmin(value))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def online_costs(X: np.ndarray, centroids: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Forward recursion costs. Row t depends on rows up to t only."""
    losses = _losses(np.asarray(X, dtype=float), np.asarray(centroids, dtype=float))
    out = np.empty_like(losses)
    out[0] = losses[0]
    for t in range(1, len(losses)):
        prev = out[t - 1]
        out[t] = losses[t] + np.minimum(prev, prev.min() + jump_penalty)
    return out


def _kmeanspp(X: np.ndarray, k: int, rng) -> np.ndarray:
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = _losses(X, np.asarray(centroids)).min(axis=1)
        total = d2.sum()
        idx = rng.integers(len(X)) if total <= 0 else rng.choice(len(X), p=d2 / total)
        centroids.append(X[idx])
    return np.asarray(centroids, dtype=float)


def fit_jump(
    X,
    n_states: int = N_JUMP_STATES,
    jump_penalty: float = 50.0,
    n_init: int = 5,
    max_iter: int = 30,
    seed: int = 0,
    init_centroids=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Best of ``n_init`` fits (the first starts from ``init_centroids`` when given)."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    for i in range(int(n_init)):
        if i == 0 and init_centroids is not None:
            centroids = np.array(init_centroids, dtype=float, copy=True)
        else:
            centroids = _kmeanspp(X, n_states, rng)
        states = None
        for _ in range(int(max_iter)):
            new_states = jump_dp(_losses(X, centroids), jump_penalty)
            if states is not None and np.array_equal(new_states, states):
                break
            states = new_states
            for g in range(n_states):
                members = states == g
                if members.any():
                    centroids[g] = X[members].mean(axis=0)
        loss = _losses(X, centroids)[np.arange(len(X)), states].sum() + jump_penalty * np.sum(states[1:] != states[:-1])
        if best is None or loss < best[2]:
            best = (centroids.copy(), states.copy(), float(loss))
    return best


def _named_probs(costs: np.ndarray, mapping: dict[int, str], temperature: float) -> np.ndarray:
    rel = costs - costs.min(axis=1, keepdims=True)
    weights = np.exp(-rel / temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((len(costs), len(REGIMES)))
    for comp, name in mapping.items():
        out[:, REGIMES.index(name)] += weights[:, comp]
    return out


def track_jump(
    X,
    burn: int = BURN,
    stride: int = JUMP_STRIDE,
    jump_penalty: float = 50.0,
    seed: int = 0,
    warm: int = JUMP_WARM,
    temperature: float | None = None,
    n_init: int = 4,
) -> tuple[list[str], np.ndarray]:
    """Walk-forward jump-model labels and soft probabilities.

    Features are standardised by the burn-in. Every ``stride`` bars the model is refitted
    on all earlier rows (warm-started from the last centroids) and its states are named
    by moment sorting of the centroids (columns 0 and 1 are return and volatility). Each
    window is decoded online after running the recursion over ``warm`` earlier rows.
    Probabilities are softmax(-(V - min V) / temperature), temperature = jump_penalty by
    default, so a state one switch-penalty worse gets weight e^-1.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    burn = int(min(max(burn, 4 * N_JUMP_STATES), n))
    tau = float(jump_penalty if temperature is None else temperature)
    mu = X[:burn].mean(axis=0)
    Xs = (X - mu) / feature_scale(X[:burn])
    labels: list[str] = []
    probs = np.zeros((n, len(REGIMES)))

    centroids, _states, _loss = fit_jump(Xs[:burn], jump_penalty=jump_penalty, n_init=n_init, seed=seed)
    mapping = lock_states(centroids)
    costs = online_costs(Xs[:burn], centroids, jump_penalty)
    probs[:burn] = _named_probs(costs, mapping, tau)
    labels.extend(mapping[int(k)] for k in costs.argmin(axis=1))

    t = burn
    while t < n:
        end = min(t + int(stride), n)
        centroids, _states, _loss = fit_jump(
            Xs[:t], jump_penalty=jump_penalty, n_init=n_init, seed=seed, init_centroids=centroids
        )
        mapping = lock_states(centroids)
        lead = min(int(warm), t)
        costs = online_costs(Xs[t - lead : end], centroids, jump_penalty)[lead:]
        probs[t:end] = _named_probs(costs, mapping, tau)
        labels.extend(mapping[int(k)] for k in costs.argmin(axis=1))
        t = end
    return labels, probs

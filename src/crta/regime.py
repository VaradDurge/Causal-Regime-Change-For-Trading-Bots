"""4-state Gaussian HMM. The burn-in is named by moments; later refits keep those names by Wasserstein distance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM

REGIMES = ("bull", "bear", "calm", "crisis")
# 4h bars. 30 bars = 5 days, so a single session cannot flip the label.
# A 1-day dwell (6 bars) still reads as barcode noise on a five-year plate.
MIN_DWELL = 30
N_STATES = 4
# Six weeks to name the templates. Each later window is one week, fit on earlier rows only.
BURN = 252
STRIDE = 42
ETA = 0.2


def enforce_dwell(labels, min_dwell: int = MIN_DWELL, prev: str | None = None) -> list[str]:
    # ponytail: post-hoc merge of short runs. A sticky transition prior is the upgrade.
    seq = [str(prev)] if prev is not None else []
    seq.extend(str(x) for x in labels)
    i = 0
    n = len(seq)
    while i < n:
        j = i + 1
        while j < n and seq[j] == seq[i]:
            j += 1
        if (j - i) < min_dwell and i > 0:
            previous = seq[i - 1]
            for k in range(i, j):
                seq[k] = previous
        i = j
    if prev is not None:
        return seq[1:]
    return seq


def lock_states(means: np.ndarray) -> dict[int, str]:
    """Low return + high vol = crisis. Names the burn-in only."""
    order = sorted(range(len(means)), key=lambda i: (float(means[i, 0]), i))
    low, high = order[:2], order[2:]
    low = sorted(low, key=lambda i: (float(means[i, 1]), i))
    high = sorted(high, key=lambda i: (float(means[i, 1]), i))
    return {low[0]: "bear", low[1]: "crisis", high[0]: "calm", high[1]: "bull"}


@dataclass
class RegimeModel:
    hmm: GaussianHMM
    mapping: dict[int, str]
    mu: np.ndarray
    sd: np.ndarray


def _state_index(mapping: dict[int, str], name: str) -> int:
    for key, value in mapping.items():
        if value == name:
            return int(key)
    raise KeyError(name)


def _fit_hmm(X: np.ndarray, seed: int = 42) -> RegimeModel:
    X = np.asarray(X, dtype=float)
    if len(X) < N_STATES * 5:
        raise ValueError(f"need at least {N_STATES * 5} rows to fit, got {len(X)}")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xs = (X - mu) / sd
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=200,
        random_state=seed,
        min_covar=1e-3,
    )
    model.fit(xs)
    # EM can leave a state with no observed exit. predict() then refuses the matrix.
    trans = np.array(model.transmat_, dtype=float, copy=True)
    dead = trans.sum(axis=1) < 1e-8
    if np.any(dead):
        trans[dead] = 1.0 / trans.shape[1]
        model.transmat_ = trans
    start = np.array(model.startprob_, dtype=float, copy=True)
    if start.sum() < 1e-8:
        model.startprob_ = np.full(len(start), 1.0 / len(start))
    return RegimeModel(hmm=model, mapping={}, mu=mu, sd=sd)


def fit_regime(X: np.ndarray, seed: int = 42) -> RegimeModel:
    fitted = _fit_hmm(X, seed)
    fitted.mapping = lock_states(fitted.hmm.means_)
    return fitted


def component_params(model: RegimeModel) -> tuple[np.ndarray, np.ndarray]:
    """Emission mean and std, undoing this fit's scaler so windows can be compared."""
    mu = np.asarray(model.hmm.means_, dtype=float) * model.sd + model.mu
    cov = np.asarray(model.hmm.covars_, dtype=float)
    var = np.diagonal(cov, axis1=-2, axis2=-1) if cov.ndim == 3 else cov
    sd = np.sqrt(np.maximum(var, 0.0)) * model.sd
    return np.asarray(mu, dtype=float), np.asarray(sd, dtype=float)


def w2_diag(mu_a, sd_a, mu_b, sd_b) -> float:
    """Squared 2-Wasserstein distance between two diagonal Gaussians."""
    mu_a = np.asarray(mu_a, dtype=float)
    sd_a = np.asarray(sd_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    sd_b = np.asarray(sd_b, dtype=float)
    return float(np.sum((mu_a - mu_b) ** 2 + (sd_a - sd_b) ** 2))


def assign_templates(mu, sd, template_mu, template_sd) -> np.ndarray:
    """Nearest template for each component. Two components may share one template."""
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    template_mu = np.asarray(template_mu, dtype=float)
    template_sd = np.asarray(template_sd, dtype=float)
    delta_mu = mu[:, None, :] - template_mu[None, :, :]
    delta_sd = sd[:, None, :] - template_sd[None, :, :]
    dist = np.sum(delta_mu**2 + delta_sd**2, axis=2)
    return np.argmin(dist, axis=1).astype(int)


def update_templates(template_mu, template_sd, mu, sd, proba, assign, eta: float = ETA):
    """Move each matched template toward its components. An empty template stays put."""
    template_mu = np.array(template_mu, dtype=float, copy=True)
    template_sd = np.array(template_sd, dtype=float, copy=True)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    proba = np.asarray(proba, dtype=float)
    assign = np.asarray(assign, dtype=int)
    for g in range(len(template_mu)):
        members = np.flatnonzero(assign == g)
        if len(members) == 0:
            continue
        weights = proba[:, members].sum(axis=0)
        total = float(weights.sum())
        if total < 1e-12:
            continue
        weights = weights / total
        template_mu[g] = (1.0 - eta) * template_mu[g] + eta * (weights @ mu[members])
        template_sd[g] = (1.0 - eta) * template_sd[g] + eta * (weights @ sd[members])
    return template_mu, template_sd


def decode(model: RegimeModel, X: np.ndarray, prev: str | None = None) -> tuple[list[str], np.ndarray]:
    X = np.asarray(X, dtype=float)
    xs = (X - model.mu) / model.sd
    raw = model.hmm.predict(xs)
    proba = model.hmm.predict_proba(xs)
    names = [model.mapping[int(state)] for state in raw]
    sticky = enforce_dwell(names, prev=prev)
    ordered = np.column_stack([proba[:, _state_index(model.mapping, name)] for name in REGIMES])
    return sticky, ordered


def _name_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(REGIMES)}


def _templates_from_lock(mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = lock_states(mu)
    index = _name_index()
    assign = np.array([index[mapping[k]] for k in range(len(mu))], dtype=int)
    template_mu = np.zeros((len(REGIMES), mu.shape[1]))
    template_sd = np.zeros((len(REGIMES), sd.shape[1]))
    for k, g in enumerate(assign):
        template_mu[g] = mu[k]
        template_sd[g] = sd[k]
    return template_mu, template_sd, assign


def _aggregate(proba: np.ndarray, assign: np.ndarray) -> np.ndarray:
    ordered = np.zeros((len(proba), len(REGIMES)))
    for k, g in enumerate(assign):
        ordered[:, int(g)] += proba[:, k]
    return ordered


def _window_labels(model: RegimeModel, X: np.ndarray, assign: np.ndarray, prev: str | None, dwell: int = MIN_DWELL):
    xs = (np.asarray(X, dtype=float) - model.mu) / model.sd
    raw = model.hmm.predict(xs)
    proba = model.hmm.predict_proba(xs)
    names = [REGIMES[int(assign[int(state)])] for state in raw]
    # Closed here. A later window must not relabel bars already issued.
    sticky = enforce_dwell(names, min_dwell=dwell, prev=prev)
    return sticky, _aggregate(proba, assign), proba


def track(
    X: np.ndarray,
    burn: int = BURN,
    stride: int = STRIDE,
    eta: float = ETA,
    seed: int = 42,
    dwell: int = MIN_DWELL,
) -> tuple[list[str], np.ndarray]:
    """One causal pass. The burn-in is named by moments; every later window matches templates by W2."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < N_STATES * 5:
        raise ValueError(f"need at least {N_STATES * 5} rows to track, got {n}")
    burn = int(min(max(burn, N_STATES * 5), n))
    stride = int(max(stride, 1))
    labels: list[str] = []
    probs = np.zeros((n, len(REGIMES)))

    fitted = _fit_hmm(X[:burn], seed)
    mu, sd = component_params(fitted)
    template_mu, template_sd, assign = _templates_from_lock(mu, sd)
    window_labels, window_probs, _proba = _window_labels(fitted, X[:burn], assign, None, dwell)
    labels.extend(window_labels)
    probs[:burn] = window_probs

    t = burn
    while t < n:
        end = min(t + stride, n)
        fitted = _fit_hmm(X[:t], seed)
        mu, sd = component_params(fitted)
        assign = assign_templates(mu, sd, template_mu, template_sd)
        window_labels, window_probs, proba = _window_labels(fitted, X[t:end], assign, labels[-1], dwell)
        labels.extend(window_labels)
        probs[t:end] = window_probs
        template_mu, template_sd = update_templates(template_mu, template_sd, mu, sd, proba, assign, eta)
        t = end
    return labels, probs


def label_span(X, start: int, end: int, prev: str | None = None, seed: int = 42):
    """Label rows [start, end) from a fit on the rows before start. Later rows are not used."""
    X = np.asarray(X, dtype=float)
    start = int(start)
    end = int(end)
    if start < 0 or end > len(X) or start >= end:
        raise IndexError(f"bad span {start}:{end} for {len(X)} rows")
    fit_end = start if start >= N_STATES * 5 else min(len(X), N_STATES * 5)
    fitted = _fit_hmm(X[:fit_end], seed)
    mu, sd = component_params(fitted)
    _template_mu, _template_sd, assign = _templates_from_lock(mu, sd)
    labels, probs, _proba = _window_labels(fitted, X[start:end], assign, prev)
    return labels, probs

"""4-state Gaussian HMM. The burn-in is named by moments; later refits keep those names by Wasserstein distance.

Probabilities are filtered: the value at bar t uses rows up to t only. The smoothed
(forward-backward) posterior lets bar t see later bars of its window, including the
next bar's return, so it is kept only to reproduce and audit the published runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM

REGIMES = ("bull", "bear", "calm", "crisis")
# 4h bars. 30 bars = 5 days, so a single session cannot flip the label.
# A 1-day dwell (6 bars) still reads as barcode noise on a five-year plate.
MIN_DWELL = 30
N_STATES = 4
# Fold 0's training window (1,080 4h bars, six months) names the templates, so the
# burn-in costs no test bar. Each later window is one week, fit on earlier rows only.
BURN = 1080
STRIDE = 42
ETA = 0.2
# A window's forward filter first runs over this many strides of earlier rows, so it
# does not restart from the fit's start vector. Those rows are all before the window.
WARM_STRIDES = 6
POSTERIORS = ("filtered", "smoothed")
SCALES = ("burn", "raw")
# Changes whenever the same input gets different labels, so label caches go stale.
LABELLER_VERSION = "filtered-scaled-guard-1"
# The published pipeline: six-week burn-in, raw-unit matching and in-window smoothing.
# For leakage audits and reproductions only.
LEGACY_V1 = {"burn": 252, "posterior": "smoothed", "scale": "raw", "warm": 0}
LEGACY_V2 = {"burn": 252, "scale": "raw", "warm": 0, "semantic_guard": False}
# The Tier 1 labeller, before the semantic guard.
TIER1_V2 = {"burn": 1080, "semantic_guard": False}


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


def feature_scale(X: np.ndarray) -> np.ndarray:
    """Per-feature standard deviation of the burn-in rows, with a floor for flat columns."""
    sd = np.asarray(X, dtype=float).std(axis=0)
    return np.where(sd < 1e-12, 1.0, sd)


def _units(X: np.ndarray, burn: int, scale: str) -> np.ndarray:
    """Units for W2. ``burn`` divides by the burn-in std so returns are not drowned by trend."""
    if scale == "burn":
        return feature_scale(X[:burn])
    if scale == "raw":
        return np.ones(np.asarray(X).shape[1])
    raise ValueError(f"unknown scale {scale}, expected one of {SCALES}")


def _component_units(model: RegimeModel, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = component_params(model)
    return mu / units, sd / units


def _warm_rows(warm: int | None, stride: int) -> int:
    return WARM_STRIDES * int(stride) if warm is None else max(int(warm), 0)


def _history(X: np.ndarray, start: int, rows: int, posterior: str) -> np.ndarray | None:
    if posterior != "filtered" or rows <= 0 or start <= 0:
        return None
    return X[max(0, start - rows) : start]


def _posterior_states(
    model: RegimeModel, X: np.ndarray, posterior: str = "filtered", history: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Most likely state and state probabilities for each row of X.

    ``filtered`` uses rows up to t; ``history`` rows run through the filter first and
    are dropped. ``smoothed`` is Viterbi and forward-backward over all of X, so bar t
    depends on later rows. Only audits ask for it.
    """
    X = np.asarray(X, dtype=float)
    if posterior == "smoothed":
        xs = (X - model.mu) / model.sd
        return model.hmm.predict(xs), model.hmm.predict_proba(xs)
    if posterior != "filtered":
        raise ValueError(f"unknown posterior {posterior}, expected one of {POSTERIORS}")
    lead = 0 if history is None else len(history)
    rows = X if lead == 0 else np.vstack([np.asarray(history, dtype=float), X])
    proba = forward_filter(model, rows)[lead:]
    return proba.argmax(axis=1), proba


def _sticky(names: list[str], posterior: str, dwell: int, prev: str | None) -> list[str]:
    # The post-hoc merge needs a run's end, which lies in the future. Filtered labels use the online rule.
    if posterior == "smoothed":
        return enforce_dwell(names, min_dwell=dwell, prev=prev)
    return enforce_dwell_online(names, min_dwell=dwell, prev=prev)


def decode(
    model: RegimeModel, X: np.ndarray, prev: str | None = None, posterior: str = "filtered"
) -> tuple[list[str], np.ndarray]:
    states, proba = _posterior_states(model, X, posterior)
    names = [model.mapping[int(state)] for state in states]
    sticky = _sticky(names, posterior, MIN_DWELL, prev)
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


def _window_labels(
    model: RegimeModel,
    X: np.ndarray,
    assign: np.ndarray,
    prev: str | None,
    dwell: int = MIN_DWELL,
    posterior: str = "filtered",
    history: np.ndarray | None = None,
):
    states, proba = _posterior_states(model, X, posterior, history)
    names = [REGIMES[int(assign[int(state)])] for state in states]
    # Closed here. A later window must not relabel bars already issued.
    sticky = _sticky(names, posterior, dwell, prev)
    return sticky, _aggregate(proba, assign), proba


def track(
    X: np.ndarray,
    burn: int = BURN,
    stride: int = STRIDE,
    eta: float = ETA,
    seed: int = 42,
    dwell: int = MIN_DWELL,
    diagnostics: dict | None = None,
    posterior: str = "filtered",
    scale: str = "burn",
    warm: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """One causal pass. The burn-in is named by moments; every later window matches templates by W2.

    ``LEGACY_V1`` reproduces the published run, whose smoothed probabilities look ahead.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < N_STATES * 5:
        raise ValueError(f"need at least {N_STATES * 5} rows to track, got {n}")
    burn = int(min(max(burn, N_STATES * 5), n))
    stride = int(max(stride, 1))
    units = _units(X, burn, scale)
    warm_rows = _warm_rows(warm, stride)
    labels: list[str] = []
    probs = np.zeros((n, len(REGIMES)))
    if diagnostics is not None:
        diagnostics["units"] = units

    fitted = _fit_hmm(X[:burn], seed)
    mu, sd = _component_units(fitted, units)
    template_mu, template_sd, assign = _templates_from_lock(mu, sd)
    window_labels, window_probs, _proba = _window_labels(fitted, X[:burn], assign, None, dwell, posterior)
    labels.extend(window_labels)
    probs[:burn] = window_probs

    t = burn
    while t < n:
        end = min(t + stride, n)
        fitted = _fit_hmm(X[:t], seed)
        mu, sd = _component_units(fitted, units)
        assign = assign_templates(mu, sd, template_mu, template_sd)
        window_labels, window_probs, proba = _window_labels(
            fitted, X[t:end], assign, labels[-1], dwell, posterior, _history(X, t, warm_rows, posterior)
        )
        labels.extend(window_labels)
        probs[t:end] = window_probs
        if diagnostics is not None:
            cost = pairwise_w2(mu, sd, template_mu, template_sd)
            diagnostics.setdefault("matched_costs", []).append(
                float(np.mean([cost[k, int(assign[k])] for k in range(len(assign))]))
            )
            diagnostics.setdefault("template_mu", []).append(np.array(template_mu, copy=True))
            diagnostics.setdefault("template_sd", []).append(np.array(template_sd, copy=True))
            diagnostics.setdefault("component_mu", []).append(np.array(mu, copy=True))
            diagnostics.setdefault("component_sd", []).append(np.array(sd, copy=True))
            diagnostics.setdefault("assign", []).append(np.array(assign, copy=True))
            diagnostics.setdefault("responsibility", []).append(np.asarray(proba).sum(axis=0))
            diagnostics.setdefault("bars", []).append(int(t))
        template_mu, template_sd = update_templates(template_mu, template_sd, mu, sd, proba, assign, eta)
        t = end
    return labels, probs


def label_span(
    X,
    start: int,
    end: int,
    prev: str | None = None,
    seed: int = 42,
    posterior: str = "filtered",
    warm: int | None = None,
    dwell: int = MIN_DWELL,
):
    """Label rows [start, end) from a fit on the rows before start. Bar t uses rows up to t only."""
    X = np.asarray(X, dtype=float)
    start = int(start)
    end = int(end)
    if start < 0 or end > len(X) or start >= end:
        raise IndexError(f"bad span {start}:{end} for {len(X)} rows")
    fit_end = start if start >= N_STATES * 5 else min(len(X), N_STATES * 5)
    fitted = _fit_hmm(X[:fit_end], seed)
    mu, sd = component_params(fitted)
    _template_mu, _template_sd, assign = _templates_from_lock(mu, sd)
    history = _history(X, start, _warm_rows(warm, STRIDE), posterior)
    labels, probs, _proba = _window_labels(fitted, X[start:end], assign, prev, dwell, posterior, history)
    return labels, probs


def log_gauss_diag(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """Diagonal-Gaussian log likelihood. X is (T, D), means and covars are (K, D)."""
    var = np.maximum(np.asarray(covars, dtype=float), 1e-12)
    xc = np.asarray(X, dtype=float)[:, None, :] - np.asarray(means, dtype=float)[None, :, :]
    quad = np.sum((xc * xc) / var[None, :, :], axis=2)
    logdet = np.sum(np.log(2.0 * np.pi * var), axis=1)
    return -0.5 * (quad + logdet[None, :])


def _covars_diag(model: RegimeModel) -> np.ndarray:
    cov = np.asarray(model.hmm.covars_, dtype=float)
    if cov.ndim == 3:
        cov = np.diagonal(cov, axis1=-2, axis2=-1)
    return cov


def forward_filter(model: RegimeModel, X: np.ndarray) -> np.ndarray:
    """Filtered posteriors p(z_t | x_1..x_t). Bar t does not see x after t.

    Log-space forward recursion from the fitted start vector, transitions and
    diagonal Gaussian emissions. Each step is renormalized; the scale cancels,
    so the rows are the filtered posteriors.
    """
    from scipy.special import logsumexp

    X = np.asarray(X, dtype=float)
    xs = (X - model.mu) / model.sd
    log_start = np.log(np.clip(np.asarray(model.hmm.startprob_, dtype=float), 1e-300, None))
    log_trans = np.log(np.clip(np.asarray(model.hmm.transmat_, dtype=float), 1e-300, None))
    log_emit = log_gauss_diag(xs, np.asarray(model.hmm.means_, dtype=float), _covars_diag(model))
    log_post = np.empty_like(log_emit)
    log_post[0] = log_start + log_emit[0]
    log_post[0] -= logsumexp(log_post[0])
    for t in range(1, len(xs)):
        mixed = logsumexp(log_post[t - 1][:, None] + log_trans, axis=0)
        log_post[t] = log_emit[t] + mixed
        log_post[t] -= logsumexp(log_post[t])
    return np.exp(log_post)


def enforce_dwell_online(labels, min_dwell: int = MIN_DWELL, prev: str | None = None) -> list[str]:
    """Switch only after `min_dwell` consecutive new labels. Past labels are never rewritten."""
    out: list[str] = []
    current = None if prev is None else str(prev)
    candidate = None
    run = 0
    for lab in labels:
        lab = str(lab)
        if current is None:
            current = lab
            out.append(current)
            continue
        if lab == current:
            candidate = None
            run = 0
            out.append(current)
            continue
        if lab == candidate:
            run += 1
        else:
            candidate = lab
            run = 1
        if run >= min_dwell:
            current = candidate
            candidate = None
            run = 0
        out.append(current)
    return out


def pairwise_w2(mu, sd, template_mu, template_sd) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    template_mu = np.asarray(template_mu, dtype=float)
    template_sd = np.asarray(template_sd, dtype=float)
    delta_mu = mu[:, None, :] - template_mu[None, :, :]
    delta_sd = sd[:, None, :] - template_sd[None, :, :]
    return np.sum(delta_mu**2 + delta_sd**2, axis=2)


def assign_matching(
    mu,
    sd,
    template_mu,
    template_sd,
    matching: str = "nearest",
    historical: list | None = None,
    quantile: float = 0.95,
    min_history: int = 8,
) -> np.ndarray:
    """Match components to templates.

    ``nearest`` is the historical rule and may send two components to one template.
    ``hungarian`` is one-to-one. ``hungarian_with_threshold`` drops a Hungarian pair
    whose cost exceeds `quantile` of previously accepted costs and merges that
    component into its nearest template. Until `min_history` costs exist there is
    no quantile, so every Hungarian pair is kept.
    """
    cost = pairwise_w2(mu, sd, template_mu, template_sd)
    if matching == "nearest":
        assign = np.argmin(cost, axis=1).astype(int)
        if historical is not None:
            historical.extend(float(cost[k, assign[k]]) for k in range(len(assign)))
        return assign
    if matching not in ("hungarian", "hungarian_with_threshold"):
        raise ValueError(f"unknown matching {matching}")
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    assign = np.argmin(cost, axis=1).astype(int)
    for row, col in zip(rows, cols):
        assign[int(row)] = int(col)
    if matching == "hungarian":
        if historical is not None:
            historical.extend(float(cost[int(r), int(c)]) for r, c in zip(rows, cols))
        return assign
    thresh = None
    if historical is not None and len(historical) >= min_history:
        thresh = float(np.quantile(np.asarray(historical, dtype=float), quantile))
    accepted = []
    for row, col in zip(rows, cols):
        row, col = int(row), int(col)
        if thresh is not None and cost[row, col] > thresh:
            assign[row] = int(np.argmin(cost[row]))
        else:
            accepted.append(float(cost[row, col]))
    if historical is not None:
        historical.extend(accepted)
    return assign


def rebirth_source(mu, sd, template_mu, template_sd, dead: int) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Component farthest from the other templates, or the moment-sorted name if that distance is ~0."""
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    others = [j for j in range(len(template_mu)) if j != dead]
    best_k, best_d = 0, -1.0
    for k in range(len(mu)):
        if others:
            dmin = min(
                w2_diag(mu[k], sd[k], template_mu[j], template_sd[j]) for j in others
            )
        else:
            dmin = 0.0
        if dmin > best_d:
            best_d, best_k = dmin, k
    if best_d < 1e-6:
        mapping = lock_states(mu)
        name = REGIMES[dead] if dead < len(REGIMES) else None
        for comp, nm in mapping.items():
            if nm == name:
                return mu[int(comp)].copy(), sd[int(comp)].copy(), "moment", int(comp)
    return mu[best_k].copy(), sd[best_k].copy(), "farthest", int(best_k)


def rename_by_moments(template_mu, template_sd) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-name templates by the burn-in rule applied to their own moments.

    Templates drift under the update, and nothing else stops the crisis template from
    ending up calmer than calm. Returns the re-ordered arrays and ``perm`` with
    ``new = old[perm]``. Templates move slowly, so names swap only when two cross.
    """
    template_mu = np.asarray(template_mu, dtype=float)
    template_sd = np.asarray(template_sd, dtype=float)
    perm = np.empty(len(REGIMES), dtype=int)
    for idx, name in lock_states(template_mu).items():
        perm[REGIMES.index(name)] = int(idx)
    return template_mu[perm].copy(), template_sd[perm].copy(), perm


def _moment_source(mu, sd, dead: int) -> tuple[np.ndarray, np.ndarray, str, int]:
    """The component this fit's moments give the dead template's name."""
    name = REGIMES[dead]
    for comp, label in lock_states(np.asarray(mu, dtype=float)).items():
        if label == name:
            return np.asarray(mu)[int(comp)].copy(), np.asarray(sd)[int(comp)].copy(), "moment", int(comp)
    raise KeyError(name)


def apply_birth_death(
    template_mu,
    template_sd,
    mu,
    sd,
    assign,
    empty_run,
    death_after: int,
    events: list,
    refit_index: int,
    bar: int,
    date: str | None = None,
    source: str = "farthest",
):
    """Re-seed a template that has had no member for `death_after` refits. One pass, then the caller re-matches.

    ``farthest`` takes the component farthest from the other templates. ``moment`` takes
    the component that moment sorting names after the dead template.
    """
    template_mu = np.array(template_mu, dtype=float, copy=True)
    template_sd = np.array(template_sd, dtype=float, copy=True)
    empty_run = np.array(empty_run, dtype=int, copy=True)
    assign = np.asarray(assign, dtype=int)
    for g in range(len(template_mu)):
        if np.any(assign == g):
            empty_run[g] = 0
        else:
            empty_run[g] += 1
    reborn = False
    for g in range(len(template_mu)):
        if empty_run[g] < death_after:
            continue
        if source == "moment":
            src_mu, src_sd, how, comp = _moment_source(mu, sd, g)
        elif source == "farthest":
            src_mu, src_sd, how, comp = rebirth_source(mu, sd, template_mu, template_sd, g)
        else:
            raise ValueError(f"unknown rebirth source {source}")
        template_mu[g] = src_mu
        template_sd[g] = src_sd
        empty_run[g] = 0
        reborn = True
        events.append(
            {
                "kind": "death_birth",
                "refit_index": int(refit_index),
                "bar": int(bar),
                "date": date,
                "template": REGIMES[g] if g < len(REGIMES) else str(g),
                "source": how,
                "component": int(comp),
            }
        )
    return template_mu, template_sd, empty_run, reborn


def _window_labels_v2(
    model: RegimeModel,
    X: np.ndarray,
    assign: np.ndarray,
    prev: str | None,
    dwell: int,
    history: np.ndarray | None = None,
):
    return _window_labels(model, X, assign, prev, dwell, "filtered", history)


def track_v2(
    X: np.ndarray,
    burn: int = BURN,
    stride: int = STRIDE,
    eta: float = ETA,
    seed: int = 42,
    dwell: int = MIN_DWELL,
    matching: str = "nearest",
    death_after: int = 20,
    cost_quantile: float = 0.95,
    timestamps=None,
    diagnostics: dict | None = None,
    scale: str = "burn",
    warm: int | None = None,
    semantic_guard: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Causal tracker with template birth, optional Hungarian matching, and filtered posteriors.

    A template update still uses only the fit on rows before the window and the
    filtered responsibilities of that window. Closed windows are not rewritten.
    W2 compares components in units of the burn-in std (``scale="burn"``), and each
    window's filter is warmed on earlier rows. ``LEGACY_V2`` reproduces the published v2.
    ``semantic_guard`` re-names templates by their moments after every rebirth and
    update, and rebirths take the component the moments name, so "crisis" stays the
    low-return, high-volatility template.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < N_STATES * 5:
        raise ValueError(f"need at least {N_STATES * 5} rows to track, got {n}")
    burn = int(min(max(burn, N_STATES * 5), n))
    stride = int(max(stride, 1))
    units = _units(X, burn, scale)
    warm_rows = _warm_rows(warm, stride)
    labels: list[str] = []
    probs = np.zeros((n, len(REGIMES)))
    events: list = []
    if diagnostics is not None:
        diagnostics["units"] = units
        diagnostics.setdefault("events", events)
        diagnostics.setdefault("matched_costs", [])
        diagnostics.setdefault("template_mu", [])
        diagnostics.setdefault("template_sd", [])
        diagnostics.setdefault("component_mu", [])
        diagnostics.setdefault("component_sd", [])
        diagnostics.setdefault("assign", [])
        diagnostics.setdefault("responsibility", [])
        events = diagnostics["events"]

    fitted = _fit_hmm(X[:burn], seed)
    mu, sd = _component_units(fitted, units)
    template_mu, template_sd, assign = _templates_from_lock(mu, sd)
    window_labels, window_probs, _proba = _window_labels_v2(fitted, X[:burn], assign, None, dwell)
    labels.extend(window_labels)
    probs[:burn] = window_probs
    historical: list[float] = []
    empty_run = np.zeros(len(REGIMES), dtype=int)

    t = burn
    refit = 0
    while t < n:
        end = min(t + stride, n)
        fitted = _fit_hmm(X[:t], seed)
        mu, sd = _component_units(fitted, units)
        date = None
        if timestamps is not None:
            date = str(timestamps[min(t, len(timestamps) - 1)])
        assign = assign_matching(mu, sd, template_mu, template_sd, matching)
        template_mu, template_sd, empty_run, reborn = apply_birth_death(
            template_mu,
            template_sd,
            mu,
            sd,
            assign,
            empty_run,
            death_after,
            events,
            refit,
            t,
            date,
            source="moment" if semantic_guard else "farthest",
        )
        if semantic_guard:
            template_mu, template_sd, perm = rename_by_moments(template_mu, template_sd)
            empty_run = empty_run[perm]
        assign = assign_matching(
            mu, sd, template_mu, template_sd, matching, historical, cost_quantile
        )
        window_labels, window_probs, proba = _window_labels_v2(
            fitted, X[t:end], assign, labels[-1], dwell, _history(X, t, warm_rows, "filtered")
        )
        labels.extend(window_labels)
        probs[t:end] = window_probs
        if diagnostics is not None:
            cost = pairwise_w2(mu, sd, template_mu, template_sd)
            diagnostics["matched_costs"].append(float(np.mean([cost[k, int(assign[k])] for k in range(len(assign))])))
            diagnostics["template_mu"].append(np.array(template_mu, copy=True))
            diagnostics["template_sd"].append(np.array(template_sd, copy=True))
            diagnostics["component_mu"].append(np.array(mu, copy=True))
            diagnostics["component_sd"].append(np.array(sd, copy=True))
            diagnostics["assign"].append(np.array(assign, copy=True))
            diagnostics["responsibility"].append(proba.sum(axis=0))
            events.append(
                {
                    "kind": "refit",
                    "refit_index": int(refit),
                    "bar": int(t),
                    "date": date,
                    "fit_end": int(t),
                    "window_end": int(end),
                }
            )
        template_mu, template_sd = update_templates(
            template_mu, template_sd, mu, sd, proba, assign, eta
        )
        if semantic_guard:
            template_mu, template_sd, perm = rename_by_moments(template_mu, template_sd)
            empty_run = empty_run[perm]
        t = end
        refit += 1
    return labels, probs


def label_span_v2(X, start: int, end: int, prev: str | None = None, seed: int = 42, dwell: int = MIN_DWELL):
    """Same as ``label_span``, which is now filtered as well. Kept for existing callers."""
    return label_span(X, start, end, prev=prev, seed=seed, posterior="filtered", dwell=dwell)

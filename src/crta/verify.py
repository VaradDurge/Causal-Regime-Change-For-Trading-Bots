"""Ablation, counterfactual, and sensitivity checks on one frozen policy."""

from __future__ import annotations

import numpy as np

from crta.regime import REGIMES

ABLATION_MIN_DELTA = 0.15
CF_MIN_DELTA = 0.10
N_SWEEP = 11
# crisis is the most defensive, bull the least.
DEFENSE = {"bull": 0, "bear": 2, "calm": 1, "crisis": 3}
_WEIGHTS = np.array([1.0, -0.4, 0.5, -1.0])  # bull, bear, calm, crisis


def probe_action(probs) -> float:
    p = np.asarray(probs, dtype=float)
    return float(np.clip(np.dot(p, _WEIGHTS), -1.0, 1.0))


def ignore_action(probs) -> float:
    return 0.5


def inverted_action(probs) -> float:
    p = np.asarray(probs, dtype=float)
    return float(np.clip(np.dot(p, -_WEIGHTS), -1.0, 1.0))


def one_hot(name: str) -> np.ndarray:
    probs = np.zeros(len(REGIMES))
    probs[REGIMES.index(name)] = 1.0
    return probs


def with_crisis_prob(probs, crisis: float) -> np.ndarray:
    base = np.asarray(probs, dtype=float).copy()
    idx = REGIMES.index("crisis")
    base[idx] = 0.0
    if float(base.sum()) <= 1e-12:
        base[:] = 1.0
        base[idx] = 0.0
    base = base / base.sum()
    out = base * (1.0 - float(crisis))
    out[idx] = float(crisis)
    return out


def _violations(curve: np.ndarray) -> int:
    return int(np.sum(curve[1:] > curve[:-1] + 1e-8))


def audit_bar(action_fn, probs) -> dict:
    probs = np.asarray(probs, dtype=float)
    actual = float(action_fn(probs))
    ablated = float(action_fn(np.zeros(len(REGIMES))))
    delta = abs(actual - ablated)
    current = REGIMES[int(np.argmax(probs))]
    rank = DEFENSE[current]
    forced_crisis = None
    forced_bull = None
    cf_ok = True
    if rank < DEFENSE["crisis"]:
        forced_crisis = float(action_fn(one_hot("crisis")))
        cf_ok = cf_ok and (actual - forced_crisis) >= CF_MIN_DELTA
    if rank > DEFENSE["bull"]:
        forced_bull = float(action_fn(one_hot("bull")))
        cf_ok = cf_ok and (forced_bull - actual) >= CF_MIN_DELTA
    curve = np.array(
        [float(action_fn(with_crisis_prob(probs, c))) for c in np.linspace(0.0, 1.0, N_SWEEP)]
    )
    violations = _violations(curve)
    sens_ok = violations <= 1
    ab_ok = delta >= ABLATION_MIN_DELTA
    return {
        "action": actual,
        "ablation_delta": float(delta),
        "ablation": ab_ok,
        "counterfactual": cf_ok,
        "forced_crisis": forced_crisis,
        "forced_bull": forced_bull,
        "sensitivity": sens_ok,
        "violations": violations,
        "curve": curve,
        "passed": bool(ab_ok and cf_ok and sens_ok),
    }


def audit_many(predict, probs) -> dict:
    """predict(array of shape (n, 4)) -> actions. Same rules as audit_bar, batched."""
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    zeros = np.zeros_like(probs)
    crisis = np.repeat(one_hot("crisis")[None, :], n, axis=0)
    bull = np.repeat(one_hot("bull")[None, :], n, axis=0)
    grid = np.linspace(0.0, 1.0, N_SWEEP)
    sweeps = [np.stack([with_crisis_prob(row, c) for row in probs]) for c in grid]
    packed = np.concatenate([probs, zeros, crisis, bull, *sweeps], axis=0)
    actions = np.asarray(predict(packed), dtype=float).reshape(-1)
    actual = actions[:n]
    ablated = actions[n : 2 * n]
    forced_c = actions[2 * n : 3 * n]
    forced_b = actions[3 * n : 4 * n]
    curves = actions[4 * n :].reshape(N_SWEEP, n).T
    delta = np.abs(actual - ablated)
    ab_ok = delta >= ABLATION_MIN_DELTA
    ranks = np.array([DEFENSE[REGIMES[i]] for i in np.argmax(probs, axis=1)])
    cf_ok = np.ones(n, dtype=bool)
    need_down = ranks < DEFENSE["crisis"]
    cf_ok[need_down] &= (actual[need_down] - forced_c[need_down]) >= CF_MIN_DELTA
    need_up = ranks > DEFENSE["bull"]
    cf_ok[need_up] &= (forced_b[need_up] - actual[need_up]) >= CF_MIN_DELTA
    violations = np.sum(curves[:, 1:] > curves[:, :-1] + 1e-8, axis=1)
    sens_ok = violations <= 1
    passed = ab_ok & cf_ok & sens_ok
    return {
        "action": actual,
        "ablation_delta": delta,
        "pass_ablation": ab_ok,
        "pass_cf": cf_ok,
        "pass_sens": sens_ok,
        "passed": passed,
        "curve": curves,
    }

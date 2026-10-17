"""Ablation, counterfactual, and sensitivity checks on one frozen policy."""

from __future__ import annotations

import numpy as np

from crta.regime import REGIMES

ABLATION_MIN_DELTA = 0.15
CF_MIN_DELTA = 0.10
N_SWEEP = 11
# Verifier v3 leaves a regime unranked below this much training-fold probability weight.
MIN_ORDER_WEIGHT = 30.0
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


def with_regime_prob(probs, name: str, q: float) -> np.ndarray:
    """Set p(name) = q and rescale the other regimes to fill 1 - q."""
    base = np.asarray(probs, dtype=float).copy()
    idx = REGIMES.index(name)
    base[idx] = 0.0
    if float(base.sum()) <= 1e-12:
        base[:] = 1.0
        base[idx] = 0.0
    base = base / base.sum()
    out = base * (1.0 - float(q))
    out[idx] = float(q)
    return out


def with_crisis_prob(probs, crisis: float) -> np.ndarray:
    return with_regime_prob(probs, "crisis", crisis)


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


def empirical_banks(probs) -> dict[str, np.ndarray]:
    """Rows of tracker probabilities grouped by argmax. Empty regimes stay empty."""
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0:
        return {name: np.zeros((0, len(REGIMES))) for name in REGIMES}
    names = np.array(REGIMES)[np.argmax(probs, axis=1)]
    return {name: probs[names == name] for name in REGIMES}


def pick_empirical(probs, rows, pick: str = "nearest", rng=None) -> np.ndarray | None:
    """Nearest training vector in L1, or a random one among the 20 nearest."""
    rows = np.asarray(rows, dtype=float)
    if len(rows) == 0:
        return None
    dist = np.abs(rows - np.asarray(probs, dtype=float)).sum(axis=1)
    if pick == "top20":
        k = min(20, len(rows))
        pool = np.argpartition(dist, k - 1)[:k]
        if rng is None:
            rng = np.random.default_rng(0)
        return rows[int(rng.choice(pool))]
    if pick != "nearest":
        raise ValueError(f"unknown cf_pick {pick}")
    return rows[int(np.argmin(dist))]


def _cf_pair(probs, banks, pick: str, rng):
    """In-distribution crisis and bull vectors. Missing regimes are untestable, not failures."""
    probs = np.asarray(probs, dtype=float)
    rank = DEFENSE[REGIMES[int(np.argmax(probs))]]
    crisis = bull = None
    untestable = []
    if rank < DEFENSE["crisis"]:
        crisis = pick_empirical(probs, banks.get("crisis", ()), pick, rng)
        if crisis is None:
            untestable.append("crisis")
    if rank > DEFENSE["bull"]:
        bull = pick_empirical(probs, banks.get("bull", ()), pick, rng)
        if bull is None:
            untestable.append("bull")
    return crisis, bull, untestable


def audit_bar_v2(
    action_fn,
    probs,
    banks,
    *,
    delta_ab: float = ABLATION_MIN_DELTA,
    delta_cf: float = CF_MIN_DELTA,
    delta_sens: float = 0.05,
    pick: str = "nearest",
    rng=None,
) -> dict:
    """Legacy checks, with in-distribution counterfactuals and a sensitivity range."""
    probs = np.asarray(probs, dtype=float)
    actual = float(action_fn(probs))
    ablated = float(action_fn(np.zeros(len(REGIMES))))
    delta = abs(actual - ablated)
    crisis, bull, untestable = _cf_pair(probs, banks, pick, rng)
    cf_ok = True
    forced_crisis = forced_bull = None
    if crisis is not None:
        forced_crisis = float(action_fn(crisis))
        cf_ok = cf_ok and (actual - forced_crisis) >= delta_cf
    if bull is not None:
        forced_bull = float(action_fn(bull))
        cf_ok = cf_ok and (forced_bull - actual) >= delta_cf
    curve = np.array(
        [float(action_fn(with_crisis_prob(probs, c))) for c in np.linspace(0.0, 1.0, N_SWEEP)]
    )
    violations = _violations(curve)
    span = float(curve.max() - curve.min()) if len(curve) else 0.0
    sens_ok = violations <= 1 and span >= delta_sens
    ab_ok = delta >= delta_ab
    testable = len(untestable) == 0
    return {
        "action": actual,
        "ablation_delta": float(delta),
        "ablation": ab_ok,
        "counterfactual": cf_ok if testable else "untestable",
        "untestable": untestable,
        "forced_crisis": forced_crisis,
        "forced_bull": forced_bull,
        "sensitivity": sens_ok,
        "sensitivity_span": span,
        "violations": violations,
        "curve": curve,
        # A missing regime is untestable, not a failure. The other checks still have to pass.
        "passed": bool(ab_ok and cf_ok and sens_ok),
    }


def audit_many_v2(
    predict,
    probs,
    banks,
    *,
    delta_ab: float = ABLATION_MIN_DELTA,
    delta_cf: float = CF_MIN_DELTA,
    delta_sens: float = 0.05,
    pick: str = "nearest",
    rng=None,
) -> dict:
    """Batched verifier_v2. A starved regime is untestable on the bars that needed it."""
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    if rng is None:
        rng = np.random.default_rng(0)
    crisis_rows = []
    bull_rows = []
    need_c = np.zeros(n, dtype=bool)
    need_b = np.zeros(n, dtype=bool)
    untestable = [[] for _ in range(n)]
    for i, row in enumerate(probs):
        crisis, bull, missing = _cf_pair(row, banks, pick, rng)
        need_c[i] = crisis is not None
        need_b[i] = bull is not None
        untestable[i] = missing
        crisis_rows.append(crisis if crisis is not None else row)
        bull_rows.append(bull if bull is not None else row)
    zeros = np.zeros_like(probs)
    grid = np.linspace(0.0, 1.0, N_SWEEP)
    sweeps = [np.stack([with_crisis_prob(row, c) for row in probs]) for c in grid]
    packed = np.concatenate(
        [probs, zeros, np.stack(crisis_rows), np.stack(bull_rows), *sweeps], axis=0
    )
    actions = np.asarray(predict(packed), dtype=float).reshape(-1)
    actual = actions[:n]
    ablated = actions[n : 2 * n]
    forced_c = actions[2 * n : 3 * n]
    forced_b = actions[3 * n : 4 * n]
    curves = actions[4 * n :].reshape(N_SWEEP, n).T
    delta = np.abs(actual - ablated)
    ab_ok = delta >= delta_ab
    cf_ok = np.ones(n, dtype=bool)
    cf_ok[need_c] &= (actual[need_c] - forced_c[need_c]) >= delta_cf
    cf_ok[need_b] &= (forced_b[need_b] - actual[need_b]) >= delta_cf
    testable = np.array([len(item) == 0 for item in untestable])
    violations = np.sum(curves[:, 1:] > curves[:, :-1] + 1e-8, axis=1)
    span = curves.max(axis=1) - curves.min(axis=1)
    sens_ok = (violations <= 1) & (span >= delta_sens)
    passed = ab_ok & cf_ok & sens_ok
    return {
        "action": actual,
        "ablation_delta": delta,
        "pass_ablation": ab_ok,
        "pass_cf": cf_ok,
        "pass_sens": sens_ok,
        "testable": testable,
        "untestable": untestable,
        "passed": passed,
        "curve": curves,
        "sensitivity_span": span,
    }


def regime_order(probs, next_ret, min_weight: float = MIN_ORDER_WEIGHT, var_floor: float = 1e-12) -> dict:
    """Risk order fixed from training rows by the mean-variance position mu / sigma^2.

    mu and sigma^2 are the probability-weighted mean and variance of the next return.
    Rank 0 is the least defensive regime, the one a mean-variance reward pays to hold
    most of. A regime with less total weight than ``min_weight`` is left unranked.
    """
    probs = np.asarray(probs, dtype=float)
    nxt = np.asarray(next_ret, dtype=float)
    weight = probs.sum(axis=0)
    mv: dict[str, float | None] = {}
    for g, name in enumerate(REGIMES):
        if weight[g] < min_weight:
            mv[name] = None
            continue
        w = probs[:, g]
        mean = float(w @ nxt / weight[g])
        var = float(w @ (nxt - mean) ** 2 / weight[g])
        mv[name] = mean / max(var, var_floor)
    ranked = sorted((name for name in REGIMES if mv[name] is not None), key=lambda name: -mv[name])
    return {
        "rank": {name: (ranked.index(name) if name in ranked else None) for name in REGIMES},
        "ranked": ranked,
        "mv_position": mv,
        "least_defensive": ranked[0] if ranked else None,
        "most_defensive": ranked[-1] if ranked else None,
    }


def reference_weights(order: dict) -> np.ndarray:
    """Linear weights from +1 (least defensive) to -1 (most defensive). Unranked regimes get 0."""
    ranked = order["ranked"]
    weights = np.zeros(len(REGIMES))
    for name, w in zip(ranked, np.linspace(1.0, -1.0, len(ranked))):
        weights[REGIMES.index(name)] = w
    return weights


def _v3_pair(row, rank: dict, most: str | None, least: str | None, banks, pick: str, rng):
    """In-distribution vectors for the most and least defensive regimes, as this bar needs them."""
    current = rank.get(REGIMES[int(np.argmax(row))])
    if current is None or most is None:
        return None, None, ["unranked"]
    down = up = None
    missing = []
    if current < rank[most]:
        down = pick_empirical(row, banks.get(most, ()), pick, rng)
        if down is None:
            missing.append(most)
    if current > rank[least]:
        up = pick_empirical(row, banks.get(least, ()), pick, rng)
        if up is None:
            missing.append(least)
    return down, up, missing


def audit_many_v3(
    predict,
    probs,
    banks,
    order: dict,
    baseline,
    *,
    delta_ab: float = ABLATION_MIN_DELTA,
    delta_cf: float = CF_MIN_DELTA,
    delta_sens: float = 0.05,
    pick: str = "nearest",
    rng=None,
) -> dict:
    """Verifier v3: v2 with an in-distribution ablation and a risk order fixed in advance.

    Ablation replaces p_t by ``baseline`` (the training-mean vector), which the policy has
    seen the like of, instead of zeros. The counterfactual and the sensitivity sweep use
    ``order`` from ``regime_order`` on the training fold: forcing the most defensive regime
    must cut the position, forcing the least defensive must raise it, and injecting the most
    defensive regime must not raise it more than once. A bar is faithful only if it is
    testable and passes all three.
    """
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    if rng is None:
        rng = np.random.default_rng(0)
    rank, most, least = order["rank"], order["most_defensive"], order["least_defensive"]
    down_rows, up_rows = [], []
    need_down = np.zeros(n, dtype=bool)
    need_up = np.zeros(n, dtype=bool)
    untestable = [[] for _ in range(n)]
    for i, row in enumerate(probs):
        down, up, missing = _v3_pair(row, rank, most, least, banks, pick, rng)
        need_down[i] = down is not None
        need_up[i] = up is not None
        untestable[i] = missing
        down_rows.append(down if down is not None else row)
        up_rows.append(up if up is not None else row)
    base = np.repeat(np.asarray(baseline, dtype=float)[None, :], n, axis=0)
    target = most or "crisis"
    grid = np.linspace(0.0, 1.0, N_SWEEP)
    sweeps = [np.stack([with_regime_prob(row, target, q) for row in probs]) for q in grid]
    packed = np.concatenate([probs, base, np.stack(down_rows), np.stack(up_rows), *sweeps], axis=0)
    actions = np.asarray(predict(packed), dtype=float).reshape(-1)
    actual = actions[:n]
    ablated = actions[n : 2 * n]
    forced_down = actions[2 * n : 3 * n]
    forced_up = actions[3 * n : 4 * n]
    curves = actions[4 * n :].reshape(N_SWEEP, n).T
    delta = np.abs(actual - ablated)
    ab_ok = delta >= delta_ab
    cf_ok = np.ones(n, dtype=bool)
    cf_ok[need_down] &= (actual[need_down] - forced_down[need_down]) >= delta_cf
    cf_ok[need_up] &= (forced_up[need_up] - actual[need_up]) >= delta_cf
    testable = np.array([len(item) == 0 for item in untestable], dtype=bool)
    violations = np.sum(curves[:, 1:] > curves[:, :-1] + 1e-8, axis=1)
    span = curves.max(axis=1) - curves.min(axis=1)
    sens_ok = (violations <= 1) & (span >= delta_sens)
    return {
        "action": actual,
        "ablation_delta": delta,
        "pass_ablation": ab_ok,
        "pass_cf": cf_ok,
        "pass_sens": sens_ok,
        "testable": testable,
        "untestable": untestable,
        "passed": ab_ok & cf_ok & sens_ok & testable,
        "curve": curves,
        "sensitivity_span": span,
        "order": order,
    }


def verifier_report(result: dict, regimes=None) -> dict:
    """Pass rates, per-regime breakdown, and the median absolute ablation effect."""
    passed = np.asarray(result["passed"], dtype=bool)
    ab = np.asarray(result["pass_ablation"], dtype=bool)
    cf = np.asarray(result["pass_cf"], dtype=bool)
    sens = np.asarray(result["pass_sens"], dtype=bool)
    testable = np.asarray(result.get("testable", np.ones(len(passed), dtype=bool)))
    delta = np.asarray(result["ablation_delta"], dtype=float)
    untestable = result.get("untestable", [[] for _ in range(len(passed))])
    starved = np.array([len(item) > 0 for item in untestable])
    out = {
        "ablation": float(ab.mean()) if len(ab) else None,
        "counterfactual": float(cf[testable].mean()) if testable.any() else None,
        "sensitivity": float(sens.mean()) if len(sens) else None,
        "all": float(passed.mean()) if len(passed) else None,
        "untestable_share": float(starved.mean()) if len(starved) else None,
        "median_ablation_delta": float(np.median(np.abs(delta))) if len(delta) else None,
        "n": int(len(passed)),
    }
    if regimes is not None:
        labels = np.asarray(regimes)
        per = {}
        for name in REGIMES:
            mask = labels == name
            per[name] = {
                "n": int(mask.sum()),
                "ablation": float(ab[mask].mean()) if mask.any() else None,
                "counterfactual": float(cf[mask & testable].mean()) if np.any(mask & testable) else None,
                "sensitivity": float(sens[mask].mean()) if mask.any() else None,
                "all": float(passed[mask].mean()) if mask.any() else None,
                "untestable_share": float(starved[mask].mean()) if mask.any() else None,
            }
        out["per_regime"] = per
    return out


def label_agreement(probs, dwell_labels) -> float:
    """Share of bars where argmax p_t equals the dwell-smoothed label."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(dwell_labels)
    guessed = np.array(REGIMES)[np.argmax(probs, axis=1)]
    return float(np.mean(guessed == labels)) if len(labels) else float("nan")


def audit_many(predict, probs, bound_aware: bool = False) -> dict:
    """predict(array of shape (n, 4)) -> actions. Same rules as audit_bar, batched.

    With ``bound_aware`` a counterfactual direction that the position limit makes
    impossible is not scored. Those bars are counted in ``cf_untestable`` and left
    out of ``faith_rate_testable``; ``passed`` keeps the original rule.
    """
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
    extra = {}
    if bound_aware:
        blocked_down = need_down & (actual < -1.0 + CF_MIN_DELTA)
        blocked_up = need_up & (actual > 1.0 - CF_MIN_DELTA)
        untestable = blocked_down | blocked_up
        testable = ~untestable
        extra = {
            "cf_untestable": untestable,
            "n_testable": int(testable.sum()),
            "faith_rate_testable": float(passed[testable].mean()) if testable.any() else float("nan"),
        }
    return {
        **extra,
        "action": actual,
        "ablation_delta": delta,
        "pass_ablation": ab_ok,
        "pass_cf": cf_ok,
        "pass_sens": sens_ok,
        "passed": passed,
        "curve": curves,
    }

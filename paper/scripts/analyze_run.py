"""Post-hoc analysis of the pipeline run in paper/results/run.

Reloads the saved PPO fold models, recomputes every intervention used by
crta.verify.audit_many, and splits the counterfactual check into its two
directions. Also scores the reference policies (probe, ignore, inverted) on
the same regime probabilities, per-fold Sharpe ratios and position statistics.
The archived run in outputs/run (earlier per-fold labelling) is summarised
alongside for comparison.

Writes paper/results/analysis.json and paper/results/analysis.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from crta.agent import observation
from crta.data import FEATURE_COLS, ROOT
from crta.eval import sharpe
from crta.regime import REGIMES
from crta.verify import (
    ABLATION_MIN_DELTA,
    CF_MIN_DELTA,
    DEFENSE,
    N_SWEEP,
    audit_many,
    ignore_action,
    inverted_action,
    one_hot,
    probe_action,
    with_crisis_prob,
)

PAPER = Path(__file__).resolve().parents[1]
RUN = PAPER / "results" / "run"
OLD = ROOT / "outputs" / "run"


def interventions(model, sub):
    feats = sub[list(FEATURE_COLS)].to_numpy(dtype=float)
    pos = sub["position_before"].to_numpy(dtype=float)
    probs = sub[list(REGIMES)].to_numpy(dtype=float)
    n = len(sub)

    def act(batch):
        obs = np.vstack([observation(feats[i % n], batch[i], float(pos[i % n]), True) for i in range(len(batch))])
        a, _ = model.predict(obs, deterministic=True)
        return np.clip(np.asarray(a, dtype=float).reshape(-1), -1, 1)

    out = {"actual": act(probs), "ablated": act(np.zeros_like(probs))}
    for name in REGIMES:
        out[f"forced_{name}"] = act(np.repeat(one_hot(name)[None], n, axis=0))
    grid = np.linspace(0, 1, N_SWEEP)
    out["curve"] = np.stack([act(np.stack([with_crisis_prob(r, c) for r in probs])) for c in grid], axis=1)
    return out


def check_rates(d):
    return {
        "ablation": float(d["pass_ablation"].mean()),
        "counterfactual": float(d["pass_cf"].mean()),
        "sensitivity": float(d["pass_sens"].mean()),
        "all": float(d["passed"].mean()),
        "n": int(len(d)),
    }


def reference_policies(probs):
    out = {}
    for name, fn in [("probe", probe_action), ("ignore", ignore_action), ("inverted", inverted_action)]:
        res = audit_many(lambda batch, fn=fn: np.array([fn(r) for r in batch]), probs)
        out[name] = {
            "ablation": float(res["pass_ablation"].mean()),
            "counterfactual": float(res["pass_cf"].mean()),
            "sensitivity": float(res["pass_sens"].mean()),
            "all": float(res["passed"].mean()),
        }
    return out


def main() -> int:
    new = pd.read_parquet(RUN / "bars.parquet")
    old = pd.read_parquet(OLD / "bars.parquet") if (OLD / "bars.parquet").exists() else None

    parts = {k: [] for k in ["actual", "ablated", *[f"forced_{r}" for r in REGIMES], "curve"]}
    for fold in sorted(new["fold"].unique()):
        sub = new[new["fold"] == fold]
        model = PPO.load(str(RUN / f"fold{fold}_conditioned"), device="cpu")
        res = interventions(model, sub)
        for k in parts:
            parts[k].append(res[k])
    iv = {k: np.concatenate(v) for k, v in parts.items()}
    assert np.allclose(iv["actual"], new["action_conditioned"].to_numpy(), atol=1e-5)
    assert np.allclose(np.abs(iv["actual"] - iv["ablated"]), new["ablation_delta"].to_numpy(), atol=1e-5)

    probs = new[list(REGIMES)].to_numpy(dtype=float)
    ranks = np.array([DEFENSE[REGIMES[i]] for i in np.argmax(probs, axis=1)])
    down_needed = ranks < DEFENSE["crisis"]
    up_needed = ranks > DEFENSE["bull"]
    down_ok = (iv["actual"] - iv["forced_crisis"]) >= CF_MIN_DELTA
    up_ok = (iv["forced_bull"] - iv["actual"]) >= CF_MIN_DELTA
    cf_recomputed = np.where(down_needed, down_ok, True) & np.where(up_needed, up_ok, True)
    assert np.array_equal(cf_recomputed, new["pass_cf"].to_numpy())

    regime_arg = np.array([REGIMES[i] for i in np.argmax(probs, axis=1)])
    labels = new["regime"].to_numpy()
    argmax_agree = float(np.mean(regime_arg == labels))

    per_fold = {}
    for fold in sorted(new["fold"].unique()):
        m = new["fold"].to_numpy() == fold
        per_fold[int(fold)] = {
            "start": str(new["timestamp"][m].iloc[0]),
            "end": str(new["timestamp"][m].iloc[-1]),
            "sharpe_conditioned": sharpe(new["ret_conditioned"][m]),
            "sharpe_blind": sharpe(new["ret_blind"][m]),
            "sharpe_buy_hold": sharpe(new["ret_bh"][m]),
            "checks": check_rates(new[m]),
        }
        if old is not None:
            per_fold[int(fold)]["sharpe_conditioned_archived"] = sharpe(old["ret_conditioned"][m])
            per_fold[int(fold)]["checks_archived"] = check_rates(old[m])

    pos = {}
    for system, col in [("conditioned", "action_conditioned"), ("blind", "action_blind")]:
        a = new[col].to_numpy()
        turn = np.abs(np.diff(np.concatenate([[0.0], a])))
        pos[system] = {
            "mean": float(a.mean()),
            "mean_abs": float(np.abs(a).mean()),
            "share_long": float(np.mean(a > 0.05)),
            "share_short": float(np.mean(a < -0.05)),
            "turnover_per_bar": float(turn.mean()),
            "by_regime": {r: float(a[labels == r].mean()) if np.any(labels == r) else None for r in REGIMES},
        }

    curves = iv["curve"]
    mean_curve = {r: curves[labels == r].mean(axis=0).tolist() for r in REGIMES if np.any(labels == r)}
    analysis = {
        "checks_w2": check_rates(new),
        "checks_w2_by_regime": {r: check_rates(new[labels == r]) for r in REGIMES if np.any(labels == r)},
        "checks_archived": check_rates(old) if old is not None else None,
        "checks_archived_by_regime": (
            {r: check_rates(old[old["regime"] == r]) for r in REGIMES if np.any(old["regime"] == r)} if old is not None else None
        ),
        "archived_regime_counts": old["regime"].value_counts().to_dict() if old is not None else None,
        "archived_sharpe_conditioned": sharpe(old["ret_conditioned"]) if old is not None else None,
        "counterfactual_split": {
            "down_needed": int(down_needed.sum()),
            "down_ok_rate": float(down_ok[down_needed].mean()),
            "up_needed": int(up_needed.sum()),
            "up_ok_rate": float(up_ok[up_needed].mean()),
            "median_actual_minus_forced_crisis": float(np.median(iv["actual"] - iv["forced_crisis"])),
            "median_forced_bull_minus_actual": float(np.median(iv["forced_bull"] - iv["actual"])),
            "median_forced": {r: float(np.median(iv[f"forced_{r}"])) for r in REGIMES},
            "median_actual": float(np.median(iv["actual"])),
            "median_ablated": float(np.median(iv["ablated"])),
            "forced_crisis_below_forced_calm": float(np.mean(iv["forced_crisis"] < iv["forced_calm"] - CF_MIN_DELTA)),
            "forced_bear_below_forced_calm": float(np.mean(iv["forced_bear"] < iv["forced_calm"] - CF_MIN_DELTA)),
        },
        "ablation": {
            "threshold": ABLATION_MIN_DELTA,
            "median_delta": float(new["ablation_delta"].median()),
            "mean_delta": float(new["ablation_delta"].mean()),
        },
        "sensitivity_mean_curve": mean_curve,
        "reference_policies_w2": reference_policies(probs),
        "reference_policies_archived": reference_policies(old[list(REGIMES)].to_numpy(dtype=float)) if old is not None else None,
        "argmax_vs_dwell_label_agreement": argmax_agree,
        "mean_max_prob": float(probs.max(axis=1).mean()),
        "per_fold": per_fold,
        "positions": pos,
    }
    (PAPER / "results" / "analysis.json").write_text(json.dumps(analysis, indent=2))
    np.savez(PAPER / "results" / "analysis.npz", **iv)
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

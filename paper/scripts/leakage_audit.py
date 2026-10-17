"""Leakage audit of the published four-hour run: the same weights under three regime inputs.

  published  tracker v1 as published: forward-backward smoothing inside each 42-bar
             window, so bar t has seen later bars of its window, next_ret included.
  lagged     the published probabilities delayed by one bar.
  filtered   the same tracker (burn-in 252, raw-unit matching) with filtered
             probabilities, so only the posterior changes.

A signal that is real survives a one-bar delay; a look-ahead does not. Nothing is
retrained. The published fold weights in paper/results/run are replayed on the frozen
frame, and the published variant is checked against the archived positions first.

Usage: .venv/Scripts/python paper/scripts/leakage_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.agent import rollout  # noqa: E402
from crta.baselines import regime_rule_position  # noqa: E402
from crta.data import HMM_COLS  # noqa: E402
from crta.eval import max_drawdown, strategy_returns, walk_forward_bounds  # noqa: E402
from crta.leakage import lag_rows  # noqa: E402
from crta.regime import LEGACY_V1, REGIMES, track  # noqa: E402
from rerun_4h import circular_block_ci  # noqa: E402

FRAME = ROOT / "paper" / "results" / "4h_rerun" / "frame.parquet"
PUBLISHED_LABELS = ROOT / "paper" / "results" / "4h_rerun" / "labels.npz"
RUN = ROOT / "paper" / "results" / "run"
OUT = ROOT / "paper" / "results" / "causal" / "leakage_audit"
MIN_TRAIN = 1080
# Position mismatch allowed between the replay and the archived run (float32 policy output).
REPLAY_TOL = 1e-4


def _metrics(ret: np.ndarray, act: np.ndarray) -> dict:
    return {
        "sharpe": circular_block_ci(ret, "sharpe"),
        "total_return": float(np.exp(ret.sum()) - 1.0),
        "max_drawdown": max_drawdown(ret),
        "mean_position": float(np.mean(act)),
        "extreme_share": float(np.mean(np.abs(act) > 0.95)),
    }


def _with_probs(frame: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    for i, name in enumerate(REGIMES):
        out[name] = probs[:, i]
    return out


def _replay(frame: pd.DataFrame, probs: np.ndarray, bounds) -> np.ndarray:
    from stable_baselines3 import PPO

    stamped = _with_probs(frame, probs)
    carry = 0.0
    pieces = []
    for fold, (train_end, test_end) in enumerate(bounds):
        model = PPO.load(str(RUN / f"fold{fold}_conditioned.zip"), device="cpu")
        test = stamped.iloc[train_end:test_end].reset_index(drop=True)
        actions = rollout(model, test, True, carry)
        carry = float(actions[-1])
        pieces.append(actions)
    return np.concatenate(pieces)


def main() -> int:
    frame = pd.read_parquet(FRAME)
    X = frame[list(HMM_COLS)].to_numpy(dtype=float)
    bounds = walk_forward_bounds(len(frame), min_train=MIN_TRAIN)
    oos = slice(MIN_TRAIN, len(frame))
    nxt = frame["next_ret"].to_numpy(dtype=float)[oos]

    print("tracker v1 as published (smoothed)", flush=True)
    _labels, published = track(X, **LEGACY_V1)
    stored = np.load(PUBLISHED_LABELS)["probs_v1"]
    if not np.allclose(published, stored, atol=1e-8):
        raise RuntimeError("legacy tracker no longer reproduces the published probabilities")
    print("tracker v1 with filtered probabilities", flush=True)
    _labels, filtered = track(X, burn=LEGACY_V1["burn"], scale="raw", posterior="filtered")
    variants = {"published": published, "lagged_1": lag_rows(published, 1), "filtered": filtered}

    archived = pd.read_parquet(RUN / "bars.parquet")
    results: dict = {"n_oos_bars": int(len(nxt)), "variants": {}}
    for name, probs in variants.items():
        print(f"replaying {name}", flush=True)
        rule = regime_rule_position(probs[oos])
        ppo = _replay(frame, probs, bounds)
        if name == "published":
            gap = float(np.max(np.abs(ppo - archived["action_conditioned"].to_numpy(dtype=float))))
            results["published_replay_max_gap"] = gap
            if gap > REPLAY_TOL:
                raise RuntimeError(f"replay differs from the archived positions by {gap}")
        results["variants"][name] = {
            "regime_rule": _metrics(strategy_returns(rule, nxt), rule),
            "conditioned_ppo": _metrics(strategy_returns(ppo, nxt), ppo),
        }
    blind = archived["action_blind"].to_numpy(dtype=float)
    results["blind_twin_published"] = _metrics(strategy_returns(blind, nxt), blind)
    results["buy_hold"] = _metrics(strategy_returns(np.ones(len(nxt)), nxt), np.ones(len(nxt)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_audit.json").write_text(json.dumps(results, indent=2))
    rows = []
    for system in ("regime_rule", "conditioned_ppo"):
        row = {"system": system}
        for name in variants:
            block = results["variants"][name][system]
            row[f"{name}_sharpe"] = block["sharpe"]["point"]
            row[f"{name}_return"] = block["total_return"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "leakage_audit.csv", index=False)
    print(table.to_string(index=False), flush=True)
    print(f"blind twin Sharpe {results['blind_twin_published']['sharpe']['point']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

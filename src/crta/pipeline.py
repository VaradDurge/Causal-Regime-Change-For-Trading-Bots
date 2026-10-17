"""One entry: data, regimes, optional PPO, verification, figures."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from crta.agent import TIMESTEPS, observation, rollout, train_ppo
from crta.baselines import baseline_positions
from crta.data import FEATURE_COLS, HMM_COLS, RAW_PATH, ROOT, fetch_ohlcv, model_frame, synthetic_ohlcv
from crta.eval import (
    SEED,
    by_regime,
    crash_index,
    performance,
    strategy_returns,
    walk_forward_bounds,
    wealth_curve,
)
from crta.plots import save_figures
from crta.regime import LABELLER_VERSION, REGIMES, track_v2
from crta.verify import audit_many, probe_action

RUN = ROOT / "outputs" / "run"


def load_market(synthetic: bool) -> tuple[pd.DataFrame, str]:
    if synthetic:
        return synthetic_ohlcv(), "synthetic"
    if RAW_PATH.exists():
        df = pd.read_parquet(RAW_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df, "binance"
    try:
        return fetch_ohlcv(), "binance"
    except Exception as exc:
        print(f"binance fetch failed ({exc}); using a synthetic path")
        return synthetic_ohlcv(), "synthetic"


def _stamp(frame, labels, probs, fold: int) -> pd.DataFrame:
    out = frame.copy()
    for i, name in enumerate(REGIMES):
        out[name] = probs[:, i]
    out["regime"] = labels
    out["fold"] = fold
    return out


def _audit(out: pd.DataFrame, mode: str, models: dict) -> np.ndarray:
    n = len(out)
    delta = np.zeros(n)
    pass_ab = np.zeros(n, dtype=bool)
    pass_cf = np.zeros(n, dtype=bool)
    pass_sens = np.zeros(n, dtype=bool)
    passed = np.zeros(n, dtype=bool)
    curves = np.zeros((n, 11))
    for fold in sorted(out["fold"].unique()):
        idx = np.flatnonzero(out["fold"].to_numpy() == fold)
        sub = out.iloc[idx]
        feats = sub[list(FEATURE_COLS)].to_numpy(dtype=float)
        pos = sub["position_before"].to_numpy(dtype=float)
        rows = len(sub)
        if mode == "probe":

            def predict(batch, _rows=rows):
                return np.array([probe_action(row) for row in batch])

        else:
            model = models[int(fold)]

            def predict(batch, model=model, feats=feats, pos=pos, rows=rows):
                obs = np.vstack(
                    [
                        observation(feats[i % rows], batch[i], float(pos[i % rows]), True)
                        for i in range(len(batch))
                    ]
                )
                action, _ = model.predict(obs, deterministic=True)
                return np.asarray(action, dtype=float).reshape(-1)

        result = audit_many(predict, sub[list(REGIMES)].to_numpy(dtype=float))
        delta[idx] = result["ablation_delta"]
        pass_ab[idx] = result["pass_ablation"]
        pass_cf[idx] = result["pass_cf"]
        pass_sens[idx] = result["pass_sens"]
        passed[idx] = result["passed"]
        curves[idx] = result["curve"]
    out["ablation_delta"] = delta
    out["pass_ablation"] = pass_ab
    out["pass_cf"] = pass_cf
    out["pass_sens"] = pass_sens
    out["passed"] = passed
    return curves


def build(mode: str = "trained", timesteps: int = TIMESTEPS, synthetic: bool = False) -> dict:
    raw, source = load_market(synthetic)
    frame = model_frame(raw)
    min_train = 1080 if len(frame) > 1280 else max(80, len(frame) // 3)
    bounds = walk_forward_bounds(len(frame), min_train=min_train)
    RUN.mkdir(parents=True, exist_ok=True)

    pieces = []
    models: dict = {}
    carry_c = 0.0
    carry_b = 0.0
    # One causal pass with filtered probabilities. Folds slice it; a test bar is never fit
    # on rows after its window. Fold 0's training rows are the burn-in, so no test bar is spent.
    labels, label_probs = track_v2(frame[list(HMM_COLS)].to_numpy(), burn=min_train)
    for fold, (train_end, test_end) in enumerate(bounds):
        # Features at t do not include the next close. The last train reward does.
        train = frame.iloc[:train_end].reset_index(drop=True)
        test = frame.iloc[train_end:test_end].reset_index(drop=True)
        train_s = _stamp(train, labels[:train_end], label_probs[:train_end], fold)
        test_s = _stamp(test, labels[train_end:test_end], label_probs[train_end:test_end], fold)
        print(f"fold {fold}: train {len(train)} test {len(test)}", flush=True)
        if mode == "probe":
            actions = np.array([probe_action(p) for p in label_probs[train_end:test_end]])
            blind = None
        else:
            steps = max(timesteps, 10 * len(train))
            model_c = train_ppo(train_s, True, timesteps=steps)
            print(f"fold {fold} conditioned trained ({steps} steps)", flush=True)
            model_b = train_ppo(train_s, False, timesteps=steps)
            print(f"fold {fold} blind trained ({steps} steps)", flush=True)
            model_c.save(str(RUN / f"fold{fold}_conditioned"))
            model_b.save(str(RUN / f"fold{fold}_blind"))
            models[fold] = model_c
            actions = rollout(model_c, test_s, True, carry_c)
            blind = rollout(model_b, test_s, False, carry_b)
            carry_b = float(blind[-1])
        carry_c = float(actions[-1])
        test_s = test_s.copy()
        test_s["action_conditioned"] = actions
        if blind is not None:
            test_s["action_blind"] = blind
        pieces.append(test_s)

    out = pd.concat(pieces, ignore_index=True)
    out["position_before"] = np.concatenate([[0.0], out["action_conditioned"].to_numpy()[:-1]])
    out["action_bh"] = 1.0
    ret_c = strategy_returns(out["action_conditioned"], out["next_ret"])
    ret_h = strategy_returns(out["action_bh"], out["next_ret"])
    out["ret_conditioned"] = ret_c
    out["ret_bh"] = ret_h
    out["equity_conditioned"] = wealth_curve(ret_c)
    out["equity_bh"] = wealth_curve(ret_h)
    if "action_blind" in out.columns:
        ret_b = strategy_returns(out["action_blind"], out["next_ret"])
        out["ret_blind"] = ret_b
        out["equity_blind"] = wealth_curve(ret_b)

    curves = _audit(out, mode, models)

    metrics = {
        "conditioned": performance(out["ret_conditioned"]),
        "buy_hold": performance(out["ret_bh"]),
    }
    grouped = {
        "conditioned": by_regime(out["ret_conditioned"], out["regime"]),
        "buy_hold": by_regime(out["ret_bh"], out["regime"]),
    }
    if "ret_blind" in out.columns:
        metrics["blind"] = performance(out["ret_blind"])
        grouped["blind"] = by_regime(out["ret_blind"], out["regime"])
    # Rules see the full close history, so the first test bars already have a lookback.
    first_test = bounds[0][0]
    for name, position in baseline_positions(frame["close"], label_probs).items():
        if name != "buy_hold":
            metrics[name] = performance(strategy_returns(position[first_test:], out["next_ret"]))
    by = {}
    for name in REGIMES:
        by[name] = {"n": grouped["conditioned"][name]["n"]}
        for system, table in grouped.items():
            by[name][system] = {
                "sharpe": table[name]["sharpe"],
                "mean_return": table[name]["mean_return"],
            }

    spot = crash_index(out)
    summary = {
        "model": "probe" if mode == "probe" else "trained PPO",
        "source": source,
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "labeller": LABELLER_VERSION,
        "pass_rate": float(np.mean(out["passed"].to_numpy(dtype=float))),
        "n_bars": int(len(out)),
        "timesteps": None if mode == "probe" else int(timesteps),
        "train_passes": None if mode == "probe" else 10,
        "seed": SEED,
        "metrics": metrics,
        "by_regime": by,
        "spotlight": pd.Timestamp(out.iloc[spot]["timestamp"]).isoformat(),
    }
    curve = curves[spot]
    out.to_parquet(RUN / "bars.parquet", index=False)
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))
    save_figures(out, summary, spot, curve)
    print(
        json.dumps(
            {"model": summary["model"], "source": source, "pass_rate": summary["pass_rate"], "bars": summary["n_bars"]},
            indent=2,
        ),
        flush=True,
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit regimes, train the twins, write figures.")
    parser.add_argument("--probe", action="store_true", help="Audit the probe policy and skip PPO.")
    parser.add_argument("--synthetic", action="store_true", help="Skip Binance and use the built-in path.")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    args = parser.parse_args(argv)
    build("probe" if args.probe else "trained", timesteps=args.timesteps, synthetic=args.synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

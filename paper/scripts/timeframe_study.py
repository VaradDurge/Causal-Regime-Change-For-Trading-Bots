"""Twins trained on their own timeframe and asset: 15 min and 1 h candles, BTC and ETH.

The tracker keeps the calendar spans of the 4 h study (six-week burn-in, one-week
stride, five-day dwell). Training draws random 2,000-bar episodes with a scaled reward
and a small entropy bonus; a single pass over five years of an uptrend otherwise
collapses both twins to buy-and-hold.

  validate: train before VAL_START, score [VAL_START, HOLDOUT_START). Inside training data.
  final:    train before HOLDOUT_START, score the next year and the frozen live window.

Usage: .venv/bin/python paper/scripts/timeframe_study.py --asset BTC --tf 1h --stage validate
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import live_study as L
import numpy as np
import pandas as pd

from crta.agent import train_ppo
from crta.data import FEATURE_COLS, HMM_COLS, features
from crta.eval import strategy_returns
from crta.regime import REGIMES, track

OUT = L.PAPER / "results" / "tf"
HISTORY_START = pd.Timestamp("2021-01-01", tz="UTC")
VAL_START = pd.Timestamp("2024-09-22 16:00", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-09-22 16:00", tz="UTC")
SYMBOLS = {"BTC": "BTC/USDT", "ETH": "ETH/USDT"}
# Calendar spans of the 4 h tracker (252, 42, 30 bars) in each timeframe's own bars.
CAL = {
    "1h": {"burn": 1008, "stride": 168, "dwell": 120},
    "15m": {"burn": 4032, "stride": 672, "dwell": 480},
}
# ent_coef 0.05 was also tried on validation only (results/tf/val_ent005) and collapsed both
# 15 min tapes to a constant position, so 0.01 stays.
RECIPE = {"window": 2000, "reward_scale": 100.0, "ent_coef": 0.01, "timesteps": 500_000}


def candles(asset: str, tf: str) -> pd.DataFrame:
    live = L.snapshot(tf, False)
    end = live["timestamp"].iloc[-1]
    path = OUT / f"{asset.lower()}_usdt_{tf}_history.parquet"
    if path.exists():
        hist = pd.read_parquet(path)
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    else:
        hist = L._fetch(tf, HISTORY_START, SYMBOLS[asset])
        cut = live["timestamp"].iloc[0] if asset == "BTC" else end
        keep = hist["timestamp"] < cut if asset == "BTC" else hist["timestamp"] <= cut
        hist = hist.loc[keep].reset_index(drop=True)
        OUT.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(path, index=False)
    df = pd.concat([hist, live], ignore_index=True) if asset == "BTC" else hist
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def labelled(asset: str, tf: str, raw: pd.DataFrame) -> pd.DataFrame:
    framed = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    path = OUT / f"{asset.lower()}_{tf}_labels.parquet"
    if path.exists():
        lab = pd.read_parquet(path)
        lab["timestamp"] = pd.to_datetime(lab["timestamp"], utc=True)
        if len(lab) == len(framed) and lab["timestamp"].equals(framed["timestamp"]):
            for col in [*REGIMES, "regime"]:
                framed[col] = lab[col].to_numpy()
            return framed
    names, probs = track(framed[list(HMM_COLS)].to_numpy(), **CAL[tf])
    for i, name in enumerate(REGIMES):
        framed[name] = probs[:, i]
    framed["regime"] = names
    framed[["timestamp", *REGIMES, "regime"]].to_parquet(path, index=False)
    return framed


def twins(tag: str, train: pd.DataFrame, retrain: bool):
    from stable_baselines3 import PPO

    paths = {side: OUT / f"{tag}_{side}.zip" for side in ("conditioned", "blind")}
    if retrain or not all(p.exists() for p in paths.values()):
        for side, include in (("conditioned", True), ("blind", False)):
            t0 = time.time()
            model = train_ppo(
                train,
                include,
                timesteps=RECIPE["timesteps"],
                window=RECIPE["window"],
                reward_scale=RECIPE["reward_scale"],
                ent_coef=RECIPE["ent_coef"],
                threads=2,
            )
            model.save(str(paths[side].with_suffix("")))
            print(f"{tag} {side}: {RECIPE['timesteps']} steps in {time.time() - t0:.0f}s", flush=True)
    return {side: PPO.load(str(p), device="cpu") for side, p in paths.items()}


def _likely(frame: pd.DataFrame) -> np.ndarray:
    return np.array(REGIMES)[frame[list(REGIMES)].to_numpy(dtype=float).argmax(axis=1)]


def score(tf: str, models: dict, seg: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    a_c, before_c = L.trade(models["conditioned"], seg, True)
    a_b, _ = L.trade(models["blind"], seg, False)
    a_h = np.ones(len(seg))
    nxt = seg["next_ret"].to_numpy(dtype=float)
    r_c, r_b, r_h = (strategy_returns(a, nxt) for a in (a_c, a_b, a_h))
    # The five-day dwell outlasts most fast-candle bear and crisis runs, so group by p_t.
    likely = _likely(seg)
    regimes = {}
    for name in REGIMES:
        m = likely == name
        regimes[name] = {
            "share": float(m.mean()),
            "cond_position": float(a_c[m].mean()) if m.any() else None,
            "blind_position": float(a_b[m].mean()) if m.any() else None,
            "cond_return": float(np.exp(r_c[m].sum()) - 1.0) if m.any() else None,
            "blind_return": float(np.exp(r_b[m].sum()) - 1.0) if m.any() else None,
        }
    stats = {
        "n_bars": len(seg),
        "first_bar_utc": seg["timestamp"].iloc[0].isoformat(),
        "last_bar_utc": seg["timestamp"].iloc[-1].isoformat(),
        "asset_return": float(np.exp(nxt.sum()) - 1.0),
        "conditioned": L.system_stats(r_c, a_c, L.PERIODS[tf]),
        "blind": L.system_stats(r_b, a_b, L.PERIODS[tf]),
        "buy_hold": L.system_stats(r_h, a_h, L.PERIODS[tf]),
        "position_std": {"conditioned": float(a_c.std()), "blind": float(a_b.std())},
        "gap_cond_minus_blind": L.block_diff(r_c, r_b),
        "gap_cond_minus_hold": L.block_diff(r_c, r_h),
        "regimes": regimes,
        "faithfulness": L.audit_live(models["conditioned"], seg, before_c),
    }
    bars = pd.DataFrame(
        {
            "timestamp": seg["timestamp"],
            "close": seg["close"],
            "regime": likely,
            "action_conditioned": a_c,
            "action_blind": a_b,
            "equity_conditioned": np.exp(np.cumsum(r_c)),
            "equity_blind": np.exp(np.cumsum(r_b)),
            "equity_bh": np.exp(np.cumsum(r_h)),
        }
    )
    return stats, bars


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=sorted(SYMBOLS), required=True)
    parser.add_argument("--tf", choices=sorted(CAL), required=True)
    parser.add_argument("--stage", choices=("validate", "final"), required=True)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    asset, tf, stage = args.asset, args.tf, args.stage
    tag = f"{asset.lower()}_{tf}" + ("_val" if stage == "validate" else "")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    framed = labelled(asset, tf, candles(asset, tf))
    print(f"{tag}: {len(framed)} labelled rows, {time.time() - t0:.0f}s", flush=True)
    ts = framed["timestamp"]
    usable = framed["next_ret"].notna()
    cut = VAL_START if stage == "validate" else HOLDOUT_START
    train = framed.loc[(ts < cut) & usable].reset_index(drop=True)
    models = twins(tag, train, args.retrain)
    if stage == "validate":
        segments = {"validation": framed.loc[(ts >= VAL_START) & (ts < HOLDOUT_START) & usable]}
    else:
        segments = {
            "holdout": framed.loc[(ts >= HOLDOUT_START) & (ts < L.LIVE_START) & usable],
            "live": framed.loc[(ts >= L.LIVE_START) & usable],
        }
    likely = _likely(train.iloc[CAL[tf]["burn"]:])
    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "asset": asset,
        "timeframe": tf,
        "stage": stage,
        "tracker": CAL[tf],
        "recipe": RECIPE,
        "train": {
            "n_bars": len(train),
            "first_bar_utc": train["timestamp"].iloc[0].isoformat(),
            "last_bar_utc": train["timestamp"].iloc[-1].isoformat(),
            "likely_regime_share": {g: float((likely == g).mean()) for g in REGIMES},
        },
    }
    frames = []
    for name, seg in segments.items():
        report[name], bars = score(tf, models, seg.reset_index(drop=True))
        bars.insert(0, "segment", name)
        frames.append(bars)
        print(f"{tag} {name}: {len(seg)} bars, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "timeframe", tf)
    out.insert(0, "asset", asset)
    out.to_parquet(OUT / f"{tag}_bars.parquet", index=False)
    (OUT / f"{tag}_{stage}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

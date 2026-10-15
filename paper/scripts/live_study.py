"""Live paper-trading comparison of the conditioned and blind twins on Binance BTC/USDT.

Every scored candle opened after the last bar of the research run, so neither the
agents (fold 3, trained on bars before 3 Jun 2025) nor the walk-forward study saw it.
Regime labels come from the same causal tracker as the paper (burn-in 252, stride 42),
run on each timeframe's own candles. The candles are frozen to paper/results/live so
the numbers can be reproduced; pass --refresh to fetch a new snapshot.

Usage: .venv/bin/python paper/scripts/live_study.py [--refresh]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crta.agent import observation
from crta.data import FEATURE_COLS, HMM_COLS, RAW_PATH, features
from crta.eval import TURNOVER_COST, bootstrap_ci, max_drawdown, strategy_returns
from crta.regime import BURN, REGIMES, STRIDE, track
from crta.verify import audit_many

PAPER = Path(__file__).resolve().parents[1]
RUN = PAPER / "results" / "run"
OUT = PAPER / "results" / "live"
LIVE_START = pd.Timestamp("2026-09-22 16:00", tz="UTC")
PERIODS = {"15m": 4 * 24 * 365, "1h": 24 * 365, "4h": 6 * 365}
HISTORY = {"15m": 1200, "1h": 1200}
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _step(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(timeframe.replace("m", "min"))


def _fetch(timeframe: str, since: pd.Timestamp, symbol: str = "BTC/USDT") -> pd.DataFrame:
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    width = int(ex.parse_timeframe(timeframe) * 1000)
    since_ms = int(since.timestamp() * 1000)
    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        since_ms = batch[-1][0] + 1
    now = int(ex.milliseconds())
    rows = [r for r in rows if int(r[0]) + width <= now]
    df = pd.DataFrame(rows, columns=COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def snapshot(timeframe: str, refresh: bool) -> pd.DataFrame:
    path = OUT / f"btc_usdt_{timeframe}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    if timeframe == "4h":
        df = pd.read_parquet(RAW_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        tail = _fetch("4h", df["timestamp"].iloc[-1])
        df = pd.concat([df.loc[df["timestamp"] < tail["timestamp"].min()], tail], ignore_index=True)
    else:
        df = _fetch(timeframe, LIVE_START - HISTORY[timeframe] * _step(timeframe))
    df = df[COLS].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def labelled(raw: pd.DataFrame) -> pd.DataFrame:
    framed = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    names, probs = track(framed[list(HMM_COLS)].to_numpy())
    for i, name in enumerate(REGIMES):
        framed[name] = probs[:, i]
    framed["regime"] = names
    return framed


def trade(model, frame: pd.DataFrame, include_regime: bool) -> tuple[np.ndarray, np.ndarray]:
    feats = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    probs = frame[list(REGIMES)].to_numpy(dtype=float)
    pos = 0.0
    actions, before = [], []
    for i in range(len(frame)):
        obs = observation(feats[i], probs[i], pos, include_regime)
        a, _ = model.predict(obs, deterministic=True)
        before.append(pos)
        pos = float(np.clip(np.asarray(a, dtype=float).reshape(-1)[0], -1.0, 1.0))
        actions.append(pos)
    return np.asarray(actions), np.asarray(before)


def sharpe_at(periods: int):
    def fn(r):
        r = np.asarray(r, dtype=float)
        if len(r) < 2:
            return None
        sd = float(r.std(ddof=1))
        if not np.isfinite(sd) or sd < 1e-12:
            return None
        return float(r.mean() / sd * np.sqrt(periods))

    return fn


def block_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Circular block bootstrap of the summed log-return gap a - b. Blocks keep serial dependence."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    block = max(5, int(np.ceil(np.sqrt(n))))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    sums = d[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {
        "gap_log": float(d.sum()),
        "low": float(lo),
        "high": float(hi),
        "share_positive": float((sums > 0).mean()),
        "block": block,
    }


def audit_live(model, frame: pd.DataFrame, before: np.ndarray) -> dict:
    feats = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    probs = frame[list(REGIMES)].to_numpy(dtype=float)
    n = len(frame)

    def act(batch):
        obs = np.vstack([observation(feats[i % n], batch[i], float(before[i % n]), True) for i in range(len(batch))])
        a, _ = model.predict(obs, deterministic=True)
        return np.clip(np.asarray(a, dtype=float).reshape(-1), -1, 1)

    res = audit_many(act, probs)
    return {
        "ablation": float(res["pass_ablation"].mean()),
        "counterfactual": float(res["pass_cf"].mean()),
        "sensitivity": float(res["pass_sens"].mean()),
        "all": float(res["passed"].mean()),
        "median_ablation_delta": float(np.median(res["ablation_delta"])),
    }


def system_stats(ret: np.ndarray, act: np.ndarray, periods: int) -> dict:
    prev = np.concatenate([[0.0], act[:-1]])
    cost = TURNOVER_COST * np.abs(act - prev)
    return {
        "total_return": float(np.exp(ret.sum()) - 1.0),
        "gross_return": float(np.exp((ret + cost).sum()) - 1.0),
        "sharpe": bootstrap_ci(ret, sharpe_at(periods)),
        "max_drawdown": max_drawdown(ret),
        "mean_position": float(act.mean()),
        "turnover_per_bar": float(np.abs(act - prev).mean()),
        "cost_paid": float(cost.sum()),
    }


def main() -> int:
    from stable_baselines3 import PPO

    refresh = "--refresh" in sys.argv
    t0 = time.time()
    cond = PPO.load(str(RUN / "fold3_conditioned.zip"), device="cpu")
    blind = PPO.load(str(RUN / "fold3_blind.zip"), device="cpu")
    report: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "live_start_utc": LIVE_START.isoformat(),
        "agents": "paper/results/run fold3 (trained on bars before 2025-06-03)",
        "tracker": {"burn": BURN, "stride": STRIDE},
        "timeframes": {},
    }
    bars = []
    for tf in ("15m", "1h", "4h"):
        raw = snapshot(tf, refresh)
        framed = labelled(raw)
        live = framed.loc[(framed["timestamp"] >= LIVE_START) & framed["next_ret"].notna()].reset_index(drop=True)
        a_c, before_c = trade(cond, live, True)
        a_b, _ = trade(blind, live, False)
        a_h = np.ones(len(live))
        nxt = live["next_ret"].to_numpy(dtype=float)
        r_c = strategy_returns(a_c, nxt)
        r_b = strategy_returns(a_b, nxt)
        r_h = strategy_returns(a_h, nxt)
        labels = live["regime"].to_numpy()
        probs = live[list(REGIMES)].to_numpy(dtype=float)
        regimes = {}
        for name in REGIMES:
            m = labels == name
            regimes[name] = {
                "share": float(m.mean()),
                "cond_position": float(a_c[m].mean()) if m.any() else None,
                "blind_position": float(a_b[m].mean()) if m.any() else None,
                "cond_return": float(np.exp(r_c[m].sum()) - 1.0) if m.any() else None,
                "blind_return": float(np.exp(r_b[m].sum()) - 1.0) if m.any() else None,
            }
        step = _step(tf)
        report["timeframes"][tf] = {
            "n_bars": len(live),
            "first_bar_utc": live["timestamp"].iloc[0].isoformat(),
            "last_bar_utc": live["timestamp"].iloc[-1].isoformat(),
            "settled_through_utc": (live["timestamp"].iloc[-1] + 2 * step).isoformat(),
            "history_bars": int((framed["timestamp"] < LIVE_START).sum()),
            "btc_return": float(np.exp(nxt.sum()) - 1.0),
            "conditioned": system_stats(r_c, a_c, PERIODS[tf]),
            "blind": system_stats(r_b, a_b, PERIODS[tf]),
            "buy_hold": system_stats(r_h, a_h, PERIODS[tf]),
            "gap_cond_minus_blind": block_diff(r_c, r_b),
            "gap_cond_minus_hold": block_diff(r_c, r_h),
            "regimes": regimes,
            "mean_max_prob": float(probs.max(axis=1).mean()),
            "position_corr": float(np.corrcoef(a_c, a_b)[0, 1]) if a_c.std() > 0 and a_b.std() > 0 else None,
            "faithfulness": audit_live(cond, live, before_c),
        }
        bars.append(
            pd.DataFrame(
                {
                    "timeframe": tf,
                    "timestamp": live["timestamp"],
                    "close": live["close"],
                    "regime": labels,
                    "action_conditioned": a_c,
                    "action_blind": a_b,
                    "equity_conditioned": np.exp(np.cumsum(r_c)),
                    "equity_blind": np.exp(np.cumsum(r_b)),
                    "equity_bh": np.exp(np.cumsum(r_h)),
                }
            )
        )
        print(f"{tf}: {len(live)} live bars, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(bars, ignore_index=True).to_parquet(OUT / "live_bars.parquet", index=False)
    (OUT / "live_study.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

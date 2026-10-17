"""Pre-clip policy means on the validation split of the native twins.

The executed position is the deterministic Gaussian mean clipped to [-1, 1].
Validation weights were not in the repo (the final-stage zips are a later cut,
and val_ent005 is the rejected entropy-0.05 run), so this retrains seed 42
with the frozen native recipe and labels the source
``retrained_seed42_validation``.

Row order for the three-line decision file, and for the CSV sort:
BTC then ETH, 1h then 15m, conditioned then blind.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.agent import observation, save_twin, train_ppo  # noqa: E402
from crta.data import FEATURE_COLS  # noqa: E402
from crta.regime import REGIMES  # noqa: E402

import timeframe_study as study  # noqa: E402

WEIGHT_SOURCE = "retrained_seed42_validation"
RECIPE = {
    "reward_mode": "raw",
    "reward_scale": 100.0,
    "ent_coef": 0.01,
    "window": 2000,
    "window_sampling": "uniform",
    "tracker": "v1",
    "timesteps": 500_000,
    "threads": 2,
    "seed": 42,
}
TAPES = (("BTC", "1h"), ("BTC", "15m"), ("ETH", "1h"), ("ETH", "15m"))
TWINS = ("conditioned", "blind")
WEIGHTS = ROOT / "paper" / "results" / "saturation" / "weights"
CANON_CSV = ROOT / "results" / "saturation_diagnostic.csv"
COPY_CSV = ROOT / "paper" / "results" / "saturation_diagnostic.csv"
DECISION = ROOT / "results" / "saturation_decision.txt"
DECISION_COPY = ROOT / "paper" / "results" / "saturation_decision.txt"
RAW_PATH = ROOT / "results" / "saturation_raw_means.parquet"
RAW_COPY = ROOT / "paper" / "results" / "saturation_raw_means.parquet"


def weight_file(asset: str, timeframe: str, twin: str) -> Path:
    return WEIGHTS / f"{asset.lower()}_{timeframe}_{twin}.zip"


def split_frames(asset: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = study.candles(asset, timeframe)
    framed = study.labelled(asset, timeframe, raw)
    ts = pd.to_datetime(framed["timestamp"], utc=True)
    framed = framed.copy()
    framed["timestamp"] = ts
    usable = framed["next_ret"].notna()
    train = framed.loc[(ts < study.VAL_START) & usable].reset_index(drop=True)
    val = framed.loc[(ts >= study.VAL_START) & (ts < study.HOLDOUT_START) & usable].reset_index(drop=True)
    if len(train) < RECIPE["window"] or len(val) < 2:
        raise RuntimeError(f"{asset} {timeframe}: train {len(train)} val {len(val)}")
    if train["timestamp"].iloc[-1] >= study.VAL_START:
        raise RuntimeError(f"{asset} {timeframe}: training row on or after the validation cut")
    if val["timestamp"].iloc[0] < study.VAL_START or val["timestamp"].iloc[-1] >= study.HOLDOUT_START:
        raise RuntimeError(f"{asset} {timeframe}: validation window is not inside the split")
    return train, val


def train_one(asset: str, timeframe: str, twin: str) -> Path:
    path = weight_file(asset, timeframe, twin)
    meta_path = path.with_suffix(".meta.json")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("weight_source") == WEIGHT_SOURCE and int(meta.get("seed", -1)) == RECIPE["seed"]:
            print(f"keep {path.name}", flush=True)
            return path
    train, _val = split_frames(asset, timeframe)
    t0 = time.time()
    model = train_ppo(
        train,
        twin == "conditioned",
        timesteps=RECIPE["timesteps"],
        seed=RECIPE["seed"],
        window=RECIPE["window"],
        reward_scale=RECIPE["reward_scale"],
        ent_coef=RECIPE["ent_coef"],
        threads=RECIPE["threads"],
        reward_mode=RECIPE["reward_mode"],
        window_sampling=RECIPE["window_sampling"],
        tracker=RECIPE["tracker"],
    )
    model.crta_meta.update(
        {
            "weight_source": WEIGHT_SOURCE,
            "stage": "validate",
            "seed": RECIPE["seed"],
            "asset": asset,
            "timeframe": timeframe,
            "twin": twin,
            "train_before_utc": study.VAL_START.isoformat(),
            "train_first_bar_utc": train["timestamp"].iloc[0].isoformat(),
            "train_last_bar_utc": train["timestamp"].iloc[-1].isoformat(),
            "n_train": int(len(train)),
            "timesteps": RECIPE["timesteps"],
            "window": RECIPE["window"],
        }
    )
    save_twin(model, path)
    print(
        f"trained {asset} {timeframe} {twin}: {len(train)} rows, "
        f"last {train['timestamp'].iloc[-1].isoformat()}, {time.time() - t0:.0f}s",
        flush=True,
    )
    return path


def _raw_means(model, obs: np.ndarray) -> np.ndarray:
    import torch as th

    tensor, _vectorized = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        dist = model.policy.get_distribution(tensor)
        mean = dist.mode()
    return mean.detach().cpu().numpy().reshape(-1)


def rollout_raw(model, frame: pd.DataFrame, include_regime: bool) -> tuple[np.ndarray, np.ndarray]:
    """Sequential rollout. The next observation sees the clipped position, not the raw mean."""
    feats = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    probs = frame[list(REGIMES)].to_numpy(dtype=float)
    raw = np.empty(len(frame))
    executed = np.empty(len(frame))
    pos = 0.0
    check_at = set(np.linspace(0, len(frame) - 1, num=min(32, len(frame)), dtype=int).tolist())
    for i in range(len(frame)):
        obs = observation(feats[i], probs[i], pos, include_regime)
        mean = float(_raw_means(model, obs.reshape(1, -1))[0])
        clipped = float(np.clip(mean, -1.0, 1.0))
        if i in check_at:
            pred, _ = model.predict(obs, deterministic=True)
            pred = float(np.clip(np.asarray(pred, dtype=float).reshape(-1)[0], -1.0, 1.0))
            if abs(pred - clipped) > 1e-5:
                raise RuntimeError(f"clipped mean {clipped} != deterministic predict {pred} at bar {i}")
        raw[i] = mean
        executed[i] = clipped
        pos = clipped
    return raw, executed


def summarize(raw: np.ndarray, executed: np.ndarray) -> dict:
    delta = np.abs(np.diff(executed))
    return {
        "n_bars": int(len(executed)),
        "share_raw_abs_gt_1": float(np.mean(np.abs(raw) > 1.0)),
        "share_exec_abs_gt_095": float(np.mean(np.abs(executed) > 0.95)),
        "mean_abs_delta_position": float(delta.mean()) if len(delta) else float("nan"),
    }


def row_order() -> list[tuple[str, str, str]]:
    return [(asset, tf, twin) for asset, tf in TAPES for twin in TWINS]


def diagnose() -> pd.DataFrame:
    from stable_baselines3 import PPO

    rows = []
    pieces = []
    for asset, timeframe, twin in row_order():
        path = train_one(asset, timeframe, twin)
        _train, val = split_frames(asset, timeframe)
        model = PPO.load(str(path), device="cpu")
        sidecar = json.loads(path.with_suffix(".meta.json").read_text())
        model.crta_meta = sidecar
        t0 = time.time()
        raw, executed = rollout_raw(model, val, twin == "conditioned")
        stats = summarize(raw, executed)
        print(
            f"{asset} {timeframe} {twin}: raw>1 {stats['share_raw_abs_gt_1']:.4f} "
            f"exec>0.95 {stats['share_exec_abs_gt_095']:.4f} "
            f"|da| {stats['mean_abs_delta_position']:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        rows.append(
            {
                "asset": asset,
                "timeframe": timeframe,
                "twin": twin,
                **stats,
                "weight_source": WEIGHT_SOURCE,
            }
        )
        pieces.append(
            pd.DataFrame(
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "twin": twin,
                    "timestamp": val["timestamp"].to_numpy(),
                    "raw_mean": raw,
                    "executed": executed,
                }
            )
        )
    table = pd.DataFrame(rows)
    CANON_CSV.parent.mkdir(parents=True, exist_ok=True)
    COPY_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CANON_CSV, index=False)
    table.to_csv(COPY_CSV, index=False)
    pd.concat(pieces, ignore_index=True).to_parquet(RAW_PATH, index=False)
    pd.concat(pieces, ignore_index=True).to_parquet(RAW_COPY, index=False)
    write_decision(table)
    return table


def write_decision(table: pd.DataFrame) -> str:
    ordered = []
    for asset, timeframe, twin in row_order():
        hit = table[(table.asset == asset) & (table.timeframe == timeframe) & (table.twin == twin)]
        if len(hit) != 1:
            raise RuntimeError(f"missing decision row {asset} {timeframe} {twin}")
        ordered.append(hit.iloc[0])
    raw_shares = [float(row.share_raw_abs_gt_1) for row in ordered]
    exec_shares = [float(row.share_exec_abs_gt_095) for row in ordered]
    deltas = [float(row.mean_abs_delta_position) for row in ordered]
    n_cause = sum(share >= 0.50 and executed >= 0.80 for share, executed in zip(raw_shares, exec_shares))
    verdict = "IS" if n_cause >= 6 else "NOT"
    def fmt(values: list[float]) -> str:
        return " ".join(f"{value:.6f}" for value in values)

    text = fmt(raw_shares) + "\n" + fmt(exec_shares + deltas) + "\n" + verdict + "\n"
    DECISION.parent.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(text)
    DECISION_COPY.write_text(text)
    if len(text.splitlines()) != 3:
        raise RuntimeError("decision file is not exactly 3 lines")
    print(text, end="", flush=True)
    print(f"rows meeting both thresholds: {n_cause} of 8 -> {verdict}", flush=True)
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--asset", choices=[a for a, _ in TAPES])
    parser.add_argument("--tf", choices=["1h", "15m"])
    parser.add_argument("--twin", choices=list(TWINS))
    args = parser.parse_args()
    if args.train_only:
        if not (args.asset and args.tf and args.twin):
            raise SystemExit("--train-only needs --asset --tf --twin")
        train_one(args.asset, args.tf, args.twin)
        return 0
    diagnose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

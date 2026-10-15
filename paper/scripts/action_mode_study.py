"""Screen tanh_env and discrete5 against the clipped Gaussian, then score one mode.

Selection reads the validation split only. gaussian_clip reuses the seed-42
validation weights from the saturation diagnostic. Holdout and live are scored
only after results/action_mode_choice.txt exists, and only for the frozen mode.

Tie-break, fixed before the screen: higher minimum ablation across the four
tapes, then higher minimum twin position std, then tanh_env, discrete5,
gaussian_clip. Returns are not an input.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.agent import audit_positions, batch_observation, load_twin, rollout, save_twin, train_ppo  # noqa: E402
from crta.data import FEATURE_COLS  # noqa: E402
from crta.eval import strategy_returns  # noqa: E402
from crta.regime import REGIMES  # noqa: E402
from crta.select import choose_action_mode, meets_criterion  # noqa: E402
from crta.verify import audit_many, audit_many_v2, empirical_banks  # noqa: E402

import live_study as live  # noqa: E402
import saturation_diagnostic as sat  # noqa: E402
import timeframe_study as study  # noqa: E402

MODES = ("gaussian_clip", "tanh_env", "discrete5")
TRAIN_MODES = ("tanh_env", "discrete5")
SEEDS = (42, 43, 44, 45, 46)
PERIODS = {"15m": 4 * 24 * 365, "1h": 24 * 365}
VAL_DIR = ROOT / "paper" / "results" / "action_mode" / "validation"
FINAL_DIR = ROOT / "paper" / "results" / "action_mode" / "final"
CHOICE = ROOT / "results" / "action_mode_choice.txt"
CHOICE_JSON = ROOT / "results" / "action_mode_choice.json"
SCREEN_CSV = ROOT / "results" / "action_mode_screen.csv"
HOLDOUT_CSV = ROOT / "results" / "action_mode_holdout.csv"


def _copy(path: Path) -> None:
    dest = ROOT / "paper" / "results" / path.name
    dest.write_bytes(path.read_bytes())


def _sharpe(ret, periods: int):
    r = np.asarray(ret, dtype=float)
    if len(r) < 2:
        return None
    scale = float(r.std(ddof=1))
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    return float(r.mean() / scale * np.sqrt(periods))


def validation_bundle(asset: str, timeframe: str):
    train, val = sat.split_frames(asset, timeframe)
    return train, val


def final_bundle(asset: str, timeframe: str):
    raw = study.candles(asset, timeframe)
    framed = study.labelled(asset, timeframe, raw)
    ts = pd.to_datetime(framed["timestamp"], utc=True)
    framed = framed.copy()
    framed["timestamp"] = ts
    usable = framed["next_ret"].notna()
    train = framed.loc[(ts < study.HOLDOUT_START) & usable].reset_index(drop=True)
    holdout = framed.loc[(ts >= study.HOLDOUT_START) & (ts < live.LIVE_START) & usable].reset_index(drop=True)
    live_seg = framed.loc[(ts >= live.LIVE_START) & usable].reset_index(drop=True)
    if train["timestamp"].iloc[-1] >= study.HOLDOUT_START:
        raise RuntimeError("final train window crosses the holdout cut")
    return train, {"holdout": holdout, "live": live_seg}


def weight_path(stage: str, mode: str, asset: str, timeframe: str, twin: str, seed: int) -> Path:
    name = f"{asset.lower()}_{timeframe}_{twin}.zip"
    if stage == "validate":
        if mode == "gaussian_clip":
            return sat.weight_file(asset, timeframe, twin)
        return VAL_DIR / mode / name
    return FINAL_DIR / mode / f"seed_{seed}" / name


def train_one(stage: str, mode: str, asset: str, timeframe: str, twin: str, seed: int) -> Path:
    if mode == "gaussian_clip" and stage == "validate":
        path = sat.weight_file(asset, timeframe, twin)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    path = weight_path(stage, mode, asset, timeframe, twin, seed)
    meta_path = path.with_suffix(".meta.json")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("action_mode") == mode
            and int(meta.get("seed", -1)) == seed
            and meta.get("stage") == stage
            and int(meta.get("timesteps", -1)) == sat.RECIPE["timesteps"]
        ):
            print(f"keep {path}", flush=True)
            return path
    if stage == "validate":
        train, _val = validation_bundle(asset, timeframe)
        cut = study.VAL_START
    else:
        train, _segments = final_bundle(asset, timeframe)
        cut = study.HOLDOUT_START
    t0 = time.time()
    model = train_ppo(
        train,
        twin == "conditioned",
        timesteps=sat.RECIPE["timesteps"],
        seed=seed,
        window=sat.RECIPE["window"],
        reward_scale=sat.RECIPE["reward_scale"],
        ent_coef=sat.RECIPE["ent_coef"],
        threads=sat.RECIPE["threads"],
        reward_mode=sat.RECIPE["reward_mode"],
        window_sampling=sat.RECIPE["window_sampling"],
        tracker=sat.RECIPE["tracker"],
        action_mode=mode,
    )
    model.crta_meta.update(
        {
            "stage": stage,
            "seed": int(seed),
            "asset": asset,
            "timeframe": timeframe,
            "twin": twin,
            "train_before_utc": cut.isoformat(),
            "train_last_bar_utc": train["timestamp"].iloc[-1].isoformat(),
            "n_train": int(len(train)),
            "timesteps": sat.RECIPE["timesteps"],
        }
    )
    save_twin(model, path)
    print(f"trained {stage} {mode} {asset} {timeframe} {twin} seed {seed} in {time.time() - t0:.0f}s", flush=True)
    return path


def _load(path: Path):
    model, meta = load_twin(path)
    return model


def _audit(model, seg, actions, banks):
    feats = seg[list(FEATURE_COLS)].to_numpy(dtype=float)
    pos = np.concatenate([[0.0], np.asarray(actions, dtype=float)[:-1]])
    probs = seg[list(REGIMES)].to_numpy(dtype=float)
    n = len(seg)

    def predict(batch):
        batch = np.asarray(batch, dtype=float)
        out = np.empty(len(batch))
        step = 8192
        for start in range(0, len(batch), step):
            sl = slice(start, start + step)
            idx = np.arange(start, min(start + step, len(batch))) % n
            obs = batch_observation(feats[idx], batch[sl], pos[idx], True)
            out[sl] = audit_positions(model, obs)
        return out

    legacy = audit_many(predict, probs)
    v2 = audit_many_v2(predict, probs, banks)
    return legacy, v2


def score_segment(models, seg, timeframe: str, train_probs) -> dict:
    act_c = rollout(models["conditioned"], seg, True)
    act_b = rollout(models["blind"], seg, False)
    nxt = seg["next_ret"].to_numpy(dtype=float)
    ret_c = strategy_returns(act_c, nxt)
    ret_b = strategy_returns(act_b, nxt)
    banks = empirical_banks(train_probs)
    legacy, v2 = _audit(models["conditioned"], seg, act_c, banks)
    periods = PERIODS[timeframe]
    return {
        "n_bars": int(len(seg)),
        "std_c": float(np.std(act_c)),
        "std_b": float(np.std(act_b)),
        "extreme_c": float(np.mean(np.abs(act_c) > 0.95)),
        "extreme_b": float(np.mean(np.abs(act_b) > 0.95)),
        "ablation": float(np.mean(legacy["pass_ablation"])),
        "faithfulness": float(np.mean(legacy["passed"])),
        "faithfulness_v2": float(np.mean(v2["passed"])),
        "return_c": float(np.exp(ret_c.sum()) - 1.0),
        "return_b": float(np.exp(ret_b.sum()) - 1.0),
        "sharpe_c": _sharpe(ret_c, periods),
        "sharpe_b": _sharpe(ret_b, periods),
        "beats": bool(ret_c.sum() > ret_b.sum()),
        "passes": bool(
            meets_criterion(
                float(np.std(act_c)),
                float(np.std(act_b)),
                float(np.mean(np.abs(act_c) > 0.95)),
                float(np.mean(np.abs(act_b) > 0.95)),
                float(np.mean(legacy["pass_ablation"])),
            )
        ),
    }


def _criterion_row(mode: str, asset: str, timeframe: str, stats: dict) -> dict:
    return {
        "mode": mode,
        "asset": asset,
        "timeframe": timeframe,
        "std_c": stats["std_c"],
        "std_b": stats["std_b"],
        "extreme_c": stats["extreme_c"],
        "extreme_b": stats["extreme_b"],
        "ablation": stats["ablation"],
    }


def screen_score() -> dict:
    rows = []
    criterion = []
    for mode in MODES:
        for asset, timeframe in sat.TAPES:
            train, val = validation_bundle(asset, timeframe)
            models = {
                twin: _load(weight_path("validate", mode, asset, timeframe, twin, 42))
                for twin in sat.TWINS
            }
            t0 = time.time()
            stats = score_segment(models, val, timeframe, train[list(REGIMES)].to_numpy(dtype=float))
            print(
                f"screen {mode} {asset} {timeframe}: std {stats['std_c']:.3f}/{stats['std_b']:.3f} "
                f"ext {stats['extreme_c']:.3f}/{stats['extreme_b']:.3f} "
                f"ab {stats['ablation']:.3f} pass {stats['passes']} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            rows.append({"mode": mode, "asset": asset, "timeframe": timeframe, "seed": 42, "segment": "validation", **stats})
            criterion.append(_criterion_row(mode, asset, timeframe, stats))
    choice = choose_action_mode(criterion)
    frame = pd.DataFrame(rows)
    SCREEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SCREEN_CSV, index=False)
    _copy(SCREEN_CSV)
    CHOICE.write_text(choice["mode"] + "\n")
    _copy(CHOICE)
    CHOICE_JSON.write_text(json.dumps(choice, indent=2) + "\n")
    _copy(CHOICE_JSON)
    print(f"frozen mode: {choice['mode']}", flush=True)
    return choice


def _mean_std(values) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def holdout_score(mode: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for asset, timeframe in sat.TAPES:
            train, segments = final_bundle(asset, timeframe)
            models = {
                twin: _load(weight_path("final", mode, asset, timeframe, twin, seed))
                for twin in sat.TWINS
            }
            probs = train[list(REGIMES)].to_numpy(dtype=float)
            for name, seg in segments.items():
                stats = score_segment(models, seg, timeframe, probs)
                rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "asset": asset,
                        "timeframe": timeframe,
                        "segment": name,
                        **stats,
                    }
                )
                print(
                    f"final {mode} seed {seed} {asset} {timeframe} {name}: "
                    f"ret {stats['return_c']:.3f} sharpe {stats['sharpe_c']} ab {stats['ablation']:.3f}",
                    flush=True,
                )
    detail = pd.DataFrame(rows)
    agg_rows = []
    keys = ["return_c", "return_b", "sharpe_c", "sharpe_b", "std_c", "std_b", "extreme_c", "extreme_b", "ablation", "faithfulness", "faithfulness_v2"]
    for (asset, timeframe, segment), group in detail.groupby(["asset", "timeframe", "segment"], sort=False):
        row = {
            "mode": mode,
            "seed": "mean",
            "asset": asset,
            "timeframe": timeframe,
            "segment": segment,
            "n_seeds": int(group["seed"].nunique()),
            "beats_share": float(group["beats"].mean()),
        }
        for key in keys:
            mean, std = _mean_std(group[key].tolist())
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    HOLDOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(HOLDOUT_CSV.with_name("action_mode_holdout_by_seed.csv"), index=False)
    aggregate.to_csv(HOLDOUT_CSV, index=False)
    _copy(HOLDOUT_CSV)
    _copy(HOLDOUT_CSV.with_name("action_mode_holdout_by_seed.csv"))
    return aggregate


def _jobs(stage: str, mode: str, seeds) -> list[tuple]:
    jobs = []
    for seed in seeds:
        for asset, timeframe in sat.TAPES:
            for twin in sat.TWINS:
                jobs.append((stage, mode, asset, timeframe, twin, seed))
    return jobs


def _run_jobs(jobs: list[tuple], parallel: int) -> None:
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, tuple]] = []
    failed = []
    while pending or running:
        while pending and len(running) < parallel:
            job = pending.pop(0)
            stage, mode, asset, timeframe, twin, seed = job
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--stage",
                stage,
                "--mode",
                mode,
                "--asset",
                asset,
                "--tf",
                timeframe,
                "--twin",
                twin,
                "--seed",
                str(seed),
            ]
            print("launch", stage, mode, asset, timeframe, twin, seed, flush=True)
            running.append((subprocess.Popen(cmd), job))
        time.sleep(5)
        still = []
        for proc, job in running:
            code = proc.poll()
            if code is None:
                still.append((proc, job))
            elif code != 0:
                failed.append((job, code))
                print(f"FAILED {job} exit {code}", flush=True)
            else:
                print(f"finished {job[1:]}", flush=True)
        running = still
    if failed:
        raise SystemExit(f"training jobs failed: {failed}")


def frozen_mode() -> str:
    if not CHOICE.exists():
        raise SystemExit("results/action_mode_choice.txt is missing; screen has not been frozen")
    return CHOICE.read_text().strip().splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--stage", choices=("validate", "final"), required=True)
    train_p.add_argument("--mode", choices=MODES, required=True)
    train_p.add_argument("--asset", required=True)
    train_p.add_argument("--tf", required=True)
    train_p.add_argument("--twin", required=True)
    train_p.add_argument("--seed", type=int, required=True)
    screen_p = sub.add_parser("screen")
    screen_p.add_argument("--jobs", type=int, default=2)
    hold_p = sub.add_parser("holdout")
    hold_p.add_argument("--jobs", type=int, default=2)
    sub.add_parser("score-screen")
    args = parser.parse_args()
    if args.cmd == "train":
        train_one(args.stage, args.mode, args.asset, args.tf, args.twin, args.seed)
        return 0
    if args.cmd == "score-screen":
        screen_score()
        return 0
    if args.cmd == "screen":
        jobs = []
        for mode in TRAIN_MODES:
            jobs.extend(_jobs("validate", mode, (42,)))
        _run_jobs(jobs, args.jobs)
        screen_score()
        return 0
    mode = frozen_mode()
    if mode == "none":
        note = ROOT / "results" / "action_mode_holdout_skipped.txt"
        note.write_text("none passed the validation bar; holdout and live were not scored\n")
        _copy(note)
        print(note.read_text(), end="", flush=True)
        return 0
    if mode not in MODES:
        raise SystemExit(f"unexpected frozen mode {mode}")
    _run_jobs(_jobs("final", mode, SEEDS), args.jobs)
    holdout_score(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Screen the mean-variance reward on the native twins, then score one lambda.

Both twins train from configs/final.yaml with only reward_mode and
risk_aversion changed, through run_experiment._train_kwargs. The raw-reward
control is the seed-42 saturation weights, scored by the same code here.

Pass bar, fixed before the screen, on all four validation tapes: both twins
position std > 0.05, both twins share at |a| > 0.95 below 0.80, conditioned
ablation >= 10% of bars. Among passing lambdas the highest minimum ablation
wins; a tie goes to the smaller lambda. Returns and Sharpe are not inputs.
Holdout and live are scored only after paper/results/mean_variance/choice.json
exists, and only for the frozen lambda.
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

from crta.agent import batch_observation, audit_positions, load_twin, save_twin, train_ppo  # noqa: E402
from crta.config import ExperimentConfig  # noqa: E402
from crta.data import FEATURE_COLS  # noqa: E402
from crta.eval import max_drawdown, strategy_returns  # noqa: E402
from crta.regime import REGIMES  # noqa: E402
from crta.select import choose_risk_aversion, meets_criterion  # noqa: E402
from crta.verify import audit_many, audit_many_v2, empirical_banks  # noqa: E402

import action_mode_study as amode  # noqa: E402
import run_experiment as runner  # noqa: E402
import saturation_diagnostic as sat  # noqa: E402
import timeframe_study as study  # noqa: E402

LAMBDAS = (2.0, 5.0, 10.0)
SEEDS = (42, 43, 44, 45, 46)
PERIODS = {"15m": 4 * 24 * 365, "1h": 24 * 365}
BASE_CONFIG = ROOT / "configs" / "final.yaml"
OUT = ROOT / "paper" / "results" / "mean_variance"
WEIGHTS = OUT / "weights"
SCREEN_CSV = OUT / "screen.csv"
CHOICE_JSON = OUT / "choice.json"
SEED_CSV = OUT / "seed_check.csv"
SEED_JSON = OUT / "seed_check.json"
HOLDOUT_CSV = OUT / "holdout_live.csv"
HOLDOUT_SEED_CSV = OUT / "holdout_live_by_seed.csv"


def config(lam: float, asset: str, timeframe: str, stage: str, seed: int) -> ExperimentConfig:
    return ExperimentConfig.load(BASE_CONFIG).with_overrides(
        reward_mode="mean_variance",
        risk_aversion=float(lam),
        asset=asset,
        timeframe=timeframe,
        stage=stage,
        seed=int(seed),
    )


def _lam_tag(lam: float) -> str:
    return f"lam_{lam:g}"


def weight_path(stage: str, lam: float, asset: str, timeframe: str, twin: str, seed: int) -> Path:
    if stage == "validate" and lam == 0.0:
        if seed != 42:
            raise ValueError("the raw control exists for seed 42 only")
        return sat.weight_file(asset, timeframe, twin)
    sub = "validation" if stage == "validate" else "final"
    return WEIGHTS / sub / _lam_tag(lam) / f"seed_{seed}" / f"{asset.lower()}_{timeframe}_{twin}.zip"


def train_one(stage: str, lam: float, asset: str, timeframe: str, twin: str, seed: int) -> Path:
    if lam <= 0.0:
        raise ValueError("training is for mean-variance lambdas only")
    path = weight_path(stage, lam, asset, timeframe, twin, seed)
    meta_path = path.with_suffix(".meta.json")
    cfg = config(lam, asset, timeframe, stage, seed)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("reward_mode") == "mean_variance"
            and float(meta.get("risk_aversion", -1)) == float(lam)
            and int(meta.get("seed", -1)) == seed
            and meta.get("stage") == stage
            and int(meta.get("timesteps", -1)) == int(cfg.timesteps)
        ):
            print(f"keep {path}", flush=True)
            return path
    if stage == "validate":
        train, _val = sat.split_frames(asset, timeframe)
        cut = study.VAL_START
    else:
        train, _segments = amode.final_bundle(asset, timeframe)
        cut = study.HOLDOUT_START
    if train["timestamp"].iloc[-1] >= cut:
        raise RuntimeError("training rows cross the cut")
    kwargs = runner._train_kwargs(cfg, seed, False, timeframe)
    t0 = time.time()
    model = train_ppo(train, twin == "conditioned", **kwargs)
    model.crta_meta.update(
        {
            "config": str(BASE_CONFIG.relative_to(ROOT)),
            "stage": stage,
            "seed": int(seed),
            "asset": asset,
            "timeframe": timeframe,
            "twin": twin,
            "train_before_utc": cut.isoformat(),
            "train_first_bar_utc": train["timestamp"].iloc[0].isoformat(),
            "train_last_bar_utc": train["timestamp"].iloc[-1].isoformat(),
            "n_train": int(len(train)),
            "timesteps": int(cfg.timesteps),
            "window": int(cfg.window),
            "threads": int(cfg.threads),
        }
    )
    save_twin(model, path)
    print(f"trained {stage} lam {lam:g} {asset} {timeframe} {twin} seed {seed} in {time.time() - t0:.0f}s", flush=True)
    return path


def _load(path: Path):
    model, _meta = load_twin(path)
    return model


def _audits(model, seg: pd.DataFrame, executed: np.ndarray, train_probs: np.ndarray) -> dict:
    feats = seg[list(FEATURE_COLS)].to_numpy(dtype=float)
    held = np.concatenate([[0.0], np.asarray(executed, dtype=float)[:-1]])
    probs = seg[list(REGIMES)].to_numpy(dtype=float)
    n = len(seg)

    def predict(batch):
        batch = np.asarray(batch, dtype=float)
        out = np.empty(len(batch))
        step = 8192
        for start in range(0, len(batch), step):
            sl = slice(start, start + step)
            idx = np.arange(start, min(start + step, len(batch))) % n
            obs = batch_observation(feats[idx], batch[sl], held[idx], True)
            out[sl] = audit_positions(model, obs)
        return out

    legacy = audit_many(predict, probs, bound_aware=True)
    v2 = audit_many_v2(predict, probs, empirical_banks(train_probs))
    return {
        "ablation": float(np.mean(legacy["pass_ablation"])),
        "faith_legacy": float(np.mean(legacy["passed"])),
        "faith_rate_testable": float(legacy["faith_rate_testable"]),
        "n_testable": int(legacy["n_testable"]),
        "cf_untestable_share": float(np.mean(legacy["cf_untestable"])),
        "faith_v2": float(np.mean(v2["passed"])),
    }


def _twin_stats(raw: np.ndarray, executed: np.ndarray, ret: np.ndarray, periods: int) -> dict:
    prev = np.concatenate([[0.0], executed[:-1]])
    delta = np.abs(np.diff(executed))
    return {
        "std": float(np.std(executed)),
        "extreme": float(np.mean(np.abs(executed) > 0.95)),
        "raw_gt1": float(np.mean(np.abs(raw) > 1.0)),
        "mean_abs_delta": float(delta.mean()) if len(delta) else float("nan"),
        "turnover": float(np.mean(np.abs(executed - prev))),
        "mean_position": float(np.mean(executed)),
        "log_return": float(np.sum(ret)),
        "return": float(np.exp(np.sum(ret)) - 1.0),
        "sharpe": amode._sharpe(ret, periods),
        "max_drawdown": float(max_drawdown(ret)),
    }


def score_pair(models: dict, seg: pd.DataFrame, timeframe: str, train_probs: np.ndarray, seed: int) -> dict:
    periods = PERIODS[timeframe]
    nxt = seg["next_ret"].to_numpy(dtype=float)
    out = {"n_bars": int(len(seg))}
    rets = {}
    for twin, short in (("conditioned", "c"), ("blind", "b")):
        raw, executed = sat.rollout_raw(models[twin], seg, twin == "conditioned")
        ret = strategy_returns(executed, nxt)
        rets[short] = ret
        for key, value in _twin_stats(raw, executed, ret, periods).items():
            out[f"{key}_{short}"] = value
        if twin == "conditioned":
            out.update(_audits(models[twin], seg, executed, train_probs))
    gap = runner.paired_block_gap(rets["c"], rets["b"], n_boot=2000, seed=int(seed))
    out.update(
        {
            "gap_log": gap["gap_log"],
            "gap_low": gap["low"],
            "gap_high": gap["high"],
            "gap_share_positive": gap["share_positive"],
            "gap_block": gap["block"],
            "beats": bool(gap["cond_beats_blind"]),
        }
    )
    out["passes"] = bool(
        meets_criterion(out["std_c"], out["std_b"], out["extreme_c"], out["extreme_b"], out["ablation"])
    )
    return out


def _print(tag: str, stats: dict) -> None:
    print(
        f"{tag}: std {stats['std_c']:.3f}/{stats['std_b']:.3f} "
        f"ext {stats['extreme_c']:.3f}/{stats['extreme_b']:.3f} "
        f"u>1 {stats['raw_gt1_c']:.3f}/{stats['raw_gt1_b']:.3f} "
        f"ab {stats['ablation']:.3f} faith {stats['faith_legacy']:.3f} "
        f"testable {stats['faith_rate_testable']:.3f} (n {stats['n_testable']}) v2 {stats['faith_v2']:.3f} "
        f"pass {stats['passes']}",
        flush=True,
    )


def score_validation(lam: float, seed: int) -> list[dict]:
    rows = []
    for asset, timeframe in sat.TAPES:
        train, val = sat.split_frames(asset, timeframe)
        models = {twin: _load(weight_path("validate", lam, asset, timeframe, twin, seed)) for twin in sat.TWINS}
        t0 = time.time()
        stats = score_pair(models, val, timeframe, train[list(REGIMES)].to_numpy(dtype=float), seed)
        _print(f"val lam {lam:g} seed {seed} {asset} {timeframe} ({time.time() - t0:.0f}s)", stats)
        rows.append(
            {
                "reward_mode": "raw" if lam == 0.0 else "mean_variance",
                "risk_aversion": float(lam),
                "role": "control" if lam == 0.0 else "candidate",
                "seed": int(seed),
                "asset": asset,
                "timeframe": timeframe,
                "segment": "validation",
                **stats,
            }
        )
    return rows


def screen_score() -> dict:
    rows = []
    for lam in (0.0, *LAMBDAS):
        rows.extend(score_validation(lam, 42))
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SCREEN_CSV, index=False)
    candidates = [
        {k: row[k] for k in ("risk_aversion", "std_c", "std_b", "extreme_c", "extreme_b", "ablation")}
        for row in rows
        if row["role"] == "candidate"
    ]
    choice = choose_risk_aversion(candidates)
    control = [row for row in rows if row["role"] == "control"]
    payload = {
        "risk_aversion": choice["risk_aversion"],
        "reward_mode": "mean_variance" if choice["risk_aversion"] != "none" else None,
        "screen_seed": 42,
        "split": "validation: train before 2024-09-22T16:00Z, score 2024-09-22T16:00Z to 2025-09-22T16:00Z",
        "base_config": str(BASE_CONFIG.relative_to(ROOT)),
        "rule": (
            "all four tapes: both twins position std > 0.05, both twins share |a|>0.95 < 0.80, "
            "conditioned ablation >= 0.10; among passing lambdas highest minimum ablation, tie to smaller lambda; "
            "returns and Sharpe not used"
        ),
        "candidates": list(LAMBDAS),
        "passed": choice["passed"],
        "rejected": choice["rejected"],
        "control_raw_seed42": {
            "n_tapes_passed": int(sum(bool(row["passes"]) for row in control)),
            "min_ablation": float(min(row["ablation"] for row in control)),
            "min_position_std": float(min(min(row["std_c"], row["std_b"]) for row in control)),
        },
        "holdout_live": (
            "not scored: no lambda passed, so no final-split models were trained"
            if choice["risk_aversion"] == "none"
            else "pending: frozen before any holdout or live number"
        ),
    }
    CHOICE_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"frozen lambda: {choice['risk_aversion']}", flush=True)
    return payload


def frozen_lambda() -> float | None:
    if not CHOICE_JSON.exists():
        raise SystemExit("choice.json is missing; the screen has not been frozen")
    value = json.loads(CHOICE_JSON.read_text())["risk_aversion"]
    return None if value == "none" else float(value)


def seed_check(lam: float) -> dict:
    rows = []
    for seed in SEEDS:
        rows.extend(score_validation(lam, seed))
    frame = pd.DataFrame(rows)
    frame.to_csv(SEED_CSV, index=False)
    tapes = {}
    keys = ("std_c", "std_b", "extreme_c", "extreme_b", "ablation")
    for (asset, timeframe), group in frame.groupby(["asset", "timeframe"], sort=False):
        mean = {k: float(group[k].mean()) for k in keys}
        tapes[f"{asset} {timeframe}"] = {
            **{f"{k}_mean": v for k, v in mean.items()},
            "passes_on_seed_mean": bool(meets_criterion(*(mean[k] for k in keys))),
            "seeds_passing": int(group["passes"].sum()),
            "n_seeds": int(group["seed"].nunique()),
        }
    summary = {
        "risk_aversion": lam,
        "seeds": list(SEEDS),
        "all_tapes_pass_on_seed_mean": bool(all(t["passes_on_seed_mean"] for t in tapes.values())),
        "tapes": tapes,
    }
    SEED_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _strategy_row(base: dict, name: str, stats: dict) -> dict:
    return {**base, "strategy": name, **stats}


def holdout_score(lam: float) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for asset, timeframe in sat.TAPES:
            train, segments = amode.final_bundle(asset, timeframe)
            models = {twin: _load(weight_path("final", lam, asset, timeframe, twin, seed)) for twin in sat.TWINS}
            probs = train[list(REGIMES)].to_numpy(dtype=float)
            for name, seg in segments.items():
                stats = score_pair(models, seg, timeframe, probs, seed)
                print(
                    f"final lam {lam:g} seed {seed} {asset} {timeframe} {name}: "
                    f"ret {stats['return_c']:.3f}/{stats['return_b']:.3f} "
                    f"gap {stats['gap_log']:.3f} [{stats['gap_low']:.3f}, {stats['gap_high']:.3f}] "
                    f"ab {stats['ablation']:.3f}",
                    flush=True,
                )
                rows.append(
                    {
                        "risk_aversion": lam,
                        "seed": seed,
                        "asset": asset,
                        "timeframe": timeframe,
                        "segment": name,
                        "first_bar_utc": seg["timestamp"].iloc[0].isoformat(),
                        "last_bar_utc": seg["timestamp"].iloc[-1].isoformat(),
                        **stats,
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(HOLDOUT_SEED_CSV, index=False)
    out = []
    pair_keys = ("ablation", "faith_legacy", "faith_rate_testable", "n_testable", "faith_v2")
    twin_keys = ("return", "sharpe", "max_drawdown", "turnover", "std", "extreme", "raw_gt1", "mean_abs_delta")
    for (asset, timeframe, segment), group in detail.groupby(["asset", "timeframe", "segment"], sort=False):
        base = {
            "risk_aversion": lam,
            "asset": asset,
            "timeframe": timeframe,
            "segment": segment,
            "n_bars": int(group["n_bars"].iloc[0]),
            "n_seeds": int(group["seed"].nunique()),
        }
        for twin, short in (("conditioned", "c"), ("blind", "b")):
            stats = {}
            for key in twin_keys:
                values = pd.to_numeric(group[f"{key}_{short}"], errors="coerce")
                stats[key] = float(values.mean())
                stats[f"{key}_sd"] = float(values.std(ddof=1))
            if twin == "conditioned":
                stats["beats_share"] = float(group["beats"].mean())
                stats["gap_log"] = float(group["gap_log"].mean())
                stats["gap_ci_excludes_zero_share"] = float(((group["gap_low"] > 0) | (group["gap_high"] < 0)).mean())
                stats["gap_ci_above_zero_seeds"] = int((group["gap_low"] > 0).sum())
                for key in pair_keys:
                    stats[key] = float(group[key].mean())
            out.append(_strategy_row(base, twin, stats))
        _train, segments = amode.final_bundle(asset, timeframe)
        seg = segments[segment]
        nxt = seg["next_ret"].to_numpy(dtype=float)
        for name, pos in (("buy_hold", np.ones(len(seg))), ("cash", np.zeros(len(seg)))):
            ret = strategy_returns(pos, nxt)
            stats = _twin_stats(pos, pos, ret, PERIODS[timeframe])
            stats.pop("log_return")
            out.append(_strategy_row(base, name, stats))
    aggregate = pd.DataFrame(out)
    aggregate.to_csv(HOLDOUT_CSV, index=False)
    return aggregate


def _jobs(stage: str, lams, seeds) -> list[tuple]:
    return [
        (stage, lam, asset, timeframe, twin, seed)
        for lam in lams
        for seed in seeds
        for asset, timeframe in sat.TAPES
        for twin in sat.TWINS
    ]


def _run_jobs(jobs: list[tuple], parallel: int) -> None:
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, tuple]] = []
    failed = []
    while pending or running:
        while pending and len(running) < parallel:
            job = pending.pop(0)
            stage, lam, asset, timeframe, twin, seed = job
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--stage", stage,
                "--lam", str(lam),
                "--asset", asset,
                "--tf", timeframe,
                "--twin", twin,
                "--seed", str(seed),
            ]
            print("launch", *job, flush=True)
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
                print(f"finished {job}", flush=True)
        running = still
    if failed:
        raise SystemExit(f"training jobs failed: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--stage", choices=("validate", "final"), required=True)
    train_p.add_argument("--lam", type=float, required=True)
    train_p.add_argument("--asset", required=True)
    train_p.add_argument("--tf", required=True)
    train_p.add_argument("--twin", required=True)
    train_p.add_argument("--seed", type=int, required=True)
    for name in ("screen", "seeds", "final"):
        p = sub.add_parser(name)
        p.add_argument("--jobs", type=int, default=2)
    sub.add_parser("score-screen")
    sub.add_parser("score-seeds")
    sub.add_parser("score-final")
    args = parser.parse_args()
    if args.cmd == "train":
        train_one(args.stage, args.lam, args.asset, args.tf, args.twin, args.seed)
        return 0
    if args.cmd == "screen":
        _run_jobs(_jobs("validate", LAMBDAS, (42,)), args.jobs)
        screen_score()
        return 0
    if args.cmd == "score-screen":
        screen_score()
        return 0
    lam = frozen_lambda()
    if lam is None:
        print("no lambda passed the validation bar; final-split models are not trained and holdout/live are not scored")
        return 0
    if args.cmd == "seeds":
        _run_jobs(_jobs("validate", (lam,), SEEDS), args.jobs)
        seed_check(lam)
    elif args.cmd == "score-seeds":
        seed_check(lam)
    elif args.cmd == "final":
        _run_jobs(_jobs("final", (lam,), SEEDS), args.jobs)
        holdout_score(lam)
    elif args.cmd == "score-final":
        holdout_score(lam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

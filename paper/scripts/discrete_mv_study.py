"""Screen discrete5 with the mean-variance reward on the native twins, then score one lambda.

Both twins train from configs/final.yaml with only action_mode, reward_mode and
risk_aversion changed, through run_experiment._train_kwargs. Two seed-42
controls are rescored here without retraining: discrete5 with the raw reward
(paper/results/action_mode/validation/discrete5) and gaussian_clip with the raw
reward (paper/results/saturation/weights).

Pass bar, fixed before the screen, on all four validation tapes: both twins
position std > 0.05, both twins share at |a| > 0.95 below 0.80, conditioned
ablation >= 10% of bars. For discrete5, |a| is the executed level and the
verifier reads the expected position. Among passing lambdas the highest minimum
ablation wins; a tie goes to the smaller lambda. Returns and Sharpe are not inputs.
Holdout and live are scored only after paper/results/discrete_mv/choice.json
exists, only for the frozen lambda, and only if its 5-seed validation means pass.
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

from crta.agent import DISCRETE_LEVELS, action_mode_of, load_twin, observation, save_twin, train_ppo  # noqa: E402
from crta.config import ExperimentConfig  # noqa: E402
from crta.data import FEATURE_COLS  # noqa: E402
from crta.eval import max_drawdown, strategy_returns  # noqa: E402
from crta.regime import REGIMES  # noqa: E402
from crta.select import choose_risk_aversion, meets_criterion  # noqa: E402

import action_mode_study as amode  # noqa: E402
import mean_variance_study as mv  # noqa: E402
import run_experiment as runner  # noqa: E402
import saturation_diagnostic as sat  # noqa: E402
import timeframe_study as study  # noqa: E402

LAMBDAS = (2.0, 5.0, 10.0)
SEEDS = (42, 43, 44, 45, 46)
PERIODS = mv.PERIODS
ACTION_MODE = "discrete5"
BASE_CONFIG = ROOT / "configs" / "final.yaml"
OUT = ROOT / "paper" / "results" / "discrete_mv"
WEIGHTS = OUT / "weights"
SCREEN_CSV = OUT / "screen.csv"
SCREEN_TAPES_CSV = OUT / "screen_tapes.csv"
CHOICE_JSON = OUT / "choice.json"
SEED_CSV = OUT / "seed_check.csv"
SEED_JSON = OUT / "seed_check.json"
HOLDOUT_CSV = OUT / "holdout_live.csv"
HOLDOUT_SEED_CSV = OUT / "holdout_live_by_seed.csv"
LEVEL_NAMES = ("m1", "m05", "0", "p05", "p1")
CONTROLS = {
    "discrete5_raw": {"action_mode": "discrete5", "reward_mode": "raw", "risk_aversion": 0.0},
    "gaussian_clip_raw": {"action_mode": "gaussian_clip", "reward_mode": "raw", "risk_aversion": 0.0},
}


def cell_name(lam: float) -> str:
    return f"discrete5_mv_lam_{lam:g}"


def config(lam: float, asset: str, timeframe: str, stage: str, seed: int) -> ExperimentConfig:
    return ExperimentConfig.load(BASE_CONFIG).with_overrides(
        action_mode=ACTION_MODE,
        reward_mode="mean_variance",
        risk_aversion=float(lam),
        asset=asset,
        timeframe=timeframe,
        stage=stage,
        seed=int(seed),
    )


def weight_path(stage: str, lam: float, asset: str, timeframe: str, twin: str, seed: int) -> Path:
    sub = "validation" if stage == "validate" else "final"
    return WEIGHTS / sub / f"lam_{lam:g}" / f"seed_{seed}" / f"{asset.lower()}_{timeframe}_{twin}.zip"


def control_path(control: str, asset: str, timeframe: str, twin: str) -> Path:
    mode = CONTROLS[control]["action_mode"]
    return amode.weight_path("validate", mode, asset, timeframe, twin, 42)


def train_one(stage: str, lam: float, asset: str, timeframe: str, twin: str, seed: int) -> Path:
    if lam <= 0.0:
        raise ValueError("training is for mean-variance lambdas only")
    path = weight_path(stage, lam, asset, timeframe, twin, seed)
    meta_path = path.with_suffix(".meta.json")
    cfg = config(lam, asset, timeframe, stage, seed)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("action_mode") == ACTION_MODE
            and meta.get("reward_mode") == "mean_variance"
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


def _discrete_probs(model, obs: np.ndarray) -> np.ndarray:
    import torch as th

    tensor, _vectorized = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        probs = model.policy.get_distribution(tensor).distribution.probs
    return probs.detach().cpu().numpy().astype(float)


def rollout_positions(model, frame: pd.DataFrame, include_regime: bool) -> dict:
    """Sequential rollout. executed is the stored position; expected is what the verifier reads.

    For discrete5, executed is the argmax level and expected is the probability-weighted level.
    For gaussian_clip, executed is the clipped mean, expected equals executed, raw is the mean.
    """
    mode = action_mode_of(model)
    if mode == "gaussian_clip":
        raw, executed = sat.rollout_raw(model, frame, include_regime)
        return {"executed": executed, "expected": executed.copy(), "raw": raw}
    if mode != "discrete5":
        raise ValueError(f"unsupported action_mode {mode}")
    feats = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    probs = frame[list(REGIMES)].to_numpy(dtype=float)
    executed = np.empty(len(frame))
    expected = np.empty(len(frame))
    pos = 0.0
    check_at = set(np.linspace(0, len(frame) - 1, num=min(32, len(frame)), dtype=int).tolist())
    for i in range(len(frame)):
        obs = observation(feats[i], probs[i], pos, include_regime)
        p = _discrete_probs(model, obs.reshape(1, -1))[0]
        level = float(DISCRETE_LEVELS[int(np.argmax(p))])
        if i in check_at:
            pred, _ = model.predict(obs, deterministic=True)
            if float(DISCRETE_LEVELS[int(np.asarray(pred).reshape(-1)[0])]) != level:
                raise RuntimeError(f"argmax level {level} != deterministic predict at bar {i}")
        executed[i] = level
        expected[i] = float(p @ DISCRETE_LEVELS)
        pos = level
    return {"executed": executed, "expected": expected, "raw": np.full(len(frame), np.nan)}


def _twin_stats(path: dict, ret: np.ndarray, periods: int) -> dict:
    executed = path["executed"]
    prev = np.concatenate([[0.0], executed[:-1]])
    delta = np.abs(np.diff(executed))
    raw = path["raw"]
    out = {
        "std": float(np.std(executed)),
        "expected_std": float(np.std(path["expected"])),
        "extreme": float(np.mean(np.abs(executed) > 0.95)),
        "raw_gt1": float(np.mean(np.abs(raw) > 1.0)) if np.isfinite(raw).all() else float("nan"),
        "mean_abs_delta": float(delta.mean()) if len(delta) else float("nan"),
        "turnover": float(np.mean(np.abs(executed - prev))),
        "mean_position": float(np.mean(executed)),
        "log_return": float(np.sum(ret)),
        "return": float(np.exp(np.sum(ret)) - 1.0),
        "sharpe": amode._sharpe(ret, periods),
        "max_drawdown": float(max_drawdown(ret)),
    }
    for name, level in zip(LEVEL_NAMES, DISCRETE_LEVELS):
        out[f"share_{name}"] = float(np.mean(np.abs(executed - level) < 1e-9))
    return out


def score_pair(models: dict, seg: pd.DataFrame, timeframe: str, train_probs: np.ndarray, seed: int) -> dict:
    periods = PERIODS[timeframe]
    nxt = seg["next_ret"].to_numpy(dtype=float)
    out = {"n_bars": int(len(seg))}
    rets = {}
    for twin, short in (("conditioned", "c"), ("blind", "b")):
        path = rollout_positions(models[twin], seg, twin == "conditioned")
        ret = strategy_returns(path["executed"], nxt)
        rets[short] = ret
        for key, value in _twin_stats(path, ret, periods).items():
            out[f"{key}_{short}"] = value
        if twin == "conditioned":
            # audit_positions inside mv._audits reads the expected level for discrete5.
            out.update(mv._audits(models[twin], seg, path["executed"], train_probs))
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


def _print(tag: str, s: dict) -> None:
    levels = lambda short: "/".join(f"{s[f'share_{n}_{short}']:.2f}" for n in LEVEL_NAMES)  # noqa: E731
    print(
        f"{tag}: std {s['std_c']:.3f}/{s['std_b']:.3f} E-std {s['expected_std_c']:.3f}/{s['expected_std_b']:.3f} "
        f"ext {s['extreme_c']:.3f}/{s['extreme_b']:.3f} lv c {levels('c')} b {levels('b')} "
        f"ab {s['ablation']:.3f} faith {s['faith_legacy']:.3f} "
        f"testable {s['faith_rate_testable']:.3f} (n {s['n_testable']}) v2 {s['faith_v2']:.3f} pass {s['passes']}",
        flush=True,
    )


def _cell_meta(cell: str) -> dict:
    if cell in CONTROLS:
        return {"cell": cell, "role": "control", **CONTROLS[cell]}
    lam = float(cell.rsplit("_", 1)[1])
    return {"cell": cell, "role": "candidate", "action_mode": ACTION_MODE, "reward_mode": "mean_variance", "risk_aversion": lam}


def score_validation(cell: str, seed: int) -> list[dict]:
    meta = _cell_meta(cell)
    rows = []
    for asset, timeframe in sat.TAPES:
        train, val = sat.split_frames(asset, timeframe)
        if meta["role"] == "control":
            if seed != 42:
                raise ValueError("controls exist for seed 42 only")
            paths = {twin: control_path(cell, asset, timeframe, twin) for twin in sat.TWINS}
        else:
            paths = {twin: weight_path("validate", meta["risk_aversion"], asset, timeframe, twin, seed) for twin in sat.TWINS}
        models = {}
        for twin, path in paths.items():
            model, wmeta = load_twin(path)
            for key in ("action_mode", "reward_mode"):
                if str(wmeta.get(key, "gaussian_clip" if key == "action_mode" else None)) != str(meta[key]):
                    raise RuntimeError(f"{path}: {key} {wmeta.get(key)} != {meta[key]}")
            if float(wmeta.get("risk_aversion", 0.0)) != float(meta["risk_aversion"]):
                raise RuntimeError(f"{path}: risk_aversion {wmeta.get('risk_aversion')} != {meta['risk_aversion']}")
            if wmeta.get("train_before_utc") != study.VAL_START.isoformat() or int(wmeta.get("seed", -1)) != seed:
                raise RuntimeError(f"{path}: not a seed-{seed} validation-split model")
            models[twin] = model
        t0 = time.time()
        stats = score_pair(models, val, timeframe, train[list(REGIMES)].to_numpy(dtype=float), seed)
        _print(f"val {cell} seed {seed} {asset} {timeframe} ({time.time() - t0:.0f}s)", stats)
        rows.append({**meta, "seed": int(seed), "asset": asset, "timeframe": timeframe, "segment": "validation", **stats})
    return rows


TWIN_KEYS = (
    "std", "expected_std", "extreme", "raw_gt1", "mean_abs_delta", "turnover", "mean_position",
    *(f"share_{n}" for n in LEVEL_NAMES), "return", "sharpe", "max_drawdown",
)
AUDIT_KEYS = ("ablation", "faith_legacy", "faith_rate_testable", "n_testable", "cf_untestable_share", "faith_v2")


def long_rows(rows: list[dict]) -> pd.DataFrame:
    """One row per cell, tape and twin. The blind twin has no regime input, so the audit is not run on it."""
    out = []
    for row in rows:
        base = {k: row[k] for k in ("cell", "role", "action_mode", "reward_mode", "risk_aversion", "seed", "asset", "timeframe", "segment", "n_bars")}
        for twin, short in (("conditioned", "c"), ("blind", "b")):
            item = {**base, "twin": twin, **{k: row[f"{k}_{short}"] for k in TWIN_KEYS}}
            for key in AUDIT_KEYS:
                item[key] = row[key] if twin == "conditioned" else np.nan
            item["tape_passes"] = bool(row["passes"])
            out.append(item)
    return pd.DataFrame(out)


def screen_score() -> dict:
    rows = []
    for cell in (*CONTROLS, *(cell_name(lam) for lam in LAMBDAS)):
        rows.extend(score_validation(cell, 42))
    OUT.mkdir(parents=True, exist_ok=True)
    long_rows(rows).to_csv(SCREEN_CSV, index=False)
    pd.DataFrame(rows).to_csv(SCREEN_TAPES_CSV, index=False)
    candidates = [
        {k: row[k] for k in ("risk_aversion", "std_c", "std_b", "extreme_c", "extreme_b", "ablation")}
        for row in rows
        if row["role"] == "candidate"
    ]
    choice = choose_risk_aversion(candidates)
    controls = {}
    for control in CONTROLS:
        sub = [row for row in rows if row["cell"] == control]
        controls[control] = {
            "weights": str(control_path(control, "BTC", "1h", "conditioned").parent.relative_to(ROOT)),
            "retrained": False,
            "n_tapes_passed": int(sum(bool(row["passes"]) for row in sub)),
            "min_ablation": float(min(row["ablation"] for row in sub)),
            "min_position_std": float(min(min(row["std_c"], row["std_b"]) for row in sub)),
            "max_extreme_share": float(max(max(row["extreme_c"], row["extreme_b"]) for row in sub)),
        }
    frozen = choice["risk_aversion"]
    payload = {
        "cell": "none" if frozen == "none" else cell_name(frozen),
        "action_mode": None if frozen == "none" else ACTION_MODE,
        "reward_mode": None if frozen == "none" else "mean_variance",
        "risk_aversion": frozen,
        "screen_seed": 42,
        "split": "validation: train before 2024-09-22T16:00Z, score 2024-09-22T16:00Z to 2025-09-22T16:00Z",
        "base_config": str(BASE_CONFIG.relative_to(ROOT)),
        "rule": (
            "all four tapes: both twins position std > 0.05, both twins share |a|>0.95 < 0.80 "
            "(discrete5: executed level), conditioned ablation >= 0.10 (discrete5: verifier on the expected position, "
            "bound_aware=True); among passing lambdas highest minimum ablation, tie to smaller lambda; "
            "returns and Sharpe not used"
        ),
        "candidates": [cell_name(lam) for lam in LAMBDAS],
        "passed": choice["passed"],
        "rejected": choice["rejected"],
        "controls_seed42": controls,
        "seed_check": "pending" if frozen != "none" else "not run: no lambda passed",
        "holdout_live": (
            "not scored: no lambda passed, so no final-split models were trained"
            if frozen == "none"
            else "pending: frozen before any holdout or live number; scored only if the 5-seed validation means pass"
        ),
    }
    CHOICE_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"frozen cell: {payload['cell']}", flush=True)
    return payload


def frozen_lambda() -> float | None:
    if not CHOICE_JSON.exists():
        raise SystemExit("choice.json is missing; the screen has not been frozen")
    value = json.loads(CHOICE_JSON.read_text())["risk_aversion"]
    return None if value == "none" else float(value)


def _record(field: str, value) -> None:
    payload = json.loads(CHOICE_JSON.read_text())
    payload[field] = value
    CHOICE_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def seed_check(lam: float) -> dict:
    rows = []
    for seed in SEEDS:
        rows.extend(score_validation(cell_name(lam), seed))
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
        "cell": cell_name(lam),
        "risk_aversion": lam,
        "seeds": list(SEEDS),
        "all_tapes_pass_on_seed_mean": bool(all(t["passes_on_seed_mean"] for t in tapes.values())),
        "tapes": tapes,
    }
    SEED_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    _record(
        "seed_check",
        {
            "all_tapes_pass_on_seed_mean": summary["all_tapes_pass_on_seed_mean"],
            "seeds_passing": {k: v["seeds_passing"] for k, v in tapes.items()},
            "detail": str(SEED_JSON.relative_to(ROOT)),
        },
    )
    if not summary["all_tapes_pass_on_seed_mean"]:
        _record("holdout_live", "not scored: the 5-seed validation means fail the bar, so no final-split models were trained")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def holdout_score(lam: float) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for asset, timeframe in sat.TAPES:
            train, segments = amode.final_bundle(asset, timeframe)
            models = {twin: load_twin(weight_path("final", lam, asset, timeframe, twin, seed))[0] for twin in sat.TWINS}
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
                        "cell": cell_name(lam),
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
    twin_keys = ("return", "sharpe", "max_drawdown", "turnover", "std", "expected_std", "extreme", "mean_abs_delta")
    for (asset, timeframe, segment), group in detail.groupby(["asset", "timeframe", "segment"], sort=False):
        base = {
            "cell": cell_name(lam),
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
                stats["gap_ci_above_zero_seeds"] = int((group["gap_low"] > 0).sum())
                stats["gap_ci_below_zero_seeds"] = int((group["gap_high"] < 0).sum())
                for key in AUDIT_KEYS:
                    stats[key] = float(group[key].mean())
            out.append({**base, "strategy": twin, **stats})
        _train, segments = amode.final_bundle(asset, timeframe)
        seg = segments[segment]
        nxt = seg["next_ret"].to_numpy(dtype=float)
        for name, pos in (("buy_hold", np.ones(len(seg))), ("cash", np.zeros(len(seg)))):
            ret = strategy_returns(pos, nxt)
            stats = _twin_stats({"executed": pos, "expected": pos, "raw": pos}, ret, PERIODS[timeframe])
            out.append({**base, "strategy": name, **{k: stats[k] for k in twin_keys}})
    aggregate = pd.DataFrame(out)
    aggregate.to_csv(HOLDOUT_CSV, index=False)
    _record("holdout_live", f"scored once: {HOLDOUT_CSV.relative_to(ROOT)}")
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
                sys.executable, str(Path(__file__).resolve()), "train",
                "--stage", stage, "--lam", str(lam), "--asset", asset,
                "--tf", timeframe, "--twin", twin, "--seed", str(seed),
            ]
            print("launch", *job, flush=True)
            log = OUT / "logs" / f"{stage}_lam_{lam:g}_seed_{seed}_{asset.lower()}_{timeframe}_{twin}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            running.append((subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT), job))
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
    for name in ("score-screen", "score-seeds", "score-final"):
        sub.add_parser(name)
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
    if args.cmd in ("seeds", "score-seeds"):
        if args.cmd == "seeds":
            _run_jobs(_jobs("validate", (lam,), SEEDS), args.jobs)
        seed_check(lam)
        return 0
    if not SEED_JSON.exists() or not json.loads(SEED_JSON.read_text())["all_tapes_pass_on_seed_mean"]:
        print("the 5-seed validation means do not pass on all four tapes; holdout and live are not scored")
        return 0
    if args.cmd == "final":
        _run_jobs(_jobs("final", (lam,), SEEDS), args.jobs)
    holdout_score(lam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

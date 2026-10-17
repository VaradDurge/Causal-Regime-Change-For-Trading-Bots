"""Redesigned task: daily long/flat decisions with momentum features and a risk-aware reward.

Stage A (labeller guard_dd) trains every cell of the grid
  head in {clip01, tanh01, beta} x risk aversion in {1, 2, 5} x twin in {blind, conditioned,
  monotone} x seeds 42-46
on the four walk-forward folds of the 4h study (same calendar cuts, daily bars). The
conditioned and monotone twins are audited by verifier v3 with each fold's training order
and baseline, its thresholds halved because a long/flat position spans [0, 1], not [-1, 1].

Selection, fixed before any result: on folds 0-1 only, a (head, risk aversion) cell passes
if the blind and conditioned twins both have mean position std above 0.05 and a mean
share of bars below 0.05 or above 0.95 under 0.80, and the conditioned ablation (v3)
averages at least 0.10. Among passing cells the highest median conditioned Sharpe on
folds 0-1 wins. If none passes, the highest score wins and the failure is recorded.

The report gives the chosen cell on folds 2-3 (never used to choose) beside buy-and-hold,
2/4/8-week momentum and the regime rule, with seed medians and IQRs and deflated Sharpe
ratios counting every grid cell and seed as a trial. Stage B reruns the chosen cell with
the jump-model and Wasserstein k-means labels chosen by label_alternatives.py.

Usage:
  .venv/Scripts/python paper/scripts/daily_study.py grid --jobs 20
  .venv/Scripts/python paper/scripts/daily_study.py select
  .venv/Scripts/python paper/scripts/daily_study.py alternatives --jobs 20
  .venv/Scripts/python paper/scripts/daily_study.py report
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.baselines import baseline_positions  # noqa: E402
from crta.daily import (  # noqa: E402
    DAILY_FEATURES,
    PERIODS_DAILY,
    Scaler,
    audit_predictor,
    daily_bounds,
    daily_frame,
    run_positions,
    train_daily,
)
from crta.eval import max_drawdown, strategy_returns, walk_forward_bounds  # noqa: E402
from crta.policies import regime_rank_vector  # noqa: E402
from crta.regime import REGIMES  # noqa: E402
from crta.stats import deflated_sharpe, per_period_sharpe, seed_summary  # noqa: E402
from crta.verify import audit_many_v3, empirical_banks, regime_order, verifier_report  # noqa: E402
from rerun_4h import _jsonable  # noqa: E402

OUT = ROOT / "paper" / "results" / "daily"
FRAME4H = ROOT / "paper" / "results" / "causal_4h_guard_dd" / "frame.parquet"
LABEL_SOURCES = {
    "guard_dd": (ROOT / "paper" / "results" / "causal_4h_guard_dd" / "labels.npz", "probs"),
    "alternatives": ROOT / "paper" / "results" / "causal" / "label_alternatives",
}
HEADS = ("clip01", "tanh01", "beta")
LAMBDAS = (1.0, 2.0, 5.0)
TWINS = ("blind", "conditioned", "monotone")
SEEDS = (42, 43, 44, 45, 46)
SELECT_FOLDS = (0, 1)
EVAL_FOLDS = (2, 3)
MIN_TRAIN_4H = 1080
EXTREME = 0.05
CONSERVATIVE_TRIALS = 200
# The 4h verifier thresholds (0.15, 0.10, 0.05) are 7.5%, 5% and 2.5% of the [-1, 1] range.
# A long/flat position spans [0, 1], so the same shares are half as large.
DELTA_AB = 0.075
DELTA_CF = 0.05
DELTA_SENS = 0.025


def _sharpe(ret) -> float | None:
    r = np.asarray(ret, dtype=float)
    if len(r) < 2 or r.std(ddof=1) < 1e-12:
        return None
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_DAILY))


def _extreme_share(pos) -> float:
    pos = np.asarray(pos, dtype=float)
    return float(np.mean((pos < EXTREME) | (pos > 1.0 - EXTREME)))


def load_probs(labeller: str) -> np.ndarray:
    if labeller == "guard_dd":
        path, key = LABEL_SOURCES["guard_dd"]
        return np.load(path)["probs"]
    alt = LABEL_SOURCES["alternatives"]
    return np.load(alt / "labels.npz")[f"probs_{labeller}"]


def build(labeller: str) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    frame4h = pd.read_parquet(FRAME4H)
    daily = daily_frame(frame4h, load_probs(labeller))
    bounds = daily_bounds(daily["timestamp"], frame4h["timestamp"], walk_forward_bounds(len(frame4h), min_train=MIN_TRAIN_4H))
    return daily, bounds


def _cell_dir(labeller: str, head: str, lam: float, twin: str, seed: int) -> Path:
    return OUT / labeller / f"{head}_l{lam:g}" / twin / f"seed_{seed}"


def run_job(spec: tuple[str, str, float, str, int]) -> str:
    labeller, head, lam, twin, seed = spec
    out = _cell_dir(labeller, head, lam, twin, seed)
    if (out / "eval.json").exists():
        return f"skip {spec}"
    t0 = time.time()
    daily, bounds = build(labeller)
    include = twin != "blind"
    carry = 0.0
    pieces, folds = [], []
    for fold, (start, end) in enumerate(bounds):
        train, test = daily.iloc[:start], daily.iloc[start:end].reset_index(drop=True)
        scaler = Scaler.fit(train[list(DAILY_FEATURES)].to_numpy(dtype=float))
        train_probs = train[list(REGIMES)].to_numpy(dtype=float)
        order = regime_order(train_probs, train["next_ret"].to_numpy(dtype=float))
        model = train_daily(
            train.reset_index(drop=True),
            include,
            head,
            lam,
            seed=seed,
            timesteps=max(8192, 10 * len(train)),
            scaler=scaler,
            monotone=twin == "monotone",
            regime_rank=regime_rank_vector(order) if twin == "monotone" else None,
        )
        pos = run_positions(model, test, include, head, scaler, start=carry)
        before = np.concatenate([[carry], pos[:-1]])
        carry = float(pos[-1])
        block = {"fold": fold, "n_train": int(len(train)), "n_test": int(len(test)), "order": order["ranked"]}
        if include:
            predict = audit_predictor(model, test, before, scaler)
            result = audit_many_v3(
                predict,
                test[list(REGIMES)].to_numpy(dtype=float),
                empirical_banks(train_probs),
                order,
                train_probs.mean(axis=0),
                delta_ab=DELTA_AB,
                delta_cf=DELTA_CF,
                delta_sens=DELTA_SENS,
            )
            block["v3"] = verifier_report(result)
        folds.append(block)
        pieces.append(pd.DataFrame({"timestamp": test["timestamp"], "fold": fold, "position": pos, "next_ret": test["next_ret"]}))
    bars = pd.concat(pieces, ignore_index=True)
    bars["ret"] = strategy_returns(bars["position"].to_numpy(), bars["next_ret"].to_numpy())
    out.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out / "bars.parquet", index=False)
    payload = {"labeller": labeller, "head": head, "risk_aversion": lam, "twin": twin, "seed": seed, "folds": folds, "seconds": time.time() - t0}
    (out / "eval.json").write_text(json.dumps(_jsonable(payload), indent=2))
    return f"done {spec} in {time.time() - t0:.0f}s"


def _run_many(specs: list, jobs: int) -> None:
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for line in pool.map(run_job, specs):
            print(line, flush=True)


def grid(jobs: int) -> None:
    specs = [("guard_dd", h, lam, tw, s) for h, lam, tw, s in itertools.product(HEADS, LAMBDAS, TWINS, SEEDS)]
    _run_many(specs, jobs)


def _cell_stats(labeller: str, head: str, lam: float, twin: str, seed: int, folds: tuple[int, ...]) -> dict:
    out = _cell_dir(labeller, head, lam, twin, seed)
    bars = pd.read_parquet(out / "bars.parquet")
    payload = json.loads((out / "eval.json").read_text())
    sub = bars[bars["fold"].isin(folds)]
    pos, ret = sub["position"].to_numpy(dtype=float), sub["ret"].to_numpy(dtype=float)
    v3 = [f["v3"] for f in payload["folds"] if f["fold"] in folds and "v3" in f]
    n = sum(b["n"] for b in v3)
    return {
        "sharpe": _sharpe(ret),
        "total_return": float(np.exp(ret.sum()) - 1.0),
        "max_drawdown": max_drawdown(ret),
        "mean_position": float(pos.mean()),
        "position_std": float(pos.std()),
        "extreme_share": _extreme_share(pos),
        "ablation_v3": (sum(b["ablation"] * b["n"] for b in v3) / n) if n else None,
        "faithful_v3": (sum(b["all"] * b["n"] for b in v3) / n) if n else None,
        "returns": ret,
    }


def select() -> dict:
    rows = []
    for head, lam in itertools.product(HEADS, LAMBDAS):
        stats = {tw: [_cell_stats("guard_dd", head, lam, tw, s, SELECT_FOLDS) for s in SEEDS] for tw in TWINS}
        mean = lambda tw, key: float(np.mean([x[key] for x in stats[tw] if x[key] is not None]))  # noqa: E731
        spread_ok = all(mean(tw, "position_std") > 0.05 and mean(tw, "extreme_share") < 0.80 for tw in ("blind", "conditioned"))
        ablation_ok = mean("conditioned", "ablation_v3") >= 0.10
        score = seed_summary([x["sharpe"] for x in stats["conditioned"]])["median"]
        rows.append(
            {
                "head": head,
                "risk_aversion": lam,
                "passes": bool(spread_ok and ablation_ok),
                "score": score,
                **{f"{tw}_position_std": mean(tw, "position_std") for tw in TWINS},
                **{f"{tw}_extreme_share": mean(tw, "extreme_share") for tw in TWINS},
                "conditioned_ablation_v3": mean("conditioned", "ablation_v3"),
            }
        )
    table = pd.DataFrame(rows)
    pool = table[table["passes"]] if table["passes"].any() else table
    best = pool.loc[pool["score"].idxmax()]
    decision = {"head": best["head"], "risk_aversion": float(best["risk_aversion"]), "fallback": not bool(table["passes"].any())}
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "selection.csv", index=False)
    (OUT / "selection.json").write_text(json.dumps({"decision": decision, "cells": rows}, indent=2, default=float))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))
    print(json.dumps(decision), flush=True)
    return decision


def _decision() -> dict:
    return json.loads((OUT / "selection.json").read_text())["decision"]


def _alt_labellers() -> list[str]:
    chosen = json.loads((LABEL_SOURCES["alternatives"] / "label_alternatives.json").read_text())["chosen"]
    return [chosen["jump"], chosen["wkmeans"]]


def alternatives(jobs: int) -> None:
    d = _decision()
    specs = [(lab, d["head"], d["risk_aversion"], tw, s) for lab in _alt_labellers() for tw in TWINS for s in SEEDS]
    _run_many(specs, jobs)


def _baselines(daily: pd.DataFrame, bounds) -> dict[str, np.ndarray]:
    mask = np.zeros(len(daily), dtype=bool)
    for fold in EVAL_FOLDS:
        mask[bounds[fold][0] : bounds[fold][1]] = True
    nxt = daily["next_ret"].to_numpy(dtype=float)
    out = {}
    for name, pos in baseline_positions(daily["close"].to_numpy(dtype=float), daily[list(REGIMES)].to_numpy(dtype=float), bars_per_week=7).items():
        start = int(np.argmax(mask))
        ret = strategy_returns(pos[mask], nxt[mask], start_position=float(pos[start - 1]) if start > 0 else 0.0)
        out[name] = ret
    return out


def report() -> None:
    d = _decision()
    labellers = ["guard_dd", *_alt_labellers()]
    trial_returns = {}
    for head, lam, tw, s in itertools.product(HEADS, LAMBDAS, TWINS, SEEDS):
        trial_returns[f"{head}_l{lam:g}_{tw}_{s}"] = _cell_stats("guard_dd", head, lam, tw, s, EVAL_FOLDS)["returns"]
    daily, bounds = build("guard_dd")
    base = _baselines(daily, bounds)
    trial_returns.update({f"baseline_{k}": v for k, v in base.items()})
    trials = [per_period_sharpe(r) for r in trial_returns.values()]
    summary = {"decision": d, "eval_folds": list(EVAL_FOLDS), "n_trials": len(trials), "labellers": {}, "baselines": {}}
    rows = []
    for lab in labellers:
        block = {}
        for tw in TWINS:
            cells = [_cell_stats(lab, d["head"], d["risk_aversion"], tw, s, EVAL_FOLDS) for s in SEEDS]
            dsr = [deflated_sharpe(c["returns"], trials)["dsr"] for c in cells]
            block[tw] = {
                key: seed_summary([c[key] for c in cells])
                for key in ("sharpe", "total_return", "max_drawdown", "position_std", "extreme_share", "ablation_v3", "faithful_v3")
            }
            block[tw]["dsr"] = seed_summary(dsr)
            row = {"labeller": lab, "twin": tw}
            row.update({f"{k}_median": v["median"] for k, v in block[tw].items()})
            row.update({f"{k}_iqr": v["iqr"] for k, v in block[tw].items()})
            rows.append(row)
        cond = [_cell_stats(lab, d["head"], d["risk_aversion"], "conditioned", s, EVAL_FOLDS)["total_return"] for s in SEEDS]
        mono = [_cell_stats(lab, d["head"], d["risk_aversion"], "monotone", s, EVAL_FOLDS)["total_return"] for s in SEEDS]
        blind = [_cell_stats(lab, d["head"], d["risk_aversion"], "blind", s, EVAL_FOLDS)["total_return"] for s in SEEDS]
        block["conditioned_beats_blind"] = float(np.mean([c > b for c, b in zip(cond, blind)]))
        block["monotone_beats_blind"] = float(np.mean([m > b for m, b in zip(mono, blind)]))
        summary["labellers"][lab] = block
    for name, ret in base.items():
        summary["baselines"][name] = {
            "sharpe": _sharpe(ret),
            "total_return": float(np.exp(ret.sum()) - 1.0),
            "max_drawdown": max_drawdown(ret),
            "dsr": deflated_sharpe(ret, trials)["dsr"],
            "dsr_conservative": deflated_sharpe(ret, trials, n_trials=max(CONSERVATIVE_TRIALS, len(trials)))["dsr"],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "report_twins.csv", index=False)
    pd.DataFrame([{"system": k, **v} for k, v in summary["baselines"].items()]).to_csv(OUT / "report_baselines.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(_jsonable(summary), indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["labeller", "twin", "sharpe_median", "sharpe_iqr", "total_return_median", "max_drawdown_median",
                "position_std_median", "extreme_share_median", "ablation_v3_median", "faithful_v3_median", "dsr_median"]
        print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))
        print(pd.DataFrame([{"system": k, **v} for k, v in summary["baselines"].items()]).round(3).to_string(index=False))
    for lab in labellers:
        b = summary["labellers"][lab]
        print(lab, "conditioned beats blind", b["conditioned_beats_blind"], "monotone beats blind", b["monotone_beats_blind"])


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("grid", "alternatives"):
        p = sub.add_parser(name)
        p.add_argument("--jobs", type=int, default=20)
    sub.add_parser("select")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "grid":
        grid(args.jobs)
    elif args.cmd == "alternatives":
        alternatives(args.jobs)
    elif args.cmd == "select":
        select()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rerun the five-seed native twins with filtered regime probabilities.

The frozen native recipe (configs/final.yaml: raw reward, scale 100, entropy 0.01,
2,000-bar windows, 500k steps, tracker v1) is unchanged except that crta.regime now
returns filtered probabilities on burn-in-scaled matching. The published native tables
used the smoothed decoder. Each tape is labelled once, then seeds 42-46 train in
parallel processes and score the holdout year and the live window.

Usage (seed jobs inherit this interpreter; a CPU-only torch keeps each process small):
  .venv-cpu/Scripts/python paper/scripts/native_filtered.py labels
  .venv-cpu/Scripts/python paper/scripts/native_filtered.py run --jobs 8
  .venv-cpu/Scripts/python paper/scripts/native_filtered.py tables
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

OUT = ROOT / "paper" / "results" / "native_filtered"
CONFIG = ROOT / "configs" / "final.yaml"
TAPES = (("BTC", "1h"), ("BTC", "15m"), ("ETH", "1h"), ("ETH", "15m"))
SEEDS = (42, 43, 44, 45, 46)
# A 16 GB laptop holds about eight seed processes with the 15-minute tapes in memory.
DEFAULT_JOBS = 8


def _label_tape(tape: tuple[str, str]) -> str:
    import timeframe_study as study

    asset, tf = tape
    t0 = time.time()
    study.labelled(asset, tf, study.candles(asset, tf))
    return f"{asset} {tf} labelled in {time.time() - t0:.0f}s"


def labels() -> None:
    with ProcessPoolExecutor(max_workers=len(TAPES)) as pool:
        for line in pool.map(_label_tape, TAPES):
            print(line, flush=True)


def _tape_dir(asset: str, tf: str) -> Path:
    return OUT / f"{asset.lower()}_{tf}_final"


def run(jobs: int, tapes: tuple[tuple[str, str], ...] = TAPES) -> None:
    pending = [(asset, tf, seed) for asset, tf in tapes for seed in SEEDS]
    running: list[tuple[subprocess.Popen, str]] = []
    failed = []
    while pending or running:
        while pending and len(running) < jobs:
            asset, tf, seed = pending.pop(0)
            cmd = [
                sys.executable,
                str(HERE / "run_experiment.py"),
                "--config", str(CONFIG),
                "--asset", asset,
                "--tf", tf,
                "--stage", "final",
                "--seeds", str(seed),
                "--set", "threads=1",
                "--out", str(_tape_dir(asset, tf)),
            ]
            print("launch", asset, tf, seed, flush=True)
            running.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL), f"{asset}:{tf}:{seed}"))
        time.sleep(10)
        still = []
        for proc, name in running:
            code = proc.poll()
            if code is None:
                still.append((proc, name))
            elif code != 0:
                failed.append((name, code))
                print(f"FAILED {name} exit {code}", flush=True)
            else:
                print(f"finished {name}", flush=True)
        running = still
    if failed:
        raise SystemExit(f"native jobs failed: {failed}")
    if all(_tape_dir(a, t).joinpath(f"seed_{s}", "eval.json").exists() for a, t in TAPES for s in SEEDS):
        tables()


def tables() -> None:
    from run_experiment import aggregate

    rows = []
    for asset, tf in TAPES:
        summary = aggregate(_tape_dir(asset, tf))
        for window, block in summary["segments"].items():
            rows.append(
                {
                    "tape": f"{asset} {tf}",
                    "window": window,
                    "cond_return": block["conditioned"]["total_return"]["mean"],
                    "cond_return_std": block["conditioned"]["total_return"]["std"],
                    "cond_sharpe": block["conditioned"]["sharpe"]["mean"],
                    "cond_extreme_share": block["conditioned"]["extreme_share"]["mean"],
                    "blind_return": block["blind"]["total_return"]["mean"],
                    "blind_sharpe": block["blind"]["sharpe"]["mean"],
                    "blind_extreme_share": block["blind"]["extreme_share"]["mean"],
                    "buy_hold_return": block["buy_hold"]["total_return"]["mean"],
                    "beats_share": block["cond_beats_blind_share"],
                    "ablation": block["ablation_pass_rate"]["mean"],
                    "faithfulness": block["faithfulness_rate"]["mean"],
                }
            )
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "native_filtered.csv", index=False)
    (OUT / "native_filtered.json").write_text(json.dumps(rows, indent=2, default=float))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.round(3).to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels")
    run_p = sub.add_parser("run")
    run_p.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    run_p.add_argument("--tapes", default=None, help="comma list such as BTC:1h,ETH:1h (default: all four)")
    sub.add_parser("tables")
    args = parser.parse_args()
    if args.cmd == "labels":
        labels()
    elif args.cmd == "run":
        chosen = TAPES if args.tapes is None else tuple(tuple(item.split(":")) for item in args.tapes.split(","))
        run(args.jobs, chosen)
    else:
        tables()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

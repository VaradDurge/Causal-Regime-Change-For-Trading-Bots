"""Four-hour walk-forward study with causal regime inputs. This replaces the published run.

Labels: tracker v2 with filtered probabilities, W2 on burn-in-scaled features, fold 0's
1,080 training bars as the burn-in, and a warmed filter (crta.regime defaults). Twins,
budget, reward and verifiers are those of rerun_4h.py, on the same frozen candles.
Baselines: buy-and-hold, long/flat momentum over 2, 4 and 8 weeks, and the regime rule
1 - p(bear) - p(crisis) on the same filtered probabilities. Results are also split by
hindsight 20% bull/bear phases so one falling year does not decide the comparison.

Usage:
  .venv/Scripts/python paper/scripts/causal_4h.py all --jobs 5
  .venv/Scripts/python paper/scripts/causal_4h.py seed --seed 42
  .venv/Scripts/python paper/scripts/causal_4h.py tables
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

from crta.baselines import baseline_positions  # noqa: E402
from crta.data import HMM_COLS  # noqa: E402
from crta.dating import hindsight_phases, label_quality  # noqa: E402
from crta.eval import PERIODS, sharpe, strategy_returns  # noqa: E402
from crta.regime import LABELLER_VERSION, REGIMES, track_v2  # noqa: E402
from crta.report import label_statistics  # noqa: E402
from rerun_4h import _jsonable, _mean_std, _system_metrics, load_frozen_frame, walk_forward_seed  # noqa: E402

OUT = ROOT / "paper" / "results" / "causal_4h"
SEEDS = (42, 43, 44, 45, 46)
MIN_TRAIN = 1080
LABELLING = "causal_v2"
SYSTEMS = ("conditioned", "blind")


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "labels.npz").exists() and (OUT / "frame.parquet").exists():
        print("labels already prepared", flush=True)
        return
    t0 = time.time()
    frame = load_frozen_frame()
    diag: dict = {}
    labels, probs = track_v2(frame[list(HMM_COLS)].to_numpy(dtype=float), burn=MIN_TRAIN, seed=42, diagnostics=diag)
    phases = hindsight_phases(frame["close"].to_numpy(dtype=float))
    oos = slice(MIN_TRAIN, len(frame))
    stats = {
        "labeller": LABELLER_VERSION,
        "settings": {"burn": MIN_TRAIN, "stride": 42, "dwell": 30, "matching": "nearest", "death_after": 20},
        "scored_from_bar": MIN_TRAIN,
        "quality": label_quality(list(labels)[oos], phases[oos]),
        "statistics": label_statistics(list(labels)[oos], frame.iloc[oos], PERIODS, diag),
        "rebirths": [e for e in diag.get("events", []) if e.get("kind") == "death_birth"],
    }
    frame.to_parquet(OUT / "frame.parquet", index=False)
    np.savez_compressed(OUT / "labels.npz", labels=np.array(labels), probs=probs, phases=phases.astype(str))
    (OUT / "label_quality.json").write_text(json.dumps(_jsonable(stats), indent=2))
    write_baselines(frame, probs, phases)
    print(f"prepared labels and baselines in {time.time() - t0:.0f}s", flush=True)


def _load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(OUT / "frame.parquet")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    packed = np.load(OUT / "labels.npz", allow_pickle=False)
    return frame, packed["labels"], packed["probs"], packed["phases"]


def _phase_sharpe(ret: np.ndarray, phases: np.ndarray) -> dict:
    return {phase: (sharpe(ret[phases == phase]) if (phases == phase).sum() > 1 else None) for phase in ("bull", "bear")}


def write_baselines(frame: pd.DataFrame, probs: np.ndarray, phases: np.ndarray) -> dict:
    oos = slice(MIN_TRAIN, len(frame))
    nxt = frame["next_ret"].to_numpy(dtype=float)[oos]
    out = {}
    for name, position in baseline_positions(frame["close"].to_numpy(dtype=float), probs).items():
        act = position[oos]
        ret = strategy_returns(act, nxt)
        out[name] = {**_system_metrics(ret, act), "phase_sharpe": _phase_sharpe(ret, phases[oos])}
    (OUT / "baselines.json").write_text(json.dumps(_jsonable(out), indent=2))
    return out


def run_seed(seed: int) -> dict:
    prepare()
    frame, labels, probs, _phases = _load()
    return walk_forward_seed(frame, labels, probs, LABELLING, seed, OUT / "seeds" / f"seed_{seed}")


def run_all(jobs: int) -> None:
    prepare()
    pending = list(SEEDS)
    running: list[tuple[subprocess.Popen, int]] = []
    failed = []
    while pending or running:
        while pending and len(running) < jobs:
            seed = pending.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "seed", "--seed", str(seed)]
            print("launch seed", seed, flush=True)
            running.append((subprocess.Popen(cmd), seed))
        time.sleep(5)
        still = []
        for proc, seed in running:
            code = proc.poll()
            if code is None:
                still.append((proc, seed))
            elif code != 0:
                failed.append((seed, code))
                print(f"FAILED seed {seed} exit {code}", flush=True)
            else:
                print(f"finished seed {seed}", flush=True)
        running = still
    if failed:
        raise SystemExit(f"seed jobs failed: {failed}")
    write_tables()


def _seed_rows(phases_oos: np.ndarray) -> list[dict]:
    rows = []
    for seed in SEEDS:
        seed_dir = OUT / "seeds" / f"seed_{seed}"
        payload = json.loads((seed_dir / "eval.json").read_text())
        bars = pd.read_parquet(seed_dir / "bars.parquet")
        if len(bars) != len(phases_oos):
            raise RuntimeError(f"seed {seed}: {len(bars)} bars for {len(phases_oos)} phases")
        cond = payload["systems"]["conditioned"]
        blind = payload["systems"]["blind"]
        row = {"seed": seed, "beats": float(cond["total_return"] > blind["total_return"])}
        for system in SYSTEMS:
            block = payload["systems"][system]
            row[f"{system}_sharpe"] = block["sharpe"]["point"]
            row[f"{system}_return"] = block["total_return"]
            row[f"{system}_max_drawdown"] = block["max_drawdown"]
            row[f"{system}_position_std"] = block["position_std"]
            row[f"{system}_extreme_share"] = block["extreme_share"]
            phase = _phase_sharpe(bars[f"ret_{system}"].to_numpy(dtype=float), phases_oos)
            row[f"{system}_bull_sharpe"] = phase["bull"]
            row[f"{system}_bear_sharpe"] = phase["bear"]
        for verifier in ("legacy", "v2"):
            row[f"ablation_{verifier}"] = payload["verifiers"][verifier]["ablation"]
            row[f"faithful_{verifier}"] = payload["verifiers"][verifier]["all"]
        rows.append(row)
    return rows


def write_tables() -> None:
    frame, _labels, _probs, phases = _load()
    phases_oos = phases[MIN_TRAIN:]
    by_seed = pd.DataFrame(_seed_rows(phases_oos))
    summary = {col: _mean_std(by_seed[col].tolist()) for col in by_seed.columns if col != "seed"}
    baselines = json.loads((OUT / "baselines.json").read_text())
    quality = json.loads((OUT / "label_quality.json").read_text())
    by_seed.to_csv(OUT / "metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [{"metric": k, "mean": v["mean"], "std": v["std"], "n": v["n"]} for k, v in summary.items()]
    ).to_csv(OUT / "metrics_mean_std.csv", index=False)
    pd.DataFrame(
        [
            {
                "system": name,
                "sharpe": block["sharpe"]["point"],
                "sharpe_low": block["sharpe"]["low"],
                "sharpe_high": block["sharpe"]["high"],
                "total_return": block["total_return"],
                "max_drawdown": block["max_drawdown"],
                "bull_sharpe": block["phase_sharpe"]["bull"],
                "bear_sharpe": block["phase_sharpe"]["bear"],
            }
            for name, block in baselines.items()
        ]
    ).to_csv(OUT / "baselines.csv", index=False)
    payload = {
        "labeller": LABELLER_VERSION,
        "n_oos_bars": int(len(phases_oos)),
        "seeds": list(SEEDS),
        "twins_mean_std": summary,
        "baselines": baselines,
        "label_quality": quality["quality"],
        "label_shares": {g: quality["quality"]["label_share"][g] for g in REGIMES},
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(json.dumps(_jsonable({k: payload[k] for k in ("label_shares", "twins_mean_std")}), indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    seed_p = sub.add_parser("seed")
    seed_p.add_argument("--seed", type=int, required=True)
    all_p = sub.add_parser("all")
    all_p.add_argument("--jobs", type=int, default=len(SEEDS))
    sub.add_parser("tables")
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "seed":
        run_seed(args.seed)
    elif args.cmd == "all":
        run_all(args.jobs)
    else:
        write_tables()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

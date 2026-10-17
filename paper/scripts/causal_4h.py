"""Four-hour walk-forward study with causal regime inputs. This replaces the published run.

Labellers (all tracker v2 with filtered probabilities, burn-in-scaled W2, fold 0's
1,080 training bars as the burn-in and a warmed filter):

  guard_dd  semantic guard plus the 90-day drawdown input, chosen on fold 1's training
            span by label_selection.py. The default and the canonical study.
  tier1     the Tier 1 labeller, kept so its committed results can be re-tabled.

Twins, budget and reward are those of rerun_4h.py, on the same frozen candles. Each
conditioned twin is audited by the legacy verifier, verifier v2 and verifier v3, whose
ablation baseline and risk order come from that fold's training rows. Verifier v3 is
first checked on reference policies built from the same order.
Baselines: buy-and-hold, long/flat momentum over 2, 4 and 8 weeks, and the regime rule
1 - p(bear) - p(crisis). Results are split by hindsight 20% bull/bear phases, seeds are
summarised by median and interquartile range, and every system gets a deflated Sharpe
ratio that counts all systems and seeds of this study as trials.

Usage:
  .venv/Scripts/python paper/scripts/causal_4h.py all --jobs 5
  .venv/Scripts/python paper/scripts/causal_4h.py seed --seed 42
  .venv/Scripts/python paper/scripts/causal_4h.py tables
  (add --labeller tier1 for the Tier 1 labeller)
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
from crta.data import HMM_COLS, HMM_COLS_DD, drawdown  # noqa: E402
from crta.dating import hindsight_phases, label_quality  # noqa: E402
from crta.eval import PERIODS, sharpe, strategy_returns, walk_forward_bounds  # noqa: E402
from crta.regime import LABELLER_VERSION, REGIMES, TIER1_V2, track_v2  # noqa: E402
from crta.report import label_statistics  # noqa: E402
from crta.stats import deflated_sharpe, per_period_sharpe, seed_summary  # noqa: E402
from crta.verify import audit_many_v3, empirical_banks, reference_weights, regime_order  # noqa: E402
from rerun_4h import _jsonable, _system_metrics, load_frozen_frame, walk_forward_seed  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
MIN_TRAIN = 1080
SYSTEMS = ("conditioned", "blind")
VERIFIERS = ("legacy", "v2", "v3")
# A conservative trial count for the deflated Sharpe ratio, covering screens run before this study.
CONSERVATIVE_TRIALS = 100
LABELLERS = {
    "guard_dd": {"cols": HMM_COLS_DD, "kwargs": {"burn": MIN_TRAIN, "semantic_guard": True}, "dir": "causal_4h_guard_dd"},
    "tier1": {"cols": HMM_COLS, "kwargs": dict(TIER1_V2), "dir": "causal_4h"},
}
DEFAULT_LABELLER = "guard_dd"


def out_dir(labeller: str) -> Path:
    return ROOT / "paper" / "results" / LABELLERS[labeller]["dir"]


def prepare(labeller: str) -> None:
    out = out_dir(labeller)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "labels.npz").exists() and (out / "frame.parquet").exists():
        print("labels already prepared", flush=True)
        return
    t0 = time.time()
    frame = load_frozen_frame()
    frame["drawdown"] = drawdown(frame["close"])
    spec = LABELLERS[labeller]
    diag: dict = {}
    labels, probs = track_v2(frame[list(spec["cols"])].to_numpy(dtype=float), seed=42, diagnostics=diag, **spec["kwargs"])
    phases = hindsight_phases(frame["close"].to_numpy(dtype=float))
    oos = slice(MIN_TRAIN, len(frame))
    stats = {
        "labeller": labeller,
        "labeller_version": LABELLER_VERSION,
        "settings": {"cols": list(spec["cols"]), **spec["kwargs"], "stride": 42, "dwell": 30, "death_after": 20},
        "scored_from_bar": MIN_TRAIN,
        "quality": label_quality(list(labels)[oos], phases[oos]),
        "statistics": label_statistics(list(labels)[oos], frame.iloc[oos], PERIODS, diag),
        "rebirths": [e for e in diag.get("events", []) if e.get("kind") == "death_birth"],
    }
    frame.to_parquet(out / "frame.parquet", index=False)
    np.savez_compressed(out / "labels.npz", labels=np.array(labels), probs=probs, phases=phases.astype(str))
    (out / "label_quality.json").write_text(json.dumps(_jsonable(stats), indent=2))
    write_baselines(out, frame, probs, phases)
    (out / "reference_policies.json").write_text(json.dumps(_jsonable(reference_check(frame, probs)), indent=2))
    print(f"prepared labels, baselines and the v3 reference check in {time.time() - t0:.0f}s", flush=True)


def _load(out: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(out / "frame.parquet")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    packed = np.load(out / "labels.npz", allow_pickle=False)
    return frame, packed["labels"], packed["probs"], packed["phases"]


def _phase_sharpe(ret: np.ndarray, phases: np.ndarray) -> dict:
    return {phase: (sharpe(ret[phases == phase]) if (phases == phase).sum() > 1 else None) for phase in ("bull", "bear")}


def write_baselines(out: Path, frame: pd.DataFrame, probs: np.ndarray, phases: np.ndarray) -> dict:
    oos = slice(MIN_TRAIN, len(frame))
    nxt = frame["next_ret"].to_numpy(dtype=float)[oos]
    result = {}
    returns = {}
    for name, position in baseline_positions(frame["close"].to_numpy(dtype=float), probs).items():
        act = position[oos]
        ret = strategy_returns(act, nxt)
        returns[name] = ret
        result[name] = {**_system_metrics(ret, act), "phase_sharpe": _phase_sharpe(ret, phases[oos])}
    pd.DataFrame(returns).to_parquet(out / "baseline_returns.parquet", index=False)
    (out / "baselines.json").write_text(json.dumps(_jsonable(result), indent=2))
    return result


def reference_check(frame: pd.DataFrame, probs: np.ndarray) -> dict:
    """Verifier v3 on policies that follow, ignore and invert each fold's training order."""
    nxt = frame["next_ret"].to_numpy(dtype=float)
    totals = {name: {"passed": 0, "testable": 0} for name in ("faithful", "ignore", "inverted")}
    folds = []
    n = 0
    for train_end, test_end in walk_forward_bounds(len(frame), min_train=MIN_TRAIN):
        train = probs[:train_end]
        order = regime_order(train, nxt[:train_end])
        w = reference_weights(order)
        policies = {
            "faithful": lambda batch, w=w: np.clip(np.asarray(batch) @ w, -1.0, 1.0),
            "ignore": lambda batch: np.full(len(batch), 0.5),
            "inverted": lambda batch, w=w: np.clip(np.asarray(batch) @ -w, -1.0, 1.0),
        }
        test = probs[train_end:test_end]
        n += len(test)
        fold = {"train_end": train_end, "order": order["ranked"], "mv_position": order["mv_position"]}
        for name, policy in policies.items():
            res = audit_many_v3(policy, test, empirical_banks(train), order, train.mean(axis=0))
            totals[name]["passed"] += int(res["passed"].sum())
            totals[name]["testable"] += int(res["testable"].sum())
            fold[name] = float(res["passed"].mean())
        folds.append(fold)
    return {
        "n": n,
        "pass_rate": {name: t["passed"] / n for name, t in totals.items()},
        "testable_share": {name: t["testable"] / n for name, t in totals.items()},
        "folds": folds,
    }


def run_seed(labeller: str, seed: int) -> dict:
    prepare(labeller)
    out = out_dir(labeller)
    frame, labels, probs, _phases = _load(out)
    return walk_forward_seed(frame, labels, probs, f"causal_{labeller}", seed, out / "seeds" / f"seed_{seed}")


def run_all(labeller: str, jobs: int) -> None:
    prepare(labeller)
    pending = list(SEEDS)
    running: list[tuple[subprocess.Popen, int]] = []
    failed = []
    while pending or running:
        while pending and len(running) < jobs:
            seed = pending.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "seed", "--seed", str(seed), "--labeller", labeller]
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
    write_tables(labeller)


def _seed_rows(out: Path, phases_oos: np.ndarray) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    returns = {}
    for seed in SEEDS:
        seed_dir = out / "seeds" / f"seed_{seed}"
        payload = json.loads((seed_dir / "eval.json").read_text())
        bars = pd.read_parquet(seed_dir / "bars.parquet")
        if len(bars) != len(phases_oos):
            raise RuntimeError(f"seed {seed}: {len(bars)} bars for {len(phases_oos)} phases")
        cond = payload["systems"]["conditioned"]
        blind = payload["systems"]["blind"]
        row = {"seed": seed, "beats": float(cond["total_return"] > blind["total_return"])}
        for system in SYSTEMS:
            block = payload["systems"][system]
            ret = bars[f"ret_{system}"].to_numpy(dtype=float)
            returns[f"{system}_seed{seed}"] = ret
            row[f"{system}_sharpe"] = block["sharpe"]["point"]
            row[f"{system}_return"] = block["total_return"]
            row[f"{system}_max_drawdown"] = block["max_drawdown"]
            row[f"{system}_position_std"] = block["position_std"]
            row[f"{system}_extreme_share"] = block["extreme_share"]
            phase = _phase_sharpe(ret, phases_oos)
            row[f"{system}_bull_sharpe"] = phase["bull"]
            row[f"{system}_bear_sharpe"] = phase["bear"]
        for verifier in VERIFIERS:
            block = payload["verifiers"].get(verifier)
            row[f"ablation_{verifier}"] = None if block is None else block["ablation"]
            row[f"faithful_{verifier}"] = None if block is None else block["all"]
        rows.append(row)
    return rows, returns


def _deflate(returns: dict[str, np.ndarray]) -> dict:
    trial_sharpes = [per_period_sharpe(r) for r in returns.values()]
    out = {}
    for name, ret in returns.items():
        study = deflated_sharpe(ret, trial_sharpes)
        conservative = deflated_sharpe(ret, trial_sharpes, n_trials=max(CONSERVATIVE_TRIALS, len(trial_sharpes)))
        out[name] = {"dsr": study["dsr"], "dsr_conservative": conservative["dsr"], "sr0": study["sr0"], "n_trials": study["n_trials"]}
    return out


def write_tables(labeller: str) -> None:
    out = out_dir(labeller)
    frame, _labels, probs, phases = _load(out)
    phases_oos = phases[MIN_TRAIN:]
    if not (out / "baseline_returns.parquet").exists():
        # Runs from before the deflated Sharpe ratio kept no baseline returns.
        write_baselines(out, frame, probs, phases)
    seed_rows, ppo_returns = _seed_rows(out, phases_oos)
    by_seed = pd.DataFrame(seed_rows)
    baselines = json.loads((out / "baselines.json").read_text())
    base_returns = pd.read_parquet(out / "baseline_returns.parquet")
    deflated = _deflate({**ppo_returns, **{k: base_returns[k].to_numpy(dtype=float) for k in base_returns.columns}})
    for system in SYSTEMS:
        by_seed[f"{system}_dsr"] = [deflated[f"{system}_seed{seed}"]["dsr"] for seed in SEEDS]
    summary = {col: seed_summary(by_seed[col].tolist()) for col in by_seed.columns if col != "seed"}
    by_seed.to_csv(out / "metrics_by_seed.csv", index=False)
    pd.DataFrame([{"metric": k, **v} for k, v in summary.items()]).to_csv(out / "metrics_seed_summary.csv", index=False)
    base_rows = []
    for name, block in baselines.items():
        base_rows.append(
            {
                "system": name,
                "sharpe": block["sharpe"]["point"],
                "sharpe_low": block["sharpe"]["low"],
                "sharpe_high": block["sharpe"]["high"],
                "total_return": block["total_return"],
                "max_drawdown": block["max_drawdown"],
                "bull_sharpe": block["phase_sharpe"]["bull"],
                "bear_sharpe": block["phase_sharpe"]["bear"],
                "dsr": deflated[name]["dsr"],
                "dsr_conservative": deflated[name]["dsr_conservative"],
            }
        )
    pd.DataFrame(base_rows).to_csv(out / "baselines.csv", index=False)
    quality = json.loads((out / "label_quality.json").read_text())
    reference = out / "reference_policies.json"
    payload = {
        "labeller": labeller,
        "labeller_version": quality.get("labeller_version"),
        "n_oos_bars": int(len(phases_oos)),
        "seeds": list(SEEDS),
        "twins_seed_summary": summary,
        "baselines": base_rows,
        "deflated_sharpe": deflated,
        "verifier_v3_reference": json.loads(reference.read_text()) if reference.exists() else None,
        "label_quality": quality["quality"],
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2))
    keys = [k for k in summary if k.endswith(("_sharpe", "_dsr", "beats")) or k.startswith(("faithful_", "ablation_"))]
    for key in keys:
        s = summary[key]
        if s["n"]:
            print(f"{key:28s} median {s['median']:+.3f}  IQR [{s['q25']:+.3f}, {s['q75']:+.3f}]  mean {s['mean']:+.3f}", flush=True)
    print(pd.DataFrame(base_rows).round(3).to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "seed", "all", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--labeller", choices=sorted(LABELLERS), default=DEFAULT_LABELLER)
        if name == "seed":
            p.add_argument("--seed", type=int, required=True)
        if name == "all":
            p.add_argument("--jobs", type=int, default=len(SEEDS))
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.labeller)
    elif args.cmd == "seed":
        run_seed(args.labeller, args.seed)
    elif args.cmd == "all":
        run_all(args.labeller, args.jobs)
    else:
        write_tables(args.labeller)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

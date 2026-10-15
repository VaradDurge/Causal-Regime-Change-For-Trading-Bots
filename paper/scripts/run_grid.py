"""Validation-only one-factor grid, then a few combinations.

Selection is fixed in crta.select and does not read holdout or live files.
Returns are not an input. Resume is safe: a cell with seed_42/eval.json is skipped.

  python paper/scripts/run_grid.py --workers 2
  python paper/scripts/run_grid.py --select
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crta.select import criterion_deficit, meets_criterion  # noqa: E402

GRID = ROOT / "paper" / "results" / "v2" / "grid"
RUNNER = Path(__file__).resolve().parent / "run_experiment.py"
TAPES = (("BTC", "1h"), ("BTC", "15m"), ("ETH", "1h"), ("ETH", "15m"))
# One factor away from configs/default.yaml. The control cell is "baseline".
FACTORS = (
    ("baseline", None, None),
    ("reward_excess_bh", "reward_mode", "excess_bh"),
    ("reward_diff_sharpe", "reward_mode", "diff_sharpe"),
    ("sample_regime_balanced", "window_sampling", "regime_balanced"),
    ("sample_return_balanced", "window_sampling", "return_balanced"),
    ("ent_002", "ent_coef", "0.02"),
    ("perturb_005", "regime_perturb_prob", "0.05"),
    ("perturb_01", "regime_perturb_prob", "0.1"),
    ("tracker_v2", "tracker", "v2"),
)


def _cell(name: str, asset: str, timeframe: str) -> Path:
    return GRID / name / f"{asset.lower()}_{timeframe}"


def _done(path: Path) -> bool:
    return (path / "seed_42" / "eval.json").exists()


def _cmd(name: str, asset: str, timeframe: str, key, value) -> list[str]:
    out = _cell(name, asset, timeframe)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(ROOT / "configs" / "default.yaml"),
        "--asset",
        asset,
        "--tf",
        timeframe,
        "--stage",
        "validate",
        "--seeds",
        "42",
        "--out",
        str(out),
    ]
    if key is not None:
        cmd.extend(["--set", f"{key}={value}"])
    return cmd


def _jobs(names) -> list[tuple[str, list[str]]]:
    jobs = []
    for name, key, value in FACTORS:
        if names and name not in names:
            continue
        for asset, timeframe in TAPES:
            path = _cell(name, asset, timeframe)
            if _done(path):
                continue
            jobs.append((f"{name} {asset} {timeframe}", _cmd(name, asset, timeframe, key, value)))
    return jobs


def _run(label: str, cmd: list[str]) -> tuple[str, int]:
    log = GRID / "logs"
    log.mkdir(parents=True, exist_ok=True)
    path = log / (label.replace(" ", "_") + ".log")
    with path.open("w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return label, proc.returncode


def launch(workers: int, names) -> int:
    jobs = _jobs(names)
    print(f"{len(jobs)} cells to run", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, label, cmd) for label, cmd in jobs]
        for future in as_completed(futures):
            label, code = future.result()
            print(f"{label}: exit {code}", flush=True)
            if code != 0:
                failed.append(label)
    if failed:
        print("failed:", ", ".join(failed), flush=True)
        return 1
    return 0


def _metrics(path: Path) -> dict | None:
    eval_path = path / "seed_42" / "eval.json"
    if not eval_path.exists():
        return None
    payload = json.loads(eval_path.read_text())
    seg = payload["segments"].get("validation")
    if not seg or seg.get("skipped"):
        return None
    cond = seg["conditioned"]
    blind = seg["blind"]
    row = {
        "std_c": cond["position_std"],
        "std_b": blind["position_std"],
        "extreme_c": cond["extreme_share"],
        "extreme_b": blind["extreme_share"],
        "ablation": seg["ablation_pass_rate"],
        "faithfulness": seg["faithfulness_rate"],
        "ret_c": cond["total_return"],
        "ret_b": blind["total_return"],
        "sharpe_c": cond["sharpe"],
        "sharpe_b": blind["sharpe"],
    }
    row["meets"] = meets_criterion(row["std_c"], row["std_b"], row["extreme_c"], row["extreme_b"], row["ablation"])
    row["deficit"] = criterion_deficit(
        row["std_c"], row["std_b"], row["extreme_c"], row["extreme_b"], row["ablation"]
    )
    return row


def _cell_table() -> list[dict]:
    rows = []
    for name, key, value in FACTORS:
        deficits = []
        meets = []
        for asset, timeframe in TAPES:
            metrics = _metrics(_cell(name, asset, timeframe))
            if metrics is None:
                continue
            metrics.update(cell=name, factor=key, value=value, asset=asset, timeframe=timeframe)
            rows.append(metrics)
            deficits.append(metrics["deficit"])
            meets.append(metrics["meets"])
        if deficits:
            rows.append(
                {
                    "cell": name,
                    "asset": "ALL",
                    "timeframe": "ALL",
                    "deficit": float(sum(deficits) / len(deficits)),
                    "meets_all": all(meets) and len(meets) == len(TAPES),
                    "n": len(deficits),
                }
            )
    return rows


def _combo_rows() -> list[dict]:
    rows = []
    if not GRID.exists():
        return rows
    plan = {}
    plan_path = GRID / "combo_plan.json"
    if plan_path.exists():
        for combo in json.loads(plan_path.read_text()):
            plan[combo["name"]] = combo
    for path in sorted(p for p in GRID.iterdir() if p.is_dir() and p.name.startswith("combo_")):
        deficits, meets = [], []
        for asset, timeframe in TAPES:
            metrics = _metrics(path / f"{asset.lower()}_{timeframe}")
            if metrics is None:
                continue
            metrics.update(
                cell=path.name,
                factor="combo",
                value=",".join(plan.get(path.name, {}).get("sets", [])),
                asset=asset,
                timeframe=timeframe,
            )
            rows.append(metrics)
            deficits.append(metrics["deficit"])
            meets.append(metrics["meets"])
        if deficits:
            rows.append(
                {
                    "cell": path.name,
                    "asset": "ALL",
                    "timeframe": "ALL",
                    "deficit": float(sum(deficits) / len(deficits)),
                    "meets_all": all(meets) and len(meets) == len(TAPES),
                    "n": len(deficits),
                }
            )
    return rows


def select() -> int:
    rows = _cell_table() + _combo_rows()
    if not rows:
        raise SystemExit("no validation cells to select from")
    out = ROOT / "paper" / "results" / "v2"
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "validation_grid.csv", index=False)
    summary = [row for row in rows if row.get("asset") == "ALL"]
    summary.sort(key=lambda row: (not row.get("meets_all", False), row["deficit"], row["cell"]))
    (out / "validation_ranking.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def freeze() -> int:
    """Write configs/final.yaml from validation deficits only. Holdout files are not read."""
    select()
    ranking = json.loads((ROOT / "paper" / "results" / "v2" / "validation_ranking.json").read_text())
    complete = [row for row in ranking if row.get("n") == len(TAPES)]
    if not complete:
        raise SystemExit("no complete validation cell")
    winner = sorted(complete, key=lambda row: (not row.get("meets_all", False), row["deficit"], row["cell"]))[0]
    from crta.config import ExperimentConfig

    cfg = ExperimentConfig.load(ROOT / "configs" / "default.yaml")
    lookup = {name: (key, value) for name, key, value in FACTORS}
    plan = {}
    plan_path = GRID / "combo_plan.json"
    if plan_path.exists():
        plan = {combo["name"]: combo for combo in json.loads(plan_path.read_text())}
    overrides = []
    if winner["cell"] in plan:
        overrides = plan[winner["cell"]]["sets"]
    elif winner["cell"] in lookup and lookup[winner["cell"]][0] is not None:
        key, value = lookup[winner["cell"]]
        overrides = [f"{key}={value}"]
    for item in overrides:
        key, value = item.split("=", 1)
        cfg = cfg.with_overrides(**{key: _parse(value)})
    cfg.dump(ROOT / "configs" / "final.yaml")
    import pandas as pd

    grid = pd.read_csv(ROOT / "paper" / "results" / "v2" / "validation_grid.csv")
    detail = grid[(grid["cell"] == winner["cell"]) & (grid["asset"] != "ALL")]
    note = {
        "cell": winner["cell"],
        "meets_all": bool(winner.get("meets_all")),
        "mean_deficit": winner["deficit"],
        "overrides": overrides,
        "criterion": "both twins position std > 0.05, |a|>0.95 share < 0.80, conditioned ablation >= 0.10",
        "tapes": detail.to_dict(orient="records"),
        "reason": (
            "Every validation tape met the pre-registered criterion."
            if winner.get("meets_all")
            else "No cell met the criterion on all four tapes. This is the lowest mean deficit."
        ),
    }
    (ROOT / "paper" / "results" / "v2" / "selection.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2))
    return 0


def _parse(value: str):
    from crta.config import _parse_scalar

    return _parse_scalar(value)


FAMILIES = {
    "reward_mode": ("reward_excess_bh", "reward_diff_sharpe"),
    "window_sampling": ("sample_regime_balanced", "sample_return_balanced"),
    "ent_coef": ("ent_002",),
    "regime_perturb_prob": ("perturb_005", "perturb_01"),
    "tracker": ("tracker_v2",),
}


def _summary_deficits() -> dict[str, float]:
    rows = [row for row in _cell_table() if row.get("asset") == "ALL" and row.get("n") == len(TAPES)]
    return {row["cell"]: float(row["deficit"]) for row in rows}


def plan_combos(max_combos: int = 4) -> list[dict]:
    """Factors that beat the control, best level per family, at most `max_combos` stacks.

    Written before the factor cells were scored. Returns are not used.
    """
    deficits = _summary_deficits()
    if "baseline" not in deficits:
        raise SystemExit("baseline cell is incomplete")
    base = deficits["baseline"]
    helpful = []
    for family, names in FAMILIES.items():
        scored = [(name, deficits[name]) for name in names if name in deficits]
        if not scored:
            continue
        name, deficit = min(scored, key=lambda item: item[1])
        if deficit < base - 1e-12:
            helpful.append((name, base - deficit))
    helpful.sort(key=lambda item: -item[1])
    specs = []
    if len(helpful) >= 2:
        chosen = [name for name, _gap in helpful[:4]]
        specs.append(chosen)
        if len(chosen) >= 3 and len(specs) < max_combos:
            # Leave out the weakest helpful factor once, still ignoring returns.
            specs.append(chosen[:-1])
        if len(chosen) >= 3 and len(specs) < max_combos:
            specs.append(chosen[:2])
    lookup = {name: (key, value) for name, key, value in FACTORS}
    combos = []
    for spec in specs[:max_combos]:
        sets = []
        for name in spec:
            key, value = lookup[name]
            sets.append(f"{key}={value}")
        combos.append({"name": "combo_" + "__".join(spec), "sets": sets, "cells": spec})
    return combos


def _combo_cmd(combo: dict, asset: str, timeframe: str) -> list[str]:
    out = GRID / combo["name"] / f"{asset.lower()}_{timeframe}"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(ROOT / "configs" / "default.yaml"),
        "--asset",
        asset,
        "--tf",
        timeframe,
        "--stage",
        "validate",
        "--seeds",
        "42",
        "--out",
        str(out),
    ]
    for item in combo["sets"]:
        cmd.extend(["--set", item])
    return cmd


def launch_combos(workers: int) -> int:
    combos = plan_combos()
    (GRID / "combo_plan.json").write_text(json.dumps(combos, indent=2))
    print(json.dumps(combos, indent=2), flush=True)
    jobs = []
    for combo in combos:
        for asset, timeframe in TAPES:
            path = GRID / combo["name"] / f"{asset.lower()}_{timeframe}"
            if _done(path):
                continue
            label = f"{combo['name']} {asset} {timeframe}"
            jobs.append((label, _combo_cmd(combo, asset, timeframe)))
    print(f"{len(jobs)} combo cells", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, label, cmd) for label, cmd in jobs]
        for future in as_completed(futures):
            label, code = future.result()
            print(f"{label}: exit {code}", flush=True)
            if code != 0:
                failed.append(label)
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", default="")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--launch-combos", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze:
        return freeze()
    if args.select:
        return select()
    if args.launch_combos:
        return launch_combos(args.workers)
    names = {part for part in args.only.split(",") if part}
    return launch(args.workers, names)


if __name__ == "__main__":
    raise SystemExit(main())

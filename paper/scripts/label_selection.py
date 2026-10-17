"""Choose the regime labeller on the early span only, by a rule fixed before any result was seen.

Candidates (all tracker v2 with filtered probabilities, burn-in-scaled W2, 1,080-bar
burn-in and a warmed filter):

  fixed     the Tier 1 labeller
  guard     templates re-named by their moments after every rebirth and update
  dd        HMM inputs plus the distance below the 90-day high
  guard_dd  both

Selection span: bars [1080, 3937), fold 1's training window, Jul 2021 to Oct 2022, which
holds one hindsight bull and one hindsight bear phase. Evaluation span: bars 3937 on,
reported for every candidate and never used to choose.

Rule, applied on the selection span:
  1. Semantic check. crisis-labelled bars have a higher mean vol_20 than calm- and
     bull-labelled bars, and bull-labelled bars have a higher mean log return than
     bear-labelled bars. A label with fewer than 30 bars fails the check it is needed for.
  2. Among candidates that pass, the highest balanced recall: the mean of the share of
     hindsight bull bars labelled bull and of hindsight bear bars labelled bear or crisis.
  3. Within 0.01 of the best, the candidate listed first above (fewest changes) wins.
  If none passes, the highest balanced recall wins and the failure is recorded.

Usage: .venv/Scripts/python paper/scripts/label_selection.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crta.data import HMM_COLS, HMM_COLS_DD, drawdown  # noqa: E402
from crta.dating import hindsight_phases, label_quality  # noqa: E402
from crta.regime import track_v2  # noqa: E402

FRAME = ROOT / "paper" / "results" / "4h_rerun" / "frame.parquet"
OUT = ROOT / "paper" / "results" / "causal" / "label_selection"
BURN = 1080
SELECT = (1080, 3937)
MIN_BARS = 30
TIE = 0.01
CANDIDATES = {
    "fixed": {"cols": HMM_COLS, "semantic_guard": False},
    "guard": {"cols": HMM_COLS, "semantic_guard": True},
    "dd": {"cols": HMM_COLS_DD, "semantic_guard": False},
    "guard_dd": {"cols": HMM_COLS_DD, "semantic_guard": True},
}


def load_frame() -> pd.DataFrame:
    frame = pd.read_parquet(FRAME)
    frame["drawdown"] = drawdown(frame["close"])
    return frame


def _label(name: str) -> tuple[str, list[str], np.ndarray, list]:
    frame = load_frame()
    spec = CANDIDATES[name]
    diag: dict = {}
    labels, probs = track_v2(
        frame[list(spec["cols"])].to_numpy(dtype=float),
        burn=BURN,
        semantic_guard=spec["semantic_guard"],
        diagnostics=diag,
    )
    rebirths = [e for e in diag.get("events", []) if e.get("kind") == "death_birth"]
    return name, list(labels), probs, rebirths


def _means(labels: np.ndarray, values: np.ndarray, name: str) -> float | None:
    mask = labels == name
    return float(values[mask].mean()) if mask.sum() >= MIN_BARS else None


def span_report(labels, frame: pd.DataFrame, phases: np.ndarray, start: int, end: int) -> dict:
    lab = np.asarray(labels[start:end], dtype=object)
    vol = frame["vol_20"].to_numpy(dtype=float)[start:end]
    ret = frame["log_ret"].to_numpy(dtype=float)[start:end]
    q = label_quality(lab, phases[start:end])
    vols = {g: _means(lab, vol, g) for g in ("bull", "bear", "calm", "crisis")}
    rets = {g: _means(lab, ret, g) for g in ("bull", "bear", "calm", "crisis")}
    crisis_ok = vols["crisis"] is not None and all(
        vols[g] is None or vols["crisis"] > vols[g] for g in ("calm", "bull")
    )
    order_ok = rets["bull"] is not None and rets["bear"] is not None and rets["bull"] > rets["bear"]
    recalls = [q["bull_labelled_bull"], q["bear_labelled_bear_or_crisis"]]
    balanced = None if any(r is None for r in recalls) else float(np.mean(recalls))
    return {
        "bars": [start, end],
        "quality": q,
        "mean_vol": vols,
        "mean_return": rets,
        "semantic_ok": bool(crisis_ok and order_ok),
        "balanced_recall": balanced,
    }


def choose(reports: dict) -> dict:
    names = list(CANDIDATES)
    passing = [n for n in names if reports[n]["selection"]["semantic_ok"]]
    pool = passing or names
    score = {n: reports[n]["selection"]["balanced_recall"] or 0.0 for n in pool}
    best = max(score.values())
    winner = next(n for n in names if n in pool and score[n] >= best - TIE)
    return {"chosen": winner, "passing": passing, "scores": score, "fallback": not passing}


def main() -> int:
    frame = load_frame()
    phases = hindsight_phases(frame["close"].to_numpy(dtype=float))
    with ProcessPoolExecutor(max_workers=len(CANDIDATES)) as pool:
        runs = list(pool.map(_label, CANDIDATES))
    reports = {}
    labels_out = {}
    for name, labels, probs, rebirths in runs:
        reports[name] = {
            "settings": {"cols": list(CANDIDATES[name]["cols"]), "semantic_guard": CANDIDATES[name]["semantic_guard"]},
            "rebirths": rebirths,
            "selection": span_report(labels, frame, phases, *SELECT),
            "evaluation": span_report(labels, frame, phases, SELECT[1], len(frame)),
        }
        labels_out[f"labels_{name}"] = np.array(labels)
        labels_out[f"probs_{name}"] = probs
    decision = choose(reports)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_selection.json").write_text(json.dumps({"decision": decision, "candidates": reports}, indent=2, default=str))
    np.savez_compressed(OUT / "labels.npz", **labels_out)
    rows = []
    for name, rep in reports.items():
        for span in ("selection", "evaluation"):
            s = rep[span]
            q = s["quality"]
            rows.append(
                {
                    "candidate": name,
                    "span": span,
                    "semantic_ok": s["semantic_ok"],
                    "balanced_recall": s["balanced_recall"],
                    "bull_as_bull": q["bull_labelled_bull"],
                    "bear_as_bear_crisis": q["bear_labelled_bear_or_crisis"],
                    "bear_as_bull": q["bear_labelled_bull"],
                    **{f"{g}_share": q["label_share"][g] for g in ("bull", "bear", "calm", "crisis")},
                    **{f"{g}_vol_pct": (None if v is None else 100 * v) for g, v in s["mean_vol"].items()},
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "label_selection.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

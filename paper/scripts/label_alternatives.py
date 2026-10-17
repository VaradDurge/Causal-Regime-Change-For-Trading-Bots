"""Alternative regime labellers next to the HMM: statistical jump model and Wasserstein k-means.

Each family's setting is chosen on the early span only, by the rule of label_selection.py
(semantic check, then balanced recall against hindsight dating, ties to the first listed),
fixed before any result was seen:

  jump      HMM_COLS_DD features, jump penalty in {20, 80, 320}, monthly refits
  wkmeans   sorted windows of 4h log returns, window in {42, 180} bars (one week, one month)

The chosen jump model and Wasserstein k-means are then reported on the evaluation span
beside guard_dd, the HMM labeller of the 4h study. Probabilities are saved for the daily study.

Usage: .venv/Scripts/python paper/scripts/label_alternatives.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.data import HMM_COLS_DD  # noqa: E402
from crta.dating import hindsight_phases  # noqa: E402
from crta.jump import track_jump  # noqa: E402
from crta.wkmeans import track_wkmeans  # noqa: E402
from label_selection import SELECT, TIE, load_frame, span_report  # noqa: E402

OUT = ROOT / "paper" / "results" / "causal" / "label_alternatives"
GUARD_DD = ROOT / "paper" / "results" / "causal_4h_guard_dd" / "labels.npz"
BURN = 1080
FAMILIES = {
    "jump": [("jump_l20", 20.0), ("jump_l80", 80.0), ("jump_l320", 320.0)],
    "wkmeans": [("wk_w42", 42), ("wk_w180", 180)],
}


def _run(spec: tuple[str, str, float]) -> tuple[str, list[str], np.ndarray]:
    family, name, value = spec
    frame = load_frame()
    if family == "jump":
        labels, probs = track_jump(frame[list(HMM_COLS_DD)].to_numpy(dtype=float), burn=BURN, jump_penalty=value, seed=0)
    else:
        labels, probs = track_wkmeans(frame["log_ret"].to_numpy(dtype=float), window=int(value), burn=BURN, seed=0)
    return name, list(labels), probs


def _choose(names: list[str], reports: dict) -> str:
    passing = [n for n in names if reports[n]["selection"]["semantic_ok"]]
    pool = passing or names
    score = {n: reports[n]["selection"]["balanced_recall"] or 0.0 for n in pool}
    best = max(score.values())
    return next(n for n in names if n in pool and score[n] >= best - TIE)


def main() -> int:
    frame = load_frame()
    phases = hindsight_phases(frame["close"].to_numpy(dtype=float))
    specs = [(family, name, value) for family, items in FAMILIES.items() for name, value in items]
    with ProcessPoolExecutor(max_workers=len(specs)) as pool:
        runs = list(pool.map(_run, specs))
    guard = np.load(GUARD_DD)
    runs.append(("guard_dd", list(guard["labels"]), guard["probs"]))
    reports, saved = {}, {}
    for name, labels, probs in runs:
        reports[name] = {
            "selection": span_report(labels, frame, phases, *SELECT),
            "evaluation": span_report(labels, frame, phases, SELECT[1], len(frame)),
        }
        saved[f"labels_{name}"] = np.array(labels)
        saved[f"probs_{name}"] = probs
    chosen = {family: _choose([n for n, _v in items], reports) for family, items in FAMILIES.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "labels.npz", **saved)
    (OUT / "label_alternatives.json").write_text(json.dumps({"chosen": chosen, "candidates": reports}, indent=2, default=str))
    rows = []
    for name, rep in reports.items():
        for span in ("selection", "evaluation"):
            s = rep[span]
            q = s["quality"]
            rows.append(
                {
                    "labeller": name,
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
    table.to_csv(OUT / "label_alternatives.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))
    print(json.dumps({"chosen": chosen}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

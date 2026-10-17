"""Re-score the published four-hour run with the counterfactual limited to feasible directions.

The original pass flags are read from the run, not recomputed. A bar is untestable when the
direction its regime requires is out of reach from the position the agent already holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crta.regime import REGIMES
from crta.verify import CF_MIN_DELTA, DEFENSE

ROOT = Path(__file__).resolve().parents[2]
BARS = ROOT / "paper" / "results" / "run" / "bars.parquet"
OUT = ROOT / "paper" / "results" / "bound_aware_4h.json"


def main() -> None:
    bars = pd.read_parquet(BARS)
    action = bars["action_conditioned"].to_numpy(dtype=float)
    rank = np.array([DEFENSE[REGIMES[i]] for i in bars[list(REGIMES)].to_numpy().argmax(axis=1)])
    blocked = ((rank > DEFENSE["bull"]) & (action > 1.0 - CF_MIN_DELTA)) | (
        (rank < DEFENSE["crisis"]) & (action < -1.0 + CF_MIN_DELTA)
    )
    testable = ~blocked
    passed = bars["passed"].to_numpy(dtype=bool)
    pass_cf = bars["pass_cf"].to_numpy(dtype=bool)
    out = {
        "n_bars": int(len(bars)),
        "n_passed": int(passed.sum()),
        "faith_rate": float(passed.mean()),
        "n_untestable": int(blocked.sum()),
        "untestable_share": float(blocked.mean()),
        "n_testable": int(testable.sum()),
        "faith_rate_testable": float(passed[testable].mean()),
        "cf_rate_testable": float(pass_cf[testable].mean()),
        "share_abs_action_gt_095": float(np.mean(np.abs(action) > 0.95)),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

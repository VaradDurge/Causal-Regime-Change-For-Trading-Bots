"""Run the project pipeline (Wasserstein tracker + PPO twins + audit) into paper/results.

Wraps crta.pipeline.build so that nothing under outputs/ is overwritten.
Usage: .venv/bin/python paper/scripts/run_pipeline.py [--probe]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import crta.pipeline as pipeline
from crta.plots import save_figures

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"


def main() -> int:
    mode = "probe" if "--probe" in sys.argv else "trained"
    run_dir = RESULTS / ("run_probe" if mode == "probe" else "run")
    fig_dir = RESULTS / ("console_probe" if mode == "probe" else "console")
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pipeline.RUN = run_dir
    pipeline.save_figures = lambda df, summary, spot, curve: save_figures(df, summary, spot, curve, out_dir=fig_dir)
    t0 = time.time()
    pipeline.build(mode)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

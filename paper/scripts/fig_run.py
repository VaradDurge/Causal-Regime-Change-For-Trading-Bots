"""Result figures from paper/results/run (current W2 pipeline) and outputs/run (archived)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from crta.data import ROOT
from crta.regime import REGIMES

PAPER = Path(__file__).resolve().parents[1]
RES = PAPER / "results"
FIG = PAPER / "figures"


def equity(new, old):
    ts = pd.DatetimeIndex(pd.to_datetime(new["timestamp"], utc=True))
    fig, ax = plt.subplots(figsize=(style.COL, 1.95))
    ax.plot(ts, new["equity_bh"], color=style.SYSTEM_COLORS["buy_hold"], lw=0.8, label="buy and hold")
    ax.plot(ts, new["equity_conditioned"], color=style.SYSTEM_COLORS["conditioned"], lw=0.9, label="conditioned ($W_2$ labels)")
    ax.plot(ts, new["equity_blind"], color=style.SYSTEM_COLORS["blind"], lw=0.9, label="blind")
    if old is not None:
        ax.plot(ts, old["equity_conditioned"], color=style.SYSTEM_COLORS["conditioned"], lw=0.7, ls=(0, (2, 1.2)), alpha=0.8,
                label="conditioned (archived labels)")
    starts = new.groupby("fold").head(1).index.to_numpy()
    for i in starts[1:]:
        ax.axvline(ts[i], color="0.5", lw=0.5, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("wealth (log scale)")
    ax.axhline(1.0, color="0.7", lw=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handlelength=1.8, columnspacing=0.8)
    fig.savefig(FIG / "equity.pdf")
    plt.close(fig)


def faithfulness(an):
    groups = [
        ("PPO\n$W_2$ labels", an["checks_w2"]),
        ("PPO\narchived labels", an["checks_archived"]),
        ("probe", an["reference_policies_w2"]["probe"]),
        ("ignore", an["reference_policies_w2"]["ignore"]),
        ("inverted", an["reference_policies_w2"]["inverted"]),
    ]
    checks = ["ablation", "counterfactual", "sensitivity", "all"]
    names = ["ablation", "counterfact.", "sensitivity", "all three"]
    colors = ["#999999", "#E69F00", "#56B4E9", "#000000"]
    fig, ax = plt.subplots(figsize=(style.COL, 1.75))
    x = np.arange(len(groups))
    w = 0.19
    for k, (c, label, col) in enumerate(zip(checks, names, colors)):
        vals = [g[1][c] * 100 for g in groups]
        bars = ax.bar(x + (k - 1.5) * w, vals, w, color=col, label=label, lw=0)
        if c == "all":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
    ax.set_ylabel("bars passing (%)")
    ax.set_ylim(0, 112)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6.5, handlelength=1.0, columnspacing=0.8)
    fig.savefig(FIG / "faithfulness.pdf")
    plt.close(fig)


def interventions(new, iv):
    labels = new["regime"].to_numpy()
    grid = np.linspace(0, 1, iv["curve"].shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(style.COL, 1.75), gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.42})
    ax = axes[0]
    for r in REGIMES:
        m = labels == r
        if not m.any():
            continue
        c = iv["curve"][m]
        q1, med, q3 = np.percentile(c, [25, 50, 75], axis=0)
        ax.plot(grid, c.mean(axis=0), color=style.REGIME_COLORS[r], lw=1.1, label=f"{r} ($n$={m.sum()})")
        ax.plot(grid, med, color=style.REGIME_COLORS[r], lw=0.6, ls=":")
    ax.set_xlabel("injected $p$(crisis)")
    ax.set_ylabel("position")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower right", fontsize=5.5, handlelength=1.0)
    ax.set_title("(a) crisis sweep", fontsize=7)

    ax = axes[1]
    keys = ["actual", "ablated", "forced_bull", "forced_calm", "forced_bear", "forced_crisis"]
    ticks = ["actual", "zeroed", "bull", "calm", "bear", "crisis"]
    data = [iv[k] for k in keys]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.8),
                    whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
    fills = ["#BBBBBB", "#DDDDDD", style.REGIME_COLORS["bull"], style.REGIME_COLORS["calm"], style.REGIME_COLORS["bear"], style.REGIME_COLORS["crisis"]]
    for patch, col in zip(bp["boxes"], fills):
        patch.set_facecolor(col)
    for i, d in enumerate(data):
        ax.scatter(i + 1, np.mean(d), marker="D", s=6, color="white", ec="black", lw=0.4, zorder=3)
    ax.set_xticks(range(1, len(ticks) + 1))
    ax.set_xticklabels(ticks, rotation=40, fontsize=6)
    ax.axvline(2.5, color="0.6", lw=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("(b) forced one-hot regime", fontsize=7)
    fig.savefig(FIG / "interventions.pdf")
    plt.close(fig)


def main():
    style.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    new = pd.read_parquet(RES / "run" / "bars.parquet")
    old_path = ROOT / "outputs" / "run" / "bars.parquet"
    old = pd.read_parquet(old_path) if old_path.exists() else None
    an = json.loads((RES / "analysis.json").read_text())
    iv = np.load(RES / "analysis.npz")
    equity(new, old)
    faithfulness(an)
    interventions(new, iv)


if __name__ == "__main__":
    main()

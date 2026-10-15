"""Tables and figures for the v2 study. Reads saved JSON and parquet only.

  python paper/scripts/v2_outputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style

V2 = ROOT / "paper" / "results" / "v2"
FIG = ROOT / "paper" / "figures" / "v2"
TAB = V2 / "tables"
TAPES = (("BTC", "1h"), ("BTC", "15m"), ("ETH", "1h"), ("ETH", "15m"))


def _fmt(mean, std, digits=2) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def performance_table() -> pd.DataFrame | None:
    rows = []
    for asset, timeframe in TAPES:
        summary = _load(V2 / "final" / f"{asset.lower()}_{timeframe}" / "summary.json")
        if summary is None:
            continue
        for segment, block in summary["segments"].items():
            for side in ("conditioned", "blind"):
                stats = block[side]
                rows.append(
                    {
                        "asset": asset,
                        "timeframe": timeframe,
                        "segment": segment,
                        "side": side,
                        "sharpe": _fmt(stats["sharpe"]["mean"], stats["sharpe"]["std"]),
                        "sortino": _fmt(stats["sortino"]["mean"], stats["sortino"]["std"]),
                        "total_return": _fmt(stats["total_return"]["mean"], stats["total_return"]["std"], 3),
                        "max_drawdown": _fmt(stats["max_drawdown"]["mean"], stats["max_drawdown"]["std"], 3),
                        "mean_position": _fmt(stats["mean_position"]["mean"], stats["mean_position"]["std"], 3),
                        "position_std": _fmt(stats["position_std"]["mean"], stats["position_std"]["std"], 3),
                        "extreme_share": _fmt(stats["extreme_share"]["mean"], stats["extreme_share"]["std"], 3),
                        "turnover": _fmt(stats["turnover"]["mean"], stats["turnover"]["std"], 4),
                        "ablation": _fmt(block["ablation_pass_rate"]["mean"], block["ablation_pass_rate"]["std"], 3),
                        "faithfulness": _fmt(block["faithfulness_rate"]["mean"], block["faithfulness_rate"]["std"], 3),
                        "cond_beats_blind_share": block.get("cond_beats_blind_share"),
                        "regularized": block.get("regularized", False),
                        "sharpe_mean": stats["sharpe"]["mean"],
                        "sharpe_std": stats["sharpe"]["std"],
                        "return_mean": stats["total_return"]["mean"],
                        "return_std": stats["total_return"]["std"],
                        "std_mean": stats["position_std"]["mean"],
                        "std_std": stats["position_std"]["std"],
                        "extreme_mean": stats["extreme_share"]["mean"],
                        "extreme_std": stats["extreme_share"]["std"],
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows)


def collapse_table(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in perf.iterrows():
        std = row["std_mean"]
        rows.append(
            {
                "asset": row["asset"],
                "timeframe": row["timeframe"],
                "segment": row["segment"],
                "side": row["side"],
                "position_std": row["position_std"],
                "extreme_share": row["extreme_share"],
                "collapsed": bool(std is not None and std <= 0.05),
            }
        )
    return pd.DataFrame(rows)


def per_regime_table() -> pd.DataFrame | None:
    rows = []
    for asset, timeframe in TAPES:
        parent = V2 / "final" / f"{asset.lower()}_{timeframe}"
        paths = sorted(parent.glob("seed_*/eval.json"))
        if not paths:
            continue
        for path in paths:
            payload = json.loads(path.read_text())
            for segment, block in payload["segments"].items():
                if block.get("skipped"):
                    continue
                for name, stats in block.get("regimes", {}).items():
                    rows.append(
                        {
                            "asset": asset,
                            "timeframe": timeframe,
                            "segment": segment,
                            "seed": payload["seed"],
                            "regime": name,
                            **stats,
                        }
                    )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    value_cols = [
        col
        for col in (
            "share",
            "cond_position",
            "blind_position",
            "cond_position_std",
            "blind_position_std",
            "cond_return",
            "blind_return",
        )
        if col in frame.columns
    ]
    grouped = frame.groupby(["asset", "timeframe", "segment", "regime"], as_index=False)[value_cols].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped


def faithfulness_chart(path_stem: Path) -> bool:
    rows = []
    found = False
    for asset, timeframe in TAPES:
        parent = V2 / "final" / f"{asset.lower()}_{timeframe}"
        for eval_path in sorted(parent.glob("seed_*/eval.json")):
            payload = json.loads(eval_path.read_text())
            for segment, block in payload["segments"].items():
                faith = block.get("faithfulness") or {}
                legacy = faith.get("legacy") or {}
                newer = faith.get("v2") or {}
                for verifier, report in (("legacy", legacy), ("v2", newer)):
                    if not report:
                        continue
                    rows.append(
                        {
                            "tape": f"{asset} {timeframe}",
                            "segment": segment,
                            "verifier": verifier,
                            "rate": report.get("all"),
                            "regularized": faith.get("regularized", False),
                        }
                    )
                    found = True
    if not found:
        return False
    frame = pd.DataFrame(rows)
    frame.to_csv(TAB / "faithfulness.csv", index=False)
    style.apply()
    segments = list(frame["segment"].unique())
    fig, axes = plt.subplots(1, len(segments), figsize=(style.FULL, 2.4), sharey=True)
    if len(segments) == 1:
        axes = [axes]
    for ax, segment in zip(axes, segments):
        sub = frame[frame["segment"] == segment]
        tapes = list(sub["tape"].unique())
        x = np.arange(len(tapes))
        width = 0.36
        for offset, verifier, color in ((-width / 2, "legacy", "#0072B2"), (width / 2, "v2", "#D55E00")):
            vals = [
                float(sub[(sub["tape"] == tape) & (sub["verifier"] == verifier)]["rate"].mean())
                if ((sub["tape"] == tape) & (sub["verifier"] == verifier)).any()
                else 0.0
                for tape in tapes
            ]
            ax.bar(x + offset, vals, width=width, label=verifier, color=color)
        ax.set_xticks(x, tapes, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(segment)
    axes[0].set_ylabel("Faithfulness rate")
    axes[-1].legend(loc="upper right")
    regularized = bool(frame["regularized"].any())
    title = "Faithfulness rate: legacy verifier and verifier v2"
    if regularized:
        title += " (regularized agent)"
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".pdf"))
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return True


def wealth_chart() -> bool:
    drew = False
    style.apply()
    for segment in ("holdout", "live"):
        fig, axes = plt.subplots(2, 2, figsize=(style.FULL, 4.2), sharex=False)
        empty = True
        for ax, (asset, timeframe) in zip(axes.ravel(), TAPES):
            parent = V2 / "final" / f"{asset.lower()}_{timeframe}"
            curves_c, curves_b = [], []
            for path in sorted(parent.glob("seed_*/bars.parquet")):
                bars = pd.read_parquet(path)
                sub = bars[bars["segment"] == segment]
                if sub.empty:
                    continue
                curves_c.append(sub["equity_conditioned"].to_numpy(dtype=float))
                curves_b.append(sub["equity_blind"].to_numpy(dtype=float))
            ax.set_title(f"{asset} {timeframe}")
            if not curves_c:
                ax.set_axis_off()
                continue
            empty = False
            drew = True
            n = min(len(c) for c in curves_c)
            stack_c = np.stack([c[:n] for c in curves_c])
            stack_b = np.stack([c[:n] for c in curves_b])
            for row in stack_c:
                ax.plot(row, color="#0072B2", alpha=0.35, lw=0.6)
            for row in stack_b:
                ax.plot(row, color="#CC79A7", alpha=0.35, lw=0.6)
            ax.plot(stack_c.mean(axis=0), color="#0072B2", lw=1.4, label="conditioned")
            ax.plot(stack_b.mean(axis=0), color="#CC79A7", lw=1.4, label="blind")
            ax.axhline(1.0, color="#444444", lw=0.4)
        if empty:
            plt.close(fig)
            continue
        axes.ravel()[0].legend(loc="best")
        fig.suptitle(f"{segment} wealth, one line per seed", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"wealth_{segment}.pdf")
        fig.savefig(FIG / f"wealth_{segment}.png", dpi=200)
        plt.close(fig)
    return drew


def label_table_figure(frame: pd.DataFrame) -> None:
    frame.to_csv(TAB / "label_stats.csv", index=False)
    style.apply()
    fig, ax = plt.subplots(figsize=(style.FULL, 2.6))
    ax.axis("off")
    show = frame.copy()
    for col in show.columns:
        if show[col].dtype.kind == "f":
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    table = ax.table(cellText=show.values, colLabels=list(show.columns), loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.2)
    fig.savefig(FIG / "label_stats.pdf")
    fig.savefig(FIG / "label_stats.png", dpi=200)
    plt.close(fig)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    perf = performance_table()
    if perf is not None:
        perf.to_csv(TAB / "walkforward.csv", index=False)
        hold = perf[perf["segment"].isin(["holdout", "live"])]
        hold.to_csv(TAB / "holdout_live.csv", index=False)
        collapse_table(perf).to_csv(TAB / "collapse.csv", index=False)
        style.apply()
        fig, ax = plt.subplots(figsize=(style.FULL, 2.8))
        ax.axis("off")
        show = hold[
            ["asset", "timeframe", "segment", "side", "total_return", "sharpe", "position_std", "extreme_share"]
        ]
        table = ax.table(cellText=show.values, colLabels=list(show.columns), loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.15)
        fig.savefig(FIG / "holdout_live.pdf")
        fig.savefig(FIG / "holdout_live.png", dpi=200)
        plt.close(fig)
        print(f"performance rows {len(perf)}")
    regimes = per_regime_table()
    if regimes is not None:
        regimes.to_csv(TAB / "per_regime.csv", index=False)
        print(f"per-regime rows {len(regimes)}")
    label_path = TAB / "label_stats_raw.csv"
    if label_path.exists():
        label_table_figure(pd.read_csv(label_path))
        print("label figure")
    if faithfulness_chart(FIG / "faithfulness_verifiers"):
        print("faithfulness chart")
    if wealth_chart():
        print("wealth charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

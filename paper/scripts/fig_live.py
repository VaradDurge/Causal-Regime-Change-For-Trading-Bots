"""Twin wealth on 15 min, 1 h and 4 h candles: the live window and the one-year holdout.

The 15 min and 1 h twins come from timeframe_study.py, the 4 h twins from live_study.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import style

PAPER = Path(__file__).resolve().parents[1]
LIVE = PAPER / "results" / "live"
TF = PAPER / "results" / "tf"
FIG = PAPER / "figures"
NAMES = {"15m": "15 min", "1h": "1 h", "4h": "4 h"}


def _shade(ax, ts, labels):
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            end = ts[i] if i < len(ts) else ts[-1]
            ax.axvspan(ts[start], end, color=style.REGIME_COLORS[labels[start]], alpha=0.13, lw=0)
            start = i


def _segment(asset: str, tf: str, segment: str) -> pd.DataFrame:
    if tf == "4h":
        bars = pd.read_parquet(LIVE / "live_bars.parquet")
        return bars.loc[bars["timeframe"] == "4h"].reset_index(drop=True)
    bars = pd.read_parquet(TF / f"{asset.lower()}_{tf}_bars.parquet")
    return bars.loc[bars["segment"] == segment].reset_index(drop=True)


def _panels(subs: dict, path: Path, shape: tuple, width: float, height: float, locator, fmt: str) -> None:
    fig, axes = plt.subplots(*shape, figsize=(width, height), sharex=True, squeeze=False)
    axes = axes.ravel()
    seen: set[str] = set()
    for ax, (name, sub) in zip(axes, subs.items()):
        ts = pd.DatetimeIndex(pd.to_datetime(sub["timestamp"], utc=True)).tz_convert("Asia/Kolkata")
        _shade(ax, ts, sub["regime"].to_numpy())
        seen |= set(sub["regime"])
        ax.plot(ts, sub["equity_conditioned"], color=style.SYSTEM_COLORS["conditioned"], lw=0.9, label="conditioned")
        ax.plot(ts, sub["equity_blind"], color=style.SYSTEM_COLORS["blind"], lw=0.9, label="blind")
        # Drawn last and dashed: a twin that holds +1 would otherwise hide it.
        ax.plot(ts, sub["equity_bh"], color=style.SYSTEM_COLORS["buy_hold"], lw=0.7, ls=(0, (3, 1.5)),
                label="buy and hold")
        ax.axhline(1.0, color="0.6", lw=0.5, ls=":")
        ax.text(0.01, 0.94, f"{name} ({len(sub):,} bars)", transform=ax.transAxes, va="top", fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.2})
    for ax in axes[:: shape[1]]:
        ax.set_ylabel("wealth")
    for ax in axes[-shape[1]:]:
        ax.xaxis.set_major_locator(locator(ts.tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=ts.tz))
        ax.set_xlabel("date (IST)")
    handles, labels = axes[0].get_legend_handles_labels()
    for name in [n for n in style.REGIME_COLORS if n in seen]:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=style.REGIME_COLORS[name], alpha=0.3, lw=0))
        labels.append(name)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4 + 3 * (shape[1] > 1),
               fontsize=6, handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.align_ylabels(axes)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    style.apply()
    days = mdates.DayLocator
    live = {f"BTC {NAMES[tf]}": _segment("BTC", tf, "live") for tf in ("15m", "1h", "4h")}
    _panels(live, FIG / "live_equity.pdf", (3, 1), style.COL, 3.6, lambda tz: days(interval=4, tz=tz), "%d %b")
    eth = {f"ETH {NAMES[tf]}": _segment("ETH", tf, "live") for tf in ("15m", "1h")}
    _panels(eth, FIG / "live_equity_eth.pdf", (2, 1), style.COL, 2.5, lambda tz: days(interval=4, tz=tz), "%d %b")
    hold = {f"{a} {NAMES[tf]}": _segment(a, tf, "holdout") for tf in ("15m", "1h") for a in ("BTC", "ETH")}
    _panels(hold, FIG / "holdout_equity.pdf", (2, 2), style.FULL, 3.4,
            lambda tz: mdates.MonthLocator(bymonth=[1, 4, 7, 10], tz=tz), "%b %y")


if __name__ == "__main__":
    main()

"""Figures for the Wasserstein identity tracker, from paper/results/tracker.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

import style
from crta.data import RAW_PATH, model_frame
from crta.eval import walk_forward_bounds
from crta.regime import BURN, REGIMES

PAPER = Path(__file__).resolve().parents[1]
RES = PAPER / "results"
FIG = PAPER / "figures"


def runs(labels):
    out = []
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        out.append((i, j, labels[i]))
        i = j
    return out


def strip(ax, ts, labels, y0=0.0, h=1.0):
    num = mdates.date2num(ts)
    step = num[1] - num[0]
    for i, j, name in runs(list(labels)):
        ax.broken_barh([(num[i], num[j - 1] - num[i] + step)], (y0, h), color=style.REGIME_COLORS[name], lw=0)


def regime_price(d, ts):
    close = d["close"]
    lab_w2 = d["labels_w2"]
    lab_ms = d["labels_ms"]
    probs = d["probs_w2"]
    n = len(close)
    bounds = walk_forward_bounds(n, min_train=1080)

    fig, axes = plt.subplots(
        3, 1, figsize=(style.FULL, 2.75), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.1, 0.75], "hspace": 0.08}
    )
    ax = axes[0]
    num = mdates.date2num(ts)
    step = num[1] - num[0]
    for i, j, name in runs(list(lab_w2)):
        ax.axvspan(num[i], num[j - 1] + step, color=style.REGIME_COLORS[name], alpha=0.28, lw=0)
    ax.plot(ts, close, color="black", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("BTC/USDT close")
    ax.axvline(ts[BURN], color="0.3", lw=0.6, ls=":")
    ax.text(ts[BURN], close.min() * 0.97, " burn-in end", fontsize=6, va="bottom", color="0.3")
    for k, (train_end, _end) in enumerate(bounds):
        for a in axes:
            a.axvline(ts[train_end], color="0.35", lw=0.5, ls="--")
        ax.text(ts[train_end], close.max() * 1.05, f" test {k}", fontsize=6, va="top", color="0.25")
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.REGIME_COLORS[r], alpha=0.6) for r in REGIMES]
    ax.legend(handles, REGIMES, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.17), handlelength=1.2, columnspacing=1.0)

    ax = axes[1]
    ax.fill_between(ts, probs[:, REGIMES.index("crisis")], color=style.REGIME_COLORS["crisis"], alpha=0.7, lw=0)
    ax.fill_between(
        ts, -probs[:, REGIMES.index("bear")], color=style.REGIME_COLORS["bear"], alpha=0.7, lw=0
    )
    ax.axhline(0, color="0.2", lw=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["1", "0", "1"])
    ax.set_ylabel("$p$(bear) / $p$(crisis)", fontsize=6.5)

    ax = axes[2]
    strip(ax, ts, lab_w2, 1.1, 0.9)
    strip(ax, ts, lab_ms, 0.0, 0.9)
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.45, 1.55])
    ax.set_yticklabels(["moment sort", "$W_2$ tracker"], fontsize=6.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(ts[0], ts[-1])
    fig.align_ylabels(axes)
    fig.savefig(FIG / "regime_price.pdf")
    plt.close(fig)


def templates(d, ts):
    t = d["t"]
    tmu = d["tmu"]  # (refits, 4, 3) in original units, before the update at that refit
    mu = d["mu"]
    cost = d["cost"]
    A = d["assign"]
    M = d["moment"]
    when = ts[t]
    fig, axes = plt.subplots(2, 1, figsize=(style.COL, 2.4), sharex=True, gridspec_kw={"hspace": 0.12})
    ax = axes[0]
    for k in range(mu.shape[1]):
        colors = [style.REGIME_COLORS[REGIMES[g]] for g in A[:, k]]
        ax.scatter(when, mu[:, k, 1] * 100, s=2.0, c=colors, lw=0, alpha=0.55)
    for g, name in enumerate(REGIMES):
        ax.plot(when, tmu[:, g, 1] * 100, color=style.REGIME_COLORS[name], lw=1.3, label=name)
    ax.set_ylabel(r"$\sigma_{20}$ mean (%)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.5, 1, 2, 4])
    ax.legend(ncol=4, loc="upper right", handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(1.0, 1.18))

    ax = axes[1]
    idx = np.arange(cost.shape[1])
    c_w2 = np.array([c[idx, a].mean() for c, a in zip(cost, A)])
    c_ms = np.array([c[idx, m].mean() for c, m in zip(cost, M)])
    ax.plot(when, c_ms * 1e4, color="0.45", lw=0.8, label="moment sort")
    ax.plot(when, c_w2 * 1e4, color=style.SYSTEM_COLORS["conditioned"], lw=0.9, label="$W_2$ nearest template")
    ax.set_yscale("log")
    ax.set_ylabel(r"mean $W_2^2$ to template ($\times10^{-4}$)", fontsize=6.5)
    ax.legend(loc="upper right", ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.align_ylabels(axes)
    fig.savefig(FIG / "templates.pdf")
    plt.close(fig)


def cost_matrix(d, ts, target="2022-06-18"):
    t = d["t"]
    when = pd.DatetimeIndex(ts[t])
    r = int(np.argmin(np.abs((when - pd.Timestamp(target, tz="UTC")).total_seconds())))
    C = d["cost"][r] * 1e4
    A = d["assign"][r]
    M = d["moment"][r]
    mu = d["mu"][r]
    tmu = d["tmu"][r]
    fig, axes = plt.subplots(1, 2, figsize=(style.COL, 1.75), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.55})
    ax = axes[0]
    logc = np.log10(C)
    im = ax.imshow(logc, cmap="Blues_r", aspect="auto")
    lo, hi = logc.min(), logc.max()
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            val = C[i, j]
            txt = f"{val:.1f}" if val < 100 else f"{val:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color="white" if (logc[i, j] - lo) / (hi - lo) < 0.45 else "black")
        ax.add_patch(plt.Rectangle((A[i] - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.3))
        ax.plot(M[i], i + 0.33, marker="^", ms=3, color="#D55E00")
    ax.set_xticks(range(4))
    ax.set_xticklabels(REGIMES, rotation=40, fontsize=6)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"comp. {k}" for k in range(4)], fontsize=6)
    ax.set_xlabel("template", fontsize=7)
    ax.spines[:].set_visible(False)

    ax = axes[1]
    for g, name in enumerate(REGIMES):
        ax.scatter(tmu[g, 1] * 100, tmu[g, 0] * 1e4, marker="s", s=28, color=style.REGIME_COLORS[name], ec="black", lw=0.4, zorder=3)
        ax.annotate(name, (tmu[g, 1] * 100, tmu[g, 0] * 1e4), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    for k in range(len(mu)):
        g = A[k]
        ax.scatter(mu[k, 1] * 100, mu[k, 0] * 1e4, marker="o", s=14, color=style.REGIME_COLORS[REGIMES[g]], ec="black", lw=0.3, zorder=4)
        ax.annotate("", xy=(tmu[g, 1] * 100, tmu[g, 0] * 1e4), xytext=(mu[k, 1] * 100, mu[k, 0] * 1e4),
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="0.3"))
        ax.annotate(str(k), (mu[k, 1] * 100, mu[k, 0] * 1e4), fontsize=5.5, xytext=(-6, -6), textcoords="offset points")
    ax.set_xlabel(r"$\sigma_{20}$ mean (%)", fontsize=7)
    ax.set_ylabel("log-return mean (bp)", fontsize=7)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1, 1.5, 2, 3])
    fig.savefig(FIG / "w2_matrix.pdf")
    plt.close(fig)
    return str(when[r]), int(t[r])


def main():
    style.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    d = np.load(RES / "tracker.npz", allow_pickle=True)
    raw = pd.read_parquet(RAW_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    frame = model_frame(raw)
    assert np.allclose(frame["close"].to_numpy(), d["close"])
    ts = pd.DatetimeIndex(frame["timestamp"])
    regime_price(d, ts)
    templates(d, ts)
    print("w2 matrix refit at", cost_matrix(d, ts))


if __name__ == "__main__":
    main()

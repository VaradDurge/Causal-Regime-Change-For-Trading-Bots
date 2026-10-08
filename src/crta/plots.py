"""Shared paper palette for the console and the PNG pack."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.font_manager import FontProperties
from plotly.subplots import make_subplots

from crta.data import ROOT
from crta.regime import REGIMES

PAPER = "#f4f0e6"
INK = "#1c1915"
RULE = "#d9d2c5"
MUTED = "#8a8175"
BULL = "#b5813a"
BEAR = "#3d5a80"
CALM = "#6b7f6a"
CRISIS = "#9c3b2e"
OK = "#2f6b4f"
REGIME_COLOR = {"bull": BULL, "bear": BEAR, "calm": CALM, "crisis": CRISIS}
DISCLAIMER = "Historical research. Not a trading signal."
FIG = ROOT / "outputs" / "figures"


def _fonts():
    folder = ROOT / "assets" / "fonts"
    return {
        "display": FontProperties(fname=str(folder / "Fraunces-600.ttf")),
        "mono": FontProperties(fname=str(folder / "IBMPlexMono-400.ttf")),
        "body": FontProperties(fname=str(folder / "Literata-400.ttf")),
    }


def _spans(labels):
    spans = []
    start = 0
    labels = list(labels)
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            spans.append((start, i, labels[start]))
            start = i
    return spans


def _style_plotly(fig, title: str, height: int):
    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        height=height,
        margin=dict(l=48, r=24, t=92, b=36),
        title=dict(
            text=title,
            font=dict(family="Fraunces, Georgia, serif", size=22, color=INK),
            x=0,
            xref="paper",
            y=1,
            yref="container",
            yanchor="top",
            pad=dict(t=14),
        ),
        font=dict(family="IBM Plex Mono, ui-monospace, monospace", color=INK, size=12),
        # Fixed entry width: Plotly measures before the web font loads and would clip the last label.
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            entrywidth=170,
            entrywidthmode="pixels",
        ),
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=11))
    # Price axis refits to the visible window in nav.js, like a trading chart.
    fig.update_yaxes(gridcolor="#e6dece", zeroline=False, tickfont=dict(size=11), fixedrange=True)
    return fig


def _colorscale():
    # Normalized heatmap positions for z = 0,1,2,3.
    return [
        [0.0, BULL],
        [0.166, BULL],
        [0.167, BEAR],
        [0.50, BEAR],
        [0.501, CALM],
        [0.833, CALM],
        [0.834, CRISIS],
        [1.0, CRISIS],
    ]


def price_figure(
    df: pd.DataFrame,
    title: str,
    mark=None,
    height: int = 460,
    ribbon: float = 0.16,
    kind: str = "line",
    xrange=None,
) -> go.Figure:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    codes = np.array([REGIMES.index(r) for r in df["regime"]])
    names = np.array(df["regime"]).reshape(1, -1)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[1 - ribbon, ribbon],
        vertical_spacing=0.04,
    )
    if kind == "candle":
        fig.add_trace(
            go.Candlestick(
                x=ts,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=INK,
                decreasing_line_color=CRISIS,
                increasing_fillcolor=INK,
                decreasing_fillcolor=CRISIS,
                name="BTC",
                customdata=np.stack([df["regime"]], axis=1),
                hovertemplate=(
                    "%{x}<br>open %{open:,.0f}<br>high %{high:,.0f}"
                    "<br>low %{low:,.0f}<br>close %{close:,.0f}<br>%{customdata[0]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        lows, highs = df["low"], df["high"]
    else:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=df["close"],
                mode="lines",
                line=dict(color=INK, width=1.6),
                name="BTC",
                customdata=np.stack([df["regime"], df["crisis"]], axis=1),
                hovertemplate="%{x}<br>%{y:,.0f}<br>%{customdata[0]}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        lows = highs = df["close"]
    fig.add_trace(
        go.Heatmap(
            x=ts,
            z=codes.reshape(1, -1),
            y=["regime"],
            colorscale=_colorscale(),
            zmin=0,
            zmax=3,
            showscale=False,
            customdata=names,
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{customdata}<extra></extra>",
            name="regime",
        ),
        row=2,
        col=1,
    )
    seen = np.ones(len(df), dtype=bool)
    if xrange is not None:
        seen = ((ts >= xrange[0]) & (ts <= xrange[1])).to_numpy()
        fig.update_xaxes(range=list(xrange))
    low = float(lows[seen].min())
    high = float(highs[seen].max())
    pad = (high - low) * 0.06 or 1.0
    fig.update_yaxes(title_text="USD", tickformat=",.0f", range=[low - pad, high + pad], row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    if mark is not None:
        fig.add_vline(x=mark, line_width=1, line_dash="dot", line_color=CRISIS, row=1, col=1)
    return _style_plotly(fig, title, height)


def callout(fig: go.Figure, when, price: float, text: str, color: str) -> None:
    """Small box on one candle. The text is the bot's decision at that time."""
    fig.add_annotation(
        x=when,
        y=float(price),
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowwidth=1,
        arrowcolor=color,
        ax=-78,
        ay=-58,
        xanchor="right",
        yanchor="bottom",
        bgcolor=PAPER,
        bordercolor=color,
        borderwidth=1,
        borderpad=6,
        font=dict(family="IBM Plex Mono, ui-monospace, monospace", size=12, color=INK),
        align="left",
        row=1,
        col=1,
    )


def trade_marks(fig: go.Figure, tape: pd.DataFrame, rows: list) -> None:
    """Buy, sell, and flat marks for paper trades already filled."""
    if not rows:
        return
    lookup = pd.Series(
        tape["close"].to_numpy(dtype=float),
        index=pd.to_datetime(tape["timestamp"], utc=True),
    )
    grouped: dict[str, tuple[list, list, list]] = {
        "buy": ([], [], []),
        "sell": ([], [], []),
        "flat": ([], [], []),
    }
    for item in rows:
        stamp = pd.Timestamp(item["timestamp"])
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        if stamp not in lookup.index:
            continue
        price = lookup.loc[stamp]
        verb = {"buy": "Bought", "sell": "Sold", "flat": "Stood aside"}[item["side"]]
        told = item.get("fed")
        mix = ""
        if told:
            parts = [
                f"{name} {float(weight):.2f}"
                for name, weight in zip(REGIMES, told)
                if float(weight) >= 0.05
            ]
            mix = "<br>" + (" · ".join(parts) if parts else "no regime")
        grouped[item["side"]][0].append(stamp)
        grouped[item["side"]][1].append(float(price.iloc[0] if hasattr(price, "iloc") else price))
        grouped[item["side"]][2].append(f"{verb} {float(item['bet']):+.2f}{mix}")
    style = {
        "buy": ("triangle-up", OK, "Buy"),
        "sell": ("triangle-down", CRISIS, "Sell"),
        "flat": ("circle", MUTED, "Flat"),
    }
    for side, (xs, ys, texts) in grouped.items():
        if not xs:
            continue
        symbol, color, label = style[side]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=label,
                customdata=texts,
                marker=dict(symbol=symbol, color=color, size=13, line=dict(width=1, color=INK)),
                hovertemplate="%{customdata}<br>%{x|%d %b %Y %H:%M}<extra></extra>",
            ),
            row=1,
            col=1,
        )


def equity_figure(df: pd.DataFrame, model: str, mark=None, height: int = 360) -> go.Figure:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if model == "probe":
        cond_name = "Probe policy"
    elif str(model).startswith("plug-in"):
        cond_name = str(model)
    else:
        cond_name = "Regime-conditioned"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=df["equity_conditioned"],
            name=cond_name,
            line=dict(color=INK, width=2.0),
        )
    )
    if "equity_blind" in df.columns and df["equity_blind"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=df["equity_blind"],
                name="Regime-blind",
                line=dict(color=MUTED, width=1.6),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=df["equity_bh"],
            name="Buy and hold",
            line=dict(color=BULL, width=1.6),
        )
    )
    fig.add_hline(y=1, line_color=RULE, line_width=1)
    if mark is not None:
        fig.add_vline(x=mark, line_width=1, line_dash="dot", line_color=CRISIS)
    return _style_plotly(fig, "Out-of-fold wealth", height)


def position_figure(when, bets, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(when), y=list(bets), name="Paper bet", line=dict(color=INK, width=2)))
    fig.add_hline(y=0, line_color=RULE, line_width=1)
    return _style_plotly(fig, "Paper bet", height)


def sensitivity_figure(curve, crisis: float) -> go.Figure:
    xs = np.linspace(0, 1, len(curve))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xs, y=curve, mode="lines", line=dict(color=INK, width=2), name="Position")
    )
    k = int(np.argmin(np.abs(xs - crisis)))
    fig.add_trace(
        go.Scatter(
            x=[xs[k]],
            y=[curve[k]],
            mode="markers",
            marker=dict(size=11, color=CRISIS),
            name="This bar",
        )
    )
    fig.update_xaxes(title="Crisis probability")
    fig.update_yaxes(title="Position")
    return _style_plotly(fig, "Sensitivity curve", 340)


def _new_ax(figsize=(11.4, 6.4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    return fig, ax, _fonts()


def _finish(fig, ax, fonts, title, path: Path):
    ax.set_title(title, fontproperties=fonts["display"], fontsize=18, color=INK, loc="left", pad=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fonts["mono"])
        label.set_color(INK)
    ax.tick_params(colors=INK, length=0)
    for spine in ax.spines.values():
        spine.set_color(RULE)
    fig.text(0.01, 0.01, DISCLAIMER, fontproperties=fonts["body"], fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _right_edge(ts, end: int):
    if end < len(ts):
        return ts.iloc[end]
    return ts.iloc[-1] + (ts.iloc[-1] - ts.iloc[-2])


def save_figures(df: pd.DataFrame, summary: dict, spot: int, curve, out_dir: Path = FIG) -> None:
    fonts = _fonts()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax, strip) = plt.subplots(
        2,
        1,
        figsize=(11.4, 6.4),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [5, 0.45], "hspace": 0.05},
    )
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    strip.set_facecolor(PAPER)
    for start, end, name in _spans(df["regime"]):
        color = REGIME_COLOR[name]
        right = _right_edge(ts, end)
        ax.axvspan(ts.iloc[start], right, color=color, alpha=0.22, lw=0)
        strip.axvspan(ts.iloc[start], right, color=color, lw=0)
    ax.plot(ts, df["close"], color=INK, lw=1.1)
    strip.set_yticks([])
    strip.set_ylim(0, 1)
    for spine in list(ax.spines.values()) + list(strip.spines.values()):
        spine.set_color(RULE)
    ax.set_title(
        f"BTC/USDT · 4h · {summary['model']}",
        fontproperties=fonts["display"],
        fontsize=18,
        color=INK,
        loc="left",
    )
    ax.tick_params(colors=INK, length=0)
    strip.tick_params(colors=INK, length=0)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + list(strip.get_xticklabels()):
        label.set_fontproperties(fonts["mono"])
    handles = [plt.Line2D([0], [0], color=REGIME_COLOR[name], lw=6, label=name) for name in REGIMES]
    ax.legend(handles=handles, frameon=False, prop=fonts["mono"], loc="upper left", ncol=4)
    fig.text(0.01, 0.01, DISCLAIMER, fontproperties=fonts["body"], fontsize=9, color=MUTED)
    fig.savefig(out_dir / "regime_ribbon.png", facecolor=PAPER)
    plt.close(fig)

    fig, ax, fonts = _new_ax()
    ax.plot(ts, df["equity_conditioned"], color=INK, lw=1.8, label="Regime-conditioned" if summary["model"] != "probe" else "Probe policy")
    if "equity_blind" in df.columns and df["equity_blind"].notna().any():
        ax.plot(ts, df["equity_blind"], color=MUTED, lw=1.3, label="Regime-blind")
    ax.plot(ts, df["equity_bh"], color=BULL, lw=1.3, label="Buy and hold")
    ax.axhline(1, color=RULE, lw=0.8)
    ax.legend(frameon=False, prop=fonts["mono"])
    _finish(fig, ax, fonts, "Out-of-fold wealth", out_dir / "equity.png")

    fig, ax, fonts = _new_ax()
    xs = np.linspace(0, 1, len(curve))
    ax.plot(xs, curve, color=INK, lw=1.8)
    crisis = float(df.iloc[spot]["crisis"])
    k = int(np.argmin(np.abs(xs - crisis)))
    ax.scatter([xs[k]], [curve[k]], color=CRISIS, s=40, zorder=3)
    ax.set_xlabel("Crisis probability", fontproperties=fonts["body"], fontsize=11, color=INK)
    ax.set_ylabel("Position", fontproperties=fonts["body"], fontsize=11, color=INK)
    when = ts.iloc[spot].strftime("%d %b %Y %H:%M UTC")
    _finish(fig, ax, fonts, f"Sensitivity · {when}", out_dir / "sensitivity.png")

    row = df.iloc[spot]
    fig, ax, fonts = _new_ax()
    ax.set_axis_off()
    regime = str(row["regime"])
    mix = "   ".join(f"{name} {float(row[name]):.2f}" for name in REGIMES)
    ax.text(0.02, 0.88, "REGIME AUDIT", fontproperties=fonts["mono"], fontsize=11, color=MUTED)
    ax.text(0.02, 0.74, when, fontproperties=fonts["display"], fontsize=22, color=INK)
    ax.text(0.02, 0.62, regime, fontproperties=fonts["display"], fontsize=20, color=REGIME_COLOR[regime])
    ax.text(
        0.02,
        0.42,
        f"{float(row['action_conditioned']):+.2f}",
        fontproperties=fonts["mono"],
        fontsize=48,
        color=INK,
    )
    ax.text(0.02, 0.28, mix, fontproperties=fonts["mono"], fontsize=11, color=MUTED)
    lines = [
        ("Ablation", bool(row["pass_ablation"]), f"removing regime moves position by {float(row['ablation_delta']):.2f}"),
        (
            "Counterfactual",
            bool(row["pass_cf"]),
            "a forced regime moves the position the right way"
            if row["pass_cf"]
            else "a forced regime does not move the position the right way",
        ),
        (
            "Sensitivity",
            bool(row["pass_sens"]),
            "position falls as crisis probability rises"
            if row["pass_sens"]
            else "position does not fall as crisis probability rises",
        ),
    ]
    y = 0.18
    for name, ok, detail in lines:
        ax.text(0.02, y, "Pass" if ok else "Fail", fontproperties=fonts["mono"], fontsize=12, color=OK if ok else CRISIS)
        ax.text(0.16, y, f"{name}  {detail}", fontproperties=fonts["body"], fontsize=12, color=INK)
        y -= 0.055
    fig.text(0.02, 0.04, DISCLAIMER, fontproperties=fonts["body"], fontsize=9, color=MUTED)
    fig.savefig(out_dir / "audit_card.png", facecolor=PAPER)
    plt.close(fig)

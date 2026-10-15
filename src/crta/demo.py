"""One-screen audit desk. Reads saved artifacts. Does not train and does not send orders."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from crta.agent import predict_action
from crta.data import FEATURE_COLS, HMM_COLS, RAW_PATH, ROOT, features, fetch_last_price, fetch_window, refresh_ohlcv
from crta.eval import by_regime, crash_index, strategy_returns
from crta.plug import call, list_bots, load_act, score, step_bar, trade_side
from crta.plots import (
    CRISIS,
    DISCLAIMER,
    FIG,
    INK,
    MUTED,
    OK,
    PAPER,
    REGIME_COLOR,
    RULE,
    callout,
    equity_figure,
    paper_race_figure,
    price_figure,
    trade_marks,
    save_figures,
    sensitivity_figure,
)
from crta.regime import REGIMES, label_span
from crta.verify import ABLATION_MIN_DELTA, CF_MIN_DELTA, audit_bar, one_hot, probe_action, with_crisis_prob

RUN = ROOT / "outputs" / "run"
IST = ZoneInfo("Asia/Kolkata")
TF_NAME = {"15m": "15-minute", "1h": "1-hour", "4h": "4-hour"}

st.set_page_config(page_title="Regime Audit", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,640&family=IBM+Plex+Mono:wght@400;500&family=Literata:opsz,wght@7..72,400;7..72,650&display=swap');
      html, body, .stApp {{ background: {PAPER}; color: {INK}; }}
      header[data-testid="stHeader"] {{ background: transparent; pointer-events: none; }}
      #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
      .block-container {{ padding-top: 1.1rem; max-width: 1440px; }}
      h1, h2, h3, .mast h1 {{
        font-family: Fraunces, "Iowan Old Style", Palatino, Georgia, serif !important;
        font-weight: 560; letter-spacing: -0.03em; color: {INK};
      }}
      p, li, label, .stMarkdown {{
        font-family: Literata, Georgia, serif;
        color: {INK};
      }}
      .mast {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; }}
      .kicker {{
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: {MUTED}; margin: 0 0 0.2rem 0;
      }}
      .mast h1 {{ font-size: 3rem; line-height: 0.95; margin: 0; }}
      .mast.compact h1 {{ font-size: 1.85rem; line-height: 1.05; }}
      .deck {{ margin: 0.35rem 0 0 0; font-size: 1.05rem; }}
      .badge {{
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        border: 1px solid {INK}; padding: 0.45rem 0.7rem; letter-spacing: 0.08em;
        text-transform: uppercase; font-size: 0.75rem;
      }}
      .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.3rem 0 0.4rem; }}
      .metrics > div {{ border: 1px solid {RULE}; border-top: 3px solid {INK}; padding: 0.7rem 0.85rem 0.8rem; }}
      .metrics > div.lead {{ background: #ebe4d6; }}
      .metrics b {{
        display: block; font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        font-size: 1.35rem; font-weight: 500; letter-spacing: -0.03em; margin: 0.25rem 0;
      }}
      .metrics span {{
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED};
      }}
      .metrics em {{ display: block; font-style: normal; font-size: 0.8rem; line-height: 1.3; color: {MUTED}; }}
      .sec {{ border-top: 2px solid {INK}; margin: 2.4rem 0 0.9rem; padding-top: 0.7rem; }}
      .sec h2 {{ font-size: 1.75rem; margin: 0; padding: 0; }}
      .sec p {{ margin: 0.25rem 0 0; color: {MUTED}; font-size: 0.92rem; }}
      .st-key-setup, .st-key-whatif {{ border: 1px solid {RULE}; padding: 1rem 1.1rem 0.6rem; background: #f7f2e9; }}
      .legend {{ font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; letter-spacing: 0.08em;
                 text-transform: uppercase; color: {MUTED}; display: flex; gap: 16px; margin: 0.3rem 0 0.4rem; }}
      .legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: -1px; }}
      .card {{ border: 1px solid {RULE}; border-top: 3px solid {INK}; padding: 0.9rem 1rem 0.6rem; }}
      .stMarkdown p.pos {{
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        font-size: 3.4rem; line-height: 1; margin: 0.3rem 0 0.4rem;
      }}
      .stMarkdown p.pos.small {{ font-size: 2.4rem; }}
      .verdict {{
        font-family: "IBM Plex Mono", monospace; font-size: 0.8rem; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.45rem 0.7rem; margin: 0.8rem 0 0.2rem; color: {PAPER};
      }}
      .mix {{ display: flex; height: 22px; width: 100%; margin: 0.4rem 0 0.8rem; }}
      .mix i {{
        display: block; height: 100%;
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        font-style: normal; font-size: 0.62rem; color: {PAPER};
        padding: 4px 0 0 4px; overflow: hidden; white-space: nowrap;
      }}
      .check {{
        border: 1px solid #e4dccf; border-left: 4px solid {RULE}; padding: 0.6rem 0.75rem; margin: 0.55rem 0 0;
      }}
      .check.pass {{ border-left-color: {OK}; }}
      .check.fail {{ border-left-color: {CRISIS}; }}
      .check header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }}
      .check h3 {{ font-size: 1.05rem; margin: 0; padding: 0; }}
      .check .fact {{ font-family: "IBM Plex Mono", monospace; font-size: 0.82rem; margin: 0.25rem 0 0; }}
      .check p {{ margin: 0.25rem 0 0; font-size: 0.86rem; line-height: 1.35; color: #4a443c; }}
      .ok {{ color: {OK}; font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; letter-spacing: 0.08em; }}
      .bad {{ color: {CRISIS}; font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; letter-spacing: 0.08em; }}
      .note {{ color: {MUTED}; font-size: 0.82rem; }}
      table.reg {{ width: 100%; border-collapse: collapse; margin-top: 0.4rem; }}
      table.reg th {{
        font-family: "IBM Plex Mono", monospace; font-size: 0.68rem; letter-spacing: 0.12em;
        text-transform: uppercase; text-align: left; color: {MUTED}; font-weight: 400;
        padding: 0.35rem 0.4rem; border-bottom: 1px solid {INK};
      }}
      table.reg td {{
        font-family: "IBM Plex Mono", monospace; font-size: 0.85rem;
        padding: 0.4rem; border-bottom: 1px solid #e4dccf;
      }}
      .stButton > button {{
        background: transparent; color: {INK}; border: 1px solid {INK}; border-radius: 0;
        font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      .st-key-nav-audit button[data-testid="stBaseButton-primary"],
      .st-key-nav-sim button[data-testid="stBaseButton-primary"] {{
        background: {INK}; color: {PAPER};
      }}
      .st-key-sim-inputs {{
        border: 1px solid {RULE}; padding: 0.85rem 1rem 0.2rem; background: #f7f2e9;
      }}
      .st-key-sim-go button {{
        background: {INK}; color: {PAPER};
      }}
      .sp {{
        display: grid; grid-template-columns: 6.4rem 2.8rem 1fr; gap: 8px;
        align-items: baseline; border-left: 4px solid {RULE};
        padding: 0.28rem 0.45rem; margin-top: 0.35rem;
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 0.72rem;
      }}
      .sp.pass {{ border-left-color: {OK}; }}
      .sp.fail {{ border-left-color: {CRISIS}; }}
      .sp em {{ font-style: normal; color: {MUTED}; }}
      .band {{ border-top: 2px solid {INK}; margin: 0.8rem 0 0.35rem; padding-top: 0.45rem; }}
      .crta-nav {{
        position: absolute; top: 10px; right: 14px; z-index: 5;
        display: flex; border: 1px solid {INK}; background: {PAPER};
      }}
      .crta-nav button {{
        width: 2rem; height: 2rem; border: 0; border-left: 1px solid {RULE};
        background: transparent; color: {INK}; cursor: pointer;
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 1.05rem; line-height: 1;
      }}
      .crta-nav button:first-child {{ border-left: 0; }}
      .crta-nav button:hover {{ background: #ebe4d6; }}
      .js-plotly-plot .draglayer {{ cursor: grab; }}
      .js-plotly-plot .draglayer:active {{ cursor: grabbing; }}
      [data-testid="stTabs"] [role="tablist"] {{
        gap: 4px; border-bottom: 1px solid {INK};
      }}
      [data-testid="stTabs"] [role="tab"] {{
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        border: 1px solid {INK}; border-bottom: none; border-radius: 8px 8px 0 0;
        padding: 0.35rem 0.8rem; background: transparent; color: {MUTED};
      }}
      [data-testid="stTabs"] [role="tab"] p {{
        font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
        font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; margin: 0;
      }}
      [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: {PAPER}; color: {INK};
      }}
      [data-testid="stTabs"] .react-aria-SelectionIndicator {{ display: none; }}
      [data-testid="stPlotlyChart"] {{ background: transparent; }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.html(
    f"<script>{(Path(__file__).with_name('nav.js')).read_text()}</script>",
    unsafe_allow_javascript=True,
)
PAN = {"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"}
OPEN_BARS = 6 * 365


def _ci(block) -> str:
    if not block or block.get("point") is None or block.get("low") is None:
        return "—"
    return f"{block['point']:.2f}  [{block['low']:.2f}, {block['high']:.2f}]"


def _sharpe(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


@st.cache_data
def _read(mtime: float):
    df = pd.read_parquet(RUN / "bars.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    summary = json.loads((RUN / "summary.json").read_text())
    return df, summary


@st.cache_resource
def _models(mtime: float):
    from crta.agent import load_twin

    loaded = {}
    for path in sorted(RUN.glob("fold*_conditioned.zip")):
        fold = int(path.name.split("_")[0].replace("fold", ""))
        loaded[fold], _meta = load_twin(path)
    return loaded


@st.cache_resource
def _blind_latest(mtime: float):
    from crta.agent import load_twin

    paths = sorted(RUN.glob("fold*_blind.zip"))
    if not paths:
        return None
    model, _meta = load_twin(paths[-1])
    return model


def _ensure():
    if not (RUN / "bars.parquet").exists():
        from crta.pipeline import build

        with st.spinner("Fitting the probe on cached or synthetic bars…"):
            build("probe")


def _refresh_market() -> None:
    """Bring the 4h cache up to the latest closed Binance candle. A failure keeps the saved file."""
    if st.session_state.get("market_checked"):
        return
    try:
        with st.spinner("Refreshing the Binance 4h tape…"):
            refresh_ohlcv()
        st.session_state.market_status = "live"
    except Exception:
        st.session_state.market_status = "stale" if RAW_PATH.exists() else "down"
    st.session_state.market_checked = True


def _policy(row, summary, models):
    if summary["model"] != "trained PPO" or not models:
        return probe_action
    from crta.agent import predict_action

    model = models[int(row["fold"])]
    feats = row[list(FEATURE_COLS)].to_numpy(dtype=float)
    pos = float(row["position_before"])

    def fn(probs, model=model, feats=feats, pos=pos):
        return predict_action(model, feats, probs, pos, True)

    return fn


def _mark(ok: bool) -> str:
    word = "Pass" if ok else "Fail"
    klass = "ok" if ok else "bad"
    return f'<span class="{klass}">{word}</span>'


def _check(title: str, ok: bool, fact: str, text: str) -> str:
    klass = "pass" if ok else "fail"
    return (
        f'<section class="check {klass}"><header><h3>{title}</h3>{_mark(ok)}</header>'
        f'<p class="fact">{fact}</p><p>{text}</p></section>'
    )


def _section(num: int, title: str, lead: str) -> None:
    st.markdown(
        f'<div class="sec"><p class="kicker">{num:02d}</p><h2>{title}</h2><p>{lead}</p></div>',
        unsafe_allow_html=True,
    )


def _facts(result) -> tuple[str, str, str]:
    ablation = f"moved {result['ablation_delta']:.2f} · need ≥ {ABLATION_MIN_DELTA:.2f}"
    lies = []
    if result["forced_crisis"] is not None:
        lies.append(f"crash lie {result['action']:+.2f} → {result['forced_crisis']:+.2f}")
    if result["forced_bull"] is not None:
        lies.append(f"bull lie {result['action']:+.2f} → {result['forced_bull']:+.2f}")
    cf = " · ".join(lies) + f" · need {CF_MIN_DELTA:.2f}"
    n = int(result["violations"])
    sens = f"{n} rise{'s' if n != 1 else ''} as crash certainty grows · 1 allowed"
    return ablation, cf, sens


def _ablation_text(result) -> str:
    delta = result["ablation_delta"]
    need = ABLATION_MIN_DELTA
    if result["ablation"]:
        return (
            f"We hid the market mood and asked for the bet again. "
            f"It moved by {delta:.2f}. A pass needs a move of at least {need:.2f}. "
            f"This bar passes: the mood changed the decision."
        )
    return (
        f"We hid the market mood and asked for the bet again. "
        f"It moved by only {delta:.2f}. Nothing on this check passes. "
        f"A pass needs a move of at least {need:.2f}. "
        f"The agent mostly ignored the mood."
    )


def _cf_text(result) -> str:
    parts = []
    if result["forced_crisis"] is None:
        parts.append(
            "Already a crisis, so there is no worse mood to force. That half is not tested."
        )
    else:
        drop = result["action"] - result["forced_crisis"]
        if drop >= CF_MIN_DELTA:
            parts.append(
                f"Passes the crash lie. Forcing a crisis cut the bet from {result['action']:+.2f} "
                f"to {result['forced_crisis']:+.2f} (down {drop:.2f}; need {CF_MIN_DELTA:.2f})."
            )
        else:
            parts.append(
                f"Fails the crash lie. Forcing a crisis should cut the bet by at least {CF_MIN_DELTA:.2f}. "
                f"It went from {result['action']:+.2f} to {result['forced_crisis']:+.2f}."
            )
    if result["forced_bull"] is None:
        parts.append(
            "Already a bull market, so there is no calmer mood to force. That half is not tested."
        )
    else:
        lift = result["forced_bull"] - result["action"]
        if lift >= CF_MIN_DELTA:
            parts.append(
                f"Passes the bull lie. Pretending this is a bull market raised the bet from "
                f"{result['action']:+.2f} to {result['forced_bull']:+.2f} "
                f"(up {lift:.2f}; need {CF_MIN_DELTA:.2f})."
            )
        else:
            parts.append(
                f"Fails the bull lie. Pretending this is a bull market should raise the bet by at least "
                f"{CF_MIN_DELTA:.2f}. It stayed {result['action']:+.2f} → {result['forced_bull']:+.2f}."
            )
    return " ".join(parts)


def _sens_text(result) -> str:
    n = int(result["violations"])
    word = "time" if n == 1 else "times"
    if result["sensitivity"]:
        return (
            f"We turned “how sure is this a crash?” from 0 to 1. "
            f"The bet got bigger {n} {word}. One rise is allowed. "
            f"This bar passes: the bet does not get more aggressive as the crash looks more certain."
        )
    return (
        f"We turned “how sure is this a crash?” from 0 to 1. "
        f"The bet should get smaller or stay flat. It got bigger {n} {word}. "
        f"One rise would still pass. {n} rises fail. "
        f"On this bar the agent buys harder the more certain the crash is."
    )


def _window(frame, center: int, span: int):
    span = int(max(96, min(len(frame), span)))
    start = int(max(0, min(center - span // 2, len(frame) - span)))
    return frame.iloc[start : start + span]


def _fresh_paper(cursor: int) -> dict:
    return {"cursor": int(cursor), "position": 0.0, "wealth": 1.0, "rows": []}


def _step_for(row, plug, models):
    if plug is not None:

        def step(features, regime, position, plug=plug):
            return call(plug, features, regime, position)

        return step
    if models:
        fold = int(row["fold"]) if "fold" in row.index else max(models)
        model = models.get(fold) or models[max(models)]

        def step(features, regime, position, model=model):
            return predict_action(model, features, regime, position, True)

        return step

    def step(features, regime, position):
        return probe_action(regime)

    return step


FEAT_FIELDS = (
    ("log_ret", "Return", -0.1, 0.1, 0.001, "%.3f"),
    ("vol_20", "Volatility", 0.0, 0.1, 0.001, "%.3f"),
    ("trend", "Trend", -0.2, 0.2, 0.005, "%.3f"),
    ("rsi", "RSI", 0.0, 1.0, 0.01, "%.2f"),
    ("macd", "MACD", -0.02, 0.02, 0.001, "%.3f"),
)


def _mark_custom() -> None:
    st.session_state.sim_custom = True


def _bounds(lo: float, hi: float, value: float, step: float) -> tuple[float, float]:
    lo = min(float(lo), float(value))
    hi = max(float(hi), float(value))
    if hi <= lo:
        hi = lo + step
    return lo, hi


def _snap(value: float, lo: float, hi: float, step: float) -> float:
    snapped = lo + round((float(value) - lo) / step) * step
    return float(min(hi, max(lo, snapped)))


def _mix(raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    total = float(raw.sum())
    if total > 1e-8:
        return raw / total
    return raw


def _inputs(row, paper):
    """Features, regime probabilities, and position the bot will actually see."""
    custom = st.session_state.get("sim_custom") and all(
        f"sim-f-{name}" in st.session_state for name in FEATURE_COLS
    )
    if custom:
        features = np.array([float(st.session_state[f"sim-f-{name}"]) for name in FEATURE_COLS])
        regime = _mix([float(st.session_state[f"sim-r-{name}"]) for name in REGIMES])
    else:
        features = row[list(FEATURE_COLS)].to_numpy(dtype=float)
        regime = _mix(row[list(REGIMES)].to_numpy(dtype=float))
    if st.session_state.get("sim_carry", True):
        position = float(paper["position"])
    else:
        position = float(st.session_state.get("sim-pos", paper["position"]))
    return features, regime, position


def _advance_sim(frame, plug, models) -> None:
    paper = st.session_state.paper
    if paper["cursor"] >= len(frame):
        st.session_state.sim_play = False
        return
    i = int(paper["cursor"])
    row = frame.iloc[i]
    features, regime, position = _inputs(row, paper)
    out = step_bar(
        _step_for(row, plug, models),
        frame,
        i,
        position,
        paper["wealth"],
        regime,
        features,
    )
    audit = {
        key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in out["audit"].items()
    }
    paper["rows"].append(
        {
            "timestamp": row["timestamp"],
            "bet": out["bet"],
            "side": out["side"],
            "wealth": out["wealth"],
            "fed": [float(x) for x in regime],
            "audit": audit,
        }
    )
    paper["position"] = out["position"]
    paper["wealth"] = out["wealth"]
    paper["cursor"] = i + 1
    if paper["cursor"] >= len(frame):
        st.session_state.sim_play = False


def _preview(step, features, regime, position):
    bet = float(np.clip(step(features, regime, float(position)), -1.0, 1.0))
    held = float(position)
    feats = np.asarray(features, dtype=float)
    fed = np.asarray(regime, dtype=float)

    def action_fn(probs, feats=feats, held=held, step=step):
        return float(np.clip(step(feats, np.asarray(probs, dtype=float), held), -1.0, 1.0))

    return bet, trade_side(bet), audit_bar(action_fn, fed)


def _told(regime) -> str:
    parts = [
        f"{name} {float(weight):.2f}"
        for name, weight in zip(REGIMES, regime, strict=True)
        if float(weight) >= 0.05
    ]
    if not parts:
        return "Told no regime"
    return "Told " + " · ".join(parts)


def _date_index(labels) -> int:
    picked = pd.Timestamp(st.session_state.when)
    if picked.tzinfo is None:
        picked = picked.tz_localize("UTC")
    index = int(labels.searchsorted(picked))
    return min(max(index, 0), len(labels) - 1)


def _sync_shown(shown, paper) -> None:
    """Copy this candle into the sliders when the tape moves, or when asked."""
    cursor = int(paper["cursor"])
    reload = bool(st.session_state.pop("sim_reload", False))
    missing = any(f"sim-f-{name}" not in st.session_state for name in FEATURE_COLS)
    moved = st.session_state.get("sim_synced") != cursor
    if not (reload or missing or (moved and not st.session_state.get("sim_custom"))):
        return
    for name, _label, lo, hi, step, _fmt in FEAT_FIELDS:
        bound_lo, bound_hi = _bounds(lo, hi, float(shown[name]), step)
        st.session_state[f"sim-f-{name}"] = _snap(float(shown[name]), bound_lo, bound_hi, step)
    for name in REGIMES:
        st.session_state[f"sim-r-{name}"] = _snap(float(shown[name]), 0.0, 1.0, 0.01)
    st.session_state.sim_synced = cursor


def _held_slider(parent, key: str, label: str, lo: float, hi: float, step: float, fmt: str):
    bound_lo, bound_hi = _bounds(lo, hi, float(st.session_state[key]), step)
    parent.slider(
        label,
        min_value=bound_lo,
        max_value=bound_hi,
        step=step,
        format=fmt,
        key=key,
        on_change=_mark_custom,
    )


def _fmt_ist(value, seconds: bool = False) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    stamp = stamp.tz_convert(IST)
    pattern = "%d %b %Y · %H:%M:%S IST" if seconds else "%d %b %Y · %H:%M IST"
    return stamp.strftime(pattern)


def _now_ist() -> str:
    return datetime.now(IST).strftime("%d %b %Y · %H:%M:%S IST")


def _fmt_axis(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.tz_convert(IST)


def _as_ist_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(IST)
    return out


def _label_window(raw: pd.DataFrame) -> pd.DataFrame:
    framed = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    framed["timestamp"] = pd.to_datetime(framed["timestamp"], utc=True)
    if len(framed) < 40:
        raise RuntimeError("Not enough closed candles yet.")
    burn = min(80, max(20, len(framed) // 5))
    names, probs = label_span(framed[list(HMM_COLS)].to_numpy(), burn, len(framed), prev=None)
    framed = framed.iloc[burn:].reset_index(drop=True)
    for i, name in enumerate(REGIMES):
        framed[name] = probs[:, i]
    framed["regime"] = names
    return framed


def _empty_book() -> dict:
    return {"position": 0.0, "wealth": 1.0, "rows": [], "cursor": 0}


def _roll_books(frame, cond_model, blind_model) -> dict:
    """Settle every candle whose next close is already known. Both bots share that tape."""
    end = len(frame)
    while end > 0 and not _can_step(frame, end - 1):
        end -= 1
    books = {"conditioned": _empty_book(), "blind": _empty_book(), "cursor": 0}
    if cond_model is None or blind_model is None:
        books["cursor"] = end
        return books
    for i in range(end):
        _fill_books(frame, i, books, cond_model, blind_model)
    books["cursor"] = end
    return books


def _fill_books(frame, index: int, books: dict, cond_model, blind_model) -> None:
    row = frame.iloc[int(index)]
    feats = row[list(FEATURE_COLS)].to_numpy(dtype=float)
    regime = row[list(REGIMES)].to_numpy(dtype=float)
    nxt = float(row["next_ret"])
    for name, model, see_regime in (
        ("conditioned", cond_model, True),
        ("blind", blind_model, False),
    ):
        book = books[name]
        bet = float(predict_action(model, feats, regime, float(book["position"]), see_regime))
        ret = float(strategy_returns([bet], [nxt], start_position=float(book["position"]))[0])
        wealth = float(book["wealth"]) * float(np.exp(ret))
        book["rows"].append(
            {
                "timestamp": row["timestamp"],
                "bet": bet,
                "side": trade_side(bet),
                "wealth": wealth,
                "fed": [float(x) for x in regime],
            }
        )
        book["position"] = bet
        book["wealth"] = wealth
    books["cursor"] = int(index) + 1


def _prime_live(timeframe: str, cond_model, blind_model) -> None:
    try:
        raw, stale = fetch_window(timeframe)
        framed = _label_window(raw)
    except Exception:
        st.session_state.paper_frame = None
        st.session_state.live_books = None
        st.session_state.live_tf = timeframe
        if timeframe == "4h" and RAW_PATH.exists():
            st.session_state.paper_live_error = "Binance did not answer. The tape is the saved file."
        else:
            st.session_state.paper_live_error = (
                "Binance is unreachable and there is no saved file for this timeframe."
            )
        return
    st.session_state.paper_frame = framed
    st.session_state.live_tf = timeframe
    st.session_state.live_caught_up = True
    st.session_state.paper_live_error = (
        "Binance did not answer. The tape is the saved file." if stale else ""
    )
    if cond_model is None or blind_model is None:
        st.session_state.live_books = None
        st.session_state.paper_live_error = "The conditioned and blind agents are not both on disk."
        return
    st.session_state.live_books = _roll_books(framed, cond_model, blind_model)
    st.session_state.paper = _fresh_paper(int(st.session_state.live_books["cursor"]))


def _extend_live(frame, timeframe: str):
    """Append newly closed candles. The new labels are fit only on the rows before them."""
    try:
        raw, stale = fetch_window(timeframe)
    except Exception:
        return frame, "Binance is unreachable and there is no saved file for this timeframe."
    framed = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    framed["timestamp"] = pd.to_datetime(framed["timestamp"], utc=True)
    last = pd.Timestamp(frame["timestamp"].iloc[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    fresh = framed.index[framed["timestamp"] > last]
    note = "Binance did not answer. The tape is the saved file." if stale else ""
    if len(fresh) == 0:
        return frame, note
    start = int(fresh[0])
    prev = str(frame["regime"].iloc[-1]) if "regime" in frame.columns and len(frame) else None
    names, probs = label_span(framed[list(HMM_COLS)].to_numpy(), start, len(framed), prev=prev)
    extra = framed.iloc[start:].copy()
    for i, name in enumerate(REGIMES):
        extra[name] = probs[:, i]
    extra["regime"] = names
    merged = pd.concat([frame, extra], ignore_index=True)
    return merged, note


def _settle_open(frame, books: dict, cond_model, blind_model) -> None:
    while _can_step(frame, int(books["cursor"])):
        _fill_books(frame, int(books["cursor"]), books, cond_model, blind_model)


def _pair_preview(row, books, cond_model, blind_model):
    custom = st.session_state.get("sim_custom") and all(
        f"sim-f-{name}" in st.session_state for name in FEATURE_COLS
    )
    if custom:
        feats = np.array([float(st.session_state[f"sim-f-{name}"]) for name in FEATURE_COLS])
        regime = _mix([float(st.session_state[f"sim-r-{name}"]) for name in REGIMES])
    else:
        feats = row[list(FEATURE_COLS)].to_numpy(dtype=float)
        regime = row[list(REGIMES)].to_numpy(dtype=float)
    bet_c = float(predict_action(cond_model, feats, regime, float(books["conditioned"]["position"]), True))
    bet_b = float(predict_action(blind_model, feats, regime, float(books["blind"]["position"]), False))
    return bet_c, bet_b, regime


def _race_card(timeframe: str, books: dict, spot, closed_at: str, bet_c, bet_b, regime) -> str:
    def cell(title: str, book: dict, bet: float) -> str:
        wealth = float(book["wealth"])
        move = wealth - 1.0
        return (
            f"<div><p class='kicker'>{title}</p>"
            f"<p class='pos small'>{wealth:.3f}</p>"
            f"<p class='note'>{move:+.1%} · {len(book['rows'])} fills · open {bet:+.2f} {trade_side(bet)}</p></div>"
        )

    price = f"{float(spot):,.2f}" if spot is not None else "—"
    mix = _mix_html(regime) or '<i style="width:100%;background:#8a8175">no regime</i>'
    name = TF_NAME.get(timeframe, timeframe)
    return (
        f'<div class="card"><p class="kicker">{_now_ist()}</p>'
        f'<p class="pos small">{price}</p>'
        f'<p class="note">Binance last price. Waiting for the next {name} close. '
        f"Latest closed candle {closed_at}. Nothing is sent to an exchange.</p>"
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:0.8rem">'
        f"{cell('Conditioned', books['conditioned'], bet_c)}"
        f"{cell('Blind', books['blind'], bet_b)}"
        f"</div>"
        f'<p class="kicker" style="margin-top:0.8rem">Regime the conditioned bot sees</p>'
        f'<div class="mix">{mix}</div>'
        f'<p class="note">Paper wealth starts at 1. The blind bot trades the same candles and never sees this mix. '
        f"Move the sliders to change only the conditioned bot's open bet.</p></div>"
    )


def _can_step(frame, cursor: int) -> bool:
    """A bar can be settled only once the next candle has closed."""
    if cursor < 0 or cursor >= len(frame):
        return False
    return bool(pd.notna(frame["next_ret"].iloc[int(cursor)]))


def _price_at(tape, stamp) -> float:
    ts = pd.to_datetime(tape["timestamp"], utc=True)
    when = pd.Timestamp(stamp)
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    hit = np.flatnonzero(ts == when)
    row = tape.iloc[int(hit[-1])] if len(hit) else tape.iloc[-1]
    if "high" in row.index and pd.notna(row["high"]):
        return float(row["high"])
    return float(row["close"])


def _append_live(frame) -> str:
    """Add closed 4h bars after the tape. The fit uses only the rows before those bars."""
    stale = False
    try:
        refresh_ohlcv()
    except Exception:
        stale = True
    if not RAW_PATH.exists():
        return "Binance is unreachable and there is no saved file."
    raw = pd.read_parquet(RAW_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    framed = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    framed["timestamp"] = pd.to_datetime(framed["timestamp"], utc=True)
    last = pd.Timestamp(frame["timestamp"].iloc[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    fresh = framed.index[framed["timestamp"] > last]
    if len(fresh) == 0:
        return "Binance did not answer. The tape is the saved file." if stale else ""
    start = int(fresh[0])
    prev = str(frame["regime"].iloc[-1]) if "regime" in frame.columns else None
    try:
        names, probs = label_span(framed[list(HMM_COLS)].to_numpy(), start, len(framed), prev=prev)
    except Exception as exc:
        return str(exc)
    extra = framed.iloc[start:].copy()
    for i, name in enumerate(REGIMES):
        extra[name] = probs[:, i]
    extra["regime"] = names
    if "fold" in frame.columns:
        extra["fold"] = int(frame["fold"].iloc[-1])
    st.session_state.paper_frame = pd.concat([frame, extra], ignore_index=True)
    return "Binance did not answer. The tape is the saved file." if stale else ""


def _mix_html(regime) -> str:
    parts = []
    for name, weight in zip(REGIMES, regime, strict=True):
        share = float(weight)
        if share < 0.05:
            continue
        parts.append(
            f'<i style="width:{100 * share:.1f}%;background:{REGIME_COLOR[name]}">{name} {share:.2f}</i>'
        )
    return "".join(parts)


def _pills(audit) -> str:
    ablation, fairness, sensitivity = _facts(audit)
    rows = (
        ("Ablation", bool(audit["ablation"]), ablation),
        ("Fairness", bool(audit["counterfactual"]), fairness),
        ("Sensitivity", bool(audit["sensitivity"]), sensitivity),
    )
    chunks = []
    for title, ok, fact in rows:
        word = "Pass" if ok else "Fail"
        chunks.append(
            f'<div class="sp {"pass" if ok else "fail"}"><b>{title}</b><span class="{"ok" if ok else "bad"}">{word}</span><em>{fact}</em></div>'
        )
    return "".join(chunks)


def _live_price_line(price: float | None) -> str:
    if price is None:
        return ""
    return f"Last price {price:,.2f}. "


def _draw_live_price(fig, when, price: float) -> None:
    fig.add_hline(y=float(price), line_width=1, line_dash="dot", line_color=OK, row=1, col=1)
    y0, y1 = fig.layout.yaxis.range
    if y0 is not None and y1 is not None:
        pad = (y1 - y0) * 0.04 or 1.0
        fig.update_yaxes(range=[min(y0, float(price) - pad), max(y1, float(price) + pad)], row=1, col=1)
    fig.add_annotation(
        x=when,
        y=float(price),
        text=f"  {price:,.0f} now",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(family="IBM Plex Mono, ui-monospace, monospace", size=12, color=INK),
        row=1,
        col=1,
    )


def _sim_card(
    when: str,
    verb: str,
    bet: float,
    regime,
    audit,
    wealth: float,
    fills: int,
    custom: bool,
    last_price: float | None = None,
    pending: bool = False,
) -> str:
    lead = REGIMES[int(np.argmax(regime))] if float(np.max(regime)) > 0 else None
    tone = REGIME_COLOR[lead] if lead else INK
    source = "Your numbers" if custom else "This candle"
    mix = _mix_html(regime) or '<i style="width:100%;background:#8a8175">no regime</i>'
    return (
        f'<div class="card" style="border-top-color:{tone}">'
        f'<p class="kicker">{when}</p>'
        f'<p class="kicker" style="color:{tone}">{source} · {verb}</p>'
        f'<p class="pos">{bet:+.2f}</p>'
        f'<p class="note">The bet. −1 means the price will fall. +1 means it will rise.</p>'
        f'<p class="kicker">Regime it was told</p>'
        f'<div class="mix">{mix}</div>'
        f"{_pills(audit)}"
        f'<p class="note" style="margin-top:0.7rem">Paper wealth {wealth:.3f}. {fills} filled. '
        f"{'Waiting for the next closed 4h candle. ' if pending else ''}"
        f"{_live_price_line(last_price)}Nothing is sent to an exchange.</p>"
        f"</div>"
    )


def _render_simulate(df, labels, models, plug, bot_options) -> None:
    """Chart, the decision card, then the regime the bot is told."""
    if "sim_clock" not in st.session_state:
        st.session_state.sim_clock = "Live"
    if "sim_tf" not in st.session_state:
        st.session_state.sim_tf = "4h"
    clock = st.session_state.sim_clock
    choice = st.session_state.bot
    timeframe = st.session_state.sim_tf
    sig = (clock, choice, timeframe if clock == "Live" else None)
    if st.session_state.get("sim_sig") != sig:
        st.session_state.sim_sig = sig
        st.session_state.sim_play = False
        st.session_state.sim_custom = False
        st.session_state.sim_reload = True
        st.session_state.paper_frame = None
        st.session_state.live_books = None
        st.session_state.live_tf = None
        st.session_state.live_caught_up = False
        start = len(df) if clock == "Live" else _date_index(labels)
        st.session_state.paper = _fresh_paper(start)
    from crta.agent import load_twin

    cond_model = models[max(models)] if models else None
    blind_paths = sorted(RUN.glob("fold*_blind.zip"))
    blind_model = _blind_latest(blind_paths[-1].stat().st_mtime) if blind_paths else None
    twin_note = ""
    picked = bot_paths.get(choice) if choice != "Saved run" else None
    if picked is not None and picked.suffix == ".zip":
        cond_model, meta = load_twin(picked)
        sibling = picked.with_name(picked.name.replace("conditioned", "blind"))
        if sibling.exists() and sibling != picked:
            blind_model, _blind_meta = load_twin(sibling)
        label = "regularized" if meta.get("faithfulness_regularizer") else "unregularized"
        twin_note = (
            f" Loaded twin metadata: reward {meta.get('reward_mode', 'raw')}, "
            f"probability temperature {meta.get('prob_temperature', 1)}, "
            f"tracker {meta.get('tracker', 'v1')}, {label}."
        )
    if clock == "Live" and (
        st.session_state.get("paper_frame") is None or st.session_state.get("live_tf") != timeframe
    ):
        with st.spinner(f"Loading live Binance {TF_NAME[timeframe]} candles…"):
            _prime_live(timeframe, cond_model, blind_model)

    setup_l, setup_m, setup_r = st.columns([1, 1.15, 1.15])
    setup_l.radio("Clock", ["Replay", "Live"], horizontal=True, key="sim_clock")
    setup_m.radio("Candles", ["15m", "1h", "4h"], horizontal=True, key="sim_tf")
    setup_r.selectbox("Bot", bot_options, key="bot")
    if clock == "Live":
        st.caption(
            f"Now {_now_ist()}. Binance BTC/USDT, paper money only. "
            "The weights were trained on 4-hour bars; 15m and 1h run those same weights on the faster candle. "
            "Replay still walks the saved 4-hour research tape."
            f"{twin_note}"
        )
    pace = timedelta(seconds=20) if clock == "Live" else timedelta(seconds=1) if st.session_state.get("sim_play") else None

    @st.fragment(run_every=pace)
    def _sim_desk():
        now = time.time()
        if clock == "Live" and st.session_state.get("paper_frame") is None:
            st.caption(st.session_state.get("paper_live_error") or "No live tape.")
            return
        live_tape = df if clock == "Replay" else st.session_state.paper_frame
        acted = False
        if st.session_state.pop("sim_do_step", False):
            _advance_sim(live_tape, plug, models)
            acted = True
        run_n = int(st.session_state.pop("sim_do_run", 0) or 0)
        if run_n:
            acted = True
            with st.spinner("Walking the next 2 days…"):
                for _ in range(run_n):
                    if st.session_state.paper["cursor"] >= len(live_tape):
                        break
                    _advance_sim(live_tape, plug, models)
        if clock == "Live":
            if now - float(st.session_state.get("paper_poll", 0.0)) >= 20:
                st.session_state.paper_poll = now
                live_tape, note = _extend_live(st.session_state.paper_frame, timeframe)
                st.session_state.paper_frame = live_tape
                st.session_state.paper_live_error = note
                try:
                    st.session_state.live_price = fetch_last_price()
                except Exception:
                    if not st.session_state.paper_live_error:
                        st.session_state.paper_live_error = "Binance did not answer with a last price."
                books = st.session_state.get("live_books")
                if books is not None and cond_model is not None and blind_model is not None:
                    _settle_open(live_tape, books, cond_model, blind_model)
                    st.session_state.paper["cursor"] = int(books["cursor"])
        elif (
            not acted
            and st.session_state.get("sim_play")
            and now - float(st.session_state.get("paper_tick", 0.0)) >= 0.8
        ):
            st.session_state.paper_tick = now
            _advance_sim(df, plug, models)
            live_tape = df
            if not st.session_state.get("sim_play"):
                st.rerun()
        err = st.session_state.get("paper_live_error") or ""
        if err:
            st.caption(err)
        paper = st.session_state.paper
        deciding = int(paper["cursor"]) < len(live_tape)
        audit = None
        if deciding:
            row = live_tape.iloc[int(paper["cursor"])]
            features, regime, position = _inputs(row, paper)
            bet, side, audit = _preview(_step_for(row, plug, models), features, regime, position)
            verb = {"buy": "Buys", "sell": "Sells", "flat": "Stands aside"}[side]
            when = pd.Timestamp(row["timestamp"]).strftime("%d %b %Y · %H:%M UTC")
            text = f"{verb} {bet:+.2f}<br>{_told(regime)}"
            mark_at = row["timestamp"]
            mark_y = float(row["high"]) if "high" in row.index and pd.notna(row["high"]) else float(row["close"])
            lead = REGIMES[int(np.argmax(regime))] if float(np.max(regime)) > 0 else None
        elif paper["rows"]:
            last = paper["rows"][-1]
            regime = np.asarray(last["fed"], dtype=float)
            bet = float(last["bet"])
            side = last["side"]
            audit = last["audit"]
            verb = {"buy": "Bought", "sell": "Sold", "flat": "Stood aside"}[side]
            when = pd.Timestamp(last["timestamp"]).strftime("%d %b %Y · %H:%M UTC")
            text = f"{verb} {bet:+.2f}<br>{_told(regime)}"
            mark_at = last["timestamp"]
            mark_y = _price_at(live_tape, mark_at)
            lead = REGIMES[int(np.argmax(regime))] if float(np.max(regime)) > 0 else None
        else:
            regime = np.zeros(len(REGIMES))
            bet = 0.0
            verb = "Waiting"
            when = "Next closed 4h candle"
            text = None
            mark_at = None
            mark_y = None
            lead = None
        books = st.session_state.get("live_books") if clock == "Live" else None
        bet_c = bet_b = None
        if books is not None and cond_model is not None and blind_model is not None:
            row = live_tape.iloc[min(int(books["cursor"]), len(live_tape) - 1)]
            bet_c, bet_b, regime = _pair_preview(row, books, cond_model, blind_model)
            text = f"Conditioned {bet_c:+.2f}<br>Blind {bet_b:+.2f}"
            mark_at = row["timestamp"]
            mark_y = float(row["high"]) if "high" in row.index and pd.notna(row["high"]) else float(row["close"])
            lead = REGIMES[int(np.argmax(regime))] if float(np.max(regime)) > 0 else None
            paper["cursor"] = int(books["cursor"])
        head = min(int(paper["cursor"]), len(live_tape) - 1)
        start_i = max(0, head - 119)
        view = live_tape.iloc[start_i : head + 1]
        draw = _as_ist_frame(view) if clock == "Live" else view
        opening = (draw["timestamp"].iloc[0], draw["timestamp"].iloc[-1])
        chart_col, card_col = st.columns([1.55, 0.72], gap="large")
        with chart_col:
            legend = "".join(
                f'<span><i style="background:{REGIME_COLOR[name]}"></i>{name}</span>' for name in REGIMES
            )
            st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
            kind = "candle" if {"open", "high", "low", "close"} <= set(view.columns) else "line"
            axis_at = _fmt_axis(mark_at) if clock == "Live" and mark_at is not None else (pd.Timestamp(mark_at) if mark_at is not None else None)
            fig = price_figure(
                draw,
                "",
                mark=axis_at,
                height=520,
                ribbon=0.16,
                kind=kind,
                xrange=opening,
            )
            marks = books["conditioned"]["rows"][-24:] if books is not None else paper["rows"]
            if clock == "Live":
                marks = [{**item, "timestamp": _fmt_axis(item["timestamp"])} for item in marks]
            trade_marks(fig, draw, marks)
            spot = st.session_state.get("live_price") if clock == "Live" else None
            if spot is not None:
                _draw_live_price(fig, draw["timestamp"].iloc[-1], float(spot))
            if text is not None and mark_at is not None:
                color = REGIME_COLOR[lead] if lead else INK
                y0, y1 = fig.layout.yaxis.range
                span = (y1 - y0) or 1.0
                fig.update_yaxes(range=[y0, y1 + span * 0.18], row=1, col=1)
                callout(fig, axis_at, float(mark_y), text, color)
            fig.update_layout(uirevision=f"sim-{head}", showlegend=False, title_text="", margin=dict(t=12, l=48, r=12, b=28))
            st.plotly_chart(fig, width="stretch", config=PAN, key="sim-price")
            if books is not None and books["conditioned"]["rows"]:
                st.plotly_chart(paper_race_figure(books), width="stretch", config=PAN, key="sim-race")
            live_note = " Times are IST. The dotted line is the Binance price now." if spot is not None else ""
            st.markdown(
                '<p class="note">Drag to pan. Scroll to zoom. Double-click resets. '
                f"Triangles are the conditioned bot's recent paper fills. The box is this candle.{live_note}</p>",
                unsafe_allow_html=True,
            )
        with card_col:
            live_wait = clock == "Live" and (
                audit is None or not _can_step(live_tape, int(paper["cursor"]))
            )
            if books is not None and bet_c is not None:
                closed_at = _fmt_ist(live_tape["timestamp"].iloc[-1])
                st.markdown(
                    _race_card(timeframe, books, spot, closed_at, bet_c, bet_b, regime),
                    unsafe_allow_html=True,
                )
            elif live_wait:
                closed_at = pd.Timestamp(live_tape["timestamp"].iloc[-1]).strftime("%d %b %Y · %H:%M UTC")
                price_html = (
                    f'<p class="pos">{float(spot):,.2f}</p>'
                    if spot is not None
                    else '<p class="note">Last price unavailable.</p>'
                )
                held = ""
                if paper["rows"]:
                    last = paper["rows"][-1]
                    held = (
                        f" Last fill {last['side']} {float(last['bet']):+.2f}"
                        f" at {pd.Timestamp(last['timestamp']).strftime('%d %b %H:%M UTC')}."
                    )
                st.markdown(
                    '<div class="card"><p class="kicker">Live BTC/USDT</p>'
                    f"{price_html}"
                    f'<p class="note">Waiting for the next closed 4h candle. '
                    f"Latest closed bar {closed_at}.{held} "
                    "The next bet is made when that candle closes. "
                    "Nothing is sent to an exchange.</p></div>",
                    unsafe_allow_html=True,
                )
            elif audit is not None:
                st.markdown(
                    _sim_card(
                        when,
                        verb,
                        float(bet),
                        regime,
                        audit,
                        float(paper["wealth"]),
                        len(paper["rows"]),
                        bool(st.session_state.get("sim_custom")),
                        float(spot) if spot is not None else None,
                        pending=clock == "Live" and not _can_step(live_tape, int(paper["cursor"])),
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="card"><p class="kicker">Replay</p>'
                    '<p class="note">The tape has reached the last saved candle.</p></div>',
                    unsafe_allow_html=True,
                )
            at_end = clock == "Live" or int(paper["cursor"]) >= len(live_tape)
            go_label = "Pause" if st.session_state.get("sim_play") else "Play"
            play_col, step_col = st.columns(2)
            if play_col.button(go_label, key="sim-go", disabled=at_end, width="stretch"):
                st.session_state.sim_play = go_label == "Play"
                st.rerun()
            if step_col.button("Step", key="sim-step", disabled=at_end, width="stretch"):
                st.session_state.sim_do_step = True
                st.rerun()
            run_col, restart_col = st.columns(2)
            if run_col.button("Run 2 days", key="sim-run", disabled=at_end, width="stretch"):
                st.session_state.sim_do_run = 12
                st.rerun()
            if restart_col.button("Restart", key="sim-restart", width="stretch"):
                st.session_state.sim_play = False
                st.session_state.sim_custom = False
                st.session_state.sim_reload = True
                st.session_state.paper_frame = None
                st.session_state.live_books = None
                st.session_state.live_tf = None
                st.session_state.live_caught_up = False
                fresh_at = len(df) if clock == "Live" else _date_index(labels)
                st.session_state.paper = _fresh_paper(fresh_at)
                st.rerun()
        shown = live_tape.iloc[head]
        _sync_shown(shown, paper)
        st.markdown(
            '<div class="band"><p class="kicker">Tell the conditioned bot the regime</p>'
            "<p class=\"note\">These four moods are what this project adds. "
            "Move one and only the conditioned bot's open bet changes. The blind bot never sees them. "
            "They are rescaled so they sum to 1.</p></div>",
            unsafe_allow_html=True,
        )
        regime_cols = st.columns(4)
        for column, name in zip(regime_cols, REGIMES, strict=True):
            _held_slider(column, f"sim-r-{name}", name.capitalize(), 0.0, 1.0, 0.01, "%.2f")
        if st.button("Use this candle", key="sim-reset-inputs"):
            st.session_state.sim_custom = False
            st.session_state.sim_reload = True
            st.rerun()
        if st.session_state.get("sim_custom"):
            st.caption("The bot keeps seeing your numbers on each new candle.")
        else:
            st.caption("These match this candle.")
        with st.expander("Other numbers the bot also sees"):
            st.caption("Return, volatility, trend, RSI, and MACD, plus the position already held.")
            for name, label, lo, hi, step, fmt in FEAT_FIELDS:
                _held_slider(st, f"sim-f-{name}", label, lo, hi, step, fmt)
            carry = st.checkbox("Carry the last bet", value=True, key="sim_carry")
            pos_key = "sim-pos"
            if carry or pos_key not in st.session_state:
                st.session_state[pos_key] = _snap(float(paper["position"]), -1.0, 1.0, 0.01)
            st.slider(
                "Position already held",
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                format="%+.2f",
                key=pos_key,
                disabled=carry,
            )

    _sim_desk()


_ensure()
_refresh_market()
mtime = (RUN / "bars.parquet").stat().st_mtime
df, summary = _read(mtime)
models = _models(mtime) if summary["model"] == "trained PPO" else {}
spot = crash_index(df)
labels = pd.to_datetime(df["timestamp"], utc=True)
if "when" not in st.session_state:
    st.session_state.when = labels.iloc[spot].to_pydatetime()
bot_paths = {path.stem: path for path in list_bots()}
bot_options = ["Saved run", *bot_paths]
if st.session_state.get("bot") not in bot_options:
    st.session_state.bot = "Saved run"
choice = st.session_state.bot
guest = st.session_state.get("guest")
if not guest or guest.get("name") != choice:
    guest = None

cond = summary["metrics"]["conditioned"]["sharpe"]
blind = summary["metrics"].get("blind", {}).get("sharpe")
hold = summary["metrics"]["buy_hold"]["sharpe"]
if guest:
    cond = guest["sharpe"]
    blind = None
    badge = f"plug-in · {choice}"
    pass_rate = guest["pass_rate"]
    sharpe_name = choice
else:
    badge = summary["model"]
    pass_rate = summary["pass_rate"]
    sharpe_name = "conditioned"
market_status = st.session_state.get("market_status")
if market_status == "stale":
    source = "Saved Binance file · tape is stale"
elif market_status == "down" and summary["source"] != "binance":
    source = "Synthetic series · Binance unreachable"
elif market_status == "down":
    source = "Binance unreachable · saved run only"
elif summary["source"] == "binance":
    source = "Binance BTC/USDT · 4h"
else:
    source = "Synthetic series · not Binance history"
if "view" not in st.session_state:
    st.session_state.view = "audit"
view = st.session_state.view
plug = None if choice == "Saved run" else load_act(bot_paths[choice])
if view == "simulate":
    title = "Conditioned against blind<br>on this candle"
    deck = (
        f"Paper money on live Binance candles. Clock {_now_ist()}. "
        "The conditioned bot sees the regime. The blind bot does not. Nothing is sent to an exchange."
    )
    if market_status == "stale":
        deck += " Binance did not answer, so this tape is the saved file."
    elif market_status == "down":
        deck += " Binance is unreachable and there is no saved file." if summary["source"] != "binance" else " Binance is unreachable."
else:
    title = "Does the regime<br>move the position?"
    deck = f"{source}. Out-of-fold bars only. {DISCLAIMER}"

head_l, head_r = st.columns([4.6, 1.25], vertical_alignment="bottom")
with head_l:
    st.markdown(
        f"""
        <div class="mast{" compact" if view == "simulate" else ""}">
          <div>
            <p class="kicker">PICT · Electronics &amp; Computer · regime audit</p>
            <h1>{title}</h1>
            <p class="deck">{deck}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown(f'<div class="badge">{badge}</div>', unsafe_allow_html=True)
    nav_audit, nav_sim = st.columns(2)
    if nav_audit.button(
        "Audit",
        key="nav-audit",
        type="primary" if view == "audit" else "secondary",
        width="stretch",
    ):
        st.session_state.view = "audit"
        st.rerun()
    if nav_sim.button(
        "Simulate",
        key="nav-sim",
        type="primary" if view == "simulate" else "secondary",
        width="stretch",
    ):
        st.session_state.view = "simulate"
        st.rerun()

if view == "simulate":
    _render_simulate(df, labels, models, plug, bot_options)
    st.stop()

st.markdown(
    f"""
    <div class="metrics">
      <div class="lead"><span>Faithfulness</span><b>{100 * pass_rate:.1f}%</b>
        <em>Share of bars where all three checks pass.</em></div>
      <div><span>Sharpe · {sharpe_name}</span><b>{_ci(cond)}</b>
        <em>Profit per unit of bumpiness. Higher is smoother.</em></div>
      <div><span>Sharpe · blind</span><b>{_ci(blind) if blind else "—"}</b>
        <em>Same bot trained without the regime.</em></div>
      <div><span>Sharpe · buy &amp; hold</span><b>{_ci(hold)}</b>
        <em>Just holding BTC. The bracket is the likely range.</em></div>
    </div>
    """,
    unsafe_allow_html=True,
)

_section(1, "Setup", "Pick the bot to audit and the moment in history to look at.")
setup = st.container(key="setup")
pick, score_col, crash_col = setup.columns([1.3, 0.9, 0.8], vertical_alignment="bottom")
choice = pick.selectbox("Bot under test", bot_options, key="bot")
guest = st.session_state.get("guest")
if not guest or guest.get("name") != choice:
    guest = None
plug = None if choice == "Saved run" else load_act(bot_paths[choice])
if plug is not None and score_col.button("Score this bot on every bar", width="stretch"):
    with st.spinner("Running the three checks on every saved bar…"):
        try:
            scored = score(plug, df)
        except Exception as exc:
            st.error(str(exc))
            scored = None
    if scored is not None:
        scored["name"] = choice
        scored["by_regime"] = by_regime(scored["returns"], df["regime"])
        st.session_state.guest = scored
        st.rerun()
if crash_col.button("Jump to the 2022 crash", width="stretch"):
    st.session_state.when = labels.iloc[spot].to_pydatetime()
    st.rerun()
if choice == "Saved run":
    setup.caption("The saved run stays loaded. Drop a file in bots/ and pick it here. No training.")
else:
    setup.caption(
        "This file’s act(features, regime, position) is the bot. "
        "The three checks replace regime. Score every bar to replace the header numbers."
    )

picked = setup.slider(
    "Date",
    min_value=labels.iloc[0].to_pydatetime(),
    max_value=labels.iloc[-1].to_pydatetime(),
    step=pd.Timedelta(hours=4).to_pytimedelta(),
    format="DD MMM YYYY  HH:mm",
    key="when",
)
picked_ts = pd.Timestamp(picked)
if picked_ts.tzinfo is None:
    picked_ts = picked_ts.tz_localize("UTC")
index = int(labels.searchsorted(picked_ts))
index = min(max(index, 0), len(df) - 1)
row = df.iloc[index]
probs = row[list(REGIMES)].to_numpy(dtype=float)
if plug is None:
    fn = _policy(row, summary, models)
else:
    feats = row[list(FEATURE_COLS)].to_numpy(dtype=float)
    held = 0.0 if guest is None or index == 0 else float(guest["actions"][index - 1])

    def fn(next_probs, feats=feats, held=held, plug=plug):
        return call(plug, feats, next_probs, held)
result = audit_bar(fn, probs)

shown = df
if guest is not None:
    shown = df.copy()
    shown["equity_conditioned"] = guest["equity"]
    shown = shown.drop(columns=["equity_blind"], errors="ignore")
view = _window(shown, index, OPEN_BARS)
opening = (view["timestamp"].iloc[0], view["timestamp"].iloc[-1])
legend = "".join(
    f'<span><i style="background:{REGIME_COLOR[name]}"></i>{name}</span>' for name in REGIMES
)
nav_hint = (
    '<p class="note">Drag to pan. Scroll or ‹ − + › to zoom and shift. '
    "Double-click goes back to this date.</p>"
)

_section(
    2,
    "This bar",
    "What the bot bet on the chosen date, and whether that bet really listened to the market mood.",
)
chart_col, card_col = st.columns([1.45, 0.85], gap="large")
with chart_col:
    st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
    chart_args = dict(mark=labels.iloc[index], height=600, ribbon=0.24, xrange=opening)
    line_tab, candle_tab = st.tabs(["Line chart", "Candlestick"])
    for tab, kind in ((line_tab, "line"), (candle_tab, "candle")):
        with tab:
            fig = price_figure(shown, "Price and the dwell-smoothed regime", kind=kind, **chart_args)
            fig.update_layout(uirevision=f"price-{index}")
            st.plotly_chart(fig, width="stretch", config=PAN, key=f"price-{kind}")
    st.markdown(nav_hint, unsafe_allow_html=True)
with card_col:
    when = labels.iloc[index].strftime("%d %b %Y · %H:%M UTC")
    mix = "".join(
        f'<i style="width:{100 * float(row[name]):.2f}%;background:{REGIME_COLOR[name]}">{name} {float(row[name]):.2f}</i>'
        for name in REGIMES
        if float(row[name]) >= 0.08
    )
    checks = (result["ablation"], result["counterfactual"], result["sensitivity"])
    passed = sum(bool(ok) for ok in checks)
    tone = OK if passed == 3 else CRISIS
    ablation_fact, cf_fact, sens_fact = _facts(result)
    st.markdown(
        f"""
        <div class="card">
          <p class="kicker">{when}</p>
          <p class="kicker" style="color:{REGIME_COLOR[row['regime']]}">{row['regime']} label · dwell smoothed</p>
          <p class="pos">{result['action']:+.2f}</p>
          <p class="note">The bet, from −1 (price will fall) to +1 (price will rise). It is not the crash probability.</p>
          <p class="kicker">What the detector saw</p>
          <div class="mix">{mix}</div>
          <div class="verdict" style="background:{tone}">{passed} of 3 checks pass · {'faithful' if passed == 3 else 'not faithful'}</div>
          {_check("Ablation", result["ablation"], ablation_fact, _ablation_text(result))}
          {_check("Counterfactual fairness", result["counterfactual"], cf_fact, _cf_text(result))}
          {_check("Sensitivity curve", result["sensitivity"], sens_fact, _sens_text(result))}
          <p class="note" style="margin-top:0.7rem">The bot sees probabilities. The ribbon is the smoothed label. They can disagree on a short run.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

_section(
    3,
    "What if",
    "Lie to the bot about the market mood and watch its bet change. The audit above does not change.",
)
whatif = st.container(key="whatif")
knobs, curve_col = whatif.columns([0.8, 1.2], gap="large")
with knobs:
    force = st.selectbox(
        "Tell the agent the regime is",
        ["As detected", "Bull", "Bear", "Calm", "Crisis"],
        key=f"force-{index}",
    )
    crisis = st.slider(
        "Crisis probability",
        0.0,
        1.0,
        float(row["crisis"]),
        key=f"crisis-{index}",
    )
    if force == "As detected":
        live_probs = with_crisis_prob(probs, crisis)
        note = "Bet at this crash certainty."
        dot = crisis
    else:
        live_probs = one_hot(force.lower())
        note = f"Bet if the regime is forced to {force.lower()}. The slider is paused."
        dot = float(live_probs[REGIMES.index("crisis")])
    live = float(fn(live_probs))
    same_number = abs(dot - 1.0) < 1e-9 and abs(live - 1.0) < 1e-9
    twin = (
        " Both read 1.00, but they are different facts: crash certainty is 100%, and the bet is still fully long."
        if same_number
        else ""
    )
    st.markdown(
        f'<p class="kicker" style="margin-top:0.6rem">Bet under this story</p>'
        f'<p class="pos small">{live:+.2f}</p>'
        f'<p class="note">{note}{twin} Drag the slider left to see the bet at a lower crash certainty.</p>',
        unsafe_allow_html=True,
    )
with curve_col:
    sens = sensitivity_figure(result["curve"], dot)
    sens.update_layout(uirevision=f"sens-{index}")
    st.plotly_chart(sens, width="stretch", config=PAN, key="sensitivity")

_section(
    4,
    "Whole history",
    "How the bot’s money grew over every out-of-fold bar, and how it did in each market mood.",
)
wealth_col, table_col = st.columns([1.6, 0.9], gap="large")
with wealth_col:
    wealth = equity_figure(shown, badge, labels.iloc[index], height=480)
    wealth.update_layout(uirevision=f"wealth-{index}")
    st.plotly_chart(wealth, width="stretch", config=PAN, key="wealth")
    st.markdown(nav_hint, unsafe_allow_html=True)

body = []
if guest is not None:
    headers = ["Regime", "Bars", "This bot"]
    for name in REGIMES:
        block = guest["by_regime"][name]
        body.append(
            "<tr>"
            f'<td style="color:{REGIME_COLOR[name]}">{name}</td>'
            f"<td>{block['n']}</td><td>{_sharpe(block['sharpe'])}</td></tr>"
        )
else:
    headers = ["Regime", "Bars", "Conditioned"]
    if "blind" in summary["by_regime"]["bull"]:
        headers.append("Blind")
    headers.append("Buy and hold")
    for name in REGIMES:
        block = summary["by_regime"][name]
        cells = [
            f'<td style="color:{REGIME_COLOR[name]}">{name}</td>',
            f"<td>{block['n']}</td>",
            f"<td>{_sharpe(block['conditioned']['sharpe'])}</td>",
        ]
        if "blind" in block:
            cells.append(f"<td>{_sharpe(block['blind']['sharpe'])}</td>")
        cells.append(f"<td>{_sharpe(block['buy_hold']['sharpe'])}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
head = "".join(f"<th>{h}</th>" for h in headers)
with table_col:
    st.markdown(
        f"""
        <div class="card">
          <p class="kicker">Sharpe by regime</p>
          <table class="reg"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>
          <p class="note" style="margin-top:0.6rem">Point estimates, without the bootstrap bracket. Positive means the bot made smooth money in that mood.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Export figures", width="stretch"):
        save_figures(df, summary, index, result["curve"], FIG)
        st.caption(f"Wrote four PNGs to {FIG}")

st.markdown(f'<p class="note" style="margin-top:2rem">{DISCLAIMER}</p>', unsafe_allow_html=True)


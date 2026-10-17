"""BTC/USDT cache and causal features. Column t uses rows <= t only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = ("log_ret", "vol_20", "trend", "rsi", "macd")
HMM_COLS = ("log_ret", "vol_20", "trend")
# Optional regime input: distance below the 90-day high (540 4h bars). A rally inside a
# bear market stays far below its high; a bull market sits near it.
DD_WINDOW = 540
HMM_COLS_DD = (*HMM_COLS, "drawdown")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "crta").exists():
            return parent
    return here.parents[2]


ROOT = project_root()
RAW_PATH = ROOT / "data" / "raw" / "btc_usdt_4h.parquet"
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _binance():
    import ccxt

    return ccxt.binance({"enableRateLimit": True})


def _page_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int) -> list[list]:
    rows: list[list] = []
    last_ts = since_ms - 1
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not batch or batch[-1][0] <= last_ts:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        since_ms = last_ts + 1
        if len(batch) < 1000:
            break
    return rows


def _closed_only(rows: list[list], now_ms: int, width_ms: int) -> list[list]:
    return [row for row in rows if int(row[0]) + width_ms <= now_ms]


def _ohlcv_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=OHLCV_COLS)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _read_cache(cache: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def merge_closed(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Keep saved rows, replace any overlap, and append later closed bars."""
    existing = existing.copy()
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    if incoming.empty:
        return existing.sort_values("timestamp").reset_index(drop=True)
    incoming = incoming.copy()
    incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True)
    first = incoming["timestamp"].min()
    kept = existing.loc[existing["timestamp"] < first]
    out = pd.concat([kept, incoming], ignore_index=True)
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _write_cache(df: pd.DataFrame, cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    since: str = "2021-01-01",
    cache: Path = RAW_PATH,
) -> pd.DataFrame:
    """Return the saved file when it exists. A missing file is fetched once from 2021."""
    if cache.exists():
        return _read_cache(cache)

    exchange = _binance()
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    width = int(exchange.parse_timeframe(timeframe) * 1000)
    rows = _closed_only(_page_ohlcv(exchange, symbol, timeframe, since_ms), int(exchange.milliseconds()), width)
    df = _ohlcv_frame(rows)
    if df.empty:
        raise RuntimeError("Binance returned no closed candles")
    _write_cache(df, cache)
    return df


def refresh_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    since: str = "2021-01-01",
    cache: Path = RAW_PATH,
) -> pd.DataFrame:
    """Append closed candles after the last saved bar. A failed fetch leaves the file untouched."""
    if not cache.exists():
        return fetch_ohlcv(symbol, timeframe, since, cache)

    existing = _read_cache(cache)
    exchange = _binance()
    if existing.empty:
        since_ms = exchange.parse8601(f"{since}T00:00:00Z")
        width = int(exchange.parse_timeframe(timeframe) * 1000)
        rows = _closed_only(_page_ohlcv(exchange, symbol, timeframe, since_ms), int(exchange.milliseconds()), width)
        df = _ohlcv_frame(rows)
        if df.empty:
            raise RuntimeError("Binance returned no closed candles")
        _write_cache(df, cache)
        return df

    width = int(exchange.parse_timeframe(timeframe) * 1000)
    since_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
    rows = _closed_only(_page_ohlcv(exchange, symbol, timeframe, since_ms), int(exchange.milliseconds()), width)
    incoming = _ohlcv_frame(rows)
    merged = merge_closed(existing, incoming)
    if not incoming.empty:
        _write_cache(merged, cache)
    return merged


def fetch_closed(symbol: str = "BTC/USDT", timeframe: str = "4h", limit: int = 3) -> pd.DataFrame:
    """Latest Binance candles whose close has already happened. The forming bar is dropped."""
    exchange = _binance()
    rows = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    width = int(exchange.parse_timeframe(timeframe) * 1000)
    closed = _closed_only(rows, int(exchange.milliseconds()), width)
    return _ohlcv_frame(closed)


LIVE_BARS = {"15m": 1000, "1h": 1000, "4h": 500}


def cache_for(timeframe: str) -> Path:
    """4h stays on the training cache. Faster candles keep their own files."""
    if timeframe == "4h":
        return RAW_PATH
    if timeframe not in LIVE_BARS:
        raise ValueError(f"unsupported timeframe {timeframe}")
    return ROOT / "data" / "raw" / f"btc_usdt_{timeframe}.parquet"


def _refresh_window(timeframe: str, bars: int, cache: Path) -> pd.DataFrame:
    exchange = _binance()
    width = int(exchange.parse_timeframe(timeframe) * 1000)
    now = int(exchange.milliseconds())
    if timeframe == "4h":
        if cache.exists():
            return refresh_ohlcv(timeframe=timeframe, cache=cache)
        return fetch_ohlcv(timeframe=timeframe, cache=cache)
    if cache.exists():
        existing = _read_cache(cache)
        since_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
    else:
        existing = None
        since_ms = now - int(bars * width)
    rows = _closed_only(_page_ohlcv(exchange, "BTC/USDT", timeframe, since_ms), now, width)
    incoming = _ohlcv_frame(rows)
    merged = incoming if existing is None else merge_closed(existing, incoming)
    if merged.empty:
        raise RuntimeError("Binance returned no closed candles")
    if len(merged) > bars * 2:
        merged = merged.tail(bars * 2).reset_index(drop=True)
    if existing is None or not incoming.empty:
        _write_cache(merged, cache)
    return merged


def fetch_window(timeframe: str = "4h", bars: int | None = None) -> tuple[pd.DataFrame, bool]:
    """Recent closed candles for the live desk. A failed fetch returns the saved file and marks it stale."""
    if timeframe not in LIVE_BARS:
        raise ValueError(f"unsupported timeframe {timeframe}")
    bars = int(bars or LIVE_BARS[timeframe])
    cache = cache_for(timeframe)
    try:
        merged = _refresh_window(timeframe, bars, cache)
        stale = False
    except Exception:
        if not cache.exists():
            raise
        merged = _read_cache(cache)
        stale = True
    if merged.empty:
        raise RuntimeError("No closed candles")
    return merged.tail(bars).reset_index(drop=True), stale


def fetch_last_price(symbol: str = "BTC/USDT") -> float:
    """Current Binance last price. This is the market now, not a closed 4h candle."""
    ticker = _binance().fetch_ticker(symbol)
    price = ticker.get("last")
    if price is None:
        price = ticker.get("close")
    if price is None:
        raise RuntimeError("Binance ticker had no last price")
    return float(price)


def synthetic_ohlcv(seed: int = 7) -> pd.DataFrame:
    """A 4h path with a 2022-shaped crash, used when Binance is unreachable."""
    rng = np.random.default_rng(seed)
    segments = [
        (0.0014, 0.0045, 2200),
        (-0.0055, 0.028, 1100),
        (0.00015, 0.0035, 2000),
        (0.0009, 0.007, 2800),
        (-0.0012, 0.011, 900),
        (0.0005, 0.006, 3400),
    ]
    rets = np.concatenate([rng.normal(m, s, n) for m, s, n in segments])
    close = 30000.0 * np.exp(np.cumsum(rets))
    ts = pd.date_range("2021-01-01", periods=len(close), freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_ratio(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return (macd - signal) / close


def drawdown(close, window: int = DD_WINDOW) -> np.ndarray:
    """close / (highest close of the last ``window`` bars) - 1. Uses closes up to t only."""
    close = pd.Series(np.asarray(close, dtype=float))
    peak = close.rolling(int(window), min_periods=1).max()
    return (close / peak - 1.0).to_numpy()


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    log_ret = np.log(close).diff()
    out["log_ret"] = log_ret
    out["vol_20"] = log_ret.rolling(20, min_periods=20).std()
    sma = close.rolling(20, min_periods=20).mean()
    out["trend"] = close / sma - 1.0
    out["rsi"] = _rsi(close) / 100.0
    out["macd"] = _macd_ratio(close)
    out["drawdown"] = drawdown(close)
    out["next_ret"] = log_ret.shift(-1)
    return out


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = features(df)
    keep = list(FEATURE_COLS) + ["next_ret"]
    return feat.dropna(subset=keep).reset_index(drop=True)

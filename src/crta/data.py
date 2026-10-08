"""BTC/USDT cache and causal features. Column t uses rows <= t only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = ("log_ret", "vol_20", "trend", "rsi", "macd")
HMM_COLS = ("log_ret", "vol_20", "trend")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "crta").exists():
            return parent
    return here.parents[2]


ROOT = project_root()
RAW_PATH = ROOT / "data" / "raw" / "btc_usdt_4h.parquet"


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    since: str = "2021-01-01",
    cache: Path = RAW_PATH,
) -> pd.DataFrame:
    if cache.exists():
        df = pd.read_parquet(cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    rows: list[list] = []
    last_ts = -1
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not batch or batch[-1][0] <= last_ts:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        since_ms = last_ts + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def fetch_closed(symbol: str = "BTC/USDT", timeframe: str = "4h", limit: int = 3) -> pd.DataFrame:
    """Latest Binance candles whose close has already happened. The forming bar is dropped."""
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    width = int(exchange.parse_timeframe(timeframe) * 1000)
    now = int(exchange.milliseconds())
    closed = [row for row in rows if row[0] + width <= now]
    df = pd.DataFrame(closed, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


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
    out["next_ret"] = log_ret.shift(-1)
    return out


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = features(df)
    keep = list(FEATURE_COLS) + ["next_ret"]
    return feat.dropna(subset=keep).reset_index(drop=True)

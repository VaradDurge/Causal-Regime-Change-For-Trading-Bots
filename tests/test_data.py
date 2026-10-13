import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from crta.data import FEATURE_COLS, RAW_PATH, cache_for, features, fetch_window, merge_closed, refresh_ohlcv


def _frame(n=500, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC"),
        }
    )


def test_merge_closed_keeps_older_rows_and_replaces_overlap():
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T04:00:00"], utc=True),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 9.0],
            "volume": [1.0, 1.0],
        }
    )
    incoming = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T04:00:00", "2024-01-01T08:00:00"], utc=True),
            "open": [2.0, 3.0],
            "high": [2.0, 3.0],
            "low": [2.0, 3.0],
            "close": [2.5, 3.0],
            "volume": [4.0, 5.0],
        }
    )
    merged = merge_closed(existing, incoming)
    assert list(merged["close"]) == [1.0, 2.5, 3.0]
    assert len(merge_closed(existing, incoming.iloc[:0])) == 2


def test_refresh_leaves_the_file_when_binance_fails(tmp_path, monkeypatch):
    cache = tmp_path / "btc_usdt_4h.parquet"
    saved = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    saved.to_parquet(cache, index=False)

    def offline():
        raise RuntimeError("offline")

    monkeypatch.setattr("crta.data._binance", offline)
    try:
        refresh_ohlcv(cache=cache)
    except RuntimeError:
        pass
    else:
        raise AssertionError("refresh should fail when Binance is down")
    kept = pd.read_parquet(cache)
    assert list(kept["close"]) == [1.0]
    assert not cache.with_suffix(".parquet.tmp").exists()


def test_faster_timeframes_do_not_share_the_training_cache():
    assert cache_for("4h") == RAW_PATH
    assert cache_for("15m") != RAW_PATH
    assert cache_for("1h") != cache_for("15m")


def test_fetch_window_keeps_the_file_when_binance_fails(tmp_path, monkeypatch):
    cache = tmp_path / "btc_usdt_15m.parquet"
    saved = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:15:00"], utc=True),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1.0, 1.0],
        }
    )
    saved.to_parquet(cache, index=False)
    monkeypatch.setattr("crta.data.cache_for", lambda _timeframe: cache)

    def offline():
        raise RuntimeError("offline")

    monkeypatch.setattr("crta.data._binance", offline)
    frame, stale = fetch_window("15m", bars=10)
    assert stale
    assert list(frame["close"]) == [1.0, 2.0]
    assert list(pd.read_parquet(cache)["close"]) == [1.0, 2.0]


def test_prefix_matches_full_features():
    df = _frame()
    full = features(df)
    k = 320
    cut = features(df.iloc[:k].copy())
    assert_frame_equal(
        full.iloc[:k][list(FEATURE_COLS)].reset_index(drop=True),
        cut[list(FEATURE_COLS)].reset_index(drop=True),
        check_dtype=False,
    )
    assert_series_equal(
        full["next_ret"].iloc[: k - 1].reset_index(drop=True),
        cut["next_ret"].iloc[:-1].reset_index(drop=True),
        check_dtype=False,
    )

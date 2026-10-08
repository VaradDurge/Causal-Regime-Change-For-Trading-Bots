import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from crta.data import FEATURE_COLS, features


def _frame(n=500, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "close": close,
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC"),
        }
    )


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

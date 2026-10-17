import numpy as np
import pytest
from scipy.stats import norm

from crta.eval import bootstrap_ci, sharpe
from crta.stats import (
    EULER_GAMMA,
    circular_block_indices,
    deflated_sharpe,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe,
    seed_summary,
)


def test_circular_blocks_are_contiguous_runs_that_wrap():
    rng = np.random.default_rng(0)
    idx = circular_block_indices(10, 3, 50, rng)
    assert idx.shape == (50, 10)
    for row in idx:
        for start in range(0, 10, 3):
            run = row[start : start + 3]
            assert all((run[k + 1] - run[k]) % 10 == 1 for k in range(len(run) - 1))


def test_bootstrap_uses_blocks_and_reports_their_length():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 400)
    ci = bootstrap_ci(r, sharpe, n=300, seed=0)
    assert ci["block"] == 20
    assert ci["low"] <= ci["point"] <= ci["high"]
    iid = bootstrap_ci(r, sharpe, n=300, seed=0, block=1)
    assert iid["block"] == 1


def test_expected_max_sharpe_matches_the_closed_form():
    want = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / 100) + EULER_GAMMA * norm.ppf(1 - 1 / (100 * np.e))
    assert expected_max_sharpe(1.0, 100) == pytest.approx(want)
    assert expected_max_sharpe(0.04, 100) == pytest.approx(0.2 * want)
    assert expected_max_sharpe(1.0, 1) == 0.0


def test_probabilistic_sharpe_is_a_half_for_a_symmetric_zero_mean_series():
    r = np.tile([0.01, -0.01], 500)
    assert probabilistic_sharpe(r, 0.0) == pytest.approx(0.5)


def test_more_trials_deflate_the_sharpe_ratio():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0008, 0.01, 2000)
    trials = rng.normal(0.0, 0.02, 30)
    one = deflated_sharpe(r, trials, n_trials=1)
    many = deflated_sharpe(r, trials, n_trials=30)
    assert one["dsr"] == pytest.approx(probabilistic_sharpe(r, 0.0))
    assert many["dsr"] < one["dsr"]
    assert many["sr0"] > 0.0
    assert many["sr"] == pytest.approx(per_period_sharpe(r))


def test_seed_summary_gives_median_and_interquartile_range():
    s = seed_summary([1.0, 2.0, 3.0, 4.0, 5.0, None])
    assert s["n"] == 5
    assert s["median"] == 3.0
    assert s["q25"] == 2.0 and s["q75"] == 4.0
    assert s["iqr"] == 2.0
    assert s["mean"] == 3.0
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))

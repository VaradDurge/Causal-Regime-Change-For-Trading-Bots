import numpy as np
import pytest

from crta.dating import hindsight_phases, label_quality


def _path():
    up = np.linspace(100.0, 150.0, 51)  # peak at index 50
    down = np.linspace(150.0, 100.0, 51)[1:]  # -33%, trough at index 100
    rally = np.linspace(100.0, 130.0, 31)[1:]  # +30%, never confirmed as a peak
    return np.concatenate([up, down, rally])


def test_twenty_percent_moves_date_bull_and_bear_phases():
    phases = hindsight_phases(_path())
    assert set(phases[10:51]) == {"bull"}
    assert set(phases[51:101]) == {"bear"}
    assert set(phases[101:]) == {"bull"}


def test_a_drawdown_short_of_the_threshold_does_not_end_a_bull_phase():
    close = np.concatenate([np.linspace(100, 150, 51), np.linspace(150, 125, 26)[1:], np.linspace(125, 160, 36)[1:]])
    phases = hindsight_phases(close)
    assert set(phases[10:]) == {"bull"}


def test_a_series_that_never_moves_enough_is_undated():
    phases = hindsight_phases(np.linspace(100, 110, 30))
    assert set(phases) == {"undated"}


def test_label_quality_counts_phase_recall():
    labels = ["bull", "calm", "bear", "crisis", "bull"]
    phases = ["bull", "bull", "bear", "bear", "bear"]
    q = label_quality(labels, phases)
    assert q["bull_bars"] == 2
    assert q["bear_bars"] == 3
    assert q["bull_labelled_bull"] == pytest.approx(0.5)
    assert q["bear_labelled_bear_or_crisis"] == pytest.approx(2 / 3)
    assert q["bear_labelled_bull"] == pytest.approx(1 / 3)
    assert q["label_share"]["bull"] == pytest.approx(0.4)
    assert q["confusion"]["bear"]["crisis"] == 1


def test_label_quality_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        label_quality(["bull"], ["bull", "bear"])

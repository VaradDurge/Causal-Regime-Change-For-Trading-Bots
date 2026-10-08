import numpy as np
import pandas as pd

from crta.data import FEATURE_COLS, ROOT
from crta.eval import TURNOVER_COST
from crta.plug import load_act, score, step_bar
from crta.regime import REGIMES
from crta.verify import audit_bar, one_hot, probe_action


def test_example_file_listens_to_regime():
    act = load_act(ROOT / "bots" / "example.py")
    bull = np.array([1.0, 0.0, 0.0, 0.0])
    crisis = np.array([0.0, 0.0, 0.0, 1.0])
    assert act(np.zeros(5), bull, 0.0) > 0
    assert act(np.zeros(5), crisis, 0.0) < 0
    assert audit_bar(lambda probs: act(np.zeros(5), probs, 0.0), bull)["passed"]


def test_a_bot_that_ignores_regime_fails_and_still_scores():
    def act(features, regime, position):
        del features, regime, position
        return 0.5

    bull = np.array([1.0, 0.0, 0.0, 0.0])
    assert not audit_bar(lambda probs: act(None, probs, 0.0), bull)["ablation"]
    n = 40
    frame = pd.DataFrame({name: np.linspace(-0.01, 0.01, n) for name in FEATURE_COLS})
    frame["next_ret"] = 0.001
    for name in REGIMES:
        frame[name] = 0.0
    frame["bull"] = 1.0
    result = score(act, frame)
    assert result["pass_rate"] == 0.0
    assert len(result["equity"]) == n


def _tape(n=2, next_ret=0.01):
    frame = pd.DataFrame({name: np.linspace(-0.01, 0.01, n) for name in FEATURE_COLS})
    frame["next_ret"] = next_ret
    frame["timestamp"] = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    for name in REGIMES:
        frame[name] = 0.0
    frame["bull"] = 1.0
    return frame


def test_paper_fill_earns_the_next_return_minus_turnover():
    def step(features, regime, position):
        del features, regime, position
        return 1.0

    out = step_bar(step, _tape(), 0, 0.0, 1.0)
    assert out["side"] == "buy"
    assert out["wealth"] == np.exp(0.01 - TURNOVER_COST)


def test_crisis_order_is_a_sell_and_the_curve_does_not_rise():
    def step(features, regime, position):
        del features, position
        return probe_action(regime)

    out = step_bar(step, _tape(), 0, 0.0, 1.0, regime=one_hot("crisis"))
    assert out["bet"] == -1.0
    assert out["side"] == "sell"
    assert np.all(np.diff(out["audit"]["curve"]) <= 1e-8)


def test_a_deaf_bot_fails_ablation_on_the_stepped_bar():
    def step(features, regime, position):
        del features, regime, position
        return 0.5

    out = step_bar(step, _tape(), 0, 0.0, 1.0, regime=one_hot("bull"))
    assert out["audit"]["ablation"] is False


def test_a_custom_feature_vector_is_what_the_bot_sees():
    seen = {}

    def step(features, regime, position):
        del regime, position
        seen["features"] = np.asarray(features, dtype=float).copy()
        return 0.0

    custom = np.array([0.2, 0.3, -0.1, 0.4, 0.05])
    step_bar(step, _tape(), 0, 0.0, features=custom)
    assert np.allclose(seen["features"], custom)


def test_stepping_a_bar_ignores_a_row_appended_after_it():
    def step(features, regime, position):
        return probe_action(regime)

    frame = _tape()
    first = step_bar(step, frame, 0, 0.0, 1.0)
    extra = frame.iloc[[1]].copy()
    extra["log_ret"] = 5.0
    extra["next_ret"] = 9.0
    extra["timestamp"] = frame["timestamp"].iloc[-1] + pd.Timedelta(hours=4)
    longer = pd.concat([frame, extra], ignore_index=True)
    second = step_bar(step, longer, 0, 0.0, 1.0)
    assert first["bet"] == second["bet"]
    assert first["wealth"] == second["wealth"]
    assert first["audit"]["action"] == second["audit"]["action"]

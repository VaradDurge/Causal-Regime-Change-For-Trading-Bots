import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crta.agent import load_twin, save_twin, train_ppo
from crta.data import FEATURE_COLS
from crta.regime import REGIMES
from crta.select import choose_risk_aversion, criterion_deficit, meets_criterion


def _frame(n=80):
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(0.0, 0.01, size=(n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
    for name in REGIMES:
        df[name] = 0.25
    df["bull"] = 0.7
    df["next_ret"] = rng.normal(0.0, 0.01, n)
    df["close"] = 100.0
    df["timestamp"] = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return df


def test_metadata_round_trip(tmp_path):
    frame = _frame()
    model = train_ppo(
        frame,
        True,
        timesteps=128,
        window=32,
        reward_mode="excess_bh",
        prob_temperature=1.5,
        prob_floor=0.05,
        tracker="v2",
        threads=1,
        collapse_log_every=64,
    )
    path = save_twin(model, tmp_path / "btc_1h_conditioned.zip")
    loaded, meta = load_twin(path)
    assert meta["reward_mode"] == "excess_bh"
    assert meta["prob_temperature"] == 1.5
    assert meta["prob_floor"] == 0.05
    assert meta["tracker"] == "v2"
    assert path.with_suffix(".meta.json").exists()
    assert loaded.crta_meta["reward_mode"] == "excess_bh"
    assert model.collapse_log


def test_runner_quick(tmp_path):
    script = Path("paper/scripts/run_experiment.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--quick", "--seeds", "42", "--out", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads((tmp_path / "seed_42" / "eval.json").read_text())
    seg = payload["segments"]["validation"]
    for side in ("conditioned", "blind"):
        assert "position_std" in seg[side]
        assert "extreme_share" in seg[side]
        assert "sharpe" in seg[side]
    assert "ablation_pass_rate" in seg
    assert "faithfulness_rate" in seg
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_seeds"] == 1
    assert (tmp_path / "summary.csv").exists()


def test_config_risk_aversion_reaches_trade_env(tmp_path):
    import importlib.util

    from crta.config import ExperimentConfig

    spec = importlib.util.spec_from_file_location("run_experiment", Path("paper/scripts/run_experiment.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = ExperimentConfig.load("configs/final.yaml").with_overrides(reward_mode="mean_variance", risk_aversion=5.0)
    kwargs = runner._train_kwargs(cfg, 42, True, "1h")
    assert kwargs["reward_mode"] == "mean_variance"
    assert kwargs["risk_aversion"] == 5.0
    kwargs.update(timesteps=64, collapse_log_every=64)
    model = train_ppo(_frame(), True, **kwargs)
    assert model.get_env().get_attr("reward_mode") == ["mean_variance"]
    assert model.get_env().get_attr("risk_aversion") == [5.0]
    _, meta = load_twin(save_twin(model, tmp_path / "mv.zip"))
    assert meta["reward_mode"] == "mean_variance"
    assert meta["risk_aversion"] == 5.0
    assert runner._train_kwargs(ExperimentConfig.load("configs/final.yaml"), 42, True, "1h")["risk_aversion"] == 0.0


def test_selection_criterion_ignores_returns():
    assert meets_criterion(0.2, 0.1, 0.5, 0.4, 0.2)
    assert not meets_criterion(0.2, 0.01, 0.5, 0.4, 0.2)
    assert not meets_criterion(0.2, 0.1, 0.9, 0.4, 0.2)
    assert not meets_criterion(0.2, 0.1, 0.5, 0.4, 0.05)
    assert criterion_deficit(0.2, 0.1, 0.5, 0.4, 0.2) == 0.0
    assert criterion_deficit(0.0, 0.0, 1.0, 1.0, 0.0) > 0.0


def test_risk_aversion_choice_takes_min_ablation_then_smaller_lambda():
    def tape(lam, ab, std=0.2, ext=0.3):
        return {"risk_aversion": lam, "std_c": std, "std_b": std, "extreme_c": ext, "extreme_b": ext, "ablation": ab}

    rows = [tape(2.0, 0.3), tape(2.0, 0.2), tape(5.0, 0.2), tape(5.0, 0.4), tape(10.0, 0.5), tape(10.0, 0.05)]
    choice = choose_risk_aversion(rows)
    assert choice["risk_aversion"] == 2.0
    assert [r["risk_aversion"] for r in choice["rejected"]] == [10.0]
    assert choose_risk_aversion([tape(2.0, 0.5, ext=0.9)])["risk_aversion"] == "none"

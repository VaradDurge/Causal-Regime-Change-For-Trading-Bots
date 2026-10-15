"""Load any frozen bot and run the same three checks. Does not train."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from crta.agent import observation
from crta.data import FEATURE_COLS, ROOT
from crta.eval import performance, strategy_returns, wealth_curve
from crta.regime import REGIMES
from crta.verify import audit_bar, audit_many

# A bet inside this band is flat. Outside it the paper book is long or short.
FLAT = 0.05

BOTS = ROOT / "bots"


def list_bots() -> list[Path]:
    if not BOTS.exists():
        return []
    py = [p for p in BOTS.glob("*.py") if not p.name.startswith("_")]
    zips = list(BOTS.glob("*.zip"))
    return sorted(py + zips)


def _clip(value) -> float:
    number = float(np.asarray(value, dtype=float).reshape(-1)[0])
    if not np.isfinite(number):
        raise ValueError("bot returned a non-finite bet")
    return float(np.clip(number, -1.0, 1.0))


def _from_zip(path: Path):
    from crta.agent import load_twin, smooth_probs

    model, meta = load_twin(path)
    dim = int(np.prod(model.observation_space.shape))
    full = len(FEATURE_COLS) + 1 + len(REGIMES)
    blind = len(FEATURE_COLS) + 1
    if dim not in (full, blind):
        raise ValueError(
            f"{path.name} expects {dim} inputs. This desk feeds {full} "
            f"(5 features, position, 4 regime probabilities) or {blind} without the regime. "
            "Wrap the model in bots/<name>.py act() and build its own vector there."
        )
    include = dim == full
    temperature = float(meta.get("prob_temperature", 1.0))
    floor = float(meta.get("prob_floor", 0.0))

    def act(features, regime, position, model=model, include=include, temperature=temperature, floor=floor):
        fed = smooth_probs(regime, temperature, floor) if include else regime
        obs = observation(features, fed, position, include)
        action, _ = model.predict(obs, deterministic=True)
        return _clip(action)

    act.meta = meta

    def batch(feats, regimes, positions, model=model, include=include, temperature=temperature, floor=floor):
        fed = smooth_probs(regimes, temperature, floor) if include else regimes
        obs = np.vstack(
            [
                observation(feats[i], fed[i], float(positions[i]), include)
                for i in range(len(fed))
            ]
        )
        action, _ = model.predict(obs, deterministic=True)
        out = np.asarray(action, dtype=float).reshape(-1)
        if not np.isfinite(out).all():
            raise ValueError(f"{path.name} returned a non-finite bet")
        return np.clip(out, -1.0, 1.0)

    act.batch = batch  # ponytail: attribute, not a class. Zip models are the only batch path.
    return act


def call(act, features, regime, position) -> float:
    return _clip(act(features, regime, position))


def trade_side(bet: float) -> str:
    if bet > FLAT:
        return "buy"
    if bet < -FLAT:
        return "sell"
    return "flat"


def step_bar(step, frame, index, position, wealth=1.0, regime=None, features=None) -> dict:
    """One paper fill. The bot sees this row only; the checks then replace the regime it was fed."""
    row = frame.iloc[int(index)]
    if features is None:
        features = row[list(FEATURE_COLS)].to_numpy(dtype=float)
    else:
        features = np.asarray(features, dtype=float)
    fed = row[list(REGIMES)].to_numpy(dtype=float) if regime is None else np.asarray(regime, dtype=float)
    bet = _clip(step(features, fed, float(position)))
    ret = float(strategy_returns([bet], [float(row["next_ret"])], start_position=float(position))[0])
    held = float(position)

    def action_fn(probs, features=features, held=held, step=step):
        return _clip(step(features, np.asarray(probs, dtype=float), held))

    return {
        "bet": bet,
        "side": trade_side(bet),
        "position": bet,
        "ret": ret,
        "wealth": float(wealth) * float(np.exp(ret)),
        "audit": audit_bar(action_fn, fed),
    }


def load_act(path: Path):
    path = Path(path)
    if path.suffix == ".zip":
        return _from_zip(path)
    spec = importlib.util.spec_from_file_location(f"crta_bot_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    act = getattr(module, "act", None)
    if not callable(act):
        raise ValueError(f"{path.name} needs act(features, regime, position)")
    return act


def score(act, frame: pd.DataFrame) -> dict:
    """Walk the saved bars. The regime column is what the bot saw; the checks then replace it."""
    feats = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    regime = frame[list(REGIMES)].to_numpy(dtype=float)
    n = len(frame)
    actions = np.zeros(n)
    position = 0.0
    batch = getattr(act, "batch", None)
    for i in range(n):
        actions[i] = call(act, feats[i], regime[i], position)
        position = actions[i]
    prev = np.concatenate([[0.0], actions[:-1]])

    def predict(packed, feats=feats, prev=prev, n=n):
        if batch is not None:
            idx = np.arange(len(packed)) % n
            return batch(feats[idx], packed, prev[idx])
        return np.array(
            [_clip(act(feats[i % n], packed[i], float(prev[i % n]))) for i in range(len(packed))]
        )

    audited = audit_many(predict, regime)
    returns = strategy_returns(actions, frame["next_ret"].to_numpy(dtype=float))
    return {
        "actions": actions,
        "passed": audited["passed"],
        "pass_rate": float(np.mean(audited["passed"])),
        "sharpe": performance(returns)["sharpe"],
        "equity": wealth_curve(returns),
        "returns": returns,
    }


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score a frozen bot on the saved bars.")
    parser.add_argument("bot", type=Path, help="bots/name.py with act(), or a .zip with our observation")
    args = parser.parse_args(argv)
    bars = ROOT / "outputs" / "run" / "bars.parquet"
    if not bars.exists():
        raise SystemExit("no saved bars. Run: uv run python -m crta.pipeline --probe")
    frame = pd.read_parquet(bars)
    act = load_act(args.bot)
    result = score(act, frame)
    sharpe = result["sharpe"]
    print(
        json.dumps(
            {
                "bot": args.bot.name,
                "pass_rate": result["pass_rate"],
                "sharpe": sharpe["point"],
                "sharpe_low": sharpe["low"],
                "sharpe_high": sharpe["high"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

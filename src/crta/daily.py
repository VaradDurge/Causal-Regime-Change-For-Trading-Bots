"""Daily long/flat task: one decision at each daily close, paid the next day's return.

The decision bar is the 4h candle that opens at 20:00 UTC and closes the day. Features
are rebuilt on daily closes and joined by 2, 4 and 8-week momentum. The regime block is
the 4h labeller's filtered probability at the decision bar, so it uses candles up to
that close only. The reward is a * r - c |a - a_prev| - lambda a^2 sigma^2 with sigma the
20-day volatility known at the decision, which has an interior optimum
a* = E[r] / (2 lambda sigma^2) whenever the edge is small next to the risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from crta.data import FEATURE_COLS, features
from crta.eval import TURNOVER_COST
from crta.policies import HEADS
from crta.predictability import momentum_features
from crta.regime import REGIMES

DECISION_HOUR = 20
DAYS_PER_WEEK = 7
MOMENTUM_WEEKS = (2, 4, 8)
DAILY_FEATURES = (*FEATURE_COLS, *(f"mom_{w}w" for w in MOMENTUM_WEEKS))
PERIODS_DAILY = 365
OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


def daily_frame(frame4h: pd.DataFrame, probs4h) -> pd.DataFrame:
    """One row per daily close with daily features, momentum, next-day return and the regime block."""
    frame4h = frame4h.reset_index(drop=True)
    stamps = pd.to_datetime(frame4h["timestamp"], utc=True)
    rows = np.flatnonzero(stamps.dt.hour.to_numpy() == DECISION_HOUR)
    raw = frame4h.loc[rows, OHLCV].reset_index(drop=True)
    out = features(raw)
    momentum = momentum_features(out["close"], DAYS_PER_WEEK, MOMENTUM_WEEKS)
    for col in momentum.columns:
        out[col] = momentum[col].to_numpy()
    out["row4h"] = rows
    probs = np.asarray(probs4h, dtype=float)[rows]
    for i, name in enumerate(REGIMES):
        out[name] = probs[:, i]
    return out.dropna(subset=[*DAILY_FEATURES, "next_ret"]).reset_index(drop=True)


def daily_bounds(daily_ts, ts4h, bounds4h) -> list[tuple[int, int]]:
    """Daily walk-forward folds that cut on the same calendar instants as the 4h folds."""
    days = pd.to_datetime(pd.Series(daily_ts), utc=True).to_numpy()
    bars = pd.to_datetime(pd.Series(ts4h), utc=True).to_numpy()
    out = []
    for k, (start4h, end4h) in enumerate(bounds4h):
        start = int(np.searchsorted(days, bars[start4h], side="left"))
        end = len(days) if k == len(bounds4h) - 1 else int(np.searchsorted(days, bars[end4h], side="left"))
        out.append((start, end))
    return out


def execute_position(head: str, action) -> float:
    """Executed position in [0, 1] for a raw action from the policy."""
    value = float(np.asarray(action, dtype=float).reshape(-1)[0])
    if head in ("clip01", "beta"):
        return float(np.clip(value, 0.0, 1.0))
    if head == "tanh01":
        return float((np.tanh(value) + 1.0) / 2.0)
    raise ValueError(f"unknown head {head}, expected one of {HEADS}")


@dataclass(frozen=True)
class Scaler:
    """Feature standardisation fitted on a training fold."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X) -> "Scaler":
        X = np.asarray(X, dtype=float)
        sd = X.std(axis=0)
        return cls(mean=X.mean(axis=0), std=np.where(sd < 1e-12, 1.0, sd))

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.std


class PositionEnv(gym.Env):
    """Walks the frame once from a given start position. Observation: features, position, regime block."""

    metadata: dict = {}

    def __init__(
        self,
        frame: pd.DataFrame,
        include_regime: bool,
        head: str,
        risk_aversion: float = 0.0,
        reward_scale: float = 100.0,
        scaler: Scaler | None = None,
        cost: float = TURNOVER_COST,
    ):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head {head}, expected one of {HEADS}")
        self.head = head
        self.include_regime = bool(include_regime)
        self.risk_aversion = float(risk_aversion)
        self.reward_scale = float(reward_scale)
        self.cost = float(cost)
        feats = frame[list(DAILY_FEATURES)].to_numpy(dtype=float)
        self.scaler = scaler or Scaler.fit(feats)
        self._scaled = self.scaler.transform(feats)
        self._next = frame["next_ret"].to_numpy(dtype=float)
        self._vol = frame["vol_20"].to_numpy(dtype=float)
        self._probs = frame[list(REGIMES)].to_numpy(dtype=float) if self.include_regime else None
        n_obs = len(DAILY_FEATURES) + 1 + (len(REGIMES) if self.include_regime else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        if head == "tanh01":
            # Wide enough that Stable-Baselines3's clip does not bind before tanh.
            self.action_space = spaces.Box(-1.0e3, 1.0e3, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self._i = 0
        self.position = 0.0

    def _obs(self) -> np.ndarray:
        parts = [self._scaled[self._i], [self.position]]
        if self._probs is not None:
            parts.append(self._probs[self._i])
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self.position = float((options or {}).get("position", 0.0))
        return self._obs(), {}

    def step(self, action):
        target = execute_position(self.head, action)
        nxt = float(self._next[self._i])
        sigma = float(self._vol[self._i])
        reward = target * nxt - self.cost * abs(target - self.position) - self.risk_aversion * target * target * sigma * sigma
        self.position = target
        self._i += 1
        done = self._i >= len(self._next)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if done else self._obs()
        return obs, float(reward * self.reward_scale), done, False, {}


def train_daily(
    frame: pd.DataFrame,
    include_regime: bool,
    head: str,
    risk_aversion: float,
    seed: int,
    timesteps: int,
    scaler: Scaler,
    monotone: bool = False,
    regime_rank: list[int] | None = None,
    reward_scale: float = 100.0,
    ent_coef: float = 0.0,
    device: str = "cpu",
    threads: int = 1,
):
    """PPO with the study's 4h settings (lr 3e-4, gamma 0.99, GAE 0.95, 256-step rollouts, batch 64)."""
    import torch
    from stable_baselines3 import PPO

    from crta.policies import PositionPolicy

    torch.set_num_threads(int(threads))
    env = PositionEnv(frame, include_regime, head, risk_aversion, reward_scale, scaler)
    n_steps = 256 if len(frame) >= 256 else max(32, len(frame) // 2)
    model = PPO(
        PositionPolicy,
        env,
        seed=seed,
        device=device,
        verbose=0,
        n_steps=n_steps,
        batch_size=min(64, n_steps),
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        policy_kwargs={
            "head": head,
            "n_regime": len(REGIMES) if include_regime else 0,
            "monotone": bool(monotone),
            "regime_rank": regime_rank,
        },
    )
    model.learn(total_timesteps=int(timesteps), progress_bar=False)
    return model


def run_positions(model, frame: pd.DataFrame, include_regime: bool, head: str, scaler: Scaler, start: float = 0.0) -> np.ndarray:
    """Deterministic positions bar by bar; each decision sees the previous executed position."""
    env = PositionEnv(frame, include_regime, head, reward_scale=1.0, scaler=scaler)
    obs, _ = env.reset(options={"position": start})
    out = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, _t, _i = env.step(action)
        out.append(env.position)
        if done:
            return np.asarray(out, dtype=float)


def audit_predictor(model, frame: pd.DataFrame, before: np.ndarray, scaler: Scaler):
    """predict(prob rows) -> positions, holding each bar's features and previous position fixed."""
    import torch as th

    scaled = scaler.transform(frame[list(DAILY_FEATURES)].to_numpy(dtype=float))
    before = np.asarray(before, dtype=float)
    n = len(frame)

    def predict(batch) -> np.ndarray:
        batch = np.asarray(batch, dtype=float)
        idx = np.arange(len(batch)) % n
        obs = np.column_stack([scaled[idx], before[idx], batch]).astype(np.float32)
        with th.no_grad():
            return model.policy.position(th.as_tensor(obs, device=model.device)).cpu().numpy().astype(float)

    return predict

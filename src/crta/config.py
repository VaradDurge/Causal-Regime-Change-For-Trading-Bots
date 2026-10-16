"""Experiment flags. Code defaults reproduce the previous trainer; YAML selects a recipe.

``ExperimentConfig()`` matches ``train_ppo`` as it stood before the v2 flags:
raw reward, scale 1, no window, entropy 0. ``configs/default.yaml`` is the
timeframe-native recipe (window 2000, scale 100, entropy 0.01, 500k steps)
with every new flag left at the old behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path


def _parse_scalar(text: str):
    value = text.strip()
    if value == "" or value == "null" or value == "~":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if any(c in value for c in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_yaml(path: str | Path) -> dict:
    """Flat YAML (scalars, null, booleans, inline lists). Enough for our configs."""
    data: dict = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("---"):
            continue
        if ":" not in line:
            raise ValueError(f"cannot parse {raw!r} in {path}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def dump_yaml(path: str | Path, data: dict) -> None:
    lines = []
    for key, value in data.items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        elif isinstance(value, float):
            rendered = repr(value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    Path(path).write_text("\n".join(lines) + "\n")


@dataclass
class ExperimentConfig:
    """One recipe. Every new field defaults to the behavior before that change."""

    reward_mode: str = "raw"
    reward_scale: float = 1.0
    diff_sharpe_eta: float = 0.01
    pos_extreme_penalty: float = 0.0
    pos_variance_bonus: float = 0.0
    variance_every: int = 256
    ent_coef: float = 0.0
    ent_coef_start: float | None = None
    collapse_log_every: int = 10_000
    window: int | None = None
    window_sampling: str = "uniform"
    regime_balance_beta: float = 3.0
    sign_flip_prob: float = 0.0
    regime_perturb_prob: float = 0.0
    faithfulness_regularizer: bool = False
    faithfulness_reg_coef: float = 0.1
    prob_temperature: float = 1.0
    prob_floor: float = 0.0
    tracker: str = "v1"
    tracker_matching: str = "nearest"
    tracker_death_after: int = 20
    tracker_cost_quantile: float = 0.95
    verifier: str = "legacy"
    delta_ab: float = 0.15
    delta_cf: float = 0.10
    delta_sens: float = 0.05
    cf_pick: str = "nearest"
    timesteps: int = 8192
    threads: int | None = None
    seed: int = 42
    protocol: str = "timeframe"
    asset: str = "BTC"
    timeframe: str = "1h"
    stage: str = "validate"
    action_mode: str = "gaussian_clip"
    risk_aversion: float = 0.0
    faith_bound_aware: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise KeyError(f"unknown config keys: {unknown}")
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        dump_yaml(path, self.to_dict())

    def with_overrides(self, **kwargs) -> "ExperimentConfig":
        data = self.to_dict()
        data.update(kwargs)
        return self.from_dict(data)

"""Five-seed rerun of the four-hour walk-forward study.

Uses the frozen candle prefix already in the repo (the live snapshot's rows
through 2026-09-22 12:00 UTC, which match tracker.npz). Does not download.

Labellings: tracker v1, tracker v2 (nearest match, birth after 20 empty refits),
and moment sorting on the same v1 refits. Verifiers: legacy and v2.
Budget stays max(8192, 10 * n_train), raw reward, scale 1, entropy 0, no window.

Bootstrap intervals are a circular block bootstrap with block length
ceil(sqrt(n)) and 1000 resamples, the four-hour code's resample count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.agent import batch_observation, rollout, save_twin, train_ppo  # noqa: E402
from crta.data import FEATURE_COLS, HMM_COLS, model_frame  # noqa: E402
from crta.eval import (  # noqa: E402
    PERIODS,
    max_drawdown,
    sharpe,
    sortino,
    strategy_returns,
    walk_forward_bounds,
)
from crta.regime import (  # noqa: E402
    ETA,
    LEGACY_V1,
    LEGACY_V2,
    REGIMES,
    STRIDE,
    _fit_hmm,
    _templates_from_lock,
    _window_labels,
    assign_templates,
    component_params,
    lock_states,
    pairwise_w2,
    track,
    track_v2,
    update_templates,
)
from crta.report import label_statistics  # noqa: E402
from crta.verify import audit_many, audit_many_v2, empirical_banks, verifier_report  # noqa: E402

# Reproduces the published five-seed rerun, smoothed v1 probabilities included.
# New results come from causal_4h.py.
BURN = LEGACY_V1["burn"]
OUT = ROOT / "paper" / "results" / "4h_rerun"
TAB = ROOT / "paper" / "results" / "tables_4h"
TAB_COPY = ROOT / "results" / "tables"
LIVE_PARQUET = ROOT / "paper" / "results" / "live" / "btc_usdt_4h.parquet"
TRACKER_NPZ = ROOT / "paper" / "results" / "tracker.npz"
ORIGINAL_BARS = ROOT / "paper" / "results" / "run" / "bars.parquet"
STUDY_END = pd.Timestamp("2026-09-22 12:00", tz="UTC")
SEEDS = (42, 43, 44, 45, 46)
LABELLINGS = ("tracker_v1", "tracker_v2", "moment_sorted")
VERIFIERS = ("legacy", "v2")
N_BOOT = 1000
BOOT_SEED = 42
THREADS = 1

NOTES = """\
Four-hour rerun notes
=====================

Candle. data/raw/btc_usdt_4h.parquet is gitignored and was not in the checkout.
paper/results/live/btc_usdt_4h.parquet was built from that cache plus a later tail.
This rerun keeps rows with timestamp <= 2026-09-22 12:00 UTC (12,544 bars, the
study end in the paper) and checks closes against paper/results/tracker.npz.
No candle was downloaded. The live file's 12:00 close does not reproduce the
archived run's last next_ret, so that one close is restored from
paper/results/run/bars.parquet before features are built. Every other out-of-fold
close and next_ret already matched.

Moment-sorted labelling. Names on the same causal refits as tracker v1, using
lock_states on the unscaled component means. Templates still move under the
Wasserstein assignment. This is the third labelling next to tracker v1 and
tracker v2, not a separate per-fold HMM.

Bootstrap. Circular blocks of length ceil(sqrt(n)), 1000 resamples, seed 42.
src/crta/eval.py uses 1000 i.i.d. resamples. The 2000-resample circular
bootstrap in the repo belongs to the native-twin gap, not the four-hour table.
The task said to match 2000 only if the four-hour code uses it.

Threads. Each PPO call passes threads=1. train_ppo's default is unchanged.
Budget is max(8192, 10 * n_train), reward raw, scale 1, entropy 0, no window.

Verifier v2 banks are the training rows of that fold. Position is carried
across folds the way crta.pipeline.build carries it. Faithfulness is the share
of out-of-fold bars passing all three checks. Ablation is that verifier's
ablation pass rate. Both are properties of the conditioned agent.

Canonical table directory: paper/results/tables_4h/
Copy: results/tables/
"""


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def load_frozen_frame() -> pd.DataFrame:
    live = pd.read_parquet(LIVE_PARQUET)
    live["timestamp"] = pd.to_datetime(live["timestamp"], utc=True)
    raw = live.loc[live["timestamp"] <= STUDY_END].reset_index(drop=True)
    if len(raw) != 12544 or raw["timestamp"].iloc[-1] != STUDY_END:
        raise RuntimeError(
            f"frozen prefix is {len(raw)} bars ending {raw['timestamp'].iloc[-1]}, expected 12544 at {STUDY_END}"
        )
    if raw["timestamp"].iloc[0] != pd.Timestamp("2021-01-01", tz="UTC"):
        raise RuntimeError(f"unexpected first bar {raw['timestamp'].iloc[0]}")
    original = pd.read_parquet(ORIGINAL_BARS)
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True)
    # The live file's 12:00 close was replaced when that snapshot appended a later
    # tail. The archived run's last next_ret is the study's frozen return into
    # that candle. Restore it before features are built. It only changes the
    # last model row's reward; that row is never inside a training window.
    archived_next = float(original["next_ret"].iloc[-1])
    prev_close = float(raw["close"].iloc[-2])
    raw = raw.copy()
    raw.loc[raw.index[-1], "close"] = prev_close * np.exp(archived_next)
    frame = model_frame(raw)
    if len(frame) != 12510:
        raise RuntimeError(f"model frame has {len(frame)} rows, expected 12510")
    ref = np.load(TRACKER_NPZ)
    if not np.allclose(frame["close"].to_numpy(dtype=float), ref["close"].astype(float)):
        raise RuntimeError("frozen closes do not match tracker.npz")
    oos = frame.iloc[1080:].reset_index(drop=True)
    if len(oos) != len(original) or not oos["timestamp"].equals(original["timestamp"]):
        raise RuntimeError("out-of-fold timestamps do not match paper/results/run/bars.parquet")
    if not np.allclose(oos["close"].to_numpy(dtype=float), original["close"].to_numpy(dtype=float)):
        raise RuntimeError("out-of-fold closes do not match the original run")
    if not np.allclose(oos["next_ret"].to_numpy(dtype=float), original["next_ret"].to_numpy(dtype=float), atol=1e-10):
        raise RuntimeError("out-of-fold next_ret does not match the archived run")
    return frame


def _moment_assign(mu: np.ndarray) -> np.ndarray:
    mapping = lock_states(mu)
    index = {name: i for i, name in enumerate(REGIMES)}
    return np.array([index[mapping[k]] for k in range(len(mu))], dtype=int)


def moment_sorted_on_v1_refits(X: np.ndarray) -> tuple[list[str], np.ndarray, dict, dict]:
    """Moment names on the v1 refits. Template updates still use the W2 assignment."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    labels_w2: list[str] = []
    labels_ms: list[str] = []
    probs_w2 = np.zeros((n, len(REGIMES)))
    probs_ms = np.zeros((n, len(REGIMES)))
    diag_w2 = {"component_mu": [], "component_sd": [], "assign": [], "responsibility": [], "matched_costs": []}
    diag_ms = {"component_mu": [], "component_sd": [], "assign": [], "responsibility": [], "matched_costs": []}

    fitted = _fit_hmm(X[:BURN], 42)
    mu, sd = component_params(fitted)
    tmu, tsd, assign = _templates_from_lock(mu, sd)
    wl, wp, _proba = _window_labels(fitted, X[:BURN], assign, None, posterior="smoothed")
    labels_w2.extend(wl)
    labels_ms.extend(list(wl))
    probs_w2[:BURN] = wp
    probs_ms[:BURN] = wp

    t = BURN
    while t < n:
        end = min(t + STRIDE, n)
        fitted = _fit_hmm(X[:t], 42)
        mu, sd = component_params(fitted)
        cost = pairwise_w2(mu, sd, tmu, tsd)
        assign = assign_templates(mu, sd, tmu, tsd)
        moment = _moment_assign(mu)
        w2_cost = float(cost[np.arange(len(mu)), assign].mean())
        ms_cost = float(cost[np.arange(len(mu)), moment].mean())
        wl, wp, proba = _window_labels(fitted, X[t:end], assign, labels_w2[-1], posterior="smoothed")
        labels_w2.extend(wl)
        probs_w2[t:end] = wp
        wl_ms, wp_ms, proba_ms = _window_labels(fitted, X[t:end], moment, labels_ms[-1], posterior="smoothed")
        labels_ms.extend(wl_ms)
        probs_ms[t:end] = wp_ms
        weight = np.asarray(proba).sum(axis=0)
        for diag, chosen, chosen_cost, chosen_weight in (
            (diag_w2, assign, w2_cost, weight),
            (diag_ms, moment, ms_cost, np.asarray(proba_ms).sum(axis=0)),
        ):
            diag["component_mu"].append(np.array(mu, copy=True))
            diag["component_sd"].append(np.array(sd, copy=True))
            diag["assign"].append(np.array(chosen, copy=True))
            diag["responsibility"].append(np.array(chosen_weight, copy=True))
            diag["matched_costs"].append(chosen_cost)
        tmu, tsd = update_templates(tmu, tsd, mu, sd, proba, assign, ETA)
        t = end

    ref_labels, ref_probs = track(X, **LEGACY_V1)
    if ref_labels != labels_w2 or not np.allclose(ref_probs, probs_w2):
        raise RuntimeError("moment-sort replay diverged from track()")
    return labels_ms, probs_ms, diag_w2, diag_ms


def _stamp(frame: pd.DataFrame, labels, probs) -> pd.DataFrame:
    out = frame.copy()
    for i, name in enumerate(REGIMES):
        out[name] = np.asarray(probs)[:, i]
    out["regime"] = list(labels)
    return out


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame_path = OUT / "frame.parquet"
    label_path = OUT / "labels.npz"
    stats_path = OUT / "label_stats.json"
    if frame_path.exists() and label_path.exists() and stats_path.exists():
        print("labels already prepared", flush=True)
        return
    t0 = time.time()
    frame = load_frozen_frame()
    X = frame[list(HMM_COLS)].to_numpy(dtype=float)
    print("fitting tracker v1", flush=True)
    labels_v1, probs_v1 = track(X, **LEGACY_V1)
    print("fitting tracker v2", flush=True)
    diag_v2: dict = {}
    labels_v2, probs_v2 = track_v2(X, diagnostics=diag_v2, seed=42, **LEGACY_V2)
    print("fitting moment sort on the v1 refits", flush=True)
    labels_ms, probs_ms, diag_w2, diag_ms = moment_sorted_on_v1_refits(X)
    if labels_v1 != list(labels_v1):
        raise RuntimeError("v1 labels")
    after = slice(BURN, None)
    stats = {
        "tracker_v1": label_statistics(list(labels_v1)[after], frame.iloc[after], PERIODS, diag_w2),
        "tracker_v2": label_statistics(list(labels_v2)[after], frame.iloc[after], PERIODS, diag_v2),
        "moment_sorted": label_statistics(list(labels_ms)[after], frame.iloc[after], PERIODS, diag_ms),
        "n_model_bars": int(len(frame)),
        "burn": BURN,
        "after_burn_bars": int(len(frame) - BURN),
    }
    frame.to_parquet(frame_path, index=False)
    np.savez_compressed(
        label_path,
        labels_v1=np.array(labels_v1),
        probs_v1=probs_v1,
        labels_v2=np.array(labels_v2),
        probs_v2=probs_v2,
        labels_ms=np.array(labels_ms),
        probs_ms=probs_ms,
    )
    stats_path.write_text(json.dumps(_jsonable(stats), indent=2))
    print(f"prepared labels in {time.time() - t0:.0f}s", flush=True)


def _labels_for(labelling: str, packed) -> tuple[np.ndarray, np.ndarray]:
    key = {"tracker_v1": ("labels_v1", "probs_v1"), "tracker_v2": ("labels_v2", "probs_v2"), "moment_sorted": ("labels_ms", "probs_ms")}[
        labelling
    ]
    return packed[key[0]], packed[key[1]]


def _stat_point(returns, kind: str):
    if kind == "sharpe":
        return sharpe(returns)
    if kind == "sortino":
        return sortino(returns)
    raise ValueError(kind)


def circular_block_ci(returns, kind: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    block = max(1, int(np.ceil(np.sqrt(n))))
    point = _stat_point(r, kind)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    samples = r[idx]
    mean = samples.mean(axis=1)
    if kind == "sharpe":
        scale = samples.std(axis=1, ddof=1)
    else:
        scale = np.sqrt(np.mean(np.minimum(samples, 0.0) ** 2, axis=1))
    stat = np.full(n_boot, np.nan)
    ok = np.isfinite(scale) & (scale >= 1e-12)
    stat[ok] = mean[ok] / scale[ok] * np.sqrt(PERIODS)
    stat = stat[np.isfinite(stat)]
    if len(stat) == 0 or point is None:
        return {"point": point, "low": None, "high": None, "block": block, "n_boot": n_boot}
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return {"point": float(point), "low": float(lo), "high": float(hi), "block": block, "n_boot": n_boot}


def _system_metrics(ret, act) -> dict:
    return {
        "sharpe": circular_block_ci(ret, "sharpe"),
        "sortino": circular_block_ci(ret, "sortino"),
        "max_drawdown": max_drawdown(ret),
        "total_return": float(np.exp(np.asarray(ret).sum()) - 1.0),
        "position_std": float(np.std(act)),
        "extreme_share": float(np.mean(np.abs(act) > 0.95)),
    }


def _predict_fn(model, feats, pos):
    def predict(batch):
        batch = np.asarray(batch, dtype=float)
        n = len(feats)
        out = np.empty(len(batch))
        step = 8192
        for start in range(0, len(batch), step):
            sl = slice(start, start + step)
            idx = np.arange(start, min(start + step, len(batch))) % n
            obs = batch_observation(feats[idx], batch[sl], pos[idx], True)
            action, _ = model.predict(obs, deterministic=True)
            out[sl] = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        return out

    return predict


def _audit_fold(model, sub, banks) -> tuple[dict, dict]:
    feats = sub[list(FEATURE_COLS)].to_numpy(dtype=float)
    pos = sub["position_before"].to_numpy(dtype=float)
    probs = sub[list(REGIMES)].to_numpy(dtype=float)
    predict = _predict_fn(model, feats, pos)
    legacy = audit_many(predict, probs)
    v2 = audit_many_v2(predict, probs, banks)
    likely = sub["regime"].to_numpy()
    return verifier_report(legacy, regimes=likely), verifier_report(v2, regimes=likely)


def _concat_reports(parts: list[dict]) -> dict:
    """Bar-weighted means of pass rates. n is the total out-of-fold count."""
    n = sum(int(part["n"]) for part in parts)
    out = {"n": n}
    for key in ("ablation", "counterfactual", "sensitivity", "all", "untestable_share", "median_ablation_delta"):
        if key == "median_ablation_delta":
            continue
        num = 0.0
        for part in parts:
            value = part.get(key)
            if value is None:
                value = 0.0
            num += float(value) * int(part["n"])
        out[key] = num / n if n else None
    # Median of medians is not the pooled median. Recompute is not available here;
    # the seed payload stores the pooled value separately when bars are in hand.
    out["median_ablation_delta"] = None
    return out


def run_seed(labelling: str, seed: int, timesteps_override: int | None = None) -> dict:
    prepare()
    frame = pd.read_parquet(OUT / "frame.parquet")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    packed = np.load(OUT / "labels.npz", allow_pickle=False)
    labels, probs = _labels_for(labelling, packed)
    if len(labels) != len(frame):
        raise RuntimeError(f"{labelling} labels {len(labels)} != frame {len(frame)}")
    return walk_forward_seed(frame, labels, probs, labelling, seed, OUT / labelling / f"seed_{seed}", timesteps_override)


def walk_forward_seed(
    frame: pd.DataFrame,
    labels,
    probs,
    labelling: str,
    seed: int,
    seed_dir: Path,
    timesteps_override: int | None = None,
) -> dict:
    """Train both twins per fold, audit the conditioned twin, and write eval.json under seed_dir."""
    bounds = walk_forward_bounds(len(frame), min_train=1080)
    seed_dir.mkdir(parents=True, exist_ok=True)
    carry_c = 0.0
    carry_b = 0.0
    pieces = []
    fold_rows = []
    audits = {"legacy": [], "v2": []}
    for fold, (train_end, test_end) in enumerate(bounds):
        steps = int(timesteps_override) if timesteps_override is not None else max(8192, 10 * train_end)
        train = _stamp(frame.iloc[:train_end].reset_index(drop=True), labels[:train_end], probs[:train_end])
        test = _stamp(
            frame.iloc[train_end:test_end].reset_index(drop=True),
            labels[train_end:test_end],
            probs[train_end:test_end],
        )
        models = {}
        for side, include in (("conditioned", True), ("blind", False)):
            path = seed_dir / f"fold{fold}_{side}.zip"
            model = None
            if path.exists() and timesteps_override is None:
                loaded, meta = _load(path)
                if int(meta.get("timesteps", -1)) == steps and meta.get("labelling") == labelling and int(meta.get("seed", -1)) == seed:
                    model = loaded
                else:
                    path.unlink()
                    meta_path = path.with_suffix(".meta.json")
                    if meta_path.exists():
                        meta_path.unlink()
            if model is None:
                t0 = time.time()
                model = train_ppo(
                    train,
                    include,
                    timesteps=steps,
                    seed=seed,
                    window=None,
                    reward_scale=1.0,
                    ent_coef=0.0,
                    threads=THREADS,
                    reward_mode="raw",
                    window_sampling="uniform",
                )
                model.crta_meta.update(
                    {
                        "labelling": labelling,
                        "seed": int(seed),
                        "fold": int(fold),
                        "twin": side,
                        "timesteps": int(steps),
                        "n_train": int(train_end),
                        "recipe": "4h_walkforward",
                    }
                )
                save_twin(model, path)
                print(
                    f"{labelling} seed {seed} fold {fold} {side}: {steps} steps in {time.time() - t0:.0f}s",
                    flush=True,
                )
            models[side] = model
        actions_c = rollout(models["conditioned"], test, True, carry_c)
        actions_b = rollout(models["blind"], test, False, carry_b)
        carry_c = float(actions_c[-1])
        carry_b = float(actions_b[-1])
        test = test.copy()
        test["fold"] = fold
        test["action_conditioned"] = actions_c
        test["action_blind"] = actions_b
        test["n_train"] = train_end
        test["timesteps"] = steps
        pieces.append(test)
        banks = empirical_banks(train[list(REGIMES)].to_numpy(dtype=float))
        # position_before is filled after concat; audit this fold after that.
        fold_rows.append({"fold": fold, "train_end": train_end, "test_end": test_end, "banks": banks, "model": models["conditioned"]})

    out = pd.concat(pieces, ignore_index=True)
    out["position_before"] = np.concatenate([[0.0], out["action_conditioned"].to_numpy()[:-1]])
    out["action_bh"] = 1.0
    out["ret_conditioned"] = strategy_returns(out["action_conditioned"], out["next_ret"])
    out["ret_blind"] = strategy_returns(out["action_blind"], out["next_ret"])
    out["ret_bh"] = strategy_returns(out["action_bh"], out["next_ret"])

    legacy_parts = []
    v2_parts = []
    legacy_pass = []
    v2_pass = []
    legacy_ab = []
    v2_ab = []
    for spec in fold_rows:
        sub = out.loc[out["fold"] == spec["fold"]]
        legacy, v2 = _audit_fold(spec["model"], sub, spec["banks"])
        legacy_parts.append(legacy)
        v2_parts.append(v2)
        # Keep bar-level pass arrays by re-auditing is wasteful. Reconstruct from rates
        # only for the pooled rate via weights. Bar-level flags are needed for the
        # pooled rate, which _concat_reports weights by n. Fold n is exact.
        legacy_pass.append(legacy["all"])
        v2_pass.append(v2["all"])
        legacy_ab.append(legacy["ablation"])
        v2_ab.append(v2["ablation"])
        del spec["model"]
        del spec["banks"]

    def pooled(parts):
        n = sum(part["n"] for part in parts)
        def rate(key):
            return float(sum(part[key] * part["n"] for part in parts) / n)
        return {"ablation": rate("ablation"), "all": rate("all"), "n": int(n)}

    systems = {
        "conditioned": _system_metrics(out["ret_conditioned"], out["action_conditioned"].to_numpy(dtype=float)),
        "blind": _system_metrics(out["ret_blind"], out["action_blind"].to_numpy(dtype=float)),
        "buy_hold": _system_metrics(out["ret_bh"], out["action_bh"].to_numpy(dtype=float)),
    }
    fold_metrics = []
    for fold, (train_end, test_end) in enumerate(bounds):
        mask = out["fold"].to_numpy() == fold
        block = out.loc[mask]
        fold_metrics.append(
            {
                "fold": fold,
                "n_train": int(train_end),
                "n_test": int(mask.sum()),
                "timesteps": int(block["timesteps"].iloc[0]),
                "test_start": block["timestamp"].iloc[0].isoformat(),
                "test_end": block["timestamp"].iloc[-1].isoformat(),
                "conditioned": _system_metrics(block["ret_conditioned"], block["action_conditioned"].to_numpy(dtype=float)),
                "blind": _system_metrics(block["ret_blind"], block["action_blind"].to_numpy(dtype=float)),
                "buy_hold": _system_metrics(block["ret_bh"], block["action_bh"].to_numpy(dtype=float)),
            }
        )
    regimes = {}
    for name in REGIMES:
        mask = out["regime"].to_numpy() == name
        chunk = out.loc[mask]
        regimes[name] = {"n": int(mask.sum())}
        if mask.sum() == 0:
            continue
        for side, ret_col, act_col in (
            ("conditioned", "ret_conditioned", "action_conditioned"),
            ("blind", "ret_blind", "action_blind"),
            ("buy_hold", "ret_bh", "action_bh"),
        ):
            regimes[name][side] = {
                "sharpe": sharpe(chunk[ret_col]) if mask.sum() >= 2 else None,
                "mean_position": float(chunk[act_col].mean()),
                "total_return": float(np.exp(chunk[ret_col].sum()) - 1.0),
            }
    payload = {
        "seed": int(seed),
        "labelling": labelling,
        "n_bars": int(len(out)),
        "threads": THREADS,
        "bootstrap": {"method": "circular_block", "n_boot": N_BOOT, "seed": BOOT_SEED, "block_rule": "ceil(sqrt(n))"},
        "timesteps_override": timesteps_override,
        "systems": systems,
        "folds": fold_metrics,
        "verifiers": {"legacy": pooled(legacy_parts), "v2": pooled(v2_parts)},
        "verifier_detail": {"legacy": legacy_parts, "v2": v2_parts},
        "regimes": regimes,
    }
    if timesteps_override is None:
        (seed_dir / "eval.json").write_text(json.dumps(_jsonable(payload), indent=2))
        keep = [
            "timestamp",
            "fold",
            "regime",
            "action_conditioned",
            "action_blind",
            "action_bh",
            "ret_conditioned",
            "ret_blind",
            "ret_bh",
            "next_ret",
            "close",
        ]
        out[keep].to_parquet(seed_dir / "bars.parquet", index=False)
    return payload


def _load(path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    sidecar = path.with_suffix(".meta.json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    model.crta_meta = meta
    return model, meta


def _mean_std(values) -> dict:
    clean = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            clean.append(number)
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(clean, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def _flat_metric_row(seed, labelling, verifier, system, metrics, audit) -> dict:
    row = {
        "seed": seed,
        "labelling": labelling,
        "verifier": verifier,
        "system": system,
        "sharpe": metrics["sharpe"]["point"],
        "sharpe_low": metrics["sharpe"]["low"],
        "sharpe_high": metrics["sharpe"]["high"],
        "sharpe_block": metrics["sharpe"]["block"],
        "sortino": metrics["sortino"]["point"],
        "sortino_low": metrics["sortino"]["low"],
        "sortino_high": metrics["sortino"]["high"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": metrics["total_return"],
        "position_std": metrics["position_std"],
        "extreme_share": metrics["extreme_share"],
        "ablation_pass_rate": None if system != "conditioned" else audit["ablation"],
        "faithfulness_rate": None if system != "conditioned" else audit["all"],
    }
    return row


def write_tables() -> None:
    rows = []
    fold_rows = []
    regime_rows = []
    for labelling in LABELLINGS:
        for seed in SEEDS:
            path = OUT / labelling / f"seed_{seed}" / "eval.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            for verifier in VERIFIERS:
                audit = payload["verifiers"][verifier]
                for system, metrics in payload["systems"].items():
                    rows.append(_flat_metric_row(seed, labelling, verifier, system, metrics, audit))
            for fold in payload["folds"]:
                for system in ("conditioned", "blind", "buy_hold"):
                    metrics = fold[system]
                    fold_rows.append(
                        {
                            "seed": seed,
                            "labelling": labelling,
                            "fold": fold["fold"],
                            "n_train": fold["n_train"],
                            "n_test": fold["n_test"],
                            "timesteps": fold["timesteps"],
                            "test_start": fold["test_start"],
                            "test_end": fold["test_end"],
                            "system": system,
                            "sharpe": metrics["sharpe"]["point"],
                            "sharpe_low": metrics["sharpe"]["low"],
                            "sharpe_high": metrics["sharpe"]["high"],
                            "sortino": metrics["sortino"]["point"],
                            "max_drawdown": metrics["max_drawdown"],
                            "total_return": metrics["total_return"],
                            "position_std": metrics["position_std"],
                            "extreme_share": metrics["extreme_share"],
                        }
                    )
            for name in REGIMES:
                block = payload["regimes"][name]
                regime_rows.append(
                    {
                        "seed": seed,
                        "labelling": labelling,
                        "regime": name,
                        "n": block["n"],
                        "sharpe_conditioned": None if block["n"] < 2 else block.get("conditioned", {}).get("sharpe"),
                        "sharpe_blind": None if block["n"] < 2 else block.get("blind", {}).get("sharpe"),
                        "sharpe_buy_hold": None if block["n"] < 2 else block.get("buy_hold", {}).get("sharpe"),
                        "position_conditioned": None if block["n"] == 0 else block.get("conditioned", {}).get("mean_position"),
                        "position_blind": None if block["n"] == 0 else block.get("blind", {}).get("mean_position"),
                        "position_buy_hold": None if block["n"] == 0 else block.get("buy_hold", {}).get("mean_position"),
                    }
                )
    by_seed = pd.DataFrame(rows)
    metric_cols = [
        "sharpe",
        "sortino",
        "max_drawdown",
        "total_return",
        "position_std",
        "extreme_share",
        "ablation_pass_rate",
        "faithfulness_rate",
    ]
    agg_rows = []
    for (labelling, verifier, system), group in by_seed.groupby(["labelling", "verifier", "system"], sort=False):
        row = {"seed": "mean", "labelling": labelling, "verifier": verifier, "system": system, "n_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            stats = _mean_std(group[col].tolist())
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    headline_rows = []
    for _, row in aggregate.iterrows():
        if row["system"] == "buy_hold":
            continue
        headline_rows.append(
            {
                "labelling": row["labelling"],
                "verifier": row["verifier"],
                "system": row["system"],
                "sharpe": _pm(row["sharpe_mean"], row["sharpe_std"]),
                "total_return": _pm(row["total_return_mean"], row["total_return_std"]),
                "position_std": _pm(row["position_std_mean"], row["position_std_std"]),
                "extreme_share": _pm(row["extreme_share_mean"], row["extreme_share_std"]),
                "ablation_pass_rate": _pm(row["ablation_pass_rate_mean"], row["ablation_pass_rate_std"]),
                "faithfulness_rate": _pm(row["faithfulness_rate_mean"], row["faithfulness_rate_std"]),
            }
        )
    folds = pd.DataFrame(fold_rows)
    fold_agg = []
    for (labelling, fold, system), group in folds.groupby(["labelling", "fold", "system"], sort=False):
        base = group.iloc[0]
        row = {
            "labelling": labelling,
            "fold": int(fold),
            "system": system,
            "n_train": int(base["n_train"]),
            "n_test": int(base["n_test"]),
            "timesteps": int(base["timesteps"]),
            "test_start": base["test_start"],
            "test_end": base["test_end"],
        }
        for col in ("sharpe", "sortino", "max_drawdown", "total_return", "position_std", "extreme_share"):
            stats = _mean_std(group[col].tolist())
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
        fold_agg.append(row)
    regimes = pd.DataFrame(regime_rows)
    regime_agg = []
    for (labelling, regime), group in regimes.groupby(["labelling", "regime"], sort=False):
        row = {"labelling": labelling, "regime": regime, "n": int(group["n"].iloc[0])}
        for col in (
            "sharpe_conditioned",
            "sharpe_blind",
            "sharpe_buy_hold",
            "position_conditioned",
            "position_blind",
            "position_buy_hold",
        ):
            stats = _mean_std(group[col].tolist())
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
        regime_agg.append(row)
    labels = _label_table()
    settings = _settings_table()
    TAB.mkdir(parents=True, exist_ok=True)
    files = {
        "metrics_by_seed.csv": by_seed,
        "metrics_mean_std.csv": aggregate,
        "headline_mean_std.csv": pd.DataFrame(headline_rows),
        "table_I_settings.csv": settings,
        "table_I_folds.csv": pd.DataFrame(fold_agg),
        "table_I_folds_by_seed.csv": folds,
        "table_III_labels.csv": labels,
        "table_IV_performance.csv": aggregate,
        "table_V_regime.csv": pd.DataFrame(regime_agg),
        "table_V_regime_by_seed.csv": regimes,
    }
    mapping = (
        "Table mapping for this rerun\n"
        "=============================\n"
        "The current paper's tables, in order, are tab:folds, tab:hyper, tab:labels, tab:perf, tab:regime.\n"
        "The task groups settings and folds as Table I, label statistics as Table III,\n"
        "out-of-fold performance as Table IV, and per-regime results as Table V.\n"
        "table_I_settings.csv + table_I_folds.csv -> Table I (paper tab:hyper and tab:folds).\n"
        "table_III_labels.csv -> Table III (paper tab:labels), after burn-in.\n"
        "table_IV_performance.csv -> Table IV (paper tab:perf), mean and std across seeds 42-46.\n"
        "table_V_regime.csv -> Table V (paper tab:regime), mean and std across seeds.\n"
        "metrics_by_seed.csv is the per-seed source. Verifier rows repeat return metrics\n"
        "because the verifier does not change the executed position. Ablation and faithfulness\n"
        "are filled for the conditioned agent only.\n"
        "Canonical directory: paper/results/tables_4h/. Copy: results/tables/.\n"
    )
    (TAB / "MAPPING.txt").write_text(mapping)
    (TAB / "NOTES.txt").write_text(NOTES)
    for name, frame in files.items():
        frame.to_csv(TAB / name, index=False)
    _mirror()
    print(f"wrote tables under {TAB}", flush=True)


def _pm(mean, std) -> str:
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return ""
    if std is None:
        return f"{mean:.6f}"
    return f"{mean:.6f}±{std:.6f}"


def _label_table() -> pd.DataFrame:
    stats = json.loads((OUT / "label_stats.json").read_text())
    rows = []
    for labelling in LABELLINGS:
        block = stats[labelling]
        row = {
            "labelling": labelling,
            "n_bars_after_burn": block["n"],
            "switches_per_year": block["switches_per_year"],
            "median_run_bars": block["median_run_bars"],
            "mean_matched_w2": block["mean_matched_w2"],
            "identity_jump": block["identity_jump"],
        }
        for name in REGIMES:
            per = block["per_regime"][name]
            row[f"{name}_share"] = per["share"]
            mean_ret = per["mean_return"]
            vol = per["volatility"]
            row[f"{name}_ret_bp"] = None if mean_ret is None else mean_ret * 1e4
            row[f"{name}_sigma_pct"] = None if vol is None else vol * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def _settings_table() -> pd.DataFrame:
    rows = [
        ("data", "BTC/USDT 4h, rows of paper/results/live/btc_usdt_4h.parquet with timestamp <= 2026-09-22 12:00 UTC"),
        ("n_raw_bars", "12544"),
        ("n_model_bars", "12510 after feature warm-up"),
        ("folds", "4 expanding folds, min train 1080"),
        ("budget", "max(8192, 10 * n_train) steps per twin per fold"),
        ("reward", "raw a*r - 0.001*|da|, scale 1"),
        ("window", "none"),
        ("entropy", "0"),
        ("policy", "MLP 64-64, tanh, Gaussian, executed action clipped to [-1, 1]"),
        ("ppo", "lr 3e-4, gamma 0.99, gae_lambda 0.95, n_steps 256, batch 64, n_epochs 10 (SB3 defaults)"),
        ("seeds", "42,43,44,45,46 for both twins; tracker seed 42"),
        ("threads", "1 passed to train_ppo; the function default is unchanged"),
        ("tracker_v1", "burn 252, stride 42, eta 0.2, dwell 30"),
        ("tracker_v2", "same calendar, matching nearest, death_after 20, cost_quantile 0.95, filtered probabilities, online dwell"),
        ("moment_sorted", "moment names on the same v1 refits; templates still update by W2"),
        ("verifier_legacy", "one-hot counterfactual, delta_ab 0.15, delta_cf 0.10, at most one uptick"),
        ("verifier_v2", "nearest training probability vector, delta_sens 0.05, untestable regime does not fail the bar"),
        ("bootstrap", "circular block, block length ceil(sqrt(n)), 1000 resamples, seed 42, 95% percentile"),
    ]
    return pd.DataFrame(rows, columns=["component", "setting"])


def _mirror() -> None:
    TAB_COPY.mkdir(parents=True, exist_ok=True)
    for path in TAB.iterdir():
        if path.is_file():
            (TAB_COPY / path.name).write_bytes(path.read_bytes())


def _run_seed_process(labelling: str, seed: int) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "seed",
        "--labelling",
        labelling,
        "--seed",
        str(seed),
    ]
    print("launch", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd).wait()


def run_all(jobs: int) -> None:
    prepare()
    pending = [(lab, seed) for lab in LABELLINGS for seed in SEEDS]
    running: list[tuple[subprocess.Popen, str]] = []
    failed = []
    while pending or running:
        while pending and len(running) < jobs:
            lab, seed = pending.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "seed", "--labelling", lab, "--seed", str(seed)]
            print("launch", lab, seed, flush=True)
            running.append((subprocess.Popen(cmd), f"{lab}:{seed}"))
        time.sleep(5)
        still = []
        for proc, name in running:
            code = proc.poll()
            if code is None:
                still.append((proc, name))
            elif code != 0:
                failed.append((name, code))
                print(f"FAILED {name} exit {code}", flush=True)
            else:
                print(f"finished {name}", flush=True)
        running = still
    if failed:
        raise SystemExit(f"seed jobs failed: {failed}")
    write_tables()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    seed_p = sub.add_parser("seed")
    seed_p.add_argument("--labelling", choices=LABELLINGS, required=True)
    seed_p.add_argument("--seed", type=int, required=True)
    seed_p.add_argument("--timesteps", type=int, default=None)
    sub.add_parser("tables")
    all_p = sub.add_parser("all")
    all_p.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "seed":
        run_seed(args.labelling, args.seed, args.timesteps)
    elif args.cmd == "tables":
        write_tables()
    else:
        run_all(args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

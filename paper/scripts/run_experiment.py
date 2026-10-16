"""Train both twins for a config and a list of seeds, then aggregate.

Lives next to the other study scripts. One seed writes only its own folder.

  python paper/scripts/run_experiment.py --config configs/default.yaml --asset BTC --tf 1h --stage validate --seeds 42
  python paper/scripts/run_experiment.py --quick --out /tmp/crta-quick
  python paper/scripts/run_experiment.py --aggregate paper/results/v2/grid/excess_bh/btc_1h

The timeframe protocol is one walk-forward split: train on bars before the cut,
score the following window. The tracker seed stays 42 so seeds change only the PPO
draw. ``--quick`` is a synthetic smoke test and is not a reported result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crta.agent import prob_peak_report, save_twin, train_ppo  # noqa: E402
from crta.config import ExperimentConfig, _parse_scalar  # noqa: E402
from crta.data import FEATURE_COLS, HMM_COLS, features, synthetic_ohlcv  # noqa: E402
from crta.eval import TURNOVER_COST, max_drawdown, strategy_returns  # noqa: E402
from crta.regime import REGIMES, track, track_v2  # noqa: E402
from crta.verify import audit_many, audit_many_v2, empirical_banks, label_agreement, verifier_report  # noqa: E402

LABELS = ROOT / "paper" / "results" / "v2" / "labels"
PERIODS = {"15m": 4 * 24 * 365, "1h": 24 * 365, "4h": 6 * 365, "synthetic": 6 * 365}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _ratio(returns, periods: int, downside: bool):
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return None
    if downside:
        scale = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    else:
        scale = float(r.std(ddof=1))
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    return float(r.mean() / scale * np.sqrt(periods))


def paired_block_gap(a, b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Circular block bootstrap of summed log-return gap a - b. Block length is ceil(sqrt(n))."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    block = max(1, int(np.ceil(np.sqrt(n))))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    sums = d[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {
        "gap_log": float(d.sum()),
        "low": float(lo),
        "high": float(hi),
        "share_positive": float((sums > 0).mean()),
        "block": block,
        "cond_beats_blind": bool(d.sum() > 0.0),
    }


def _stats(ret, act, periods: int) -> dict:
    prev = np.concatenate([[0.0], act[:-1]])
    return {
        "sharpe": _ratio(ret, periods, downside=False),
        "sortino": _ratio(ret, periods, downside=True),
        "total_return": float(np.exp(np.asarray(ret).sum()) - 1.0),
        "max_drawdown": max_drawdown(ret),
        "mean_position": float(np.mean(act)),
        "position_std": float(np.std(act)),
        "extreme_share": float(np.mean(np.abs(act) > 0.95)),
        "turnover": float(np.mean(np.abs(act - prev))),
        "cost_paid": float((TURNOVER_COST * np.abs(act - prev)).sum()),
    }


def _tracker_kwargs(cfg: ExperimentConfig, timeframe: str, quick: bool) -> dict:
    if quick:
        base = {"burn": 40, "stride": 20, "dwell": 8, "seed": 42}
    else:
        import timeframe_study as study

        base = {**study.CAL[timeframe], "seed": 42}
    if cfg.tracker == "v2":
        base.update(
            matching=cfg.tracker_matching,
            death_after=int(cfg.tracker_death_after),
            cost_quantile=float(cfg.tracker_cost_quantile),
        )
    return base


def _stamp(frame: pd.DataFrame, labels, probs) -> pd.DataFrame:
    out = frame.copy()
    for i, name in enumerate(REGIMES):
        out[name] = probs[:, i]
    out["regime"] = list(labels)
    return out


def synthetic_frame(cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict]:
    raw = synthetic_ohlcv(7).iloc[:800].reset_index(drop=True)
    frame = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    kwargs = _tracker_kwargs(cfg, "1h", quick=True)
    call = track_v2 if cfg.tracker == "v2" else track
    labels, probs = call(frame[list(HMM_COLS)].to_numpy(dtype=float), **kwargs)
    frame = _stamp(frame, labels, probs)
    cut = int(len(frame) * 0.7)
    usable = frame["next_ret"].notna()
    return frame, {
        "train": frame.loc[usable].iloc[:cut].reset_index(drop=True),
        "segments": {"validation": frame.loc[usable].iloc[cut:].reset_index(drop=True)},
        "periods": PERIODS["synthetic"],
    }


def timeframe_frame(cfg: ExperimentConfig, asset: str, timeframe: str) -> tuple[pd.DataFrame, dict]:
    import live_study as live
    import timeframe_study as study

    raw = study.candles(asset, timeframe)
    framed = features(raw).dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    framed["timestamp"] = pd.to_datetime(framed["timestamp"], utc=True)
    kwargs = _tracker_kwargs(cfg, timeframe, quick=False)
    if cfg.tracker == "v1":
        labelled = study.labelled(asset, timeframe, raw)
        frame = labelled
    else:
        LABELS.mkdir(parents=True, exist_ok=True)
        cache = LABELS / f"{asset.lower()}_{timeframe}_v2.parquet"
        key_path = LABELS / f"{asset.lower()}_{timeframe}_v2.key.json"
        key = {"n": len(framed), "kwargs": kwargs, "first": str(framed["timestamp"].iloc[0])}
        if cache.exists() and key_path.exists() and json.loads(key_path.read_text()) == key:
            lab = pd.read_parquet(cache)
            for col in [*REGIMES, "regime"]:
                framed[col] = lab[col].to_numpy()
            frame = framed
        else:
            diag: dict = {}
            labels, probs = track_v2(
                framed[list(HMM_COLS)].to_numpy(dtype=float), diagnostics=diag, **kwargs
            )
            frame = _stamp(framed, labels, probs)
            tmp = cache.with_suffix(".parquet.tmp")
            frame[["timestamp", *REGIMES, "regime"]].to_parquet(tmp, index=False)
            tmp.replace(cache)
            key_path.write_text(json.dumps(key))
            if diag.get("component_mu"):
                np.savez_compressed(
                    cache.with_suffix(".diag.npz"),
                    matched_costs=np.asarray(diag["matched_costs"], dtype=float),
                    component_mu=np.stack(diag["component_mu"]),
                    component_sd=np.stack(diag["component_sd"]),
                    assign=np.stack(diag["assign"]),
                    responsibility=np.stack(diag["responsibility"]),
                    bars=np.asarray(diag.get("bars", []), dtype=int) if diag.get("bars") else np.arange(len(diag["matched_costs"])),
                )
    ts = frame["timestamp"]
    usable = frame["next_ret"].notna()
    if cfg.stage == "validate":
        cut = study.VAL_START
        train = frame.loc[(ts < cut) & usable].reset_index(drop=True)
        segments = {
            "validation": frame.loc[(ts >= study.VAL_START) & (ts < study.HOLDOUT_START) & usable].reset_index(drop=True)
        }
    else:
        cut = study.HOLDOUT_START
        train = frame.loc[(ts < cut) & usable].reset_index(drop=True)
        segments = {
            "holdout": frame.loc[(ts >= study.HOLDOUT_START) & (ts < live.LIVE_START) & usable].reset_index(drop=True),
            "live": frame.loc[(ts >= live.LIVE_START) & usable].reset_index(drop=True),
        }
    return frame, {"train": train, "segments": segments, "periods": PERIODS[timeframe], "cut": str(cut)}


def _trade(model, frame: pd.DataFrame, include_regime: bool):
    from crta.agent import predict_action

    feats = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    probs = frame[list(REGIMES)].to_numpy(dtype=float)
    pos = 0.0
    actions = np.empty(len(frame))
    before = np.empty(len(frame))
    for i in range(len(frame)):
        before[i] = pos
        pos = predict_action(model, feats[i], probs[i], pos, include_regime)
        actions[i] = pos
    return actions, before


def _audit(model, frame, before, banks, cfg: ExperimentConfig) -> dict:
    from crta.agent import batch_observation

    feats = frame[list(FEATURE_COLS)].to_numpy(dtype=float)
    probs = frame[list(REGIMES)].to_numpy(dtype=float)
    meta = getattr(model, "crta_meta", {}) or {}
    temperature = float(meta.get("prob_temperature", 1.0))
    floor = float(meta.get("prob_floor", 0.0))
    n = len(frame)

    def predict(batch):
        batch = np.asarray(batch, dtype=float)
        out = np.empty(len(batch))
        step = 8192
        for start in range(0, len(batch), step):
            sl = slice(start, start + step)
            idx = np.arange(start, min(start + step, len(batch))) % n
            # batch_observation smooths again; pass temperature 1 and floor 0 here
            # because the caller already needs the intervened probabilities untouched
            # except for the model's own smoothing. Apply that smoothing inside.
            obs = batch_observation(feats[idx], batch[sl], before[idx], True, temperature, floor)
            if cfg.action_mode == "gaussian_clip":
                action, _ = model.predict(obs, deterministic=True)
                out[sl] = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
            else:
                from crta.agent import audit_positions

                out[sl] = audit_positions(model, obs)
        return out

    legacy = audit_many(predict, probs)
    v2 = audit_many_v2(
        predict,
        probs,
        banks,
        delta_ab=cfg.delta_ab,
        delta_cf=cfg.delta_cf,
        delta_sens=cfg.delta_sens,
        pick=cfg.cf_pick,
    )
    likely = np.array(REGIMES)[np.argmax(probs, axis=1)]
    legacy_report = verifier_report(legacy, regimes=likely)
    # The legacy auditor has no untestable list; give the report a full testable mask.
    legacy["testable"] = np.ones(len(frame), dtype=bool)
    legacy["untestable"] = [[] for _ in range(len(frame))]
    legacy_report = verifier_report(legacy, regimes=likely)
    v2_report = verifier_report(v2, regimes=likely)
    return {
        "legacy": legacy_report,
        "v2": v2_report,
        "label_agreement": label_agreement(probs, frame["regime"].to_numpy()) if "regime" in frame.columns else None,
        "regularized": bool(meta.get("faithfulness_regularizer")),
    }


def _per_regime(act_c, act_b, ret_c, ret_b, likely) -> dict:
    out = {}
    for name in REGIMES:
        mask = likely == name
        out[name] = {
            "share": float(mask.mean()) if len(mask) else None,
            "n": int(mask.sum()),
            "cond_position": float(act_c[mask].mean()) if mask.any() else None,
            "blind_position": float(act_b[mask].mean()) if mask.any() else None,
            "cond_position_std": float(act_c[mask].std()) if mask.sum() > 1 else None,
            "blind_position_std": float(act_b[mask].std()) if mask.sum() > 1 else None,
            "cond_return": float(np.exp(ret_c[mask].sum()) - 1.0) if mask.any() else None,
            "blind_return": float(np.exp(ret_b[mask].sum()) - 1.0) if mask.any() else None,
        }
    return out


def score_pair(models, segments, periods, banks, cfg: ExperimentConfig) -> tuple[dict, pd.DataFrame]:
    report = {}
    frames = []
    for name, seg in segments.items():
        if len(seg) < 2:
            report[name] = {"n_bars": int(len(seg)), "skipped": True}
            continue
        act_c, before_c = _trade(models["conditioned"], seg, True)
        act_b, _ = _trade(models["blind"], seg, False)
        nxt = seg["next_ret"].to_numpy(dtype=float)
        ret_c = strategy_returns(act_c, nxt)
        ret_b = strategy_returns(act_b, nxt)
        ret_h = strategy_returns(np.ones(len(seg)), nxt)
        likely = np.array(REGIMES)[seg[list(REGIMES)].to_numpy(dtype=float).argmax(axis=1)]
        audit = _audit(models["conditioned"], seg, before_c, banks, cfg)
        gap = paired_block_gap(ret_c, ret_b, seed=int(cfg.seed))
        report[name] = {
            "n_bars": int(len(seg)),
            "first_bar_utc": seg["timestamp"].iloc[0].isoformat() if "timestamp" in seg.columns else None,
            "last_bar_utc": seg["timestamp"].iloc[-1].isoformat() if "timestamp" in seg.columns else None,
            "conditioned": _stats(ret_c, act_c, periods),
            "blind": _stats(ret_b, act_b, periods),
            "buy_hold": _stats(ret_h, np.ones(len(seg)), periods),
            "gap_cond_minus_blind": gap,
            "cond_beats_blind": gap["cond_beats_blind"],
            "regimes": _per_regime(act_c, act_b, ret_c, ret_b, likely),
            "faithfulness": audit,
            "ablation_pass_rate": audit["legacy"]["ablation"],
            "faithfulness_rate": audit["v2"]["all"] if cfg.verifier == "v2" else audit["legacy"]["all"],
        }
        frames.append(
            pd.DataFrame(
                {
                    "segment": name,
                    "timestamp": seg["timestamp"] if "timestamp" in seg.columns else np.arange(len(seg)),
                    "close": seg["close"] if "close" in seg.columns else np.nan,
                    "regime": likely,
                    "action_conditioned": act_c,
                    "action_blind": act_b,
                    "equity_conditioned": np.exp(np.cumsum(ret_c)),
                    "equity_blind": np.exp(np.cumsum(ret_b)),
                    "equity_bh": np.exp(np.cumsum(ret_h)),
                }
            )
        )
    bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return report, bars


def _train_kwargs(cfg: ExperimentConfig, seed: int, quick: bool, timeframe: str) -> dict:
    steps = 256 if quick else int(cfg.timesteps)
    window = 64 if quick else cfg.window
    return dict(
        timesteps=steps,
        seed=seed,
        window=window,
        reward_scale=float(cfg.reward_scale if not quick else min(cfg.reward_scale, 100.0)),
        ent_coef=float(cfg.ent_coef),
        threads=1 if quick else cfg.threads,
        reward_mode=cfg.reward_mode,
        diff_sharpe_eta=float(cfg.diff_sharpe_eta),
        pos_extreme_penalty=float(cfg.pos_extreme_penalty),
        pos_variance_bonus=float(cfg.pos_variance_bonus),
        variance_every=int(cfg.variance_every),
        ent_coef_start=cfg.ent_coef_start,
        collapse_log_every=32 if quick else int(cfg.collapse_log_every),
        window_sampling=cfg.window_sampling,
        regime_balance_beta=float(cfg.regime_balance_beta),
        sign_flip_prob=0.0 if quick else float(cfg.sign_flip_prob),
        regime_perturb_prob=float(cfg.regime_perturb_prob),
        faithfulness_regularizer=bool(cfg.faithfulness_regularizer),
        faithfulness_reg_coef=float(cfg.faithfulness_reg_coef),
        prob_temperature=float(cfg.prob_temperature),
        prob_floor=float(cfg.prob_floor),
        tracker=cfg.tracker,
        tracker_kwargs=_tracker_kwargs(cfg, timeframe, quick),
        action_mode=cfg.action_mode,
        risk_aversion=float(cfg.risk_aversion),
    )


def run_seed(cfg: ExperimentConfig, seed: int, out_dir: Path, quick: bool, asset: str, timeframe: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if quick or cfg.protocol == "synthetic":
        _frame, bundle = synthetic_frame(cfg)
        periods = bundle["periods"]
    else:
        _frame, bundle = timeframe_frame(cfg, asset, timeframe)
        periods = bundle["periods"]
    train = bundle["train"]
    banks = empirical_banks(train[list(REGIMES)].to_numpy(dtype=float))
    peaks = prob_peak_report(train[list(REGIMES)].to_numpy(dtype=float), cfg.prob_temperature, cfg.prob_floor)
    kwargs = _train_kwargs(cfg, seed, quick, timeframe)
    models = {}
    for side, include in (("conditioned", True), ("blind", False)):
        t1 = time.time()
        model = train_ppo(train, include, **kwargs)
        save_twin(model, out_dir / f"{side}.zip")
        models[side] = model
        print(f"seed {seed} {side}: {kwargs['timesteps']} steps in {time.time() - t1:.1f}s", flush=True)
    report, bars = score_pair(models, bundle["segments"], periods, banks, cfg.with_overrides(seed=seed))
    if len(bars):
        bars.to_parquet(out_dir / "bars.parquet", index=False)
    payload = {
        "seed": int(seed),
        "quick": bool(quick),
        "asset": asset,
        "timeframe": timeframe if not quick else "synthetic",
        "stage": cfg.stage if not quick else "quick",
        "config": cfg.to_dict(),
        "tracker_kwargs": kwargs["tracker_kwargs"],
        "prob_peak": peaks,
        "train_bars": int(len(train)),
        "regularized": bool(models["conditioned"].crta_meta.get("faithfulness_regularizer")),
        "collapse_log": {
            "conditioned": models["conditioned"].collapse_log[-5:],
            "blind": models["blind"].collapse_log[-5:],
        },
        "segments": report,
        "seconds": time.time() - t0,
    }
    (out_dir / "eval.json").write_text(json.dumps(_jsonable(payload), indent=2))
    return payload


def _mean_std(values) -> dict:
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
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


def aggregate(parent: Path) -> dict:
    paths = sorted(parent.glob("seed_*/eval.json"))
    if not paths:
        raise FileNotFoundError(f"no seed_*/eval.json under {parent}")
    rows = [json.loads(path.read_text()) for path in paths]
    segments = sorted({name for row in rows for name in row["segments"]})
    summary = {"seeds": [row["seed"] for row in rows], "n_seeds": len(rows), "segments": {}}
    fields = (
        "sharpe",
        "sortino",
        "total_return",
        "max_drawdown",
        "mean_position",
        "position_std",
        "extreme_share",
        "turnover",
    )
    for name in segments:
        block = {}
        for side in ("conditioned", "blind", "buy_hold"):
            block[side] = {
                field: _mean_std(
                    [
                        row["segments"][name][side][field]
                        for row in rows
                        if name in row["segments"] and "skipped" not in row["segments"][name]
                    ]
                )
                for field in fields
            }
        usable = [
            row["segments"][name]
            for row in rows
            if name in row["segments"] and "skipped" not in row["segments"][name]
        ]
        block["ablation_pass_rate"] = _mean_std([item["ablation_pass_rate"] for item in usable])
        block["faithfulness_rate"] = _mean_std([item["faithfulness_rate"] for item in usable])
        block["cond_beats_blind_share"] = float(np.mean([item["cond_beats_blind"] for item in usable])) if usable else None
        block["gap_log"] = _mean_std([item["gap_cond_minus_blind"]["gap_log"] for item in usable])
        block["regularized"] = any(row.get("regularized") for row in rows)
        summary["segments"][name] = block
    (parent / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    flat = []
    for name, block in summary["segments"].items():
        for side in ("conditioned", "blind"):
            flat.append(
                {
                    "segment": name,
                    "side": side,
                    **{f"{field}_mean": block[side][field]["mean"] for field in fields},
                    **{f"{field}_std": block[side][field]["std"] for field in fields},
                    "ablation_mean": block["ablation_pass_rate"]["mean"],
                    "faithfulness_mean": block["faithfulness_rate"]["mean"],
                    "cond_beats_blind_share": block["cond_beats_blind_share"],
                }
            )
    pd.DataFrame(flat).to_csv(parent / "summary.csv", index=False)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train and score twin agents for one config.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--asset", default=None)
    parser.add_argument("--tf", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--aggregate", type=Path, default=None)
    parser.add_argument("--prepare-labels", action="store_true")
    args = parser.parse_args(argv)
    if args.aggregate is not None:
        print(json.dumps(_jsonable(aggregate(args.aggregate)), indent=2))
        return 0
    if args.prepare_labels:
        cfg = ExperimentConfig.load(args.config).with_overrides(tracker="v2")
        for item in args.overrides:
            key, value = item.split("=", 1)
            cfg = cfg.with_overrides(**{key: _parse_scalar(value)})
        asset = args.asset or cfg.asset
        timeframe = args.tf or cfg.timeframe
        timeframe_frame(cfg, asset, timeframe)
        print(f"labels ready {asset} {timeframe}", flush=True)
        return 0
    cfg = ExperimentConfig.load(args.config)
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item}")
        key, value = item.split("=", 1)
        cfg = cfg.with_overrides(**{key: _parse_scalar(value)})
    if args.asset:
        cfg = cfg.with_overrides(asset=args.asset)
    if args.tf:
        cfg = cfg.with_overrides(timeframe=args.tf)
    if args.stage:
        cfg = cfg.with_overrides(stage=args.stage)
    seeds = [int(part) for part in str(args.seeds).split(",") if part]
    out = args.out
    if out is None:
        tag = "quick" if args.quick else f"{cfg.asset.lower()}_{cfg.timeframe}_{cfg.stage}"
        out = ROOT / "paper" / "results" / "v2" / tag
    for seed in seeds:
        run_seed(cfg, seed, out / f"seed_{seed}", args.quick, cfg.asset, cfg.timeframe)
    if len(seeds) >= 1:
        aggregate(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

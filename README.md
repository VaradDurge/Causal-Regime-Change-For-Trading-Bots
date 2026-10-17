# Regime Audit

A single-asset research desk. A sticky hidden Markov model labels BTC/USDT 4h bars as bull, bear, calm, or crisis. Two PPO agents share one trainer; one of them also sees the regime probabilities. Ablation, a counterfactual regime, and a crisis-probability sweep then test whether that input actually moves the position.

This is a historical backtest. It does not send orders.

## Setup

```bash
uv sync
uv run python -m crta.pipeline --probe   # regimes + probe policy, no PPO
uv run python -m crta.pipeline           # train both agents, then verify
uv run streamlit run src/crta/demo.py
```

`--synthetic` skips Binance. The console badge says `probe` or `trained PPO`, and the header says when the series is synthetic.

## What you are looking at

The ribbon is a dwell-held label. A new label replaces the old one only after five days (30 bars of 4h) in a row, so a single day cannot flip it and a label already shown is never rewritten. The policy sees the filtered probabilities, not that label: the probability at bar `t` uses candles up to `t` only. The wealth curves are out-of-fold only: each test block is scored by a model fit on the bars before it.

The conditioned and blind agents use the same seed, learning rate, network width, and number of passes over the training window (ten). The first linear layer is wider by four inputs on the conditioned agent. That width is the experiment.

Faithfulness on a bar means all three of these:

- Ablation moves the position by at least 0.15 when the regime input is zeroed.
- Forcing a more defensive regime cuts the position by at least 0.10, and forcing a calmer one raises it.
- An 11-point sweep of crisis probability is non-increasing, with at most one uptick.

## Current result

Binance BTC/USDT, 4h, out-of-fold only, ten training passes per fold:

- Faithfulness: **33.5%** of bars pass all three checks.
- Sharpe, regime-conditioned: **0.41** [−0.41, 1.20]
- Sharpe, regime-blind: **−1.12** [−1.96, −0.27]
- Sharpe, buy and hold: **0.35** [−0.51, 1.24]

On 9 Nov 2022 the label is crisis and the trained position is +1.00. Ablation, the counterfactual, and the sensitivity sweep all fail on that bar. The conditioned agent beats the blind twin. It does not beat buy-and-hold by a margin the bootstrap can separate. That is the result to show.

## Five minutes with a mentor

1. Open the console. Say it is a backtest of BTC/USDT 4-hour bars, not a live trader.
2. Click **Open the 2022 crash**. The ribbon should sit on crisis through the drawdown. Buy-and-hold falls. Read whether the conditioned position actually cut risk — if it did not, say so.
3. Read the three lines on the card: ablation, counterfactual, sensitivity. One bar is a story. The faithfulness percent is the result.
4. Drag **Crisis probability**. The position under the card should fall as the probability rises, on a bar that passed the sensitivity check.
5. Set **Tell the agent the regime is** to Calm. On a crash bar, a faithful policy gets longer. A policy that ignores the regime does not move.
6. End on the Sharpe intervals in the header. The bracket is a bootstrap, not a point estimate. Losing to buy-and-hold is a valid result. The question is whether the regime input is load-bearing, and whether the gap shows up inside a regime rather than only in the average.

## What to post

Export the four PNGs with **Export figures**, or use the files written by the pipeline:

- `outputs/figures/regime_ribbon.png` — price with the regime strip
- `outputs/figures/equity.png` — conditioned, blind, buy-and-hold
- `outputs/figures/sensitivity.png` — position against crisis probability
- `outputs/figures/audit_card.png` — one bar, three verdicts

Record the crisis slider and the force-regime control on a 2022 crash bar. Suggested caption:

> Regime-conditioned BTC agent, 4h bars, walk-forward only. The check is not the equity curve. Ablation, a forced regime, and a crisis sweep have to agree that the regime input moves the position. Historical research. Not a trading signal.

Do not describe the probe policy as a trained agent. The badge on the screen is the one to repeat.

## Plug in a bot

Put a file in `bots/`. It needs one function. The console lists it. Nothing is retrained.

```python
def act(features, regime, position):
    # features: log_ret, vol_20, trend, rsi, macd
    # regime: bull, bear, calm, crisis. The checks replace this.
    # position: your previous bet. Return the next bet in [-1, 1].
    return 0.0
```

`bots/example.py` is a copy of the hand rule. A `.zip` from this desk’s own trainer also loads. A model with a different input vector goes inside `act()`, where you build the vector it expects.

```bash
uv run python -m crta.plug bots/example.py
```

Open the console, choose the file under **Bot under test**, and read the card. **Score this bot on every bar** replaces the faithfulness number and the wealth line for that session.

## Tests

```bash
uv run pytest -q
```

The leakage test checks that a truncated history produces the same features as the prefix of the full history. The verification tests check that a regime-following probe passes and that an ignoring policy and an inverted policy fail.

## Code map

The package is `src/crta`.

- `data.py` builds causal features. Column `t` uses rows at or before `t` only. The candle cache is written to a temporary file and renamed over the old one. A candle still forming is dropped.
- `regime.py` is the walk-forward 4-state Gaussian HMM. `track` and `track_v2` name the burn-in by moments and match later refits by diagonal Wasserstein distance on burn-in-scaled features. Probabilities are filtered (bar `t` uses rows up to `t`), and each window's filter is warmed on the six strides before it. 4h defaults are burn 1080 (fold 0's training window), stride 42, dwell 30. `track_v2` adds template rebirth and, by default, a semantic guard that re-names templates by their moments after every update so "crisis" stays the low-return, high-volatility state. The 4h pipeline runs `track_v2` on `HMM_COLS_DD`, the three HMM features plus the distance below the 90-day high. `LEGACY_V1` and `LEGACY_V2` reproduce the published, smoothed labels for audits only; `TIER1_V2` is the labeller before the guard.
- `dating.py` is hindsight 20% bull/bear dating, the yardstick for label quality. `baselines.py` is buy-and-hold, long/flat 2/4/8-week momentum and the regime rule `1 - p(bear) - p(crisis)`. `predictability.py` is the walk-forward forecast test.
- `agent.py` is the Gymnasium `TradeEnv` and the two PPO twins. `include_regime` is the only architectural difference: 10 inputs with the four probabilities, 6 without. `train_ppo` is the shared trainer.
- `verify.py` is ablation, counterfactual, and sensitivity. `audit_bar` is the one-hot verifier. `audit_many_v2` is the in-distribution verifier. `audit_many_v3` ablates to the training-mean probability vector and takes its risk order from `regime_order`, which ranks regimes by the mean-variance position mu/sigma^2 on the training fold.
- `eval.py` is the walk-forward split and the return statistics; its intervals use a circular block bootstrap. `stats.py` adds the deflated Sharpe ratio and the median and interquartile range across seeds. `pipeline.py` is the 4h probe and the trained run.
- `train_ppo(device=...)` and `CRTA_DEVICE` select the PPO device. CPU is the default: on an RTX 4060 laptop GPU one 4h twin trained about twice as slowly as on the CPU, because the 64x64 policy is too small to pay for the transfers. `uv sync` reinstalls the CPU build of torch.
- `paper/scripts/timeframe_study.py` trains the native BTC and ETH twins. Validation trains before 2024-09-22 16:00 UTC and scores until 2025-09-22. The final stage trains before 2025-09-22 16:00 UTC, then scores the holdout year and the live window from 2026-09-22 16:00 UTC. The native recipe is window 2000, reward scale 100, entropy 0.01, 500k steps, 2 threads.
- `paper/scripts/run_experiment.py` is the multi-seed runner. `paper/scripts/run_grid.py` is the validation grid.
- `demo.py` is the Streamlit console. It only steps on closed candles.

`configs/default.yaml` is the native recipe with every new flag at the old behavior. `configs/v2.yaml` turns the new options on and is not a frozen selection. `configs/final.yaml` is written from the validation grid only.

## Reproducing v2 results

From the repository root, with the package installed:

```bash
python -m pytest -q
python -m crta.pipeline --probe --synthetic
python paper/scripts/run_experiment.py --quick --out /tmp/crta-quick
python paper/scripts/run_grid.py --workers 2
python paper/scripts/run_grid.py --select
```

The grid trains both twins for 500k steps on the validation split only (BTC and ETH, 1h and 15m, seed 42). `configs/final.yaml` is frozen from that grid before holdout. In this run the criterion was not met and the baseline had the lowest deficit, so the frozen file is the native recipe. Holdout and live then use five seeds, 42 through 46:

```bash
python paper/scripts/run_experiment.py --config configs/final.yaml --asset BTC --tf 1h --stage final --seeds 42,43,44,45,46 --out paper/results/v2/final/btc_1h
```

Repeat for BTC 15m, ETH 1h, and ETH 15m. `--quick` is a smoke test, not a reported number.

The published timeframe JSON under `paper/results/tf/` is the baseline record in `paper/results/baseline/manifest.json`. New runs go to `paper/results/v2/` and do not overwrite those files.

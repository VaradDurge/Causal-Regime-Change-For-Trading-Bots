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

The ribbon is a dwell-smoothed label. Runs shorter than five days (30 bars of 4h) are merged into the previous label, so a single day cannot flip it. The policy sees the raw probabilities, not that label. The wealth curves are out-of-fold only: each test block is scored by a model fit on the bars before it.

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
- `regime.py` is the walk-forward 4-state Gaussian HMM. `track(X, burn, stride, eta, seed, dwell)` names the burn-in by moments and matches later refits by diagonal Wasserstein distance. 4h defaults are burn 252, stride 42, dwell 30. `track_v2` is off unless a config asks for it.
- `agent.py` is the Gymnasium `TradeEnv` and the two PPO twins. `include_regime` is the only architectural difference: 10 inputs with the four probabilities, 6 without. `train_ppo` is the shared trainer.
- `verify.py` is ablation, counterfactual, and sensitivity. `audit_bar` is the one-hot verifier. `audit_many_v2` is the in-distribution verifier.
- `eval.py` is the walk-forward split and the return statistics. `pipeline.py` is the 4h probe and the trained run.
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

The grid trains both twins for 500k steps on the validation split only (BTC and ETH, 1h and 15m, seed 42). Holdout and live are a later step, five seeds 42..46, and they use `configs/final.yaml` after it is frozen. `--quick` is a smoke test, not a reported number.

The published timeframe JSON under `paper/results/tf/` is the baseline record in `paper/results/baseline/manifest.json`. New runs go to `paper/results/v2/` and do not overwrite those files.

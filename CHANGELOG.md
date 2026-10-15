# Changelog

Defaults reproduce the trainer and the tracker as of `af843a5`. Every new flag is off, or set to the previous behavior, unless a YAML file says otherwise.

## Baseline

`paper/results/tf/*_{validate,final}.json` from that commit is the reproduced timeframe baseline. The files load, and their ablation rates match the paper (about 1.1% and 0.8% on the 15-minute tapes, 26.1% and 7.5% on the 1-hour tapes). The synthetic probe (`python -m crta.pipeline --probe --synthetic`) was re-run. The eight native 500k twins were not retrained for that record. An in-environment control cell of `configs/default.yaml` is part of the validation grid so factor comparisons are not mixed with the earlier library build.

## 1. Reward mode

`reward_mode` defaults to `raw`: `a_t * r_{t+1} - c |a_t - a_{t-1}|` with `c = 10^{-3}`.

`excess_bh` subtracts the buy-and-hold return, `(a_t - 1) * r_{t+1} - c |Δa|`. At scale 1 this is the raw reward minus `r_{t+1}`.

`diff_sharpe` is the Moody and Saffell differential Sharpe on the scaled portfolio return `reward_scale * (a_t r_{t+1} - c |Δa|)`. The EMA moments `A` and `B` reset at every episode (`diff_sharpe_eta` default 0.01). The ratio is homogeneous of degree 0, so the scale cancels in the differential and is applied only inside the return. The first steps are 0 until the variance estimate is positive.

`reward_scale` multiplies return and cost in `raw` and `excess_bh`.

## 2. Anti-collapse

`pos_extreme_penalty` (default 0) subtracts `λ * max(0, |a_t| - 0.95)` each bar, after scaling.

`pos_variance_bonus` (default 0) adds `λ * min(std of the last K positions, 0.5)` every `variance_every` bars (default 256) and on the leftover at the end of the episode. The `min(., 0.5)` cap is the bound. It is not multiplied by `reward_scale`.

`ent_coef` is the PPO entropy coefficient. `ent_coef_start: null` keeps it constant. A number decays linearly from that start to `ent_coef` over the training budget.

Every `collapse_log_every` steps (default 10000) the trainer prints mean position, position standard deviation, and the share of steps with `|a| > 0.95`. Three logs in a row with standard deviation below 0.02 print a warning. Training does not stop.

## 3. Window sampling

`window_sampling` defaults to `uniform`, the previous random start.

`regime_balanced` weights each start by `1 + β * share`, where the share is the fraction of bars in the window whose argmax probability is bear or crisis, using the labels already on the training rows. `regime_balance_beta` defaults to 3.

`return_balanced` splits starts into terciles of the window's summed log return and draws the three terciles with equal probability.

`sign_flip_prob` defaults to 0. When it is positive, that fraction of episodes negates log returns, rebuilds the price path, recomputes features, and re-runs the causal tracker. The flipped frame is cached and used by both twins, so the blind twin is not left on the original series. It is off by default because the re-track is a full causal pass. The quick runner forces it off.

A window is one contiguous slice. It is not used when it is at least as long as the series, which was already the rule.

## 4. Regime input

`regime_perturb_prob` defaults to 0. With that probability an episode replaces each `p_t` by a draw from the empirical training-row probabilities, resampling once if the draw matches the current row. Rewards stay on the real returns. Both twins consume the same draws so the next window stays aligned. The draws are not one-hot.

`faithfulness_regularizer` defaults to false. When it is on, `FaithfulPPO` (Stable-Baselines3 2.9 `PPO.train` plus one term) adds a hinge on the conditioned twin only: the mean action at the nearest training crisis vector must not exceed the realised action, and the nearest training bull vector must not fall below it. A missing regime skips that side. Any run with the flag on is marked `regularized` and is not an independent audit. The grid leaves it off.

`prob_temperature` defaults to 1 and `prob_floor` to 0, which leave probabilities unchanged. Otherwise `p = softmax(log p / T)` and then `p = (1-ε)p + ε/4`. The same smoothing is applied at training and at scoring. Mean max probability before and after is logged.

## 5. Tracker v2

`tracker: v1` is the unchanged `track`. `track_v2` adds:

- Birth: a template with no member for `tracker_death_after` refits (default 20) is re-seeded from the component farthest in diagonal W2 from the other templates, or from the moment-sorted component of that name when that distance is about 0. The event records refit index, bar, and date. The caller re-matches once. Template updates still use only the fit on rows before the window and that window's filtered responsibilities.
- `tracker_matching`: `nearest` (default, same collisions as v1), `hungarian`, or `hungarian_with_threshold`. The threshold variant keeps a Hungarian pair only when its cost is within `tracker_cost_quantile` (default 0.95) of previously accepted costs. Until 8 accepted costs exist, every pair is kept.
- Filtered posteriors from the forward algorithm in log space (`startprob_`, `transmat_`, diagonal Gaussian log likelihoods). Bar t does not use later bars. Reported labels use an online dwell that only looks backward. v1 still uses forward-backward probabilities and the post-hoc dwell.

Passing `diagnostics` to `track` only records costs. It does not change labels.

## 6. Verifier v2

`verifier: legacy` is the one-hot counterfactual. `v2` replaces those one-hots with the nearest training-set probability vector (L1) whose argmax is crisis or bull, or a random vector among the 20 nearest when `cf_pick: top20`. A regime with no training vectors is `untestable` and does not fail the bar. Sensitivity still allows at most one uptick and now also requires `max(y) - min(y) >= delta_sens` (default 0.05). Ablation stays at `delta_ab = 0.15`. Reports include per-check rates, the combined rate, per-regime rates, the median absolute ablation effect, and the share of bars where argmax `p_t` equals the dwell label.

The selection ablation rate is the legacy ablation, so the grid stays comparable to the published baseline.

## 7. Multi-seed runner

`paper/scripts/run_experiment.py` trains both twins with the same reward, window sampler, entropy, seed, and budget. The tracker seed stays 42. Each seed writes `seed_<n>/` only. `--quick` is a synthetic smoke test (256 steps) and is not a result. `--aggregate` writes mean and standard deviation across seeds. The conditioned-minus-blind gap is a paired circular block bootstrap with block length `ceil(sqrt(n))` and 2000 resamples. "Beats" means the summed log-return gap is positive.

The timeframe protocol is one walk-forward split (train before the cut, score after). It is not the four-fold 4-hour scheme.

## 8. Selection

Fixed before holdout, in `src/crta/select.py`. On each validation tape both twins need position standard deviation above 0.05 and an `|a| > 0.95` share below 80%, and the conditioned ablation rate must be at least 10%. The deficit is the sum of relative shortfalls. Returns are ignored. A tie would be broken by lower deficit, not by return.

The one-factor grid did not meet that criterion on any tape: every twin still spent more than 90% of validation bars at `|a| > 0.95`. The baseline had the lowest mean deficit (0.985). No other factor reduced it, so the pre-registered combination stack was empty and `configs/final.yaml` keeps the native recipe. Holdout and live use that frozen file and were not used to choose it.

## 10. Console

Saved twins store `reward_mode`, `prob_temperature`, `prob_floor`, and `tracker` on the model and in a `.meta.json` sidecar. Loading a zip applies that smoothing. A `*_conditioned.zip` dropped in `bots/` with a sibling `*_blind.zip` is the live pair. Closed candles and the atomic cache rename are unchanged.

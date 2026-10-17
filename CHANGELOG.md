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

## 11. Validation saturation and the five-seed four-hour rerun

No zip in the tree carried metadata for a validation-split native twin (train rows before 2024-09-22 16:00 UTC). `paper/results/tf/val_ent005/` is the rejected entropy-0.05 run and was not used. The final-stage zips were not used. Seed 42 was retrained with the frozen native recipe and labelled `retrained_seed42_validation`.

The saturation CSV is canonical at `results/saturation_diagnostic.csv` (the path named for the check). A copy is at `paper/results/saturation_diagnostic.csv`. The eight decision numbers are ordered BTC then ETH, 1h then 15m, conditioned then blind. `mean_abs_delta_position` is the mean of consecutive executed-position changes, with no fictional start at 0. The verdict file is `results/saturation_decision.txt`.

The four-hour candle cache `data/raw/btc_usdt_4h.parquet` is gitignored and was not in the checkout. The rerun uses rows of `paper/results/live/btc_usdt_4h.parquet` at or before 2026-09-22 12:00 UTC and checks them against `paper/results/tracker.npz`. No candle was downloaded. That live file's 12:00 close does not reproduce the archived run's last `next_ret`, so that one close is restored from `paper/results/run/bars.parquet` before features are built. Every other out-of-fold close and next return already matched.

Moment-sorted is a third labelling on the same v1 refits (`lock_states` on unscaled means). Templates still update under the Wasserstein assignment. It is not a separate per-fold HMM.

Sharpe and Sortino intervals use a circular block bootstrap with block length `ceil(sqrt(n))` and 1000 resamples (seed 42). `src/crta/eval.py` uses 1000 i.i.d. resamples. The 2000-resample circular bootstrap in the repo is the native-twin gap, not the four-hour table. Each PPO call in the rerun passes `threads=1`. `train_ppo`'s default is unchanged. The budget stays `max(8192, 10 * n_train)`, raw reward, scale 1, entropy 0, no window.

Canonical tables: `paper/results/tables_4h/`. Copy: `results/tables/`. Table I is settings and folds (paper `tab:hyper` and `tab:folds`). Table III is label statistics after burn-in (paper `tab:labels`). Table IV is out-of-fold performance (paper `tab:perf`). Table V is per-regime (paper `tab:regime`). `paper/main.tex` was not edited.

## 12. Action mode

The saturation rule called clipping the cause (8 of 8 tapes), so `action_mode` was added. The default is `gaussian_clip`, the previous Gaussian policy with the executed action clipped to [-1, 1]. Omitting the flag keeps that path. `tanh_env` lets the environment execute `tanh(u)`. `discrete5` is `Discrete(5)` on `{-1, -0.5, 0, 0.5, 1}`. The verifier audits the expected level under that categorical distribution, not a sampled action.

The screen used the validation split only. `gaussian_clip` reused the seed-42 validation weights. `tanh_env` and `discrete5` were trained at seed 42 with the same native recipe. A mode had to pass on all four tapes: both twins with position standard deviation above 0.05, an `|a| > 0.95` share below 0.80, and a conditioned ablation rate of at least 10%. The pre-registered tie-break was the highest minimum ablation across the four tapes, then the minimum twin position standard deviation, then `tanh_env`, `discrete5`, `gaussian_clip`. Returns were not an input.

No mode passed. `gaussian_clip` and `tanh_env` missed the extreme-share bar on every tape. `discrete5` passed BTC 15m, ETH 1h, and ETH 15m, and failed BTC 1h because the conditioned twin was constant. The frozen file is `results/action_mode_choice.txt` and it says `none`. Holdout and live were not scored.

## Mean-variance reward and limit-aware audit

- `reward_mode: mean_variance` with `risk_aversion` (default reward stays `raw`, `risk_aversion` defaults to 0). The reward is `a*r - c*|a - a_prev| - risk_aversion * a^2 * vol_20^2`, scaled by `reward_scale`. `vol_20` is the volatility already known at the decision bar. A reward linear in the position is maximised at a rail, so this mode gives an interior optimum `a* = E[r] / (2 * risk_aversion * vol^2)` instead of forcing variety.
- `audit_many(..., bound_aware=True)` reports bars where the position limit makes the required counterfactual move impossible (`cf_untestable`) and the all-check rate on the rest (`faith_rate_testable`). `passed` keeps the original rule, and the original rate is still the headline.
- `paper/scripts/bound_aware_audit.py` re-scores the published four-hour run from its saved pass flags: 7,040 of 11,430 bars are untestable, and the all-check rate on the 4,390 testable bars is 0.25% (11 bars overall).

## Causal regime inputs (breaks the tracker default above)

The published tracker decoded each 42-bar window with forward-backward smoothing (`predict_proba`), so the probability at bar `t` had already seen later bars of its window, including `next_ret`, the return the agent is paid on. Defaults now change on purpose; the old behavior is still reachable, for audits only.

- `track`, `track_v2`, `label_span` and `decode` return filtered probabilities and online-dwell labels by default. `posterior="smoothed"` keeps the old decoder. Each window's filter first runs over `WARM_STRIDES * stride` earlier rows, so it does not restart from the fit's start vector.
- W2 matching compares components in units of the burn-in standard deviation (`scale="burn"`), so per-state mean returns are no longer drowned by trend. `scale="raw"` is the old rule.
- `BURN` is 1,080 bars, fold 0's training window, so naming the templates costs no test bar.
- `LEGACY_V1` and `LEGACY_V2` hold the published settings. `rerun_4h.py`, `tracker_diagnostics.py` and `tracker_one_to_one.py` pass them, so they still reproduce the published numbers.
- `crta.pipeline` labels with `track_v2(burn=min_train)` and scores the baselines next to the twins.
- `timeframe_study.py` and `run_experiment.py` key their label caches on `LABELLER_VERSION`, so labels cached before this change are rebuilt.
- New: `crta.dating` (hindsight 20% bull/bear dating and label quality), `crta.baselines`, `crta.predictability`, `crta.leakage`.
- New scripts: `leakage_audit.py` (published weights under published, one-bar-lagged and filtered probabilities), `label_quality.py`, `predictability_study.py`, and `causal_4h.py`, the five-seed walk-forward study that replaces the published four-hour run. Outputs go to `paper/results/causal/` and `paper/results/causal_4h/`.

## Tier 2: verifier v3, regularizer, statistics and labels

- Verifier v3 (`audit_many_v3`): ablation replaces `p_t` by the training-mean probability vector instead of zeros, which the policy never saw. The risk order is fixed from the training fold by `regime_order`, the mean-variance position mu/sigma^2 of the next return per regime, so the counterfactual and the sweep test what a risk-aware reward pays for. A bar is faithful only if it is testable. `reference_weights` builds faithful, ignoring and inverted reference policies from the same order. Legacy and v2 are unchanged.
- `FaithfulPPO` applies the hinge to the raw Gaussian mean (or tanh of it for `tanh_env`) instead of the mean clamped to [-1, 1], so a violation beyond the limits still has a gradient.
- `eval.bootstrap_ci` is a circular block bootstrap (block `ceil(sqrt(n))`; `block=1` is the old i.i.d. one). `crta.stats` adds the probabilistic and deflated Sharpe ratios and the median and IQR across seeds.
- Labels: `track_v2(semantic_guard=True)` is the new default. Templates are re-named by their moments after every rebirth and update, and a rebirth takes the component the moments name, so the crisis template can no longer drift calmer than calm. `crta.data.drawdown` and `HMM_COLS_DD` add the distance below the 90-day high as an HMM input. `label_selection.py` chose the labeller on fold 1's training span by a rule written before the run; the guard with the drawdown input was the only candidate whose names matched their moments, and the pipeline now uses it. `LABELLER_VERSION` is `filtered-scaled-guard-1`.
- The filter warm-up listed in Tier 2 was already part of the causal fix.
- `causal_4h.py --labeller guard_dd` (the default) writes the canonical study to `paper/results/causal_4h_guard_dd/`, with verifier v3, the v3 reference check, phase splits, seed medians and IQRs, and deflated Sharpe ratios. `--labeller tier1` re-tables `paper/results/causal_4h/`.
- `train_ppo` takes `device`, and `rerun_4h.walk_forward_seed` reads `CRTA_DEVICE`. CPU stays the default because it was about twice as fast as the GPU for this policy.

## Tier 3: daily long/flat task, monotone head, alternative labellers, native rerun

- `crta.daily`: one decision per daily close (the 4h bar opening 20:00 UTC), long/flat positions in [0, 1], daily features plus 2/4/8-week momentum standardised on each training fold, and the reward `a*r - c|a - a_prev| - lambda a^2 sigma^2` with the 20-day volatility known at the decision. Daily folds cut on the same calendar instants as the 4h folds. The regime block is the 4h labeller's filtered probability at the decision bar.
- `crta.policies.PositionPolicy`: actor-critic heads `clip01`, `tanh01` (tanh with the mean-variance reward, never tested before) and `beta` (Beta policy with a learned concentration). `monotone=True` builds z = base(features, position) + sum_g w_g p_g, where the base network never sees the regime block and the weights fall along the training fold's risk order, so the head is faithful by construction.
- `crta.jump` (statistical jump model with online forward inference) and `crta.wkmeans` (Wasserstein k-means on sorted rolling return windows) are causal walk-forward labellers named by moment sorting. `label_alternatives.py` picks each family's setting on the early span with the rule of `label_selection.py`.
- `daily_study.py`: a 3-head x 3-risk-aversion x 3-twin x 5-seed grid on the guard_dd labels, a cell chosen on folds 0-1 by a non-return bar fixed in advance, the report on folds 2-3 beside the baselines with seed medians, IQRs and deflated Sharpe ratios, and the chosen cell rerun with the jump-model and Wasserstein labels. Verifier v3 thresholds are halved for [0, 1] positions.
- `native_filtered.py`: the five-seed native twins rerun under the frozen recipe with filtered probabilities. Rebuilding the labels overwrites the cached `paper/results/tf/*_labels.parquet`; the smoothed versions remain in git history.

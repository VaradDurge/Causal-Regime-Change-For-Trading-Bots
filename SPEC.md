# Project Specification
## Causal Regime-Conditioned Trading Agent — A Verifiable Alternative to Black-Box Regime-Aware RL

*Companion files: `reading_list.md` (papers behind each design choice), `project_architecture.html` (visual pipeline diagram). This document is self-contained and can be handed to Claude Code as the planning input for the repository.*

---

## 1. Problem Statement

Regime-aware reinforcement learning for trading — agents that detect market conditions (bull/bear/calm/crisis) and adapt their policy accordingly — is an active research area with multiple 2025–2026 papers. However, the published work in this space consistently shares a weakness its own authors acknowledge: the link between the detected regime and the agent's resulting behavior is asserted, not verified. A model can report "I reduced exposure because volatility spiked" without any check on whether that claim is actually true of the model's internal decision process, versus being a plausible-sounding post-hoc story.

This matters beyond academic tidiness. Financial regulators (MiFID II in the EU, SEC guidance in the US) increasingly require that algorithmic trading decisions be explainable and auditable. An explanation that merely *sounds* right without being *provably* tied to the model's actual computation does not satisfy that bar — and general ML research (see `reading_list.md`, Section 2) has repeatedly shown that popular explanation methods can produce plausible output that is completely disconnected from what a model actually learned.

**The gap this project targets:** build a regime-conditioned trading agent whose regime-to-decision link is not just interpretable by construction, but is actively tested for causal faithfulness using multiple independent verification methods — and report how often, and how convincingly, it passes.

## 2. Research Question

> Does explicitly verifying (not just asserting) that a regime signal has a genuine causal effect on a trading agent's behavior — via ablation, counterfactual perturbation, and sensitivity analysis — produce a system that is both more trustworthy and more robust across regime transitions than a standard black-box regime-switching baseline?

Secondary question: how often does a plausible-looking regime-conditioned agent actually fail one or more of these faithfulness checks?

## 3. Scope

### In scope
- A regime detection module using an interpretable statistical model (HMM-based), not a black-box classifier.
- A single-asset (crypto) RL trading agent, built on an existing open-source RL-for-finance framework.
- A causal verification suite: ablation testing, counterfactual perturbation, and policy sensitivity curves.
- Walk-forward backtesting with regime-segmented, statistically rigorous evaluation.
- Open-source repository + a paper targeting ACM ICAIF / ALGODEFI (2027 cycle).

### Out of scope (explicitly — do not build these)
- Real HFT infrastructure: no colocation, no sub-millisecond execution, no order-book-level market making. This is a decision-frequency RL agent (minute/hour/day bars), not a latency-sensitive system.
- Live trading with real capital, or any brokerage integration. All experiments run on historical data in backtest.
- Multi-asset portfolio optimization (single asset keeps the verification methodology the primary focus, not portfolio construction).
- Production deployment concerns (monitoring, live risk limits, failover). This is a research prototype, not a trading product.

## 4. System Architecture

### 4.1 Data Layer
**Purpose:** Provide clean, causal (no-lookahead) historical market data with enough regime diversity to make the research question testable.
- Ingest crypto OHLCV and/or order-book data.
- Compute derived features: returns, rolling volatility (multiple windows), trend indicators (e.g., moving average crossovers, RSI, MACD).
- Enforce strict walk-forward splitting: every experiment must be structured so that no model ever sees data from after the point it is making a decision about.

### 4.2 Regime Detection Layer
**Purpose:** Produce an interpretable "market weather" signal.
- Default model: Gaussian HMM over rolling volatility/return/trend features, with a **minimum-dwell-time (sticky) constraint** so regimes cannot flip on single-day noise.
- Stretch goal: replace/augment with a causal Wasserstein-distance-based identity-tracking scheme (see `reading_list.md` §1) to keep regime identity stable across rolling re-fits — this is a known open problem (plain HMMs relabel regimes arbitrarily across windows) that the stretch goal directly addresses.
- Output per timestep: a regime label and a confidence/probability score. Must be computed only from data available up to that timestep.

### 4.3 RL Trading Agent
**Purpose:** Make position-sizing decisions; provide both a regime-aware and a regime-blind variant for comparison.
- Base algorithm: policy-gradient RL (PPO or SAC) via an existing framework (see Section 5) rather than building an RL library from scratch.
- Observation space: price/volume-derived features + the regime probability from 4.2 as an explicit additional input.
- Action space: continuous or discretized position size (e.g., -1 to +1, or short/flat/long weightings).
- **Required baseline twin:** an identical agent architecture and training procedure with the regime feature removed. This is not optional — it is the comparison point for every result in Section 4.4 and 4.5.

### 4.4 Causal Verification Layer — *the core contribution*
**Purpose:** Prove the regime signal causally drives the agent's behavior, rather than asserting it.
- **Ablation:** re-run the trained agent's decision process with the regime input zeroed out / removed, holding everything else fixed. Measure the magnitude of the resulting change in action. A near-zero change means the signal is not actually load-bearing.
- **Counterfactual perturbation:** feed the trained agent synthetic regime readings it did not actually observe (e.g., tell it "calm" during a real crash) and confirm the direction of the behavioral change matches what the stated logic would predict. A wrong-direction response indicates the regime signal is miswired, not just unused.
- **Sensitivity curve:** sweep the regime confidence input continuously from 0 to 1 (not just on/off) and plot the resulting action. A trustworthy signal produces a smooth, monotonic-ish response; a jagged curve with a single sharp threshold suggests memorization of specific training examples rather than genuine sensitivity to the underlying variable.
- **Aggregate faithfulness verdict:** combine the three tests into a per-decision pass/fail, then report the pass rate across the full backtest as a headline result.

### 4.5 Evaluation & Backtesting
**Purpose:** Rigorous, leakage-free performance measurement — the layer most published trading-ML work gets wrong.
- Walk-forward (expanding or rolling window) backtesting only. No single static train/test split.
- Report Sharpe and Sortino ratios with bootstrapped confidence intervals, not bare point estimates.
- Break down performance **by regime**, not only in aggregate — the central empirical claim is that the verified regime-aware agent degrades less during regime transitions/crises than the baseline, which an aggregate number alone cannot show.
- Compare three systems throughout: regime-conditioned + verified agent, regime-blind baseline agent, and buy-and-hold.

### 4.6 Outputs
- Open-source GitHub repository (code, trained model artifacts or training scripts, evaluation notebook).
- Paper draft targeting ACM ICAIF / ALGODEFI 2027 submission cycles.
- Final-year project submission and live demo.

## 5. Recommended Tech Stack

| Layer | Recommendation | Notes |
|---|---|---|
| Language / env | Python 3.10+, venv or conda | Standard for this ecosystem |
| RL framework | **Classic FinRL** (`AI4Finance-Foundation/FinRL`) + **FinRL-Meta** for market environments | Do **not** use FinRL-X / FinRL-Trading — that is AI4Finance's newer production/live-brokerage-oriented stack; classic FinRL is explicitly positioned by its maintainers for education/research prototyping, which fits this project |
| Crypto-specific starting point | `AI4Finance-Foundation/FinRL_Crypto` | An existing crypto-specialized FinRL variant — check it before building the environment from scratch |
| Underlying RL algorithms | Stable-Baselines3 (PPO, SAC) | What FinRL is built on top of |
| Env interface | Gymnasium | Standard RL environment API |
| Data acquisition | `ccxt` (multi-exchange) and/or `python-binance`; `kaggle` package for the LOB dataset | ccxt is the more actively maintained general-purpose option |
| Data wrangling | pandas, numpy | — |
| Technical indicators | `pandas-ta` | Lighter dependency footprint than TA-Lib |
| Regime detection | `hmmlearn` (Gaussian HMM), `scikit-learn` (GMM utilities) | Core of Section 4.2 |
| Wasserstein / optimal transport (stretch goal) | `POT` (Python Optimal Transport) | Needed only if pursuing the causal Wasserstein HMM stretch goal |
| Verification layer | Primarily custom code (numpy/scipy) since the methodology is bespoke; `captum` optional if you want additional PyTorch-native attribution methods for comparison | Ablation/counterfactual/sensitivity logic is project-specific — don't force it into a generic XAI library |
| Statistics | `scipy.stats`, `statsmodels`, `arch` (bootstrap) | For confidence intervals, significance testing |
| Performance metrics | `quantstats` | Sharpe/Sortino/drawdown reporting, standard in the field |
| Visualization | `matplotlib`, `plotly` | Sensitivity curves and regime-segmented performance charts |
| Experiment tracking | `wandb` (optional) or structured CSV/JSON logging | Nice-to-have, not required for scope |
| Testing | `pytest` | Cover the regime detector and verification functions especially — these are your novel contributions and need to be correct |
| Notebook | Jupyter | For the final evaluation/results notebook |

## 6. Data Sources

- **Binance API** (via `ccxt` or `python-binance`) — free historical and live OHLCV/order-book data.
- **Kaggle: high-frequency crypto limit order book dataset** — free, second-level LOB data for cryptocurrency.
- Rationale for crypto over equities: crypto's 2021 bull run → 2022 crash → 2023–25 chop-and-recovery cycle provides real, labeled-by-hindsight regime diversity within a short, easily-obtainable history. Equities would require decades of data to see comparable regime variety.

## 7. Suggested Repository Structure

```
project-root/
├── README.md
├── SPEC.md                        # this document
├── reading_list.md
├── pyproject.toml                 # or requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_binance.py
│   │   ├── fetch_kaggle_lob.py
│   │   └── feature_engineering.py
│   ├── regime/
│   │   ├── sticky_hmm.py
│   │   └── wasserstein_identity.py   # stretch goal
│   ├── agents/
│   │   ├── baseline_agent.py         # regime-blind
│   │   └── regime_conditioned_agent.py
│   ├── verification/
│   │   ├── ablation.py
│   │   ├── counterfactual.py
│   │   └── sensitivity_curve.py
│   ├── evaluation/
│   │   ├── backtester.py             # walk-forward only
│   │   └── metrics.py
│   └── utils/
├── notebooks/
│   └── eval_report.ipynb
├── tests/
└── outputs/
    ├── figures/
    └── results/
```

## 8. Milestones (mapped to a ~9-month final-year timeline)

1. **Months 1–2:** Literature review pass over `reading_list.md`; data pipeline; reproduce a vanilla FinRL baseline on crypto data end-to-end.
2. **Months 3–4:** Build the regime detection module (sticky HMM) and the fusion mechanism feeding it into the agent's observation space.
3. **Months 5–6:** Train and tune both the baseline and regime-conditioned agents; initial experiments.
4. **Month 7:** Implement and run the full causal verification suite (Section 4.4); run walk-forward, regime-segmented evaluation (Section 4.5).
5. **Months 8–9:** Write the paper, incorporate feedback, polish the repo. Target the 2027 ICAIF/ALGODEFI submission cycle; consider an AI4Finance/FinRL contest as an earlier, lower-stakes checkpoint if timing allows.

## 9. Definition of Done

- [ ] Baseline FinRL agent trains and backtests reproducibly on crypto data with a fixed seed.
- [ ] Regime detector outputs labels + confidence that plausibly align with known historical regime changes (sanity-checked against, e.g., the 2022 crash) — not the delivered PASS test itself, just a basic reality check that the detector isn't nonsense.
- [ ] Regime-conditioned agent variant trains successfully using the regime signal as input.
- [ ] All three verification tests (ablation, counterfactual, sensitivity curve) are implemented, unit-tested, and produce a per-decision faithfulness verdict.
- [ ] Walk-forward, regime-segmented evaluation comparing regime-conditioned agent vs. baseline vs. buy-and-hold is complete, with bootstrapped confidence intervals reported (not bare point estimates).
- [ ] Entire pipeline is reproducible from a single entry point / notebook.
- [ ] Repository is public, documented, and citation-ready.

## 10. Key Constraints

- Every backtest must be strictly walk-forward. No result derived from a lookahead-contaminated split is acceptable, even for a preliminary check.
- The baseline (regime-blind) agent must be trained under identical conditions to the regime-conditioned agent — same architecture, same hyperparameter search budget — so any performance difference is attributable to the regime signal, not to an unfair comparison.
- The verification layer is the actual novel contribution of this project. If time runs short, protect Section 4.4 before adding scope elsewhere (e.g., before attempting the Wasserstein stretch goal in 4.2).

## 11. References

See `reading_list.md` for full citations. Most load-bearing for implementation decisions above:
- *Explainable Regime-Aware Investing* (arXiv:2603.04441) — regime model design (4.2)
- *RegimeFolio* (arXiv:2510.14986) — motivates why verification, not just architecture, is the novelty
- *Sanity Checks for Saliency Maps* (arXiv:1810.03292) — the foundational proof that explanations can be unfaithful; motivates the entire verification methodology (4.4)

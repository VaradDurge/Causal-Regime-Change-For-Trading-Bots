# Reading List — Regime-Aware, Causally-Verified RL Trading Agent

Organized by the three pillars of the project: regime detection, explanation faithfulness (general ML), and explainability in finance specifically.

---

## 1. Regime Detection & Regime-Aware Trading Models

**Explainable Regime-Aware Investing**
Amine Boukardagha, Columbia University (2026)
https://arxiv.org/abs/2603.04441
The "causal Wasserstein HMM" paper — rolling Gaussian HMM with regime-identity tracking via 2-Wasserstein distance, built to fix label-switching while staying interpretable. This is your most direct methodological reference for the regime-detection layer.

**Adaptive and Regime-Aware RL for Portfolio Optimization**
arXiv:2509.14385 (2025)
https://arxiv.org/abs/2509.14385
Regime-conditioned RL portfolio agent using macro-informed Gaussian Mixture Model transitions. Good baseline/comparison point — shows the "plain" version of what you're upgrading past.

**RegimeFolio: A Regime-Aware ML System for Sectoral Portfolio Optimization in Dynamic Markets**
arXiv:2510.14986 (2025)
https://arxiv.org/abs/2510.14986
Explicitly critiques prior regime-aware DRL work as "regime agnostic" or "opaque" — this is the paper that most directly names the exact gap your project targets. Good to cite when framing your motivation.

**Behaviorally Informed Deep Reinforcement Learning for Portfolio Optimization with Loss Aversion and Overconfidence (BBAPT)**
Scientific Reports (Nature), 2026
https://www.nature.com/articles/s41598-026-35902-x
Uses TimesNet as a forecasting-guided regime selector combined with behavioral bias modeling. Useful for seeing how others structure a "regime → policy switch" pipeline.

**Clustering Market Regimes using the Wasserstein Distance**
arXiv:2110.11848 (2021)
https://arxiv.org/abs/2110.11848
Foundational paper on using Wasserstein distance (instead of plain HMMs) for regime clustering — useful background before reading the 2026 causal Wasserstein HMM paper above.

**Non-parametric Online Market Regime Detection and Regime Clustering for Multidimensional and Path-Dependent Data Structures**
arXiv:2306.15835 (2023)
https://arxiv.org/abs/2306.15835
Discusses limitations of Wasserstein-based clustering (computational cost, loss of temporal ordering) — good for your limitations/future-work section.

**Geometric Observables for Financial Regime Detection**
arXiv:2605.17117 (2026)
https://arxiv.org/abs/2605.17117
Very recent, more exotic (differential-geometry-based) regime detection. Not essential, but shows how active this sub-field currently is — good for a "related work is still evolving" framing line.

---

## 2. Explanation Faithfulness (General ML — foundational evidence)

**Sanity Checks for Saliency Maps**
Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim — NeurIPS 2018
https://arxiv.org/abs/1810.03292
The landmark paper. Shows several popular explanation methods produce near-identical outputs whether the model is trained or randomized — i.e., proof that plausible-looking explanations can be completely disconnected from the model. This is your core citation for "explanations can lie."

**Is Ignorance Bliss? The Role of Post Hoc Explanation Faithfulness and Alignment in Model Trust in Laypeople and Domain Experts**
arXiv:2312.05690 (2023)
https://arxiv.org/abs/2312.05690
Directly distinguishes "faithfulness" (is it true) from "alignment" (does it sound right to a human) — the exact distinction your whole verification layer is built on.

**An Empirical Study on Explanations in Out-of-Domain Settings**
arXiv:2203.00056 (2022)
https://arxiv.org/abs/2203.00056
Shows post-hoc explanation faithfulness metrics can be misleading outside the training distribution — relevant since markets constantly shift "out of domain" (that's what a regime shift is).

**BEExAI: Benchmark to Evaluate Explainable AI**
arXiv:2407.19897 (2024)
https://arxiv.org/abs/2407.19897
A general benchmark for scoring explanation quality — useful if you want an off-the-shelf way to quantify your faithfulness results rather than building all metrics from scratch.

**A Comprehensive Guide to Explainable AI: From Classical Models to LLMs**
arXiv:2412.00800 (2024)
https://arxiv.org/abs/2412.00800
Good general reference with a clean formal definition of "faithfulness score" via feature perturbation — useful for writing your methodology section precisely.

---

## 3. Explainability in Finance Specifically

**A Comprehensive Review on Financial Explainable AI**
Artificial Intelligence Review (Springer), 2025
https://link.springer.com/article/10.1007/s10462-024-11077-7
Survey of how SHAP/LIME-style post-hoc methods are actually used across finance (credit scoring, fraud, portfolio construction). Confirms finance leans on the same explanation methods shown to fail sanity checks in Section 2 — your key "connects the two literatures" citation.

**Trade-offs in Financial AI: Explainability in a Trilemma with Accuracy and Compliance**
arXiv:2602.01368 (2026)
https://arxiv.org/abs/2602.01368
Very recent — explicitly flags that LLM/AI explanation layers in finance risk being "plausible but not faithful." Strong, current motivation citation.

**How Can I Choose an Explainer? An Application-Grounded Evaluation of Post-Hoc Explanations**
arXiv:2101.08758 (2021)
https://arxiv.org/abs/2101.08758
Empirical XAI evaluation specifically in financial fraud detection. Opens by noting the field lacks solid evaluation methodology — good citation for "this is still an open problem."

**Explainable AI in Finance: Addressing the Needs of Diverse Stakeholders**
CFA Institute Research and Policy Center, 2025
https://rpc.cfainstitute.org/sites/default/files/docs/research-reports/wilson_explainableaiinfinance_online.pdf
Practitioner-oriented (not peer-reviewed academic), but useful for framing the regulatory angle (MiFID II-style compliance pressure) in your intro.

---

## 4. Tooling / Infrastructure (not papers, but load-bearing resources)

**FinRL / FinRL-Meta / AI4Finance Foundation**
https://github.com/AI4Finance-Foundation
Open-source RL-for-finance framework, data loaders, and yearly contests run alongside ACM ICAIF. This is your recommended base to build on rather than starting from scratch — check their GitHub for the current year's contest and starter code before you begin.

---

### Suggested reading order
1. RegimeFolio (motivates the gap) → 2. Sanity Checks for Saliency Maps (proves the failure mode exists) → 3. Explainable Regime-Aware Investing (your regime-model reference) → 4. Trade-offs in Financial AI (ties it back to finance) → then skim the rest as needed while writing.

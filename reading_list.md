# Reading List — Regime-Aware, Causally-Verified RL Trading Agent

This reading list has been updated to include peer-reviewed IEEE/Springer literature directly relevant to regime-aware trading, reinforcement learning for portfolio optimization, XAI, and causal inference in financial decision-making.

---

## 1. Regime Detection & Regime-Aware Trading Models

**HMM-Based Market Regime Detection with RL for Portfolio Management**  
IEEE IDS 2025  
https://www.cloud-conf.net/datasec/2025/proceedings/pdfs/IDS2025-3SVVEmiJ6JbFRviTl4Otnv/966100a067/966100a067.pdf  
Integrates Hidden Markov Models with reinforcement learning to build regime-aware trading strategies. Demonstrates improved Sharpe ratio and risk-adjusted performance relative to conventional RL approaches.

**RegimeFolio: A Regime-Aware ML System for Sectoral Portfolio Optimization**  
IEEE Xplore  
https://ieeexplore.ieee.org/abstract/document/11215751  
Presents a regime-aware ML framework for sectoral portfolio optimization under non-stationary market conditions. This is one of the most relevant works for framing the core research gap.

**Regime Change Detection Using Directional Change Indicators in Financial Markets**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/8363125  
Explores data-driven methods for identifying structural breaks and regime shifts in financial markets, which is directly relevant to adaptive trading models.

**Explainable Machine Learning for Regime-Based Asset Allocation**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/9378332  
Uses explainable ML and regime classification to support asset allocation decisions, combining interpretability with market-state analysis.

**A Causal Inference Method for Stock Trend Forecasting (CISTF)**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/11033554  
Introduces a causal inference framework for stock trend forecasting, highlighting the need for causal reasoning in financial time-series analysis.

---

## 2. Reinforcement Learning for Portfolio Optimization and Trading

**Portfolio Optimization using Deep Reinforcement Learning**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/10744403  
Presents a deep Q-network-based approach to dynamic capital allocation and portfolio optimization.

**Deep Reinforcement Learning for Real-Time Portfolio Optimization**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/11294821  
Discusses real-time adaptive portfolio rebalancing using PPO and DDPG in equity markets.

**Reinforcement Learning Application in Portfolio Optimization: A Review and Research Perspective**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/10620129  
Provides a review of reinforcement learning and deep reinforcement learning for financial decision-making and portfolio optimization.

**Reinforcement Learning-Based Dynamic Portfolio Optimization Under Adversarial Perturbations**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/11291673  
Introduces robust RL strategies to improve portfolio optimization under hostile or perturbed market conditions.

**Research on Portfolio Optimization Based on Deep Reinforcement Learning**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/10125509  
Frames portfolio trading as a Markov decision process and applies DRL for asset-weight management and diversification.

**Research on the Implementation and Effectiveness Evaluation of Reinforcement Learning in A-Share Markets**  
Springer  
https://link.springer.com/article/10.1007/s44163-025-00547-8  
Applies RL algorithms to A-share trading, respecting actual market trading rules and evaluating effectiveness in a realistic setting.

**Reinforcement Learning for Single-Stock Portfolio Optimization**  
Springer  
https://link.springer.com/article/10.1007/s44196-025-01105-x  
Proposes an adaptive RL portfolio management framework for single-stock optimization using DDPG-style policy learning.

**Systematic Review of Reinforcement Learning for Automated Equity Portfolio Management**  
Springer  
https://link.springer.com/article/10.1007/s10791-026-10336-1  
A comprehensive review of reinforcement learning approaches in automated equity portfolio management.

---

## 3. Explainable AI and Explanation Faithfulness

**Effectiveness of Explainable Artificial Intelligence (XAI) Techniques in Machine Learning**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/11017606  
Evaluates the effectiveness of XAI methods and their ability to provide faithful and interpretable explanations.

**Solving the Enigma: Enhancing Faithfulness and Comprehensibility in Explainable AI**  
ScienceDirect / Elsevier  
https://www.sciencedirect.com/science/article/pii/S266665102500004X  
Examines the tension between explanation plausibility and actual model faithfulness, a problem highly relevant to financial decision support.

**XAI: On Explainability and the Obligation to Explain**  
Springer  
https://link.springer.com/article/10.1007/s44206-025-00215-5  
Discusses why explainability is not merely optional in AI, especially in regulated and high-stakes domains such as finance.

**A Comprehensive Review on Financial Explainable AI**  
Artificial Intelligence Review (Springer), 2025  
https://link.springer.com/article/10.1007/s10462-024-11077-7  
A major survey of how SHAP/LIME-style post-hoc explanations are used in finance, and how their limitations affect real-world decision-making.

---

## 4. Causal Inference in Finance and Dynamic Decision-Making

**ML-Based Causal Inference for Decision-Making in Dynamic Environments**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/11390025  
Introduces temporal machine learning frameworks for causal inference in dynamic systems, with relevance to market regime transitions and adaptive trading.

**CISTF: A Causal Inference Method for Stock Trend Forecasting**  
IEEE Xplore  
https://ieeexplore.ieee.org/document/11033554  
Provides a causal framework for identifying dependencies in financial data and forecasting stock trends using causal structure.

---

## 5. Why these references matter for this project

These papers collectively support three central themes of the project:

1. Regime detection and market-state modeling are essential because financial markets are non-stationary and regime-dependent.  
2. RL and DRL offer powerful adaptive trading mechanisms, but they need to be paired with interpretability and verification.  
3. XAI and causal reasoning are necessary to ensure the model is not only profitable, but also faithful, trustworthy, and operationally defensible under changing market conditions.

---

### Suggested reading order
1. RegimeFolio → 2. HMM-Based Market Regime Detection with RL for Portfolio Management → 3. Research on Portfolio Optimization Based on Deep Reinforcement Learning → 4. A Comprehensive Review on Financial Explainable AI → 5. Solving the Enigma: Enhancing Faithfulness and Comprehensibility in Explainable AI → 6. CISTF: A Causal Inference Method for Stock Trend Forecasting


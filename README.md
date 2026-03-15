# Advanced Credit ECL Simulator (IFRS 9 / CECL Framework)

This repository contains an interactive, quantitative dashboard for simulating and analyzing Expected Credit Loss (ECL) under IFRS 9 and CECL frameworks. It is designed to demonstrate deep mathematical rigor in modern credit underwriting, pricing, and portfolio risk management.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31+-red.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.0+-green.svg)

## The Mathematical Framework

Unlike superficial slider dashboards, this simulator computes ECL through a rigorous decomposition:

### $$ECL = PD \times LGD \times EAD$$

#### 1. Probability of Default (PD)
- **Base Model:** XGBoost classifier (`binary:logistic`) trained on synthetic borrower data (FICO, DTI, LTV, Employment History, Loan Purpose).
- **Calibration:** Platt scaling (sigmoid calibration) applied via a held-out calibration set to ensure output probabilities are mathematically sound default rates, not just rank-ordering scores.
- **Attribution:** SHAP (Shapley Additive Explanations) TreeExplainer is used to generate feature-level log-odds attributions (waterfall charts) and marginal effects (partial dependence plots).

#### 2. Loss Given Default (LGD)
- **Model:** Modeled as a Continuous Beta Distribution: $LGD \sim Beta(\alpha, \beta)$
- **Collateral Driven:** Base parameters $(\alpha, \beta)$ are determined by the collateral type (Real Estate vs. Vehicle vs. Unsecured).
- **LTV Sensitivity:** The $\alpha$ parameter is dynamically adjusted upward for secured loans based on the Loan-to-Value (LTV) ratio exceeding 70%, shifting the expected severity distribution rightward.

#### 3. Exposure at Default (EAD) & Amortisation
- **Model:** Full reducing-balance amortisation schedule.
- **Timing:** Lifetime ECL is calculated via a discounted monthly marginal-loss summation:
  $$ECL = \sum_{t} [ h \cdot S(t-1) \times LGD \times EAD(t) \times (1+d/12)^{-t} ]$$
  *(where $h$ is the monthly hazard rate and $S$ is survival probability)*

#### 4. Macroeconomic Stress Overlay (Wilson 1997)
- **Methodology:** We use a bounds-preserving logit-space macro overlay. 
- **Formula:** $logit(PD_{adj}) = logit(PD_{base}) + \beta_u \Delta U + \beta_g \Delta G + \beta_r \Delta R$
- This allows stress testing (e.g., Stagflation, Severe Recession) while preserving the underlying idiosyncratic rank-ordering of the XGBoost model.

#### 5. IFRS 9 Staging
- **Stage 1:** 12-month ECL (No Significant Increase in Credit Risk).
- **Stage 2:** Lifetime ECL (SICR triggered via 30+ DPD or absolute PD threshold).
- **Stage 3:** Lifetime ECL (Credit-impaired, 90+ DPD).

## Running the Dashboard Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vamsh1/credit-risk-simulator.git
   cd credit-risk-simulator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Architecture Details
- The portfolio generation engine uses a Merton-style latent factor model to simulate structurally coherent correlated defaults.
- UI built entirely in Python via `streamlit`, leveraging customized dark-themed `matplotlib` subplots for high-performance financial data visualization.

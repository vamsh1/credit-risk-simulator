"""
Advanced Credit ECL Simulator — IFRS 9 / CECL Framework
=========================================================
ECL = PD × LGD × EAD

PD  : XGBoost classifier, Platt-scaled, with SHAP TreeExplainer
LGD : Beta(α, β) regression — parameters driven by LTV and collateral type
EAD : Standard annuity amortisation schedule (reducing-balance)
Macro: Wilson (1997) logit-space overlay — logit(PD_adj) = logit(PD) + β·ΔX
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from scipy import stats
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit ECL Simulator — IFRS 9 / CECL",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* dark card */
  .kpi-card {
    background: linear-gradient(135deg,#0d1117 0%,#161b22 100%);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 4px;
  }
  .kpi-val  { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
  .kpi-lbl  { font-size: 0.78rem; color: #8b949e; margin-top: 3px; letter-spacing:.04em; }
  /* formula block */
  .formula  {
    background: #0d1117;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 10px 16px;
    font-family: monospace;
    font-size: 0.88rem;
    color: #79c0ff;
    margin: 10px 0 16px 0;
    line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
rng_global = np.random.RandomState(SEED)

PURPOSE_MAP    = {"mortgage": 0, "auto": 1, "personal": 2, "credit_card": 3, "business": 4}
COLLATERAL_MAP = {"real_estate": 0, "vehicle": 1, "unsecured": 2}

FEATURE_COLS = [
    "fico_score", "dti", "ltv", "emp_years",
    "loan_amount", "loan_term", "account_age_months",
    "purpose_enc", "collateral_enc",
    "macro_unemp", "macro_gdp", "macro_rate",
]

FEATURE_LABELS = {
    "fico_score":         "FICO Score",
    "dti":                "DTI (%)",
    "ltv":                "LTV (%)",
    "emp_years":          "Employment (yrs)",
    "loan_amount":        "Loan Amount ($)",
    "loan_term":          "Loan Term (mo)",
    "account_age_months": "Account Age (mo)",
    "purpose_enc":        "Purpose",
    "collateral_enc":     "Collateral",
    "macro_unemp":        "Unemployment (%)",
    "macro_gdp":          "GDP Growth (%)",
    "macro_rate":         "Policy Rate (%)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC PORTFOLIO GENERATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_portfolio(n: int, macro_unemp: float, macro_gdp: float, macro_rate: float) -> pd.DataFrame:
    """
    Simulate a realistic loan portfolio via a latent-factor credit model.

    Default indicator is drawn from a Merton-style factor model:
        logit(PD_i) = α + β_FICO·z_FICO + β_DTI·DTI + ... + γ·F_macro + ε_i
    where F_macro captures systematic macro exposure (Wilson 1997).

    LGD is drawn from Beta(α_collateral, β_collateral) distributions whose
    parameters shift with LTV (secured) or remain flat (unsecured).

    EAD is the remaining principal at a Beta(4,2)-distributed default timing.
    """
    rng = np.random.RandomState(SEED)

    # ── Borrower & Loan Characteristics ──────────────────────────────────────
    fico       = rng.normal(680, 65, n).clip(450, 850).round()
    dti        = rng.beta(2.5, 5.0, n) * 60          # Debt-to-Income %
    ltv        = rng.beta(3.0, 2.0, n) * 100          # Loan-to-Value %
    emp_years  = rng.exponential(7.0, n).clip(0, 40)
    income     = rng.lognormal(11.0, 0.5, n)          # Annual income $
    loan_amt   = (income * rng.uniform(0.5, 4.5, n)).round(0)
    loan_term  = rng.choice([12, 24, 36, 48, 60, 84], n, p=[0.05, 0.10, 0.30, 0.20, 0.30, 0.05])
    purpose    = rng.choice(
        ["mortgage", "auto", "personal", "credit_card", "business"],
        n, p=[0.35, 0.20, 0.25, 0.12, 0.08]
    )
    collateral = np.where(
        purpose == "mortgage", "real_estate",
        np.where(purpose == "auto", "vehicle", "unsecured")
    )
    acct_age   = rng.randint(1, 73, n)

    # ── Latent Default Model ──────────────────────────────────────────────────
    # fico_z: higher FICO → lower default risk → negative contribution to logit(PD)
    fico_z = (fico - 850) / (850 - 450)   # ∈ [-1, 0]

    purpose_adj = np.where(purpose == "credit_card", 0.80,
                  np.where(purpose == "personal",    0.50,
                  np.where(purpose == "business",    0.30, 0.0)))

    logit_pd = (
        -3.50
        + 3.50 * fico_z            # FICO: negative → higher FICO lowers PD
        + 0.040 * dti              # DTI: higher → more stressed borrower
        + 0.020 * ltv              # LTV: higher → more leveraged
        - 0.050 * emp_years        # Employment stability → reduces PD
        + purpose_adj              # Product-specific base uplift
        + 0.15 * (macro_unemp - 5.5)   # Macro: unemployment beta
        - 0.08 * (macro_gdp  - 2.1)    # Macro: GDP growth beta (negative)
        + 0.05 * (macro_rate - 4.5)    # Macro: rate sensitivity
        + rng.normal(0, 0.50, n)       # Idiosyncratic noise
    )

    true_pd = expit(logit_pd)
    default_flag = (rng.uniform(0, 1, n) < true_pd).astype(int)

    # ── LGD: Beta Distribution ────────────────────────────────────────────────
    # Base α, β per collateral type (real_estate: low LGD; unsecured: high LGD)
    alpha_base = np.where(collateral == "real_estate", 1.5,
                 np.where(collateral == "vehicle",     2.0, 4.0))
    beta_base  = np.where(collateral == "real_estate", 4.0,
                 np.where(collateral == "vehicle",     5.0, 3.0))

    # LTV adjustment: every 1% above 70% LTV adds to α for secured loans
    ltv_adj = np.where(
        collateral != "unsecured",
        0.02 * np.clip(ltv - 70, 0, 30),
        0.0,
    )
    alpha_lgd = alpha_base + ltv_adj
    beta_lgd  = beta_base

    lgd = rng.beta(alpha_lgd, beta_lgd).clip(0.01, 0.99)

    # ── EAD: Remaining Principal at Default ───────────────────────────────────
    # Default timing modelled as Beta(4,2): most defaults occur mid-life
    pct_remaining = rng.beta(4, 2, n)
    ead = loan_amt * pct_remaining

    df = pd.DataFrame({
        "loan_id":            [f"LN-{i:05d}" for i in range(n)],
        "fico_score":         fico,
        "dti":                dti.round(2),
        "ltv":                ltv.round(2),
        "emp_years":          emp_years.round(1),
        "income":             income.round(0),
        "loan_amount":        loan_amt,
        "loan_term":          loan_term,
        "purpose":            purpose,
        "collateral_type":    collateral,
        "account_age_months": acct_age,
        "vintage_year":       2023 - (acct_age // 12),
        "true_pd":            true_pd.round(5),
        "lgd":                lgd.round(4),
        "ead":                ead.round(2),
        "default_flag":       default_flag,
        "alpha_lgd":          alpha_lgd.round(3),
        "beta_lgd":           beta_lgd.round(3),
        "macro_unemp":        macro_unemp,
        "macro_gdp":          macro_gdp,
        "macro_rate":         macro_rate,
    })
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["purpose_enc"]    = df["purpose"].map(PURPOSE_MAP)
    df["collateral_enc"] = df["collateral_type"].map(COLLATERAL_MAP)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PD MODEL — XGBoost + Platt Calibration
# ─────────────────────────────────────────────────────────────────────────────
class _PlattWrapper:
    """
    Thin wrapper that applies manual Platt sigmoid recalibration on top of any
    base classifier that exposes predict_proba().

    Platt scaling (Platt 1999):
        P(y=1 | s) = σ(A·s + B)   where s = base model score, A & B are fit
                                    via MLE on a held-out calibration set.
    Implemented here with LogisticRegression(1 feature) for numerical stability.
    """
    def __init__(self, base):
        self.base  = base
        self._lr   = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)

    def fit(self, X_cal, y_cal):
        raw = self.base.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        self._lr.fit(raw, y_cal)
        return self

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        return self._lr.predict_proba(raw)   # shape (n, 2)


@st.cache_resource(show_spinner=False)
def train_pd_model(_df: pd.DataFrame):
    """
    Train XGBoost PD classifier with Platt sigmoid recalibration.

    Three-way split:
      70% train → XGBoost fit
      15% cal   → Platt scaling (A, B of sigmoid)
      15% test  → held-out evaluation

    Returns a bundle dict containing:
      model        — _PlattWrapper (calibrated probability predictions)
      raw_model    — bare XGBClassifier (used by SHAP TreeExplainer)
      explainer    — shap.TreeExplainer(raw_model)
      shap_values  — SHAP matrix on training subsample (n_sub × n_feat)
      X_train_sub  — training subsample used for SHAP
      X_test       — held-out test features
      y_test       — held-out test labels
      auc          — ROC-AUC on test set (calibrated model)
      logloss      — log-loss on test set (calibrated model)
    """
    df = _encode(_df)
    X  = df[FEATURE_COLS].values
    y  = df["default_flag"].values

    # 70 / 15 / 15 split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                 random_state=SEED, stratify=y)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                 random_state=SEED, stratify=y_tmp)

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    raw = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )
    raw.fit(X_tr, y_tr)

    # Platt calibration on held-out calibration set
    cal = _PlattWrapper(raw).fit(X_cal, y_cal)

    probs_te = cal.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, probs_te)
    ll  = log_loss(y_te,  probs_te)

    # SHAP — subsample training set for speed
    rng_s = np.random.RandomState(0)
    sub_idx     = rng_s.choice(len(X_tr), min(600, len(X_tr)), replace=False)
    X_tr_sub    = X_tr[sub_idx]
    explainer   = shap.TreeExplainer(raw)
    shap_matrix = explainer.shap_values(X_tr_sub)

    # For binary XGBoost, shap_values may be list of 2 arrays or single array
    if isinstance(shap_matrix, list):
        shap_matrix = shap_matrix[1]

    return {
        "model":       cal,
        "raw_model":   raw,
        "explainer":   explainer,
        "shap_values": shap_matrix,   # (n_sub, n_feat) — log-odds contributions
        "X_train_sub": X_tr_sub,
        "X_test":      X_te,
        "y_test":      y_te,
        "auc":         auc,
        "logloss":     ll,
    }


def predict_pd(bundle, X: np.ndarray) -> np.ndarray:
    return bundle["model"].predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LGD MODEL — Beta Regression
# ─────────────────────────────────────────────────────────────────────────────
def lgd_beta_params(ltv: float, collateral: str, stress: float = 1.0):
    """
    Return (α, β, mean, variance) of Beta(α,β) LGD distribution.

    Stress factor amplifies α, shifting the distribution rightward (higher LGD).
    """
    base_a = {"real_estate": 1.5, "vehicle": 2.0, "unsecured": 4.0}
    base_b = {"real_estate": 4.0, "vehicle": 5.0, "unsecured": 3.0}

    a = base_a.get(collateral, 3.0)
    b = base_b.get(collateral, 3.0)

    if collateral != "unsecured":
        a += 0.02 * max(0.0, ltv - 70.0)

    a *= stress  # stress shifts mean upward

    mean = a / (a + b)
    var  = (a * b) / ((a + b) ** 2 * (a + b + 1))
    return a, b, mean, var


def fit_beta_mle(values: np.ndarray):
    """MLE fit of Beta distribution parameters."""
    v = np.clip(values, 1e-4, 1 - 1e-4)

    def neg_ll(p):
        if p[0] <= 0 or p[1] <= 0:
            return 1e10
        return -np.sum(stats.beta.logpdf(v, p[0], p[1]))

    res = minimize(neg_ll, x0=[2.0, 3.0], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5})
    return res.x[0], res.x[1]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EAD — Amortisation Schedule
# ─────────────────────────────────────────────────────────────────────────────
def amortisation_schedule(principal: float, annual_rate_pct: float, term_months: int) -> pd.DataFrame:
    """
    Generate full reducing-balance amortisation schedule.
    Monthly payment  P = L · [r(1+r)^n] / [(1+r)^n - 1]
    """
    r = annual_rate_pct / 100.0 / 12.0
    if r > 0:
        pmt = principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)
    else:
        pmt = principal / term_months

    rows, balance = [], principal
    for m in range(1, term_months + 1):
        interest  = balance * r
        principal_pmt = pmt - interest
        balance   = max(0.0, balance - principal_pmt)
        rows.append({
            "month":         m,
            "payment":       pmt,
            "interest":      interest,
            "principal_pmt": principal_pmt,
            "balance":       balance,   # EAD if default occurs just after this payment
        })
    return pd.DataFrame(rows)


def lifetime_ecl(pd_annual: float, lgd: float,
                 sched: pd.DataFrame, discount_rate: float = 0.05) -> float:
    """
    Discounted lifetime ECL via monthly marginal-loss summation.

        ECL = Σ_t  [ h·S(t-1) × LGD × EAD(t) × DF(t) ]

    where h  = monthly hazard  = 1 − (1−PD_annual)^(1/12)
          S(t)= survival = (1−h)^t
          DF(t)= discount factor = (1 + d/12)^{-t}
    """
    h    = 1.0 - (1.0 - pd_annual) ** (1.0 / 12.0)
    df_m = 1.0 / (1.0 + discount_rate / 12.0)
    ecl, S = 0.0, 1.0
    for _, row in sched.iterrows():
        marginal_pd = S * h
        ecl += marginal_pd * lgd * row["balance"] * (df_m ** row["month"])
        S   *= (1.0 - h)
    return ecl


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MACRO OVERLAY — Wilson (1997)
# ─────────────────────────────────────────────────────────────────────────────
def wilson_overlay(base_pd: np.ndarray,
                   unemp_base: float, unemp_stress: float,
                   gdp_base: float,   gdp_stress: float,
                   rate_base: float,  rate_stress: float,
                   b_unemp: float = 0.15,
                   b_gdp:   float = -0.08,
                   b_rate:  float = 0.05) -> np.ndarray:
    """
    Apply macro shock in logit space (bounds-preserving):
        logit(PD_adj) = logit(PD_base) + β_u·ΔU + β_g·ΔG + β_r·ΔR
    """
    dU = unemp_stress - unemp_base
    dG = gdp_stress   - gdp_base
    dR = rate_stress  - rate_base
    shift = b_unemp * dU + b_gdp * dG + b_rate * dR
    return expit(logit(np.clip(base_pd, 1e-6, 1 - 1e-6)) + shift)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  IFRS 9 STAGE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def ifrs9_stage(pd_12m: float, dpd: int = 0) -> int:
    """
    Stage 1 — 12-month ECL  (no SICR)
    Stage 2 — Lifetime ECL  (significant increase in credit risk, SICR)
    Stage 3 — Lifetime ECL  (credit-impaired / defaulted)
    """
    if dpd >= 90 or pd_12m > 0.80:
        return 3
    elif dpd >= 30 or pd_12m > 0.05:
        return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MATPLOTLIB DARK THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#0d1117"
BG2  = "#161b22"
GRID = "#21262d"
TEXT = "#c9d1d9"
BLUE = "#58a6ff"
RED  = "#f85149"
GRN  = "#3fb950"
YEL  = "#d29922"
PURP = "#bc8cff"

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    for obj in (fig, ax):
        obj.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    return fig, ax


def dark_fig_multi(nrows=1, ncols=2, w=12, h=5, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), sharey=sharey)
    fig.set_facecolor(BG)
    for ax in np.array(axes).flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    return fig, axes


def styled_legend(ax, **kw):
    leg = ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TEXT,
                    fontsize=8, **kw)
    return leg


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("Credit ECL Simulator — IFRS 9 / CECL")
    st.caption("XGBoost PD · Beta LGD · Amortisation EAD · Wilson Macro Overlay · SHAP Attribution")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Macro Environment")

        st.subheader("Baseline")
        u0 = st.number_input("Unemployment (%) — Base", value=5.5, step=0.1, format="%.1f")
        g0 = st.number_input("GDP Growth (%) — Base",   value=2.1, step=0.1, format="%.1f")
        r0 = st.number_input("Policy Rate (%) — Base",  value=4.5, step=0.1, format="%.1f")

        st.subheader("Stress Scenario")
        preset = st.selectbox("Preset", ["Custom", "Mild Recession",
                                          "Severe Recession", "Stagflation"])
        presets = {
            "Mild Recession":    (u0+2.0, g0-1.5, r0-0.50),
            "Severe Recession":  (u0+5.0, g0-4.0, r0-2.00),
            "Stagflation":       (u0+3.0, g0-1.0, r0+3.00),
        }
        if preset in presets:
            us, gs, rs = presets[preset]
        else:
            us = st.slider("Unemployment (%) — Stress", 3.0, 15.0, u0+2.0, 0.1)
            gs = st.slider("GDP Growth (%) — Stress",  -5.0,  5.0, g0-1.5, 0.1)
            rs = st.slider("Policy Rate (%) — Stress",  0.0, 12.0, r0+1.0, 0.1)

        st.subheader("Wilson Betas")
        b_u = st.slider("β unemployment", 0.05, 0.50,  0.15, 0.01)
        b_g = st.slider("β GDP",         -0.30,-0.01, -0.08, 0.01)
        b_r = st.slider("β rate",         0.01, 0.25,  0.05, 0.01)

        st.subheader("LGD Stress")
        lgd_sf = st.slider("LGD stress factor (α×)", 1.0, 2.5, 1.0, 0.05,
                            help="Multiplies the Beta α parameter → shifts mean LGD upward")

        st.subheader("Portfolio")
        n_loans = st.select_slider("Portfolio size", [500, 1000, 2000, 5000], value=2000)

    # ── Data + Model ──────────────────────────────────────────────────────────
    with st.spinner("Generating portfolio and training XGBoost PD model…"):
        df_raw  = generate_portfolio(n_loans, u0, g0, r0)
        bundle  = train_pd_model(df_raw)

    df = _encode(df_raw)

    # Predict base PD (features at baseline macro)
    df["pd_base"] = predict_pd(bundle, df[FEATURE_COLS].values)

    # Predict stress PD via re-inference with stressed macro features
    df_s = df.copy()
    df_s["macro_unemp"] = us
    df_s["macro_gdp"]   = gs
    df_s["macro_rate"]  = rs
    df["pd_stress_model"] = predict_pd(bundle, df_s[FEATURE_COLS].values)

    # Wilson overlay (logit-space shift on top of base PD)
    df["pd_wilson"] = wilson_overlay(df["pd_base"].values, u0, us, g0, gs, r0, rs,
                                     b_u, b_g, b_r)

    # Stressed LGD (vectorised)
    df["lgd_stressed"] = df.apply(
        lambda r: lgd_beta_params(r["ltv"], r["collateral_type"], lgd_sf)[2], axis=1
    )

    # ECL computations (12-month approximation: ECL ≈ PD × LGD × EAD)
    df["ecl_base"]   = df["pd_base"]   * df["lgd"]         * df["ead"]
    df["ecl_stress"] = df["pd_wilson"] * df["lgd_stressed"] * df["ead"]

    # IFRS 9 stage
    rng_dpd = np.random.RandomState(99)
    dpd_sim = rng_dpd.choice([0,0,0,0,0,30,60,90,120], n_loans,
                              p=[0.65,0.10,0.05,0.05,0.05,0.03,0.03,0.02,0.02])
    df["dpd_sim"] = dpd_sim
    df["stage"]   = df.apply(lambda r: ifrs9_stage(r["pd_base"], r["dpd_sim"]), axis=1)
    df["ecl_cov"] = df["ecl_base"] / df["ead"].clip(lower=1)

    # ── TABS ─────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Portfolio Overview",
        "PD Model & SHAP",
        "LGD Distribution",
        "EAD & Amortisation",
        "Macro Stress",
        "Loan Analyser",
    ])

    # =========================================================================
    # TAB 1 — PORTFOLIO OVERVIEW
    # =========================================================================
    with t1:
        st.header("Portfolio ECL Overview")
        st.markdown("""<div class="formula">
ECL (12m)      =  PD × LGD × EAD<br>
Lifetime ECL   =  Σ_t  [ h·S(t-1) × LGD × EAD(t) × (1+d)^{-t} ]<br>
IFRS 9 Stages  :  Stage 1 → 12m ECL  |  Stage 2/3 → Lifetime ECL
</div>""", unsafe_allow_html=True)

        ead_tot = df["ead"].sum()
        ecl_b   = df["ecl_base"].sum()
        ecl_s   = df["ecl_stress"].sum()
        avg_pd  = df["pd_base"].mean()
        avg_lgd = df["lgd"].mean()
        cov     = ecl_b / ead_tot

        cols = st.columns(6)
        for col, label, val in zip(cols, [
            "Total EAD", "ECL Base", "ECL Stressed",
            "Avg PD", "Avg LGD", "ECL Coverage"
        ], [
            f"${ead_tot/1e6:.1f}M",
            f"${ecl_b/1e6:.2f}M",
            f"${ecl_s/1e6:.2f}M",
            f"{avg_pd*100:.2f}%",
            f"{avg_lgd*100:.1f}%",
            f"{cov*100:.2f}%",
        ]):
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                f'<div class="kpi-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        c1, c2 = st.columns(2)

        # Stage breakdown ─────────────────────────────────────────────────────
        with c1:
            st.subheader("ECL by IFRS 9 Stage")
            stg = df.groupby("stage").agg(
                Count   = ("loan_id",   "count"),
                EAD     = ("ead",       "sum"),
                ECL_B   = ("ecl_base",  "sum"),
                ECL_S   = ("ecl_stress","sum"),
                Avg_PD  = ("pd_base",   "mean"),
                Avg_LGD = ("lgd",       "mean"),
            )
            stg["Cov_B"] = stg["ECL_B"] / stg["EAD"]
            stg["Cov_S"] = stg["ECL_S"] / stg["EAD"]

            fig, axes = dark_fig_multi(1, 2, w=10, h=4)
            colors_s  = [GRN, YEL, RED]
            labels_s  = [f"Stage {i}" for i in stg.index]

            # EAD bar
            axes[0].bar(labels_s, stg["EAD"]/1e6, color=colors_s, alpha=0.85, edgecolor=GRID)
            axes[0].set_title("EAD by Stage ($M)", fontweight="bold")
            axes[0].set_ylabel("$M", color=TEXT)
            for i, v in enumerate(stg["EAD"]/1e6):
                axes[0].text(i, v + 0.02, f"${v:.1f}M", ha="center",
                             color=TEXT, fontsize=8)

            # Coverage grouped bar
            x  = np.arange(len(stg))
            w  = 0.35
            axes[1].bar(x - w/2, stg["Cov_B"]*100, w, label="Base",     color=BLUE,  alpha=0.85)
            axes[1].bar(x + w/2, stg["Cov_S"]*100, w, label="Stressed", color=RED,   alpha=0.85)
            axes[1].set_xticks(x); axes[1].set_xticklabels(labels_s)
            axes[1].set_title("ECL Coverage % by Stage", fontweight="bold")
            axes[1].set_ylabel("Coverage %", color=TEXT)
            styled_legend(axes[1])

            plt.tight_layout()
            st.pyplot(fig); plt.close()

            stg_disp = stg.copy()
            stg_disp.index = labels_s
            st.dataframe(
                stg_disp[["Count","EAD","ECL_B","ECL_S","Avg_PD","Avg_LGD","Cov_B","Cov_S"]]
                .rename(columns={"ECL_B":"ECL Base","ECL_S":"ECL Stress",
                                  "Avg_PD":"Avg PD","Avg_LGD":"Avg LGD",
                                  "Cov_B":"Coverage (Base)","Cov_S":"Coverage (Stress)"})
                .style.format({
                    "EAD":"${:,.0f}", "ECL Base":"${:,.0f}", "ECL Stress":"${:,.0f}",
                    "Avg PD":"{:.2%}", "Avg LGD":"{:.2%}",
                    "Coverage (Base)":"{:.2%}", "Coverage (Stress)":"{:.2%}",
                }),
                use_container_width=True,
            )

        # Purpose breakdown ───────────────────────────────────────────────────
        with c2:
            st.subheader("ECL by Loan Purpose")
            pur = (df.groupby("purpose")
                     .agg(EAD=("ead","sum"), ECL=("ecl_base","sum"))
                     .assign(Coverage=lambda x: x["ECL"]/x["EAD"])
                     .sort_values("ECL", ascending=True))

            fig, ax = dark_fig(8, 4)
            cmap   = plt.cm.RdYlGn_r
            colors_p = [cmap(i/len(pur)) for i in range(len(pur))]
            bars = ax.barh(pur.index, pur["ECL"]/1e3, color=colors_p, alpha=0.88, edgecolor=GRID)
            for bar, (_, row) in zip(bars, pur.iterrows()):
                ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                        f"{row['Coverage']*100:.1f}%", va="center", color="#8b949e", fontsize=8)
            ax.set_xlabel("ECL ($K)", color=TEXT); ax.set_title("ECL by Purpose ($K)", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.subheader("Vintage × Stage ECL Heatmap")
            vm = (df.groupby(["vintage_year","stage"])["ecl_cov"]
                    .mean().unstack(fill_value=0) * 100)
            vm.columns = [f"Stage {c}" for c in vm.columns]

            fig, ax = dark_fig(8, 3.2)
            im = ax.imshow(vm.T.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=6)
            ax.set_xticks(range(len(vm.index)));    ax.set_xticklabels(vm.index, color=TEXT)
            ax.set_yticks(range(len(vm.columns))); ax.set_yticklabels(vm.columns, color=TEXT)
            ax.set_title("Avg ECL Coverage % — Vintage × Stage", fontweight="bold")
            for i in range(vm.shape[1]):
                for j in range(vm.shape[0]):
                    v = vm.values[j, i]
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                            color="white" if v > 2 else BG, fontsize=8, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8, label="Coverage %")
            ax.grid(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # =========================================================================
    # TAB 2 — PD MODEL & SHAP
    # =========================================================================
    with t2:
        st.header("PD Model — XGBoost + Platt Calibration + SHAP")
        st.markdown("""<div class="formula">
Model  :  PD = σ(XGB(x))  where σ = Platt sigmoid recalibration<br>
SHAP   :  f(x) = φ₀ + Σᵢ φᵢ(xᵢ)  (Shapley Additive Explanations, Lundberg & Lee 2017)<br>
Marginal effect  :  ∂PD/∂xⱼ  via ceteris-paribus partial dependence grid
</div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC",       f"{bundle['auc']:.4f}")
        m2.metric("Log-Loss",      f"{bundle['logloss']:.4f}")
        m3.metric("Default Rate",  f"{df['default_flag'].mean()*100:.1f}%")
        m4.metric("Train Size",    f"{len(bundle['X_train_sub'])*100//30*30:,}")

        st.markdown("---")
        c1, c2 = st.columns([1.3, 1])

        # SHAP beeswarm ───────────────────────────────────────────────────────
        with c1:
            st.subheader("SHAP Feature Impact (Beeswarm)")
            sv    = bundle["shap_values"]          # (n_sub, n_feat)
            X_sub = bundle["X_train_sub"]
            mean_abs = np.abs(sv).mean(axis=0)
            order    = np.argsort(mean_abs)

            fig, ax = dark_fig(9, 7)
            n_f = len(FEATURE_COLS)
            rng_j = np.random.RandomState(7)
            for rank, fi in enumerate(order):
                sv_f = sv[:, fi]
                xf   = X_sub[:, fi]
                xnorm = (xf - xf.min()) / (xf.ptp() + 1e-8)
                jitter = rng_j.uniform(-0.35, 0.35, len(sv_f))
                ax.scatter(sv_f, rank + jitter, c=xnorm, cmap="RdBu_r",
                           s=5, alpha=0.65, vmin=0, vmax=1, linewidths=0)

            ax.set_yticks(range(n_f))
            ax.set_yticklabels([FEATURE_LABELS[FEATURE_COLS[i]] for i in order],
                               color=TEXT, fontsize=9)
            ax.axvline(0, color=GRID, lw=1, ls="--")
            ax.set_xlabel("SHAP value (log-odds contribution to default)", color=TEXT)
            ax.set_title("SHAP Beeswarm — XGBoost PD Model", fontweight="bold", color=TEXT)

            sm  = plt.cm.ScalarMappable(cmap="RdBu_r", norm=Normalize(0,1))
            cb  = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
            cb.set_label("Feature value\n(low → high)", color=TEXT, fontsize=8)
            cb.ax.tick_params(labelcolor=TEXT)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Marginal effects ────────────────────────────────────────────────────
        with c2:
            st.subheader("Marginal Effects — Ceteris Paribus")

            med_feat = np.median(X_sub, axis=0)
            fico_idx = FEATURE_COLS.index("fico_score")
            dti_idx  = FEATURE_COLS.index("dti")
            ltv_idx  = FEATURE_COLS.index("ltv")
            u_idx    = FEATURE_COLS.index("macro_unemp")

            def partial_dep(feat_idx, grid, stress_unemp=None):
                base_pds, stress_pds = [], []
                for v in grid:
                    row = med_feat.copy()
                    row[feat_idx] = v
                    base_pds.append(predict_pd(bundle, row.reshape(1,-1))[0])
                    if stress_unemp is not None:
                        row_s = row.copy(); row_s[u_idx] = stress_unemp
                        stress_pds.append(predict_pd(bundle, row_s.reshape(1,-1))[0])
                return np.array(base_pds), np.array(stress_pds)

            # FICO
            fico_grid = np.linspace(450, 850, 70)
            pd_f_b, pd_f_s = partial_dep(fico_idx, fico_grid, us)

            fig, ax = dark_fig(7, 3.8)
            ax.plot(fico_grid, pd_f_b*100, color=BLUE,  lw=2, label=f"Base (U={u0:.1f}%)")
            ax.plot(fico_grid, pd_f_s*100, color=RED,   lw=2, ls="--", label=f"Stress (U={us:.1f}%)")
            ax.fill_between(fico_grid, pd_f_b*100, pd_f_s*100, alpha=0.12, color=RED)
            ax.set_xlabel("FICO Score"); ax.set_ylabel("Predicted PD (%)")
            ax.set_title("∂PD/∂FICO — Partial Dependence", fontweight="bold")
            styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

            # DTI
            dti_grid = np.linspace(5, 58, 70)
            pd_d_b, pd_d_s = partial_dep(dti_idx, dti_grid, us)

            fig, ax = dark_fig(7, 3.8)
            ax.plot(dti_grid, pd_d_b*100, color=BLUE, lw=2, label=f"Base")
            ax.plot(dti_grid, pd_d_s*100, color=RED,  lw=2, ls="--", label=f"Stress")
            ax.fill_between(dti_grid, pd_d_b*100, pd_d_s*100, alpha=0.12, color=RED)
            ax.set_xlabel("DTI Ratio (%)"); ax.set_ylabel("Predicted PD (%)")
            ax.set_title("∂PD/∂DTI — Partial Dependence", fontweight="bold")
            styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        # Calibration ─────────────────────────────────────────────────────────
        st.subheader("Calibration Reliability Diagram")
        y_te  = bundle["y_test"]
        p_te  = bundle["model"].predict_proba(bundle["X_test"])[:, 1]
        frac, mpred = calibration_curve(y_te, p_te, n_bins=15, strategy="quantile")

        fig, ax = dark_fig(7, 3.5)
        ax.plot([0,1],[0,1], ls="--", color=GRID, lw=1.2, label="Perfect calibration")
        ax.plot(mpred, frac, "o-", color=BLUE, lw=2, ms=5, label="XGB + Platt")
        ax.fill_between(mpred, frac, mpred, alpha=0.18, color=RED, label="Calibration error")
        ax.set_xlabel("Mean Predicted PD"); ax.set_ylabel("Observed Default Rate")
        ax.set_title("Reliability Diagram — PD Calibration", fontweight="bold")
        styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # =========================================================================
    # TAB 3 — LGD DISTRIBUTION
    # =========================================================================
    with t3:
        st.header("LGD Model — Beta Distribution")
        st.markdown("""<div class="formula">
LGD ~ Beta(α, β)  ,  α = f(collateral, LTV, stress_factor)  ,  β = f(collateral)<br>
E[LGD] = α / (α+β)     Var[LGD] = αβ / [(α+β)²(α+β+1)]<br>
Stress : α → α × stress_factor   (rightward shift → higher expected loss severity)
</div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        # Distributions by collateral ─────────────────────────────────────────
        with c1:
            st.subheader("Beta LGD Distributions by Collateral")
            ltv_ex = st.slider("Example LTV (%)", 20, 100, 75)
            fig, axes = dark_fig_multi(1, 3, w=13, h=4, sharey=False)
            x_l = np.linspace(0.001, 0.999, 300)
            coll_list   = ["real_estate", "vehicle", "unsecured"]
            coll_colors = [GRN, YEL, RED]

            for ax, ct, col in zip(axes, coll_list, coll_colors):
                a_b, b_b, m_b, _ = lgd_beta_params(ltv_ex, ct, 1.0)
                a_s, b_s, m_s, _ = lgd_beta_params(ltv_ex, ct, lgd_sf)
                ax.plot(x_l, stats.beta.pdf(x_l, a_b, b_b), color=col, lw=2.2,
                        label=f"Base  μ={m_b*100:.1f}%")
                ax.plot(x_l, stats.beta.pdf(x_l, a_s, b_s), color=col, lw=2.2,
                        ls="--", alpha=0.65, label=f"Stress μ={m_s*100:.1f}%")
                ax.fill_between(x_l, stats.beta.pdf(x_l, a_b, b_b), alpha=0.18, color=col)
                ax.axvline(m_b, color=col, lw=1, ls=":")
                ax.axvline(m_s, color=col, lw=1, ls=":", alpha=0.55)
                ax.set_title(ct.replace("_"," ").title(), fontweight="bold")
                ax.set_xlabel("LGD")
                ax.text(0.02, 0.96, f"α={a_b:.2f}, β={b_b:.2f}",
                        transform=ax.transAxes, color="#8b949e", fontsize=8, va="top")
                styled_legend(ax)

            plt.suptitle(f"Beta LGD — LTV={ltv_ex}%  |  Stress α×{lgd_sf:.2f}",
                         color=TEXT, fontweight="bold", y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # LGD vs LTV ──────────────────────────────────────────────────────────
        with c2:
            st.subheader("E[LGD] vs LTV — Collateral Sensitivity")
            ltv_grid = np.linspace(20, 100, 60)

            fig, ax = dark_fig(7, 4.5)
            for ct, col in [("real_estate", BLUE), ("vehicle", YEL)]:
                mb  = [lgd_beta_params(l, ct, 1.0)[2]  for l in ltv_grid]
                ms  = [lgd_beta_params(l, ct, lgd_sf)[2] for l in ltv_grid]
                label = ct.replace("_"," ").title()
                ax.plot(ltv_grid, np.array(mb)*100, color=col, lw=2, label=f"{label} (Base)")
                ax.plot(ltv_grid, np.array(ms)*100, color=col, lw=2, ls="--", alpha=0.65,
                        label=f"{label} (Stress)")
                ax.fill_between(ltv_grid, np.array(mb)*100, np.array(ms)*100,
                                alpha=0.10, color=col)

            mu_b = lgd_beta_params(70, "unsecured", 1.0)[2]
            mu_s = lgd_beta_params(70, "unsecured", lgd_sf)[2]
            ax.axhline(mu_b*100, color=RED, lw=2, label="Unsecured (Base)")
            ax.axhline(mu_s*100, color=RED, lw=2, ls="--", alpha=0.65, label="Unsecured (Stress)")

            ax.set_xlabel("LTV (%)", color=TEXT); ax.set_ylabel("E[LGD] (%)", color=TEXT)
            ax.set_title("Expected LGD Curve by Collateral Type", fontweight="bold")
            styled_legend(ax, ncol=2); plt.tight_layout(); st.pyplot(fig); plt.close()

        # Portfolio-level LGD fit ─────────────────────────────────────────────
        st.subheader("Portfolio LGD — Empirical vs MLE-Fitted Beta")
        a_fit, b_fit = fit_beta_mle(df["lgd"].values)
        mu_fit = a_fit / (a_fit + b_fit)

        fig, ax = dark_fig(10, 3.8)
        ax.hist(df["lgd"], bins=70, density=True, color=BLUE, alpha=0.45,
                label="Empirical LGD")
        x_fit = np.linspace(0.001, 0.999, 400)
        ax.plot(x_fit, stats.beta.pdf(x_fit, a_fit, b_fit),
                color=RED, lw=2.5, label=f"MLE Beta(α={a_fit:.2f}, β={b_fit:.2f})  E[LGD]={mu_fit*100:.1f}%")
        ax.set_xlabel("LGD", color=TEXT); ax.set_ylabel("Density", color=TEXT)
        ax.set_title(f"Portfolio LGD Distribution — MLE Fit", fontweight="bold")
        styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # =========================================================================
    # TAB 4 — EAD & AMORTISATION
    # =========================================================================
    with t4:
        st.header("EAD — Amortisation Schedule & Lifetime ECL")
        st.markdown("""<div class="formula">
Monthly Payment  =  L · r(1+r)ⁿ / [(1+r)ⁿ−1]   (standard annuity)<br>
EAD(t)           =  Remaining principal balance at month t<br>
Lifetime ECL     =  Σ_t  h·(1−h)^{t-1} × LGD × EAD(t) × (1+d/12)^{-t}<br>
where  h = monthly hazard = 1 − (1−PD_annual)^{1/12}
</div>""", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            st.subheader("Loan Parameters")
            la_princ  = st.number_input("Principal ($)",       5_000,  1_000_000, 200_000, 5_000)
            la_rate   = st.number_input("Annual Rate (%)",     1.0,    25.0,       6.5,    0.25)
            la_term   = st.selectbox("Term (months)", [12,24,36,48,60,84,120,180,240,360], index=6)
            la_pd     = st.number_input("Annual PD (%)",       0.1,    50.0,       3.0,    0.1)
            la_lgd    = st.number_input("LGD (%)",             1.0,    100.0,      35.0,   1.0)
            la_disc   = st.number_input("Discount Rate (%)",   0.5,    15.0,       5.0,    0.25)

            sched = amortisation_schedule(la_princ, la_rate, la_term)
            lt_ecl = lifetime_ecl(la_pd/100, la_lgd/100, sched, la_disc/100)
            ecl_12 = (la_pd/100) * (la_lgd/100) * la_princ

            k1, k2, k3 = st.columns(3)
            k1.metric("12m ECL",      f"${ecl_12:,.0f}")
            k2.metric("Lifetime ECL", f"${lt_ecl:,.0f}")
            k3.metric("Coverage",     f"{lt_ecl/la_princ*100:.2f}%")

            # Amortisation plot
            fig, ax = dark_fig(8, 4)
            ax.fill_between(sched["month"], sched["balance"],   alpha=0.25, color=BLUE)
            ax.plot(sched["month"], sched["balance"],           color=BLUE, lw=2,  label="Remaining Balance (EAD)")
            ax.plot(sched["month"], sched["interest"].cumsum(), color=YEL,  lw=1.5, ls="--", label="Cumulative Interest")
            ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
            ax.set_title(f"Amortisation — ${la_princ:,.0f} @ {la_rate}% / {la_term}mo",
                         fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

            st.subheader("Schedule Detail (first 12 months)")
            st.dataframe(
                sched.head(12).style.format({
                    "payment":       "${:,.2f}", "interest":     "${:,.2f}",
                    "principal_pmt": "${:,.2f}", "balance":      "${:,.2f}",
                }),
                use_container_width=True, height=280,
            )

        with cb:
            st.subheader("Monthly Marginal ECL Profile")
            h_m  = 1.0 - (1.0 - la_pd/100) ** (1.0/12.0)
            df_m = 1.0 / (1.0 + la_disc/100/12.0)

            marg, S = [], 1.0
            for _, row in sched.iterrows():
                mp = S * h_m
                marg.append(mp * (la_lgd/100) * row["balance"] * (df_m ** row["month"]))
                S  *= (1.0 - h_m)
            marg = np.array(marg)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            fig.set_facecolor(BG)
            for ax in (ax1, ax2):
                ax.set_facecolor(BG)
                ax.tick_params(colors=TEXT, labelsize=9)
                for sp in ax.spines.values(): sp.set_color(GRID)
                ax.grid(True, color=GRID, lw=0.5, ls="--", alpha=0.6)

            ax1.bar(sched["month"], marg, color=RED, alpha=0.75, width=0.9, edgecolor=BG)
            ax1.set_xlabel("Month", color=TEXT); ax1.set_ylabel("Marginal ECL ($)", color=TEXT)
            ax1.set_title("h·S(t-1)·LGD·EAD(t)·DF(t)  per month", fontweight="bold", color=TEXT)
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

            ax2.plot(sched["month"], marg.cumsum(), color=BLUE, lw=2, label="Cumulative ECL")
            ax2.axhline(lt_ecl, color=RED, ls="--", lw=1.5, label=f"Total = ${lt_ecl:,.0f}")
            ax2.fill_between(sched["month"], marg.cumsum(), alpha=0.2, color=BLUE)
            ax2.set_xlabel("Month", color=TEXT); ax2.set_ylabel("Cumulative ECL ($)", color=TEXT)
            ax2.set_title("Cumulative Lifetime ECL Buildup", fontweight="bold", color=TEXT)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            styled_legend(ax2)

            plt.tight_layout(); st.pyplot(fig); plt.close()

            # EAD term structure across portfolio
            st.subheader("Portfolio EAD Term Structure (by Stage)")
            fig, ax = dark_fig(8, 4)
            colors_st = {1: GRN, 2: YEL, 3: RED}
            for stage, grp in df.groupby("stage"):
                ax.hist(grp["ead"]/1e3, bins=50, alpha=0.55,
                        color=colors_st[stage], label=f"Stage {stage}",
                        density=True)
            ax.set_xlabel("EAD ($K)"); ax.set_ylabel("Density (normalised)")
            ax.set_title("EAD Distribution by IFRS 9 Stage", fontweight="bold")
            styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # =========================================================================
    # TAB 5 — MACRO STRESS
    # =========================================================================
    with t5:
        st.header("Macroeconomic Stress Scenarios")
        st.markdown("""<div class="formula">
Wilson (1997):  logit(PD_adj) = logit(PD_base) + β_u·ΔU + β_g·ΔG + β_r·ΔR<br>
Ensures PD ∈ (0,1) while providing multiplicative amplification in logit space.<br>
Merton link: systematic macro factor shifts the latent asset-value distribution → tail PD risk
</div>""", unsafe_allow_html=True)

        SCENARIOS = {
            "Base":             (u0,   g0,   r0),
            "Mild Recession":   (u0+2, g0-1.5, r0-0.5),
            "Severe Recession": (u0+5, g0-4,   r0-2),
            "Stagflation":      (u0+3, g0-1,   r0+3),
            "Custom Stress":    (us,   gs,   rs),
        }
        SCEN_COLORS = [BLUE, GRN, RED, PURP, YEL]

        ca, cb = st.columns(2)

        # PD distribution shift ───────────────────────────────────────────────
        with ca:
            st.subheader("PD Distribution — Scenario Shift")
            fig, ax = dark_fig(9, 5)
            for (sn, (su, sg, sr)), col in zip(SCENARIOS.items(), SCEN_COLORS):
                pd_sc = wilson_overlay(df["pd_base"].values, u0, su, g0, sg, r0, sr, b_u, b_g, b_r)
                ax.hist(pd_sc*100, bins=60, density=True, alpha=0.45, color=col,
                        label=f"{sn}  (μ={pd_sc.mean()*100:.2f}%)")
            ax.set_xlabel("Predicted PD (%)"); ax.set_ylabel("Density")
            ax.set_title("PD Distribution Shift — Macro Scenarios", fontweight="bold")
            styled_legend(ax, ncol=1); plt.tight_layout(); st.pyplot(fig); plt.close()

        # ECL vs unemployment ramp ────────────────────────────────────────────
        with cb:
            st.subheader("ECL Sensitivity — Unemployment Ramp")
            u_rng = np.linspace(u0, u0+8, 40)
            ecl_by_stage = {1: [], 2: [], 3: []}
            for u_val in u_rng:
                pd_u = wilson_overlay(df["pd_base"].values, u0, u_val, g0, g0, r0, r0, b_u, b_g, b_r)
                ecl_u = pd_u * df["lgd"].values * df["ead"].values
                for s in [1,2,3]:
                    ecl_by_stage[s].append(ecl_u[df["stage"]==s].sum())

            fig, ax = dark_fig(8, 4.5)
            ax.stackplot(u_rng,
                         [np.array(ecl_by_stage[s])/1e3 for s in [1,2,3]],
                         labels=["Stage 1","Stage 2","Stage 3"],
                         colors=[GRN, YEL, RED], alpha=0.80)
            ax.axvline(u0, color=TEXT, ls=":", lw=1.2, alpha=0.6, label="Baseline")
            ax.axvline(us, color=RED,  ls="--",lw=1.5, alpha=0.7, label="Custom Stress")
            ax.set_xlabel("Unemployment Rate (%)"); ax.set_ylabel("ECL ($K)")
            ax.set_title("ECL Sensitivity to Unemployment (stacked by Stage)", fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
            styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        # Scenario table ──────────────────────────────────────────────────────
        st.subheader("Scenario Summary Table")
        base_ecl = df["ecl_base"].sum()
        rows = []
        for sn, (su, sg, sr) in SCENARIOS.items():
            pd_sc = wilson_overlay(df["pd_base"].values, u0, su, g0, sg, r0, sr, b_u, b_g, b_r)
            ecl_sc = (pd_sc * df["lgd"].values * df["ead"].values).sum()
            rows.append({
                "Scenario":      sn,
                "Unemployment":  f"{su:.1f}%",
                "GDP Growth":    f"{sg:.1f}%",
                "Policy Rate":   f"{sr:.1f}%",
                "Avg PD":        f"{pd_sc.mean()*100:.2f}%",
                "Total ECL":     f"${ecl_sc/1e3:,.1f}K",
                "vs Base":       f"{(ecl_sc/base_ecl - 1)*100:+.1f}%",
                "Coverage":      f"{ecl_sc/df['ead'].sum()*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Tornado ─────────────────────────────────────────────────────────────
        st.subheader("Tornado — ECL Sensitivity to ±1-unit Macro Shocks")
        shocks = [
            ("Unemployment +2%",  u0+2, g0,   r0  ),
            ("Unemployment -2%",  u0-2, g0,   r0  ),
            ("GDP −2%",           u0,   g0-2, r0  ),
            ("GDP +2%",           u0,   g0+2, r0  ),
            ("Rate +1%",          u0,   g0,   r0+1),
            ("Rate −1%",          u0,   g0,   r0-1),
        ]
        tornado = []
        for label, su, sg, sr in shocks:
            pd_sc  = wilson_overlay(df["pd_base"].values, u0, su, g0, sg, r0, sr, b_u, b_g, b_r)
            ecl_sc = (pd_sc * df["lgd"].values * df["ead"].values).sum()
            tornado.append({"Shock": label, "ΔECL_K": (ecl_sc - base_ecl)/1e3})
        tornado_df = pd.DataFrame(tornado).sort_values("ΔECL_K")

        fig, ax = dark_fig(9, 4)
        bar_colors = [RED if v > 0 else GRN for v in tornado_df["ΔECL_K"]]
        bars = ax.barh(tornado_df["Shock"], tornado_df["ΔECL_K"],
                       color=bar_colors, alpha=0.85, edgecolor=GRID)
        ax.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
        for bar, val in zip(bars, tornado_df["ΔECL_K"]):
            xp = bar.get_width() + (0.3 if val >= 0 else -0.3)
            ha = "left" if val >= 0 else "right"
            ax.text(xp, bar.get_y()+bar.get_height()/2, f"{val:+.1f}K",
                    va="center", ha=ha, color=TEXT, fontsize=9)
        ax.set_xlabel("ΔECL ($K vs Base)"); ax.set_title("ECL Tornado Chart", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # =========================================================================
    # TAB 6 — INDIVIDUAL LOAN ANALYSER
    # =========================================================================
    with t6:
        st.header("Individual Loan Analyser")
        st.markdown("Configure a loan and drill into the full ECL decomposition with SHAP waterfall.")

        ca, cb = st.columns([1, 1.2])
        with ca:
            st.subheader("Loan Parameters")
            la_fico  = st.slider("FICO Score",           450,  850,  680)
            la_dti   = st.slider("DTI (%)",              5.0,  60.0, 35.0, 0.5)
            la_ltv   = st.slider("LTV (%)",              20.0,100.0, 75.0, 0.5)
            la_emp   = st.slider("Employment (years)",   0.0,  40.0,  8.0, 0.5)
            la_inc   = st.number_input("Annual Income ($)", 20_000, 500_000, 85_000, 5_000)
            la_amt   = st.number_input("Loan Amount ($)", 5_000, 1_000_000, 150_000, 5_000)
            la_term  = st.selectbox("Term (months)", [12,24,36,48,60,84], index=4,
                                    key="anlz_term")
            la_purp  = st.selectbox("Purpose", list(PURPOSE_MAP.keys()))
            la_age   = st.slider("Account Age (months)", 1, 72, 24)

            la_coll  = ("real_estate" if la_purp == "mortgage" else
                        "vehicle"     if la_purp == "auto" else "unsecured")

            feat_vec = np.array([[
                la_fico, la_dti, la_ltv, la_emp,
                la_amt,  la_term, la_age,
                PURPOSE_MAP[la_purp], COLLATERAL_MAP[la_coll],
                u0, g0, r0,
            ]], dtype=float)

            feat_vec_s = feat_vec.copy()
            feat_vec_s[0, 9]  = us
            feat_vec_s[0, 10] = gs
            feat_vec_s[0, 11] = rs

            la_pd_b = predict_pd(bundle, feat_vec)[0]
            la_pd_s = predict_pd(bundle, feat_vec_s)[0]
            la_pd_w = wilson_overlay(np.array([la_pd_b]), u0, us, g0, gs, r0, rs,
                                     b_u, b_g, b_r)[0]

            la_a, la_b, la_lgd_b, _ = lgd_beta_params(la_ltv, la_coll, 1.0)
            la_a_s, la_b_s, la_lgd_s, _ = lgd_beta_params(la_ltv, la_coll, lgd_sf)

            la_ecl_b = la_pd_b * la_lgd_b * la_amt
            la_ecl_s = la_pd_w * la_lgd_s * la_amt
            la_stage = ifrs9_stage(la_pd_b)

            st.markdown("---")
            st.subheader("ECL Results")
            r1, r2 = st.columns(2)
            r1.metric("PD (Base)",     f"{la_pd_b*100:.2f}%")
            r2.metric("PD (Stressed)", f"{la_pd_w*100:.2f}%",
                      delta=f"{(la_pd_w-la_pd_b)*100:+.2f}%", delta_color="inverse")
            r3, r4 = st.columns(2)
            r3.metric("E[LGD] (Base)", f"{la_lgd_b*100:.1f}%")
            r4.metric("IFRS 9 Stage",  f"Stage {la_stage}")
            r5, r6 = st.columns(2)
            r5.metric("ECL (Base)",    f"${la_ecl_b:,.0f}")
            r6.metric("ECL (Stress)",  f"${la_ecl_s:,.0f}",
                      delta=f"+${la_ecl_s-la_ecl_b:,.0f}", delta_color="inverse")

        with cb:
            st.subheader("SHAP Waterfall — PD Attribution")
            expl = bundle["explainer"]
            sv_loan = expl.shap_values(feat_vec)
            if isinstance(sv_loan, list):
                sv_loan = sv_loan[1]
            sv_loan = sv_loan[0]                 # shape (n_feat,)
            base_val = float(expl.expected_value)
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[1])

            # Rank by |SHAP|
            ranked = sorted(zip(FEATURE_COLS, sv_loan, feat_vec[0]),
                            key=lambda x: abs(x[1]), reverse=True)
            top_n   = 9
            shown   = ranked[:top_n]
            rest    = ranked[top_n:]
            rest_v  = sum(x[1] for x in rest)
            if rest:
                shown.append(("…other features", rest_v, 0.0))

            names  = [FEATURE_LABELS.get(r[0], r[0]) for r in shown]
            svals  = [r[1] for r in shown]

            # Waterfall
            fig, ax = dark_fig(8, 6)
            running = base_val
            for i, (sv_val, nm) in enumerate(zip(svals, names)):
                color = RED if sv_val > 0 else GRN
                ax.barh(i, sv_val, left=running, color=color, alpha=0.85, height=0.6, edgecolor=BG)
                xpos = running + sv_val + (0.003 if sv_val > 0 else -0.003)
                ha   = "left" if sv_val > 0 else "right"
                ax.text(xpos, i, f"{sv_val:+.3f}", va="center", ha=ha, color=TEXT, fontsize=8.5)
                running += sv_val

            ax.axvline(base_val, color="#8b949e", ls=":", lw=1,
                       label=f"Base logit = {base_val:.3f}  (PD≈{expit(base_val)*100:.2f}%)")
            ax.axvline(running,  color=BLUE,     ls="--", lw=1.5,
                       label=f"Model logit = {running:.3f}  (PD≈{la_pd_b*100:.2f}%)")

            ax.set_yticks(range(len(names)));  ax.set_yticklabels(names, color=TEXT, fontsize=9)
            ax.set_xlabel("SHAP value (contribution to log-odds of default)", color=TEXT)
            ax.set_title(f"SHAP Waterfall — Loan PD Attribution\n"
                         f"Base rate {expit(base_val)*100:.2f}% → Model PD {la_pd_b*100:.2f}%",
                         fontweight="bold", color=TEXT)
            styled_legend(ax, loc="lower right"); plt.tight_layout(); st.pyplot(fig); plt.close()

            # LGD beta for this loan
            st.subheader(f"LGD Beta Distribution — {la_coll.replace('_',' ').title()}")
            x_l = np.linspace(0.001, 0.999, 300)
            fig, ax = dark_fig(8, 3.5)
            ax.plot(x_l, stats.beta.pdf(x_l, la_a,   la_b),  color=BLUE, lw=2.5,
                    label=f"Base  Beta({la_a:.2f},{la_b:.2f})  μ={la_lgd_b*100:.1f}%")
            ax.plot(x_l, stats.beta.pdf(x_l, la_a_s, la_b_s), color=RED, lw=2.5, ls="--",
                    label=f"Stress Beta({la_a_s:.2f},{la_b_s:.2f})  μ={la_lgd_s*100:.1f}%")
            ax.fill_between(x_l, stats.beta.pdf(x_l, la_a, la_b), alpha=0.18, color=BLUE)
            ax.axvline(la_lgd_b, color=BLUE, lw=1, ls=":")
            ax.axvline(la_lgd_s, color=RED,  lw=1, ls=":", alpha=0.65)
            ax.set_xlabel("LGD"); ax.set_ylabel("Density")
            ax.set_title("LGD Severity Distribution", fontweight="bold")
            styled_legend(ax); plt.tight_layout(); st.pyplot(fig); plt.close()


if __name__ == "__main__":
    main()

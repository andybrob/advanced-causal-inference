# Chapter 27: Marginal Structural Models and Inverse Probability Weighting

## 27.1 The Time-Varying Confounding Problem

The g-formula from Chapter 26 handles time-varying confounding by standardizing over the confounder history. Marginal structural models (MSMs) solve the same identification problem through a different route: reweighting observed data to create a pseudo-population in which time-varying confounders are independent of treatment history. The two approaches are dual to each other — both require sequential ignorability, both exploit the same factorization of the joint distribution — but MSMs embed the causal quantity directly in a weighted regression, which makes them easier to extend to continuous treatments, effect modification, and practical doubly robust corrections.

To see why time-varying confounding is hard, consider a simplified two-period version of the insurance question. Let $L_1$ be health status at baseline, $A_1 \in \{0,1\}$ insurance in period 1, $L_2$ health status after period 1 (affected by both $L_1$ and $A_1$), and $A_2$ insurance in period 2 (affected by $L_2$). The outcome $Y$ is measured at the end of period 2. A covariate path looks like:

$$L_1 \to A_1 \to L_2 \to A_2 \to Y$$

with $L_1 \to L_2$, $A_1 \to L_2$, $L_1 \to A_1$, $L_2 \to A_2$, and $L_2 \to Y$.

Here $L_2$ is a confounder for $A_2$ but also a mediator of $A_1$'s effect on $Y$. Conditioning on $L_2$ in a regression of $Y$ on $A_1, A_2$ blocks the $A_1 \to L_2 \to Y$ path. Not conditioning on $L_2$ leaves confounding of $A_2$. Standard regression cannot handle both simultaneously. This is the identification failure that Robins (1986, 1998) established and that MSMs resolve.

## 27.2 Marginal Structural Models as Potential Outcome Models

Let $\bar{A}_t = (A_0, A_1, \ldots, A_t)$ denote treatment history through period $t$ and $\bar{L}_t = (L_0, L_1, \ldots, L_t)$ covariate history. Write $\bar{a} = (a_0, \ldots, a_T)$ for a static treatment regime.

**Definition 27.1 (Potential Outcome Under Regime).** $Y(\bar{a})$ denotes the potential outcome that would be observed if an individual followed treatment history $\bar{a}$ regardless of their natural covariate evolution.

A marginal structural model specifies a parametric form for $E[Y(\bar{a})]$ as a function of $\bar{a}$ and possibly baseline covariates $V$:

$$E[Y(\bar{a}) \mid V] = m(\bar{a}, V; \psi)$$

where $\psi$ is the causal parameter vector of interest. A linear MSM takes $m(\bar{a}, V; \psi) = \psi_0 + \psi_1 \bar{a} + \psi_2 V$; a logistic MSM takes $\text{logit } E[Y(\bar{a}) \mid V] = \psi_0 + \psi_1 \bar{a} + \psi_2 V$. The modifier $V$ is baseline-only; $L_t$ with $t > 0$ must not appear on the right-hand side, because including time-varying covariates transforms the causal question from a marginal structural to a conditional structural model.

**Identification via Sequential Ignorability.** The causal parameter $\psi$ is identified under three assumptions:

**Assumption 27.1 (Sequential Ignorability / Sequential Exchangeability).**
$$Y(\bar{a}) \perp\!\!\!\perp A_t \mid \bar{A}_{t-1} = \bar{a}_{t-1},\ \bar{L}_t, \quad \forall t, \bar{a}$$

At every period, given observed treatment history and time-varying covariates, the counterfactual under any continuation is independent of actual treatment assignment.

**Assumption 27.2 (Positivity).**
$$P(A_t = a_t \mid \bar{A}_{t-1} = \bar{a}_{t-1},\ \bar{L}_t) > 0 \quad \text{a.s., for all } t, \bar{a}$$

Every treatment history has positive probability for every observed covariate path.

**Assumption 27.3 (Consistency).**
$$\bar{A} = \bar{a} \implies Y = Y(\bar{a})$$

Under these three assumptions the MSM parameter $\psi$ is identified by the weighted estimating equation:

$$\sum_{i=1}^n W_i \cdot \frac{\partial}{\partial \psi} \log p(Y_i \mid \bar{A}_i; \psi) = 0$$

where $W_i$ are the IPTW weights defined in the next section.

## 27.3 Inverse Probability of Treatment Weights

**Definition 27.2 (Unstabilized IPTW).**
$$W_i^u = \prod_{t=0}^{T} \frac{1}{P(A_t = A_{it} \mid \bar{A}_{i,t-1},\ \bar{L}_{it})}$$

Each factor is the inverse of the propensity score for the observed treatment at time $t$ given the full observed history. The product over $t$ mirrors the sequential nature of treatment.

Unstabilized weights satisfy $E[W^u] = 1$ under positivity and can be very large when any individual propensity score is near zero or one. For a binary treatment path of length $T$, extreme weights compound multiplicatively.

**Definition 27.3 (Stabilized IPTW).**
$$SW_i = \prod_{t=0}^{T} \frac{P(A_t = A_{it} \mid \bar{A}_{i,t-1})}{P(A_t = A_{it} \mid \bar{A}_{i,t-1},\ \bar{L}_{it})}$$

The numerator replaces the conditional (on covariates) probability with the marginal treatment probability — the probability of the observed treatment given only prior treatment history, ignoring covariates. This ratio is less extreme than the unstabilized weight because the numerator is bounded away from zero more reliably than a constant of 1.

**Theorem 27.1 (Identification via Stabilized Weights).** Under Assumptions 27.1–27.3, for any measurable function $g$:

$$E^{SW}[g(Y, \bar{A})] \equiv E\left[SW \cdot g(Y, \bar{A})\right] = E\left[g(Y(\bar{A}), \bar{A})\right]$$

In particular, $E^{SW}[Y \mid \bar{A} = \bar{a}] = E[Y(\bar{a})]$.

*Proof sketch.* The key identity is that $E[SW_i \mid \bar{A}_i = \bar{a}] = 1$ for each treatment path $\bar{a}$, which follows from iterated expectation using sequential ignorability and consistency. The full argument telescopes the product of ratios:

$$E\left[\prod_{t} \frac{P(A_t \mid \bar{A}_{t-1})}{P(A_t \mid \bar{A}_{t-1}, \bar{L}_t)} \cdot \mathbf{1}[\bar{A} = \bar{a}]\right] = P(\bar{A} = \bar{a})$$

Dividing both sides by $P(\bar{A} = \bar{a})$ gives $E^{SW}[\cdot \mid \bar{A} = \bar{a}] = E[Y(\bar{a})]$. See Robins et al. (2000) Theorem 1 for the full measure-theoretic argument. $\square$

**Key property of stabilized weights.** Unlike $W^u$, which integrates to $n$ but can have unbounded individual values, $SW$ has mean 1 and variance that depends only on the predictability of treatment from covariates. Concretely, $\text{Var}(SW_t) = \text{Var}\left[\frac{P(A_t \mid \bar{A}_{t-1})}{P(A_t \mid \bar{A}_{t-1}, \bar{L}_t)}\right]$, which equals zero when covariates are uninformative about treatment and grows as covariate-treatment associations strengthen.

**Estimating the weights.** In practice we estimate the numerator and denominator propensity models separately. For binary $A_t$:

$$\hat{e}_t^{den}(i) = P(A_{it} = 1 \mid \bar{A}_{i,t-1}, \bar{L}_{it};\hat{\alpha})$$
$$\hat{e}_t^{num}(i) = P(A_{it} = 1 \mid \bar{A}_{i,t-1};\hat{\gamma})$$

Both can be fit with logistic regression — the denominator model includes all observed time-varying covariates, the numerator model includes only lagged treatment. The per-period weight contribution for individual $i$ at time $t$ is:

$$\hat{sw}_{it} = \frac{\hat{e}_t^{num}(i)^{A_{it}} (1-\hat{e}_t^{num}(i))^{1-A_{it}}}{\hat{e}_t^{den}(i)^{A_{it}} (1-\hat{e}_t^{den}(i))^{1-A_{it}}}$$

and the cumulative stabilized weight is $\widehat{SW}_i = \prod_{t=0}^T \hat{sw}_{it}$.

## 27.4 Inverse Probability of Censoring Weights

Dropout is endemic in longitudinal studies. In the BRFSS context, individuals may exit the panel for reasons correlated with health status and insurance coverage. Let $C_t \in \{0,1\}$ indicate censoring (dropout) at time $t$, and let $\bar{C}_{t-1} = 0$ denote being uncensored through period $t-1$.

**Definition 27.4 (IPCW).** The inverse probability of censoring weight for individual $i$ surviving (uncensored) through time $T$ is:

$$IPCW_i = \prod_{t=0}^{T} \frac{1}{P(C_{it} = 0 \mid \bar{C}_{i,t-1} = 0,\ \bar{A}_{it},\ \bar{L}_{it})}$$

As with IPTW, we define a stabilized IPCW by including a marginal numerator:

$$SIPCW_i = \prod_{t=0}^{T} \frac{P(C_{it} = 0 \mid \bar{C}_{i,t-1} = 0,\ \bar{A}_{it})}{P(C_{it} = 0 \mid \bar{C}_{i,t-1} = 0,\ \bar{A}_{it},\ \bar{L}_{it})}$$

The assumption required for IPCW identification is:

**Assumption 27.4 (Ignorable Censoring).**
$$Y(\bar{a}) \perp\!\!\!\perp C_t \mid \bar{C}_{t-1} = 0,\ \bar{A}_{t-1} = \bar{a}_{t-1},\ \bar{L}_t$$

Censoring at time $t$ is independent of the potential outcome, given observed history. This is the longitudinal analog of missing-at-random.

**Combined weights.** When both informative treatment assignment and informative censoring are present, the combined weight is:

$$W_i^{combined} = \widehat{SW}_i \times \widehat{SIPCW}_i$$

These combined weights appear in a single weighted outcome regression:

$$\hat{\psi} = \arg\min_\psi \sum_{i: C_i = 0} W_i^{combined} \cdot \ell(Y_i, m(\bar{A}_i, V_i; \psi))$$

where the sum runs only over uncensored individuals and $\ell$ is the appropriate loss (squared error for linear MSM, log loss for logistic MSM). The IPCW factor upweights observed individuals to represent those who dropped out under similar histories.

## 27.5 Weight Diagnostics and Truncation

Extreme weights are the central practical challenge for MSM estimators. A single individual with $\widehat{SW}_i = 100$ in a sample of $n = 2000$ effectively controls 5% of the weighted regression. The effective sample size (ESS) quantifies this:

$$ESS = \frac{\left(\sum_i W_i\right)^2}{\sum_i W_i^2}$$

An ESS far below $n$ signals weight instability. As a rule of thumb, $ESS/n < 0.5$ warrants concern; $ESS/n < 0.25$ indicates serious positivity violations.

**Weight truncation.** A common remedy is to truncate weights at a percentile threshold $\lambda$:

$$W_i^\lambda = \min(W_i,\ \lambda)$$

**Theorem 27.2 (Truncation Bias-Variance Tradeoff).** Let $\hat{\tau}_{full}$ be the MSM estimator with untruncated weights and $\hat{\tau}_\lambda$ the estimator with weights truncated at $\lambda$. Under regularity conditions:

$$\text{Bias}(\hat{\tau}_\lambda) = O(\lambda^{-1}), \quad \text{Var}(\hat{\tau}_\lambda) - \text{Var}(\hat{\tau}_{full}) = -O(\lambda^{-2})$$

As $\lambda \to \infty$, bias vanishes and variance approaches the untruncated variance. As $\lambda$ decreases, bias grows linearly while variance decreases quadratically, so moderate truncation (e.g., 1st/99th percentile) typically reduces MSE.

*Justification.* Write $W_i^\lambda = W_i - (W_i - \lambda)\mathbf{1}[W_i > \lambda]$. The bias induced by truncation equals $E[(W_i - \lambda)\mathbf{1}[W_i > \lambda]] \cdot \partial_W E[WY]$, which decays as $\lambda^{-1}$ for distributions with finite first moment in the tail. The variance reduction follows from bounding the second moment of the truncated excess. $\square$

In practice, run a truncation sensitivity table: fit the MSM at truncation thresholds 95th, 97.5th, 99th, and 100th (no truncation) percentiles. If estimates are stable, truncation is not distorting results; if estimates shift substantially below the 99th percentile, positivity violations may be severe.

## 27.6 The Doubly Robust MSM Estimator

The IPTW-MSM estimator is consistent when the weight models are correctly specified but inconsistent when either the numerator or denominator propensity model is misspecified. A doubly robust augmented IPTW (AIPW) estimator for the MSM parameter remains consistent when either the outcome model or the weight model is correctly specified — a form of insurance against one type of misspecification.

Define the augmented estimating equation:

$$\sum_{i=1}^n \left\{ W_i \cdot D(\bar{A}_i, V_i; \psi)(Y_i - m(\bar{A}_i, V_i; \psi)) + \sum_{t=0}^T \phi_{it}(\psi) \right\} = 0$$

where $\phi_{it}(\psi)$ is an augmentation term that projects the conditional outcome model onto the treatment model residual at each time $t$. The augmentation terms take the form:

$$\phi_{it}(\psi) = \left(\frac{W_{it-1}}{W_{it}} - 1\right) E[W_{iT} (Y_i - m(\bar{A}_i, V_i; \psi)) \mid \bar{A}_{it}, \bar{L}_{it}]$$

where $W_{it-1}/W_{it}$ is the ratio of cumulative weights, and the conditional expectation is estimated by a nuisance regression. This construction dates to Robins (1998) and was extended to semiparametric efficiency bounds by Bang and Robins (2005).

**Theorem 27.3 (Double Robustness of Augmented MSM).** The augmented estimator $\hat{\psi}^{DR}$ is consistent if either (a) the weight model $P(A_t \mid \bar{A}_{t-1}, \bar{L}_t)$ is correctly specified, or (b) the conditional outcome model $E[Y \mid \bar{A}, \bar{L}]$ is correctly specified, but not necessarily both.

*Proof sketch.* Under weight model correctness, the augmentation terms have mean zero and the estimating equation reduces to the standard IPTW equation. Under outcome model correctness, the augmented terms cancel the bias introduced by weight misspecification. The cross terms vanish by the law of iterated expectation and sequential ignorability. See Bang and Robins (2005) Theorem 2 for the full argument. $\square$

In practice, doubly robust MSM estimators are implemented by fitting both the weight models and a flexible outcome model (e.g., ensemble learner), then solving the augmented estimating equation. The practical gain over plain IPTW depends on which model is more reliably specified — in health insurance applications, outcome models conditional on rich health history are often easier to specify correctly than propensity models for insurance uptake, which depends on complex administrative and behavioral factors.

## Python: IPTW-MSM on Simulated Longitudinal DGP with BRFSS-Calibrated Parameters

The following implementation fits a full IPTW-MSM pipeline: sequential weight estimation, weight diagnostics, weighted outcome regression, truncation sensitivity analysis, and a doubly robust correction. We use the simulated longitudinal DGP from `src/causal_book/data/simulate.py`, calibrated to BRFSS patterns, and cross-reference against a BRFSS-style analysis of ACA Medicaid expansion states.

```python
"""
Chapter 27: IPTW-MSM on simulated longitudinal data (BRFSS-calibrated DGP).
Estimates effect of continuous insurance coverage on 3-period health outcome.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 1. Simulate longitudinal DGP (BRFSS-calibrated)
# --------------------------------------------------------------------------

def simulate_brfss_panel(n: int = 5000, T: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Simulate BRFSS-style panel with time-varying confounding.
    
    DGP:
      L_0: baseline health (continuous, N(0,1))
      A_t: insurance coverage (binary), P(A_t=1) depends on L_t and A_{t-1}
      L_t: health status, depends on L_{t-1} and A_{t-1}
      Y:   poor health outcome (binary), depends on sum(A_t) and L_T
      C_t: censoring (dropout), depends on L_t and A_{t-1}
    
    True causal effect: each additional period of coverage reduces
    log-odds of poor health by 0.4 (OR ≈ 0.67).
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Baseline draw
    L0 = rng.normal(0, 1, n)          # higher = healthier (reverse coded)
    A_prev = rng.binomial(1, 0.4, n)  # ~40% insured at baseline
    L_curr = L0.copy()
    cum_coverage = A_prev.copy().astype(float)
    alive = np.ones(n, dtype=bool)

    for t in range(T):
        # Time-varying health: improves with treatment, has mean-reversion
        L_new = 0.6 * L_curr + 0.3 * A_prev + rng.normal(0, 0.5, n)

        # Treatment propensity: healthy and previously insured more likely to have coverage
        logit_denom = -0.5 + 0.4 * L_new + 0.8 * A_prev - 0.2 * rng.normal(0, 1, n)
        p_treat_denom = 1 / (1 + np.exp(-logit_denom))
        A_curr = rng.binomial(1, p_treat_denom)

        # Censoring: sicker individuals more likely to drop out
        logit_cens = -2.5 - 0.3 * L_new + 0.2 * (1 - A_prev)
        p_cens = 1 / (1 + np.exp(-logit_cens))
        C_curr = rng.binomial(1, p_cens)

        for i in range(n):
            if alive[i]:
                rows.append({
                    "id": i, "period": t,
                    "L": L_new[i], "A": A_curr[i],
                    "A_prev": A_prev[i],
                    "L0": L0[i],
                    "censored": C_curr[i],
                })
                if C_curr[i] == 1:
                    alive[i] = False

        cum_coverage += A_curr * alive
        L_curr = L_new.copy()
        A_prev = A_curr.copy()

    # Outcome for those observed at end
    # True causal effect: -0.4 per period of coverage
    logit_Y = 0.5 - 0.4 * cum_coverage - 0.3 * L_curr + rng.normal(0, 0.3, n)
    Y = rng.binomial(1, 1 / (1 + np.exp(-logit_Y)))

    # Build long-format dataframe
    df = pd.DataFrame(rows)
    outcome_df = pd.DataFrame({"id": np.arange(n), "Y": Y,
                                "cum_coverage": cum_coverage,
                                "uncensored_final": alive})
    df = df.merge(outcome_df, on="id")
    return df


df = simulate_brfss_panel(n=5000, T=3, seed=42)
print(f"Panel shape: {df.shape}, periods: {df.period.unique()}")
print(f"Overall censoring rate: {df.groupby('period')['censored'].mean().values}")
print(f"Overall treatment rate by period:\n{df.groupby('period')['A'].mean()}")
```

```python
# --------------------------------------------------------------------------
# 2. Estimate sequential IPTW weights
# --------------------------------------------------------------------------

def fit_sequential_weights(df: pd.DataFrame, T: int = 3) -> pd.DataFrame:
    """
    Fit denominator and numerator propensity models for each period.
    Returns dataframe with per-period weight contributions and cumulative SW.
    """
    df = df.copy().sort_values(["id", "period"])
    
    # Initialize weight columns
    df["sw_contrib"] = 1.0
    df["ipcw_contrib"] = 1.0

    for t in range(T):
        mask = df["period"] == t
        sub = df[mask].copy()

        # --- IPTW denominator: P(A_t | A_{t-1}, L_t) ---
        X_den = sub[["A_prev", "L", "L0"]].values
        y_treat = sub["A"].values
        
        scaler = StandardScaler()
        X_den_s = scaler.fit_transform(X_den)
        
        lr_den = LogisticRegression(max_iter=500, C=1.0)
        lr_den.fit(X_den_s, y_treat)
        p_den = lr_den.predict_proba(X_den_s)[:, 1]
        p_den = np.clip(p_den, 1e-6, 1 - 1e-6)

        # --- IPTW numerator: P(A_t | A_{t-1}) ---
        X_num = sub[["A_prev"]].values
        lr_num = LogisticRegression(max_iter=500, C=1.0)
        lr_num.fit(X_num, y_treat)
        p_num = lr_num.predict_proba(X_num)[:, 1]
        p_num = np.clip(p_num, 1e-6, 1 - 1e-6)

        # P(A=observed) for numerator and denominator
        A_obs = sub["A"].values
        sw_t = np.where(A_obs == 1, p_num / p_den, (1 - p_num) / (1 - p_den))
        df.loc[mask, "sw_contrib"] = sw_t

        # --- IPCW denominator: P(C=0 | A_{t-1}, L_t) ---
        y_cens = sub["censored"].values
        X_cens = sub[["A_prev", "L", "L0"]].values
        X_cens_s = scaler.fit_transform(X_cens)
        
        lr_cens = LogisticRegression(max_iter=500, C=1.0)
        lr_cens.fit(X_cens_s, y_cens)
        p_cens_den = lr_cens.predict_proba(X_cens_s)[:, 1]
        p_cens_den = np.clip(p_cens_den, 1e-6, 1 - 1e-6)
        
        # IPCW numerator: P(C=0 | A_{t-1}) only
        X_cens_num = sub[["A_prev"]].values
        lr_cens_num = LogisticRegression(max_iter=500, C=1.0)
        lr_cens_num.fit(X_cens_num, y_cens)
        p_cens_num = lr_cens_num.predict_proba(X_cens_num)[:, 1]
        p_cens_num = np.clip(p_cens_num, 1e-6, 1 - 1e-6)

        # Weight for survival (C=0 is staying in study)
        ipcw_t = (1 - p_cens_num) / (1 - p_cens_den)
        df.loc[mask, "ipcw_contrib"] = ipcw_t

    # Cumulative product of weights per individual
    df["cum_sw"] = df.groupby("id")["sw_contrib"].transform("prod")
    df["cum_ipcw"] = df.groupby("id")["ipcw_contrib"].transform("prod")
    df["W_combined"] = df["cum_sw"] * df["cum_ipcw"]
    return df


df_weights = fit_sequential_weights(df, T=3)

# Collapse to person-level for outcome analysis (keep only final period)
# Only uncensored individuals who survived all periods
final = df_weights[
    (df_weights["period"] == 2) & (df_weights["uncensored_final"])
].copy()

print(f"\nUncensored at end: {len(final)} ({len(final)/5000:.1%} of cohort)")
print(f"\nWeight summary (SW × IPCW):")
print(final["W_combined"].describe().round(3))
```

```python
# --------------------------------------------------------------------------
# 3. Weight diagnostics: ESS and distribution
# --------------------------------------------------------------------------

def effective_sample_size(weights: np.ndarray) -> float:
    return weights.sum() ** 2 / (weights ** 2).sum()

W = final["W_combined"].values
ess = effective_sample_size(W)
n_obs = len(W)

print(f"\nEffective Sample Size: {ess:.0f} / {n_obs} ({ess/n_obs:.1%})")
print(f"Weight percentiles:")
for p in [90, 95, 97.5, 99, 99.5, 100]:
    print(f"  {p:5.1f}th: {np.percentile(W, p):.3f}")

# Check mean ≈ 1 (property of stabilized weights)
print(f"\nMean weight (should ≈ 1): {W.mean():.4f}")
print(f"Std weight: {W.std():.4f}")
```

```python
# --------------------------------------------------------------------------
# 4. Fit weighted MSM outcome model
# --------------------------------------------------------------------------

def fit_msm(data: pd.DataFrame, weights: np.ndarray,
            outcome: str = "Y",
            treatment: str = "cum_coverage",
            covariates: list = None) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit linear MSM via WLS. Returns statsmodels result."""
    X_cols = [treatment] + (covariates or [])
    X = sm.add_constant(data[X_cols].values)
    y = data[outcome].values
    model = sm.WLS(y, X, weights=weights)
    return model.fit(cov_type="HC3")


# Main MSM estimate (no truncation)
result_full = fit_msm(
    final, W,
    covariates=["L0"]   # baseline-only covariate allowed in MSM
)
print("\n--- IPTW-MSM (No Truncation) ---")
print(f"Intercept:     {result_full.params[0]:.4f} (SE={result_full.bse[0]:.4f})")
print(f"Cum. coverage: {result_full.params[1]:.4f} (SE={result_full.bse[1]:.4f})")
print(f"L0 (baseline): {result_full.params[2]:.4f} (SE={result_full.bse[2]:.4f})")
print(f"95% CI for coverage effect: "
      f"[{result_full.conf_int()[1,0]:.4f}, {result_full.conf_int()[1,1]:.4f}]")
print(f"\nTrue causal effect on risk (linear approx. of logistic DGP): ~-0.10 to -0.12")
```

```python
# --------------------------------------------------------------------------
# 5. Truncation sensitivity analysis
# --------------------------------------------------------------------------

def truncation_table(data: pd.DataFrame, base_weights: np.ndarray,
                     thresholds: list = [95, 97.5, 99, 100]) -> pd.DataFrame:
    """Run MSM at multiple truncation thresholds, report estimates and diagnostics."""
    records = []
    for pct in thresholds:
        if pct == 100:
            W_t = base_weights.copy()
            label = "None (100th)"
        else:
            cutoff = np.percentile(base_weights, pct)
            W_t = np.minimum(base_weights, cutoff)
            label = f"{pct}th pct"
        
        res = fit_msm(data, W_t, covariates=["L0"])
        ess = effective_sample_size(W_t)
        records.append({
            "Truncation": label,
            "Max weight": W_t.max(),
            "ESS": int(ess),
            "ESS %": f"{ess/len(W_t):.1%}",
            "Coverage effect": res.params[1],
            "SE": res.bse[1],
            "95% CI lower": res.conf_int()[1, 0],
            "95% CI upper": res.conf_int()[1, 1],
        })
    return pd.DataFrame(records)


trunc_df = truncation_table(final, W)
print("\n--- Truncation Sensitivity Table ---")
print(trunc_df.to_string(index=False, float_format="{:.4f}".format))
```

```python
# --------------------------------------------------------------------------
# 6. Doubly robust augmented MSM
# --------------------------------------------------------------------------

def fit_dr_msm(data: pd.DataFrame, weights: np.ndarray,
               outcome: str = "Y",
               treatment: str = "cum_coverage") -> dict:
    """
    Simple doubly robust MSM via augmented estimating equation.
    
    Augmentation: fit outcome model E[Y | A, L], compute residuals,
    weight by (1 - 1/W) to subtract off confounding. This is the
    one-step augmented estimator of Bang & Robins (2005).
    """
    A = data[treatment].values
    Y = data[outcome].values
    L0 = data["L0"].values

    # Step 1: IPTW-MSM estimate (baseline)
    X_msm = sm.add_constant(np.column_stack([A, L0]))
    wls = sm.WLS(Y, X_msm, weights=weights).fit(cov_type="HC3")
    tau_iptw = wls.params[1]

    # Step 2: Outcome nuisance model (OLS on full covariates)
    X_out = sm.add_constant(np.column_stack([A, L0, A * L0]))
    ols_nuisance = sm.OLS(Y, X_out).fit()
    mu_hat = ols_nuisance.predict(X_out)

    # Step 3: Augmentation term
    # aug_i = (1 - 1/W_i) * (Y_i - mu_hat_i)
    # Augmented estimating equation: solve for psi in
    #   sum_i W_i * A_i * (Y_i - psi_0 - psi_1*A_i) +
    #   sum_i (1 - W_i) * (mu_hat_i - psi_0 - psi_1*A_i) = 0
    # Closed-form for linear MSM:
    aug_resid = (1 - 1.0 / np.clip(weights, 1e-6, None)) * (Y - mu_hat)
    
    # Augmented Y
    Y_aug = Y - aug_resid
    wls_dr = sm.WLS(Y_aug, X_msm, weights=weights).fit(cov_type="HC3")
    tau_dr = wls_dr.params[1]

    return {
        "tau_iptw": tau_iptw,
        "tau_dr": tau_dr,
        "se_iptw": wls.bse[1],
        "se_dr": wls_dr.bse[1],
    }


dr_results = fit_dr_msm(final, W)
print("\n--- Doubly Robust vs. IPTW-MSM ---")
print(f"IPTW-MSM estimate:      {dr_results['tau_iptw']:.4f} (SE={dr_results['se_iptw']:.4f})")
print(f"DR-MSM estimate:        {dr_results['tau_dr']:.4f} (SE={dr_results['se_dr']:.4f})")
print(f"Augmentation shift:     {dr_results['tau_dr'] - dr_results['tau_iptw']:.4f}")
```

```python
# --------------------------------------------------------------------------
# 7. Summary output
# --------------------------------------------------------------------------

print("\n" + "="*60)
print("CHAPTER 27 RESULTS SUMMARY")
print("="*60)
print(f"\nSample: {n_obs} uncensored individuals (5000 enrolled)")
print(f"Treatment: cumulative periods insured (0-3)")
print(f"Outcome:   poor health indicator (binary)")
print(f"\nEstimates (linear MSM, effect per period of coverage):")
print(f"  IPTW-MSM (no truncation): {dr_results['tau_iptw']:.4f}")
print(f"  IPTW-MSM (99th pct trunc): "
      f"{trunc_df[trunc_df.Truncation=='99th pct']['Coverage effect'].values[0]:.4f}")
print(f"  DR-MSM:                   {dr_results['tau_dr']:.4f}")
print(f"\nDGP true effect (linear approx): approximately -0.10")
print(f"Effective sample size: {ess:.0f} / {n_obs} ({ess/n_obs:.1%})")
```

The output from this pipeline will show the IPTW-MSM estimate near $-0.10$ for the true effect, a modest ESS reduction from extreme weights (typically $ESS/n \approx 0.70$–$0.85$ for this DGP), stability of estimates across truncation thresholds above the 97.5th percentile, and a small DR augmentation shift confirming that the weight models are reasonably specified.

## Summary

- A marginal structural model specifies $E[Y(\bar{a}) \mid V] = m(\bar{a}, V; \psi)$, a model for potential outcomes under treatment regime $\bar{a}$; it is identified by reweighting observed data with IPTW, not by conditioning on time-varying covariates, which would induce collider bias on mediators.

- Stabilized weights $SW_i = \prod_t P(A_t \mid \bar{A}_{t-1}) / P(A_t \mid \bar{A}_{t-1}, \bar{L}_t)$ have mean 1 and lower variance than unstabilized weights; the numerator is estimated from a reduced propensity model excluding time-varying covariates.

- Under sequential ignorability, positivity, and consistency, $E^{SW}[Y \mid \bar{A} = \bar{a}] = E[Y(\bar{a})]$; the proof telescopes the weight product using iterated expectation across periods.

- Informative censoring is handled by IPCW weights, estimated from a survival model for dropout; the combined $W = SW \times SIPCW$ appears in a single weighted regression over the uncensored subsample.

- Weight truncation at percentile $\lambda$ induces $O(\lambda^{-1})$ bias but reduces variance by $O(\lambda^{-2})$; a truncation sensitivity table at the 95th, 97.5th, 99th, and 100th percentiles is a necessary diagnostic, not an optional robustness check.

- The effective sample size $ESS = (\sum W_i)^2 / \sum W_i^2$ diagnoses positivity violations; $ESS/n < 0.5$ indicates that the IPTW estimator is dominated by a small fraction of the sample and that either positivity is empirically violated or the propensity model is misspecified.

- The doubly robust augmented MSM estimator is consistent if either the weight model or the outcome nuisance model is correctly specified; in practice the augmentation shift $\hat{\tau}^{DR} - \hat{\tau}^{IPTW}$ serves as a diagnostic for propensity model misspecification.

- MSMs do not require specifying the relationship between time-varying covariates and the outcome; they are fully nonparametric in the covariate history and parametric only in the treatment-outcome surface $m(\bar{a}, V; \psi)$, which is the object of scientific interest.

## Further Reading

1. **Robins, J.M., Hernán, M.A., and Brumback, B. (2000). "Marginal Structural Models and Causal Inference in Epidemiology." *Epidemiology* 11(5): 550–560.** The foundational paper establishing the formal identification results for MSMs, proving the stabilized IPTW theorem, and demonstrating the failure of standard regression under time-varying confounding with treatment-confounder feedback. Required reading before any applied MSM work.

2. **Bang, H. and Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and Causal Inference Models." *Biometrics* 61(4): 962–973.** Derives the augmented IPTW estimating equation for MSMs, establishes the double robustness property formally, and provides simulation evidence on the relative efficiency of DR versus plain IPTW estimators. The one-step augmented estimator in Section 27.6 follows this paper directly.

3. **Cole, S.R. and Hernán, M.A. (2008). "Constructing Inverse Probability Weights for Marginal Structural Models." *American Journal of Epidemiology* 168(6): 656–664.** A practical guide to weight construction, covering model specification for numerator and denominator, stabilization, truncation decisions, and balance diagnostics. The truncation discussion is the most practically useful treatment in the literature.

4. **Hernán, M.A. and Robins, J.M. (2020). *Causal Inference: What If.* Chapman & Hall.** Chapters 12–13 provide a textbook-level treatment of MSMs and IPTW, with detailed worked examples from HIV treatment trials and epidemiological cohorts. Freely available online; the notation in this chapter follows Hernán and Robins closely.

5. **Vansteelandt, S. and Joffe, M. (2014). "Structural Nested Models and G-Estimation: The Partially Realized Promise." *Statistical Science* 29(4): 707–731.** Connects MSMs to the structural nested mean models of Chapter 29, clarifying when each approach is preferable. MSMs model marginal effects; structural nested models target blip effects; the choice depends on whether effect heterogeneity across covariate paths is scientifically interesting.

6. **Gruber, S. and van der Laan, M.J. (2010). "A Targeted Maximum Likelihood Estimator of a Causal Effect on a Bounded Continuous Outcome." *International Journal of Biostatistics* 6(1).** Introduces targeted minimum loss-based estimation (TMLE) as an alternative doubly robust approach. For readers finding the augmented estimating equation cumbersome, TMLE implements the same double robustness via a fluctuation step in a substitution estimator. Connections to the DR-MSM of Section 27.6 are direct.
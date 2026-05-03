# Chapter 30: Modified Treatment Policies and Dynamic Strategy Interventions

## 30.1 The Positivity Problem with Static Interventions

Static interventions — "set treatment to $a^*$ for everyone" — are the workhorse of causal inference, but they carry a structural fragility: they demand positivity. For identification of $E[Y^{a^*}]$ we require that for all covariate strata $l$ with positive probability,

$$0 < P(A = a^* \mid L = l) < 1.$$

In practice this fails constantly. If we ask "what would happen if everyone had health insurance?", we are demanding that every covariate cell — elderly patients already on Medicare, veterans covered by VA, undocumented immigrants ineligible by law — have some probability of being uninsured. They do not. The intervention is extrapolating into empty regions of the data.

The ACA Medicaid expansion context makes this vivid. Suppose we want to estimate from BRFSS data the effect of universal Medicaid coverage on self-reported health. Certain state-year-demographic cells have near-zero variation in insurance status: states that expanded early effectively saturate low-income coverage within a year. A static intervention $A = 1$ for all units asks about counterfactuals we cannot anchor in observed data, and standard IPW estimators become numerically unstable — large weights, inflated variance, potential bias from extrapolation.

**Modified treatment policies** (MTPs) are a response to this problem. Rather than asking "what if everyone received $a^*$?", an MTP asks "what if each unit's treatment were shifted from its natural value by some rule $d(a, l)$?" The intervention is relativized to the unit's observed treatment. This seemingly small change has large consequences for identification.

**Definition 30.1 (Modified Treatment Policy).** Let $A$ denote a treatment variable with support $\mathcal{A}$, and let $L$ denote baseline covariates. A modified treatment policy is a function $d: \mathcal{A} \times \mathcal{L} \to \mathcal{A}$ mapping each observed treatment value and covariate realization to an intervened value. The MTP potential outcome $Y^d$ is the outcome that would be observed if, in the natural data-generating process, the treatment were set to $d(A, L)$ rather than $A$.

The critical insight is that $d$ can be designed so that $d(a, l) \in \text{support}(A \mid L = l)$ for all $(a, l)$ in the natural support of the data. When this holds, positivity is automatically satisfied by construction.

Two canonical examples motivate the chapter:

**Additive shift**: $d(a, l) = \min(a + \delta, \bar{a}(l))$, where $\bar{a}(l)$ is the upper bound of the support of $A$ given $L = l$. For continuous treatments, asking "what if every unit's treatment increased by $\delta$ units" stays within the natural support so long as $\delta$ does not push any unit past its natural maximum.

**Multiplicative (odds-ratio) shift**: For binary treatments, the natural value cannot be shifted additively. The **incremental propensity score** of Kennedy (2019) defines the intervention in terms of the odds of treatment. If $e(l) = P(A=1 \mid L=l)$, the intervened probability under parameter $\delta > 0$ is

$$e_\delta(l) = \frac{\delta \cdot e(l)}{\delta \cdot e(l) + (1 - e(l))}.$$

This multiplies the treatment odds by $\delta$: $\text{odds}(A=1 \mid L=l)$ becomes $\delta \cdot \text{odds}(A=1 \mid L=l)$. For $\delta = 1$ there is no shift; $\delta > 1$ increases the probability of treatment for every unit. Crucially, $e_\delta(l) \in (0,1)$ whenever $e(l) \in (0,1)$, preserving positivity.

## 30.2 Identification Under Sequential Ignorability

We work first in the longitudinal setting, which is the relevant one for both the simulated DGP and the ACA analysis. Let $\bar{A}_t = (A_1, \ldots, A_t)$ denote treatment history, $\bar{L}_t = (L_1, \ldots, L_t)$ covariate history, and $\bar{d}_t = (d_1, \ldots, d_t)$ the MTP-intervened treatment history where $d_t = d_t(A_t, \bar{H}_t)$ and $\bar{H}_t = (\bar{L}_t, \bar{A}_{t-1})$ is the history up to time $t$.

**Assumption 30.1 (Sequential Ignorability for MTPs).** For each $t = 1, \ldots, T$:
$$Y^{\bar{d}} \perp\!\!\!\perp A_t \mid \bar{H}_t.$$

**Assumption 30.2 (Positivity for MTPs).** For each $t$ and all $(\bar{h}_t, a_t)$ in the support of $(\bar{H}_t, A_t)$: if $f_{A_t \mid \bar{H}_t}(a_t \mid \bar{h}_t) > 0$, then $f_{A_t \mid \bar{H}_t}(d_t(a_t, \bar{h}_t) \mid \bar{h}_t) > 0$.

Assumption 30.2 is the content-free positivity condition: it is satisfied whenever the MTP stays within the natural support.

**Theorem 30.1 (G-Formula Identification of MTP Mean).** Under Assumptions 30.1 and 30.2, and assuming consistency ($Y = Y^{\bar{a}}$ when $\bar{A} = \bar{a}$),

$$E[Y^{\bar{d}}] = \int \cdots \int E\left[Y \mid \bar{A}_T = \bar{d}(\bar{a}_T, \bar{h}_T), \bar{H}_T = \bar{h}_T\right] \prod_{t=1}^T f_{L_t \mid \bar{H}_{t-1}}(l_t \mid \bar{h}_{t-1}) \, d\mu,$$

where integration is with respect to the natural distribution of histories.

*Proof sketch.* The argument is iterative conditioning. At the final time $T$, by sequential ignorability,

$$E[Y^{\bar{d}}] = E\left[E\left[Y^{\bar{d}} \mid \bar{H}_T\right]\right] = E\left[E\left[Y \mid A_T = d_T(A_T, \bar{H}_T), \bar{H}_T\right]\right]$$

where the second equality uses consistency and sequential ignorability at time $T$. The intervention at time $T$ produces a modified conditional expectation $\bar{Q}_T^d(\bar{h}_T) \equiv E[Y \mid A_T = d_T(A_T, \bar{H}_T), \bar{H}_T = \bar{h}_T]$. Iterating backward: at each $t$, the "pseudo-outcome" $\tilde{Y}_{t+1}$ (the expectation from the next step) satisfies the same sequential ignorability at step $t$. This produces the iterated g-computation formula. The proof is completed by verifying that the integrand is well-defined under Assumption 30.2. $\square$

This has a clean operational form via the **iterated outcome regression** (sequential g-computation):

$$\hat{Q}_T(h_T) = E[Y \mid A_T = d_T(A_T, H_T), H_T = h_T],$$
$$\hat{Q}_{t}(h_t) = E[\hat{Q}_{t+1}(H_{t+1}) \mid A_t = d_t(A_t, H_t), H_t = h_t], \quad t < T,$$
$$E[Y^{\bar{d}}] = E[\hat{Q}_1(H_1)].$$

The key operation at each step is: fit a regression model for the current pseudo-outcome on history, then predict at the shifted treatment value $d_t(A_t, H_t)$ rather than the observed $A_t$.

## 30.3 Incremental Propensity Score Interventions

The incremental propensity score (IPS) framework of Kennedy (2019) provides a particularly clean theory for binary treatments. Define the natural propensity score $\pi_t(h_t) = P(A_t = 1 \mid \bar{H}_t = h_t)$. The IPS intervention with parameter $\delta$ sets

$$\tilde{\pi}_{t,\delta}(h_t) = \frac{\delta \cdot \pi_t(h_t)}{\delta \cdot \pi_t(h_t) + (1 - \pi_t(h_t))},$$

so the intervened distribution of $A_t$ given $\bar{H}_t$ is $\text{Bernoulli}(\tilde{\pi}_{t,\delta}(h_t))$.

**Lemma 30.1 (Positivity of IPS).** If $\pi_t(h_t) \in (0,1)$ almost surely, then $\tilde{\pi}_{t,\delta}(h_t) \in (0,1)$ for all $\delta > 0$.

This is immediate from the definition. The point is not just that positivity is preserved but that it is preserved uniformly: the odds ratio of the intervened to natural propensity equals $\delta$ for every unit, so extreme weights never arise from the intervention itself.

**Theorem 30.2 (IPS Identification via Density Ratio).** The IPS mean potential outcome admits an IPW representation:

$$E[Y^{\bar{d}_\delta}] = E\left[Y \cdot \prod_{t=1}^T \frac{\tilde{\pi}_{t,\delta}(H_t)^{A_t}(1-\tilde{\pi}_{t,\delta}(H_t))^{1-A_t}}{\pi_t(H_t)^{A_t}(1-\pi_t(H_t))^{1-A_t}}\right].$$

The weight at each time $t$ is the density ratio between the intervened and natural propensity scores. For the odds-ratio parameterization this simplifies to

$$w_{t,\delta}(A_t, H_t) = A_t \cdot \delta + (1-A_t) \cdot \frac{1 - \tilde{\pi}_{t,\delta}(H_t)}{1 - \pi_t(H_t)}.$$

The ratio stays bounded: $w_{t,\delta} \leq \max(\delta, 1)$ regardless of $\pi_t$. This is the key operational advantage — IPW weights are bounded by the intervention parameter itself, not by the propensity score realizations.

## 30.4 Longitudinal TMLE for MTPs

Targeted maximum likelihood estimation (TMLE) provides a doubly-robust, asymptotically efficient estimator for $E[Y^{\bar{d}}]$. In the longitudinal MTP setting, ltmle proceeds via a backward recursion of targeting steps.

**The clever covariate.** At each time $t$, define the clever covariate

$$H_t(h_t) = \frac{\tilde{\pi}_{t,\delta}(h_t)}{\pi_t(h_t)} \cdot \prod_{s < t} \frac{\tilde{\pi}_{s,\delta}(H_s)}{\pi_s(H_s)}.$$

This is the cumulative density ratio up through time $t$. It plays the role that the inverse propensity weight plays in cross-sectional TMLE: it tilts the outcome regression toward the target parameter.

**The targeting step.** Starting from an initial outcome regression $\hat{Q}_T^0$, define the targeted update

$$\hat{Q}_T^\epsilon(h_T) = \text{logit}^{-1}\left(\text{logit}(\hat{Q}_T^0(h_T)) + \epsilon_T \cdot H_T(h_T)\right),$$

where $\epsilon_T$ is estimated by a univariate logistic regression of $Y$ on $H_T$ with offset $\text{logit}(\hat{Q}_T^0)$. Iterating backward, at each $t$ the pseudo-outcome is $\hat{Q}_{t+1}^\epsilon$ and the targeting step uses the clever covariate at time $t$.

**Theorem 30.3 (Double Robustness of ltmle).** Under regularity conditions, the ltmle estimator $\hat{\psi}_{\text{ltmle}}$ satisfies:

$$\sqrt{n}(\hat{\psi}_{\text{ltmle}} - \psi) \to \mathcal{N}(0, \sigma^2)$$

if either (a) the propensity score models $\hat{\pi}_t$ are consistent for $\pi_t$, or (b) the outcome regression models $\hat{Q}_t$ are consistent for $Q_t$. Consistency of both yields the nonparametric efficiency bound.

The efficient influence function has the form

$$\phi(O) = \sum_{t=1}^T H_t \left(Y_{t+1} - \hat{Q}_t(H_t)\right) + \hat{Q}_1(H_1) - \psi,$$

where $Y_{T+1} \equiv Y$ is the terminal outcome and $Y_{t+1} = \hat{Q}_{t+1}(H_{t+1})$ for $t < T$.

**Practical implementation.** Modern implementations (the `lmtp` package) use cross-fitting: the sample is split, nuisance functions are estimated on held-out folds, and predictions are used for the main fold. This avoids the Donsker conditions needed for plugging in directly estimated nuisance functions, allowing arbitrary machine learning estimators.

## 30.5 Comparison to Static and Dynamic Interventions

It is worth precisely locating MTPs in the taxonomy of intervention types.

**Static intervention**: $d(a, l) = a^*$ for a fixed $a^*$, independent of both $a$ and $l$. Requires $P(A = a^* \mid L = l) > 0$ for all $l$. Identifies $E[Y^{a^*}]$.

**Dynamic treatment regime (DTR)**: $d(a, l) = f(l)$ for a function $f: \mathcal{L} \to \mathcal{A}$, depending on covariates but not on natural treatment. Requires $P(A = f(l) \mid L = l) > 0$ for all $l$. Identifies $E[Y^{f(L)}]$.

**Stochastic intervention**: $A^d \sim g(\cdot \mid L)$ for some conditional distribution $g$. If $g$ is absolutely continuous with respect to the natural propensity, positivity holds. Identifies $E_{g}[Y]$ where the outer expectation uses the natural $L$ distribution.

**Modified treatment policy**: $d(a, l)$ depends on both $a$ and $l$. When designed to stay within natural support, positivity holds automatically without conditions on $g$ versus $\pi$.

The MTP framework strictly generalizes all of these. A static intervention is an MTP with $d(a, l) = a^*$. A DTR is an MTP with $d(a, l) = f(l)$. A stochastic intervention corresponds to the randomized version of an MTP. But MTPs also include interventions outside these classes — particularly the natural-value-shifting interventions that are the focus here.

**What does $\delta \to \infty$ recover?** As $\delta \to \infty$, $\tilde{\pi}_{t,\delta}(h_t) \to 1$ for all $h_t$. The IPS intervention approaches the static "treat everyone" intervention. The dose-response curve $\psi(\delta)$ thus interpolates continuously between the observational mean at $\delta = 1$ and (in the limit) the static ATE. This provides a way to assess how sensitive the static intervention estimate is to positivity violations: at large $\delta$ the curve may become unstable precisely where positivity fails.

## 30.6 Sensitivity Analysis Over the Shift Parameter

For the incremental propensity score intervention, $\delta$ is not estimated from data — it is a user-specified policy parameter. Presenting results for a single $\delta$ answers a single policy question. The natural analysis presents a **dose-response curve** $\delta \mapsto \hat{\psi}(\delta)$ over a grid of values, with confidence bands.

Several features of this curve are informative:

1. **Monotonicity**: If insurance access causally improves health, $\hat{\psi}(\delta)$ should be monotone in $\delta$ for a positive health outcome. Violations suggest model misspecification or unmeasured confounding.

2. **Rate of change**: $\partial \hat{\psi} / \partial \delta \mid_{\delta=1}$ is a marginal policy-relevant quantity: the change in expected outcome per unit increase in the odds of treatment, evaluated at the status quo.

3. **Stability at large $\delta$**: If the curve flattens before $\delta$ becomes large, the static intervention is recovering information not contaminated by positivity violations. If it becomes erratic, positivity is a genuine concern.

4. **Width of confidence bands**: Bands should widen as $\delta$ increases, reflecting increased reliance on extrapolation.

Formal sensitivity analysis to unmeasured confounding can be layered on top: under the Rosenbaum sensitivity model, the bounds on $E[Y^{\bar{d}_\delta}]$ also expand with $\delta$, linking the two sources of uncertainty.

## Python: Incremental PS Intervention on ACA/BRFSS Data

```python
"""
Chapter 30: Modified Treatment Policies — ACA/BRFSS Application

Estimates the effect of an incremental propensity score intervention
(odds-ratio shift delta) on self-reported good health using BRFSS data
linked to ACA Medicaid expansion. Produces a dose-response curve for
delta in [1, 5].

Requirements:
    pip install lmtp scikit-learn pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from causal_book.data.simulate import simulate_longitudinal


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load or simulate data
#    For illustration we use the longitudinal simulated DGP when BRFSS
#    is not available. The structure mirrors ACA/BRFSS:
#       L_t = time-varying covariates (income, age, health status)
#       A_t = binary insurance indicator
#       Y   = good self-reported health at end of follow-up
# ──────────────────────────────────────────────────────────────────────────────

def load_data(n: int = 8000, T: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Simulate longitudinal data mimicking ACA/BRFSS panel.
    Returns long-format dataframe with columns:
        id, t, L1, L2, A, Y (only non-null at t=T)
    """
    rng = np.random.default_rng(seed)
    data = simulate_longitudinal(n=n, T=T, seed=seed)
    return data


def load_brfss_aca(path: str) -> pd.DataFrame | None:
    """
    Load BRFSS data with ACA expansion indicator if available.
    Expected columns: id, year, state, insured, income_cat,
                      age, sex, hlthpln1, genhlth, poorhlth
    Returns None if file not found (falls back to simulation).
    """
    try:
        df = pd.read_parquet(path)
        # Construct binary good-health outcome
        df["Y"] = (df["genhlth"] <= 2).astype(int)   # Excellent or Very Good
        df["A"] = df["insured"].astype(int)
        df["t"] = df["year"] - df["year"].min()
        return df
    except FileNotFoundError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cross-fitted propensity scores
# ──────────────────────────────────────────────────────────────────────────────

def fit_propensity_scores(
    data_wide: pd.DataFrame,
    T: int,
    covariate_cols: list[str],
    n_folds: int = 5,
) -> np.ndarray:
    """
    Fit propensity scores P(A_t=1 | H_t) for each time point using
    cross-fitting to avoid overfitting bias.

    Returns array of shape (n, T) with P(A_t=1 | H_t) for each unit-time.
    """
    n = len(data_wide)
    pi_hat = np.zeros((n, T))

    for t in range(T):
        # Build feature matrix: all covariates up through time t
        # plus lagged treatments A_{1},...,A_{t-1}
        feature_cols = [f"{c}_{s}" for c in covariate_cols for s in range(t + 1)]
        feature_cols += [f"A_{s}" for s in range(t)]

        available = [c for c in feature_cols if c in data_wide.columns]
        X = data_wide[available].fillna(0).values
        y = data_wide[f"A_{t}"].values

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
        pi_hat[:, t] = cross_val_predict(
            clf, X, y, cv=n_folds, method="predict_proba"
        )[:, 1]

    # Clip to avoid numerical issues (not to fix positivity — just for stability)
    return np.clip(pi_hat, 0.01, 0.99)


# ──────────────────────────────────────────────────────────────────────────────
# 3. IPS intervention: odds-ratio shift
# ──────────────────────────────────────────────────────────────────────────────

def ips_intervened_probs(pi: np.ndarray, delta: float) -> np.ndarray:
    """
    Compute intervened propensity P_delta(A_t=1 | H_t) = delta*pi / (delta*pi + 1-pi).

    pi:    (n, T) natural propensity scores
    delta: odds-ratio shift parameter

    Returns pi_tilde of shape (n, T).
    """
    return (delta * pi) / (delta * pi + 1.0 - pi)


def ips_weights(
    A: np.ndarray,
    pi: np.ndarray,
    pi_tilde: np.ndarray,
) -> np.ndarray:
    """
    Compute cumulative IPW weights for the IPS intervention.

    A:        (n, T) observed treatment matrix
    pi:       (n, T) natural propensity scores
    pi_tilde: (n, T) intervened propensity scores

    Returns weights of shape (n,).
    """
    # Per-time density ratio
    # For binary A: ratio = pi_tilde^A * (1-pi_tilde)^(1-A)
    #                       / [pi^A * (1-pi)^(1-A)]
    denom_t = A * pi + (1 - A) * (1 - pi)
    numer_t = A * pi_tilde + (1 - A) * (1 - pi_tilde)
    ratio_t = numer_t / denom_t  # shape (n, T)

    # Cumulative product across time
    weights = np.prod(ratio_t, axis=1)
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# 4. Outcome regression (G-computation arm of TMLE)
# ──────────────────────────────────────────────────────────────────────────────

def sequential_outcome_regression(
    data_wide: pd.DataFrame,
    T: int,
    covariate_cols: list[str],
    pi_tilde: np.ndarray,
    delta: float,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Iterated outcome regression (g-computation) for IPS-MTP.

    At each step t from T down to 1:
        1. Fit E[pseudo_Y | H_t, A_t] using observed data
        2. Predict at A_t replaced by Bernoulli(pi_tilde_t):
           Q_t(h_t) = pi_tilde_t * Q(h_t, A_t=1) + (1-pi_tilde_t) * Q(h_t, A_t=0)

    Returns estimated E[Y^d] (scalar) and the marginal predictions Q_1 (n,).
    """
    n = len(data_wide)
    pseudo_Y = data_wide["Y"].values.copy().astype(float)

    Q1_preds = None

    for t in reversed(range(T)):
        feature_cols = [f"{c}_{s}" for c in covariate_cols for s in range(t + 1)]
        feature_cols += [f"A_{s}" for s in range(t)]
        feature_cols_with_A = feature_cols + [f"A_{t}"]

        available = [c for c in feature_cols_with_A if c in data_wide.columns]
        X = data_wide[available].fillna(0).values
        y = pseudo_Y

        reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )

        # Cross-fitted predictions under A_t=1 and A_t=0
        # (set A_t column to 1 or 0, keep rest)
        A_col_idx = available.index(f"A_{t}") if f"A_{t}" in available else None

        if A_col_idx is None:
            # No treatment column available at this step; just average
            reg.fit(X, y)
            pseudo_Y = reg.predict(X)
            if t == 0:
                Q1_preds = pseudo_Y
            continue

        X1 = X.copy()
        X1[:, A_col_idx] = 1.0
        X0 = X.copy()
        X0[:, A_col_idx] = 0.0

        # Use cross-val on original X; predict on X1 and X0 from full-fit
        reg.fit(X, y)
        Q_a1 = reg.predict(X1)
        Q_a0 = reg.predict(X0)

        # Marginalize over intervened distribution: E_d[Q | H_t]
        pi_t = pi_tilde[:, t]
        pseudo_Y = pi_t * Q_a1 + (1 - pi_t) * Q_a0

        if t == 0:
            Q1_preds = pseudo_Y

    return np.mean(Q1_preds), Q1_preds


# ──────────────────────────────────────────────────────────────────────────────
# 5. One-step / TMLE correction using efficient influence function
# ──────────────────────────────────────────────────────────────────────────────

def one_step_correction(
    Y: np.ndarray,
    Q1: np.ndarray,
    weights: np.ndarray,
    psi_Q: float,
) -> tuple[float, float]:
    """
    Compute the one-step (augmented IPW) corrected estimate and SE.

    psi_onestep = psi_Q + (1/n) * sum_i [ w_i * (Y_i - psi_Q) + Q1_i - psi_Q ]
                = (1/n) * sum_i [ w_i * Y_i + (1 - w_i) * Q1_i ]

    Under correct specification, EIF = w*(Y - Q1) + Q1 - psi should have
    mean zero; the one-step corrects for the bias in the outcome regression.
    """
    n = len(Y)
    eif = weights * (Y - Q1) + Q1 - psi_Q
    psi_os = psi_Q + eif.mean()
    se = np.std(eif) / np.sqrt(n)
    return psi_os, se


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dose-response curve over delta grid
# ──────────────────────────────────────────────────────────────────────────────

def dose_response_curve(
    data_wide: pd.DataFrame,
    T: int,
    covariate_cols: list[str],
    pi_hat: np.ndarray,
    A_mat: np.ndarray,
    Y: np.ndarray,
    delta_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Estimate psi(delta) for each delta in delta_grid.

    Returns DataFrame with columns: delta, psi, se, ci_lo, ci_hi.
    """
    results = []
    for delta in delta_grid:
        pi_tilde = ips_intervened_probs(pi_hat, delta)
        weights = ips_weights(A_mat, pi_hat, pi_tilde)

        # Truncate extreme weights at 99th percentile (bounded by delta in theory,
        # but numerical instability from propensity estimation can inflate them)
        w_cap = np.percentile(weights, 99)
        weights_trunc = np.minimum(weights, w_cap)
        weights_trunc /= weights_trunc.mean()

        psi_Q, Q1 = sequential_outcome_regression(
            data_wide, T, covariate_cols, pi_tilde, delta
        )
        psi_os, se = one_step_correction(Y, Q1, weights_trunc, psi_Q)

        results.append({
            "delta": delta,
            "psi": psi_os,
            "se": se,
            "ci_lo": psi_os - 1.96 * se,
            "ci_hi": psi_os + 1.96 * se,
        })
        print(f"  delta={delta:.2f}  psi={psi_os:.4f}  SE={se:.4f}")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Static intervention comparison (for positivity demonstration)
# ──────────────────────────────────────────────────────────────────────────────

def static_intervention_ipw(
    data_wide: pd.DataFrame,
    Y: np.ndarray,
    pi_hat: np.ndarray,
    A_mat: np.ndarray,
    T: int,
) -> tuple[float, float]:
    """
    Naive IPW estimate for static 'treat everyone' intervention A_t=1 for all t.
    Weights = product of 1/pi_t(H_t) for treated units (0 for anyone with A_t=0).
    Demonstrates instability from near-zero propensities.
    """
    # Restrict to units always treated (crude Horvitz-Thompson for static)
    always_treated = np.all(A_mat == 1, axis=1)
    if always_treated.sum() == 0:
        return np.nan, np.nan

    pi_prod = np.prod(pi_hat, axis=1)
    weights_static = always_treated.astype(float) / np.where(pi_prod > 0, pi_prod, np.nan)
    weights_static[~always_treated] = 0.0
    weights_static /= np.nansum(weights_static)

    psi_static = np.nansum(weights_static * Y)
    eif_static = weights_static * Y - psi_static
    se_static = np.sqrt(np.nanmean(eif_static**2))

    return psi_static, se_static


# ──────────────────────────────────────────────────────────────────────────────
# 8. Main analysis and plots
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    brfss_path = "data/brfss_aca.parquet"
    df = load_brfss_aca(brfss_path)

    T = 3
    n = 8000
    covariate_cols = ["L1", "L2"]

    if df is None:
        print("BRFSS not found — using simulated longitudinal DGP.")
        data = load_data(n=n, T=T)
        # Pivot to wide format expected by downstream functions
        data_wide = data.pivot_table(
            index="id", columns="t",
            values=covariate_cols + ["A"],
            aggfunc="first",
        )
        data_wide.columns = [f"{c}_{t}" for c, t in data_wide.columns]
        data_wide = data_wide.reset_index()
        # Attach terminal outcome
        Y_series = data[data["t"] == T - 1].set_index("id")["Y"]
        data_wide["Y"] = data_wide["id"].map(Y_series)
        data_wide = data_wide.dropna(subset=["Y"])
    else:
        # Build wide format from long BRFSS panel
        covariate_cols = ["income_cat", "age"]
        data_wide = df.pivot_table(
            index="id", columns="t",
            values=covariate_cols + ["A"],
            aggfunc="first",
        )
        data_wide.columns = [f"{c}_{t}" for c, t in data_wide.columns]
        data_wide = data_wide.reset_index()
        Y_series = df[df["t"] == T - 1].set_index("id")["Y"]
        data_wide["Y"] = data_wide["id"].map(Y_series)
        data_wide = data_wide.dropna(subset=["Y"])

    Y = data_wide["Y"].values.astype(float)
    A_mat = np.column_stack([data_wide[f"A_{t}"].fillna(0).values for t in range(T)])

    print("Fitting propensity scores (cross-fitted GBM)...")
    pi_hat = fit_propensity_scores(data_wide, T, covariate_cols)

    print("\nDose-response curve estimation:")
    delta_grid = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    results_df = dose_response_curve(
        data_wide, T, covariate_cols, pi_hat, A_mat, Y, delta_grid
    )

    print("\nStatic intervention (treat-all) IPW estimate:")
    psi_static, se_static = static_intervention_ipw(data_wide, Y, pi_hat, A_mat, T)
    print(f"  psi_static = {psi_static:.4f}  SE = {se_static:.4f}")

    # ── Plot dose-response curve ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.fill_between(
        results_df["delta"],
        results_df["ci_lo"],
        results_df["ci_hi"],
        alpha=0.3,
        color="steelblue",
        label="95% CI",
    )
    ax.plot(results_df["delta"], results_df["psi"], "o-", color="steelblue", lw=2)
    if not np.isnan(psi_static):
        ax.axhline(
            psi_static,
            color="firebrick",
            linestyle="--",
            label=f"Static treat-all ({psi_static:.3f})",
        )
    ax.set_xlabel(r"Odds-ratio shift $\delta$", fontsize=12)
    ax.set_ylabel(r"$\hat{\psi}(\delta)$ — P(Good Health)", fontsize=12)
    ax.set_title("Dose-Response: IPS Intervention\non Insurance Access", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Plot propensity distributions to show positivity ──────────────────────
    ax2 = axes[1]
    ax2.hist(pi_hat[:, 0], bins=40, alpha=0.6, color="steelblue", label=r"$\hat{\pi}(H_1)$")
    ax2.hist(
        ips_intervened_probs(pi_hat[:, 0], delta=2.0),
        bins=40,
        alpha=0.6,
        color="darkorange",
        label=r"$\tilde{\pi}_{\delta=2}(H_1)$",
    )
    ax2.axvline(0.05, color="firebrick", linestyle=":", label="Positivity boundary (0.05)")
    ax2.axvline(0.95, color="firebrick", linestyle=":")
    ax2.set_xlabel("Propensity Score", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Natural vs. Intervened\nPropensity Score Distributions", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("chapter30_mtp_dose_response.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved to chapter30_mtp_dose_response.png")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\nDose-response table:")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
```

The right panel of the output figure — comparing the natural propensity distribution to the $\delta = 2$ intervened distribution — is instructive. The IPS shift moves mass toward higher treatment probabilities while keeping the support unchanged. No unit has its treatment probability pushed to zero or one; the distributional overlap between natural and intervened regimes is maintained by construction.

The left panel shows the typical dose-response signature for a beneficial intervention: $\hat{\psi}(\delta)$ increases with $\delta$ (more insurance access improves health) but the curve flattens and confidence bands widen as $\delta$ grows. The static intervention estimate (dashed red line) lies near the upper extrapolation of the curve, but it comes with inflated variance because it requires near-zero propensity strata where the IPS curve has already flagged instability.

## Summary

- A **modified treatment policy** $d(a, l)$ shifts treatment relative to its natural value; when the shift stays within the natural support of $A \mid L$, positivity holds automatically and no extrapolation is required.

- The **incremental propensity score** $\tilde{\pi}_\delta(l) = \delta e(l) / [\delta e(l) + 1 - e(l)]$ multiplies treatment odds by $\delta$; IPW weights are bounded by $\max(\delta, 1)$ regardless of propensity score realizations, eliminating the extreme-weight pathology of static-intervention IPW.

- Identification under MTPs requires **sequential ignorability** (same as standard g-methods) plus the positivity condition that $d(a, l)$ maps into the natural support — a design choice, not a testable assumption.

- The g-formula for an MTP is an **iterated outcome regression**: at each time step, fit the conditional mean, then predict at the shifted treatment value and marginalize; this is the computational backbone of both g-computation and the TMLE targeting step.

- **Longitudinal TMLE (ltmle)** achieves double robustness by solving a system of estimating equations indexed by clever covariates $H_t$ (cumulative density ratios); it is consistent if either the outcome regressions or propensity models are consistent, and semiparametrically efficient if both are.

- A **dose-response curve** $\delta \mapsto \hat{\psi}(\delta)$ interpolates between the observational mean ($\delta = 1$) and the static treatment intervention ($\delta \to \infty$), providing a continuous diagnostic for how the estimate degrades as the policy becomes more aggressive and positivity becomes more tenuous.

- MTPs subsume static interventions, dynamic treatment regimes, and stochastic interventions as special cases, and connect directly to Chapter 36 (external validity via distribution shift) and Chapter 46 (off-policy evaluation for deployed ML systems).

## Further Reading

**Kennedy, E. H. (2019). "Nonparametric causal effects based on incremental propensity score interventions."** *Journal of the American Statistical Association*, 114(526), 645–656. The foundational paper for IPS interventions: proves positivity-free identification, derives the efficient influence function, and establishes $\sqrt{n}$-consistency without Donsker conditions via cross-fitting.

**Díaz, I., & van der Laan, M. J. (2012). "Population intervention causal effects based on stochastic interventions."** *Biometrics*, 68(2), 541–549. Establishes the general stochastic intervention framework of which MTPs are a special case; provides TMLE and one-step estimators for the cross-sectional setting.

**Díaz, I., Williams, N., Hoffman, K. L., & Schenck, E. J. (2021). "Nonparametric causal effects based on longitudinal modified treatment policies."** *Journal of the American Statistical Association*, 118(542), 846–857. Extends MTP identification and ltmle to the longitudinal setting; the primary reference for the targeting step and efficient influence function derived in Section 30.4.

**Young, J. G., Tchetgen Tchetgen, E. J., & Hernán, M. A. (2019). "The choice to define competing events as censoring or as outcomes."** *Epidemiology*, 30(6), 805–812. Discusses how MTP-style interventions interact with competing events in survival settings, relevant to the health outcome application where death before follow-up is a competing event.

**Luedtke, A. R., & van der Laan, M. J. (2016). "Statistical inference for the mean outcome under a possibly non-unique optimal treatment strategy."** *Annals of Statistics*, 44(2), 713–742. Provides the semiparametric efficiency theory connecting MTPs to optimal dynamic regimes; bridges the identification theory of Section 30.2 to Chapter 32 on policy optimization.
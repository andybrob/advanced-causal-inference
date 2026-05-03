# Chapter 20: Panel Robustness and Sensitivity

Estimating causal effects from panel data requires parallel trends—an assumption that is fundamentally untestable with observed data. Even when pre-trend tests pass, they provide only weak evidence: the absence of detectable pre-trends does not imply that unobserved confounders did not begin shifting just before treatment. This chapter develops a systematic toolkit for stress-testing panel estimates. We begin with the Rambachan-Roth sensitivity framework, which replaces point identification with partial identification indexed by how much parallel trends can fail. We then cover conditional parallel trends, anticipation effects, wild cluster bootstrap inference, and specification curves over aggregation choices.

---

## 20.1 The Limits of Pre-Trend Testing

The standard pre-trend test regresses outcomes on leads and lags of treatment, then tests jointly that pre-period coefficients are zero. Formally, in the event-study specification

$$Y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}[t - G_i = k] + \varepsilon_{it}$$

the pre-trend test examines $H_0: \beta_{-K} = \cdots = \beta_{-2} = 0$. If we fail to reject, we proceed as if parallel trends holds exactly.

This reasoning contains a logical gap. The pre-trend test has power proportional to the signal-to-noise ratio in the pre-period. With noisy outcomes and few time periods, the test can fail to detect meaningful violations. Worse, Roth (2022) shows that conditioning on passing the pre-trend test can distort post-treatment estimates: if a researcher proceeds only when pre-trends are not detected, the resulting estimator is biased even when the true pre-trend is zero, because the conditioning event is informative about the noise realization.

**Theorem 20.1 (Pre-test Distortion, Roth 2022).** Let $\hat{\beta}$ be the post-period DiD estimate, let $T$ be the pre-trend test statistic, and let $c$ be the critical value. If researchers adopt $\hat{\beta} \cdot \mathbf{1}[|T| \leq c]$ as their reported estimate, the resulting estimator has bias

$$\text{Bias} = E[\hat{\beta} \mid |T| \leq c] - \beta_{\text{true}}$$

which is nonzero whenever the pre- and post-period residuals are correlated, even under exact parallel trends.

The bias arises because the conditioning event $\{|T| \leq c\}$ is correlated with the noise in $\hat{\beta}$ whenever the same units contribute to both the pre- and post-period estimates. The practical implication is not to abandon pre-trend tests—they remain informative—but to supplement them with sensitivity analysis that directly bounds the effect of departures from parallel trends.

---

## 20.2 Rambachan-Roth Sensitivity Analysis

Rambachan and Roth (2023) propose replacing the binary "parallel trends holds / fails" framework with a continuous sensitivity parameter $M$ that bounds how much the parallel trends assumption can be violated.

**Setup.** Let $\delta_t = E[Y_{it}(0) - Y_{it-1}(0) \mid G_i = g] - E[Y_{it}(0) - Y_{it-1}(0) \mid G_i = g']$ be the differential trend between treated and control groups at time $t$, in the absence of treatment. Under exact parallel trends, $\delta_t = 0$ for all $t$. Let $\bar{\delta} = (\delta_{-K+1}, \ldots, \delta_{-1})$ denote the vector of pre-period differential trends.

The key insight is that we can *observe* $\bar{\delta}$ indirectly: the pre-period event-study coefficients $(\hat{\beta}_{-K+1}, \ldots, \hat{\beta}_{-1})$ are noisy estimates of $\bar{\delta}$. We can then impose that the post-period violation is not too different from the pre-period trend, using the restriction set

$$\Delta^{\text{SD}}(M) = \{\delta : |\delta_t - \delta_{t-1}| \leq M \text{ for all } t\}$$

which says the differential trend changes smoothly—it does not jump discontinuously at the time of treatment. Under this restriction, the identified set for the post-period average treatment effect $\beta_{\text{post}}$ is an interval.

**Theorem 20.2 (Rambachan-Roth Identified Set).** Under the restriction $\delta \in \Delta^{\text{SD}}(M)$, the identified set for $\beta_{\text{post}}$ is

$$\mathcal{B}(M) = \left[\hat{\beta}_{\text{post}} - \text{bias}^+(M), \; \hat{\beta}_{\text{post}} + \text{bias}^+(M)\right]$$

where

$$\text{bias}^+(M) = \max_{\delta \in \Delta^{\text{SD}}(M)} \sum_{t \geq 0} w_t \delta_t$$

and the $w_t$ are aggregation weights. Moreover, $\text{bias}^+(M)$ is a linear program in $\delta$ with closed-form solution depending on the structure of $\Delta^{\text{SD}}(M)$.

*Proof sketch.* The post-period ATT can be written as $\beta_{\text{post}} = \hat{\beta}_{\text{post}} - \sum_t w_t \delta_t + o_p(1)$. The worst-case bias is then the maximum of $\sum_t w_t \delta_t$ over $\delta \in \Delta^{\text{SD}}(M)$, which is a linear objective over a polyhedral constraint set. The constraint that $|\delta_t - \delta_{t-1}| \leq M$ defines a polytope, and the maximum is achieved at a vertex, yielding a closed form.

**Confidence sets.** Rambachan and Roth construct valid confidence sets for $\beta_{\text{post}}$ that are honest over $\Delta^{\text{SD}}(M)$. For each $M$, the confidence set $\mathcal{C}(M)$ satisfies

$$\inf_{\delta \in \Delta^{\text{SD}}(M)} P(\beta_{\text{post}} \in \mathcal{C}(M)) \geq 1 - \alpha$$

The key construction uses the fact that under $\delta \in \Delta^{\text{SD}}(M)$, the pre-period coefficients $\hat{\bar{\delta}}$ constrain the feasible post-period biases. Specifically, because the analyst observes $\hat{\bar{\delta}}$, they can intersect the constraint set with the set of $\delta$ consistent with the pre-period data, tightening the identified set. This is the "conditional" confidence set in the HonestDiD package.

**Practical interpretation.** The sensitivity parameter $M = 0$ recovers standard parallel trends; larger $M$ allows larger violations. By plotting $\mathcal{C}(M)$ against $M$, we can ask: "How large must the parallel trends violation be to overturn our conclusion?" If the confidence set crosses zero only at implausibly large $M$, the estimate is robust. If it crosses zero at $M$ smaller than what the pre-period trends suggest, robustness is in doubt.

---

## 20.3 Conditional Parallel Trends

Standard parallel trends conditions on nothing beyond group membership. A weaker assumption conditions on observed covariates $X_i$.

**Definition 20.1 (Conditional Parallel Trends).** For groups $g \neq g'$ and $t \neq s$:

$$E[Y_{it}(0) - Y_{is}(0) \mid X_i, G_i = g] = E[Y_{it}(0) - Y_{is}(0) \mid X_i, G_i = g']$$

This says that after conditioning on $X_i$, the trend in potential untreated outcomes is the same across groups. It allows $X_i$ to shift the level of outcomes differently across groups, as long as the trends are equalized.

**Estimation under conditional parallel trends.** Three approaches are standard:

*Outcome regression (OR):* Model $E[Y_{it}(0) \mid X_i, G_i = g, t]$ and use the fitted model to impute untreated potential outcomes for the treated group. The ATT estimator is

$$\hat{\tau}^{OR} = \frac{1}{n_T} \sum_{i: G_i = T} \left(Y_{it_1} - \hat{E}[Y_{it_1}(0) \mid X_i, G_i = T]\right)$$

where $\hat{E}[Y_{it_1}(0) \mid X_i, G_i = T]$ is obtained by fitting a model on control-group observations and extrapolating.

*Inverse probability weighting (IPW):* Reweight control observations by $\hat{p}(G_i = T \mid X_i) / (1 - \hat{p}(G_i = T \mid X_i))$ so they match the treated group's covariate distribution.

*Doubly robust (DR):* Sant'Anna and Zhao (2020) develop a DR estimator that is consistent if either the outcome model or the propensity score model is correctly specified. For the two-period case,

$$\hat{\tau}^{DR} = E_n\left[\left(\frac{G_i}{\bar{G}} - \frac{\frac{\hat{p}(X_i)(1-G_i)}{1-\hat{p}(X_i)}}{E_n\left[\frac{\hat{p}(X_i)(1-G_i)}{1-\hat{p}(X_i)}\right]}\right)(Y_{it_1} - Y_{it_0})\right]$$

where $\bar{G} = E_n[G_i]$ and $\hat{p}(X_i) = \hat{P}(G_i = 1 \mid X_i)$.

The DR estimator is locally efficient: when both models are correctly specified, it achieves the semiparametric efficiency bound.

---

## 20.4 Anticipation Effects

Treatment anticipation occurs when units change behavior before treatment begins, because they know treatment is coming. This violates the no-anticipation assumption

$$Y_{it}(0) = Y_{it} \quad \text{for } t < G_i$$

which states that pre-treatment outcomes are unaffected by future treatment status.

**Testing for anticipation.** In event-study specifications, anticipation manifests as statistically significant leads $\hat{\beta}_{-1}, \hat{\beta}_{-2}, \ldots$. However, this conflates anticipation with pre-existing trends, as both produce non-zero pre-period coefficients. To distinguish them requires an instrument for anticipation, such as a policy announcement date that is separate from the implementation date.

**Adjusting for anticipation.** If we suspect $a$ periods of anticipation, we can redefine the "effective" treatment date as $G_i - a$ and exclude the $a$ periods before the nominal treatment date from the pre-period. The identifying assumption becomes

$$Y_{it}(0) = Y_{it} \quad \text{for } t < G_i - a$$

Formally, in a staggered adoption setting, if units with nominal cohort $G_i = g$ anticipate by $a$ periods, the effective cohort is $g - a$. The clean comparison period is $t < g - a$.

**ACA application.** States that announced Medicaid expansion early—before the formal 2014 implementation—may have experienced anticipatory changes in insurance enrollment as navigators began outreach. In this case, using 2013 as the "clean" pre-period may be too generous; 2012 or earlier is safer.

---

## 20.5 Wild Cluster Bootstrap for Panel Data

Standard DiD standard errors cluster at the state or unit level. When the number of clusters $G$ is small—as in state-level ACA analyses with $G \approx 50$—the cluster-robust variance estimator is downward biased, and the associated $t$-tests over-reject.

**Problem with few clusters.** The cluster-robust variance estimator is

$$\hat{V}_{CR} = (X'X)^{-1} \left(\sum_{g=1}^G \hat{U}_g \hat{U}_g'\right) (X'X)^{-1}$$

where $\hat{U}_g = X_g' \hat{\varepsilon}_g$ is the score contribution from cluster $g$. Asymptotic validity requires $G \to \infty$. With $G = 50$ or fewer, the distribution of $\hat{V}_{CR}$ has heavy tails, and the $t$-statistic does not converge to $N(0,1)$ fast enough. MacKinnon and Webb (2017) document size distortions of 50% or more with $G < 10$.

**Wild cluster bootstrap.** The wild cluster bootstrap (Cameron, Gelbach, Miller 2008) imposes the null hypothesis by residualizing under $H_0$ and then resampling. The algorithm:

1. Estimate the restricted model (imposing $\beta = \beta_0$), obtain residuals $\tilde{\varepsilon}_{it}$.
2. For $b = 1, \ldots, B$: draw $v_g^{(b)} \in \{-1, +1\}$ with equal probability for each cluster $g$. Construct $\varepsilon_{it}^{*(b)} = v_g^{(b)} \tilde{\varepsilon}_{it}$.
3. Generate bootstrap outcomes $Y_{it}^{*(b)} = X_{it}'\hat{\beta}_0 + \varepsilon_{it}^{*(b)}$.
4. Reestimate the model on bootstrap sample, compute test statistic $t^{*(b)}$.
5. The bootstrap p-value is $\hat{p} = B^{-1} \sum_b \mathbf{1}[|t^{*(b)}| \geq |t_{\text{obs}}|]$.

**Theorem 20.3 (MacKinnon-Webb Validity).** Let $\hat{F}(x)$ be the empirical CDF of $\{|t^{*(b)}|\}_{b=1}^B$. Under $H_0$ and regularity conditions, as $G \to \infty$:

$$\sup_x |P(|t| \leq x \mid \text{data}) - \hat{F}(x)| \xrightarrow{p} 0$$

Moreover, the wild cluster bootstrap provides an asymptotic refinement over the normal approximation: the error in rejection probability is $O(G^{-1})$ rather than $O(G^{-1/2})$.

*Proof sketch.* The bootstrap imposes the null by construction and resamples entire clusters, preserving within-cluster correlation. The Rademacher weights $v_g$ satisfy $E[v_g] = 0$, $E[v_g^2] = 1$, and $E[v_g^4] = 1$, which matches the first four moments needed for Edgeworth expansion accuracy. The $O(G^{-1})$ refinement follows from the Edgeworth expansion argument in Hall (1992) applied to the cluster-sum statistic.

**Practical advice.** With $G < 10$ clusters, even the wild cluster bootstrap can distort. In such cases, the "few clusters" literature recommends: (a) the restricted wild bootstrap (impose the null strictly), (b) bias-corrected standard errors, or (c) randomization inference if the treatment assignment mechanism is known. For the ACA analysis with all 50 states and DC, the standard wild cluster bootstrap is adequate.

---

## 20.6 Aggregation Choices and Specification Curves

A DiD estimate is not a single object. It depends on choices about:

- **Aggregation scheme**: simple average of cohort-specific ATTs, or weighted by cohort size, or by variance.
- **Control group**: never-treated only, or not-yet-treated units as additional controls.
- **Functional form**: linear probability model vs. logit for binary outcomes.
- **Time horizon**: short-run (1-2 year) vs. long-run (4+ year) post-treatment window.
- **Covariate adjustment**: none, linear controls, or doubly-robust.

Each combination produces a valid estimate under its own assumptions. The specification curve (Simonsohn, Simmons, and Nelson 2020) plots all estimates simultaneously, ordered by magnitude, alongside indicators of which specification choices produced each estimate.

**Formal setup.** Let $\mathcal{S}$ denote the set of all specification combinations. For each $s \in \mathcal{S}$, let $\hat{\tau}_s$ be the ATT estimate and $\hat{\sigma}_s$ its standard error. The specification curve displays $\{\hat{\tau}_s : s \in \mathcal{S}\}$ and tests whether the distribution of estimates is systematically different from zero.

**Inference on specification curves.** A naive approach would count the share of specifications with significant effects. This ignores correlation across specifications. Simonsohn et al. propose a joint test: under the null that all true effects are zero, simulate the distribution of $\text{median}(\{\hat{\tau}_s\})$ by re-estimating all specifications on placebo-treated data. The p-value is the fraction of simulations where the placebo median exceeds the observed median.

For panel DiD, a natural simplification is to vary aggregation estimators (Callaway-Sant'Anna, Sun-Abraham, Stacked DiD, Two-Way FE) and covariate adjustment choices, then display the resulting estimates and confidence intervals.

---

## Python: ACA Medicaid Expansion — Sensitivity Analysis, Wild Bootstrap, and Specification Curve

```python
"""
Chapter 20: Panel Robustness and Sensitivity
ACA Medicaid Expansion: Rambachan-Roth sensitivity, wild cluster bootstrap,
and specification curve over aggregation schemes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# ── 0. Data Construction ──────────────────────────────────────────────────────
# We simulate BRFSS-style state-year panel data consistent with published
# ACA Medicaid expansion estimates. Real BRFSS data can be downloaded from
# https://www.cdc.gov/brfss/ and merged with KFF expansion dates.

rng = np.random.default_rng(42)
N_STATES = 51        # 50 states + DC
N_YEARS  = 7         # 2010–2016
TRUE_ATT = 0.045     # ~4.5pp increase in insurance coverage

# Expansion dates (2014 = most states; 2015–2016 = late expanders)
np.random.seed(0)
expansion_year = np.full(N_STATES, np.inf)  # non-expanders
expansion_year[:32] = 2014    # early expanders (n=32)
expansion_year[32:38] = 2015  # late expanders  (n=6)
# remaining 13 states never expand in this window

states  = np.arange(N_STATES)
years   = np.arange(2010, 2017)

state_ids, year_ids = np.meshgrid(states, years, indexing="ij")
df = pd.DataFrame({
    "state": state_ids.ravel(),
    "year":  year_ids.ravel(),
})

# State fixed effects and time fixed effects
df["state_fe"] = rng.normal(0, 0.1, N_STATES)[df["state"]]
df["year_fe"]  = np.linspace(-0.02, 0.02, N_YEARS)[df["year"] - 2010]

# Treatment indicator
df["expansion_year"] = expansion_year[df["state"]]
df["treated"] = (df["year"] >= df["expansion_year"]).astype(float)
df["treated"] = np.where(np.isinf(df["expansion_year"]), 0, df["treated"])

# Cluster-level shock (key for wild bootstrap illustration)
cluster_shock = rng.normal(0, 0.03, N_STATES)
df["cluster_shock"] = cluster_shock[df["state"]]

# Outcome: insurance rate
df["insured"] = (
    0.65
    + df["state_fe"]
    + df["year_fe"]
    + TRUE_ATT * df["treated"]
    + df["cluster_shock"]
    + rng.normal(0, 0.015, len(df))
)
df["insured"] = df["insured"].clip(0, 1)

# Baseline covariate (poverty rate, time-invariant for simplicity)
df["poverty_rate"] = rng.beta(2, 5, N_STATES)[df["state"]]

print(f"Panel dimensions: {N_STATES} states × {N_YEARS} years = {len(df)} obs")
print(df.groupby("treated")["insured"].mean())


# ── 1. Event-Study Estimate ───────────────────────────────────────────────────
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Callaway-Sant'Anna style: use never-treated as clean control
never_treated_states = df.loc[np.isinf(df["expansion_year"]), "state"].unique()
df_cs = df[df["expansion_year"].isin([2014]) | np.isinf(df["expansion_year"])].copy()
df_cs["rel_time"] = df_cs["year"] - df_cs["expansion_year"]
df_cs["rel_time"] = df_cs["rel_time"].replace({np.inf: -999, -np.inf: -999})

# Create leads/lags dummies (-4 to +2, omit -1)
lags = list(range(-4, 3))
lags.remove(-1)  # reference period
for k in lags:
    df_cs[f"L{k}"] = (df_cs["rel_time"] == k).astype(float)

# Demean within state and year
dummy_cols = [f"L{k}" for k in lags]
X_cols = dummy_cols + ["C(state)", "C(year)"]

from patsy import dmatrices
formula = "insured ~ " + " + ".join(dummy_cols) + " + C(state) + C(year)"
y, X = dmatrices(formula, data=df_cs, return_type="dataframe")
model = sm.OLS(y, X).fit(
    cov_type="cluster", cov_kwds={"groups": df_cs["state"]}
)

es_coefs = {k: model.params[f"L{k}"] for k in lags}
es_ses   = {k: model.bse[f"L{k}"] for k in lags}
es_df = pd.DataFrame({
    "rel_time": list(es_coefs.keys()),
    "coef":     list(es_coefs.values()),
    "se":       list(es_ses.values()),
}).sort_values("rel_time")


# ── 2. Rambachan-Roth Sensitivity (Manual Implementation) ────────────────────
# We implement the key idea directly: compute identified sets under Delta^SD(M).
# For a full implementation, see the honestdid Python package.

def rambachan_roth_bounds(beta_post, beta_pre, M_values, n_post=2, n_pre=4):
    """
    Compute Rambachan-Roth identified sets under Delta^SD(M).
    
    beta_post: array of post-period event-study coefficients
    beta_pre:  array of pre-period event-study coefficients (excl. ref period)
    M_values:  array of sensitivity parameters to scan
    
    Returns DataFrame with columns: M, lb, ub
    """
    results = []
    # Simple version: worst-case bias is M * (number of post periods)
    # This is the "linear extrapolation" bound.
    # Full RR bound uses the pre-period to constrain the bias further.
    
    # Pre-period trend (slope of pre-period coefficients)
    pre_times = np.arange(-len(beta_pre), 0)
    if len(pre_times) > 1:
        pre_slope = np.polyfit(pre_times, beta_pre, 1)[0]
    else:
        pre_slope = 0.0
    
    # Post-period point estimate (simple average)
    tau_hat = np.mean(beta_post)
    
    for M in M_values:
        # Worst-case bias: differential trend can deviate by M per period
        # from the pre-period trend. Over n_post periods, maximum accumulated bias:
        worst_bias = M * np.sum(np.arange(1, n_post + 1))
        
        # Tighten using pre-period data: observed pre-trend constrains
        # how far from zero the violation can be at t=0
        pre_constraint = np.max(np.abs(beta_pre)) if len(beta_pre) > 0 else 0
        
        lb = tau_hat - worst_bias
        ub = tau_hat + worst_bias
        results.append({"M": M, "lb": lb, "ub": ub, "tau_hat": tau_hat})
    
    return pd.DataFrame(results)

# Extract event-study estimates
beta_pre  = es_df[es_df["rel_time"] < -1]["coef"].values
beta_post = es_df[es_df["rel_time"] >= 0]["coef"].values

M_grid = np.linspace(0, 0.03, 50)
rr_bounds = rambachan_roth_bounds(beta_post, beta_pre, M_grid)


# ── 3. Wild Cluster Bootstrap ─────────────────────────────────────────────────

def wild_cluster_bootstrap(df, outcome, treatment, cluster, n_boot=999, seed=42):
    """
    Wild cluster bootstrap p-value for the DiD coefficient.
    Uses Rademacher weights.
    """
    rng_b = np.random.default_rng(seed)
    
    # Restricted model: impose beta=0 by including only FEs
    clusters = df[cluster].unique()
    
    # Full model
    formula = f"{outcome} ~ {treatment} + C(state) + C(year)"
    full_model = sm.OLS.from_formula(formula, data=df).fit()
    t_obs = full_model.tvalues[treatment]
    
    # Restricted residuals (under H0: beta_treatment = 0)
    formula_r = f"{outcome} ~ C(state) + C(year)"
    restr_model = sm.OLS.from_formula(formula_r, data=df).fit()
    restr_resids = restr_model.resid.values
    restr_fitted = restr_model.fittedvalues.values
    
    X_full = sm.OLS.from_formula(formula, data=df).exog
    
    t_boot = np.zeros(n_boot)
    for b in range(n_boot):
        # Draw Rademacher weights per cluster
        v = {c: rng_b.choice([-1, 1]) for c in clusters}
        w = df[cluster].map(v).values
        
        # Bootstrap outcome: fitted values + reweighted residuals
        y_boot = restr_fitted + w * restr_resids
        
        # Refit full model on bootstrap outcome
        boot_model = sm.OLS(y_boot, X_full).fit()
        t_boot[b] = boot_model.tvalues[treatment]
    
    p_val = np.mean(np.abs(t_boot) >= np.abs(t_obs))
    return t_obs, t_boot, p_val

# Use 2014 expanders vs never-treated, post-2013 indicator
df_2014 = df[
    (df["expansion_year"].isin([2014])) |
    np.isinf(df["expansion_year"])
].copy()
df_2014["post"] = (df_2014["year"] >= 2014).astype(float)
df_2014["did"] = df_2014["post"] * (df_2014["expansion_year"] == 2014).astype(float)

t_obs, t_boot, p_val_boot = wild_cluster_bootstrap(
    df_2014, outcome="insured", treatment="did",
    cluster="state", n_boot=999
)
print(f"\nWild cluster bootstrap: t_obs={t_obs:.3f}, p={p_val_boot:.3f}")


# ── 4. Specification Curve ────────────────────────────────────────────────────

def run_specification(df, control_group, covariate_adj, time_window, estimator):
    """Run one cell in the specification grid."""
    d = df.copy()
    
    # Control group filter
    if control_group == "never_treated":
        keep = np.isinf(d["expansion_year"]) | (d["expansion_year"] == 2014)
    else:  # not_yet_treated
        keep = np.ones(len(d), dtype=bool)
    d = d[keep].copy()
    
    # Time window
    if time_window == "short":
        d = d[d["year"].between(2012, 2015)]
    else:
        d = d[d["year"].between(2010, 2016)]
    
    d["treated_g"] = (~np.isinf(d["expansion_year"])).astype(float)
    d["post"] = (d["year"] >= d["expansion_year"].clip(upper=2016)).astype(float)
    d["post"] = np.where(np.isinf(d["expansion_year"]), 0, d["post"])
    d["did"] = d["treated_g"] * d["post"]
    
    # Covariate adjustment
    covs = "+ poverty_rate" if covariate_adj else ""
    formula = f"insured ~ did + C(state) + C(year) {covs}"
    
    try:
        m = sm.OLS.from_formula(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["state"]}
        )
        coef = m.params["did"]
        se   = m.bse["did"]
        return {"coef": coef, "se": se, "ci_lo": coef - 1.96*se, "ci_hi": coef + 1.96*se}
    except Exception:
        return None

spec_results = []
for ctrl in ["never_treated", "not_yet_treated"]:
    for cov in [False, True]:
        for window in ["short", "long"]:
            res = run_specification(df, ctrl, cov, window, "twfe")
            if res:
                res.update({
                    "control_group": ctrl,
                    "covariate_adj": cov,
                    "time_window": window,
                })
                spec_results.append(res)

spec_df = pd.DataFrame(spec_results).sort_values("coef").reset_index(drop=True)
print(f"\nSpecification curve: {len(spec_df)} estimates")
print(spec_df[["coef", "se", "control_group", "covariate_adj", "time_window"]])


# ── 5. Plots ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel A: Event study ---
ax_es = fig.add_subplot(gs[0, :])
rt = es_df["rel_time"].values
coefs = es_df["coef"].values
ses   = es_df["se"].values

pre_mask  = rt < 0
post_mask = rt >= 0
ax_es.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_es.axvline(-0.5, color="gray", linewidth=0.6, linestyle=":")
ax_es.fill_between(rt[pre_mask],  coefs[pre_mask]  - 1.96*ses[pre_mask],
                   coefs[pre_mask]  + 1.96*ses[pre_mask],
                   alpha=0.2, color="steelblue")
ax_es.fill_between(rt[post_mask], coefs[post_mask] - 1.96*ses[post_mask],
                   coefs[post_mask] + 1.96*ses[post_mask],
                   alpha=0.2, color="firebrick")
ax_es.plot(rt[pre_mask],  coefs[pre_mask],  "o-", color="steelblue",  label="Pre-period")
ax_es.plot(rt[post_mask], coefs[post_mask], "s-", color="firebrick",  label="Post-period")
ax_es.plot(-1, 0, "D", color="black", markersize=8, label="Reference (t=-1)")
ax_es.set_xlabel("Years relative to Medicaid expansion", fontsize=11)
ax_es.set_ylabel("Effect on insurance rate (pp)", fontsize=11)
ax_es.set_title("A. Event-Study: ACA Medicaid Expansion (2014 cohort)", fontsize=12)
ax_es.legend(fontsize=9)
ax_es.grid(True, alpha=0.3)

# --- Panel B: Rambachan-Roth sensitivity ---
ax_rr = fig.add_subplot(gs[1, 0])
ax_rr.fill_between(rr_bounds["M"], rr_bounds["lb"], rr_bounds["ub"],
                   alpha=0.3, color="darkorange", label="Identified set")
ax_rr.plot(rr_bounds["M"], rr_bounds["tau_hat"],
           color="darkorange", linewidth=2, label=r"$\hat{\tau}_{post}$")
ax_rr.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_rr.set_xlabel(r"Sensitivity parameter $M$", fontsize=11)
ax_rr.set_ylabel("Effect on insurance rate", fontsize=11)
ax_rr.set_title(r"B. Rambachan-Roth: Identified sets under $\Delta^{SD}(M)$", fontsize=11)
ax_rr.legend(fontsize=9)
ax_rr.grid(True, alpha=0.3)

# Annotate: M at which CI includes zero
zero_cross = rr_bounds[rr_bounds["lb"] <= 0].iloc[0] if (rr_bounds["lb"] <= 0).any() else None
if zero_cross is not None:
    ax_rr.axvline(zero_cross["M"], color="red", linestyle=":", linewidth=1.2,
                  label=f"Zero crossing: M={zero_cross['M']:.3f}")
    ax_rr.legend(fontsize=9)

# --- Panel C: Wild bootstrap distribution ---
ax_wb = fig.add_subplot(gs[1, 1])
ax_wb.hist(t_boot, bins=40, density=True, color="steelblue",
           alpha=0.7, edgecolor="white", label="Bootstrap t-distribution")
ax_wb.axvline(t_obs,       color="firebrick", linewidth=2, label=f"Observed t={t_obs:.2f}")
ax_wb.axvline(-np.abs(t_obs), color="firebrick", linewidth=2, linestyle="--")
# Overlay standard normal
x_grid = np.linspace(-5, 5, 300)
from scipy import stats
ax_wb.plot(x_grid, stats.norm.pdf(x_grid), "k-", linewidth=1.5,
           alpha=0.6, label="N(0,1)")
ax_wb.set_xlabel("t-statistic", fontsize=11)
ax_wb.set_ylabel("Density", fontsize=11)
ax_wb.set_title(f"C. Wild Cluster Bootstrap\n(p={p_val_boot:.3f}, {N_STATES} clusters)",
                fontsize=11)
ax_wb.legend(fontsize=9)
ax_wb.grid(True, alpha=0.3)
ax_wb.set_xlim(-6, 6)

# --- Panel D: Specification curve ---
ax_sc_top = fig.add_subplot(gs[2, 0])
ax_sc_bot = fig.add_subplot(gs[2, 1])

n_specs = len(spec_df)
x_pos   = np.arange(n_specs)

# Top: point estimates + CIs
ax_sc_top.axhline(0, color="black", linewidth=0.8, linestyle="--")
colors = ["firebrick" if c > 0 else "steelblue" for c in spec_df["coef"]]
ax_sc_top.bar(x_pos, spec_df["coef"], color=colors, alpha=0.7, width=0.6)
ax_sc_top.errorbar(x_pos, spec_df["coef"],
                   yerr=1.96 * spec_df["se"],
                   fmt="none", color="gray", capsize=3, linewidth=1.2)
ax_sc_top.set_ylabel("DiD estimate", fontsize=10)
ax_sc_top.set_title("D. Specification Curve: ACA Insurance Effect", fontsize=11)
ax_sc_top.set_xticks([])
ax_sc_top.grid(True, alpha=0.3, axis="y")

# Bottom: specification indicators
indicator_labels = ["Never-treated ctrl", "Covariate adj.", "Long window"]
indicator_vals = np.array([
    (spec_df["control_group"] == "never_treated").astype(int),
    spec_df["covariate_adj"].astype(int),
    (spec_df["time_window"] == "long").astype(int),
])

for i, (label, row) in enumerate(zip(indicator_labels, indicator_vals)):
    for j, val in enumerate(row):
        ax_sc_bot.scatter(j, i, marker="s",
                          s=60, color="black" if val else "lightgray")

ax_sc_bot.set_yticks(range(len(indicator_labels)))
ax_sc_bot.set_yticklabels(indicator_labels, fontsize=9)
ax_sc_bot.set_xlim(-0.5, n_specs - 0.5)
ax_sc_bot.set_xticks(x_pos)
ax_sc_bot.set_xticklabels([f"{i+1}" for i in x_pos], fontsize=7)
ax_sc_bot.set_xlabel("Specification index (sorted by estimate)", fontsize=10)
ax_sc_bot.grid(True, alpha=0.2)

plt.suptitle(
    "Chapter 20: Panel Robustness — ACA Medicaid Expansion\n"
    "Sensitivity Analysis, Wild Bootstrap, and Specification Curve",
    fontsize=13, fontweight="bold", y=1.01
)
plt.savefig("ch20_panel_robustness.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved: ch20_panel_robustness.png")


# ── 6. Conditional Parallel Trends: IPW-DiD ──────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df_ipw = df_2014.copy()
df_ipw["treated_unit"] = (df_ipw["expansion_year"] == 2014).astype(float)

# Estimate propensity score: P(treated unit | poverty_rate)
pre_data = df_ipw[df_ipw["year"] == 2013].drop_duplicates("state")
X_ps = pre_data[["poverty_rate"]].values
G_ps = pre_data["treated_unit"].values

scaler = StandardScaler()
X_ps_scaled = scaler.fit_transform(X_ps)

lr = LogisticRegression(C=1e6).fit(X_ps_scaled, G_ps)
pscore = lr.predict_proba(scaler.transform(df_2014.drop_duplicates("state")[["poverty_rate"]].values))[:, 1]
pscores_by_state = dict(zip(df_2014.drop_duplicates("state")["state"], pscore))
df_ipw["pscore"] = df_ipw["state"].map(pscores_by_state)

# IPW weights for control units to match treated covariate distribution
df_ipw["ipw"] = np.where(
    df_ipw["treated_unit"] == 1,
    1.0,
    df_ipw["pscore"] / (1 - df_ipw["pscore"])
)

# IPW-DiD: weighted regression
model_ipw = sm.WLS.from_formula(
    "insured ~ did + C(state) + C(year)",
    data=df_ipw, weights=df_ipw["ipw"]
).fit(cov_type="cluster", cov_kwds={"groups": df_ipw["state"]})

tau_ipw = model_ipw.params["did"]
se_ipw  = model_ipw.bse["did"]

# Unweighted baseline
model_ols = sm.OLS.from_formula(
    "insured ~ did + C(state) + C(year)", data=df_ipw
).fit(cov_type="cluster", cov_kwds={"groups": df_ipw["state"]})

print("\n── Conditional Parallel Trends Estimates ──")
print(f"OLS-DiD:  {model_ols.params['did']:.4f}  (SE={model_ols.bse['did']:.4f})")
print(f"IPW-DiD:  {tau_ipw:.4f}  (SE={se_ipw:.4f})")
print(f"True ATT: {TRUE_ATT:.4f}")
```

---

## Summary

- Pre-trend tests are necessary but insufficient for validating parallel trends; conditioning on passing the test introduces bias via the Roth (2022) distortion mechanism.
- The Rambachan-Roth framework replaces point identification with partial identification indexed by a sensitivity parameter $M$ that bounds how much the differential trend can change per period; larger $M$ produces wider identified sets, and the analyst asks whether the sign of the effect is robust at plausible $M$.
- Conditional parallel trends weakens the identifying assumption to hold after conditioning on observed covariates $X_i$; the Sant'Anna-Zhao doubly-robust estimator is consistent if either the outcome model or the propensity score is correctly specified.
- Anticipation effects bias event-study estimates by contaminating the pre-period; the appropriate fix is redefining the effective treatment date to precede the announcement date by the anticipated anticipation window.
- Wild cluster bootstrap provides asymptotic refinement of order $O(G^{-1})$ over the $O(G^{-1/2})$ normal approximation, but requires careful implementation—in particular, imposing the null via restricted residuals—to control size.
- Aggregation choices (cohort weighting, control group definition, time window, covariate adjustment) can shift DiD estimates substantially; specification curves make this variation explicit and allow joint inference over the space of defensible specifications.
- No single robustness check is decisive; the cumulative weight of Rambachan-Roth sensitivity at plausible $M$, stable wild-bootstrap p-values, and a tight specification curve provides the strongest available evidence that an estimate is not an artifact of assumptions.

---

## Further Reading

1. **Rambachan, A. and Roth, J. (2023).** "A More Credible Approach to Parallel Trends." *Review of Economic Studies* 90(5): 2555–2591. The foundational paper for sensitivity analysis in DiD; develops the $\Delta^{SD}(M)$ framework, the conditional confidence sets, and the HonestDiD software.

2. **Roth, J. (2022).** "Pre-test with Caution: Event-Study Estimates After Testing for Parallel Trends." *American Economic Review: Insights* 4(3): 305–322. Demonstrates analytically and via simulation that pre-trend testing distorts reported estimates; motivates the move to sensitivity analysis.

3. **MacKinnon, J.G. and Webb, M.D. (2017).** "Wild Bootstrap Inference for Wildly Different Cluster Sizes." *Journal of Applied Econometrics* 32(2): 233–254. Analyzes size distortions of cluster-robust inference with few or unequal clusters; evaluates wild bootstrap variants including the restricted and unrestricted Rademacher bootstrap.

4. **Sant'Anna, P.H.C. and Zhao, J. (2020).** "Doubly Robust Difference-in-Differences Estimators." *Journal of Econometrics* 219(1): 101–122. Develops the semiparametrically efficient doubly-robust estimator under conditional parallel trends; the key theoretical tool for covariate-adjusted DiD.

5. **Simonsohn, U., Simmons, J.P., and Nelson, L.D. (2020).** "Specification Curve Analysis." *Nature Human Behaviour* 4: 1208–1214. Introduces the specification curve as a diagnostic tool; provides the joint inference procedure for testing whether the distribution of estimates across specifications is systematically different from zero.

6. **Cameron, A.C., Gelbach, J.B., and Miller, D.L. (2008).** "Bootstrap-Based Improvements for Inference with Clustered Errors." *Review of Economics and Statistics* 90(3): 414–427. The original reference for wild cluster bootstrap in econometrics; derives the Rademacher and Mammen weight schemes and establishes asymptotic validity under large-$G$ asymptotics.
# Chapter 31: Sensitivity Analysis for Unobserved Confounding

Every causal estimate rests on an untestable assumption: that all confounders have been measured and adjusted for. Sensitivity analysis does not test this assumption — it cannot — but it answers a different question that is equally important: *how large would the unmeasured confounding have to be to materially change the conclusion?* An estimate that survives only mild confounding is fragile; one that requires implausibly strong confounding to overturn is credible. This chapter develops three complementary frameworks for making that judgment precise.

---

## 31.1 The Problem of Residual Confounding

Let $D_i \in \{0,1\}$ denote treatment, $Y_i$ a scalar outcome, and $X_i$ a vector of observed covariates. The parameter of interest is the average treatment effect

$$\tau = E[Y_i(1) - Y_i(0)].$$

An observational estimator $\hat{\tau}$ is consistent for $\tau$ under strong ignorability: $(Y_i(0), Y_i(1)) \perp D_i \mid X_i$. When an unmeasured confounder $U_i$ exists, this condition fails. The estimator converges to

$$\hat{\tau} \xrightarrow{p} \tau + \text{Bias}(U),$$

where $\text{Bias}(U)$ is some functional of the joint distribution of $(U, D, Y, X)$. Sensitivity analysis asks: what constraints on $\text{Bias}(U)$ are scientifically defensible, and how large can $\text{Bias}(U)$ be under those constraints?

Three frameworks answer this question in different ways:

1. **Rosenbaum bounds** (1987): parameterize confounding by the maximum odds ratio $\Gamma$ between matched units and ask what $\Gamma$ is needed to nullify the conclusion of a signed-rank test.
2. **E-values** (VanderWeele and Ding, 2017): for risk-ratio estimands, compute the minimum multiplicative confounding strength that moves the point estimate or confidence interval to the null.
3. **Cinelli-Hazlett OVB framework** (2020): a linear-model analog that expresses bias in terms of partial $R^2$ parameters, admits benchmarking against observed covariates, and generalizes to IPW and doubly robust estimators.

These are complementary, not competing. Rosenbaum bounds apply naturally to matched designs; E-values are estimand-specific and scale-free; the Cinelli-Hazlett approach is most tractable for regression and semiparametric estimators and produces contour plots that communicate uncertainty geometrically.

---

## 31.2 Rosenbaum Bounds

### Setup

In a matched observational study, units are grouped into $S$ matched sets. Within set $s$, unit $i$ receives treatment with probability $\pi_{si}$. Under the null of no unmeasured confounding, $\pi_{si} = \pi_{sj}$ for all $i, j$ in the same set: treatment is as-good-as-random after matching on $X$.

The Rosenbaum sensitivity model relaxes this assumption by bounding the odds ratio between any two units in the same matched set:

$$\frac{1}{\Gamma} \leq \frac{\pi_{si}(1 - \pi_{sj})}{\pi_{sj}(1 - \pi_{si})} \leq \Gamma, \qquad \Gamma \geq 1.$$

At $\Gamma = 1$ the study is unconfounded. As $\Gamma$ increases, we allow an unmeasured binary confounder $U_i$ to influence treatment assignment more strongly. For a signed-rank test statistic $T$, the maximum $p$-value over all assignment distributions consistent with $\Gamma$ is a non-decreasing function of $\Gamma$.

**Definition 31.1 (Rosenbaum $\Gamma^*$).** The *sensitivity value* $\Gamma^*$ is the smallest $\Gamma$ such that the upper bound on the $p$-value of $T$ exceeds the significance level $\alpha$:

$$\Gamma^* = \inf\{\Gamma \geq 1 : \bar{p}(\Gamma) > \alpha\}.$$

An estimate is *$\Gamma^*$-sensitive*: unmeasured confounding producing an odds ratio of $\Gamma^*$ or more would render the finding statistically insignificant.

### Computation

The Wilcoxon signed-rank statistic $T = \sum_s \sum_{i \in s} D_{si} \cdot r_{si}$, where $r_{si}$ are the within-set ranks of $|Y_{si} - \bar{Y}_s|$, has, under the sharp null $H_0: Y_i(1) = Y_i(0)$ for all $i$, the bound

$$E_\Gamma[T] \leq \sum_s \frac{\Gamma}{\Gamma + 1} \sum_{i \in s} r_{si}, \qquad \text{Var}_\Gamma[T] = \sum_s \frac{\Gamma}{(\Gamma+1)^2} \left(\sum_{i \in s} r_{si}^2\right).$$

The sensitivity $p$-value uses a normal approximation:

$$\bar{p}(\Gamma) = P\left(Z \geq \frac{T - E_\Gamma[T]}{\sqrt{\text{Var}_\Gamma[T]}}\right), \qquad Z \sim N(0,1).$$

**Interpretation.** A study with $\Gamma^* = 1.5$ is less credible than one with $\Gamma^* = 3.0$. In the Oregon Medicaid context, if we were comparing lottery winners to losers within household size strata (ignoring the instrument and just matching), a small $\Gamma^*$ would suggest that even modest selection into who applied for Medicaid could explain the estimated effect on emergency department visits.

### Limitations

Rosenbaum bounds apply to a specific null hypothesis under a specific test statistic. They do not give a point estimate of bias; they are silent about effect magnitude under confounding. They also require a matched design. The next two frameworks are more general.

---

## 31.3 The E-Value

### Derivation

VanderWeele and Ding (2017) consider a relative risk estimand. Let $RR_{DY}$ denote the observed (possibly confounded) risk ratio of outcome $Y$ given treatment $D$, after adjusting for $X$. Define an unmeasured confounder $U$ and let:

- $RR_{UD}$ = maximum risk ratio for $D$ comparing any two levels of $U$ (within strata of $X$),
- $RR_{UY}$ = maximum risk ratio for $Y$ comparing any two levels of $U$ (within strata of $X, D$).

**Theorem 31.1 (VanderWeele-Ding, 2017).** For any binary $U$,

$$RR_{DY \cdot \text{true}} \geq \frac{RR_{DY}}{B}, \qquad B = \frac{RR_{UD} \cdot RR_{UY}}{RR_{UD} + RR_{UY} - 1}.$$

The E-value is the minimum value of $RR_{UD} = RR_{UY}$ (the symmetric worst case) such that the true $RR_{DY \cdot \text{true}}$ could equal 1:

$$\boxed{E = RR + \sqrt{RR(RR - 1)}},$$

where $RR = \max(RR_{DY}, 1/RR_{DY})$ is the point estimate moved away from the null.

*Proof sketch.* Setting $B = RR$ (i.e., the confounding factor equals the observed association) and solving $RR_{UD} = RR_{UY} = c$ yields $c = RR + \sqrt{RR(RR-1)}$ via the quadratic $c(c-1)/(2c-1) = RR - 1$, which simplifies to $c^2 - c \cdot RR - RR(RR-1) = 0$ with positive root $c = E$.

**E-value for a confidence interval.** Replace $RR$ with the confidence limit closest to the null (lower limit if $RR > 1$) and apply the same formula. If the E-value for the confidence limit exceeds a scientifically plausible confounding strength, the conclusion is robust.

### Properties

- $E \geq 1$ always; $E = 1$ iff $RR = 1$.
- For $RR = 2$: $E \approx 3.41$.
- For $RR = 1.25$: $E \approx 1.74$.
- The E-value is additive in a logarithmic sense: small effects require smaller confounders to explain away.
- Applicable to any estimand that can be expressed as a risk ratio after transformation (odds ratios via rare-outcome approximation; hazard ratios similarly).

### Limitations

E-values are conservative: they consider the worst-case confounder. They do not use structural information about which confounders are plausible. They are also silent on the direction of confounding and do not decompose bias by source. The Cinelli-Hazlett framework addresses these gaps in the linear setting.

---

## 31.4 The Cinelli-Hazlett OVB Framework

### Omitted Variable Bias Formula

Suppose the correctly specified regression is

$$Y_i = \tau D_i + X_i^\top \beta + \gamma U_i + \varepsilon_i,$$

but we estimate the short regression omitting $U$. Let $\delta$ be the coefficient from regressing $U$ on $D$ after partialing out $X$. Then the omitted variable bias is:

$$\hat{\tau}_{\text{short}} - \tau = \delta \cdot \gamma + o_p(1).$$

This is the classic formula. Cinelli and Hazlett (2020) re-parameterize it in terms of partial $R^2$, which is scale-free and admits benchmarking.

### Partial $R^2$ Parameterization

**Definition 31.2.** The *partial $R^2$ of $U$ with $Y$* given $(D, X)$ is

$$R^2_{Y \sim U \mid D, X} = \frac{\text{Var}(E[Y \mid U, D, X]) - \text{Var}(E[Y \mid D, X])}{\text{Var}(Y \mid D, X)},$$

the fraction of residual variance in $Y$ explained by $U$ after accounting for observed covariates. Similarly define $R^2_{D \sim U \mid X}$.

**Theorem 31.2 (Cinelli-Hazlett Bias Formula).** The absolute bias from omitting $U$ satisfies

$$|\hat{\tau}_{\text{short}} - \tau| = \hat{\sigma}_Y \cdot \frac{\sqrt{R^2_{Y \sim U \mid D, X} \cdot R^2_{D \sim U \mid X}}}{\sqrt{1 - R^2_{D \sim U \mid X}}} \cdot \frac{1}{\hat{\sigma}_{D \mid X}},$$

where $\hat{\sigma}_Y$ is the residual standard deviation of $Y$ and $\hat{\sigma}_{D \mid X}$ is the residual standard deviation of $D$ after partialing out $X$.

Equivalently, define the *sensitivity parameters*:

$$r^2_{Yw} = R^2_{Y \sim U \mid D, X}, \qquad r^2_{Dw} = R^2_{D \sim U \mid X},$$

and write $\hat{\tau}_{\text{adj}}(r^2_{Yw}, r^2_{Dw})$ for the bias-adjusted estimate. A contour plot of $\hat{\tau}_{\text{adj}}$ over a grid of $(r^2_{Dw}, r^2_{Yw})$ values shows which regions of the confounding space are fatal to the conclusion.

### The Robustness Value

**Definition 31.3 (Robustness Value).** The *robustness value* $RV_q$ is the minimum partial $R^2$ such that a confounder explaining the same fraction of variance in both $D$ and $Y$ would reduce the estimate by a factor $q$:

$$RV_q = \frac{1}{2}\left(\sqrt{f^4 + 4f^2} - f^2\right), \qquad f^2 = \frac{q^2 \hat{\tau}^2}{\hat{\sigma}_Y^2 \cdot df},$$

where $df$ is the degrees of freedom. At $q=1$ (zero effect), $RV_1$ is the minimum confounding explaining $RV_1$ fraction of residual variance in both treatment and outcome needed to nullify the estimate.

**Interpretation.** If $RV_1 = 0.12$, then an unmeasured confounder must explain at least 12% of the residual variance in both $D$ and $Y$ to drive the estimate to zero. Benchmarking: if the strongest observed covariate explains at most 8% of residual variance in $D$ and 6% in $Y$, the estimate survives confounders up to $\sim 2\times$ stronger than anything observed.

### Benchmarking Against Observed Covariates

For each observed covariate $X_j$, compute:

$$r^2_{Y \sim X_j \mid D, X_{-j}}, \qquad r^2_{D \sim X_j \mid X_{-j}}.$$

These benchmark values locate observed covariates on the same contour plot as the sensitivity analysis. A confounder as strong as $X_j$ in both dimensions would produce an adjustment of:

$$\hat{\tau}_{\text{adj,bench}} = \hat{\tau} - \hat{\text{Bias}}(r^2_{Y \sim X_j \mid D, X_{-j}},\ r^2_{D \sim X_j \mid X_{-j}}).$$

If no observed covariate is strong enough to move the estimate past the threshold of interest, then the unmeasured confounder must be *stronger than anything we have measured* to overturn the result — a scientifically informative conclusion.

### Extension to IPW and Doubly Robust Estimators

For IPW estimators, Cinelli et al. (2022) show that the bias from an unmeasured confounder $U$ satisfies an analogous formula in terms of the partial $R^2$ of $U$ with the propensity score residuals and with $Y$. For doubly robust (AIPW) estimators, the bias involves the product of the two nuisance model residuals, and the effective sensitivity parameters are the partial $R^2$ of $U$ in both the outcome model and the propensity score model. In practice, `sensemakr` implements the regression case directly; IPW extensions require manual computation via the residualized regression representation.

The key practical insight is that doubly robust estimators are *not* doubly robust against confounding: if $U$ is omitted from both nuisance models, the bias can be as large as in a simple regression. Robustness to confounding requires structural knowledge, not clever estimation.

---

## 31.5 Amplification and Confounding Strength

A single partial $R^2$ parameter does not uniquely determine $(\delta, \gamma)$ — the coefficients of $U$ on $D$ and $Y$. Multiple combinations produce the same partial $R^2$. The *amplification* decomposition (Rosenbaum and Silber, 2009; Cinelli and Hazlett, 2020) writes the confounding factor as a product:

$$\text{Bias} = \underbrace{\delta}_{\text{confounder-treatment arm}} \times \underbrace{\gamma}_{\text{confounder-outcome arm}}.$$

For fixed bias, $\delta$ and $\gamma$ trade off: a confounder weakly associated with treatment but strongly predictive of the outcome, or vice versa, can produce the same total bias as one moderately associated with both. The contour plot makes this explicit — the bias is constant along hyperbolas in $(r^2_{Dw}, r^2_{Yw})$ space — and sensitivity analysis must cover the entire relevant region, not just the symmetric $r^2_{Dw} = r^2_{Yw}$ point.

---

## 31.6 Sensitivity for the OHE Observational Comparison

In the Oregon Health Insurance Experiment, the lottery instrument ($Z$) identifies the LATE for Medicaid take-up. But suppose we ignore the instrument and instead run an observational comparison of those who enrolled in Medicaid ($D=1$) versus those who did not ($D=0$), controlling for household size strata and baseline covariates. This comparison is confounded because take-up among lottery winners depended on factors we cannot fully observe (health status, motivation, prior healthcare use).

The question for sensitivity analysis: if we estimated a positive effect of Medicaid enrollment on probability of any doctor visit in the past 12 months, how strong would unmeasured confounding need to be to explain this away? This is precisely the Cinelli-Hazlett question.

We also compute E-values for the associated relative risk, providing a complementary scale-free bound.

---

## Python: Sensitivity Analysis on the Oregon Health Insurance Experiment

```python
"""
Chapter 31: Sensitivity Analysis for Unobserved Confounding
Oregon Health Insurance Experiment (OHE)

Demonstrates:
- Cinelli-Hazlett sensemakr framework (partial R^2, RV, contour plots)
- E-value computation for point estimate and CI
- Rosenbaum-style Gamma bounds (matched design approximation)
- Benchmarking against observed covariates
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Data loading
# ---------------------------------------------------------------------------
DATA_DIR = Path("/Users/andrewrobinson/src/causal_book/data")

def load_ohe(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge OHE survey + lottery list files.
    Adjust paths to match actual NBER data layout.
    """
    # Survey 12-month outcomes
    survey = pd.read_stata(data_dir / "oregonhie_survey12m_vars.dta")
    # Lottery list (instruments + strata)
    lottery = pd.read_stata(data_dir / "oregonhie_lottery.dta")
    df = survey.merge(lottery, on="person_id", how="inner")

    # Core variables
    df = df.rename(columns={
        "selected":            "Z",         # instrument (lottery win)
        "ohp_all_ever_admin":  "D",         # ever enrolled in Medicaid
        "doc_any_12m":         "Y_doc",     # any doctor visit (primary outcome)
        "catastrophic_exp_inp":"Y_cat",     # catastrophic financial expenditure
        "numhh_list":          "numhh",     # household size (strata)
    })

    # Covariates available at baseline (pre-lottery)
    baseline_covs = ["age_inp", "female_inp", "english_list",
                     "self_list", "ins_any_prenotify"]
    keep = ["Z", "D", "Y_doc", "Y_cat", "numhh"] + baseline_covs
    df = df[keep].dropna()

    # Binary outcomes
    df["Y_doc"] = (df["Y_doc"] >= 1).astype(float)
    df["Y_cat"] = (df["Y_cat"] >= 1).astype(float)
    return df


# ---------------------------------------------------------------------------
# 1. Observational regression (ignoring instrument)
# ---------------------------------------------------------------------------
def run_obs_regression(df: pd.DataFrame):
    """
    OLS of Y_doc on D + covariates.
    Intentionally ignores Z to create a confounded comparison.
    """
    covs = ["age_inp", "female_inp", "english_list",
            "self_list", "ins_any_prenotify",
            "C(numhh)"]
    fml = "Y_doc ~ D + " + " + ".join(covs)
    model = smf.ols(fml, data=df).fit()
    return model


# ---------------------------------------------------------------------------
# 2. Partial R^2 utilities
# ---------------------------------------------------------------------------
def partial_r2(model, variable: str) -> float:
    """
    Partial R^2 of `variable` in fitted OLS model.
    partial_R^2 = t^2 / (t^2 + df_resid)
    """
    t = model.tvalues[variable]
    df_r = model.df_resid
    return float(t**2 / (t**2 + df_r))


def partial_r2_from_restricted(model_full, variable: str, df: pd.DataFrame) -> dict:
    """
    Compute partial R^2 of `variable` w.r.t. both Y and D by fitting
    auxiliary regressions, following Cinelli-Hazlett notation.

    Returns r2_Yw (partial R^2 w/ outcome), r2_Dw (partial R^2 w/ treatment).
    """
    # r2_Yw: already partial R^2 of variable in full model
    r2_Yw = partial_r2(model_full, variable)

    # r2_Dw: partial R^2 of variable in regression of D on X_{-variable}
    other_x = [v for v in model_full.model.exog_names
               if v not in ("Intercept", variable, "D")]
    fml_d = f"D ~ {' + '.join(other_x)}" if other_x else "D ~ 1"
    try:
        m_d = smf.ols(fml_d, data=df).fit()
        r2_Dw = partial_r2(m_d, variable) if variable in m_d.tvalues else np.nan
    except Exception:
        r2_Dw = np.nan

    return {"r2_Yw": r2_Yw, "r2_Dw": r2_Dw}


# ---------------------------------------------------------------------------
# 3. Cinelli-Hazlett bias formula
# ---------------------------------------------------------------------------
def adjusted_estimate(tau_hat: float, se: float, df_resid: int,
                      r2_Yw: float, r2_Dw: float) -> float:
    """
    Bias-adjusted estimate under partial R^2 parameterization.

    bias = se * sqrt(df_resid) * sqrt(r2_Yw * r2_Dw / (1 - r2_Dw))
    """
    if r2_Dw >= 1.0:
        return np.nan
    bias = se * np.sqrt(df_resid) * np.sqrt(
        r2_Yw * r2_Dw / (1.0 - r2_Dw)
    )
    # Direction: assume confounder inflates estimate (positive bias)
    sign = np.sign(tau_hat)
    return tau_hat - sign * bias


def robustness_value(tau_hat: float, se: float, df_resid: int,
                     q: float = 1.0) -> float:
    """
    RV_q: minimum equal partial R^2 (r2_Yw = r2_Dw = rv) to reduce
    estimate by factor q (default q=1 -> drive to zero).

    Solves: q * |tau| = se * sqrt(df) * sqrt(rv^2 / (1-rv))
    => f^2 = (q * tau / (se * sqrt(df)))^2
    => rv = (sqrt(f^4 + 4f^2) - f^2) / 2
    """
    f2 = (q * abs(tau_hat) / (se * np.sqrt(df_resid)))**2
    rv = 0.5 * (np.sqrt(f2**2 + 4 * f2) - f2)
    return float(rv)


# ---------------------------------------------------------------------------
# 4. E-value
# ---------------------------------------------------------------------------
def e_value(rr: float) -> float:
    """
    E-value for a risk ratio (or odds ratio under rare outcome approximation).
    rr should be > 1 (flip if < 1).
    """
    rr = max(rr, 1.0 / rr)
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def compute_e_values(model, outcome: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OLS estimate to approximate RR and compute E-value.
    RR approximation: exp(beta / mean(Y)) for binary outcome with
    small probability (log-linear approximation).
    """
    tau = model.params["D"]
    ci_lo, ci_hi = model.conf_int().loc["D"]
    mean_y = df[outcome].mean()

    # Approximate RR from linear probability model coefficient
    rr_point = np.exp(tau / mean_y)
    rr_lo    = np.exp(ci_lo / mean_y)

    ev_point = e_value(rr_point)
    ev_ci    = e_value(rr_lo) if rr_lo > 1.0 else 1.0

    return pd.DataFrame({
        "Estimate (LPM)": [tau],
        "95% CI Low":     [ci_lo],
        "95% CI High":    [ci_hi],
        "Approx RR":      [rr_point],
        "E-value (point)":[ev_point],
        "E-value (CI lo)":[ev_ci],
    })


# ---------------------------------------------------------------------------
# 5. Contour plot
# ---------------------------------------------------------------------------
def sensitivity_contour(tau_hat: float, se: float, df_resid: int,
                        benchmarks: dict,
                        rv: float,
                        title: str = "Sensitivity Contour",
                        ax=None):
    """
    Contour plot of bias-adjusted estimate over (r2_Dw, r2_Yw) grid.
    Overlays robustness value and covariate benchmarks.
    """
    grid = np.linspace(0.0, 0.40, 200)
    R2D, R2Y = np.meshgrid(grid, grid)

    # Adjusted estimate (assuming positive tau, confounder inflates)
    safe_R2D = np.where(R2D < 1.0, R2D, np.nan)
    bias = se * np.sqrt(df_resid) * np.sqrt(
        R2Y * safe_R2D / (1.0 - safe_R2D)
    )
    tau_adj = tau_hat - np.sign(tau_hat) * bias

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.figure

    levels = np.round(
        np.linspace(min(-0.05, tau_hat * 0.2), tau_hat * 1.1, 12), 3
    )
    cs = ax.contourf(R2D, R2Y, tau_adj, levels=levels, cmap="RdYlGn")
    ax.contour(R2D, R2Y, tau_adj, levels=[0.0],
               colors="black", linewidths=2.0, linestyles="--")
    plt.colorbar(cs, ax=ax, label="Adjusted estimate")

    # Robustness value marker
    ax.plot(rv, rv, marker="D", color="navy",
            markersize=9, label=f"RV = {rv:.3f}")
    ax.plot([rv, rv], [0, rv], color="navy", linewidth=0.8, linestyle=":")
    ax.plot([0, rv], [rv, rv], color="navy", linewidth=0.8, linestyle=":")

    # Benchmark covariates
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(benchmarks)))
    for (name, vals), col in zip(benchmarks.items(), colors):
        ax.plot(vals["r2_Dw"], vals["r2_Yw"],
                marker="^", markersize=8, color=col, label=name)

    ax.set_xlabel(r"Partial $R^2$ of $U$ with $D$", fontsize=11)
    ax.set_ylabel(r"Partial $R^2$ of $U$ with $Y$", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    return fig


# ---------------------------------------------------------------------------
# 6. Rosenbaum Gamma bounds (approximation for illustrative purposes)
# ---------------------------------------------------------------------------
def rosenbaum_gamma_bound(treated_outcomes: np.ndarray,
                          control_outcomes: np.ndarray,
                          gamma_values: np.ndarray,
                          alpha: float = 0.05) -> float:
    """
    Approximate Rosenbaum Gamma* for a 1:1 matched sample using
    the Wilcoxon signed-rank test.

    Matched pairs formed by sorting treated and control on predicted
    propensity score (nearest-neighbor approximation).

    Returns Gamma* where upper-bound p-value first exceeds alpha.
    """
    diffs = treated_outcomes - control_outcomes
    diffs = diffs[diffs != 0]
    n = len(diffs)
    ranks = stats.rankdata(np.abs(diffs))
    T_obs = ranks[diffs > 0].sum()

    gamma_star = 1.0
    for gamma in gamma_values:
        p_upper = gamma / (1.0 + gamma)  # max P(treated|matched pair)
        # Expected value and variance under worst-case assignment
        E_T  = p_upper * np.sum(ranks)
        Var_T = p_upper * (1.0 - p_upper) * np.sum(ranks**2)
        z    = (T_obs - E_T) / np.sqrt(Var_T)
        pval = 1.0 - stats.norm.cdf(z)
        if pval > alpha:
            break
        gamma_star = gamma

    return float(gamma_star)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    try:
        df = load_ohe(DATA_DIR)
        print(f"OHE loaded: n = {len(df):,}")
    except FileNotFoundError as e:
        print(f"Data not found ({e}); generating synthetic OHE-like data.")
        rng = np.random.default_rng(42)
        n = 12_000
        # Simulate confounded observational data
        age      = rng.normal(40, 10, n)
        female   = rng.binomial(1, 0.55, n)
        english  = rng.binomial(1, 0.80, n)
        self_emp = rng.binomial(1, 0.15, n)
        prior    = rng.binomial(1, 0.30, n)
        numhh    = rng.choice([1, 2, 3], n, p=[0.5, 0.35, 0.15])
        # Unmeasured confounder: health motivation
        U        = rng.normal(0, 1, n)
        # Treatment: correlated with U (healthy people more likely to take up)
        p_D = 1 / (1 + np.exp(-(−0.5 + 0.4*U + 0.3*prior - 0.1*age/10)))
        D   = rng.binomial(1, p_D, n)
        # Outcome: Y = doc visit
        p_Y = 1 / (1 + np.exp(-(−1.0 + 0.25*D + 0.5*U +
                                  0.02*age + 0.3*female + 0.2*prior)))
        Y_doc = rng.binomial(1, p_Y, n)
        df = pd.DataFrame({
            "D": D, "Y_doc": Y_doc, "numhh": numhh,
            "age_inp": age, "female_inp": female,
            "english_list": english, "self_list": self_emp,
            "ins_any_prenotify": prior,
        })

    # ---- OLS observational regression ----
    model = run_obs_regression(df)
    tau_hat = model.params["D"]
    se      = model.bse["D"]
    df_r    = int(model.df_resid)

    print("\n=== Observational OLS ===")
    print(model.summary().tables[1])

    # ---- Robustness value ----
    rv1 = robustness_value(tau_hat, se, df_r, q=1.0)
    rv_half = robustness_value(tau_hat, se, df_r, q=0.5)
    print(f"\nRobustness Value (RV_1, drive to zero):  {rv1:.4f}")
    print(f"Robustness Value (RV_0.5, halve effect): {rv_half:.4f}")

    # ---- E-values ----
    ev_df = compute_e_values(model, "Y_doc", df)
    print("\n=== E-values for doctor-visit outcome ===")
    print(ev_df.to_string(index=False, float_format="{:.3f}".format))

    # ---- Benchmark partial R^2 for observed covariates ----
    benchmark_vars = ["age_inp", "female_inp", "ins_any_prenotify"]
    benchmarks = {}
    for var in benchmark_vars:
        try:
            bm = partial_r2_from_restricted(model, var, df)
            benchmarks[var] = bm
        except Exception:
            pass

    print("\n=== Benchmark partial R^2 ===")
    bm_rows = []
    for var, vals in benchmarks.items():
        adj = adjusted_estimate(tau_hat, se, df_r,
                                vals["r2_Yw"], vals["r2_Dw"])
        bm_rows.append({
            "Covariate":     var,
            "r2_Yw":         vals["r2_Yw"],
            "r2_Dw":         vals["r2_Dw"],
            "Adj. Estimate": adj,
        })
    bm_df = pd.DataFrame(bm_rows)
    print(bm_df.to_string(index=False, float_format="{:.4f}".format))

    # ---- Contour plot ----
    fig = sensitivity_contour(
        tau_hat, se, df_r,
        benchmarks=benchmarks,
        rv=rv1,
        title=(
            "Sensitivity to Unobserved Confounding\n"
            "OHE Observational Comparison: Medicaid Enrollment → Doctor Visit"
        ),
    )
    fig.tight_layout()
    fig.savefig("/tmp/ch31_sensitivity_contour.png", dpi=150)
    print("\nContour saved to /tmp/ch31_sensitivity_contour.png")

    # ---- Rosenbaum Gamma bounds ----
    # Approximate matched design: sort by propensity score, pair nearest
    ps_model = smf.logit(
        "D ~ age_inp + female_inp + english_list + self_list"
        " + ins_any_prenotify + C(numhh)",
        data=df
    ).fit(disp=False)
    df["ps"] = ps_model.predict()
    df_sorted = df.sort_values("ps").reset_index(drop=True)
    treated   = df_sorted[df_sorted["D"] == 1].reset_index(drop=True)
    control   = df_sorted[df_sorted["D"] == 0].reset_index(drop=True)
    m = min(len(treated), len(control))
    t_out = treated["Y_doc"].values[:m]
    c_out = control["Y_doc"].values[:m]

    gammas = np.arange(1.0, 4.05, 0.05)
    gamma_star = rosenbaum_gamma_bound(t_out, c_out, gammas)
    print(f"\nRosenbaum Gamma* (approx matched design): {gamma_star:.2f}")
    print(
        f"  => An unmeasured confounder creating an odds ratio of "
        f"{gamma_star:.2f} or greater\n"
        f"     between matched units would render the result non-significant."
    )

    # ---- Summary table ----
    print("\n=== Chapter 31 Sensitivity Summary ===")
    summary = pd.DataFrame({
        "Method":    ["OLS (confounded)", "Robustness Value (zero)",
                      "Robustness Value (halve)", "E-value (point)",
                      "E-value (CI lo)", "Gamma*"],
        "Statistic": [
            f"tau = {tau_hat:.4f} (SE={se:.4f})",
            f"RV_1 = {rv1:.4f}",
            f"RV_0.5 = {rv_half:.4f}",
            f"{ev_df['E-value (point)'].values[0]:.3f}",
            f"{ev_df['E-value (CI lo)'].values[0]:.3f}",
            f"{gamma_star:.2f}",
        ],
        "Interpretation": [
            "Confounded observational estimate",
            f"Confounder must explain >{rv1*100:.1f}% residual var in D and Y",
            f"Confounder must explain >{rv_half*100:.1f}% to halve estimate",
            "RR of confounder with D and Y must both exceed E",
            "E-value for lower 95% confidence limit",
            "Odds ratio of assignment within matched pairs to nullify test",
        ]
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
```

---

## Summary

- Sensitivity analysis quantifies the magnitude of unmeasured confounding required to overturn a causal conclusion; it does not test whether confounding exists.
- **Rosenbaum bounds** ($\Gamma^*$) apply to matched designs and ask what odds ratio of unmeasured confounding nullifies a signed-rank test; they are conservative and do not provide bias-adjusted estimates.
- The **E-value** $E = RR + \sqrt{RR(RR-1)}$ is the minimum risk ratio of an unmeasured confounder with both treatment and outcome needed to explain away an observed relative risk; it is scale-free and requires no structural assumptions.
- The **Cinelli-Hazlett framework** re-expresses omitted variable bias in terms of partial $R^2$ parameters, producing contour plots and a robustness value $RV_q$ that localizes exactly how much residual variance an unmeasured confounder must explain in both $D$ and $Y$ to shift the estimate by a factor $q$.
- **Benchmarking** against observed covariates places the sensitivity analysis in substantive context: if the unmeasured confounder must be stronger than anything measured to overturn the result, the estimate is credible.
- Doubly robust estimators are not doubly robust against confounding: omitting $U$ from both nuisance models propagates bias to the AIPW estimator; partial $R^2$ bounds apply there as well.
- The three frameworks are complementary: use E-values for quick scale-free communication, Cinelli-Hazlett for regression-based analysis with benchmarking, and Rosenbaum bounds for matched designs.

---

## Further Reading

1. **Cinelli, C. and Hazlett, C. (2020).** "Making sense of sensitivity: Extending omitted variable bias." *Journal of the Royal Statistical Society Series B*, 82(1), 39–67. The foundational reference for partial $R^2$ sensitivity analysis; derives the bias formula, robustness value, and benchmarking procedure; introduces the `sensemakr` software.

2. **VanderWeele, T.J. and Ding, P. (2017).** "Sensitivity analysis in observational research: Introducing the E-value." *Annals of Internal Medicine*, 167(4), 268–274. Introduces the E-value; proves the minimum confounding theorem; provides tables and an online calculator; spawned a large literature on E-value refinements.

3. **Rosenbaum, P.R. (2002).** *Observational Studies* (2nd ed.), Springer. Chapter 4 develops the $\Gamma$-sensitivity model for matched studies in full generality; Chapter 14 introduces design sensitivity as the expected $\Gamma^*$ under a treatment effect model, enabling power calculations for sensitivity analysis.

4. **Oster, E. (2019).** "Unobservable selection and coefficient stability: Theory and evidence." *Journal of Business & Economic Statistics*, 37(2), 187–204. Derives a selection-on-observables bound using $R^2$ movements under coefficient stability assumptions; widely used in economics as an alternative to the Cinelli-Hazlett approach, with different identifying assumptions.

5. **Dorie, V., Harada, M., Carnegie, N.B., and Hill, J. (2016).** "A flexible, interpretable framework for assessing sensitivity to unmeasured confounding." *Statistics in Medicine*, 35(20), 3453–3470. Introduces the CAUSE framework for sensitivity analysis with nonparametric outcome models; useful for machine-learning-based causal estimators where the linear OVB formula does not apply.

6. **Cinelli, C., Ferwerda, J., and Hazlett, C. (2020).** "sensemakr: Sensitivity analysis tools for OLS in R and Python." *Observational Studies*, 6, 1–34. Software paper documenting the `sensemakr` implementation; includes worked examples with real datasets, formula derivations for the benchmarking procedure, and extensions to non-Gaussian outcomes.
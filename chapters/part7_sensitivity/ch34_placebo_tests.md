# Chapter 34: Placebo Tests as a System

Placebo tests are among the most useful diagnostic tools in the causal inference toolkit, yet they are frequently applied in an ad hoc, piecemeal fashion. A researcher might show a single pre-trend plot and call it done, or assign a fake treatment date to a single control unit and report the null result without examining its statistical content. This chapter systematizes the practice. We develop a taxonomy of placebo tests, formalize the permutation inference framework that gives them statistical content, work through the false discovery rate problem that arises when running many placebos, and build toward a combined summary test. The ACA Medicaid expansion provides the primary empirical canvas throughout.

---

## 34.1 A Taxonomy of Placebo Tests

A placebo test is any test of a prediction that follows from $H_0$: no causal effect, and that, if violated, would undermine the identification strategy rather than just the null of zero effect. The distinction matters. Pre-trend tests do not test whether the treatment effect is zero; they test whether the parallel trends assumption—required for DiD to be identified—holds in a period when it must hold if it holds at all.

Four families cover almost all applications:

**Outcome placebos.** Replace the outcome $Y$ with a variable that the treatment cannot plausibly affect. In the ACA context, general health outcomes were affected by Medicaid expansion; mortality from external causes (accidents, homicides) should not be. A large coefficient on the external-cause mortality placebo signals confounding—some other state-level change correlated with expansion timing is driving estimates.

**Treatment placebos.** Assign a fake treatment to units that did not receive the actual treatment, or shift the treatment timing to a period before the actual policy change. The prediction under $H_0$ is that the estimated effect is near zero.

**Period placebos (pre-trend tests).** In a DiD design with panel data, restrict the sample to the pre-treatment period and assign a fake treatment cutoff at $t^* < t_0$. The true treatment indicator becomes a pure placebo: no one actually received anything. A large coefficient signals that the "treated" and "control" units were not on parallel trends before treatment.

**Geographic or in-space placebos.** Assign the actual treatment to units that did not receive it—holding the treated unit fixed—and re-estimate the effect. This is the synthetic control permutation approach. For each donor unit, construct a synthetic version, measure pre-period fit, and compute the ratio of post-to-pre RMSPE. The rank of the actual unit's ratio gives the $p$-value.

These four families are not mutually exclusive. In a well-executed empirical project, you run all four where feasible. The question is how to aggregate the evidence across them, and how to interpret any single rejection.

---

## 34.2 Permutation Inference: Foundations

The inferential engine underlying placebo tests is the permutation test, which derives $p$-values from the finite-sample distribution of a test statistic under permutations of treatment assignment.

**Setup.** Let $\mathbf{W} = (W_1, \ldots, W_N)$ be the observed treatment vector. Let $\mathcal{W}$ be the set of permissible treatment vectors under the assignment mechanism. For a completely randomized experiment, $|\mathcal{W}| = \binom{N}{N_1}$ where $N_1 = \sum_i W_i$. For a stratified design, permutations are within strata.

**Sharp null.** The Fisher sharp null is $H_0^F: Y_i(1) = Y_i(0)$ for all $i$. Under this null, all potential outcomes are known—the missing potential outcome for each unit equals the observed outcome—so the distribution of any test statistic $T(\mathbf{W}, \mathbf{Y})$ over $\mathcal{W}$ is exact and finite-sample valid.

**Theorem 34.1 (Exactness of permutation test).** Let $T(\mathbf{W}, \mathbf{Y})$ be any test statistic and suppose treatment $\mathbf{W}^{obs}$ is drawn uniformly from $\mathcal{W}$. Under $H_0^F$, potential outcomes are fixed. Then

$$p = \frac{1}{|\mathcal{W}|} \sum_{\mathbf{w} \in \mathcal{W}} \mathbf{1}[T(\mathbf{w}, \mathbf{Y}) \geq T(\mathbf{W}^{obs}, \mathbf{Y})]$$

satisfies $\Pr(p \leq \alpha) \leq \alpha$ for all $\alpha \in [0, 1]$ and all finite $N$.

*Proof sketch.* Under $H_0^F$, $\mathbf{Y}$ is the same vector regardless of $\mathbf{W}$, so $T(\mathbf{w}, \mathbf{Y})$ is a function of $\mathbf{w}$ alone. Since $\mathbf{W}^{obs}$ is uniform over $\mathcal{W}$, the rank of $T(\mathbf{W}^{obs}, \mathbf{Y})$ in the permutation distribution is uniform on $\{1, \ldots, |\mathcal{W}|\}$. Dividing by $|\mathcal{W}|$ gives an exactly uniform $p$-value. $\square$

**When $|\mathcal{W}|$ is large.** In observational settings and panel DiD with many units, enumerating all permutations is infeasible. We draw $M$ permutations $\{\mathbf{w}^{(1)}, \ldots, \mathbf{w}^{(M)}\}$ uniformly at random and compute

$$\hat{p} = \frac{1}{M} \sum_{m=1}^M \mathbf{1}\!\left[|T(\mathbf{w}^{(m)}, \mathbf{Y})| \geq |T(\mathbf{W}^{obs}, \mathbf{Y})|\right]$$

This is a Monte Carlo approximation to the exact $p$-value. By the law of large numbers, $\hat{p} \to p$ as $M \to \infty$. A common convention is $M = 999$ so that $\hat{p} \in \{0.001, 0.002, \ldots, 1.000\}$.

**The exchangeability condition.** Theorem 34.1 requires only that $\mathbf{W}^{obs}$ is uniform over $\mathcal{W}$—it does not require random assignment in the strong sense. In observational studies with staggered adoption, the permutations must respect the assignment mechanism. For state-level Medicaid expansion, we permute which states expanded and when, but we should preserve features of the assignment mechanism that are known (e.g., the number of expanding states per year). Failure to respect the assignment mechanism yields permutation tests that are not exact and may be anti-conservative.

---

## 34.3 Pre-Trend Tests in Difference-in-Differences

The canonical pre-trend test estimates an event-study regression and examines whether the leads—coefficients on indicators for periods before the treatment—are jointly zero.

**Specification.** Let $Y_{it}$ be the outcome for state $i$ at time $t$, and let $D_{it} = 1$ if state $i$ has expanded Medicaid by year $t$. Define event-time $\ell = t - T_i^*$ where $T_i^*$ is the expansion year. The event-study model is

$$Y_{it} = \alpha_i + \gamma_t + \sum_{\ell = -L}^{-2} \delta_\ell \cdot \mathbf{1}[t - T_i^* = \ell] + \sum_{\ell = 0}^{K} \delta_\ell \cdot \mathbf{1}[t - T_i^* = \ell] + \varepsilon_{it}$$

with $\ell = -1$ omitted as the reference period. The placebo test is $H_0: \delta_{-L} = \delta_{-L+1} = \cdots = \delta_{-2} = 0$.

The conventional $F$-test of joint significance uses asymptotic critical values. The permutation analog reshuffles the treatment timing across states within each year and recomputes the pre-trend statistic, building an exact distribution under the null that treatment timing is unrelated to pre-trends.

**Size calibration.** A persistent misunderstanding is that pre-trend tests are tests of parallel trends. They are not—they test the observable implication of parallel trends in the pre-period. Parallel trends is an assumption about the counterfactual post-treatment trend. Pre-trend tests can fail even when parallel trends holds (if there is heterogeneous anticipation) and can pass even when parallel trends fails (if the confounding only acts post-treatment).

**What does a well-calibrated pre-trend test look like?** Under the null of parallel trends, the distribution of the pre-trend $F$-statistic should match a central $F(L-1, \infty)$ distribution. Equivalently, if we run the permutation test across many placebo treatments drawn from units that all satisfy the null, the rejection rate at the 5% level should be approximately 5%. If we observe 25% rejection across 100 fake treatment assignments in the never-treated states, the test is over-sized—a signal of persistent heterogeneity or time-varying confounding.

---

## 34.4 In-Space Placebos for Synthetic Control

The synthetic control method (Chapter 19) constructs a weighted average of donor units to match the pre-treatment trajectory of the treated unit. Statistical inference is non-standard because there is typically one treated unit and many pre-treatment periods, making large-sample theory inapplicable.

**RMSPE ratio.** For the treated unit $i = 1$ and each donor unit $i = 2, \ldots, J+1$, construct the synthetic control and compute

$$\rho_i = \frac{\text{RMSPE}_{post}(i)}{\text{RMSPE}_{pre}(i)}$$

where $\text{RMSPE}_{pre}(i) = \sqrt{T_{pre}^{-1}\sum_{t=1}^{T_0}(Y_{it} - \hat{Y}_{it}^{SC})^2}$ and analogously for the post period. For donor units, $\text{RMSPE}_{pre}(i)$ reflects how well the synthetic control can match that unit's pre-period—units with poor pre-period fit have inflated $\rho_i$ for mechanical reasons.

**The rank-based $p$-value.** The in-space placebo $p$-value is

$$p_{SC} = \frac{\text{rank of } \rho_1 \text{ among } \{\rho_1, \rho_2, \ldots, \rho_{J+1}\}}{J+1}$$

where a higher rank corresponds to a larger ratio. Under the null that all units are exchangeable (no causal effect for unit 1), this $p$-value is exactly uniform by Theorem 34.1 applied to the permutation group of which unit is designated "treated."

**Discarding poor pre-fit donors.** The validity of this inference requires that all $\rho_i$ are comparable. If donor units with very poor pre-period fit are included, their $\rho_i$ values are inflated for reasons unrelated to treatment effects, making the inference conservative (too hard to reject the null). A common practice is to discard donor units whose $\text{RMSPE}_{pre}$ exceeds some multiple—commonly 2$\times$—of the treated unit's $\text{RMSPE}_{pre}$. The modified $p$-value uses only comparable donors; the number $J$ in the denominator changes accordingly.

---

## 34.5 False Discovery Rate Under Multiple Placebos

When running $M$ placebo tests, the probability of at least one false positive at the $\alpha$ level is $1 - (1-\alpha)^M$ under independence—which approaches 1 rapidly. This is not a reason to run fewer placebos. It is a reason to apply multiple testing corrections and to interpret the collection of test results jointly rather than individually.

**The Bonferroni bound.** For $M$ tests at family-wise error rate (FWER) $\alpha$, use individual level $\alpha/M$. This is exact for independent tests and conservative for positively correlated tests. In practice, placebo tests are positively correlated—if there is a common confounder, many placebos will show spurious effects simultaneously.

**The Benjamini-Hochberg procedure (BH).** For FDR control at level $q$, order the $M$ $p$-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(M)}$ and reject all hypotheses with $p_{(j)} \leq p_{(j^*)}$ where

$$j^* = \max\left\{j : p_{(j)} \leq \frac{j \cdot q}{M}\right\}$$

**Theorem 34.2 (BH FDR control).** Under independence of test statistics, the BH procedure at level $q$ satisfies $\mathbb{E}[V/R] \leq q \cdot M_0/M \leq q$, where $V$ is the number of false rejections and $R$ is the total number of rejections.

*This result (Benjamini and Hochberg, 1995) also holds under PRDS dependence (positive regression dependence on a subset), which typically characterizes spatially or temporally correlated placebo tests.*

**Expected rejection rate calibration.** Suppose we have $M$ outcome placebos—variables that the treatment cannot affect—and we observe $k$ rejections at the 5% level. The expected $k$ under the null is $0.05M$. Define the empirical over-rejection rate as $k/M - 0.05$. Any value substantially above zero signals either (a) a common confounder affecting both the true outcome and the placebo outcomes, or (b) a specification error that inflates rejection rates uniformly.

Placebo calibration is a two-step diagnostic:

1. Run all $M$ placebos and compute rejection rates at several $\alpha$ levels.
2. Plot the empirical CDF of placebo $p$-values against the 45-degree line. Departure above the line (stochastic dominance over Uniform[0,1]) signals over-rejection.

This calibration plot—sometimes called a "PP plot" of placebo $p$-values—is among the most informative diagnostics in empirical work.

---

## 34.6 Combining Placebos into a Summary Test

Multiple placebos provide evidence of varying strength. A researcher who runs 20 outcome placebos and finds rejections at rates consistent with $\alpha$ has stronger design evidence than one who runs two. Combining them into a single summary statistic with known size is nontrivial but feasible.

**Fisher's combination statistic.** For $M$ independent $p$-values $p_1, \ldots, p_M$, the statistic

$$Q = -2 \sum_{m=1}^M \log p_m \sim \chi^2(2M)$$

under the global null that all $M$ tests have null distributions. This can be applied to the collection of outcome placebo $p$-values: if the combined $Q$ is well within the $\chi^2(2M)$ distribution, the evidence favors design validity. If $Q$ is extreme, something in the design is off.

**Caution on dependence.** Fisher's combination assumes independence. Placebo $p$-values derived from spatially proximate outcomes or from the same dataset are dependent. The Brown (1975) extension adjusts the degrees of freedom of the combination test to account for pairwise correlations. Alternatively, the permutation distribution of $Q$ can be estimated by permuting all tests simultaneously under the same treatment assignment, preserving the dependence structure.

**A structured reporting framework.** Rather than reporting placebo tests individually, report:

- The number of placebos in each category (outcome, treatment, period, space)
- The empirical rejection rate in each category at 5% and 10%
- The PP-plot of placebo $p$-values against Uniform[0,1]
- The Fisher combined $Q$ statistic and its $p$-value (or permutation analog)
- The FDR-adjusted rejections under BH

This framework converts the usual narrative ("we ran some placebo tests and nothing was significant") into a structured evidence report with interpretable statistical content.

---

## Python: Systematic Placebo Testing for ACA DiD and Synthetic Control

```python
"""
Chapter 34: Placebo Tests as a System
ACA Medicaid Expansion DiD + Synthetic Control in-space placebos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Load and prepare ACA/BRFSS data ──────────────────────────────────────

def load_aca_data():
    """
    Load ACA Medicaid expansion panel data.
    Uses a simulated panel that mirrors BRFSS state-year structure.
    Replace with actual BRFSS download for production use.
    """
    rng = np.random.default_rng(42)
    states = [f"S{i:02d}" for i in range(1, 51)]
    years = list(range(2010, 2017))

    # Medicaid expansion years (actual ACA staggered adoption)
    expansion_years = {
        "S01": 2014, "S02": 2014, "S03": 2014, "S04": 2014, "S05": 2014,
        "S06": 2014, "S07": 2014, "S08": 2014, "S09": 2014, "S10": 2014,
        "S11": 2014, "S12": 2014, "S13": 2014, "S14": 2014, "S15": 2014,
        "S16": 2014, "S17": 2015, "S18": 2015, "S19": 2015, "S20": 2016,
        "S21": 2016,
        # States 22-50: never expanded in sample window
    }

    records = []
    state_fe = rng.normal(0, 2, size=50)
    year_fe = rng.normal(0, 1, size=7)

    for s_idx, state in enumerate(states):
        for y_idx, year in enumerate(years):
            exp_year = expansion_years.get(state, 9999)
            treated = int(year >= exp_year)
            post_years = max(0, year - exp_year) if treated else 0

            # True effect on uninsured rate: -3 pp, linear ramp-up
            effect = -3.0 * treated

            # Outcome 1: uninsured rate (true outcome)
            uninsured = (
                20
                + state_fe[s_idx]
                + year_fe[y_idx]
                + effect
                + rng.normal(0, 1.5)
            )

            # Outcome 2: poor mental health days (partial effect)
            mental_health = (
                5
                + 0.5 * state_fe[s_idx]
                + 0.3 * year_fe[y_idx]
                - 0.8 * treated
                + rng.normal(0, 0.8)
            )

            # Placebo outcomes: traffic fatalities (no causal path)
            traffic_deaths = (
                12
                + 0.2 * state_fe[s_idx]
                + 0.1 * year_fe[y_idx]
                + rng.normal(0, 1.2)
            )

            # Placebo 2: crop yields (clearly unrelated)
            crop_yield = (
                100
                + rng.normal(0, 5)
            )

            records.append({
                "state": state,
                "year": year,
                "exp_year": exp_year,
                "treated": treated,
                "uninsured": uninsured,
                "mental_health": mental_health,
                "traffic_deaths": traffic_deaths,
                "crop_yield": crop_yield,
                "state_idx": s_idx,
            })

    return pd.DataFrame(records)


df = load_aca_data()
df["event_time"] = df["year"] - df["exp_year"].clip(upper=2016)
never_expanded = df["exp_year"] == 9999

# ─── 2. Base DiD estimator ────────────────────────────────────────────────────

def did_twfe(data, outcome):
    """Two-way FE DiD: returns coefficient on 'treated' and its SE."""
    mod = ols(f"{outcome} ~ treated + C(state) + C(year)", data=data).fit()
    return mod.params["treated"], mod.bse["treated"]

coef_true, se_true = did_twfe(df, "uninsured")
print(f"True outcome (uninsured rate): β = {coef_true:.3f}  SE = {se_true:.3f}")

# ─── 3. Permutation test: 50-state treatment-timing placebos ─────────────────

def permute_treatment(data, rng):
    """
    Permute treatment timing within the set of states that actually expanded.
    Preserves the number of treated states per year (assignment mechanism).
    """
    # Identify the actual expansion-year distribution
    expanders = data[["state", "exp_year"]].drop_duplicates()
    expanders = expanders[expanders["exp_year"] < 9999].copy()
    # Shuffle which state gets which expansion year
    shuffled_years = rng.permutation(expanders["exp_year"].values)
    expanders["perm_exp_year"] = shuffled_years

    perm_map = dict(zip(expanders["state"], expanders["perm_exp_year"]))
    data = data.copy()
    data["exp_year_perm"] = data["state"].map(perm_map).fillna(9999)
    data["treated_perm"] = (data["year"] >= data["exp_year_perm"]).astype(int)
    return data


M = 999
rng_perm = np.random.default_rng(0)
perm_coefs = []

for _ in range(M):
    df_perm = permute_treatment(df, rng_perm)
    coef_p, _ = did_twfe(df_perm, "uninsured")
    perm_coefs.append(coef_p)

perm_coefs = np.array(perm_coefs)
p_perm = np.mean(np.abs(perm_coefs) >= np.abs(coef_true))
print(f"\nPermutation p-value (uninsured, treatment timing): {p_perm:.3f}")

# ─── 4. Outcome placebos: distribution of rejection rates ────────────────────

placebo_outcomes = ["traffic_deaths", "crop_yield"]
# Expand: generate 20 synthetic placebo outcomes (null by construction)
rng_out = np.random.default_rng(1)

for k in range(18):
    varname = f"placebo_null_{k}"
    df[varname] = (
        rng_out.normal(0, 2, size=len(df))
        + df["state_idx"] * rng_out.uniform(-0.2, 0.2)
    )
    placebo_outcomes.append(varname)

n_placebos = len(placebo_outcomes)
placebo_pvals = []

for outcome in placebo_outcomes:
    coef_p, se_p = did_twfe(df, outcome)
    # Two-sided t-test p-value (asymptotic, for outcome placebos)
    t_stat = coef_p / se_p
    pval = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    placebo_pvals.append(pval)

placebo_pvals = np.array(placebo_pvals)

# Rejection rates
for alpha in [0.05, 0.10]:
    rate = np.mean(placebo_pvals < alpha)
    print(f"Outcome placebo rejection rate at α={alpha}: {rate:.3f}  "
          f"(expected {alpha:.3f})")

# ─── 5. Benjamini-Hochberg FDR correction ────────────────────────────────────

def bh_correction(pvals, q=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean array of rejections."""
    M = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = (np.arange(1, M + 1) / M) * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(M, dtype=bool)
    cutoff = np.max(np.where(below)[0])
    reject = np.zeros(M, dtype=bool)
    reject[order[:cutoff + 1]] = True
    return reject

bh_reject = bh_correction(placebo_pvals, q=0.05)
print(f"\nBH rejections among {n_placebos} outcome placebos: {bh_reject.sum()}")

# ─── 6. Fisher combination test ──────────────────────────────────────────────

Q_fisher = -2 * np.sum(np.log(placebo_pvals))
df_fisher = 2 * n_placebos
p_fisher = 1 - stats.chi2.cdf(Q_fisher, df=df_fisher)
print(f"Fisher combined Q = {Q_fisher:.2f}  (df={df_fisher})  "
      f"p = {p_fisher:.3f}")

# ─── 7. In-space synthetic control permutation ───────────────────────────────

def synthetic_control_simple(treated_series, donor_matrix):
    """
    Minimal SC via constrained least squares (no Synth package dependency).
    Fits donor weights to minimize pre-period RMSPE.
    Returns post-period predictions using fitted weights.
    """
    from scipy.optimize import minimize

    n_donors = donor_matrix.shape[1]

    def objective(w):
        synth = donor_matrix @ w
        return np.sum((treated_series - synth) ** 2)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


def rmspe(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


# Build state-year pivot for uninsured rate
pivot = df.pivot(index="state", columns="year", values="uninsured")
years_arr = np.array(pivot.columns)
T0_idx = np.searchsorted(years_arr, 2014)  # pre-period: 2010-2013

# Pick one "treated" state: first expander
treated_state = "S01"
donor_states = [s for s in pivot.index if s != treated_state]

treated_vec = pivot.loc[treated_state].values
donor_mat = pivot.loc[donor_states].values.T  # shape (T, J)

pre_treated = treated_vec[:T0_idx]
pre_donor = donor_mat[:T0_idx, :]
post_treated = treated_vec[T0_idx:]
post_donor = donor_mat[T0_idx:, :]

# Fit SC weights
w_sc = synthetic_control_simple(pre_treated, pre_donor)

# RMSPE ratio for treated unit
pre_fit_treated = pre_donor @ w_sc
post_fit_treated = post_donor @ w_sc
rmspe_pre_treated = rmspe(pre_treated, pre_fit_treated)
rmspe_post_treated = rmspe(post_treated, post_fit_treated)
rho_treated = rmspe_post_treated / rmspe_pre_treated

# In-space placebos: cycle through donor states
rho_donors = []
for j_idx, donor in enumerate(donor_states):
    other_donors = [s for s in donor_states if s != donor]
    donor_vec = pivot.loc[donor].values
    other_mat = pivot.loc[other_donors].values.T

    pre_d = donor_vec[:T0_idx]
    post_d = donor_vec[T0_idx:]
    pre_other = other_mat[:T0_idx, :]
    post_other = other_mat[T0_idx:, :]

    w_d = synthetic_control_simple(pre_d, pre_other)

    pre_fit_d = pre_other @ w_d
    post_fit_d = post_other @ w_d

    rmspe_pre_d = rmspe(pre_d, pre_fit_d)
    if rmspe_pre_d < 1e-8:
        rho_donors.append(np.nan)
        continue
    rmspe_post_d = rmspe(post_d, post_fit_d)
    rho_donors.append(rmspe_post_d / rmspe_pre_d)

rho_donors = np.array(rho_donors)

# Filter: discard donors with pre-RMSPE > 2x treated pre-RMSPE
pre_rmspe_donors = []
for j_idx, donor in enumerate(donor_states):
    other_donors = [s for s in donor_states if s != donor]
    donor_vec = pivot.loc[donor].values[:T0_idx]
    other_mat = pivot.loc[other_donors].values.T[:T0_idx, :]
    w_d = synthetic_control_simple(donor_vec, other_mat)
    pre_rmspe_donors.append(rmspe(donor_vec, other_mat @ w_d))

pre_rmspe_donors = np.array(pre_rmspe_donors)
good_donors = pre_rmspe_donors <= 2 * rmspe_pre_treated
rho_comparable = rho_donors[good_donors]

all_rhos = np.concatenate([[rho_treated], rho_comparable[~np.isnan(rho_comparable)]])
p_sc = np.mean(all_rhos >= rho_treated)
print(f"\nIn-space SC p-value (rank-based): {p_sc:.3f}  "
      f"(treated RMSPE ratio: {rho_treated:.2f})")

# ─── 8. Plots ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# Panel A: permutation distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(perm_coefs, bins=50, color="#4878CF", alpha=0.75, edgecolor="white")
ax1.axvline(coef_true, color="#D65F5F", linewidth=2, label=f"Actual β = {coef_true:.2f}")
ax1.axvline(-abs(coef_true), color="#D65F5F", linewidth=2, linestyle="--")
ax1.set_xlabel("Permuted treatment coefficient")
ax1.set_ylabel("Count")
ax1.set_title(f"A. Permutation Distribution\n(Uninsured Rate, p = {p_perm:.3f})")
ax1.legend(fontsize=8)

# Panel B: PP-plot of outcome placebo p-values
ax2 = fig.add_subplot(gs[0, 1])
sorted_pvals = np.sort(placebo_pvals)
uniform_quantiles = np.arange(1, n_placebos + 1) / (n_placebos + 1)
ax2.plot(uniform_quantiles, sorted_pvals, "o", color="#6ACC65",
         markersize=5, label="Placebo p-values")
ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Uniform[0,1]")
ax2.set_xlabel("Expected quantile (Uniform)")
ax2.set_ylabel("Observed p-value quantile")
ax2.set_title("B. PP-Plot: Outcome Placebo p-values\n(Calibration check)")
ax2.legend(fontsize=8)

# Panel C: in-space SC RMSPE ratios
ax3 = fig.add_subplot(gs[1, 0])
good_rhos = rho_comparable[~np.isnan(rho_comparable)]
ax3.hist(good_rhos, bins=20, color="#B47CC7", alpha=0.75, edgecolor="white",
         label="Donor units")
ax3.axvline(rho_treated, color="#D65F5F", linewidth=2,
            label=f"Treated ρ = {rho_treated:.2f}")
ax3.set_xlabel("Post/Pre RMSPE ratio")
ax3.set_ylabel("Count")
ax3.set_title(f"C. In-Space SC Placebos\n(p = {p_sc:.3f})")
ax3.legend(fontsize=8)

# Panel D: sorted placebo p-values vs BH thresholds
ax4 = fig.add_subplot(gs[1, 1])
sorted_idx = np.argsort(placebo_pvals)
sorted_p_plot = placebo_pvals[sorted_idx]
ranks = np.arange(1, n_placebos + 1)
bh_threshold = ranks * 0.05 / n_placebos
ax4.plot(ranks, sorted_p_plot, "o", color="#4878CF", markersize=5,
         label="Sorted p-values")
ax4.plot(ranks, bh_threshold, "r--", linewidth=1.5,
         label="BH threshold (q=0.05)")
ax4.set_xlabel("Rank")
ax4.set_ylabel("p-value")
ax4.set_title(f"D. BH FDR Correction\n({bh_reject.sum()} rejections)")
ax4.legend(fontsize=8)

plt.suptitle("Chapter 34: Placebo Test System — ACA Medicaid Expansion",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig("/tmp/ch34_placebo_system.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved: /tmp/ch34_placebo_system.png")

# ─── 9. Summary table ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PLACEBO TEST SYSTEM SUMMARY")
print("="*60)
summary = pd.DataFrame({
    "Test type": [
        "Treatment-timing permutation",
        "Outcome placebos (5% level)",
        "Outcome placebos (10% level)",
        "Fisher combined test",
        "BH-adjusted rejections",
        "In-space SC (rank p-value)",
    ],
    "Statistic": [
        f"β = {coef_true:.3f}",
        f"{np.mean(placebo_pvals < 0.05):.3f} rejection rate",
        f"{np.mean(placebo_pvals < 0.10):.3f} rejection rate",
        f"Q = {Q_fisher:.1f} (df={df_fisher})",
        f"{bh_reject.sum()} / {n_placebos}",
        f"ρ = {rho_treated:.2f}",
    ],
    "p-value": [
        f"{p_perm:.3f}",
        f"Expected 0.050",
        f"Expected 0.100",
        f"{p_fisher:.3f}",
        "FDR = 0.05",
        f"{p_sc:.3f}",
    ],
    "Verdict": [
        "Reject H₀" if p_perm < 0.05 else "Fail to reject",
        "Calibrated" if abs(np.mean(placebo_pvals < 0.05) - 0.05) < 0.05 else "Miscalibrated",
        "Calibrated" if abs(np.mean(placebo_pvals < 0.10) - 0.10) < 0.05 else "Miscalibrated",
        "Fail to reject" if p_fisher > 0.05 else "Reject",
        f"{bh_reject.sum()} false positives",
        "Reject H₀" if p_sc < 0.05 else "Fail to reject",
    ],
})
print(summary.to_string(index=False))
```

---

## Summary

- A placebo test is a test of an observable implication of $H_0$: no causal effect, whose rejection signals a design flaw rather than simply a nonzero effect; the four families are outcome, treatment, period, and geographic placebos.
- Permutation $p$-values are exact in finite samples under the sharp null and exchangeability of treatment assignment; approximate permutation via $M = 999$ random draws is the practical implementation.
- Pre-trend tests do not test the parallel trends assumption directly; they test its observable implication in the pre-period, and they can pass when parallel trends fails post-treatment.
- In-space synthetic control placebos use the RMSPE ratio $\rho_i = \text{RMSPE}_{post}/\text{RMSPE}_{pre}$ for each donor unit; the rank of the treated unit's $\rho$ among comparable donors yields a rank-based $p$-value exact under exchangeability.
- Running $M$ placebo tests inflates the FWER; Bonferroni controls FWER at the cost of power, while Benjamini-Hochberg controls FDR and is more powerful under positive dependence.
- Placebo calibration—comparing the empirical rejection rate across many placebos to the nominal $\alpha$—detects systematic over-rejection that signals confounding or specification error, and is best visualized as a PP-plot of placebo $p$-values against Uniform[0,1].
- Fisher's combination statistic $Q = -2\sum_m \log p_m \sim \chi^2(2M)$ converts a collection of placebo $p$-values into a single omnibus test; when tests are dependent, the permutation analog of $Q$ preserves exact size.

---

## Further Reading

1. **Abadie, Diamond, and Hainmueller (2010).** "Synthetic Control Methods for Comparative Case Studies." *JASA* 105(490). The original synthetic control paper; Section 4 develops the in-space placebo inference framework and the RMSPE ratio test statistic in detail.

2. **Benjamini and Hochberg (1995).** "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *JRSS-B* 57(1). The original BH procedure; Theorem 1 establishes FDR control under independence, and Section 4 discusses the PRDS extension relevant to spatially correlated placebos.

3. **Roth (2022).** "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *AER: Insights* 4(3). Demonstrates that conditioning on passing a pre-trend test distorts the distribution of post-treatment estimates; develops power-adjusted pre-trend tests and recommends sensitivity analysis over binary pass/fail.

4. **Young (2019).** "Channeling Fisher: Randomization Tests and the Statistical Insignificance of Seemingly Significant Experimental Results." *QJE* 134(2). Applies permutation inference to re-examine a large body of randomized experiments; findings on over-rejection under asymptotic tests motivate the permutation approach for DiD designs.

5. **Athey and Imbens (2018).** "Design-Based Analysis in Difference-in-Differences Settings with Staggered Adoption." *NBER Working Paper* 24963. Develops the formal assignment mechanism for staggered DiD and derives exact permutation tests that respect the staggered structure; essential for understanding what permutations are valid under staggered adoption.

6. **Fisher (1935).** *The Design of Experiments.* Oliver and Boyd. Chapters III–IV develop the randomization test from first principles; the argument that the sharp null makes all potential outcomes known—and thus the permutation distribution exact—remains the cleanest statement of what permutation inference achieves.
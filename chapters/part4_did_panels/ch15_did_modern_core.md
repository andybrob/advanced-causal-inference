# Chapter 15: Difference-in-Differences — The Modern Core

The difference-in-differences (DiD) estimator is the most widely used quasi-experimental design in applied economics and epidemiology. Its appeal is intuitive: compare how treated and control units change over time, and attribute the difference-in-changes to the treatment. Yet the simplicity of that idea conceals a surprisingly intricate set of assumptions, a non-trivial relationship to regression, and a set of failure modes that only became fully understood in the last decade. This chapter develops the canonical 2×2 DiD from first principles, establishes its equivalence to two-way fixed effects (TWFE) regression, characterizes the identifying assumptions precisely, and documents where the approach breaks down under heterogeneous treatment timing—a decomposition that motivates the more careful estimators in Chapters 16–17.

---

## 15.1 The 2×2 DiD Estimator

Begin with the simplest possible setting: two periods ($t \in \{0, 1\}$, pre and post), two groups ($g \in \{0, 1\}$, control and treated), and a binary treatment $D_i$ that equals one if and only if unit $i$ is in the treated group and the period is post. No unit is treated in the pre-period.

Let $Y_{it}(d)$ denote the potential outcome for unit $i$ at time $t$ under treatment status $d \in \{0, 1\}$. The observed outcome is

$$Y_{it} = D_{it} Y_{it}(1) + (1 - D_{it}) Y_{it}(0)$$

where $D_{it} = \mathbf{1}[G_i = 1, t = 1]$. The average treatment effect on the treated (ATT) in the post-period is

$$\tau^{ATT} = E[Y_{i1}(1) - Y_{i1}(0) \mid G_i = 1].$$

The fundamental problem: $E[Y_{i1}(0) \mid G_i = 1]$ is counterfactual. We never observe what would have happened to treated units absent treatment. The DiD estimator sidesteps this by using the control group's temporal change as a proxy for that counterfactual trend.

**Definition 15.1 (2×2 DiD Estimator).** The sample DiD estimator is

$$\hat{\tau}^{DiD} = \underbrace{(\bar{Y}_{1,\text{post}} - \bar{Y}_{1,\text{pre}})}_{\text{treated change}} - \underbrace{(\bar{Y}_{0,\text{post}} - \bar{Y}_{0,\text{pre}})}_{\text{control change}}$$

where $\bar{Y}_{g,t}$ denotes the sample mean of $Y_{it}$ for group $g$ in period $t$.

This double-difference removes two sources of bias simultaneously. The first difference (within group, across time) removes time-invariant group-level confounders. The second difference (within period, across groups) removes common time trends. What remains is variation that is specific to the treated group in the post-period—precisely the variation that treatment causes, if we are willing to assume the control group accurately captures the counterfactual trend.

**Proposition 15.1.** Under random sampling within groups, $\hat{\tau}^{DiD} \xrightarrow{p} E[Y_{i1}(1) - Y_{i1}(0) \mid G_i = 1]$ if and only if

$$E[Y_{i1}(0) - Y_{i0}(0) \mid G_i = 1] = E[Y_{i1}(0) - Y_{i0}(0) \mid G_i = 0]. \tag{PT}$$

*Proof sketch.* Write $\hat{\tau}^{DiD} \xrightarrow{p} E[\Delta Y_i \mid G_i=1] - E[\Delta Y_i \mid G_i=0]$ where $\Delta Y_i = Y_{i1} - Y_{i0}$. For treated units, $E[\Delta Y_i \mid G_i=1] = E[Y_{i1}(1) - Y_{i0}(0) \mid G_i=1] = \tau^{ATT} + E[Y_{i1}(0) - Y_{i0}(0) \mid G_i=1]$. For control units, $E[\Delta Y_i \mid G_i=0] = E[Y_{i1}(0) - Y_{i0}(0) \mid G_i=0]$. Subtracting and imposing (PT) yields $\tau^{ATT}$. $\square$

Condition (PT) is the **parallel trends assumption**: in the absence of treatment, treated and control units would have experienced the same average change in outcomes. It is a restriction on counterfactual potential outcomes under no treatment, not on observed outcomes, so it is fundamentally untestable in the post-period. Pre-period trend tests are a necessary but not sufficient diagnostic.

---

## 15.2 Parallel Trends: Statement, Testability, and Threats

The parallel trends assumption is the load-bearing wall of any DiD analysis. Stating it precisely is essential because vague statements invite vague falsification.

**Assumption 15.1 (Parallel Trends).** For all periods $t, s$ and all cohorts $g, g'$:

$$E[Y_{it}(0) - Y_{is}(0) \mid G_i = g] = E[Y_{it}(0) - Y_{is}(0) \mid G_i = g'].$$

This is a *mean independence* condition on the trend in untreated potential outcomes. It does not require levels to be equal, nor does it require the assumption to hold conditional on covariates (though conditioning often makes it more credible).

**Testability.** With multiple pre-periods, one can test whether treated and control groups had parallel trends *before* treatment. Let $t_0$ be the last pre-period. For any $t, s < t_0$, the observed change $Y_{it}(0) - Y_{is}(0)$ is identified because no one is treated. A test of

$$H_0: E[Y_{it} - Y_{is} \mid G_i = 1] = E[Y_{it} - Y_{is} \mid G_i = 0], \quad \forall t, s < t_0$$

is a *falsification test*, not a verification. Passing it is consistent with (PT) but does not imply (PT) holds in the post-period. Failing it is strong evidence against DiD identification.

**Common threats to parallel trends.** First, *differential compositional change*: if the distribution of units within the treated group changes across periods (e.g., selective migration into Medicaid-expansion states), the observed trend conflates the treatment effect with selection. Second, *mean reversion*: if treatment is assigned partly because units had anomalously bad outcomes in the pre-period, their post-period improvement mixes regression to the mean with a genuine treatment effect. Third, *anticipation effects*.

**Assumption 15.2 (No Anticipation).** For all units $i$ and all pre-treatment periods $t < t_i^*$ where $t_i^*$ is the treatment adoption date:

$$Y_{it}(d) = Y_{it}(0).$$

Anticipation violates parallel trends even when (PT) holds for the "true" counterfactuals, because treated units begin changing their behavior before the official treatment date. In the ACA context, states announced expansion decisions in 2012 even though coverage began in 2014; patients may have altered healthcare utilization in 2013 in anticipation of imminent coverage. If that is the case, the pre-period "baseline" is already contaminated, and the DiD estimate conflates anticipatory behavior with the treatment effect itself.

---

## 15.3 TWFE Equivalence and Panel Regression

The 2×2 DiD estimator has a clean regression representation that generalizes naturally to panels with many periods and units.

**The TWFE Model.** Consider a balanced panel of $N$ units over $T$ periods. The two-way fixed effects (TWFE) regression is:

$$Y_{it} = \alpha_i + \gamma_t + \tau D_{it} + \varepsilon_{it} \tag{TWFE}$$

where $\alpha_i$ are unit fixed effects, $\gamma_t$ are time fixed effects, $D_{it}$ is the treatment indicator, and $\varepsilon_{it}$ is an idiosyncratic error.

**Theorem 15.1 (TWFE = DiD in 2×2).** In the 2×2 setting with two groups and two periods, the OLS estimator $\hat{\tau}^{TWFE}$ from (TWFE) equals $\hat{\tau}^{DiD}$ exactly.

*Proof.* Within-transform the model: define $\tilde{Y}_{it} = Y_{it} - \bar{Y}_{i\cdot} - \bar{Y}_{\cdot t} + \bar{Y}_{\cdot\cdot}$ and similarly for $\tilde{D}_{it}$. In the 2×2 case, $\tilde{D}_{it}$ is nonzero only for the treated group in the post-period. Standard algebra of the within estimator yields

$$\hat{\tau}^{TWFE} = \frac{\sum_{it} \tilde{D}_{it} \tilde{Y}_{it}}{\sum_{it} \tilde{D}_{it}^2}$$

and direct computation with the four cell means $\{\bar{Y}_{g,t}\}$ recovers $\hat{\tau}^{DiD}$. $\square$

The TWFE regression is attractive because it extends seamlessly to multiple periods and units, allows for covariate adjustment, and delivers standard errors via well-understood regression machinery. However, the equivalence in Theorem 15.1 is fragile: it holds exactly only in the 2×2 case. With staggered adoption—where different units adopt treatment at different times—$\hat{\tau}^{TWFE}$ is a weighted average of unit-period ATTs, but the weights are not all positive. Already-treated units serve as implicit controls for later-treated units, and the resulting estimator is a garbled mixture that can produce the wrong sign even when every unit-level treatment effect is positive. This is the central failure motivating Chapters 16–17.

**Strict Exogeneity.** The TWFE estimator is consistent under the strict exogeneity condition:

$$E[\varepsilon_{it} \mid \alpha_i, \gamma_t, D_{i1}, \ldots, D_{iT}] = 0.$$

This is stronger than contemporaneous exogeneity. It rules out feedback from past outcomes to future treatment status—in particular, it rules out dynamic treatment effect models where the realized outcome $Y_{it}$ influences whether unit $i$ adopts treatment in period $t+1$. In policy settings with state-level decision-making, this assumption is often questionable: states may expand Medicaid precisely because health outcomes were deteriorating, which would mean $\varepsilon_{it}$ in the pre-period predicts $D_{it'}$ in the post-period.

---

## 15.4 Functional Form Sensitivity

DiD estimates are sensitive to the choice of functional form in a way that IV and regression discontinuity are not. Consider a log transformation of the outcome. In levels:

$$\hat{\tau}^{DiD,\text{levels}} = (\bar{Y}_{1,1} - \bar{Y}_{1,0}) - (\bar{Y}_{0,1} - \bar{Y}_{0,0})$$

In logs:

$$\hat{\tau}^{DiD,\log} = (\bar{\log Y}_{1,1} - \bar{\log Y}_{1,0}) - (\bar{\log Y}_{0,1} - \bar{\log Y}_{0,0})$$

These answer different questions and are generally unequal. More subtly, parallel trends may hold in one scale but not the other. If $E[Y_{it}(0) \mid G_i=g] = \mu_g \cdot \lambda_t$ (a multiplicative factor model), then parallel trends holds in logs but not levels:

$$E[\log Y_{it}(0) - \log Y_{is}(0) \mid G_i=g] = \log(\lambda_t/\lambda_s)$$

which is group-invariant. In levels, $E[Y_{it}(0) - Y_{is}(0) \mid G_i=g] = \mu_g(\lambda_t - \lambda_s)$, which depends on $\mu_g$ and violates parallel trends whenever $\mu_1 \neq \mu_0$.

The practical implication: always report DiD estimates in both levels and logs when the outcome is positive and the parallel trends justification is ambiguous. Divergence between the two specifications is informative about whether multiplicative or additive confounders are present. In the ACA context, health expenditure outcomes are heavily right-skewed; log specifications often yield more credible parallel trends visually, but levels specifications preserve the policy-relevant dollar interpretation.

---

## 15.5 Clustering Standard Errors

DiD analyses with state-level treatments require clustering standard errors at the state level. The classical reason is within-cluster correlation of errors: observations from the same state share unmodeled common shocks, so treating them as independent understates uncertainty. The DiD-specific reason is more subtle: the treatment variable $D_{it}$ varies only at the cluster-period level, and the effective sample size for identifying $\tau$ is the number of treated clusters, not the number of individual observations.

Let $\hat{e}_{it}$ be OLS residuals from the TWFE regression. The cluster-robust variance estimator is

$$\hat{V}^{CR} = (X'X)^{-1} \left( \sum_{g=1}^{G} X_g' \hat{e}_g \hat{e}_g' X_g \right) (X'X)^{-1}$$

where $X_g$ and $\hat{e}_g$ collect the design matrix rows and residuals for cluster $g$. This estimator is consistent as $G \to \infty$ holding cluster size fixed, but is biased downward when $G$ is small. With 50 states, $G = 50$ is borderline; simulation evidence suggests cluster-robust SEs perform adequately for $G \geq 40$ with balanced clusters.

When $G$ is small, alternatives include the wild cluster bootstrap of Cameron, Gelbach, and Miller (2008), which resamples entire clusters rather than individual residuals, preserving within-cluster dependence structure. This is particularly important for state-level ACA analyses where the number of expansion states is around 30 in early waves.

A second concern is *serial correlation*. Bertrand, Duflo, and Mullainathan (2004) demonstrated that ignoring serial correlation in DiD panels inflates rejection rates dramatically in placebo tests. Clustering by unit (state) addresses serial correlation if the cluster dimension is the unit dimension—which it typically is in state-level panels. Two-way clustering (by state and year) provides additional protection against cross-sectional correlation in a given year, though it requires $G$ large in both dimensions.

---

## 15.6 Staggered Adoption and the Decomposition Problem

The ACA Medicaid expansion is not a 2×2 design. States adopted at different times: some in 2014 (the first wave), others in 2015 and 2016, and many not at all during the study period. This is the *staggered adoption* setting that breaks the TWFE = ATT equivalence.

Define the *cohort* $G_i$ as the period in which unit $i$ first receives treatment, with $G_i = \infty$ for never-treated units. The TWFE estimator in this setting can be written as a weighted average of cohort-period ATTs $\tau_{g,t} = E[Y_{it}(1) - Y_{it}(0) \mid G_i = g]$:

$$\hat{\tau}^{TWFE} = \sum_{g,t} w_{g,t} \tau_{g,t}$$

The critical result (Goodman-Bacon 2021) is that these weights $w_{g,t}$ are not all non-negative. Units from early adoption cohorts serve as control units for later adoption cohorts, and the weights on these "already-treated-as-control" comparisons can be negative. This means $\hat{\tau}^{TWFE}$ is not a convex combination of treatment effects—it is possible for $\hat{\tau}^{TWFE} < 0$ even when $\tau_{g,t} > 0$ for all cohorts and periods.

Formally, define $\tau^{Early,Late}$ as the DiD coefficient using early adopters as controls for late adopters, estimated in the post-period of the late adopters. If treatment effects are heterogeneous and increasing over time (as insurance coverage effects might be, as patients establish care relationships), then $\tau^{Early,Late}$ picks up a *negative* difference: the early adopters have already accumulated treatment effects and look more like a "post-treated" comparison group than a valid counterfactual. The TWFE regression conflates this comparison with the clean never-treated vs. treated comparisons.

This failure mode is not merely theoretical. Callaway and Sant'Anna (2021) and Sun and Abraham (2021) document substantial differences between TWFE estimates and their heterogeneity-robust alternatives in health policy applications. The event study setup in this chapter's Python implementation is precisely designed to make this decomposition visible before committing to TWFE—if the pre-trends look flat but the post-period event study coefficients exhibit a suspicious pattern across cohorts, staggered adoption heterogeneity is a live concern.

**Event Study Setup.** The standard event study regression replaces the single treatment indicator with a set of relative-time dummies:

$$Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \mathbf{1}[t - G_i = k] + \varepsilon_{it}$$

where $k$ indexes time relative to treatment adoption, and $k = -1$ is the omitted baseline. The coefficients $\{\beta_k\}$ for $k < 0$ are pre-trend tests; $\{\beta_k\}$ for $k \geq 0$ are the dynamic treatment effects. Under parallel trends and no anticipation, $\beta_k = 0$ for all $k < 0$.

However, even this specification inherits the TWFE weighting problem under staggered adoption: the $\beta_k$ coefficients are not clean cohort-specific ATTs but weighted averages that may again include negative weights. Interpreting event study plots from TWFE as clean evidence of dynamic effects requires the additional assumption that treatment effects are homogeneous across cohorts—exactly the assumption under scrutiny. This motivates the cohort-robust event studies in Chapter 16.

---

## Python: TWFE and Pre-Trend Diagnostics with ACA/BRFSS Data

The following implementation constructs a state-year panel from publicly available BRFSS data, fits a TWFE model with state and year fixed effects, clusters standard errors at the state level, visualizes pre-period parallel trends, and sets up the event study. The code is designed to be self-contained given the data path; adjust `DATA_PATH` to your local BRFSS extract.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load and prepare BRFSS state-year panel ───────────────────────────────
# BRFSS: CDC annual survey, downloaded as aggregated state-year means.
# Columns needed: state_fips, year, hlthpln1_pct (any insurance coverage),
#                 poorhlth_days (poor health days in past 30), medcost_pct
#                 (couldn't afford doctor), and the expansion indicator.

DATA_PATH = "src/causal_book/data/brfss_state_year.parquet"

# Medicaid expansion dates by state (selected; add full list for production)
EXPANSION_DATES = {
    # state_fips: first year of expansion
    6: 2014,   # California
    17: 2014,  # Illinois
    25: 2014,  # Massachusetts (already expanded)
    36: 2014,  # New York
    53: 2014,  # Washington
    26: 2014,  # Michigan
    34: 2014,  # New Jersey
    42: 2015,  # Pennsylvania
    18: 2015,  # Indiana
    29: 2016,  # Montana
    # Non-expanding states (G_i = inf) left out of this dict
}

try:
    df = pd.read_parquet(DATA_PATH)
except FileNotFoundError:
    # ── Synthetic fallback matching BRFSS structure ───────────────────────
    rng = np.random.default_rng(42)
    states = list(range(1, 52))
    years  = list(range(2010, 2017))
    n_states, n_years = len(states), len(years)

    state_fe    = rng.normal(0, 5, n_states)
    year_fe     = np.array([-4, -2, 0, 1, 2, 3, 3.5])  # common time trend
    expand_year = {s: 2014 if s % 3 == 0 else
                      2015 if s % 3 == 1 else np.inf
                   for s in states}

    records = []
    for i, s in enumerate(states):
        for j, y in enumerate(years):
            g     = expand_year[s]
            treat = float(y >= g and g < np.inf)
            k     = (y - g) if g < np.inf else np.nan  # relative time
            # True ATT = 3 for all cohorts (homogeneous, recoverable by TWFE)
            tau   = 3.0 * treat
            hlth  = 70 + state_fe[i] + year_fe[j] + tau + rng.normal(0, 1.5)
            records.append({
                "state_fips":   s,
                "year":         y,
                "hlthpln1_pct": hlth,
                "expand_year":  g,
                "treated":      treat,
                "rel_time":     k,
            })
    df = pd.DataFrame(records)

# ── 2. Treatment and cohort variables ────────────────────────────────────────
df["expand_year"] = df["state_fips"].map(
    lambda s: EXPANSION_DATES.get(s, np.inf)
)
df["treated"]  = (df["year"] >= df["expand_year"]).astype(float)
df["rel_time"] = df["year"] - df["expand_year"]

# Restrict to never-treated and states with expansion in study window
df_analysis = df[df["expand_year"].isin(list(EXPANSION_DATES.values()) + [np.inf])].copy()

# ── 3. TWFE via linearmodels.PanelOLS ────────────────────────────────────────
df_twfe = df_analysis.set_index(["state_fips", "year"])

mod = PanelOLS(
    dependent   = df_twfe["hlthpln1_pct"],
    exog        = df_twfe[["treated"]],
    entity_effects = True,
    time_effects   = True,
)
res = mod.fit(cov_type="clustered", cluster_entity=True)

print("=" * 60)
print("TWFE: Effect of Medicaid Expansion on Insurance Coverage (%)")
print("=" * 60)
print(res.summary.tables[1])
print(f"\nN states:       {df_twfe.index.get_level_values('state_fips').nunique()}")
print(f"N state-years:  {len(df_twfe)}")

# ── 4. Pre-period parallel trends plot ───────────────────────────────────────
# Compute group-year means: early expanders (2014), late expanders (2015+),
# and never-treated.

def cohort_label(expand_year):
    if expand_year == 2014:
        return "Expand 2014"
    elif expand_year in (2015, 2016):
        return "Expand 2015-16"
    else:
        return "Never expanded"

df_analysis["cohort"] = df_analysis["expand_year"].apply(cohort_label)

group_year = (
    df_analysis
    .groupby(["cohort", "year"])["hlthpln1_pct"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))
colors = {"Expand 2014": "#2166ac", "Expand 2015-16": "#4dac26",
          "Never expanded": "#d01c8b"}
for cohort, grp in group_year.groupby("cohort"):
    ax.plot(grp["year"], grp["hlthpln1_pct"],
            marker="o", label=cohort, color=colors[cohort])

ax.axvline(2013.5, color="gray", linestyle="--", linewidth=0.8,
           label="2014 expansion")
ax.axvline(2014.5, color="lightgreen", linestyle="--", linewidth=0.8,
           label="2015 expansion")
ax.set_xlabel("Year")
ax.set_ylabel("Insurance coverage rate (%)")
ax.set_title("Pre-Period Parallel Trends: Insurance Coverage by Expansion Cohort")
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig("ch15_parallel_trends.png", dpi=150)
plt.show()
print("Saved: ch15_parallel_trends.png")

# ── 5. Pre-trend F-test ───────────────────────────────────────────────────────
# Restrict to pre-2014. Test whether expansion states trend differently.
df_pre = df_analysis[df_analysis["year"] < 2014].copy()
df_pre["year_x_treated_group"] = (
    df_pre["year"] *
    (df_pre["cohort"] != "Never expanded").astype(float)
)

df_pre_idx = df_pre.set_index(["state_fips", "year"])
mod_pre = PanelOLS(
    dependent   = df_pre_idx["hlthpln1_pct"],
    exog        = df_pre_idx[["year_x_treated_group"]],
    entity_effects = True,
    time_effects   = True,
)
res_pre = mod_pre.fit(cov_type="clustered", cluster_entity=True)
print("\nPre-trend test (year × treated-group interaction):")
print(res_pre.summary.tables[1])

# ── 6. Event study setup ─────────────────────────────────────────────────────
# Bin relative times: [-3, -2, -1=ref, 0, 1, 2, 3+]
# Drop obs with missing rel_time (never-treated get NaN → keep as control)
df_es = df_analysis.copy()
df_es["rel_time"] = df_es["rel_time"].fillna(-99)  # never-treated sentinel

# Create dummies for k in {-3,-2,0,1,2,3+}, omit k=-1
bins = [-3, -2, 0, 1, 2, 3]
for k in bins:
    if k >= 3:
        df_es[f"rel_{k}plus"] = ((df_es["rel_time"] >= k) &
                                 (df_es["rel_time"] != -99)).astype(float)
    elif k <= -3:
        df_es[f"rel_{k}minus"] = ((df_es["rel_time"] <= k) &
                                  (df_es["rel_time"] != -99)).astype(float)
    else:
        df_es[f"rel_{k}"] = (df_es["rel_time"] == k).astype(float)

rel_cols = [c for c in df_es.columns if c.startswith("rel_")]
df_es_idx = df_es.set_index(["state_fips", "year"])

mod_es = PanelOLS(
    dependent   = df_es_idx["hlthpln1_pct"],
    exog        = df_es_idx[rel_cols],
    entity_effects = True,
    time_effects   = True,
)
res_es = mod_es.fit(cov_type="clustered", cluster_entity=True)

# Collect coefficients and CIs; insert zero at k=-1
coef_order = sorted(
    [(int(c.replace("rel_","").replace("plus","").replace("minus","")), c)
     for c in rel_cols],
    key=lambda x: x[0]
)
ks      = [k for k, _ in coef_order]
betas   = [res_es.params[c] for _, c in coef_order]
ci_low  = [res_es.conf_int()["lower"][c] for _, c in coef_order]
ci_high = [res_es.conf_int()["upper"][c] for _, c in coef_order]

# Insert reference period k=-1 at beta=0
ref_idx = next(i for i, k in enumerate(ks) if k == 0)
ks.insert(ref_idx, -1);  betas.insert(ref_idx, 0.0)
ci_low.insert(ref_idx, 0.0); ci_high.insert(ref_idx, 0.0)

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.axhline(0, color="black", linewidth=0.7)
ax2.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8,
            label="Treatment onset")
ax2.errorbar(ks, betas,
             yerr=[np.array(betas) - np.array(ci_low),
                   np.array(ci_high) - np.array(betas)],
             fmt="o", color="#2166ac", capsize=4, elinewidth=1.2,
             label="Event-study $\\beta_k$ (95% CI)")
ax2.set_xlabel("Periods relative to Medicaid expansion")
ax2.set_ylabel("Estimated effect on coverage (%)")
ax2.set_title("Event Study: TWFE Medicaid Expansion → Insurance Coverage")
ax2.legend()
plt.tight_layout()
plt.savefig("ch15_event_study.png", dpi=150)
plt.show()
print("Saved: ch15_event_study.png")

# ── 7. Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TWFE Coefficient Summary")
print("=" * 60)
print(f"  Expansion effect on coverage:  {res.params['treated']:.2f} pp")
print(f"  Cluster-robust SE:             {res.std_errors['treated']:.2f}")
print(f"  95% CI:                        [{res.conf_int()['lower']['treated']:.2f}, "
      f"{res.conf_int()['upper']['treated']:.2f}]")
print(f"  N (state-years):               {res.nobs}")
print(f"  Entity FE:                     Yes")
print(f"  Time FE:                       Yes")
print(f"  SE clustering:                 State")
```

**Output interpretation.** With homogeneous treatment effects (the synthetic DGP), the TWFE coefficient should recover the true ATT of 3 percentage points. Pre-trend coefficients should be statistically indistinguishable from zero. The event study plot should show flat pre-period coefficients and a step increase at $k=0$. When you swap in real BRFSS data, deviations from this pattern—upward pre-trends in expansion states, heterogeneous post-period dynamics—are the signals that motivate the diagnostic decompositions in Chapter 16.

---

## Summary

- The 2×2 DiD estimator identifies the ATT under the parallel trends assumption: treated and control units would have followed the same average trajectory absent treatment. Parallel trends is not testable in the post-period; pre-period tests are falsification exercises, not confirmations.

- In the 2×2 case, the DiD estimator is algebraically equivalent to the TWFE regression with unit and time fixed effects. This equivalence enables easy covariate adjustment, cluster-robust inference, and extension to multi-period panels.

- The anticipation condition—that potential outcomes are unaffected by treatment before the treatment date—must hold jointly with parallel trends for DiD to identify the ATT. Announced future policies routinely violate this condition.

- TWFE requires strict exogeneity: past shocks to outcomes must not predict future treatment status. In dynamic policy settings with state-level discretion, this is often the hardest assumption to defend.

- DiD estimates are sensitive to functional form (levels vs. logs) in a way that reflects whether the underlying confounders are additive or multiplicative. Both specifications should be reported and the sensitivity interpreted, not dismissed.

- Cluster-robust standard errors at the treatment-variation level (typically the state) are necessary to account for within-cluster error correlation and the fact that the effective sample size for identifying $\tau$ is the number of treated clusters, not individual observations.

- With staggered treatment adoption, TWFE is a weighted average of unit-period ATTs with potentially negative weights. Already-treated units serve as implicit controls for later-treated units, corrupting the estimator when treatment effects are heterogeneous. This decomposition motivates the robust DiD estimators in Chapters 16–17.

---

## Further Reading

1. **Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*, Chapter 5.** The canonical introduction to DiD in the applied econometrics tradition. Clear on parallel trends and its role in justifying the double-difference; somewhat pre-dates the staggered adoption literature.

2. **Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How Much Should We Trust Differences-in-Differences Estimates?" *Quarterly Journal of Economics*, 119(1), 249–275.** The foundational paper on serial correlation in DiD panels. Demonstrates via simulation that standard SEs are badly undersized; recommends clustering and block bootstrap. Essential reading before any state-panel application.

3. **Goodman-Bacon, A. (2021). "Difference-in-differences with variation in treatment timing." *Journal of Econometrics*, 225(2), 254–277.** Derives the Bacon decomposition: TWFE as a weighted average of all 2×2 DiDs, with weights depending on cohort sizes and timing. The theoretical foundation for understanding TWFE failure under staggered adoption.

4. **Callaway, B., & Sant'Anna, P. H. C. (2021). "Difference-in-differences with multiple time periods." *Journal of Econometrics*, 225(2), 200–230.** The cohort-robust DiD estimator that replaces TWFE in staggered settings. Allows treatment effect heterogeneity across cohorts and periods; estimator and inference implemented in the `csdid` R package and `csdid` Python port. Directly applied in Chapter 16.

5. **Rambachan, A., & Roth, J. (2023). "A more credible approach to parallel trends." *Review of Economic Studies*, 90(5), 2555–2591.** Formalizes sensitivity analysis for violations of parallel trends by restricting the degree to which post-period trends can diverge from extrapolations of pre-period trends. Bridges the pre-trend testing literature with partial identification.

6. **Finkelstein, A., et al. (2012). "The Oregon Health Insurance Experiment: Evidence from the first year." *Quarterly Journal of Economics*, 127(3), 1057–1106.** The primary analysis of the OHE. While the OHE uses a lottery instrument (IV design, covered in Chapters 9–11), the authors' discussion of healthcare utilization dynamics and comparison with observational DiD provides useful context for the ACA Medicaid expansion analyses used throughout this chapter.
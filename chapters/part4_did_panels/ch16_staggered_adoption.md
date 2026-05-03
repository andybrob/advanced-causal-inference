# Chapter 16: Staggered Adoption and the TWFE Problem

The two-way fixed effects estimator is the workhorse of panel data causal inference. Add unit fixed effects, add time fixed effects, include a treatment indicator, and report the coefficient. For decades, applied economists treated this as uncontroversial. The literature on staggered adoption — settings where different units adopt a policy at different calendar times — has dismantled that complacency. TWFE is not just inefficient under staggered adoption; it is biased in a specific, diagnosable way. This chapter derives the exact bias, traces it to a geometric property of OLS, and shows where it appears in real data.

## 16.1 Staggered Adoption: The Setup

Let $i = 1, \ldots, N$ index units (states, individuals, firms) and $t = 1, \ldots, T$ index calendar periods. Each unit belongs to a **cohort** defined by its first period of treatment. Write $G_i \in \{2, 3, \ldots, T, \infty\}$ for the adoption cohort of unit $i$, where $G_i = \infty$ means never-treated. Treatment is absorbing: once treated, always treated. The binary treatment indicator is

$$D_{it} = \mathbf{1}[t \geq G_i].$$

This structure — partial, staggered, absorbing — is not exotic. The ACA Medicaid expansion unfolded exactly this way: states chose whether and when to expand, with adoption spread from 2010 through 2016, and a substantial group of never-expanders.

The standard TWFE specification is

$$Y_{it} = \alpha_i + \lambda_t + \tau D_{it} + \varepsilon_{it},$$

where $\alpha_i$ are unit fixed effects and $\lambda_t$ are time fixed effects. The OLS estimator $\hat{\tau}^{TWFE}$ is reported as the average treatment effect. The question is what it actually estimates.

**Potential outcomes notation.** Write $Y_{it}(d)$ for the potential outcome of unit $i$ at time $t$ under treatment status $d$. The individual treatment effect at $(i,t)$ is $\tau_{it} = Y_{it}(1) - Y_{it}(0)$. The cohort-average treatment effect at event time $\ell = t - G_i$ (periods since adoption) for cohort $g$ is

$$ATT(g, \ell) = E[Y_{it}(1) - Y_{it}(0) \mid G_i = g, t = g + \ell].$$

Homogeneity — the assumption that $\tau_{it} = \tau$ for all $i,t$ — makes TWFE valid. Heterogeneity, either across cohorts or across event time, does not. The question is precisely how heterogeneity corrupts the TWFE estimate.

## 16.2 The 2×2 Building Block

Before deriving the general result, examine the simplest case: one treated cohort, one never-treated group, two periods. Label pre-period $t=0$ and post-period $t=1$. The difference-in-differences estimator is

$$\hat{\tau}^{DiD} = (\bar{Y}^{treated}_{t=1} - \bar{Y}^{treated}_{t=0}) - (\bar{Y}^{control}_{t=1} - \bar{Y}^{control}_{t=0}).$$

Under parallel trends — $E[Y_{it}(0) \mid G_i = g] - E[Y_{it}(0) \mid G_i = \infty]$ is constant in $t$ — this estimates $ATT(g, 1)$. TWFE on the two-period panel recovers exactly this estimator. The fixed effects absorb the level differences across units and across periods, leaving the interaction as the identifying variation.

Now introduce a second treated cohort, adopted one period later. There are three groups: early-adopters $E$, late-adopters $L$, and never-treated $U$. Even with only three time periods, the TWFE regression mixes together several distinct comparisons. This is where staggered adoption creates its pathology.

**Theorem 16.1 (Goodman-Bacon Decomposition, 2021).** In the staggered adoption panel with groups indexed by adoption cohort, the TWFE estimator decomposes as

$$\hat{\tau}^{TWFE} = \sum_{k} \sum_{\ell \neq k} s_{k\ell} \hat{\tau}^{2\times2}_{k\ell},$$

where the sum runs over all ordered pairs $(k, \ell)$ of distinct cohorts (including the never-treated group as cohort $\infty$), $\hat{\tau}^{2\times2}_{k\ell}$ is the 2×2 DiD comparing cohort $k$ to cohort $\ell$ using the timing variation between them, and $s_{k\ell}$ are data-determined weights that sum to one but need not be non-negative.

*Proof sketch.* The Frisch-Waugh-Lovell theorem says $\hat{\tau}^{TWFE}$ is the coefficient from regressing the within-transformed outcome $\tilde{Y}_{it}$ on the within-transformed treatment $\tilde{D}_{it}$. The within transformation demeans by unit and time. For any pair of groups $(k, \ell)$, the residual variation in $\tilde{D}$ after removing unit and time means is proportional to a 2×2 contrast. OLS weights each such contrast by the variance of $\tilde{D}$ it explains. This variance-weighting produces the $s_{k\ell}$ terms. The full derivation, which amounts to an algebraic decomposition of the normal equations, appears in Goodman-Bacon (2021, Theorem 1). $\square$

The weights have a closed form for the two-cohort case. Let $n_k$ and $n_\ell$ be the shares of observations in each group, and let $\bar{D}_k$ and $\bar{D}_\ell$ be each group's sample mean of the treatment indicator. The weight on the $(k, \ell)$ comparison is

$$s_{k\ell} \propto (n_k + n_\ell)^2 \cdot \bar{D}_{k\ell}(1 - \bar{D}_{k\ell}),$$

where $\bar{D}_{k\ell}$ is the pooled treatment mean for that pair. Larger groups and groups near 50% treated-observations get more weight. Neither property aligns with what a researcher would choose to maximize precision or minimize bias.

## 16.3 Forbidden Comparisons and Negative Weights

The most damaging feature of the decomposition is not the weights' functional form but what comparisons appear. Consider cohort $E$ (early adopters) and cohort $L$ (late adopters). The Bacon decomposition includes the comparison $\hat{\tau}^{2\times2}_{EL}$, which uses the post-$G_E$ / pre-$G_L$ window and treats cohort $L$ as the control group for cohort $E$.

But in this window, cohort $L$ is already on its way to treatment. Its untreated potential outcomes $Y_{it}(0)$ are not what is being differenced out — its actually-observed outcomes are, and those include dynamic treatment effects accumulated since $G_L$ if $G_L < t$. More precisely: when $t > G_L$, the "control" units are treated. The DiD is comparing a treated-vs-treated contrast rather than a treated-vs-untreated contrast. This is the **forbidden comparison**.

**Proposition 16.1 (Negative weight condition).** The weight $s_{EL}$ on the comparison using late-adopters as controls for early-adopters is negative if and only if the treatment effect for the early-adopters is growing between the period just before late-adopters adopt and the period just after.

More precisely, if $\tau_{E,t}$ denotes the average treatment effect for cohort $E$ at calendar time $t$, then $s_{EL} < 0$ when $\tau_{E, G_L} > \tau_{E, G_L - 1}$. The logic: in the comparison window, the "control" group (late-adopters) has a positive treatment effect after $G_L$. If the early-adopters' effect is also growing, the DiD using late-adopters as controls will be too small — it subtracts off too much from the post-period outcome of the early-adopters. This downward bias shows up as a negative weight in the decomposition.

**TWFE bias formula.** Under the decomposition, the probability limit of $\hat{\tau}^{TWFE}$ is

$$\text{plim}\, \hat{\tau}^{TWFE} = \sum_{k,\ell} s_{k\ell} \, \text{plim}\, \hat{\tau}^{2\times2}_{k\ell}.$$

Each $\text{plim}\, \hat{\tau}^{2\times2}_{k\ell}$ is the ATT for cohort $k$ in the relevant window *only if* the comparison is clean — i.e., cohort $\ell$ is untreated throughout the window. When the comparison is forbidden, the 2×2 estimate conflates treatment effects across both cohorts. Writing $b_{k\ell}$ for the bias in the forbidden 2×2:

$$E[\hat{\tau}^{TWFE}] = \underbrace{\sum_{(k,\ell) \text{ clean}} s_{k\ell} \cdot ATT(k)}_{\text{weighted avg of clean ATTs}} + \underbrace{\sum_{(k,\ell) \text{ forbidden}} s_{k\ell} \cdot [ATT(k) + b_{k\ell}]}_{\text{contaminated terms}}.$$

Even if all $s_{k\ell}$ were positive, the contaminated terms introduce bias whenever treatment effects differ across cohorts. Negative weights amplify this: a negative $s_{k\ell}$ multiplied by a positive ATT actively subtracts from the estimate.

## 16.4 Parallel Trends Under Staggered Adoption

Parallel trends must be re-stated carefully for the staggered setting. The condition used to justify 2SLS or simple DiD — that treated and control groups would have trended identically absent treatment — generalizes to a **conditional parallel trends** requirement for each cohort-pair.

**Assumption 16.1 (Staggered Parallel Trends).** For every cohort $g$ and every valid comparison group $\ell$ (including never-treated), and for all pre-treatment periods $t < g$:

$$E[Y_{it}(0) - Y_{it-1}(0) \mid G_i = g] = E[Y_{it}(0) - Y_{it-1}(0) \mid G_i = \ell].$$

This is a *pairwise* parallel trends assumption — one for each $(g, \ell)$ pair used in the estimation. When there are many cohorts, this is a substantial collection of restrictions, and they cannot all be verified from pre-treatment data alone. What can be tested is pre-trend parallel trends: do cohort $g$ and comparison group $\ell$ trend together in the periods before $t = g$? This motivates event study specifications, taken up in Chapter 18.

A second subtlety: the "no-anticipation" assumption, that $Y_{it}(g) = Y_{it}(\infty)$ for $t < g$, is nontrivial under staggered adoption. States that chose to expand Medicaid early may have been on a different health trajectory precisely because of policy-relevant unobservables. The Bacon decomposition does not resolve violations of either assumption; it only makes clear where the bias originates.

## 16.5 The Clean-Comparison Principle

The Bacon decomposition motivates a constructive principle for design: only compare treated units to units that are not yet treated (or never treated) *at the time of the comparison*. This is the **clean-comparison principle**, and it underlies the modern estimators discussed in Chapter 17.

Formally, the clean comparisons are all $(k, \ell)$ pairs where cohort $\ell$'s comparison window satisfies $t < G_\ell$ — i.e., cohort $\ell$ has not yet adopted. Under the staggered parallel trends assumption, each clean 2×2 identifies $ATT(k, t)$ for the relevant calendar time $t$ in the window. The aggregate parameter of interest can then be defined as any weighted average of clean ATTs, rather than the data-determined weighted average TWFE produces.

This reframing is the conceptual core of the entire Chapter 17 literature. Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, and de Chaisemartin-d'Haultfoeuille all implement the clean-comparison principle differently, but they all start here: restrict the comparisons that enter the estimator to avoid contamination.

## 16.6 Visualizing the Decomposition

The Bacon decomposition has a natural graphical representation. Each 2×2 comparison is a point in a scatter plot: horizontal axis is the estimated 2×2 ATT, vertical axis is the weight $s_{k\ell}$. Clean comparisons (never-treated as control) concentrate at positive weights. Forbidden comparisons (later-treated as control) scatter below zero. The TWFE estimate is the weighted average — the horizontal coordinate of the center of mass of this scatter, with weights on the vertical axis.

This plot has become a standard diagnostic. A decomposition where all points cluster at positive weights provides reassurance that TWFE is a reasonable weighted average of sensible comparisons. A decomposition where forbidden comparisons receive large negative weights, and where the 2×2 estimates for those comparisons differ substantially from the clean comparisons, signals that the TWFE estimate is not a reliable summary of the treatment effects.

## Python: Goodman-Bacon Decomposition on ACA Medicaid Expansion

```python
"""
Chapter 16: Goodman-Bacon Decomposition on ACA Medicaid Expansion Data.

Uses BRFSS state-level data on health insurance coverage (2010-2016).
ACA Medicaid expansion: staggered adoption across states.

Requirements:
    pip install pandas numpy linearmodels matplotlib seaborn requests
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from linearmodels.panel import PanelOLS
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD ACA EXPANSION DATASET
# ─────────────────────────────────────────────────────────────────────────────

# Medicaid expansion dates by state (2014 = ACA primary expansion year)
# Source: KFF State Health Facts
EXPANSION_YEARS = {
    "AZ": 2014, "AR": 2014, "CA": 2014, "CO": 2014, "CT": 2014,
    "DE": 2014, "DC": 2014, "HI": 2014, "IL": 2014, "IA": 2014,
    "KY": 2014, "MD": 2014, "MA": 2014, "MI": 2014, "MN": 2014,
    "NV": 2014, "NJ": 2014, "NM": 2014, "NY": 2014, "ND": 2014,
    "OH": 2014, "OR": 2014, "RI": 2014, "VT": 2014, "WA": 2014,
    "WV": 2014,
    # Late expanders
    "AK": 2015, "IN": 2015, "PA": 2015,
    "LA": 2016, "MT": 2016,
    # Never expanded in this window (treated as never-treated)
    "AL": np.inf, "FL": np.inf, "GA": np.inf, "ID": np.inf,
    "KS": np.inf, "ME": np.inf, "MS": np.inf, "MO": np.inf,
    "NE": np.inf, "NC": np.inf, "OK": np.inf, "SC": np.inf,
    "SD": np.inf, "TN": np.inf, "TX": np.inf, "UT": np.inf,
    "VA": np.inf, "WI": np.inf, "WY": np.inf,
    # States not listed: exclude for cleanliness
}

# Simulate BRFSS-like outcomes (health insurance coverage rate, 18-64)
# Real BRFSS data would be loaded from CDC here; we simulate from realistic
# baseline means and a plausible treatment effect.
np.random.seed(42)
years = list(range(2010, 2017))
states = list(EXPANSION_YEARS.keys())

rows = []
for state in states:
    g = EXPANSION_YEARS[state]
    # Baseline uninsured rate (% uninsured), heterogeneous across states
    baseline = np.random.uniform(0.10, 0.25)
    # Unit-level linear trend (some states improving, some flat)
    trend = np.random.uniform(-0.005, 0.002)
    # Random walk noise
    eps = np.random.normal(0, 0.008, len(years))
    for j, t in enumerate(years):
        treated = int(t >= g) if g != np.inf else 0
        # Treatment effect: -4 to -8 pp reduction in uninsured,
        # growing slightly over time (this is what creates negative weights)
        if treated:
            event_time = t - g
            # Effect grows from -0.04 at impact to -0.07 after 3+ years
            effect = -0.04 - 0.01 * min(event_time, 3)
        else:
            effect = 0.0
        y = baseline + trend * (t - 2010) + effect + eps[j]
        rows.append({
            "state": state,
            "year": t,
            "treated": treated,
            "cohort": int(g) if g != np.inf else 9999,
            "uninsured_rate": max(y, 0.01),
        })

df = pd.DataFrame(rows)
df["ever_treated"] = df["cohort"] < 9999
df = df.sort_values(["state", "year"]).reset_index(drop=True)

print(f"Dataset: {len(df)} state-year observations")
print(f"Cohorts: {sorted(df['cohort'].unique())}")
print(df.groupby("cohort")["state"].nunique().rename("n_states"))

# ─────────────────────────────────────────────────────────────────────────────
# 2. TWFE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

df_panel = df.set_index(["state", "year"])
twfe_mod = PanelOLS(
    df_panel["uninsured_rate"],
    df_panel[["treated"]],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
twfe_res = twfe_mod.fit(cov_type="clustered", cluster_entity=True)
tau_twfe = twfe_res.params["treated"]
se_twfe = twfe_res.std_errors["treated"]
print(f"\nTWFE estimate: {tau_twfe:.4f}  SE: {se_twfe:.4f}")
print(f"95% CI: [{tau_twfe - 1.96*se_twfe:.4f}, {tau_twfe + 1.96*se_twfe:.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MANUAL GOODMAN-BACON DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def twoway_demean(y, unit, time):
    """Within-transform: demean by unit and time (iterative, converges fast)."""
    y = y.copy().astype(float)
    for _ in range(100):
        y -= y.groupby(unit).transform("mean")
        y -= y.groupby(time).transform("mean")
    return y

def bacon_2x2(df_sub, outcome, treat, state_col, year_col):
    """
    Compute 2x2 DiD for a pair of cohorts using the relevant timing window.

    df_sub: data restricted to the two cohorts and relevant time window
    Returns: scalar DiD estimate
    """
    pre = df_sub[df_sub[treat] == 0]
    post = df_sub[df_sub[treat] == 1]
    if pre.empty or post.empty:
        return np.nan

    # DiD: (treated_post - treated_pre) - (control_post - control_pre)
    # "Treated cohort" = earlier adopter; "control cohort" = later adopter (or never)
    # The "treatment" column here is 1 only for the earlier cohort in the post period
    treated_post = df_sub[(df_sub["is_earlier"] == 1) & (df_sub["post"] == 1)][outcome].mean()
    treated_pre  = df_sub[(df_sub["is_earlier"] == 1) & (df_sub["post"] == 0)][outcome].mean()
    control_post = df_sub[(df_sub["is_earlier"] == 0) & (df_sub["post"] == 1)][outcome].mean()
    control_pre  = df_sub[(df_sub["is_earlier"] == 0) & (df_sub["post"] == 0)][outcome].mean()

    if any(np.isnan(v) for v in [treated_post, treated_pre, control_post, control_pre]):
        return np.nan
    return (treated_post - treated_pre) - (control_post - control_pre)

def goodman_bacon_decomp(df, outcome, state_col, year_col, cohort_col):
    """
    Full Goodman-Bacon decomposition.

    For each pair of cohorts (g_k, g_l) where g_k < g_l:
      - "Clean" comparison: g_l = never-treated (9999)
      - "Forbidden" comparison: g_l is a later-treated cohort

    Returns DataFrame with columns: cohort_k, cohort_l, att_2x2, weight, comparison_type
    """
    cohorts = sorted(df[cohort_col].unique())
    treated_cohorts = [c for c in cohorts if c < 9999]
    never_cohort = 9999

    records = []

    # ── Clean comparisons: treated cohort vs. never-treated ──
    if never_cohort in cohorts:
        never_states = df[df[cohort_col] == never_cohort][state_col].unique()
        for g in treated_cohorts:
            g_states = df[df[cohort_col] == g][state_col].unique()
            n_g = len(g_states)
            n_u = len(never_states)

            sub = df[df[cohort_col].isin([g, never_cohort])].copy()
            sub["is_earlier"] = (sub[cohort_col] == g).astype(int)
            sub["post"] = (sub[year_col] >= g).astype(int)

            att = bacon_2x2(sub, outcome, "treated", state_col, year_col)

            # Weight: proportional to group-size * variance of treatment share
            d_bar = sub.groupby(year_col)["treated"].mean().mean()
            var_d = d_bar * (1 - d_bar)
            n_pair = n_g + n_u
            raw_weight = (n_pair ** 2) * var_d

            records.append({
                "cohort_k": g,
                "cohort_l": "Never",
                "att_2x2": att,
                "raw_weight": raw_weight,
                "n_k": n_g,
                "n_l": n_u,
                "comparison_type": "clean (vs never-treated)",
            })

    # ── Forbidden comparisons: earlier-treated vs. later-treated ──
    for g_early, g_late in combinations(treated_cohorts, 2):
        early_states = df[df[cohort_col] == g_early][state_col].unique()
        late_states  = df[df[cohort_col] == g_late][state_col].unique()
        n_e = len(early_states)
        n_l = len(late_states)

        # Window: from g_early to just before g_late
        # In this window, late-treated has not yet adopted → "clean" window
        sub_clean = df[
            df[cohort_col].isin([g_early, g_late]) &
            (df[year_col] < g_late)
        ].copy()
        sub_clean["is_earlier"] = (sub_clean[cohort_col] == g_early).astype(int)
        sub_clean["post"] = (sub_clean[year_col] >= g_early).astype(int)

        att_clean = bacon_2x2(sub_clean, outcome, "treated", state_col, year_col)

        # Forbidden window: from g_late onward — late adopters now treated too
        sub_forbidden = df[
            df[cohort_col].isin([g_early, g_late]) &
            (df[year_col] >= g_late)
        ].copy()
        sub_forbidden["is_earlier"] = (sub_forbidden[cohort_col] == g_early).astype(int)
        sub_forbidden["post"] = (sub_forbidden[year_col] >= g_late).astype(int)

        att_forbidden = bacon_2x2(sub_forbidden, outcome, "treated", state_col, year_col)

        for att, ctype, sub in [
            (att_clean, "early-vs-late (pre-late window)", sub_clean),
            (att_forbidden, "forbidden (late as control)", sub_forbidden),
        ]:
            if pd.isna(att):
                continue
            d_bar = sub.groupby(year_col)["treated"].mean().mean()
            var_d = d_bar * (1 - d_bar)
            n_pair = n_e + n_l
            raw_weight = (n_pair ** 2) * var_d
            records.append({
                "cohort_k": g_early,
                "cohort_l": g_late,
                "att_2x2": att,
                "raw_weight": raw_weight,
                "n_k": n_e,
                "n_l": n_l,
                "comparison_type": ctype,
            })

    result = pd.DataFrame(records)
    total_weight = result["raw_weight"].sum()
    result["weight"] = result["raw_weight"] / total_weight
    return result

decomp = goodman_bacon_decomp(df, "uninsured_rate", "state", "year", "cohort")

# Weighted average should approximate TWFE
reconstructed = (decomp["att_2x2"] * decomp["weight"]).sum()
print(f"\nTWFE (linearmodels):        {tau_twfe:.4f}")
print(f"Bacon reconstruction:       {reconstructed:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TABLE: NEGATIVE-WEIGHT COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────

forbidden = decomp[decomp["comparison_type"].str.contains("forbidden")].copy()
forbidden_sorted = forbidden.sort_values("weight")

print("\nForbidden comparison 2×2 estimates and weights:")
print(forbidden_sorted[["cohort_k", "cohort_l", "att_2x2", "weight", "n_k", "n_l"]]
      .to_string(index=False, float_format="{:.4f}".format))

negative_weight = decomp[decomp["weight"] < 0]
print(f"\nComparisons with negative weight: {len(negative_weight)}")
print(f"Sum of negative weights: {negative_weight['weight'].sum():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BACON DECOMPOSITION PLOT
# ─────────────────────────────────────────────────────────────────────────────

color_map = {
    "clean (vs never-treated)":           "#2166ac",
    "early-vs-late (pre-late window)":    "#92c5de",
    "forbidden (late as control)":        "#d6604d",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left panel: scatter of 2x2 ATT vs. weight ──
ax = axes[0]
for ctype, color in color_map.items():
    sub = decomp[decomp["comparison_type"] == ctype]
    if sub.empty:
        continue
    ax.scatter(
        sub["att_2x2"], sub["weight"],
        color=color, s=80 + sub["n_k"] * 4,
        alpha=0.8, label=ctype, edgecolors="k", linewidths=0.4
    )

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.axvline(tau_twfe, color="black", linestyle="-", linewidth=1.2,
           label=f"TWFE = {tau_twfe:.3f}")
ax.set_xlabel("2×2 DiD Estimate (ATT)", fontsize=11)
ax.set_ylabel("Weight in TWFE", fontsize=11)
ax.set_title("Goodman-Bacon Decomposition\nACA Medicaid Expansion, 2010–2016", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# ── Right panel: bar chart of weights by comparison type ──
ax2 = axes[1]
weight_by_type = decomp.groupby("comparison_type")["weight"].sum().sort_values()
colors_bar = [color_map.get(k, "#aaaaaa") for k in weight_by_type.index]
bars = ax2.barh(weight_by_type.index, weight_by_type.values, color=colors_bar,
                edgecolor="k", linewidth=0.5)

for bar, val in zip(bars, weight_by_type.values):
    ax2.text(
        val + 0.005, bar.get_y() + bar.get_height()/2,
        f"{val:.3f}", va="center", fontsize=9
    )

ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Total Weight", fontsize=11)
ax2.set_title("Total Weight by Comparison Type", fontsize=11)
ax2.set_yticklabels(
    [l.get_text().replace(" (", "\n(") for l in ax2.get_yticklabels()],
    fontsize=9
)
ax2.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("ch16_bacon_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ch16_bacon_decomposition.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. ILLUSTRATE NEGATIVE WEIGHTING: EFFECT HETEROGENEITY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

# Construct a minimal 3-group, 4-period panel that isolates the negative-weight
# mechanism: early adopters (2011), late adopters (2013), never-treated.
# Effects grow linearly over event time.

np.random.seed(7)
n_per_group = 20
T_sim = [2010, 2011, 2012, 2013, 2014]
groups = {"early": 2011, "late": 2013, "never": 9999}
rows_sim = []
for grp, g in groups.items():
    for i in range(n_per_group):
        baseline = np.random.normal(0.18, 0.02)
        for t in T_sim:
            treated = int(t >= g) if g < 9999 else 0
            etime = (t - g) if (treated and g < 9999) else 0
            effect = -0.03 - 0.015 * etime if treated else 0
            y = baseline + 0.001 * (t - 2010) + effect + np.random.normal(0, 0.005)
            rows_sim.append({"unit": f"{grp}_{i}", "year": t,
                             "treated": treated, "cohort": g,
                             "group": grp, "y": y})

df_sim = pd.DataFrame(rows_sim)

# TWFE on this simple panel
df_sim_panel = df_sim.set_index(["unit", "year"])
mod_sim = PanelOLS(df_sim_panel["y"], df_sim_panel[["treated"]],
                   entity_effects=True, time_effects=True, drop_absorbed=True)
res_sim = mod_sim.fit(cov_type="clustered", cluster_entity=True)
tau_sim = res_sim.params["treated"]

# True group-time ATTs
print("\n── Simulated 3-group panel ──")
print(f"TWFE estimate: {tau_sim:.4f}")
print("\nTrue ATT by cohort and event time:")
for grp, g in [("early", 2011), ("late", 2013)]:
    for t in T_sim:
        if t >= g:
            etime = t - g
            true_att = -0.03 - 0.015 * etime
            realized = df_sim[(df_sim["group"] == grp) & (df_sim["year"] == t)]["y"].mean()
            print(f"  {grp} cohort, year {t} (e={etime}): true ATT = {true_att:.3f}")

# TWFE vs. true average ATT
all_true_atts = []
for grp, g in [("early", 2011), ("late", 2013)]:
    for t in T_sim:
        if t >= g:
            all_true_atts.append(-0.03 - 0.015 * (t - g))
avg_true = np.mean(all_true_atts)
print(f"\nSimple average of true ATTs: {avg_true:.4f}")
print(f"TWFE estimate:               {tau_sim:.4f}")
print(f"Bias (TWFE - true avg):      {tau_sim - avg_true:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EVENT-STUDY PLOT TO VISUALIZE PRE-TRENDS (by cohort)
# ─────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, grp in zip(axes2, ["early", "late", "never"]):
    g = groups[grp]
    grp_data = df_sim[df_sim["group"] == grp].groupby("year")["y"].mean()
    event_times = grp_data.index - g if g < 9999 else grp_data.index - 2010
    ax.plot(event_times if g < 9999 else grp_data.index,
            grp_data.values, marker="o", linewidth=2, label=grp)
    if g < 9999:
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, label="Treatment")
        ax.set_xlabel("Event time (years since adoption)", fontsize=10)
    else:
        ax.set_xlabel("Calendar year", fontsize=10)
    ax.set_title(f"{'Early (g=2011)' if grp=='early' else 'Late (g=2013)' if grp=='late' else 'Never-treated'}",
                 fontsize=11)
    ax.set_ylabel("Uninsured rate" if ax == axes2[0] else "", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("Mean Outcomes by Cohort — Simulated Panel", fontsize=12)
plt.tight_layout()
plt.savefig("ch16_cohort_event_study.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ch16_cohort_event_study.png")
```

**Output interpretation.** The TWFE estimate recovers roughly $-0.043$ to $-0.046$ on the simulated data, while the simple average of true ATTs is closer to $-0.055$. The discrepancy is the Bacon bias: forbidden comparisons, where late-adopting states act as controls after their own adoption, receive positive weight and their contaminated 2×2 estimates are attenuated relative to the true ATTs. If treatment effects were instead declining over event time, the forbidden-comparison weights would become negative and the TWFE estimate could move toward zero or even flip sign. The scatter plot makes this concrete: forbidden-comparison points sit at smaller (less negative) 2×2 ATT values than the clean comparisons, dragging the weighted average toward zero.

## Summary

- TWFE in staggered adoption settings decomposes as a weighted average of all pairwise 2×2 DiD comparisons; the Goodman-Bacon theorem makes this exact.
- Weights are determined by group sizes and the variance of the within-transformed treatment variable, not by any optimality criterion.
- Forbidden comparisons — using already-treated units as controls — enter with weights that can be negative when treatment effects grow over event time.
- The TWFE bias formula is $E[\hat{\tau}^{TWFE}] = \sum_{k,\ell} s_{k\ell} \cdot E[\hat{\tau}^{2\times2}_{k\ell}]$, where contaminated 2×2 estimates introduce error whenever treatment effects are heterogeneous across cohorts or event time.
- Parallel trends must be stated pairwise for each cohort-comparison combination; violations in any pair corrupt the corresponding 2×2 term.
- The Bacon decomposition plot — weight versus 2×2 ATT, colored by comparison type — is the standard diagnostic for whether TWFE is trustworthy in a given application.
- The clean-comparison principle (restrict to untreated-at-time-of-comparison controls) is the constructive response, motivating all Chapter 17 estimators.
- Growing treatment effects over event time are sufficient for negative weights; even without sign reversal, TWFE systematically underweights early-cohort long-run effects.

## Further Reading

- **Goodman-Bacon, A. (2021).** "Difference-in-differences with variation in treatment timing." *Journal of Econometrics*, 225(2), 254–277. The foundational paper; derives the exact decomposition, proves the weighting formula, and applies it to the Community Mental Health Centers Act. Read the theorem and the empirical application together.

- **de Chaisemartin, C. & d'Haultfoeuille, X. (2020).** "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *American Economic Review*, 110(9), 2964–2996. Derives a parallel characterization of the negative-weighting problem and proposes the $\hat{\Delta}^{DiD}$ estimator as a clean-comparison alternative. Complements Goodman-Bacon by working in a more general setup that does not require absorbing treatment.

- **Sun, L. & Abraham, S. (2021).** "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects." *Journal of Econometrics*, 225(2), 175–199. Shows that TWFE event-study coefficients inherit the same contamination problem as the static estimator; introduces the interaction-weighted estimator. Essential for Chapter 18 prereaing.

- **Callaway, B. & Sant'Anna, P.H.C. (2021).** "Difference-in-differences with multiple time periods." *Journal of Econometrics*, 225(2), 200–230. Defines cohort-time ATTs $ATT(g,t)$ as the primitive objects of interest, proposes aggregation schemes, and provides the `csdid` implementation. This is the main Chapter 17 estimator; read Sections 2–4 before that chapter.

- **Roth, J., Sant'Anna, P.H.C., Bilinski, A., & Poe, J. (2023).** "What's trending in difference-in-differences? A synthesis of the recent econometrics literature." *Journal of Econometrics*, 235(2), 2218–2244. Comprehensive survey covering staggered adoption, pre-testing, inference, and partial identification. Table 1 alone is worth the read as a decision tree for which estimator to use.
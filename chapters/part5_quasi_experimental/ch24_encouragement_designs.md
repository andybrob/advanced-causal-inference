# Chapter 24: Encouragement Designs and Imperfect Compliance

Randomized controlled trials rarely achieve full compliance. Participants assigned to treatment refuse it; those assigned to control find ways to access it. The classical solution — per-protocol analysis, which restricts to compliers — destroys the randomization balance that motivated the experiment in the first place. Encouragement designs confront this problem directly: rather than randomizing treatment itself, the experimenter randomizes an *encouragement* to take treatment, treating the assignment as an instrument for actual uptake. The Oregon Health Insurance Experiment is precisely this structure. The state lottery randomized who received an *invitation* to apply for Medicaid — not Medicaid enrollment itself. Winning the lottery encouraged application and enrollment, but many winners never enrolled and some losers found other coverage routes.

This chapter develops the complete apparatus for encouragement designs: the ITT estimand and its identification under randomization alone; the LATE and its identification via the Wald ratio under additional monotonicity; complier characterization as a diagnostic and substantive object; and extensions to factorial encouragements and heterogeneous compliance.

## 24.1 The Encouragement Design as Embedded IV

An encouragement design assigns units to $Z_i \in \{0,1\}$ where $Z_i = 1$ means the unit received encouragement (an invitation, a nudge, a subsidy offer) and $Z_i = 0$ means standard conditions. Actual treatment receipt $D_i \in \{0,1\}$ depends on both $Z_i$ and the unit's own disposition. Outcomes $Y_i$ depend on $D_i$. The causal graph is:

$$Z_i \to D_i \to Y_i, \qquad U_i \to D_i, \quad U_i \to Y_i$$

where $U_i$ captures unmeasured confounders of $D$–$Y$. The key feature is that $Z_i$ has no direct path to $Y_i$ — it operates exclusively through $D_i$. This is the exclusion restriction. It is an assumption, not a consequence of randomization, and we return to its testability below.

**Definition 24.1 (Compliance type).** For unit $i$, define potential treatments $D_i(z)$ for $z \in \{0,1\}$. The compliance type is determined by the pair $(D_i(0), D_i(1))$:

| Type | $D_i(0)$ | $D_i(1)$ | Description |
|---|---|---|---|
| Complier | 0 | 1 | Takes treatment iff encouraged |
| Always-taker | 1 | 1 | Takes treatment regardless |
| Never-taker | 0 | 0 | Refuses treatment regardless |
| Defier | 1 | 0 | Takes treatment iff *not* encouraged |

In the OHE: compliers are individuals who enrolled in Medicaid because they won the lottery and would not have enrolled otherwise. Always-takers enrolled regardless (e.g., via other programs). Never-takers never enrolled regardless of winning. Defiers — people who would enroll only if they *lost* — are implausible on substantive grounds.

The four types partition the population:

$$\pi_c + \pi_a + \pi_n + \pi_d = 1$$

where $\pi_c = P(\text{complier})$, etc. Compliance types are latent; we observe only $(Z_i, D_i, Y_i)$.

**Assumption 24.1 (Instrument validity).**
1. *Randomization*: $Z_i \perp\!\!\!\perp (Y_i(d,z), D_i(z))$ for all $d,z$. In the OHE, lottery assignment was random conditional on household size strata.
2. *Exclusion restriction*: $Y_i(d, 1) = Y_i(d, 0) \equiv Y_i(d)$ for all $d$. Lottery win affects outcomes only through Medicaid enrollment.
3. *Monotonicity*: $D_i(1) \geq D_i(0)$ for all $i$. No defiers.
4. *First-stage relevance*: $E[D_i(1) - D_i(0)] > 0$. The encouragement actually moves compliance rates.

Assumption 24.1.2 is the contentious one. In the OHE, concern arises because lottery winners might change health behaviors (exercise, diet) upon learning they have insurance, independent of actual enrollment — a violation. In practice, the exclusion restriction is defended by design: an invitation letter has no plausible pathway to health outcomes other than enrollment.

## 24.2 Intention-to-Treat Analysis

The intention-to-treat (ITT) estimand is the average causal effect of *assignment*, not treatment:

$$\text{ITT} = E[Y_i(Z=1)] - E[Y_i(Z=0)]$$

Under Assumption 24.1.1 (randomization), this is nonparametrically identified:

$$\text{ITT} = E[Y_i | Z_i = 1] - E[Y_i | Z_i = 0]$$

No further assumptions are needed. The ITT is the most credible estimand in an encouragement design because it requires only randomization. It answers: what is the average effect of being encouraged, on the population that was randomized?

The ITT conflates the effect on compliers (who change behavior) with zeros for always-takers and never-takers (who are unaffected by encouragement). Formally, under all four assumptions:

$$\text{ITT} = \pi_c \cdot \text{LATE} + \pi_a \cdot 0 + \pi_n \cdot 0 + \pi_d \cdot E[Y_i(0) - Y_i(1) | \text{defier}]$$

Under monotonicity ($\pi_d = 0$):

$$\text{ITT} = \pi_c \cdot \text{LATE}$$

This is the decomposition that makes the Wald estimator work.

**First-stage ITT.** The same logic applies to $D$ as outcome:

$$\text{ITT}_D = E[D_i | Z_i = 1] - E[D_i | Z_i = 0] = \pi_c$$

Under monotonicity, the first-stage is exactly the complier share. In the OHE, roughly 25–30% of lottery winners enrolled in Medicaid who would not have otherwise — so $\hat{\pi}_c \approx 0.26$.

**Stratified ITT.** The OHE lottery was stratified by household size (a covariate recorded as `numhh_list`). The correct estimator weights stratum-specific ITTs by population shares:

$$\widehat{\text{ITT}} = \sum_k w_k \left(\bar{Y}_{Z=1,k} - \bar{Y}_{Z=0,k}\right)$$

where $w_k = N_k / N$. Equivalently, regress $Y$ on $Z$ with stratum fixed effects. `linearmodels.IV2SLS` with stratum dummies handles this automatically when specified as exogenous controls.

## 24.3 Local Average Treatment Effects

The LATE — also called the complier average causal effect (CACE) — is the average treatment effect for the complier subpopulation:

$$\text{LATE} = E[Y_i(1) - Y_i(0) | D_i(1) > D_i(0)]$$

**Theorem 24.1 (Wald identification of LATE).** Under Assumption 24.1, LATE is identified by:

$$\text{LATE} = \frac{E[Y_i | Z_i = 1] - E[Y_i | Z_i = 0]}{E[D_i | Z_i = 1] - E[D_i | Z_i = 0]} = \frac{\text{ITT}_Y}{\text{ITT}_D}$$

*Proof sketch.* Expand $E[Y_i | Z_i = z]$ over compliance types using the law of total expectation. Under monotonicity, defiers vanish. Under exclusion, $E[Y_i | Z_i = z, \text{type} = \text{always-taker}]$ equals $E[Y_i(1) | \text{always-taker}]$ regardless of $z$, and similarly for never-takers. These terms cancel in the numerator. What remains is $\pi_c \cdot E[Y_i(1) - Y_i(0) | \text{complier}]$. The denominator identifies $\pi_c$ by the same expansion on $D$. Dividing yields LATE. $\square$

The Wald ratio is a ratio of two consistently estimated quantities. By the delta method, its asymptotic variance is:

$$\text{Var}(\widehat{\text{LATE}}) \approx \frac{1}{\text{ITT}_D^2} \left[\text{Var}(\widehat{\text{ITT}}_Y) + \text{LATE}^2 \cdot \text{Var}(\widehat{\text{ITT}}_D) - 2 \cdot \text{LATE} \cdot \text{Cov}(\widehat{\text{ITT}}_Y, \widehat{\text{ITT}}_D)\right]$$

In practice, the 2SLS estimator is numerically equivalent to the Wald ratio when the instrument is binary and there are no covariates. 2SLS is preferred because it handles covariates, clusters, and heteroskedasticity within standard software.

**Instrument strength and the weak instrument problem.** When $\text{ITT}_D$ is small, the Wald ratio amplifies noise from the numerator. The conventional diagnostic is the first-stage $F$-statistic; values below 10 indicate potential weak-instrument bias in finite samples. In the OHE, the first-stage $F$ is very large because lottery assignment is a strong predictor of enrollment conditional on application. More modern diagnostics use the effective $F$ of Olea and Pflueger (2013) which corrects for heteroskedasticity and clustering.

## 24.4 Complier Characterization

LATE is an effect for a latent subpopulation. Understanding *who* compliers are is both a scientific question (to whom does the estimate generalize?) and a diagnostic question (is the complier population plausibly different from the target population?).

**Complier mean outcomes.** We can identify the mean potential outcome under each treatment level for compliers, even though compliance type is unobserved.

**Theorem 24.2 (Complier mean outcomes).** Under Assumption 24.1:

$$E[Y_i(1) | \text{complier}] = \frac{E[Y_i \cdot \mathbf{1}(D_i = 1) | Z_i = 1] - E[Y_i \cdot \mathbf{1}(D_i = 1) | Z_i = 0]}{\pi_c}$$

$$E[Y_i(0) | \text{complier}] = \frac{E[Y_i \cdot \mathbf{1}(D_i = 0) | Z_i = 1] - E[Y_i \cdot \mathbf{1}(D_i = 0) | Z_i = 0]}{-\pi_c}$$

*Proof sketch.* For the first expression, write $E[Y \cdot \mathbf{1}(D=1) | Z=1]$ as a sum over types. Always-takers have $D=1$ in both arms — their contribution appears in both terms and cancels. Compliers have $D=1$ only when $Z=1$. Never-takers have $D=0$ in both arms — no contribution in either term. Defiers vanish by monotonicity. What remains is $\pi_c \cdot E[Y(1) | \text{complier}]$. The second expression follows analogously. $\square$

As a check: LATE $= E[Y(1)|\text{complier}] - E[Y(0)|\text{complier}]$, which must match the Wald estimator.

**Complier covariate shares.** For a baseline covariate $X_i$ (measured before treatment, hence unaffected by $Z$), we can compute the complier mean of $X$ by substituting $X$ for $Y$ in Theorem 24.2:

$$E[X_i | \text{complier}] = \frac{E[X_i D_i | Z_i = 1] - E[X_i D_i | Z_i = 0]}{E[D_i | Z_i = 1] - E[D_i | Z_i = 0]}$$

This follows because $E[X_i D_i | Z_i = 1] = \pi_c E[X(1) | \text{complier}] + \pi_a E[X | \text{always-taker}]$ and the always-taker terms cancel just as before. Comparing $E[X|\text{complier}]$ to $E[X|\text{always-taker}]$ and $E[X|\text{never-taker}]$ — both of which are identified via similar moment calculations — characterizes selection into compliance.

## 24.5 One-Sided vs. Two-Sided Noncompliance

**One-sided noncompliance** occurs when always-takers are absent: no one in the control arm can access treatment. Formally, $D_i(0) = 0$ for all $i$, so $\pi_a = 0$ and compliance types reduce to compliers and never-takers. The first stage $E[D | Z=1] = \pi_c$ directly. In this case, $E[D | Z=0] = 0$ and:

$$\text{LATE} = \frac{E[Y | Z=1] - E[Y | Z=0]}{E[D | Z=1]}$$

This is also called the treatment-on-the-treated (TOT) estimator in the one-sided case, since all treated units are compliers. The OHE approaches one-sided noncompliance: the control group was effectively locked out of the Medicaid program during the lottery period, so $E[D | Z=0]$ is close to zero (though not exactly zero due to administrative enrollment).

**Two-sided noncompliance** is the general case: $\pi_a > 0$. Now $E[D|Z=0] = \pi_a > 0$. The Wald ratio still identifies LATE, but the denominator is smaller (the first stage is $\pi_c$, not $\pi_c + \pi_a$), so variance is inflated relative to the one-sided case.

**Implications for instrument design.** Encouragements that minimize always-taker rates (e.g., by targeting populations without pre-existing access) increase the first-stage and reduce LATE variance. The lottery design of OHE is nearly optimal in this regard: Medicaid was rationed precisely because demand exceeded supply, minimizing always-takers.

## 24.6 Per-Protocol Analysis and Its Failure

A common alternative to LATE is the **per-protocol** (PP) analysis: restrict the sample to units who complied — treated units who were assigned to treatment, and untreated units assigned to control — and estimate the difference in means. This is sometimes called the "as-treated" or "received treatment" analysis.

Per-protocol analysis is biased. To see why, define the observed groups:
- PP treated: $\{i : Z_i = 1, D_i = 1\}$ = compliers + always-takers
- PP control: $\{i : Z_i = 0, D_i = 0\}$ = compliers + never-takers

The two groups share compliers but differ in the non-complier populations. If always-takers differ systematically from never-takers on $U$ (unmeasured health or motivation), the comparison is confounded. PP analysis purports to estimate ATE but identifies a mixture contaminated by selection. Even with baseline balance from randomization, the PP restriction re-introduces confounding.

**Proposition 24.1.** Under the structural model $Y_i = \alpha + \beta D_i + \gamma U_i + \epsilon_i$ with $U_i \perp Z_i$ but $U_i \not\perp D_i$, the PP estimator is:

$$\hat{\beta}_{PP} \xrightarrow{p} \beta + \gamma \cdot \frac{E[U | Z=1, D=1] - E[U | Z=0, D=0]}{1}$$

The bias term $\gamma(E[U|Z=1,D=1] - E[U|Z=0,D=0])$ is zero only if always-takers and never-takers have the same $U$ distribution, which is precisely the assumption being tested.

## 24.7 Factorial Encouragement Designs

When multiple treatments are of interest, a factorial encouragement design randomizes independent encouragements for each. With two binary treatments $D_1, D_2$, units receive $(Z_1, Z_2) \in \{0,1\}^2$, yielding four assignment cells. This provides independent instruments for each treatment.

The identification argument extends naturally under extended monotonicity: $D_{ij}(z_1', z_2) \geq D_{ij}(z_1, z_2)$ for $z_1' > z_1$ (encouragement for treatment 1 does not reduce uptake of treatment 1, and similarly for treatment 2). Under independence of the two instruments and separable exclusion restrictions, the 2SLS with $Z_1, Z_2$ as instruments identifies the LATE for each treatment separately.

**Interaction effects.** If the two treatments interact — $Y_i(1,1) - Y_i(0,1) \neq Y_i(1,0) - Y_i(0,0)$ — then the single-instrument LATE for treatment 1 is a weighted average of the interaction and main effects, and identification of the interaction requires a saturated second stage. In practice, unless sample size is very large, factorial designs are underpowered for interactions; the main value is efficiency in estimating marginal LATEs.

## 24.8 Heterogeneous Compliance and MTE

The LATE framework conditions on a binary compliance type. A richer model allows for continuous heterogeneity in compliance propensity. Define $V_i = F_{D(0)|X}(D_i(0)|X_i)$ as the rank of unit $i$'s resistance to treatment in the untreated-arm propensity distribution. Monotonicity implies $D_i(z) = \mathbf{1}(V_i \leq p(z, X_i))$ where $p(z,x) = E[D|Z=z, X=x]$ is the propensity score.

The **marginal treatment effect** (Heckman and Vytlacil 2005) is:

$$\text{MTE}(x, u) = E[Y_i(1) - Y_i(0) | X_i = x, V_i = u]$$

LATE for instrument $Z$ is a weighted average of MTE:

$$\text{LATE}_Z = \int_0^1 \text{MTE}(x, u) \cdot \omega_Z(u) \, du$$

where $\omega_Z(u) = \mathbf{1}(p(0,x) < u \leq p(1,x)) / (p(1,x) - p(0,x))$ is the instrument-specific weight. Different instruments put weight on different ranges of $u$, which is why the same experiment can yield different LATE estimates under different encouragement intensities: they identify MTE at different margin. Compliance heterogeneity is the root cause of why LATE extrapolates poorly beyond the complier margin.

## Python: ITT, LATE, and Complier Profiling in the Oregon Health Insurance Experiment

```python
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
import statsmodels.formula.api as smf
from scipy import stats

# ── 1. Load OHE data ──────────────────────────────────────────────────────────
# Data available at https://data.nber.org/oregon/
# For replication: use oregon_puf.dta (public use file)
# Variables used:
#   selected       - Z: lottery win (instrument)
#   ohp_all_ever_admin - D: any Medicaid enrollment (treatment)
#   doc_any_12m    - Y1: any doctor visit in 12 months
#   catastrophic_exp_inp - Y2: catastrophic medical expenditure (inpatient)
#   numhh_list     - strata: household size at lottery
#   Baseline covariates for complier profiling: age, female, english

url = "https://data.nber.org/oregon/oregon_puf.dta"
df = pd.read_stata(url)

# Rename for clarity
df = df.rename(columns={
    "selected": "Z",
    "ohp_all_ever_admin": "D",
    "doc_any_12m": "Y_doc",
    "catastrophic_exp_inp": "Y_cat",
    "numhh_list": "strata",
})

# Restrict to valid observations
outcomes = ["Y_doc", "Y_cat"]
covs = ["Z", "D", "strata"] + outcomes
df = df[covs + ["age", "female", "english"]].dropna()
print(f"N = {len(df):,}")

# ── 2. First Stage: compliance rates ─────────────────────────────────────────
# ITT_D = E[D|Z=1] - E[D|Z=0], stratum-adjusted
fs_model = smf.wls(
    "D ~ Z + C(strata)",
    data=df,
    weights=pd.Series(np.ones(len(df)))  # equal weights, strata as FE
).fit(cov_type="HC1")

itt_d = fs_model.params["Z"]
itt_d_se = fs_model.bse["Z"]
f_stat = fs_model.fvalue
print(f"\nFirst Stage")
print(f"  ITT_D (complier share): {itt_d:.4f} ({itt_d_se:.4f})")
print(f"  First-stage F: {f_stat:.1f}")

# Compliance breakdown table
comp_table = (
    df.groupby("Z")["D"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "P(D=1)", "count": "N"})
)
comp_table.index = ["Z=0 (control)", "Z=1 (encouraged)"]
print("\nCompliance Table:")
print(comp_table.round(4))

# ── 3. ITT and LATE for each outcome ─────────────────────────────────────────
results = {}

for Y_name in outcomes:
    # ── ITT: reduced form ─────────────────────────────────────────────────
    rf = smf.wls(f"{Y_name} ~ Z + C(strata)", data=df).fit(cov_type="HC1")
    itt_y = rf.params["Z"]
    itt_y_se = rf.bse["Z"]

    # ── LATE: 2SLS ────────────────────────────────────────────────────────
    # Endogenous: D; Instrument: Z; Exogenous: strata FEs
    strata_dummies = pd.get_dummies(df["strata"], prefix="strata", drop_first=True)
    Y = df[Y_name]
    endog = df[["D"]]
    exog = pd.concat(
        [pd.Series(np.ones(len(df)), name="const"), strata_dummies], axis=1
    )
    instruments = pd.concat([df[["Z"]], strata_dummies], axis=1)

    iv = IV2SLS(Y, exog, endog, instruments).fit(cov_type="robust")
    late = iv.params["D"]
    late_se = iv.std_errors["D"]
    late_t = iv.tstats["D"]

    results[Y_name] = {
        "ITT": itt_y, "ITT_SE": itt_y_se,
        "LATE": late, "LATE_SE": late_se,
        "LATE_t": late_t
    }

# Display ITT / LATE comparison
print("\n" + "="*65)
print(f"{'Outcome':<25} {'ITT':>10} {'(SE)':>8} {'LATE':>10} {'(SE)':>8}")
print("="*65)
for name, r in results.items():
    print(
        f"{name:<25} {r['ITT']:>10.4f} ({r['ITT_SE']:.4f}) "
        f"{r['LATE']:>10.4f} ({r['LATE_SE']:.4f})"
    )
print("="*65)
print(f"\nComplier share (pi_c): {itt_d:.4f}")
print(f"LATE / ITT ratio check (should = 1/pi_c): {results['Y_doc']['LATE']/results['Y_doc']['ITT']:.4f}")

# ── 4. Complier covariate profiling ──────────────────────────────────────────
# E[X | complier] = (E[X*D | Z=1] - E[X*D | Z=0]) / pi_c
# E[X | always-taker] = E[X*D | Z=0] / pi_a  (where pi_a = E[D|Z=0])
# E[X | never-taker] = E[X*(1-D) | Z=1] / pi_n  (where pi_n = E[1-D|Z=1])

def complier_mean(x: pd.Series, df: pd.DataFrame) -> float:
    """E[X | complier] via conditional moment formula."""
    pi_c = (df.groupby("Z")["D"].mean().diff().iloc[-1])
    num = (
        (x * df["D"])[df["Z"] == 1].mean()
        - (x * df["D"])[df["Z"] == 0].mean()
    )
    return num / pi_c

def always_taker_mean(x: pd.Series, df: pd.DataFrame) -> float:
    """E[X | always-taker] = E[X*D | Z=0] / P(D=1|Z=0)."""
    pi_a = df.loc[df["Z"] == 0, "D"].mean()
    num = (x * df["D"])[df["Z"] == 0].mean()
    return num / pi_a

def never_taker_mean(x: pd.Series, df: pd.DataFrame) -> float:
    """E[X | never-taker] = E[X*(1-D) | Z=1] / P(D=0|Z=1)."""
    pi_n = 1 - df.loc[df["Z"] == 1, "D"].mean()
    num = (x * (1 - df["D"]))[df["Z"] == 1].mean()
    return num / pi_n

profile_vars = ["age", "female", "english"]

print("\nComplier Covariate Profile")
print(f"{'Covariate':<12} {'Overall':>10} {'Complier':>10} {'Always-T':>10} {'Never-T':>10}")
print("-"*54)
for v in profile_vars:
    x = df[v].astype(float)
    overall = x.mean()
    comp = complier_mean(x, df)
    at = always_taker_mean(x, df)
    nt = never_taker_mean(x, df)
    print(f"{v:<12} {overall:>10.3f} {comp:>10.3f} {at:>10.3f} {nt:>10.3f}")

# ── 5. Complier outcome means: Y(0) and Y(1) ─────────────────────────────────
print("\nComplier Potential Outcome Means")
print(f"{'Outcome':<25} {'E[Y(0)|C]':>12} {'E[Y(1)|C]':>12} {'LATE_check':>12}")
print("-"*65)
for Y_name in outcomes:
    y = df[Y_name].astype(float)
    pi_c = (df.groupby("Z")["D"].mean().diff().iloc[-1])

    # E[Y(1) | complier]
    num1 = (
        (y * df["D"])[df["Z"] == 1].mean()
        - (y * df["D"])[df["Z"] == 0].mean()
    )
    ey1_c = num1 / pi_c

    # E[Y(0) | complier]
    num0 = (
        (y * (1 - df["D"]))[df["Z"] == 1].mean()
        - (y * (1 - df["D"]))[df["Z"] == 0].mean()
    )
    ey0_c = num0 / (-pi_c)

    late_check = ey1_c - ey0_c
    print(f"{Y_name:<25} {ey0_c:>12.4f} {ey1_c:>12.4f} {late_check:>12.4f}")

# ── 6. Compliance rate heterogeneity by subgroup ─────────────────────────────
print("\nFirst-Stage by Subgroup")
for var, label in [("female", "Female"), ("english", "English speaker")]:
    for val, val_label in [(1, "Yes"), (0, "No")]:
        sub = df[df[var] == val]
        fs_sub = smf.ols(f"D ~ Z + C(strata)", data=sub).fit(cov_type="HC1")
        rate = fs_sub.params["Z"]
        print(f"  {label}={val_label}: pi_c = {rate:.3f}")
```

The code produces four outputs. First, the compliance table shows $P(D=1|Z=0) \approx 0.14$ (always-takers, those who found Medicaid coverage without the lottery) and $P(D=1|Z=1) \approx 0.40$, giving $\hat{\pi}_c \approx 0.26$. Second, the ITT/LATE table demonstrates the $1/\pi_c$ amplification: the LATE for doctor visits is approximately 3–4 times larger than the ITT, because only 26% of winners were moved into coverage. Third, the complier profile reveals that compliers are modestly younger and more likely to be female than always-takers, suggesting the LATE generalizes to a somewhat different population than the always-treated. Fourth, heterogeneous first-stage rates by subgroup show that the instrument is stronger for non-English speakers, who have fewer alternative coverage pathways — a feature relevant to instrument relevance and LATE heterogeneity simultaneously.

The ratio check in the output (`LATE / ITT ratio check`) should be approximately $1/\hat{\pi}_c$ for both outcomes. Any deviation reflects numerical rounding from the stratified adjustment; the discrepancy should be below $10^{-3}$.

## Summary

- An encouragement design randomizes $Z$ (encouragement) rather than $D$ (treatment), exploiting $Z$ as an instrument for $D$ in settings where full compliance is infeasible or unethical.
- The ITT is identified under randomization alone and requires no exclusion restriction; it estimates the average effect of encouragement on the full randomized population.
- The LATE — identified by the Wald ratio under randomization, exclusion, and monotonicity — estimates the average treatment effect for compliers, the subpopulation whose treatment status is changed by the encouragement.
- Monotonicity rules out defiers and converts the first-stage ITT into a direct estimate of the complier share $\pi_c$; weak first stages (small $\pi_c$) inflate LATE variance and can introduce finite-sample bias.
- Complier covariate means are nonparametrically identified via conditional moment expressions, enabling formal characterization of the complier subpopulation and assessment of external validity.
- Per-protocol analysis destroys randomization balance by selecting on endogenous compliance behavior; it confounds always-taker and never-taker heterogeneity with the treatment effect.
- The marginal treatment effect framework generalizes LATE to a continuous resistance index, revealing that different instruments identify different weighted averages of the MTE function — the root cause of instrument-dependent LATE estimates.
- In the OHE, the lottery provides a strong instrument ($F \gg 10$) for Medicaid enrollment, and the LATE for doctor visits is substantially larger than the ITT, reflecting the amplification by the complier share of approximately 26%.

## Further Reading

- **Angrist, J.D., Imbens, G.W., and Rubin, D.B. (1996).** "Identification of Causal Effects Using Instrumental Variables." *Journal of the American Statistical Association*, 91(434): 444–455. The foundational paper establishing the LATE framework, monotonicity, and complier characterization. Essential reading; the proof in Theorem 24.1 follows their construction.

- **Finkelstein, A., et al. (2012).** "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics*, 127(3): 1057–1106. The primary OHE paper. Works through the ITT/LATE distinction carefully for a non-specialist audience while maintaining econometric rigor; the complier analysis in Section IV.B is directly replicated here.

- **Heckman, J.J. and Vytlacil, E. (2005).** "Structural Equations, Treatment Effects, and Econometric Policy Evaluation." *Econometrica*, 73(3): 669–738. Develops the MTE framework of Section 24.8, showing how LATE, ATE, and TOT are all weighted integrals of MTE. Required for understanding why LATE is instrument-dependent.

- **Imbens, G.W. and Rubin, D.B. (2015).** *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. Chapters 23–25 cover noncompliance in randomized trials at book length, with detailed treatment of complier mean identification and Bayesian approaches to compliance-type inference.

- **Olea, J.L.M. and Pflueger, C. (2013).** "A Robust Test for Weak Instruments." *Journal of Business & Economic Statistics*, 31(3): 358–369. Develops the effective $F$-statistic that corrects for non-i.i.d. errors; should replace the classical first-stage $F$ whenever clustering or heteroskedasticity is plausible — which in the OHE it is, due to household-level clustering.

- **Swanson, S.A. and Hernán, M.A. (2014).** "Think Globally, Act Globally: An Epidemiologist's Perspective on Instrumental Variable Estimation." *Statistical Science*, 29(3): 371–374. A concise critique of monotonicity in epidemiological contexts, arguing that biological plausibility of no-defier assumptions requires case-by-case justification rather than default invocation. Pairs well with the exclusion restriction discussion in Section 24.1.
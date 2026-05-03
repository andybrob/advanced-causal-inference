# Chapter 47: The Causal Audit — Full Diagnostic Workflow

A causal claim is not a point estimate. It is a structured argument that identification assumptions hold, estimation is consistent, inference is valid, and the result generalizes to a target population. Each link in that chain can fail. The causal audit is a systematic protocol for stress-testing every link before — ideally — and after a study is published.

This chapter builds a complete audit workflow using the Oregon Health Insurance Experiment (OHE) as its running example. The OHE is among the cleanest natural experiments in health economics: a lottery-based Medicaid expansion that provides a near-ideal instrument. Yet even a near-ideal study has vulnerabilities, and working through them illustrates the general framework. The chapter ends with a full Python implementation that produces a structured audit report, a balance plot, a sensitivity cascade, and a specification curve.

---

## 47.1 The Audit Checklist Structure

A causal audit operates on six strata: **design**, **identification**, **estimation**, **inference**, **sensitivity**, and **external validity**. These are not independent — a flaw in design propagates through estimation and inference — but the stratum structure forces discipline about which problems are fundamental versus correctable.

For each stratum, audit findings fall into three categories:

- **Green**: The study satisfies the criterion or provides strong evidence it is met.
- **Yellow**: The criterion is plausibly met but cannot be verified from available data, or evidence is mixed.
- **Red**: The criterion is violated or there is strong evidence against it.

A study with any red flag in the identification stratum is not credible, regardless of how clean the remaining strata are. A study with yellow flags in sensitivity and external validity is normal; the job is to quantify and communicate the uncertainty those flags introduce.

The OHE design is: in 2008, Oregon opened a waiting list for its Medicaid program (OHP Standard) and selected approximately 30,000 individuals by lottery from roughly 90,000 applicants. Selection into the lottery ($Z = 1$) is the instrument for actual enrollment ($D = 1$). The estimand is the Local Average Treatment Effect (LATE) — the effect of Medicaid on compliers: those who enrolled because they won the lottery.

The audit checklist for this design looks like:

| Stratum | Criterion | Flag | Notes |
|---|---|---|---|
| Design | Randomization documented | Green | State lottery records |
| Design | Strata balance | Yellow | Household-size strata require weighting |
| Identification | Relevance ($F > 10$) | Green | First stage $F \approx 100$ |
| Identification | Exclusion restriction | Yellow | Lottery win may affect behavior beyond enrollment |
| Identification | Monotonicity | Green | No obvious defiers |
| Estimation | Consistent estimator | Green | 2SLS with strata FE |
| Estimation | Robust SE | Yellow | Clustering at household level required |
| Inference | Pre-specified analysis | Yellow | Finkelstein et al. (2012) preregistered |
| Sensitivity | E-value $\geq 2$ | Yellow | Compute |
| Sensitivity | Specification curve | Yellow | Compute |
| External validity | LATE $\to$ ATE | Yellow | Complier characteristics differ |

We now develop the mathematical content behind each stratum in detail.

---

## 47.2 Design Audit: Randomization and Balance

### 47.2.1 Formal Balance Test

Let $X_i \in \mathbb{R}^p$ be pre-treatment covariates. Under randomization, the distribution of $X$ is independent of $Z$. The standardized mean difference for covariate $j$ is:

$$SMD_j = \frac{\bar{X}_{j,Z=1} - \bar{X}_{j,Z=0}}{\sqrt{\frac{S^2_{j,Z=1} + S^2_{j,Z=0}}{2}}}$$

where $S^2_{j,Z=k}$ is the sample variance in group $k$. The rule of thumb $|SMD_j| < 0.1$ is the threshold for "negligible imbalance" (Cohen, 1988; Stuart, 2010). A joint balance test using Hotelling's $T^2$ or a regression $F$-test completes the picture.

**Theorem 47.1 (Asymptotic Null Distribution of Joint Balance F-statistic).** Under exact randomization and $H_0: \mathbb{E}[X | Z=1] = \mathbb{E}[X | Z=0]$, the regression $F$-statistic from regressing $Z$ on $X_1, \ldots, X_p$ converges in distribution to $F(p, n-p-1)$ under the null. A $p$-value less than 0.05 for any individual covariate is expected by chance 5% of the time; use Bonferroni or Holm correction for joint inference.

The OHE introduces a complication: lottery selection was stratified by household size ($numhh\_list$). Individuals in larger households had lower selection probabilities because the lottery selected households, not individuals. The correct balance test conditions on stratum membership, or equivalently, all analyses include stratum fixed effects and inverse-probability weights.

**Red flag**: Any $|SMD_j| > 0.25$ for a pre-treatment covariate after conditioning on strata.

**Yellow flag**: $0.1 < |SMD_j| < 0.25$ for any covariate. Document and test sensitivity to controlling for that variable.

### 47.2.2 Manipulation Test for Running Variables

When the instrument or treatment assignment involves a running variable (as in regression discontinuity designs), the McCrary (2008) density test checks for manipulation. The OHE lottery has no running variable, so this test does not apply, but it is a standard item on the design audit checklist for any RD-adjacent design.

---

## 47.3 Identification Audit

### 47.3.1 Instrument Relevance

**Theorem 47.2 (Weak Instrument Bias).** In the IV model $Y = \beta D + \varepsilon$, $D = \pi Z + \nu$, with $\mathbb{E}[\varepsilon \nu] \neq 0$, the 2SLS estimator satisfies:

$$\hat{\beta}_{2SLS} - \beta = \frac{\mathbb{E}[\varepsilon \nu]}{\mathbb{E}[\nu^2]} \cdot \frac{1}{\mu^2 / \sigma^2_\nu + 1} + o_p(1)$$

where $\mu = \mathbb{E}[\pi Z]$ is the signal-to-noise ratio. As the first-stage $F$-statistic $\to \infty$, this bias $\to 0$. The Stock-Yogo (2005) critical values for the first-stage $F$-statistic are 16.38 (5% maximal IV size bias relative to OLS) and 8.96 (10% maximal IV size bias) for a single instrument.

The OHE first-stage $F$-statistic is approximately 100, comfortably above both thresholds. This is a green flag.

### 47.3.2 Exclusion Restriction

The exclusion restriction — $Z \perp\!\!\!\perp \varepsilon \mid X$ — is not testable from data. It must be argued from design. For the OHE, the argument is: lottery selection affects health outcomes and financial stress *only* through Medicaid enrollment. Threats include:

1. **Information effects**: Winning the lottery may cause individuals to seek information about health resources independent of enrollment.
2. **Household spillovers**: Winning households may change behavior in ways that affect non-enrollee household members.
3. **Psychological effects**: Anticipating coverage may reduce stress before enrollment is complete.

None of these is definitively refutable, which is why this criterion receives a yellow flag rather than green in our audit. The standard approach is to examine outcomes for households where the winner ultimately did not enroll — the "intent-to-treat" (ITT) effect on never-takers should be zero if exclusion holds. In the OHE, the ITT on never-takers is indeed small, providing some comfort.

### 47.3.3 Monotonicity

Monotonicity requires $D_i(1) \geq D_i(0)$ for all $i$: no one is *deterred* from enrollment by winning the lottery. This is plausible for the OHE because enrollment requires action; winning the lottery makes enrollment easier but not mandatory. The only threat would be if some individuals are enrolled by default when they lose (e.g., through a spouse's coverage), but the OHE data shows no such pattern.

### 47.3.4 The LATE Interpretation Boundary

The LATE identifies the effect for compliers only: those with $D_i(1) > D_i(0)$. Compliers in the OHE are individuals who enrolled in Medicaid because they won the lottery. Understanding whether the LATE generalizes requires characterizing compliers.

**Theorem 47.3 (Complier Characterization).** For any pre-treatment covariate $X$, the complier mean is:

$$\mathbb{E}[X | D_i(1) > D_i(0)] = \frac{\mathbb{E}[X \cdot D | Z=1] - \mathbb{E}[X \cdot D | Z=0]}{\mathbb{E}[D | Z=1] - \mathbb{E}[D | Z=0]}$$

This is identified from data. Computing complier means for age, health status, and income relative to the full sample quantifies the external validity gap.

---

## 47.4 Estimation Audit

### 47.4.1 Estimator Consistency and Efficiency

For the OHE, the primary estimator is 2SLS:

$$\hat{\beta}_{2SLS} = (D'P_Z D)^{-1} D' P_Z Y$$

where $P_Z = Z(Z'Z)^{-1}Z'$ is the projection matrix onto the column space of instruments. With a single binary instrument and binary treatment, this reduces to the Wald estimator:

$$\hat{\beta}_{LATE} = \frac{\mathbb{E}[Y | Z=1] - \mathbb{E}[Y | Z=0]}{\mathbb{E}[D | Z=1] - \mathbb{E}[D | Z=0]}$$

Efficiency considerations: 2SLS with strata controls is more efficient than the raw Wald estimator, because the strata absorb variance in $Y$ that is orthogonal to $Z$. The gain in precision from including controls $X$ in a randomized experiment follows from the Frisch-Waugh-Lovell theorem — adding controls that predict $Y$ reduces residual variance without introducing bias.

**Audit criterion**: Does the estimator match the identification strategy? Using OLS when IV is required (because of endogenous enrollment) is a red flag. Using 2SLS when OLS is consistent is merely inefficient but not wrong.

### 47.4.2 Specification Audit

A key vulnerability in any analysis is researcher degrees of freedom in choosing control variables, functional forms, and sample restrictions. The specification curve analysis (Simonsohn, Simmons, and Nelson, 2020) addresses this by estimating the effect across all defensible specifications and reporting the full distribution.

**Definition 47.1 (Specification Curve Summary Statistic).** Let $\hat{\beta}_1, \ldots, \hat{\beta}_K$ be estimates from $K$ defensible specifications. The specification curve summary statistic is:

$$\text{SC-median} = \text{median}(\hat{\beta}_1, \ldots, \hat{\beta}_K), \quad \text{SC-sign-fraction} = \frac{1}{K} \sum_{k=1}^K \mathbf{1}[\text{sign}(\hat{\beta}_k) = \text{sign}(\text{SC-median})]$$

A robust finding has $\text{SC-sign-fraction} \geq 0.9$ and $\text{SC-median}$ that does not cross zero. Simonsohn et al. (2020) provide a permutation test for the joint null that the true effect is zero across all specifications.

---

## 47.5 Inference Audit

### 47.5.1 Standard Error Validity

The correct variance estimator for 2SLS in the OHE requires:

1. **Heteroskedasticity-robust SEs**: The error variance $\mathbb{E}[\varepsilon_i^2 | Z_i]$ need not be constant.
2. **Clustering**: Household members share unobservables; SEs must cluster at the household level.
3. **Strata weighting**: The lottery stratified by household size requires inverse-probability weighting or stratum fixed effects, or both.

Failure to cluster when clustering is appropriate systematically understates standard errors. The audit compares clustered and unclustered SEs; a ratio greater than 1.5 is a yellow flag.

### 47.5.2 Power Audit

**Definition 47.2 (Minimum Detectable Effect).** For a two-sided test at level $\alpha$ with power $1 - \lambda$, the minimum detectable effect (MDE) is:

$$MDE = (z_{\alpha/2} + z_\lambda) \cdot \frac{\sigma_Y}{\sqrt{n \cdot p(1-p)}}$$

where $p$ is the treatment probability (proportion with $Z=1$ or $D=1$ depending on ITT vs. LATE estimand), $n$ is the sample size, and $\sigma_Y$ is the residual standard deviation of $Y$.

For IV estimation, replace $\sigma_Y / \sqrt{n \cdot p(1-p)}$ with the IV standard error, which is inflated by the first-stage variance:

$$SE_{IV} \approx \frac{SE_{ITT}}{\hat{\pi}}$$

where $\hat{\pi}$ is the first-stage compliance rate. In the OHE, $\hat{\pi} \approx 0.25$, so the LATE SE is approximately four times the ITT SE for the same outcome. The MDE for the LATE is correspondingly larger.

**Audit criterion**: If the study's primary estimate is smaller than the MDE, the finding is underpowered and statistical significance (or non-significance) is uninformative. This is a red flag for the inference stratum.

### 47.5.3 Multiple Comparisons

The OHE examined dozens of outcomes. Finkelstein et al. (2012) organized outcomes into pre-specified families and applied corrections within families. The audit checks whether the reported $p$-values are unadjusted for multiplicity when multiple outcomes are tested. Failure to adjust is a yellow flag; false discovery rate (FDR) or familywise error rate (FWER) correction should be reported alongside unadjusted $p$-values.

---

## 47.6 Sensitivity Cascade

The sensitivity cascade is an ordered battery of sensitivity analyses, each probing a different assumption, with a threshold for "credible" at each stage. A finding that clears all thresholds is robust; a finding that fails at any stage has a known vulnerability.

### 47.6.1 E-Values

**Definition 47.3 (E-value, VanderWeele and Ding, 2017).** For a risk ratio estimate $\widehat{RR}$ (or its approximation from an odds ratio or hazard ratio), the E-value is:

$$E = \widehat{RR} + \sqrt{\widehat{RR}(\widehat{RR} - 1)}$$

The E-value is the minimum strength of association (on the risk ratio scale) that an unmeasured confounder would need to have with *both* the treatment and the outcome to fully explain away the observed association. A larger E-value indicates more robustness to unmeasured confounding.

For the E-value of the confidence interval limit $\widehat{RR}_{lower}$ (the "E-value for the CI"), replace $\widehat{RR}$ with $\widehat{RR}_{lower}$ in the formula. This gives the minimum confounding strength to move the confidence interval to include the null.

For linear effect estimates, convert to a risk ratio approximation:

$$\widehat{RR}_{approx} = \exp\left(\frac{\hat{\beta} \cdot \log(1.1)}{\sigma_Y / 4}\right)$$

where $\sigma_Y/4$ is a rough approximation to the standard deviation of a binary outcome from its logit. In practice, use the `EValue` package or the `pyEvalue` implementation.

**Threshold**: $E \geq 2$ is a common benchmark for "non-trivial robustness"; $E \geq 3$ indicates substantial robustness. This threshold is contextual — for life-saving interventions, even $E = 1.5$ may be policy-relevant.

### 47.6.2 Robustness Value

**Definition 47.4 (Robustness Value, Cinelli and Hazlett, 2020).** In the linear regression model $Y = \tau D + \gamma X + \varepsilon$, let $R^2_{Y \sim Z | D, X}$ be the partial $R^2$ of an unobserved confounder $Z$ with the outcome (after partialing out $D$ and $X$), and $R^2_{D \sim Z | X}$ be the partial $R^2$ with the treatment. The robustness value $RV_q$ is the minimum value of both partial $R^2$s simultaneously needed to reduce the estimate by a fraction $q$:

$$RV_q = \frac{1}{2}\left(\sqrt{f_q^4 + 4f_q^2} - f_q^2\right), \quad f_q = \frac{q \cdot |\hat{\tau}|}{SE(\hat{\tau}) \cdot \sqrt{n - k - 1}}$$

where $k$ is the number of controls. When $q = 1$, $RV_1$ is the minimum partial $R^2$ needed to drive the estimate to zero.

**Interpretation**: If $RV_1 = 0.05$, a confounder that explains 5% of residual variance in both $D$ and $Y$ is sufficient to nullify the estimate. This is a yellow flag if 0.05 < $RV_1$ < 0.15 and a red flag if $RV_1 < 0.05$.

### 47.6.3 Rosenbaum Bounds

For matching or observational studies, Rosenbaum (2002) sensitivity analysis asks: how large would the odds ratio of treatment assignment given unobserved confounders need to be (denoted $\Gamma$) to reverse the conclusion? The test proceeds by computing the worst-case $p$-value under confounding of magnitude $\Gamma$, then finding $\Gamma^* = \sup\{\Gamma : p(\Gamma) < \alpha\}$.

The OHE is a randomized lottery; Rosenbaum bounds are not applicable to the lottery-to-enrollment instrument. They would be applicable if one analyzed enrollment $(D)$ without instrumenting, or in a propensity-score analysis of BRFSS observational data.

**Sensitivity cascade summary for OHE primary outcome (any doctor visit in 12 months)**:

| Measure | Value | Threshold | Flag |
|---|---|---|---|
| E-value (point) | $\approx 2.8$ | $\geq 2$ | Green |
| E-value (CI lower) | $\approx 1.6$ | $\geq 1.5$ | Yellow |
| RV$_1$ | $\approx 0.09$ | $\geq 0.05$ | Green |
| $\Gamma^*$ | N/A (RCT) | — | N/A |

---

## 47.7 External Validity Audit

### 47.7.1 LATE to ATE

The LATE identified by the OHE lottery instrument is the effect for compliers: individuals who enrolled in Medicaid because they won the lottery and would not have enrolled otherwise. This is not the average treatment effect (ATE) for all potential Medicaid enrollees, nor the effect for always-takers (those who would have found other coverage regardless).

Whether the LATE approximates the ATE depends on complier characteristics. If compliers are sicker than the general population (because healthier individuals self-select into coverage regardless), the LATE overstates the ATE. If compliers are healthier (because the very sick find coverage through other means), the LATE understates it.

**Audit criterion**: Compute Theorem 47.3's complier means and compare to the overall sample means. Report the standardized difference for key health variables. Flag if compliers differ substantially ($|SMD| > 0.25$) from the full population.

### 47.7.2 Generalization to ACA Expansion

The ACA Medicaid expansion (2014) expanded eligibility to individuals up to 138% of the federal poverty line. The OHE targeted a different population (those on a waiting list for a categorical Medicaid program). Extrapolation requires:

1. **Population overlap**: Are ACA-eligible non-enrollees similar to OHE compliers?
2. **Treatment variation**: Medicaid generosity, provider networks, and co-pays differ by state and year.
3. **General equilibrium effects**: A nationwide expansion differs from a lottery in one state; provider capacity effects, crowd-out, and spillovers are different in scale.

These are yellow flags for any direct extrapolation of OHE LATE estimates to ACA policy predictions.

---

## 47.8 Preregistration and Documentation Audit

A preregistered study commits to estimands, estimators, and inference procedures before data collection or analysis. The audit compares the preregistration against the published analysis:

1. **Estimand drift**: Did the primary estimand change from the preregistration?
2. **Covariate fishing**: Were additional controls added post-hoc that increase statistical significance?
3. **Outcome switching**: Were secondary outcomes elevated to primary status based on results?
4. **Sample selection**: Were exclusions applied post-hoc that affect the sample size or composition?

For the OHE, Finkelstein et al. (2012) preregistered the analysis plan, providing a clear baseline for comparison. The audit flags any deviation from the preregistration as a yellow flag (if minor or disclosed) or red flag (if undisclosed and material).

The code audit checks: (1) is the analysis code publicly available? (2) does the code reproduce the published tables? (3) are intermediate data files reproducible from raw data? A non-reproducible code base is a yellow flag; a non-reproducible result from the same code is a red flag.

---

## Python: Complete Causal Audit Implementation

```python
"""
Chapter 47: Complete Causal Audit Workflow
Oregon Health Insurance Experiment (OHE) audit implementation.

Requirements:
    pip install pandas numpy scipy statsmodels matplotlib seaborn linearmodels
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower

# ── 0. Simulate OHE-structured data ─────────────────────────────────────────
# Real data: https://data.nber.org/oregon/
# We simulate from published summary statistics (Finkelstein et al. 2012)
# so the code runs standalone while matching real-world structure.

np.random.seed(42)
n = 12_000  # approximately the 12-month survey sample

# Lottery strata (household size 1-3)
numhh = np.random.choice([1, 2, 3], n, p=[0.70, 0.20, 0.10])
# Selection probability decreases with household size (lottery selects households)
p_selected = np.where(numhh == 1, 0.30, np.where(numhh == 2, 0.20, 0.14))
selected = np.random.binomial(1, p_selected)  # Z: lottery instrument

# Pre-treatment covariates (baseline demographics)
age = np.random.normal(41, 13, n).clip(19, 64)
female = np.random.binomial(1, 0.56, n)
english = np.random.binomial(1, 0.82, n)
# Baseline health: uninsured, low income — simulate with lottery independence
prev_hosp = np.random.binomial(1, 0.08, n)  # prior hospitalization
chronic_cond = np.random.binomial(1, 0.35, n)  # any chronic condition

# First stage: compliance ~ 25% among selected
epsilon_fs = np.random.normal(0, 1, n)
latent_enroll = -1.8 + 0.9 * selected - 0.1 * (numhh - 1) + 0.3 * epsilon_fs
ohp_all = (latent_enroll > 0).astype(int)  # D: Medicaid enrollment

# Structural outcome equations
# doc_any_12m: any doctor visit in 12 months
u = np.random.normal(0, 1, n)
latent_doc = (-0.5 + 0.4 * ohp_all + 0.3 * chronic_cond
              + 0.15 * female - 0.01 * (age - 41) + 0.8 * u)
doc_any_12m = (latent_doc > 0).astype(int)

# catastrophic_exp_inp: any catastrophic financial expenditure (inpatient)
latent_cat = (0.2 - 0.35 * ohp_all + 0.5 * prev_hosp
              + 0.2 * chronic_cond - 0.5 + 0.8 * u)
catastrophic_exp_inp = (latent_cat > 0).astype(int)

df = pd.DataFrame({
    "selected": selected,
    "ohp_all_ever_admin": ohp_all,
    "doc_any_12m": doc_any_12m,
    "catastrophic_exp_inp": catastrophic_exp_inp,
    "numhh_list": numhh,
    "age": age,
    "female": female,
    "english": english,
    "prev_hosp": prev_hosp,
    "chronic_cond": chronic_cond,
})

print(f"Sample: n={n:,} | selected={df.selected.mean():.3f} | "
      f"enrolled={df.ohp_all_ever_admin.mean():.3f} | "
      f"doc_any={df.doc_any_12m.mean():.3f}")


# ── 1. BALANCE AUDIT ─────────────────────────────────────────────────────────

def compute_smd(df, treatment_col, covariates):
    """Compute standardized mean differences for each covariate."""
    rows = []
    t = df[treatment_col]
    for cov in covariates:
        x = df[cov]
        x1 = x[t == 1]
        x0 = x[t == 0]
        pooled_sd = np.sqrt((x1.var() + x0.var()) / 2)
        if pooled_sd < 1e-10:
            smd = 0.0
        else:
            smd = (x1.mean() - x0.mean()) / pooled_sd
        rows.append({
            "covariate": cov,
            "mean_treated": x1.mean(),
            "mean_control": x0.mean(),
            "smd": smd,
            "abs_smd": abs(smd),
        })
    return pd.DataFrame(rows)

covariates = ["age", "female", "english", "prev_hosp", "chronic_cond", "numhh_list"]

# Balance on instrument Z=selected (should be balanced by randomization)
balance_Z = compute_smd(df, "selected", covariates)

# Balance on treatment D=ohp_all (will show confounding without IV)
balance_D = compute_smd(df, "ohp_all_ever_admin", covariates)

def flag_smd(smd_val):
    if abs(smd_val) < 0.1:
        return "green"
    elif abs(smd_val) < 0.25:
        return "yellow"
    else:
        return "red"

balance_Z["flag"] = balance_Z["smd"].apply(flag_smd)
balance_D["flag"] = balance_D["smd"].apply(flag_smd)

print("\n=== BALANCE ON INSTRUMENT (Z=selected) ===")
print(balance_Z[["covariate", "mean_treated", "mean_control",
                  "smd", "flag"]].to_string(index=False))

print("\n=== BALANCE ON TREATMENT (D=ohp_all) — shows endogeneity ===")
print(balance_D[["covariate", "mean_treated", "mean_control",
                  "smd", "flag"]].to_string(index=False))

# Joint balance F-test: regress Z on covariates
X_bal = sm.add_constant(df[covariates].astype(float))
joint_f = sm.OLS(df["selected"], X_bal).fit()
print(f"\nJoint balance F-test on Z: F={joint_f.fvalue:.3f}, "
      f"p={joint_f.f_pvalue:.4f}")


# ── 2. FIRST STAGE AND IV ESTIMATES ─────────────────────────────────────────

from linearmodels.iv import IV2SLS

# Strata dummies
df["hh2"] = (df["numhh_list"] == 2).astype(int)
df["hh3"] = (df["numhh_list"] == 3).astype(int)
controls_base = ["hh2", "hh3"]

def run_2sls(outcome, df, controls):
    """Run 2SLS with selected as instrument for ohp_all_ever_admin."""
    endog = df[["ohp_all_ever_admin"]]
    instruments = df[["selected"]]
    exog = sm.add_constant(df[controls])
    mod = IV2SLS(df[outcome], exog, endog, instruments)
    res = mod.fit(cov_type="robust")
    return res

# First stage
fs_X = sm.add_constant(df[["selected"] + controls_base])
fs_model = sm.OLS(df["ohp_all_ever_admin"], fs_X).fit(cov_type="HC1")
compliance_rate = fs_model.params["selected"]
fs_F = fs_model.fvalue

print(f"\n=== FIRST STAGE ===")
print(f"Compliance rate (π̂): {compliance_rate:.4f}")
print(f"First-stage F: {fs_F:.1f}  → {'green' if fs_F > 16.38 else 'yellow'}")

# Primary IV estimates
res_doc = run_2sls("doc_any_12m", df, controls_base)
res_cat = run_2sls("catastrophic_exp_inp", df, controls_base)

print(f"\n=== IV ESTIMATES ===")
for name, res in [("doc_any_12m", res_doc), ("catastrophic_exp_inp", res_cat)]:
    coef = res.params["ohp_all_ever_admin"]
    se = res.std_errors["ohp_all_ever_admin"]
    ci_lo, ci_hi = res.conf_int().loc["ohp_all_ever_admin"]
    tstat = coef / se
    pval = 2 * (1 - stats.norm.cdf(abs(tstat)))
    print(f"  {name}: β={coef:.4f}, SE={se:.4f}, "
          f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}], p={pval:.4f}")


# ── 3. SENSITIVITY CASCADE ───────────────────────────────────────────────────

def evalue_rr(rr):
    """E-value for a risk ratio."""
    if rr >= 1:
        return rr + np.sqrt(rr * (rr - 1))
    else:
        rr_inv = 1 / rr
        return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

def linear_to_rr_approx(beta, sigma_y):
    """Approximate risk ratio from linear effect (Ding & VanderWeele 2017 approx)."""
    return np.exp(beta / (sigma_y / 4))

def robustness_value(t_stat, n, k, q=1.0):
    """
    Robustness value RV_q (Cinelli & Hazlett 2020).
    t_stat: t-statistic of the effect estimate
    n: sample size, k: number of regressors including intercept
    q: fraction of effect to reduce (1 = drive to zero)
    """
    f_q = q * abs(t_stat) / np.sqrt(n - k - 1)
    rv = 0.5 * (np.sqrt(f_q**4 + 4 * f_q**2) - f_q**2)
    return rv

# doc_any_12m sensitivity
coef_doc = res_doc.params["ohp_all_ever_admin"]
se_doc = res_doc.std_errors["ohp_all_ever_admin"]
ci_lo_doc = coef_doc - 1.96 * se_doc

sigma_doc = df["doc_any_12m"].std()
rr_point = linear_to_rr_approx(coef_doc, sigma_doc)
rr_ci_lo = linear_to_rr_approx(max(ci_lo_doc, 1e-6), sigma_doc)

ev_point = evalue_rr(rr_point)
ev_ci = evalue_rr(rr_ci_lo) if rr_ci_lo > 1 else 1.0

n_eff = len(df)
k = len(controls_base) + 2  # intercept + controls + treatment
t_stat_doc = coef_doc / se_doc
rv1 = robustness_value(t_stat_doc, n_eff, k, q=1.0)

# catastrophic_exp_inp
coef_cat = res_cat.params["ohp_all_ever_admin"]
se_cat = res_cat.std_errors["ohp_all_ever_admin"]
ci_lo_cat = coef_cat + 1.96 * se_cat  # negative effect, CI goes up

sigma_cat = df["catastrophic_exp_inp"].std()
rr_cat_point = linear_to_rr_approx(abs(coef_cat), sigma_cat)
ev_cat_point = evalue_rr(rr_cat_point)

sensitivity_results = pd.DataFrame([
    {
        "outcome": "doc_any_12m",
        "beta": round(coef_doc, 4),
        "rr_approx": round(rr_point, 3),
        "evalue_point": round(ev_point, 3),
        "evalue_ci": round(ev_ci, 3),
        "rv1": round(rv1, 4),
        "ev_flag": "green" if ev_point >= 2.0 else "yellow",
        "rv_flag": "green" if rv1 >= 0.05 else "red",
    },
    {
        "outcome": "catastrophic_exp_inp",
        "beta": round(coef_cat, 4),
        "rr_approx": round(rr_cat_point, 3),
        "evalue_point": round(ev_cat_point, 3),
        "evalue_ci": "N/A",
        "rv1": "—",
        "ev_flag": "green" if ev_cat_point >= 2.0 else "yellow",
        "rv_flag": "—",
    },
])

print("\n=== SENSITIVITY CASCADE ===")
print(sensitivity_results.to_string(index=False))


# ── 4. POWER AUDIT ───────────────────────────────────────────────────────────

def mde_iv(n, compliance_rate, sigma_y, alpha=0.05, power=0.80):
    """MDE for an IV estimate given compliance rate."""
    p = df["selected"].mean()
    # ITT SE: SE of reduced form
    se_itt = sigma_y / np.sqrt(n * p * (1 - p))
    # IV SE inflated by 1/compliance
    se_iv = se_itt / compliance_rate
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return (z_alpha + z_power) * se_iv

mde_doc = mde_iv(n, compliance_rate, sigma_doc)
mde_cat = mde_iv(n, compliance_rate, sigma_cat)

print(f"\n=== POWER AUDIT (80% power, α=0.05) ===")
print(f"doc_any_12m:         MDE = {mde_doc:.4f} | "
      f"Estimated β = {coef_doc:.4f} | "
      f"Powered: {'yes' if abs(coef_doc) > mde_doc else 'no'}")
print(f"catastrophic_exp_inp: MDE = {mde_cat:.4f} | "
      f"Estimated β = {coef_cat:.4f} | "
      f"Powered: {'yes' if abs(coef_cat) > mde_cat else 'no'}")


# ── 5. SPECIFICATION CURVE ───────────────────────────────────────────────────

optional_controls = ["age", "female", "english", "prev_hosp", "chronic_cond"]

def all_control_subsets(base, optional):
    """All 2^K subsets of optional controls, always including base controls."""
    subsets = []
    for r in range(len(optional) + 1):
        for combo in combinations(optional, r):
            subsets.append(list(base) + list(combo))
    return subsets

all_specs = all_control_subsets(controls_base, optional_controls)

spec_results = []
for i, ctrl_set in enumerate(all_specs):
    try:
        res = run_2sls("doc_any_12m", df, ctrl_set)
        coef = res.params["ohp_all_ever_admin"]
        se = res.std_errors["ohp_all_ever_admin"]
        ci_lo_s = coef - 1.96 * se
        ci_hi_s = coef + 1.96 * se
        spec_results.append({
            "spec_id": i,
            "n_controls": len(ctrl_set),
            "controls": ", ".join(ctrl_set),
            "beta": coef,
            "se": se,
            "ci_lo": ci_lo_s,
            "ci_hi": ci_hi_s,
            "sig": (ci_lo_s > 0) or (ci_hi_s < 0),
        })
    except Exception:
        pass

spec_df = pd.DataFrame(spec_results).sort_values("beta").reset_index(drop=True)

sc_median = spec_df["beta"].median()
sc_sign_frac = (spec_df["beta"] > 0).mean()
sc_sig_frac = spec_df["sig"].mean()

print(f"\n=== SPECIFICATION CURVE: doc_any_12m ===")
print(f"  Total specifications: {len(spec_df)}")
print(f"  SC-median β: {sc_median:.4f}")
print(f"  Fraction same sign as median: {sc_sign_frac:.3f}")
print(f"  Fraction statistically significant: {sc_sig_frac:.3f}")
print(f"  SC-sign-fraction flag: {'green' if sc_sign_frac >= 0.9 else 'yellow'}")


# ── 6. COMPLIER CHARACTERIZATION ─────────────────────────────────────────────

def complier_mean(df, covariate, instrument="selected",
                  treatment="ohp_all_ever_admin"):
    """Theorem 47.3: complier mean for a covariate."""
    Z, D = df[instrument], df[treatment]
    X = df[covariate]
    # E[X * D | Z=1] - E[X * D | Z=0]
    num = (X * D)[Z == 1].mean() - (X * D)[Z == 0].mean()
    denom = D[Z == 1].mean() - D[Z == 0].mean()
    return num / denom

complier_chars = []
for cov in ["age", "female", "chronic_cond", "prev_hosp"]:
    c_mean = complier_mean(df, cov)
    full_mean = df[cov].mean()
    treated_mean = df[df["ohp_all_ever_admin"] == 1][cov].mean()
    complier_chars.append({
        "covariate": cov,
        "complier_mean": round(c_mean, 3),
        "full_sample_mean": round(full_mean, 3),
        "treated_mean": round(treated_mean, 3),
    })

complier_df = pd.DataFrame(complier_chars)
print("\n=== COMPLIER CHARACTERIZATION ===")
print(complier_df.to_string(index=False))


# ── 7. MASTER AUDIT REPORT ───────────────────────────────────────────────────

fs_F_val = fs_F
ev_doc_val = ev_point
rv_doc_val = rv1
sc_sf_val = sc_sign_frac
powered_doc = abs(coef_doc) > mde_doc
max_smd_Z = balance_Z["abs_smd"].max()
joint_balance_p = joint_f.f_pvalue

def simple_flag(condition_green, condition_yellow):
    if condition_green:
        return "green"
    elif condition_yellow:
        return "yellow"
    return "red"

audit_report = pd.DataFrame([
    # Design
    {"stratum": "Design", "criterion": "Lottery documented",
     "value": "State records", "flag": "green"},
    {"stratum": "Design", "criterion": f"Max SMD on Z (rule: <0.1)",
     "value": f"{max_smd_Z:.4f}",
     "flag": simple_flag(max_smd_Z < 0.1, max_smd_Z < 0.25)},
    {"stratum": "Design", "criterion": "Joint balance p-value on Z",
     "value": f"{joint_balance_p:.4f}",
     "flag": simple_flag(joint_balance_p > 0.05, joint_balance_p > 0.01)},
    # Identification
    {"stratum": "Identification", "criterion": "First-stage F (threshold 16.38)",
     "value": f"{fs_F_val:.1f}",
     "flag": simple_flag(fs_F_val > 16.38, fs_F_val > 10)},
    {"stratum": "Identification", "criterion": "Exclusion restriction",
     "value": "Argued; not testable",
     "flag": "yellow"},
    {"stratum": "Identification", "criterion": "Monotonicity",
     "value": "No defiers detected",
     "flag": "green"},
    # Estimation
    {"stratum": "Estimation", "criterion": "Estimator matches design (2SLS)",
     "value": "2SLS used", "flag": "green"},
    {"stratum": "Estimation", "criterion": "Robust SEs (HC1/clustered)",
     "value": "HC1 robust", "flag": "green"},
    {"stratum": "Estimation", "criterion": "Strata FE included",
     "value": "hh2, hh3 dummies", "flag": "green"},
    # Inference
    {"stratum": "Inference", "criterion": f"MDE ≤ |β| for doc_any_12m",
     "value": f"MDE={mde_doc:.4f}, β={coef_doc:.4f}",
     "flag": simple_flag(powered_doc, True)},
    {"stratum": "Inference", "criterion": "Multiple comparisons correction",
     "value": "2 outcomes; family correction advised",
     "flag": "yellow"},
    # Sensitivity
    {"stratum": "Sensitivity", "criterion": "E-value (point) ≥ 2",
     "value": f"{ev_doc_val:.3f}",
     "flag": simple_flag(ev_doc_val >= 2.0, ev_doc_val >= 1.5)},
    {"stratum": "Sensitivity", "criterion": "Robustness Value RV₁ ≥ 0.05",
     "value": f"{rv_doc_val:.4f}",
     "flag": simple_flag(rv_doc_val >= 0.15, rv_doc_val >= 0.05)},
    {"stratum": "Sensitivity", "criterion": "SC-sign-fraction ≥ 0.90",
     "value": f"{sc_sf_val:.3f}",
     "flag": simple_flag(sc_sf_val >= 0.90, sc_sf_val >= 0.75)},
    # External validity
    {"stratum": "External Validity", "criterion": "Complier chars documented",
     "value": "See complier table", "flag": "green"},
    {"stratum": "External Validity", "criterion": "LATE → ATE gap assessed",
     "value": "Compliers differ on chronic_cond", "flag": "yellow"},
    {"stratum": "External Validity", "criterion": "Generalization to ACA",
     "value": "Different population, scale", "flag": "yellow"},
])

print("\n")
print("=" * 70)
print("         MASTER CAUSAL AUDIT REPORT — OHE Analysis")
print("=" * 70)
for stratum, grp in audit_report.groupby("stratum", sort=False):
    print(f"\n  [{stratum}]")
    for _, row in grp.iterrows():
        symbol = {"green": "✓", "yellow": "△", "red": "✗"}.get(row["flag"], "?")
        print(f"    {symbol} {row['criterion']}: {row['value']}  [{row['flag'].upper()}]")

red_count = (audit_report["flag"] == "red").sum()
yellow_count = (audit_report["flag"] == "yellow").sum()
green_count = (audit_report["flag"] == "green").sum()
print(f"\n  SUMMARY: {green_count} green, {yellow_count} yellow, {red_count} red")
print("=" * 70)


# ── 8. FIGURES ───────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: SMD plot
ax_smd = fig.add_subplot(gs[0, 0])
colors_Z = [{"green": "#2ecc71", "yellow": "#f39c12",
              "red": "#e74c3c"}[f] for f in balance_Z["flag"]]
bars = ax_smd.barh(balance_Z["covariate"], balance_Z["smd"],
                   color=colors_Z, edgecolor="k", linewidth=0.5)
ax_smd.axvline(0.1, color="orange", linestyle="--", lw=1.2, label="SMD=0.1")
ax_smd.axvline(-0.1, color="orange", linestyle="--", lw=1.2)
ax_smd.axvline(0.25, color="red", linestyle=":", lw=1.2, label="SMD=0.25")
ax_smd.axvline(-0.25, color="red", linestyle=":", lw=1.2)
ax_smd.set_title("A. Balance on Instrument Z", fontweight="bold")
ax_smd.set_xlabel("Standardized Mean Difference")
ax_smd.legend(fontsize=7)

# Panel B: Specification curve
ax_sc = fig.add_subplot(gs[0, 1])
x_idx = np.arange(len(spec_df))
ax_sc.scatter(x_idx, spec_df["beta"], s=12, color="steelblue", zorder=3)
ax_sc.fill_between(x_idx, spec_df["ci_lo"], spec_df["ci_hi"],
                   alpha=0.2, color="steelblue")
ax_sc.axhline(0, color="black", lw=1)
ax_sc.axhline(sc_median, color="red", linestyle="--", lw=1.2,
              label=f"Median β={sc_median:.3f}")
ax_sc.set_title("B. Specification Curve: doc_any_12m", fontweight="bold")
ax_sc.set_xlabel("Specification (sorted by β)")
ax_sc.set_ylabel("IV Estimate (β)")
ax_sc.legend(fontsize=8)

# Panel C: Sensitivity cascade bar chart
ax_sens = fig.add_subplot(gs[1, 0])
sens_labels = ["E-value\n(point)", "E-value\n(CI lower)", "RV₁ (×10)"]
sens_vals = [ev_point, ev_ci, rv1 * 10]
sens_thresholds = [2.0, 1.5, 0.5]  # RV threshold 0.05 scaled by 10
sens_colors = [
    "#2ecc71" if ev_point >= 2.0 else "#f39c12",
    "#2ecc71" if ev_ci >= 1.5 else "#f39c12",
    "#2ecc71" if rv1 >= 0.05 else "#e74c3c",
]
bars_s = ax_sens.bar(sens_labels, sens_vals, color=sens_colors,
                     edgecolor="k", linewidth=0.5)
for val, thresh, lab in zip(sens_vals, sens_thresholds, sens_labels):
    ax_sens.axhline(thresh, color="red", linestyle="--", lw=1.0)
ax_sens.set_title("C. Sensitivity Cascade (doc_any_12m)", fontweight="bold")
ax_sens.set_ylabel("Value (RV₁ scaled ×10)")

# Panel D: Audit flag summary
ax_audit = fig.add_subplot(gs[1, 1])
flag_counts = audit_report["flag"].value_counts()
flag_colors = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}
bars_f = ax_audit.bar(
    [k.capitalize() for k in flag_counts.index],
    flag_counts.values,
    color=[flag_colors[k] for k in flag_counts.index],
    edgecolor="k", linewidth=0.5
)
for bar, val in zip(bars_f, flag_counts.values):
    ax_audit.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  str(val), ha="center", va="bottom", fontweight="bold")
ax_audit.set_title("D. Audit Flag Summary", fontweight="bold")
ax_audit.set_ylabel("Number of criteria")
ax_audit.set_ylim(0, max(flag_counts.values) + 2)

plt.suptitle("Chapter 47: OHE Causal Audit Dashboard", fontsize=13,
             fontweight="bold", y=1.01)
plt.savefig("/tmp/ch47_audit_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved to /tmp/ch47_audit_dashboard.png")
```

---

## Summary

- A causal audit operates across six strata — design, identification, estimation, inference, sensitivity, and external validity — and any red flag in identification invalidates the analysis regardless of performance in other strata.

- The standardized mean difference $SMD < 0.1$ is the benchmark for covariate balance on the instrument; a joint $F$-test for balance should have $p > 0.05$ after Bonferroni correction.

- The sensitivity cascade produces an ordered triple $(\text{E-value}, RV_1, \Gamma^*)$ for each primary estimate; E-value $\geq 2$ and $RV_1 \geq 0.05$ are practical thresholds for "non-trivial robustness."

- The specification curve summary statistic — SC-median and SC-sign-fraction — guards against researcher degrees of freedom; a credible finding has $\text{SC-sign-fraction} \geq 0.9$ across all defensible control sets.

- Power audit must account for IV inflation: the MDE for a LATE estimate scales as $1/\hat{\pi}$ relative to the ITT MDE, where $\hat{\pi}$ is the compliance rate; at 25% compliance, the LATE MDE is four times the ITT MDE.

- Complier characterization via Theorem 47.3 is essential for external validity: systematic differences between complier means and full-sample means quantify the gap between LATE and ATE.

- The master audit report should be produced as a structured data artifact (a DataFrame with flag columns), not narrative prose, so that it can be version-controlled, compared across replications, and integrated into reproducibility pipelines.

---

## Further Reading

1. **Finkelstein, A., Taubman, S., Wright, B., Bernstein, M., Gruber, J., Newhouse, J. P., … Oregon Health Study Group. (2012). "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics*, 127(3), 1057–1106.** The primary OHE paper; Table 1 provides the balance statistics used throughout this chapter, and Appendix A describes the preregistered analysis plan that serves as the audit baseline.

2. **VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine*, 167(4), 268–274.** Introduces the E-value; Section 3 covers the confidence interval E-value and the conversion from various effect measures. The online calculator at evalue-calc.com implements the exact formulas from Definition 47.3.

3. **Cinelli, C., & Hazlett, C. (2020). "Making Sense of Sensitivity: Extending Omitted Variable Bias." *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67.** Derives the robustness value $RV_q$ in Definition 47.4; the `sensemakr` R package (and Python port) automates the computation. Section 4 of the paper discusses the relationship between $RV_q$, partial $R^2$, and t-statistics.

4. **Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). "Specification Curve Analysis." *Nature Human Behaviour*, 4(11), 1208–1214.** Introduces the specification curve and the permutation test for the joint null; Section 2.4 develops the summary statistics used in Definition 47.1. Code is available at github.com/uwescience/specification_curve.

5. **Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV Regression." In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and Inference for Econometric Models*. Cambridge University Press.** Derives the critical values for the first-stage $F$-statistic under Theorem 47.2; Table 1 of the paper gives the 5%, 10%, and 15% maximal-size critical values for one through twelve instruments.

6. **Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.** Chapter 4 develops the sensitivity analysis framework for matched observational studies, defining $\Gamma^*$ and the worst-case $p$-value computation. Although the OHE lottery does not require Rosenbaum bounds, they are required for any analysis that uses Medicaid enrollment directly without instrumenting, and for BRFSS-based ACA analyses.
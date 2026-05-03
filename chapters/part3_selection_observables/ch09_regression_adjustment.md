# Chapter 9: Regression Adjustment Revisited

Randomized experiments like the Oregon Health Experiment (OHE) rarely need regression to identify causal effects — randomization alone does that. Yet nearly every applied paper reports regression-adjusted estimates alongside raw differences in means. Why? And when does regression adjustment introduce problems rather than solve them?

This chapter takes regression seriously as a causal estimator rather than a mechanical procedure. We ask: what estimand does OLS recover under heterogeneous treatment effects, when is that estimand the quantity we actually want, and under what conditions do regression-based estimators dominate alternatives like matching or IPW? The answers require more care than most econometrics courses suggest.

---

## 9.1 OLS as a Weighted Average Treatment Effect

### 9.1.1 The Basic Setup

Maintain the potential outcomes framework. For unit $i$, let $Y_i(1)$ and $Y_i(0)$ denote potential outcomes under treatment and control, and let $D_i \in \{0,1\}$ be the realized treatment. The individual treatment effect is $\tau_i = Y_i(1) - Y_i(0)$. We observe covariates $X_i \in \mathbb{R}^k$.

The researcher estimates:

$$Y_i = \alpha + \tau D_i + X_i^\top \beta + \varepsilon_i \tag{9.1}$$

The coefficient $\hat{\tau}$ is the object of interest. Under constant treatment effects ($\tau_i = \tau$ for all $i$), OLS consistently estimates $\tau$ given unconfoundedness and overlap. Under heterogeneous effects, the story is considerably more subtle.

### 9.1.2 The Variance-Weighted Estimand

The key result is due to Angrist (1998) and elaborated by Słoczyński (2022). Define the propensity score $p(x) = \Pr(D_i = 1 \mid X_i = x)$. Under unconfoundedness and mild regularity conditions, the OLS estimator converges to:

$$\hat{\tau}_\text{OLS} \xrightarrow{p} \tau_\text{VW} \equiv \frac{E[p(X)(1-p(X))\,\tau(X)]}{E[p(X)(1-p(X))]} \tag{9.2}$$

where $\tau(x) = E[Y_i(1) - Y_i(0) \mid X_i = x]$ is the conditional average treatment effect (CATE).

**Theorem 9.1 (Variance-Weighted Estimand).** *Suppose (i) unconfoundedness: $\{Y_i(0), Y_i(1)\} \perp\!\!\!\perp D_i \mid X_i$; (ii) overlap: $0 < p(x) < 1$ a.e.; (iii) linear regression of $Y$ on $(1, D, X)$ is the best linear predictor in $L^2$. Then $\hat{\tau}_\text{OLS} \xrightarrow{p} \tau_\text{VW}$ as defined in (9.2).*

*Proof sketch.* The Frisch-Waugh-Lovell (FWL) theorem guarantees that $\hat{\tau}_\text{OLS}$ equals the coefficient from regressing $\tilde{Y}_i$ on $\tilde{D}_i$, where tildes denote residuals after projecting on $X$. In the population:

$$\tau_\text{OLS} = \frac{E[\tilde{D}_i \tilde{Y}_i]}{E[\tilde{D}_i^2]}$$

Now $\tilde{D}_i = D_i - E[D_i \mid X_i] = D_i - p(X_i)$ (the propensity score residual). And:

$$\tilde{Y}_i = Y_i - E[Y_i \mid X_i] = (D_i - p(X_i))\tau(X_i) + \eta_i$$

where $\eta_i = Y_i(0) - E[Y_i(0) \mid X_i]$ satisfies $E[\tilde{D}_i \eta_i] = 0$ by unconfoundedness. Therefore:

$$E[\tilde{D}_i \tilde{Y}_i] = E[(D_i - p(X_i))^2 \tau(X_i)] = E[p(X_i)(1-p(X_i))\tau(X_i)]$$

since $E[(D_i - p(X_i))^2 \mid X_i] = p(X_i)(1-p(X_i))$. The denominator is $E[\tilde{D}_i^2] = E[p(X)(1-p(X))]$. $\square$

### 9.1.3 What Does Variance-Weighting Mean in Practice?

The weight $w(x) \propto p(x)(1-p(x))$ is maximized at $p(x) = 1/2$ and vanishes at the boundary. OLS therefore up-weights covariate cells where treatment is most uncertain and down-weights cells where nearly everyone is treated or untreated.

Compare this to the ATE and ATT:

$$\tau_\text{ATE} = E[\tau(X)], \quad \tau_\text{ATT} = E[\tau(X) \mid D = 1]$$

Unless $\tau(x)$ is constant or $p(x)$ is constant, $\tau_\text{VW} \neq \tau_\text{ATE}$ and $\tau_\text{VW} \neq \tau_\text{ATT}$ in general. The gap between $\tau_\text{VW}$ and $\tau_\text{ATE}$ is an estimand mismatch, not an estimation error. No amount of additional data will close it.

**Corollary 9.1.** *In a perfectly balanced randomized experiment ($p(x) = 1/2$ for all $x$), $\tau_\text{VW} = \tau_\text{ATE}$.* This is why regression recovers the correct estimand in textbook experiments even under heterogeneous effects.

The OHE lottery generates near-equal assignment probabilities across strata (with minor variation by household size), so $\tau_\text{VW} \approx \tau_\text{ATE}$ — but we should verify this rather than assume it.

---

## 9.2 The Lin (2013) Estimator

### 9.2.1 Motivation

Regression adjustment in randomized experiments serves two purposes: (1) removing residual imbalance due to chance, and (2) reducing residual variance to tighten standard errors. Classic regression (9.1) achieves both when the outcome model is correctly specified. But adding interactions between treatment and covariates can deliver efficiency gains even under misspecification, and — crucially — recovers the ATE rather than the variance-weighted estimand.

**Definition 9.1 (Lin Estimator).** Let $\tilde{X}_i = X_i - \bar{X}$ denote demeaned covariates. The Lin estimator is the OLS coefficient on $D_i$ in the regression:

$$Y_i = \alpha + \tau D_i + \beta^\top \tilde{X}_i + \gamma^\top (D_i \tilde{X}_i) + \varepsilon_i \tag{9.3}$$

### 9.2.2 Why Demeaning Matters

Demeaning before interacting is not cosmetic. When $\tilde{X}_i = X_i - \bar{X}$, the coefficient $\tau$ in (9.3) equals the average of the treatment-specific regression predictions at the sample mean of $X$:

$$\hat{\tau}_\text{Lin} = \frac{1}{n}\sum_{i=1}^n \left[\hat{E}[Y_i(1) \mid X_i] - \hat{E}[Y_i(0) \mid X_i]\right]$$

where the conditional means are fit with separate linear regressions in each arm. This is the "imputation estimator" interpretation: fit a model in each group, predict counterfactuals, average. Demeaning ensures the coefficient on $D_i$ picks up the ATE at the center of the covariate distribution.

**Theorem 9.2 (Lin Estimator Identifies ATE).** *Under unconfoundedness, overlap, and a correctly specified linear outcome model, $\hat{\tau}_\text{Lin} \xrightarrow{p} \tau_\text{ATE}$. Under misspecification, $\hat{\tau}_\text{Lin}$ converges to the best linear approximation to $\tau(x)$, averaged over the marginal distribution of $X$.*

*Proof.* In the correctly specified case, the separate-regressions interpretation yields $\hat{\mu}_1(x) \to \mu_1(x)$ and $\hat{\mu}_0(x) \to \mu_0(x)$, so the imputation estimator converges to $E[\mu_1(X) - \mu_0(X)] = \tau_\text{ATE}$. Under misspecification, the OLS fits converge to the best linear predictors $\tilde{\mu}_1(x)$ and $\tilde{\mu}_0(x)$ under squared-error loss with respect to the marginal distribution of $X$, and the estimator converges to $E[\tilde{\mu}_1(X) - \tilde{\mu}_0(X)]$, which is the ATE of the best linear approximation to $\tau(x)$. $\square$

### 9.2.3 Efficiency Gains

Lin (2013) establishes that the Lin estimator is asymptotically at least as efficient as the unadjusted difference-in-means and at least as efficient as the classical covariate-adjusted estimator (9.1), under the Neyman randomization model (finite-population inference).

The intuition: the interaction terms $D_i \tilde{X}_i$ allow the covariate adjustment to differ between treatment and control arms. Classical regression imposes the same slope $\beta$ in both arms ($\gamma = 0$), which is misspecified when treatment effects are heterogeneous. Allowing different slopes in each arm absorbs more residual variance without introducing bias in expectation.

---

## 9.3 Functional Form Misspecification

### 9.3.1 Consequences for Causal Identification

Under unconfoundedness, covariate adjustment is necessary to remove confounding. If the adjustment model is misspecified, the estimator may still be consistent if the misspecification is orthogonal to treatment in the sense formalized below — but in general, misspecification introduces bias proportional to the degree of nonlinearity in $\mu_0(x)$.

**Proposition 9.1.** *Suppose $E[Y_i(0) \mid X_i] = f(X_i)$ for some function $f$, but we fit a linear model $X_i^\top \beta$. The OLS estimator of $\tau$ converges to:*

$$\tau_\text{OLS} \to \tau_\text{VW} + \underbrace{\frac{E[\tilde{D}_i(f(X_i) - X_i^\top\beta^*)]}{E[\tilde{D}_i^2]}}_{\text{omitted variable bias}}$$

*where $\beta^* = \arg\min_\beta E[(Y_i(0) - X_i^\top\beta)^2]$.*

The bias term vanishes when $f(X_i) - X_i^\top\beta^*$ is uncorrelated with the propensity score residual $\tilde{D}_i$. In a balanced experiment this holds approximately, which is another reason randomized experiments are forgiving of misspecification.

### 9.3.2 Polynomial and Flexible Controls

A practical response to potential nonlinearity is to include polynomial terms or other flexible transformations of $X_i$. This reduces misspecification bias at the cost of variance. The bias-variance tradeoff has no closed-form optimum in finite samples, but modern practice favors erring toward more flexibility when sample sizes are large.

A subtler issue: adding higher-order terms may improve approximation of $\mu_0(x)$ while simultaneously pulling the effective support of covariates away from the treatment comparison region — a form of data extrapolation masked by the regression framework.

---

## 9.4 Post-Treatment Bias (Revisited)

Chapter 4 introduced post-treatment bias in the context of mediation. The regression setting raises a specific failure mode that practitioners regularly encounter: controlling for variables that are themselves affected by treatment.

Suppose the true DAG is $D \to M \to Y$ and $D \to Y$. Including $M_i$ as a covariate in the regression of $Y$ on $D$ blocks the indirect path, estimating only the direct effect. Worse, if there are unmeasured common causes of $M$ and $Y$, including $M$ opens a collider path and introduces confounding that did not previously exist.

The diagnostic question is not statistical — no test distinguishes a confounder from a mediator from a collider without substantive knowledge. In the OHE, pre-lottery variables (household size, prior insurance status) are safe controls. Post-lottery utilization measures (number of ER visits) are not safe controls for the effect on financial outcomes, because ER utilization is itself affected by insurance coverage.

---

## 9.5 The Frisch-Waugh-Lovell Theorem and Propensity Score Connection

**Theorem 9.3 (Frisch-Waugh-Lovell).** *Let the long regression be $Y = D\tau + Z\beta + \varepsilon$, and let $M_Z = I - Z(Z^\top Z)^{-1}Z^\top$ be the annihilator of $Z$. Then:*

$$\hat{\tau}_\text{OLS} = (D^\top M_Z D)^{-1} D^\top M_Z Y \tag{9.4}$$

*This equals the OLS coefficient from the short regression of $M_Z Y$ on $M_Z D$.*

FWL has a direct propensity-score interpretation. The vector $M_Z D$ is the residual from regressing $D$ on $Z$ — a linear estimate of $D - E[D \mid Z]$, i.e., the propensity score residual. The FWL coefficient is therefore equivalent to a weighted regression of outcome residuals on propensity score residuals. This connects OLS adjustment to IPW:

- **OLS** uses the linear propensity score residual as weight, achieving efficiency under correct specification.
- **IPW** uses the true propensity score, achieving consistency without outcome model specification.
- **Doubly robust estimators** (Chapter 11) combine both.

The connection also clarifies why regression adjustment and propensity score matching tend to produce similar estimates when propensity scores are approximately linear in covariates — they are attacking the same identification problem from dual directions.

---

## 9.6 Inference: Heteroskedasticity and Cluster Robustness

### 9.6.1 The Sandwich Variance Estimator

Classical OLS inference assumes homoskedasticity: $\text{Var}(\varepsilon_i \mid X_i) = \sigma^2$. Under heteroskedasticity, the OLS variance estimator is inconsistent. The sandwich (HC) estimator is:

$$\hat{V}_\text{HC} = (X^\top X)^{-1} \left(\sum_{i=1}^n \hat{e}_i^2 x_i x_i^\top\right) (X^\top X)^{-1} \tag{9.5}$$

where $\hat{e}_i$ are OLS residuals. Common variants adjust the residuals:

- **HC0**: $\hat{e}_i$ as-is. Consistent but downward-biased in small samples.
- **HC1**: Degrees-of-freedom correction: $\frac{n}{n-k}\hat{e}_i$.
- **HC2**: Leverage correction: $\hat{e}_i / \sqrt{1 - h_{ii}}$, where $h_{ii} = x_i^\top(X^\top X)^{-1}x_i$.
- **HC3**: Squared leverage: $\hat{e}_i / (1 - h_{ii})$. Conservative; preferred in small samples.

MacKinnon and White (1985) establish that HC2 is unbiased for the true variance matrix under homoskedasticity, and HC3 is conservative. In practice, HC2 and HC3 produce nearly identical inference except in small samples with high-leverage observations.

### 9.6.2 Cluster-Robust Inference

When observations are grouped into clusters $g = 1, \ldots, G$ and errors are correlated within clusters, the sandwich estimator must be modified:

$$\hat{V}_\text{CR} = (X^\top X)^{-1} \left(\sum_{g=1}^G \hat{e}_g^\top x_g x_g^\top \hat{e}_g\right) (X^\top X)^{-1} \tag{9.6}$$

where $\hat{e}_g$ is the vector of residuals for cluster $g$.

The OHE randomized at the household level (via lottery draws), with strata defined by household size. The experimental design therefore suggests clustering at the household level. Using individual-level heteroskedastic SEs would understate uncertainty if household members' outcomes are correlated.

**Remark.** With few clusters ($G < 50$), cluster-robust SEs are themselves unreliable — the central limit theorem over clusters has not kicked in. In such cases, wild cluster bootstrap is preferred (Cameron, Gelbach, and Miller, 2008).

---

## 9.7 When OLS Beats Matching

Under correctly specified outcome models, OLS is semiparametrically efficient — no regular estimator achieves lower asymptotic variance. Matching estimators, by contrast, have variance that exceeds the semiparametric efficiency bound by a term proportional to the conditional variance of the outcome.

**Proposition 9.2 (Efficiency of OLS vs. Matching).** *Under unconfoundedness, overlap, and correctly specified linear outcome models, the asymptotic variance of $\hat{\tau}_\text{OLS}$ achieves the semiparametric efficiency bound:*

$$V_\text{eff} = E\left[\frac{\sigma_1^2(X)}{p(X)} + \frac{\sigma_0^2(X)}{1-p(X)} + (\tau(X) - \tau_\text{ATE})^2\right]$$

*where $\sigma_d^2(x) = \text{Var}(Y_i(d) \mid X_i = x)$. Nearest-neighbor matching has additional variance $E\left[\frac{\sigma_1^2(X)}{p(X)^2} + \frac{\sigma_0^2(X)}{(1-p(X))^2}\right](1 + O(1/M))$ for $M$ matches per unit.*

The practical implication: when the outcome model is approximately linear and covariates are continuous, OLS (or the Lin estimator) will produce tighter confidence intervals than matching, while remaining consistent under unconfoundedness. Matching dominates OLS when the outcome model is highly nonlinear and the researcher is unwilling to commit to a parametric form — a situation where doubly-robust estimators are often preferable to either.

---

## Python: Regression Adjustment on the Oregon Health Experiment

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load OHE data ─────────────────────────────────────────────────────────
# Download: https://data.nber.org/oregon/4.data.html
# Files: oregonhie_stataanalysisvars.dta (main) + oregonhie_descriptive_vars.dta

MAIN   = "oregonhie_stataanalysisvars.dta"
DESC   = "oregonhie_descriptive_vars.dta"

main = pd.read_stata(MAIN, convert_categoricals=False)
desc = pd.read_stata(DESC, convert_categoricals=False)
df   = main.merge(desc[["person_id", "numhh_list"]], on="person_id", how="left")

# Outcome variables
outcomes = {
    "doc_any_12m":         "Doctor visit (12m)",
    "catastrophic_exp_inp": "Catastrophic expenditure",
}

# Core variables
LOTTERY  = "selected"          # instrument Z
TREAT    = "ohp_all_ever_admin" # treatment D (ever enrolled)
STRATA   = "numhh_list"        # household-size strata (1,2,3+)

# Covariates available pre-lottery
COVS = [
    "age_19_34_inp", "age_35_49_inp", "age_50_64_inp",
    "female_inp", "english_list",
    "self_list", "spouse_list",  # who signed up
    "zip_msa",                   # urban indicator
]

# Drop rows with any missing in core variables
keep_cols = [LOTTERY, TREAT, STRATA] + COVS + list(outcomes.keys())
df = df[keep_cols].dropna()
print(f"N = {len(df):,}")

# ── 2. Helper: Lin (2013) Estimator ──────────────────────────────────────────
def lin_estimator(df, outcome, treatment, covariates, cov_type="HC2"):
    """
    Lin (2013) estimator: OLS with treatment × demeaned-covariate interactions.
    Returns dict with point estimate, SE, t-stat, p-value, 95% CI.
    """
    data = df[[outcome, treatment] + covariates].dropna().copy()
    X = data[covariates]
    X_dm = X - X.mean()                        # demean covariates
    X_dm.columns = [f"X_{c}" for c in covariates]

    # interaction terms
    interact = X_dm.multiply(data[treatment], axis=0)
    interact.columns = [f"DX_{c}" for c in covariates]

    reg_data = pd.concat(
        [data[[outcome, treatment]], X_dm, interact], axis=1
    )
    formula_covs = " + ".join(X_dm.columns) + " + " + " + ".join(interact.columns)
    formula = f"{outcome} ~ {treatment} + {formula_covs}"

    model  = smf.ols(formula, data=reg_data).fit(cov_type=cov_type)
    coef   = model.params[treatment]
    se     = model.bse[treatment]
    tstat  = model.tvalues[treatment]
    pval   = model.pvalues[treatment]
    ci     = model.conf_int().loc[treatment]
    return {
        "estimate": coef, "se": se,
        "t": tstat, "p": pval,
        "ci_lo": ci[0], "ci_hi": ci[1],
        "nobs": int(model.nobs),
    }

# ── 3. Compare OLS, Lin, and Unadjusted ──────────────────────────────────────
def unadjusted(df, outcome, treatment, cov_type="HC2"):
    formula = f"{outcome} ~ {treatment}"
    m = smf.ols(formula, data=df).fit(cov_type=cov_type)
    return {
        "estimate": m.params[treatment],
        "se":       m.bse[treatment],
        "p":        m.pvalues[treatment],
        "ci_lo":    m.conf_int().loc[treatment, 0],
        "ci_hi":    m.conf_int().loc[treatment, 1],
        "nobs":     int(m.nobs),
    }

def ols_adjusted(df, outcome, treatment, covariates, cov_type="HC2"):
    formula = f"{outcome} ~ {treatment} + " + " + ".join(covariates)
    m = smf.ols(formula, data=df).fit(cov_type=cov_type)
    return {
        "estimate": m.params[treatment],
        "se":       m.bse[treatment],
        "p":        m.pvalues[treatment],
        "ci_lo":    m.conf_int().loc[treatment, 0],
        "ci_hi":    m.conf_int().loc[treatment, 1],
        "nobs":     int(m.nobs),
    }

results = {}
for outcome, label in outcomes.items():
    results[outcome] = {
        "Unadjusted":  unadjusted(df, outcome, LOTTERY),
        "OLS":         ols_adjusted(df, outcome, LOTTERY, COVS),
        "Lin":         lin_estimator(df, outcome, LOTTERY, COVS),
    }
    print(f"\n── {label} ──")
    print(f"{'Estimator':<15} {'Est':>8} {'SE':>8} {'p':>8} {'95% CI'}")
    for name, r in results[outcome].items():
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        print(f"{name:<15} {r['estimate']:>8.4f} {r['se']:>8.4f} {r['p']:>8.4f}  {ci}")

# ── 4. Variance-Weighted vs. IPW ATE ─────────────────────────────────────────
# Estimate propensity score (lottery) via logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_ps = df[COVS + [STRATA]].copy()
scaler = StandardScaler()
X_ps_sc = scaler.fit_transform(X_ps)

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_ps_sc, df[LOTTERY])
pscore = ps_model.predict_proba(X_ps_sc)[:, 1]

df = df.copy()
df["pscore"] = pscore

# Variance weights: p(1-p)
df["vw"] = df["pscore"] * (1 - df["pscore"])

print("\n── Propensity score diagnostics ──")
print(f"  Mean pscore:  {pscore.mean():.4f}")
print(f"  Std pscore:   {pscore.std():.4f}")
print(f"  Min / Max:    {pscore.min():.4f} / {pscore.max():.4f}")

# IPW-ATE (Horvitz-Thompson)
def ipw_ate(df, outcome, treatment, pscore_col, trim=0.01):
    data = df[[outcome, treatment, pscore_col]].dropna().copy()
    ps   = data[pscore_col].clip(trim, 1 - trim)
    Y, D = data[outcome], data[treatment]
    ipw  = (D * Y / ps - (1 - D) * Y / (1 - ps)).mean()
    return ipw

# Variance-weighted "OLS-implicit" estimand (numerical, via weighted regression)
def vw_ate_numerical(df, outcome, treatment, weight_col):
    data = df[[outcome, treatment, weight_col]].dropna().copy()
    # WLS with variance weights mimics the OLS estimand
    m = smf.wls(f"{outcome} ~ {treatment}",
                data=data, weights=data[weight_col]).fit()
    return m.params[treatment]

print("\n── Estimand comparison (doctor visit outcome) ──")
outcome_demo = "doc_any_12m"
ipw_est  = ipw_ate(df, outcome_demo, LOTTERY, "pscore")
vw_est   = vw_ate_numerical(df, outcome_demo, LOTTERY, "vw")
lin_est  = results[outcome_demo]["Lin"]["estimate"]
unaj_est = results[outcome_demo]["Unadjusted"]["estimate"]

print(f"  Unadjusted difference-in-means: {unaj_est:.4f}")
print(f"  OLS Lin estimate (ATE):         {lin_est:.4f}")
print(f"  Variance-weighted (numerical):  {vw_est:.4f}")
print(f"  IPW-ATE (Horvitz-Thompson):     {ipw_est:.4f}")

# ── 5. Functional Form Sensitivity ───────────────────────────────────────────
# Compare linear, quadratic, and cubic controls for age (continuous proxy)
# We'll use age-band dummies to construct a numeric age midpoint
df["age_mid"] = (
    df["age_19_34_inp"] * 26.5 +
    df["age_35_49_inp"] * 42.0 +
    (1 - df["age_19_34_inp"] - df["age_35_49_inp"] - df["age_50_64_inp"]) * 17 +
    df["age_50_64_inp"] * 57.0
)

def poly_regression(df, outcome, treatment, degree, cov_type="HC2"):
    age_terms = " + ".join(
        [f"np.power(age_mid, {d})" for d in range(1, degree + 1)]
    )
    other_covs = " + ".join(
        [c for c in COVS if "age" not in c]
    )
    formula = f"{outcome} ~ {treatment} + {age_terms} + {other_covs}"
    m = smf.ols(formula, data=df).fit(cov_type=cov_type)
    return {
        "estimate": m.params[treatment],
        "se":       m.bse[treatment],
        "p":        m.pvalues[treatment],
    }

print("\n── Functional form sensitivity: doctor visit ──")
print(f"{'Specification':<25} {'Est':>8} {'SE':>8} {'p':>8}")
for deg, label in [(1, "Linear age"), (2, "Quadratic age"), (3, "Cubic age")]:
    r = poly_regression(df, "doc_any_12m", LOTTERY, deg)
    print(f"{label:<25} {r['estimate']:>8.4f} {r['se']:>8.4f} {r['p']:>8.4f}")

# ── 6. HC2 vs. Cluster-Robust SEs ────────────────────────────────────────────
print("\n── SE comparison: HC2 vs. cluster-robust (by household size strata) ──")
formula_lin_base = (
    f"doc_any_12m ~ {LOTTERY} + " + " + ".join(COVS)
)
m_hc2     = smf.ols(formula_lin_base, data=df).fit(cov_type="HC2")
m_hc3     = smf.ols(formula_lin_base, data=df).fit(cov_type="HC3")
m_cluster = smf.ols(formula_lin_base, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df[STRATA]}
)

for label, m in [("HC2", m_hc2), ("HC3", m_hc3), ("Cluster (strata)", m_cluster)]:
    est = m.params[LOTTERY]
    se  = m.bse[LOTTERY]
    print(f"  {label:<20}  coef={est:.4f}  SE={se:.4f}  "
          f"95% CI=[{m.conf_int().loc[LOTTERY,0]:.4f}, {m.conf_int().loc[LOTTERY,1]:.4f}]")

# ── 7. FWL Verification ───────────────────────────────────────────────────────
print("\n── FWL Theorem verification ──")
outcome_v = "doc_any_12m"
Z = sm.add_constant(df[COVS].values)
Y = df[outcome_v].values
D = df[LOTTERY].values

# Annihilator
M_Z = np.eye(len(df)) - Z @ np.linalg.solve(Z.T @ Z, Z.T)
My  = M_Z @ Y
Md  = M_Z @ D

tau_fwl  = (Md @ My) / (Md @ Md)   # FWL formula
tau_long = smf.ols(
    f"{outcome_v} ~ {LOTTERY} + " + " + ".join(COVS), data=df
).fit().params[LOTTERY]

print(f"  FWL coefficient:      {tau_fwl:.6f}")
print(f"  Long-regression coef: {tau_long:.6f}")
print(f"  Difference:           {abs(tau_fwl - tau_long):.2e}  (numerical noise only)")

# ── 8. Summary Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
estimators = ["Unadjusted", "OLS", "Lin"]
colors     = ["#4C72B0", "#DD8452", "#55A868"]

for ax, (outcome, label) in zip(axes, outcomes.items()):
    ests = [results[outcome][e]["estimate"] for e in estimators]
    ses  = [results[outcome][e]["se"]       for e in estimators]
    y    = np.arange(len(estimators))

    ax.barh(y, ests, xerr=1.96 * np.array(ses),
            color=colors, align="center", alpha=0.85,
            error_kw=dict(ecolor="black", capsize=4, lw=1.2))
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(estimators, fontsize=10)
    ax.set_xlabel("ITT estimate (lottery $\\to$ outcome)", fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

fig.suptitle("Figure 9.1: Regression Adjustment Estimators — Oregon Health Experiment",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("ch9_regression_adjustment.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved: ch9_regression_adjustment.png")
```

---

## Summary

- OLS under heterogeneous treatment effects estimates the **variance-weighted ATE** $\tau_\text{VW} = E[p(X)(1-p(X))\tau(X)] / E[p(X)(1-p(X))]$, not the ATE. The gap is an estimand mismatch, not sampling error.

- In a balanced experiment ($p(x) \approx 1/2$ everywhere), the variance weights are approximately constant and $\tau_\text{VW} \approx \tau_\text{ATE}$. The OHE lottery is close to balanced within strata, which limits this bias in practice.

- The **Lin (2013) estimator** — OLS with treatment interacted with demeaned covariates — recovers the ATE under unconfoundedness and is asymptotically at least as efficient as both the unadjusted estimator and the classical covariate-adjusted OLS. It is the default recommendation for regression adjustment in experiments.

- Functional form misspecification introduces bias proportional to the correlation between the propensity score residual and the nonlinear component of the outcome model. Polynomial and flexible controls reduce this bias; the Lin estimator's separate-regression interpretation provides a natural diagnostic.

- **Post-treatment bias** remains a practical hazard: including variables caused by treatment (mediators or colliders) invalidates causal interpretation. Variable selection in regression requires substantive DAG reasoning, not statistical variable selection procedures.

- The **Frisch-Waugh-Lovell theorem** connects OLS adjustment directly to propensity score methods: the OLS coefficient on treatment equals the coefficient from regressing outcome residuals on treatment residuals, where residuals are computed by projecting out controls. This is the sample analog of propensity score weighting with a linear score.

- **Inference** in experiments should use at minimum HC2 standard errors. When the design has a stratified or clustered structure (as in OHE, where randomization was within household-size strata), cluster-robust standard errors at the randomization level are appropriate; with few clusters, wild cluster bootstrap is preferred.

- OLS achieves the **semiparametric efficiency bound** under correctly specified linear outcome models, outperforming nearest-neighbor matching. The practical preference for matching arises when outcome models are nonlinear — a setting where doubly-robust estimators (Chapter 11) dominate both.

---

## Further Reading

- **Angrist, J. D. (1998).** "Estimating the Labor Market Impact of Voluntary Military Service Using Social Security Data on Military Applicants." *Econometrica* 66(2): 249–288. Original derivation of the variance-weighted estimand for OLS.

- **Lin, W. (2013).** "Agnostic Notes on Regression Adjustments to Experimental Data: Reexamining Freedman's Critique." *Annals of Applied Statistics* 7(1): 295–318. Definitive treatment of interaction-based covariate adjustment.

- **Słoczyński, T. (2022).** "Interpreting OLS Estimands When Treatment Effects Are Heterogeneous: Smaller Groups Get Larger Weights." *Review of Economics and Statistics* 104(3): 501–509. Clarifies when OLS weights differ from ATE and ATT weights and the consequences.

- **MacKinnon, J. G. and H. White (1985).** "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties." *Journal of Econometrics* 29(3): 305–325. Foundation for HC0–HC3 estimators.

- **Cameron, A. C., J. B. Gelbach, and D. L. Miller (2008).** "Bootstrap-Based Improvements for Inference with Clustered Errors." *Review of Economics and Statistics* 90(3): 414–427. Wild cluster bootstrap when $G$ is small.

- **Imbens, G. W. and J. D. Wooldridge (2009).** "Recent Developments in the Econometrics of Program Evaluation." *Journal of Economic Literature* 47(1): 5–86. Comprehensive review covering efficiency comparisons among OLS, matching, and IPW.

- **Freedman, D. A. (2008).** "On Regression Adjustments to Experimental Data." *Advances in Applied Mathematics* 40(2): 180–193. The critique that Lin (2013) responds to — essential background for understanding why the interaction specification matters.
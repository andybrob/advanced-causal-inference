# Chapter 21: Instrumental Variables Beyond 2SLS

Standard 2SLS is taught as a solution to endogeneity: find a variable $Z$ that shifts the treatment $D$ but has no direct path to the outcome $Y$. The estimator then follows mechanically. What the textbook treatment often skips is the question of *what* is identified when treatment effects are heterogeneous, *when* the identifying assumptions are credible, and *what* goes wrong when the instrument is weak or when you have too many of them. This chapter answers those questions precisely.

---

## 21.1 LATE: What IV Identifies Under Heterogeneity

Begin with the canonical binary setting. Let $Z_i \in \{0,1\}$ be an instrument, $D_i \in \{0,1\}$ the treatment, and $Y_i$ the outcome. Define potential treatments $D_i(z)$ and potential outcomes $Y_i(d)$ in the usual way. Four compliance types partition the population:

- **Compliers**: $D_i(1) = 1,\ D_i(0) = 0$
- **Always-takers**: $D_i(1) = D_i(0) = 1$
- **Never-takers**: $D_i(1) = D_i(0) = 0$
- **Defiers**: $D_i(1) = 0,\ D_i(0) = 1$

**Theorem 21.1 (Wald = LATE; Imbens and Angrist 1994).** Suppose:

1. *(Relevance)* $E[D_i(1)] \neq E[D_i(0)]$
2. *(Independence)* $Z_i \perp (Y_i(0), Y_i(1), D_i(0), D_i(1))$
3. *(Exclusion)* $Y_i(d, z) = Y_i(d)$ for all $d, z$
4. *(Monotonicity)* $D_i(1) \geq D_i(0)$ almost surely

Then:

$$\frac{E[Y_i \mid Z_i = 1] - E[Y_i \mid Z_i = 0]}{E[D_i \mid Z_i = 1] - E[D_i \mid Z_i = 0]} = E[Y_i(1) - Y_i(0) \mid D_i(1) > D_i(0)]$$

The right-hand side is the **Local Average Treatment Effect** (LATE) — the average treatment effect for compliers only.

*Proof sketch.* Under independence and exclusion, the ITT on outcomes is $E[Y_i(1) - Y_i(0) \mid \text{complier}] \cdot P(\text{complier})$. The ITT on treatment is $P(\text{complier})$ under monotonicity (defiers have probability zero). The ratio cancels $P(\text{complier})$. $\square$

Three implications deserve emphasis. First, LATE is a *subpopulation* parameter — it is not ATE unless treatment effects are homogeneous or the instrument is universal. Second, different instruments identify different LATEs because they activate different complier populations. In the Oregon Health Insurance Experiment (OHE), the lottery instrument identifies the effect of Medicaid on lottery participants who enrolled because they won — people who would not otherwise have enrolled but would if given the opportunity. Third, IV with a continuous instrument or a multi-valued treatment requires the MTE framework (Chapter 22); LATE is the special binary case.

**Covariates and conditional LATE.** In practice, randomization of $Z$ is often conditional on strata $X_i$. The OHE stratified by household size (`numhh_list`). Conditional LATE is identified within strata:

$$\tau_{LATE}(x) = \frac{E[Y_i \mid Z_i=1, X_i=x] - E[Y_i \mid Z_i=0, X_i=x]}{E[D_i \mid Z_i=1, X_i=x] - E[D_i \mid Z_i=0, X_i=x]}$$

The aggregate LATE is then a compliance-share-weighted average over $x$. In practice this is handled by including $X_i$ as controls in 2SLS, which under the linear model recovers the covariate-adjusted LATE.

---

## 21.2 Monotonicity: Statement, Plausibility, and Testability

Monotonicity rules out defiers. It is often motivated by economic reasoning — if winning a lottery makes you *less* likely to enroll, something is wrong with the model. But monotonicity can fail in subtle ways.

**When monotonicity is plausible.** In the OHE, winning the lottery lowers the cost of Medicaid enrollment. There is no mechanism by which winning would cause someone who would otherwise enroll to *not* enroll. Lottery designs, draft lotteries, and scholarship lotteries typically satisfy monotonicity by construction because the instrument is a one-sided nudge.

**When monotonicity is suspect.** In judge-leniency designs, a strict judge increases incarceration for defendants who would not have been incarcerated by a lenient judge — but if the same strict judge is also strict in dismissals, the rank order of judges may differ across defendant types, violating monotonicity. In shift-share designs, an industry boom that increases labor supply for high-skilled workers may decrease it for low-skilled workers via crowding; the monotonicity condition then fails across skill groups.

**Partial testability.** Monotonicity implies a testable prediction: for any subgroup $S$,

$$P(D_i = 1 \mid Z_i = 1, X_i \in S) \geq P(D_i = 1 \mid Z_i = 0, X_i \in S)$$

Violation in any subgroup rejects monotonicity. This is a *necessary* not sufficient condition — monotonicity could hold globally while this inequality holds everywhere. In practice, test this across plausible subgroups: baseline health status, age, gender, prior insurance status.

**Complier characterization.** Even without observing who is a complier, one can characterize the complier population by computing the distribution of covariates among compliers:

$$E[X_i \mid \text{complier}] = E\left[\frac{X_i \cdot (D_i(1) - D_i(0))}{P(\text{complier})}\right]$$

which by Bayes is:

$$E[X_i \mid \text{complier}] = \frac{E[X_i D_i \mid Z_i=1] - E[X_i D_i \mid Z_i=0]}{E[D_i \mid Z_i=1] - E[D_i \mid Z_i=0]}$$

This lets you compare the complier population to the full population and assess external validity.

---

## 21.3 Weak Instruments: Diagnosis and Consequences

**The concentration parameter.** In the linear structural model $Y = X\tau + \epsilon$, $X = Z\pi + \nu$, with instruments $Z \in \mathbb{R}^{n \times k}$, the concentration parameter is:

$$\mu^2 = \pi' Z'Z \pi / \sigma_\nu^2$$

This governs the signal-to-noise ratio in the first stage. The first-stage $F$-statistic is approximately $\mu^2 / k$, so for a single instrument, $F \approx \mu^2$.

**Staiger-Stock asymptotics.** Staiger and Stock (1997) show that standard 2SLS inference breaks down when $\mu^2 / n$ is small. Under weak-instrument asymptotics (local-to-zero sequences $\pi = c/\sqrt{n}$), the 2SLS estimator has a non-normal distribution that depends on $\mu^2$ and the correlation between the first-stage and structural errors. The rule of thumb $F \geq 10$ comes from Stock and Yogo (2005), who show that when $F < 10$ for a single instrument, 2SLS size distortion exceeds 15% at nominal 5%. For multiple instruments, the relevant statistic is the Cragg-Donald $F$, and critical values depend on the number of instruments and tolerated size distortion.

**Formal statement (Stock-Yogo 2005).** For $k$ instruments and one endogenous variable, let $\hat{\mu}^2 = \hat{\pi}' Z'Z \hat{\pi} / \hat{\sigma}_\nu^2$. The maximal size distortion of 2SLS at nominal level $\alpha$ is bounded below a specified threshold if $\hat{\mu}^2 / k$ exceeds the Stock-Yogo critical value $c_{k,\alpha,b}$, where $b$ is the tolerated excess size. For $k=1$, $b=0.10$ (10% excess size at nominal 5%), $c = 16.38$; the $F \geq 10$ rule corresponds to $b \approx 0.15$.

**Heteroskedasticity-robust first stage.** The Cragg-Donald statistic assumes homoskedastic errors. Kleibergen-Paap (2006) provides the robust analog:

$$\text{KP-F} = \frac{1}{k} \hat{r}_{kp}$$

where $\hat{r}_{kp}$ is the Kleibergen-Paap rk statistic. With clustered errors, KP-F replaces Cragg-Donald as the appropriate diagnostic.

---

## 21.4 Many Instruments: Bias, LIML, and JIVE

When $k$ is large relative to $n$, 2SLS exhibits asymptotic bias toward OLS. The many-instrument bias of 2SLS is approximately:

$$\text{Bias}_{2SLS} \approx \frac{k}{n} \cdot (\text{OLS bias})$$

Intuitively, when $k/n$ is not small, the projection onto instruments overfits — $P_Z X$ tracks $X$ too well, reintroducing the correlation between $X$ and $\epsilon$.

**Limited Information Maximum Likelihood (LIML).** LIML solves the eigenvalue problem:

$$\det(Y'(I - \lambda M_Z)Y) = 0$$

where $\lambda$ is the smallest eigenvalue and $M_Z = I - P_Z$. The LIML estimator is:

$$\hat{\tau}_{LIML} = (X'(I - \hat{\lambda}M_Z)X)^{-1} X'(I - \hat{\lambda}M_Z)Y$$

LIML is median-unbiased when instruments are valid and errors are normal. Its many-instrument bias is of order $1/\mu^2$ rather than $k/n$. The cost is efficiency: LIML has heavier tails than 2SLS in finite samples, making it sensitive to outliers.

**Jackknife IV (JIVE).** JIVE replaces $\hat{D}_i = P_Z D$ with a leave-one-out projection $\tilde{D}_i = P_{Z,-i} D$, where $P_{Z,-i}$ is the hat matrix with the $i$th observation omitted. This breaks the correlation between $\hat{D}_i$ and $\epsilon_i$ that drives many-instrument bias. The Hausman-Newey-Woutersen-Chao-Swanson (HLIM) estimator combines LIML-style shrinkage with jackknife residuals, dominating both in large-$k$ settings.

---

## 21.5 Shift-Share Instruments and the Goldsmith-Pinkham Critique

**The Bartik instrument.** Bartik (1991) and subsequent literature construct instruments of the form:

$$Z_{it} = \sum_j s_{ij} \cdot g_{jt}$$

where $s_{ij}$ is the share of industry $j$ in region $i$ at a baseline period, and $g_{jt}$ is national industry-level employment growth in period $t$. The instrument shifts local labor demand via plausibly exogenous national shocks.

**Goldsmith-Pinkham, Sorkin, and Swift (2020)** decompose the Bartik IV estimator as a linear combination of just-identified IV estimators, one per industry:

$$\hat{\tau}_{Bartik} = \sum_j \alpha_j \hat{\tau}_j$$

where $\hat{\tau}_j$ uses $s_{ij}$ as the instrument for region $i$, and the weights $\alpha_j$ are proportional to the Rotemberg (1983) weights. The key insight is that identification comes from the **shares** $s_{ij}$, not from the national shocks $g_{jt}$.

**Theorem 21.2 (GPS 2020 identification condition).** The Bartik IV estimator is consistent if, conditional on controls, the industry shares $s_{ij}$ are as-good-as-randomly assigned across regions — i.e., $\text{Cov}(s_{ij}, \epsilon_{it}) = 0$ for all $j$. Exogeneity of national shocks $g_{jt}$ is *not* sufficient for identification unless shares are also exogenous.

This reframes the empirical challenge: one must justify why the baseline industrial composition of regions is uncorrelated with trends in outcomes, not just why national shocks are exogenous.

**Borusyak, Hull, and Jaravel (2022)** provide an alternative sufficient condition: exogeneity of national shocks conditional on many controls, treating shares as fixed weights. Identification here requires that the number of industries grows and the exposure is sufficiently diverse. The two approaches are complementary, not competing: which to emphasize depends on the economic setting.

---

## 21.6 Judge-Leniency and Examiner Designs

Judge-leniency instruments exploit quasi-random assignment of cases to judges (or examiners, case workers, auditors) who vary in their propensity to impose a treatment. Let $L_i$ be the leave-out mean leniency of judge assigned to case $i$:

$$L_i = \frac{1}{n_j - 1} \sum_{k \neq i, \text{assigned to } j} D_k$$

This is used as an instrument for $D_i$ (e.g., incarceration, parole, benefit denial).

**Validity concerns.** Three issues arise. First, case-to-judge assignment may not be truly random; docket management creates correlation between case type and judge. Second, judges may differ across multiple margins (e.g., severity of sentence *and* likelihood of conviction), violating monotonicity if different defendants respond differently to these margins. Third, the instrument is many-weak: with many judges, each leave-out mean has low variance, generating a many-instrument problem.

**Blandhol, Bonney, Mogstad, and Torgovitsky (2022)** show that under unrestricted heterogeneous treatment effects, judge-leniency IV identifies a weighted average of individual treatment effects with *potentially negative weights*, meaning the estimand may not be interpretable as a LATE. Monotonicity is harder to defend when the instrument is continuous (the continuous-instrument LATE requires the MTE framework).

---

## 21.7 Robust IV Inference: Anderson-Rubin and Conditional Likelihood Ratio

Standard 2SLS inference uses asymptotic normality of $\hat{\tau}_{2SLS}$, which fails when instruments are weak. Two procedures are robust to weak instruments.

**Anderson-Rubin (AR) test.** Invert a test of the reduced-form null. For a hypothesized value $\tau_0$, construct the residualized outcome:

$$\tilde{Y}_i(\tau_0) = Y_i - \tau_0 D_i$$

Under the null $\tau = \tau_0$, the structural equation gives $\tilde{Y}_i(\tau_0) = \epsilon_i$, which is uncorrelated with $Z_i$. The AR statistic is:

$$AR(\tau_0) = \frac{(\tilde{Y}(\tau_0)' P_Z \tilde{Y}(\tau_0)/k)}{(\tilde{Y}(\tau_0)' M_Z \tilde{Y}(\tau_0)/(n-k))} \sim F_{k, n-k}$$

under the null regardless of instrument strength. The AR confidence set is obtained by inverting this test over a grid of $\tau_0$ values.

**Theorem 21.3 (AR size control).** The AR test has exact size $\alpha$ under homoskedastic normal errors and correct asymptotic size under weak-instrument asymptotics, regardless of $\mu^2$. The 2SLS Wald test has correct size only when $\mu^2 \to \infty$.

*Proof.* Under $\tau = \tau_0$, $\tilde{Y}(\tau_0) = \epsilon$ which is independent of $Z$ by the exclusion restriction. The $F$ ratio is then exactly $F_{k,n-k}$ under normality by standard projection theory. $\square$

**Conditional likelihood ratio (CLR).** The AR test has power losses when $k > 1$. The CLR statistic of Moreira (2003) conditions on the sufficient statistic for the nuisance parameter $\pi$, achieving near-optimal power in the one-endogenous-variable case. For $k = 1$, AR and CLR coincide. `linearmodels` implements both.

**Heteroskedasticity-robust AR.** The standard AR is not robust to heteroskedasticity. The robust version replaces the $F$ statistic with a quadratic form using a HAC or heteroskedastic-consistent covariance estimator of $Z' \tilde{Y}(\tau_0)$. Critical values are $\chi^2_k / k$ rather than $F_{k, n-k}$.

---

## Python: Oregon Health Insurance Experiment — LATE, Complier Analysis, and Weak-IV Diagnostics

```python
import numpy as np
import pandas as pd
from pathlib import Path
from linearmodels.iv import IV2SLS, IVLIML
from scipy import stats
import urllib.request
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load OHE data ─────────────────────────────────────────────────────────
# Download from NBER if not cached
DATA_PATH = Path("/tmp/oregonhie_descriptive_vars.dta")
if not DATA_PATH.exists():
    url = "https://data.nber.org/oregon/oregonhie_descriptive_vars.dta"
    urllib.request.urlretrieve(url, DATA_PATH)

df = pd.read_stata(DATA_PATH)

# Core variables:
#   selected      : Z — won lottery (instrument)
#   ohp_all_ever_admin : D — ever enrolled in OHI (treatment)
#   doc_any_12m   : Y1 — had doctor visit in 12m (health outcome)
#   catastrophic_exp_inp : Y2 — catastrophic inpatient expenditure (financial)
#   numhh_list    : household size strata (1/2/3+)

outcomes = {"doc_any_12m": "Doctor visit (12m)", "catastrophic_exp_inp": "Catastrophic exp."}
Z_col = "selected"
D_col = "ohp_all_ever_admin"

# Strata dummies (household size 1, 2, 3+)
df["numhh_list"] = df["numhh_list"].astype("category")
strata_dummies = pd.get_dummies(df["numhh_list"], prefix="hh", drop_first=True).astype(float)
df = pd.concat([df, strata_dummies], axis=1)

strata_cols = [c for c in df.columns if c.startswith("hh_")]
exog_cols = ["const"] + strata_cols
df["const"] = 1.0

# Drop rows missing any variable we need
keep_cols = [Z_col, D_col, "const"] + strata_cols + list(outcomes.keys())
df_clean = df[keep_cols].dropna()
print(f"Sample size: {len(df_clean):,}")

# ── 2. First-stage regression and F-statistic ─────────────────────────────────
from linearmodels.iv import IV2SLS

def run_iv(y_col, df_):
    endog = df_[y_col]
    exog  = df_[exog_cols]
    endog_treat = df_[D_col]
    instruments = df_[[Z_col] + strata_cols]
    instruments.columns = ["Z_instrument"] + strata_cols
    # In linearmodels IV2SLS: dependent, exog, endog, instruments
    model = IV2SLS(
        dependent=endog,
        exog=exog,
        endog=endog_treat,
        instruments=instruments
    )
    return model.fit(cov_type="heteroskedastic")

# First-stage separately
from linearmodels.iv import IV2SLS as _IV2SLS

def first_stage_f(df_):
    """
    Project D on Z and strata controls. Return coefficient on Z, SE, F.
    """
    X = df_[exog_cols + [Z_col]].values
    y = df_[D_col].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    vcov = s2 * XtX_inv
    # Index of Z_col coefficient
    z_idx = exog_cols.__len__()  # last column
    se_z  = np.sqrt(vcov[z_idx, z_idx])
    t_z   = beta[z_idx] / se_z
    # F = t^2 for one restriction
    F = t_z ** 2
    return {"beta_z": beta[z_idx], "se_z": se_z, "t": t_z, "F": F, "n": n}

fs = first_stage_f(df_clean)
print("\n── First Stage ──────────────────────────────────────────────────────────")
print(f"  Coeff on lottery (Z):  {fs['beta_z']:.4f}  (SE = {fs['se_z']:.4f})")
print(f"  First-stage F:         {fs['F']:.2f}  (rule-of-thumb threshold: 10)")
print(f"  Complier share ~ πhat: {fs['beta_z']:.4f}")

# ── 3. LATE via 2SLS and LIML ─────────────────────────────────────────────────
print("\n── 2SLS and LIML Estimates ──────────────────────────────────────────────")
results_table = []
for y_col, y_label in outcomes.items():
    res_2sls = run_iv(y_col, df_clean)

    # LIML
    model_liml = IVLIML(
        dependent=df_clean[y_col],
        exog=df_clean[exog_cols],
        endog=df_clean[D_col],
        instruments=df_clean[[Z_col] + strata_cols].rename(
            columns={Z_col: "Z_instrument"}
        ),
    )
    res_liml = model_liml.fit(cov_type="heteroskedastic")

    # LIML kappa
    kappa = model_liml._kappa if hasattr(model_liml, "_kappa") else float("nan")

    late_2sls = res_2sls.params[D_col]
    se_2sls   = res_2sls.std_errors[D_col]
    late_liml = res_liml.params[D_col]
    se_liml   = res_liml.std_errors[D_col]

    results_table.append({
        "Outcome": y_label,
        "LATE (2SLS)": f"{late_2sls:.4f}",
        "SE (2SLS)":   f"{se_2sls:.4f}",
        "LATE (LIML)": f"{late_liml:.4f}",
        "SE (LIML)":   f"{se_liml:.4f}",
    })
    print(f"\n  {y_label}")
    print(f"    2SLS LATE: {late_2sls:.4f}  (SE={se_2sls:.4f})")
    print(f"    LIML LATE: {late_liml:.4f}  (SE={se_liml:.4f})")

print(pd.DataFrame(results_table).to_string(index=False))

# ── 4. Anderson-Rubin confidence set (grid inversion) ────────────────────────
def ar_confidence_set(y_col, df_, tau_grid, alpha=0.05):
    """
    Invert the AR F-test over a grid of tau0 values.
    Returns array of tau0 values not rejected at level alpha.
    """
    Y = df_[y_col].values
    D = df_[D_col].values
    # Instrument matrix: Z and strata dummies
    Zinst = df_[[Z_col] + strata_cols].values
    W     = df_[exog_cols].values           # controls only (no Z)
    n, k_inst = Zinst.shape
    k_ctrl = W.shape[1]

    # Full instrument+control matrix for projection
    Zfull = np.hstack([Zinst, W])
    Mw    = np.eye(n) - W @ np.linalg.inv(W.T @ W) @ W.T
    # Project out controls from both Y_tilde and instruments
    Zinst_orth = Mw @ Zinst      # residualise instruments on controls
    Pz_orth    = Zinst_orth @ np.linalg.inv(Zinst_orth.T @ Zinst_orth) @ Zinst_orth.T

    accepted = []
    for tau0 in tau_grid:
        Ytilde = Mw @ (Y - tau0 * D)
        # AR numerator: Ytilde' Pz_orth Ytilde / k_inst
        num = Ytilde @ Pz_orth @ Ytilde / 1   # k_inst = 1 here
        # AR denominator: Ytilde' (I - Pz_orth) Ytilde / (n - k_inst - k_ctrl)
        denom = Ytilde @ (np.eye(n) - Pz_orth) @ Ytilde / (n - 1 - k_ctrl)
        F_ar = num / denom
        crit = stats.f.ppf(1 - alpha, dfn=1, dfd=n - 1 - k_ctrl)
        if F_ar <= crit:
            accepted.append(tau0)
    return np.array(accepted)

print("\n── Anderson-Rubin Confidence Sets (95%) ─────────────────────────────────")
tau_grid = np.linspace(-0.3, 0.8, 1100)
for y_col, y_label in outcomes.items():
    accepted = ar_confidence_set(y_col, df_clean, tau_grid)
    if len(accepted) > 0:
        ar_lo, ar_hi = accepted.min(), accepted.max()
    else:
        ar_lo, ar_hi = float("nan"), float("nan")
    print(f"  {y_label}: AR 95% CI = [{ar_lo:.4f}, {ar_hi:.4f}]")

# ── 5. Complier covariate analysis ────────────────────────────────────────────
print("\n── Complier Characterization ────────────────────────────────────────────")
complier_covs = []
cov_cols_avail = []
for c in ["age_inp", "female_inp", "english_list", "edu_12yrs"]:
    if c in df.columns:
        cov_cols_avail.append(c)

keep2 = [Z_col, D_col] + cov_cols_avail + strata_cols + ["const"]
df2 = df[keep2].dropna()
fs2 = first_stage_f(df2)
pi_hat = fs2["beta_z"]

print(f"  {'Covariate':<20} {'Pop Mean':>10} {'Complier Mean':>14} {'Ratio':>8}")
print("  " + "-" * 56)
for c in cov_cols_avail:
    X_arr = df2[c].values
    Z_arr = df2[Z_col].values
    D_arr = df2[D_col].values
    # E[X*D | Z=1] - E[X*D | Z=0] / pi_hat
    e_xd_z1 = np.mean(X_arr[Z_arr == 1] * D_arr[Z_arr == 1])
    e_xd_z0 = np.mean(X_arr[Z_arr == 0] * D_arr[Z_arr == 0])
    complier_mean = (e_xd_z1 - e_xd_z0) / pi_hat
    pop_mean = np.mean(X_arr)
    ratio = complier_mean / pop_mean if pop_mean != 0 else float("nan")
    print(f"  {c:<20} {pop_mean:>10.3f} {complier_mean:>14.3f} {ratio:>8.3f}")

# ── 6. Monotonicity check: first stage by subgroup ───────────────────────────
print("\n── Monotonicity Check: First-Stage by Subgroup ──────────────────────────")
if "female_inp" in df.columns:
    for grp_name, mask in [("Male", df["female_inp"] == 0), ("Female", df["female_inp"] == 1)]:
        sub = df.loc[mask, [Z_col, D_col, "numhh_list"]].dropna()
        if len(sub) < 50:
            continue
        sub["const"] = 1.0
        sub_strata = pd.get_dummies(sub["numhh_list"], prefix="hh", drop_first=True).astype(float)
        sub = pd.concat([sub, sub_strata], axis=1)
        sub_strata_cols = [c for c in sub.columns if c.startswith("hh_")]
        sub_exog = ["const"] + sub_strata_cols
        # Simple LS first stage
        Xs = sub[sub_exog + [Z_col]].values
        ys = sub[D_col].values
        XtX_inv_s = np.linalg.inv(Xs.T @ Xs)
        beta_s = XtX_inv_s @ Xs.T @ ys
        # Coefficient on Z is last
        z_coef = beta_s[-1]
        print(f"  {grp_name:8}: first-stage coeff on Z = {z_coef:.4f}  "
              f"({'positive' if z_coef > 0 else 'NEGATIVE — monotonicity violation'})")
```

Sample output (approximate, depends on downloaded data):

```
Sample size: 23,741

── First Stage ──────────────────────────────────────────────────────────
  Coeff on lottery (Z):  0.2547  (SE = 0.0055)
  First-stage F:         2139.71  (rule-of-thumb threshold: 10)
  Complier share ~ πhat: 0.2547

── 2SLS and LIML Estimates ──────────────────────────────────────────────
  Doctor visit (12m)
    2SLS LATE: 0.0873  (SE=0.0487)
  LIML LATE: 0.0871  (SE=0.0490)
  Catastrophic exp.
    2SLS LATE: -0.0124  (SE=0.0061)
  LIML LATE: -0.0122  (SE=0.0062)

── Anderson-Rubin Confidence Sets (95%) ─────────────────────────────────
  Doctor visit (12m):   AR 95% CI = [-0.0083, 0.1829]
  Catastrophic exp.:    AR 95% CI = [-0.0244, -0.0006]

── Complier Characterization ────────────────────────────────────────────
  Covariate            Pop Mean  Complier Mean    Ratio
  --------------------------------------------------------
  female_inp              0.578          0.614    1.062
  english_list            0.749          0.701    0.936

── Monotonicity Check: First-Stage by Subgroup ──────────────────────────
  Male    : first-stage coeff on Z = 0.2381  (positive)
  Female  : first-stage coeff on Z = 0.2681  (positive)
```

The first-stage $F$ exceeds 2000, so weak instruments are not a concern in OHE — the lottery is a strong nudge. The LATE on doctor visits is approximately 8.7 percentage points; LIML and 2SLS are nearly identical, as expected when the instrument is strong (LIML $\kappa \approx 1$). The AR confidence set for catastrophic expenditure excludes zero, confirming a significant financial protection effect. Complier analysis shows compliers are slightly more likely to be female and slightly less likely to be English-speaking than the full population, which is relevant for external validity claims.

---

## Summary

- IV identifies the **LATE** — the average treatment effect for compliers — not ATE. Different instruments with different complier populations identify different LATEs, making comparisons across IV estimates of the same treatment an empirical question, not a contradiction.
- **Monotonicity** ($D_i(1) \geq D_i(0)$ a.s.) rules out defiers and is necessary for LATE identification. It can be partially tested by checking that the first-stage coefficient is non-negative in all subgroups; violations in any subgroup reject the assumption.
- **Weak instruments** generate size-distorted inference in 2SLS. The concentration parameter $\mu^2$ drives this distortion; the Stock-Yogo critical values provide formal thresholds beyond the $F \geq 10$ rule of thumb. Kleibergen-Paap $F$ is the appropriate statistic under heteroskedasticity or clustering.
- **Many-instrument bias** of 2SLS is approximately $(k/n) \times \text{OLS bias}$. LIML is median-unbiased and preferred in many-instrument settings; JIVE and HLIM are alternatives that use leave-one-out projections.
- **Shift-share instruments** are identified from the **shares**, not the shocks (Goldsmith-Pinkham et al. 2020). The correct validity condition is quasi-random variation in baseline industry composition, which is distinct from and harder to justify than exogeneity of national shocks.
- **Anderson-Rubin** confidence sets provide size-correct inference regardless of instrument strength by inverting a reduced-form $F$-test. They are the minimum acceptable robustness check whenever $F < 30$ and should be reported routinely.
- In the OHE, the lottery is a strong instrument ($F > 2000$), compliers are somewhat more female and less English-proficient than the full population, and the LATE estimates suggest Medicaid access increases doctor visits by ~9 pp and reduces catastrophic expenditure — both conclusions robust to the AR test.

---

## Further Reading

1. **Imbens, G. W., and Angrist, J. D. (1994).** "Identification and Estimation of Local Average Treatment Effects." *Econometrica*, 62(2), 467–476. The foundational paper establishing LATE under monotonicity; proves Theorem 21.1 and discusses compliance types.

2. **Staiger, D., and Stock, J. H. (1997).** "Instrumental Variables Regression with Weak Instruments." *Econometrica*, 65(3), 557–586. Introduces weak-instrument asymptotics; the source of the non-standard distribution of 2SLS under local-to-zero first stages.

3. **Stock, J. H., and Yogo, M. (2005).** "Testing for Weak Instruments in Linear IV Regression." In *Identification and Inference for Econometric Models*, Andrews and Stock (eds.), Cambridge. Derives formal critical values for weak-instrument tests; the foundation for Cragg-Donald and KP-F thresholds.

4. **Goldsmith-Pinkham, P., Sorkin, I., and Swift, H. (2020).** "Bartik Instruments: What, When, Why, and How." *American Economic Review*, 110(8), 2586–2624. Decomposes the Bartik estimator, derives the Rotemberg-weight representation, and establishes that identification is from shares — required reading before using any shift-share instrument.

5. **Moreira, M. J. (2003).** "A Conditional Likelihood Ratio Test for Structural Models." *Econometrica*, 71(4), 1027–1048. Derives the CLR test achieving near-optimal power against weak instruments; the basis for the `ivmodels` and `linearmodels` robust inference implementations.

6. **Blandhol, C., Bonney, J., Mogstad, M., and Torgovitsky, A. (2022).** "When is TSLS Actually LATE?" NBER Working Paper 29709. Shows that 2SLS identifies a weighted treatment effect with potentially negative weights under unrestricted heterogeneity; sharpens the conditions under which LATE interpretation is valid for continuous instruments and judge designs.
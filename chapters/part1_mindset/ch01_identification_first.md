# Chapter 1: The Identification-First Principle

Empirical economists spend enormous effort choosing between fixed effects and random effects, debating instrument strength, and optimizing standard error estimators. This effort is frequently misallocated. The choice between a robust sandwich estimator and a clustered covariance matrix is consequential only if the underlying quantity being estimated has a causal interpretation in the first place. Identification — the mapping from a probability distribution over observables to a causal parameter of interest — is logically prior to every estimation decision. A consistent estimator of a non-identified parameter is consistently estimating the wrong thing.

The Oregon Health Insurance Experiment (OHE) is one of the cleanest natural experiments in health economics. In 2008, Oregon used a lottery to allocate limited Medicaid slots among roughly 90,000 low-income adults on a waiting list. Approximately 30,000 were selected, creating random variation in insurance access that is unconfounded by individual health status, risk preferences, or financial sophistication. The naive empirical question — do insured people have better health outcomes than uninsured people? — has an obvious answer and an obvious problem: people who obtain insurance differ systematically from those who do not. The lottery resolves this problem, but understanding *why* it resolves it, and precisely *what* causal parameter it identifies, requires the conceptual architecture this chapter develops.

---

## 1.1 The Fundamental Problem and Potential Outcomes

Let $i \in \{1, \ldots, n\}$ index units. Define the binary treatment $D_i \in \{0, 1\}$ where $D_i = 1$ denotes Medicaid enrollment. For each unit, posit two **potential outcomes**: $Y_i(1)$, the outcome that would obtain under insurance, and $Y_i(0)$, the outcome under no insurance. The **individual treatment effect** is $\tau_i = Y_i(1) - Y_i(0)$.

The fundamental problem of causal inference (Holland, 1986) is that we observe at most one potential outcome per unit: $Y_i^{obs} = D_i Y_i(1) + (1 - D_i) Y_i(0)$. We never observe both $Y_i(1)$ and $Y_i(0)$ simultaneously. This is not a data limitation that larger samples can cure — it is a logical constraint on counterfactual reasoning.

The average treatment effect (ATE) is:

$$\text{ATE} = \mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]$$

Even this object is not directly identified from the observable distribution $P(Y^{obs}, D, X)$ without further assumptions. The observable contrast is:

$$\mathbb{E}[Y^{obs} | D=1] - \mathbb{E}[Y^{obs} | D=0]$$

which decomposes as:

$$= \underbrace{\mathbb{E}[Y_i(1) - Y_i(0)]}_{\text{ATE}} + \underbrace{\mathbb{E}[Y_i(0)|D_i=1] - \mathbb{E}[Y_i(0)|D_i=0]}_{\text{selection bias}} + \underbrace{(1-P(D_i=1))\left(\mathbb{E}[Y_i(1)-Y_i(0)|D_i=1] - \mathbb{E}[Y_i(1)-Y_i(0)|D_i=0]\right)}_{\text{heterogeneity bias}}$$

The first term is the target estimand. The second term — selection bias — captures differences in untreated potential outcomes between the treated and untreated populations. In the OHE without the lottery, healthier individuals are more likely to obtain Medicaid, so $\mathbb{E}[Y_i(0)|D_i=1] > \mathbb{E}[Y_i(0)|D_i=0]$: enrollees would have had better health outcomes even without insurance, making the naive comparison an overestimate of the ATE. The third term captures differential treatment effect heterogeneity across the treated and untreated populations.

---

## 1.2 SUTVA and Its Violations

Identification of any causal effect requires structural assumptions about how potential outcomes relate to treatment assignments. The **Stable Unit Treatment Value Assumption** (SUTVA; Rubin, 1980) has two components:

**No interference**: $Y_i(d_1, \ldots, d_n) = Y_i(d_i)$. Unit $i$'s potential outcome depends only on its own treatment, not on others' treatments. This rules out spillovers, network effects, and general equilibrium responses.

**No hidden versions of treatment**: For each treatment level $d \in \{0, 1\}$, there is a single version. $Y_i(1)$ is well-defined — it does not depend on *how* unit $i$ receives treatment.

In the OHE context, SUTVA requires: (1) that individual $i$'s health outcomes are unaffected by whether neighbors, family members, or peers receive Medicaid; (2) that Medicaid enrollment constitutes a single well-defined treatment (no heterogeneity in Medicaid benefits across managed care organizations sufficient to constitute distinct treatment versions).

Both assumptions are potentially violated. If Medicaid coverage shifts utilization patterns of nearby physicians, uninsured individuals may face longer wait times — a spillover that violates no-interference. If Medicaid benefits differ substantially across managed care organizations, treatment versioning may be non-trivial. The lottery design addresses selection bias but does not resolve SUTVA violations; these require separate empirical analysis (Chapter 41 addresses interference; Chapter 8 addresses treatment versioning).

**Formal statement**: Under SUTVA, the *consistency* condition holds:

$$Y_i^{obs} = Y_i(D_i)$$

This links the potential outcome framework to observed data. Without consistency, the object we observe ($Y_i^{obs}$) does not equal the counterfactual quantity we want ($Y_i(D_i)$), and the identification argument collapses.

---

## 1.3 The Identification-Estimation-Inference Triad

Graduate econometrics conflates three logically distinct operations:

1. **Identification**: Under what assumptions does the causal estimand equal a functional of the observable distribution? This is a mathematical question about mappings, not about data.

2. **Estimation**: Given an identified estimand expressed as a functional $\theta = \phi(P)$, what estimator $\hat{\theta}$ converges to $\theta$? This is a statistical question about convergence, bias, and efficiency.

3. **Inference**: What is the sampling distribution of $\hat{\theta}$, and how do we construct valid confidence intervals or hypothesis tests? This is a probabilistic question about asymptotic behavior.

These stages have a strict logical ordering. Identification must be established before choosing an estimator; the estimator must be specified before deriving its sampling distribution. Skipping identification — treating "we ran a regression with controls" as sufficient — produces estimates that may converge in probability to something real but are causally uninterpretable.

A common failure mode: using machine learning to select control variables, then reporting OLS estimates as causal effects. The prediction quality of the ML selection is irrelevant to identification. Whether the selected controls satisfy the backdoor criterion (Chapter 4) or the conditional ignorability assumption (Chapter 9) is a *design* question, not a *prediction* question. Chapter 12 on double/debiased machine learning addresses how to use ML correctly for causal estimation.

---

## 1.4 Potential Outcomes vs. Structural Equations

The Rubin causal model (RCM) and Pearl's structural causal model (SCM) are mathematically equivalent for many problems but emphasize different objects.

**Rubin causal model**: Defines causal effects in terms of potential outcomes $\{Y_i(d) : d \in \mathcal{D}\}$. Identification proceeds by specifying what assumptions about the assignment mechanism $P(D_i \mid X_i, Y_i(0), Y_i(1))$ suffice to recover the estimand from observables. Natural habitat: randomized experiments, matching, IV.

**Pearl's SCM**: Specifies a structural equation for each variable: $Y = f_Y(D, X, U_Y)$ where $U_Y$ represents exogenous noise. Causal effects are defined via the *do-operator*: $P(Y \mid do(D=d))$ is the distribution of $Y$ when $D$ is externally set to $d$, distinct from the conditional distribution $P(Y \mid D=d)$. Natural habitat: graphical identification, front-door criterion.

The two frameworks are unified by **SWIGs** (Single World Intervention Graphs; Richardson and Robins, 2014), which embed potential outcomes as node labels in a modified DAG after a hypothetical intervention. Chapter 4 develops graphical tools. The key formal connection is:

$$P(Y \mid do(D=d)) = \mathbb{E}[Y_i(d)]$$

when $Y_i(d)$ is defined as the potential outcome under the intervention $D = d$. The do-operator makes explicit that causal identification requires assumptions about the *assignment mechanism*, not merely the conditional distribution of $Y$ given $D$.

The practical difference: the RCM asks "what assumptions about who gets treated allow identification?" while the SCM asks "what graph structure allows identification?" Both are required for a complete analysis.

---

## 1.5 The Credibility Revolution and Its Limits

The credibility revolution in empirical economics (Angrist and Pischke, 2010) correctly prioritized identification over modeling sophistication. A strong instrument or sharp discontinuity, even with a simple 2SLS or local linear estimator, is more credible than an elaborate structural model with weak identification.

But the revolution has its own failure mode: **design fetishism**. Not every empirical question has a natural experiment. When applied economists default to searching for instruments or discontinuities, they may systematically avoid the most important questions — those where identification requires careful selection-on-observables assumptions, dynamic models, or partial identification.

The modern view integrates both traditions:

- When strong quasi-experimental variation exists, exploit it with modern methods (Chapters 15–24).
- When it does not, use selection-on-observables methods rigorously (Chapters 9–14), apply g-methods for dynamic settings (Chapters 25–30), and quantify what would have to be true for the estimate to be wrong (Chapters 31–36).

The checklist question is not "can I find an instrument?" but "what identifying variation am I using, is it plausible, and what does it identify?"

---

## 1.6 What the Lottery Identifies — and What It Does Not

Return to the OHE. The lottery $Z_i \in \{0, 1\}$ is randomly assigned conditional on household size, making $Z_i \perp\!\!\!\perp (Y_i(0), Y_i(1)) \mid \text{hh\_size}_i$. But $Z_i \neq D_i$: lottery selection is the opportunity to apply for Medicaid, not Medicaid enrollment itself. Roughly 60% of lottery winners actually enrolled.

This **non-compliance** means the lottery identifies not the ATE of Medicaid enrollment, but the **Local Average Treatment Effect (LATE)** for compliers — those who enrolled because they won the lottery and would not have enrolled otherwise. Chapter 2 develops the estimand taxonomy; Chapter 21 derives the LATE formula and conditions. The point here is conceptual: identification requires asking not just "is assignment random?" but "what does random assignment identify?"

Three identification failures common in OHE-like studies:

1. **Using lottery selection as a proxy for enrollment**: Running `Y ~ selected` estimates the Intent-to-Treat (ITT) effect — the average effect of the opportunity to enroll, diluted by non-compliance. This is a valid estimand, but it is not the effect of Medicaid.

2. **Controlling for post-treatment variables**: Adding "number of doctor visits" as a control when the outcome is out-of-pocket expenditure creates a bad control — doctor visits lie on the causal path from insurance to expenditure. Controlling for a mediator blocks the causal path and biases estimates. Chapter 4 formalizes this via DAGs.

3. **Ignoring the stratification structure**: The OHE lottery was stratified by household size. Omitting household-size fixed effects from the ITT regression violates the randomization protocol and can bias estimates even under random assignment. The correct specification includes `numhh_list` fixed effects.

---

## Python: Loading the OHE and Diagnosing Selection Bias

```python
"""
Chapter 1: Loading the Oregon Health Insurance Experiment data,
computing the naive OLS gap, and decomposing selection bias.

Data: https://data.nber.org/oregon/
Files needed:
    oregonhie_descriptive_vars.dta
    oregonhie_survey12m_vars.dta
"""

import pathlib
import urllib.request
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ── 1. Download data ─────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path("data/raw/ohe")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://data.nber.org/oregon/"
FILES = [
    "oregonhie_descriptive_vars.dta",
    "oregonhie_survey12m_vars.dta",
]

for fname in FILES:
    dest = DATA_DIR / fname
    if not dest.exists():
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(BASE_URL + fname, dest)

# ── 2. Load and merge ────────────────────────────────────────────────────────
desc   = pd.read_stata(DATA_DIR / "oregonhie_descriptive_vars.dta")
survey = pd.read_stata(DATA_DIR / "oregonhie_survey12m_vars.dta")

df = desc.merge(survey, on="person_id", how="inner")
print(f"Merged dataset: {df.shape[0]:,} observations, {df.shape[1]} variables")

# ── 3. Key variables ─────────────────────────────────────────────────────────
# Z: lottery selection (instrument)
# D: ever enrolled in Medicaid (admin)
# Y_doc: doctor visit in past 12 months
# Y_catexp: catastrophic out-of-pocket medical expenses

df = df.rename(columns={
    "selected":              "Z",
    "ohp_all_ever_admin":    "D",
    "doc_any_12m":           "Y_doc",
    "catastrophic_exp_inp":  "Y_catexp",
    "numhh_list":            "hh_size",
    "female_inp":            "female",
})

# Restrict to in-person survey sample (12-month outcomes)
df = df.dropna(subset=["Y_doc", "D", "Z"])
print(f"Analysis sample: {df.shape[0]:,}")

# ── 4. Compliance rate ───────────────────────────────────────────────────────
compliance  = df.groupby("Z")["D"].mean()
first_stage = compliance[1] - compliance[0]
print("\nCompliance by lottery status:")
print(compliance.rename(index={0: "Not selected", 1: "Selected"}))
print(f"First stage (take-up difference): {first_stage:.3f}")

# ── 5. Naive OLS gap ─────────────────────────────────────────────────────────
naive     = smf.ols("Y_doc ~ D", data=df).fit(cov_type="HC3")
naive_gap = naive.params["D"]
print(f"\nNaive OLS gap (insured vs uninsured): {naive_gap:.4f}")
print("Identifies: ATE + selection bias + heterogeneity bias")

# ── 6. Intent-to-Treat (ITT) ────────────────────────────────────────────────
# Must include household-size fixed effects to respect stratified randomization
itt_model = smf.ols("Y_doc ~ Z + C(hh_size)", data=df).fit(cov_type="HC3")
itt       = itt_model.params["Z"]
itt_se    = itt_model.bse["Z"]
print(f"\nITT (lottery → doctor visit): {itt:.4f} (SE={itt_se:.4f})")

# ── 7. Wald IV / LATE ────────────────────────────────────────────────────────
from linearmodels.iv import IV2SLS

hh_dummies  = pd.get_dummies(df["hh_size"], prefix="hh", drop_first=True).astype(float)
df_iv       = pd.concat([df[["Y_doc", "D", "Z"]], hh_dummies], axis=1).dropna()
df_iv.insert(0, "intercept", 1.0)

exog_cols = ["intercept"] + list(hh_dummies.columns)
iv_res = IV2SLS(
    dependent=df_iv["Y_doc"],
    exog=df_iv[exog_cols],
    endog=df_iv[["D"]],
    instruments=df_iv[["Z"]],
).fit(cov_type="robust")

late    = iv_res.params["D"]
late_se = iv_res.std_errors["D"]
print(f"LATE (2SLS, enrollment → doctor visit): {late:.4f} (SE={late_se:.4f})")

# ── 8. Selection bias decomposition ─────────────────────────────────────────
# The Wald estimator equals ITT / first_stage.
# We can approximate the selection bias term E[Y(0)|D=1] - E[Y(0)|D=0]
# using never-takers in each lottery arm as proxies for Y(0).
#
# Never-takers from Z=0: D=0 | Z=0  (mostly never-takers + defiers)
# Never-takers from Z=1: D=0 | Z=1  (pure never-takers under monotonicity)
# This approximation is exact only if defiers are absent (LATE monotonicity).

E_Y0_D0 = df.loc[(df["Z"] == 0) & (df["D"] == 0), "Y_doc"].mean()
E_Y0_D1 = df.loc[(df["Z"] == 1) & (df["D"] == 0), "Y_doc"].mean()

naive_gap_raw        = df.loc[df["D"]==1,"Y_doc"].mean() - df.loc[df["D"]==0,"Y_doc"].mean()
selection_bias_approx = E_Y0_D1 - E_Y0_D0
residual              = naive_gap_raw - late - selection_bias_approx

print(f"\n--- Bias Decomposition (approximate, requires monotonicity) ---")
print(f"Naive gap:               {naive_gap_raw:.4f}")
print(f"LATE (causal, compliers):{late:.4f}")
print(f"Selection bias (approx): {selection_bias_approx:.4f}")
print(f"Residual (hetero. bias): {residual:.4f}")

# ── 9. Summary table ─────────────────────────────────────────────────────────
results = pd.DataFrame({
    "Estimator":  ["Naive OLS", "ITT", "LATE (2SLS)"],
    "Estimate":   [naive_gap, itt, late],
    "SE":         [naive.bse["D"], itt_se, late_se],
    "Identifies": [
        "ATE + selection + heterogeneity bias",
        "Effect of lottery access (diluted by non-compliance)",
        "Causal effect of enrollment for compliers",
    ],
})
print("\n" + results.to_string(index=False))
```

**What to expect**: The naive OLS gap ($\approx 0.17$) substantially exceeds the LATE ($\approx 0.085$), illustrating that uninsured individuals would have had worse health outcomes even absent Medicaid — exactly the selection bias the lottery resolves. The ITT falls between them, diluted by the $\approx 40\%$ non-compliance rate. The approximate Wald identity $\text{LATE} \approx \text{ITT} / \text{first stage}$ should hold up to rounding.

---

## Summary

- **Identification precedes estimation**: a consistent estimator of a non-identified quantity consistently estimates the wrong thing.
- **SUTVA** comprises no-interference and no-hidden-versions; both can be empirically scrutinized rather than merely assumed.
- **The bias decomposition** splits the naive comparison into ATE, selection bias, and heterogeneity bias; each term has a distinct empirical signature.
- **The identification-estimation-inference triad** must proceed in order; skipping identification is the most common source of published causal errors.
- **Potential outcomes (RCM) and structural equations (SCM)** are complementary frameworks unified by the SWIG formalism.
- **The credibility revolution** correctly prioritizes design but creates the failure mode of design fetishism — avoiding important questions because no natural experiment exists.
- **The lottery identifies LATE, not ATE**: non-compliance, household stratification, and post-treatment control selection are each distinct threats to valid inference.

---

## Further Reading

- **Holland (1986)** — "Statistics and Causal Inference," *JASA*. The canonical formulation of the fundamental problem of causal inference and the Rubin model.

- **Rubin (1980)** — "Randomization Analysis of Experimental Data: The Fisher Randomization Test." Introduces SUTVA explicitly and derives its role in connecting potential outcomes to observables.

- **Angrist and Pischke (2009)** — *Mostly Harmless Econometrics*. Chapter 1 on the selection problem is required reading even for advanced practitioners. (Note: the credibility revolution paper cited in the text is Angrist and Pischke, *Journal of Economic Perspectives*, 2010; the book is 2009.)

- **Pearl (2009)** — *Causality: Models, Reasoning, and Inference*, 2nd ed. The definitive treatment of SCMs, the do-operator, and graphical identification. Chapters 1–3 are most relevant here.

- **Richardson and Robins (2014)** — "Single World Intervention Graphs (SWIGs): A Unification of the Counterfactual and Graphical Approaches to Causality." Unifies RCM and SCM; foundational for Chapter 4.

- **Finkelstein et al. (2012)** — "The Oregon Health Insurance Experiment: Evidence from the First Year," *QJE*. The primary paper for the OHE; documents the lottery design, compliance structure, and first-year health results used throughout this book.
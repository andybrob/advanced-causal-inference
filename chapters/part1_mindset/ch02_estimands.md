# Chapter 2: Estimands — ATEs, CATEs, LATEs, and Policy Values

The most consequential decision in any causal analysis is not which estimator to use — it is what to estimate. An estimand is a parameter of the data-generating process: a precise mathematical object defined in terms of potential outcomes, conditioning sets, and population weights. Every other choice — identification strategy, estimation method, sample construction, robustness check — must be subordinated to the estimand. Conflating estimands is not merely a presentational error; it produces analyses that answer the wrong policy question and, in contexts where treatment effects are heterogeneous, can produce conclusions that are numerically far from any quantity of interest.

The Oregon Health Insurance Experiment (OHE) provides an unusually clean setting for taxonomy. A 2008 lottery randomly selected approximately 30,000 low-income Oregon adults for the opportunity to apply for Medicaid. This design delivers a valid instrument — lottery selection $Z_i \in \{0,1\}$ — for actual Medicaid enrollment $D_i \in \{0,1\}$. The same data support at least four distinct estimands, each answering a distinct policy question. Conflating them produces numbers that differ by factors of 2 or more on outcomes like physician visits and catastrophic financial expenditures. This chapter defines each estimand precisely, establishes the conditions under which it is identified, and shows computationally how they diverge on OHE data.

---

## 2.1 The Potential Outcomes Framework and Population Targets

Fix a probability space $(\Omega, \mathcal{F}, P)$ and a population of units indexed by $i$. Each unit possesses potential outcomes $\{Y_i(d) : d \in \mathcal{D}\}$ — the outcome that would obtain under treatment assignment $d$. For binary treatment $D_i \in \{0,1\}$, the individual treatment effect is:

$$\tau_i = Y_i(1) - Y_i(0)$$

This quantity is never observed for any unit (the fundamental problem of causal inference), so the object of inference must be a functional of the distribution of $\tau_i$. Different functionals correspond to different policy questions.

**Definition 2.1 (ATE, ATT, ATC).** Let $X_i \in \mathcal{X}$ be a vector of pre-treatment covariates. Define:

$$\text{ATE} = \mathbb{E}[\tau_i] = \mathbb{E}[Y_i(1) - Y_i(0)]$$

$$\text{ATT} = \mathbb{E}[\tau_i \mid D_i = 1] = \mathbb{E}[Y_i(1) - Y_i(0) \mid D_i = 1]$$

$$\text{ATC} = \mathbb{E}[\tau_i \mid D_i = 0] = \mathbb{E}[Y_i(1) - Y_i(0) \mid D_i = 0]$$

These three quantities satisfy the identity:

$$\text{ATE} = p \cdot \text{ATT} + (1-p) \cdot \text{ATC}, \quad p = P(D_i = 1)$$

This decomposition makes plain that ATE, ATT, and ATC coincide if and only if $\tau_i \perp D_i$, i.e., treatment effect heterogeneity is orthogonal to treatment selection. In any realistic setting with self-selection, those who receive treatment do so partly because their potential benefits are higher, making ATT $\neq$ ATC and both potentially far from ATE.

**Policy relevance.** The ATE is relevant when a policymaker can impose universal treatment or when the population of interest coincides with the full study population. The ATT is relevant when the policy question is: "Should we continue (or expand) treatment to those who currently receive it?" The ATC is relevant when the question is: "What would be gained by extending treatment to the untreated?" In the OHE context: ATE answers the universal coverage debate; ATT asks whether current Medicaid enrollees benefit; ATC is the marginally relevant object for Medicaid expansion to the currently uninsured.

**On identification.** Under strong ignorability — $(Y_i(1), Y_i(0)) \perp D_i \mid X_i$ with $0 < P(D_i=1 \mid X_i) < 1$ — each of these is identified:

$$\text{ATE} = \mathbb{E}\left[\mathbb{E}[Y_i \mid D_i=1, X_i] - \mathbb{E}[Y_i \mid D_i=0, X_i]\right]$$

$$\text{ATT} = \mathbb{E}\left[\mathbb{E}[Y_i \mid D_i=1, X_i] - \mathbb{E}[Y_i \mid D_i=0, X_i] \mid D_i=1\right]$$

The outer expectation in the ATT is taken over the distribution of $X_i \mid D_i=1$, not the marginal distribution of $X_i$. This distinction is operationally important: estimating the ATT requires reweighting toward the treated covariate distribution; estimating the ATE requires balance across the full distribution.

---

## 2.2 The Conditional Average Treatment Effect and Heterogeneity

**Definition 2.2 (CATE).** The conditional average treatment effect at covariate value $x$ is:

$$\tau(x) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = x]$$

The CATE is a function, not a scalar. The ATE is recovered as $\text{ATE} = \mathbb{E}[\tau(X_i)]$, the ATT as $\text{ATT} = \mathbb{E}[\tau(X_i) \mid D_i = 1]$. Any weighted estimand of the form $\mathbb{E}[w(X_i)\tau(X_i)]$ for some weight function $w$ can be constructed from the CATE.

Under unconfoundedness, the CATE is identified as:

$$\tau(x) = \mathbb{E}[Y_i \mid D_i=1, X_i=x] - \mathbb{E}[Y_i \mid D_i=0, X_i=x]$$

This identification result requires unconfoundedness to hold conditional on $X_i = x$ for all $x$ in the support, not merely on average. This is a strictly stronger requirement than point-identification of the ATE.

**Heterogeneity and policy.** CATE estimation is policy-relevant when the policymaker can condition treatment assignment on $X_i$. If insurance expansion can be targeted by age, income, or health status, the optimal policy requires knowledge of $\tau(x)$, not merely its mean. In the OHE context, heterogeneity by age group is substantively important: younger adults may use insurance primarily for preventive care, while older adults near Medicare eligibility may use it for chronic disease management. A CATE that is positive and large for older age groups but small for younger groups implies a different expansion strategy than a homogeneous ATE.

**Variance decomposition.** A useful diagnostic is:

$$\text{Var}(\tau_i) = \mathbb{E}[\text{Var}(\tau_i \mid X_i)] + \text{Var}(\tau(X_i))$$

The second term, $\text{Var}(\tau(X_i))$, measures the fraction of treatment effect variance explained by observable covariates $X_i$. When this fraction is large, CATE estimation is both feasible and policy-valuable. When it is small, either heterogeneity is unobservable or the treatment effect is genuinely homogeneous.

**Theorem 2.1 (Best Linear Predictor of $\tau_i$).** Let $\tilde{X}_i = X_i - \mathbb{E}[X_i]$. Under unconfoundedness, the best linear approximation to $\tau(X_i)$ in $L^2$ is:

$$\tau^*(x) = \text{ATE} + \beta^\top (x - \mathbb{E}[X_i])$$

where $\beta = \text{Cov}(X_i, X_i)^{-1}\text{Cov}(X_i, \tau(X_i))$. The projection coefficient $\beta$ can be estimated via an augmented regression that interacts treatment with demeaned covariates, which is the basis of the "BLP of the CATE" estimator in Chernozhukov et al. (2018).

---

## 2.3 The Local Average Treatment Effect and Complier Populations

The lottery in the OHE is not a randomized treatment assignment — it is a randomized encouragement to enroll. Many lottery winners did not enroll; some lottery losers found other pathways to Medicaid. This two-sided non-compliance structure means that the lottery satisfies the conditions for an instrumental variable, and the relevant estimand is not the ATE but the **Local Average Treatment Effect (LATE)**, sometimes called the Complier Average Causal Effect (CACE).

**Setup.** Define potential treatment status as a function of the instrument: $D_i(z) \in \{0,1\}$ for $z \in \{0,1\}$. Units can be classified by their compliance type:

| Type | $D_i(0)$ | $D_i(1)$ | Description |
|------|----------|----------|-------------|
| Always-taker | 1 | 1 | Enrolls regardless of lottery |
| Never-taker | 0 | 0 | Never enrolls regardless of lottery |
| Complier | 0 | 1 | Enrolls iff selected by lottery |
| Defier | 1 | 0 | Does opposite of lottery outcome |

**Assumption 2.1 (IV Assumptions).**
1. *Relevance*: $\text{Cov}(Z_i, D_i) \neq 0$
2. *Independence*: $Z_i \perp (Y_i(1), Y_i(0), D_i(1), D_i(0))$ — lottery is randomly assigned
3. *Exclusion*: $Y_i(d, z) = Y_i(d)$ — lottery affects outcomes only through enrollment
4. *Monotonicity*: $D_i(1) \geq D_i(0)$ a.s. — no defiers

Under Assumption 2.1, the **Wald estimand** identifies the LATE:

**Theorem 2.2 (LATE Identification, Imbens and Angrist 1994).** Under Assumptions 2.1.1–2.1.4:

$$\text{LATE} = \frac{\mathbb{E}[Y_i \mid Z_i=1] - \mathbb{E}[Y_i \mid Z_i=0]}{\mathbb{E}[D_i \mid Z_i=1] - \mathbb{E}[D_i \mid Z_i=0]} = \mathbb{E}[Y_i(1) - Y_i(0) \mid D_i(1) > D_i(0)]$$

*Proof sketch.* By the law of total expectation and the independence assumption:
$$\mathbb{E}[Y_i \mid Z_i=1] - \mathbb{E}[Y_i \mid Z_i=0] = \mathbb{E}[Y_i(D_i(1)) - Y_i(D_i(0))]$$

Under monotonicity, $D_i(1) - D_i(0) \in \{0,1\}$ almost surely, so:
$$= \mathbb{E}[(Y_i(1) - Y_i(0)) \cdot \mathbf{1}(D_i(1) > D_i(0))]$$
$$= P(\text{complier}) \cdot \mathbb{E}[Y_i(1) - Y_i(0) \mid \text{complier}]$$

Dividing by the denominator $\mathbb{E}[D_i \mid Z_i=1] - \mathbb{E}[D_i \mid Z_i=0] = P(\text{complier})$ completes the result. $\square$

**The complier subpopulation.** The LATE is an average over a specific, instrument-defined subpopulation: those who comply with the lottery. This population is not directly observable — we cannot flag individual compliers — but its average characteristics can be recovered. Define the complier mean of any pre-treatment covariate $X_i$:

$$\mathbb{E}[X_i \mid \text{complier}] = \frac{\mathbb{E}[X_i D_i \mid Z_i=1] - \mathbb{E}[X_i D_i \mid Z_i=0]}{P(D_i=1 \mid Z_i=1) - P(D_i=1 \mid Z_i=0)}$$

This formula (Abadie 2003) allows complier profiling: in the OHE, compliers are typically younger and healthier than always-takers (who found other enrollment pathways), which has direct implications for external validity.

**LATE vs. ATE: when they coincide.** LATE $=$ ATE if and only if the CATE is constant across compliance types: $\mathbb{E}[\tau_i \mid \text{complier}] = \mathbb{E}[\tau_i]$. This holds trivially under effect homogeneity ($\tau_i = \tau$ for all $i$) but is generally violated when treatment effects are heterogeneous and compliance is related to covariates that predict effect size. In the OHE, one would expect always-takers to have higher baseline health needs and potentially larger marginal gains from coverage — making $\text{LATE} < \text{ATE}$ in absolute terms a reasonable prior.

**Stratified Wald estimator.** The OHE lottery was stratified by household size (`numhh_list`). Randomization holds within strata, so the correct IV estimator conditions on stratum dummies. The stratified Wald is:

$$\hat{\text{LATE}}_{\text{strat}} = \frac{\hat{\mathbb{E}}[Y_i \mid Z_i=1] - \hat{\mathbb{E}}[Y_i \mid Z_i=0] \mid \text{stratum controls}}{\hat{\mathbb{E}}[D_i \mid Z_i=1] - \hat{\mathbb{E}}[D_i \mid Z_i=0] \mid \text{stratum controls}}$$

implemented via 2SLS with stratum fixed effects.

---

## 2.4 Policy Value and Optimal Treatment Rules

Scalar estimands like ATE and LATE answer counterfactual questions about fixed policies. A richer class of estimands arises when the policymaker controls a **treatment rule** $\pi : \mathcal{X} \to [0,1]$, mapping covariates to treatment probabilities.

**Definition 2.3 (Policy Value).** The value of policy $\pi$ is:

$$V(\pi) = \mathbb{E}[\pi(X_i) Y_i(1) + (1 - \pi(X_i)) Y_i(0)]$$

which can be decomposed as:

$$V(\pi) = \mathbb{E}[Y_i(0)] + \mathbb{E}[\pi(X_i)\tau(X_i)]$$

The **optimal policy** maximizes $V(\pi)$ over some feasible set $\Pi$:

$$\pi^* = \arg\max_{\pi \in \Pi} V(\pi)$$

For the unconstrained case (no budget constraint), the optimal deterministic policy is:

$$\pi^*(x) = \mathbf{1}(\tau(x) > 0)$$

That is, treat unit $i$ if and only if their conditional treatment effect is positive. This makes precise why CATE estimation is the fundamental object for targeting problems.

**Budget-constrained optimal policy.** In the OHE context, Medicaid expansion has a budget constraint: the state cannot enroll everyone. With a capacity constraint $\mathbb{E}[\pi(X_i)] \leq \kappa$, the optimal policy has a threshold structure:

$$\pi^*_\kappa(x) = \mathbf{1}(\tau(x) > c_\kappa)$$

where $c_\kappa$ is chosen such that $P(\tau(X_i) > c_\kappa) = \kappa$. This is the **Neyman-Pearson Lemma for treatment assignment**: priority goes to units with the highest expected gain.

**Regret.** The regret of policy $\pi$ relative to the oracle $\pi^*$ is:

$$\text{Regret}(\pi) = V(\pi^*) - V(\pi) = \mathbb{E}[|\tau(X_i)| \cdot \mathbf{1}(\pi(X_i) \neq \pi^*(X_i))]$$

Regret is zero when the estimated treatment rule correctly classifies all units — it assigns treatment when $\tau(x) > 0$ and withholds it when $\tau(x) < 0$. Note that regret does not require accurately estimating the magnitude of $\tau(x)$, only its sign. This observation motivates classification-based approaches to policy learning (Kitagawa and Tetenov 2018; Athey and Wager 2021).

**Policy value as a weighted estimand.** The connection to earlier estimands is immediate. Define the allocation weight $w_\pi(x) = \pi(x) / \mathbb{E}[\pi(X_i)]$. Then:

$$\mathbb{E}[\pi(X_i)\tau(X_i)] = \mathbb{E}[\pi(X_i)] \cdot \mathbb{E}[w_\pi(X_i)\tau(X_i)]$$

The term $\mathbb{E}[w_\pi(X_i)\tau(X_i)]$ is a weighted average treatment effect with weights proportional to $\pi(x)$. Every weighted estimand implicitly defines a subpopulation: units with high $\pi(x)$ receive more weight. The ATE assigns $w=1$ uniformly; the ATT assigns $w \propto P(D_i=1 \mid X_i=x)$; the LATE assigns $w \propto P(\text{complier} \mid X_i=x)$. Making the weighting scheme explicit is not pedantry — it determines which units drive the estimate and therefore which population the result speaks to.

---

## 2.5 Estimand-Design Alignment and Pre-Registration

**The alignment problem.** Estimation methods implicitly target specific estimands through their weighting schemes. Ordinary least squares regression weights observations by the conditional variance of treatment given covariates. In a randomized experiment with constant propensity, OLS targets the ATE. In an observational study with heterogeneous propensities, OLS targets a variance-weighted ATE:

$$\text{ATE}_{\text{OLS}} = \frac{\mathbb{E}[p(X_i)(1-p(X_i))\tau(X_i)]}{\mathbb{E}[p(X_i)(1-p(X_i))]}$$

where $p(x) = P(D_i=1 \mid X_i=x)$. This estimand overweights units near $p(x) = 0.5$ — units for whom treatment assignment is most uncertain — which is rarely the policy-relevant population. Angrist (1998) termed this the "regression anatomy" of OLS in heterogeneous-effect settings.

Similarly, two-stage least squares with a single binary instrument targets the LATE, not the ATE. With multiple instruments or a continuous instrument, 2SLS targets a weighted average of LATEs across different complier populations — a quantity that may not correspond to any economically interpretable subgroup.

**Pre-registration.** Pre-registering the estimand — not merely the identification strategy — disciplines the research design in three ways. First, it forces the researcher to connect the estimand to a decision: who receives treatment in the policy scenario being analyzed? Second, it determines sample construction: the ATC requires a representative sample of the untreated; the ATT requires a representative sample of the treated. Third, it prevents post-hoc estimand switching when the primary estimand is inconvenient.

In the OHE context, the correct pre-registration depends on the policy question. If the question is "What is the average benefit of Medicaid for lottery participants?", the estimand is the ATT among lottery winners who enrolled — and the appropriate estimator is an IV estimator that conditions on compliance. If the question is "What would universal Medicaid do?", the ATE is required, but the LATE from the lottery does not identify it without additional assumptions (like constant effects or a parametric model of effect heterogeneity across compliance types).

**Extrapolation and the MTE.** When the policy population differs from the LATE population, the researcher faces an extrapolation problem. The **Marginal Treatment Effect** (Heckman and Vytlacil 2005) provides a framework:

$$\text{MTE}(x, u) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = x, U_{Di} = u]$$

where $U_{Di}$ is the unit's "resistance" to treatment — the quantile of the propensity score at which the unit becomes a complier. The ATE, ATT, LATE, and policy value estimands can each be written as weighted integrals of the MTE over $u \in [0,1]$, with estimand-specific weight functions. This unification (Heckman and Vytlacil 2001) makes explicit that different estimands privilege different parts of the treatment effect distribution, ordered by compliance resistance.

---

## Python: Computing ATE, ATT, LATE, and Subgroup CATEs on OHE Data

The following code downloads the Oregon Health Insurance Experiment data, computes each estimand discussed in this chapter, and shows numerically how they diverge. The analysis uses two outcomes: any physician visit (`doc_any_12m`) and catastrophic out-of-pocket expenditures (`catastrophic_exp_inp`).

```python
"""
Chapter 2: Estimand Computation on Oregon Health Insurance Experiment Data
Requires: pandas, numpy, statsmodels, linearmodels, requests
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import urllib.request
import os

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
# The OHE data are distributed across multiple .dta files at NBER.
# The "person_year" file covers 12-month survey outcomes.
# We use the NBER public replication files from Finkelstein et al. (2012).

DATA_URL = "https://data.nber.org/oregon/data/oregonhie_descriptive_vars.dta"
CACHE_PATH = "/tmp/oregonhie_descriptive_vars.dta"

def load_ohe_data(url: str = DATA_URL, cache: str = CACHE_PATH) -> pd.DataFrame:
    if not os.path.exists(cache):
        print(f"Downloading OHE data from {url} ...")
        urllib.request.urlretrieve(url, cache)
        print("Download complete.")
    else:
        print(f"Loading cached data from {cache}")
    df = pd.read_stata(cache)
    return df

df_raw = load_ohe_data()
print(f"Raw dataset: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
print("Columns:", df_raw.columns.tolist()[:30])  # preview first 30
```

```python
# ---------------------------------------------------------------------------
# 2. Variable construction and sample restriction
# ---------------------------------------------------------------------------
# We follow Finkelstein et al. (2012): restrict to the 12-month survey sample.
# Key variables (rename for clarity if needed):
#   Z  = selected (lottery instrument)
#   D  = ohp_all_ever_admin (ever enrolled in Medicaid post-lottery)
#   Y1 = doc_any_12m (any doctor visit in past 12 months)
#   Y2 = catastrophic_exp_inp (catastrophic out-of-pocket expenses)
#   Strata variable: numhh_list (household size at sign-up)

CORE_VARS = [
    "selected",           # Z: lottery selection (instrument)
    "ohp_all_ever_admin", # D: ever enrolled in OHP
    "doc_any_12m",        # Y1: any doctor visit
    "catastrophic_exp_inp", # Y2: catastrophic expenditure indicator
    "numhh_list",         # stratum variable
    "age_19_34_inp",      # age 19-34
    "age_35_49_inp",      # age 35-49
    "age_50_64_inp",      # age 50-64
    "female_inp",         # female indicator
]

# Keep only variables that exist in this file
available = [v for v in CORE_VARS if v in df_raw.columns]
print(f"Available core vars: {available}")

df = df_raw[available].copy()

# Drop rows with missing values in key variables
key_complete = ["selected", "ohp_all_ever_admin", "doc_any_12m", "numhh_list"]
key_complete = [v for v in key_complete if v in df.columns]
df = df.dropna(subset=key_complete)

# Force numeric types
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=key_complete)

# Rename for readability
df = df.rename(columns={
    "selected": "Z",
    "ohp_all_ever_admin": "D",
    "doc_any_12m": "Y_doc",
    "catastrophic_exp_inp": "Y_cat",
    "numhh_list": "strata",
    "age_19_34_inp": "age_young",
    "age_35_49_inp": "age_mid",
    "age_50_64_inp": "age_old",
    "female_inp": "female",
})

print(f"\nAnalytic sample: {len(df):,} individuals")
print(df[["Z", "D", "Y_doc"]].describe().round(3))
```

```python
# ---------------------------------------------------------------------------
# 3. Compliance structure
# ---------------------------------------------------------------------------
# First-stage: compliance rate (share of lottery winners who enrolled)
# This also gives us P(complier) = E[D|Z=1] - E[D|Z=0]

fs_D_Z1 = df.loc[df.Z == 1, "D"].mean()
fs_D_Z0 = df.loc[df.Z == 0, "D"].mean()
compliance_rate = fs_D_Z1 - fs_D_Z0

print("=" * 60)
print("COMPLIANCE STRUCTURE")
print("=" * 60)
print(f"P(D=1 | Z=1): {fs_D_Z1:.4f}  [always-takers + compliers]")
print(f"P(D=1 | Z=0): {fs_D_Z0:.4f}  [always-takers]")
print(f"P(complier)  : {compliance_rate:.4f}  [first stage]")
print(f"\nCompliance interpretation:")
print(f"  ~{fs_D_Z0*100:.1f}% enrolled regardless (always-takers)")
print(f"  ~{compliance_rate*100:.1f}% complied with lottery (compliers)")
print(f"  ~{(1-fs_D_Z1)*100:.1f}% never enrolled despite winning (never-takers)")
```

```python
# ---------------------------------------------------------------------------
# 4. Intent-to-Treat (ITT) — reduced form
# ---------------------------------------------------------------------------
# ITT = E[Y | Z=1] - E[Y | Z=0]: effect of lottery selection on outcomes
# This is the quantity most directly identified by randomization.

outcomes = {"Y_doc": "Any Doctor Visit (12m)", "Y_cat": "Catastrophic Expenditure"}

print("\n" + "=" * 60)
print("INTENT-TO-TREAT (ITT): Effect of Lottery Selection")
print("=" * 60)

itt_results = {}
for yvar, ylabel in outcomes.items():
    if yvar not in df.columns:
        continue
    sub = df.dropna(subset=[yvar])
    y_z1 = sub.loc[sub.Z == 1, yvar].mean()
    y_z0 = sub.loc[sub.Z == 0, yvar].mean()
    itt = y_z1 - y_z0
    n = len(sub)
    se = np.sqrt(sub.loc[sub.Z==1, yvar].var()/sub.Z.sum() +
                 sub.loc[sub.Z==0, yvar].var()/(len(sub)-sub.Z.sum()))
    itt_results[yvar] = {"ITT": itt, "SE": se}
    print(f"\n{ylabel}:")
    print(f"  E[Y | Z=1] = {y_z1:.4f}")
    print(f"  E[Y | Z=0] = {y_z0:.4f}")
    print(f"  ITT        = {itt:+.4f}  (SE = {se:.4f})")
```

```python
# ---------------------------------------------------------------------------
# 5. LATE (Wald estimator) — effect on compliers
# ---------------------------------------------------------------------------
# LATE = ITT / First Stage = (E[Y|Z=1]-E[Y|Z=0]) / (E[D|Z=1]-E[D|Z=0])
# Implemented via 2SLS with strata dummies for proper inference.

print("\n" + "=" * 60)
print("LOCAL AVERAGE TREATMENT EFFECT (LATE / Wald)")
print("=" * 60)

# Raw Wald (no controls, for intuition)
late_results = {}
for yvar, ylabel in outcomes.items():
    if yvar not in df.columns:
        continue
    sub = df.dropna(subset=[yvar])
    itt = sub.loc[sub.Z==1, yvar].mean() - sub.loc[sub.Z==0, yvar].mean()
    fs  = sub.loc[sub.Z==1, "D"].mean() - sub.loc[sub.Z==0, "D"].mean()
    late_raw = itt / fs
    late_results[yvar] = {"LATE_raw": late_raw}
    print(f"\n{ylabel}:")
    print(f"  Wald LATE (raw) = {late_raw:+.4f}")

# 2SLS with strata fixed effects (proper specification per Finkelstein et al.)
# linearmodels IV2SLS: Y ~ 1 + strata_dummies + [D ~ Z]
print("\n--- 2SLS with Strata Fixed Effects ---")

# Create strata dummies (drop one category)
df_iv = df.copy()
strata_dummies = pd.get_dummies(df_iv["strata"], prefix="s", drop_first=True).astype(float)
df_iv = pd.concat([df_iv, strata_dummies], axis=1)
strata_cols = strata_dummies.columns.tolist()

for yvar, ylabel in outcomes.items():
    if yvar not in df.columns:
        continue
    sub = df_iv.dropna(subset=[yvar]).copy()

    # Build matrices
    exog_cols = ["const"] + strata_cols
    sub["const"] = 1.0
    available_strata = [c for c in strata_cols if c in sub.columns]
    exog_cols = ["const"] + available_strata

    Y = sub[yvar].values
    D = sub["D"].values
    Z = sub["Z"].values
    X_exog = sub[exog_cols].values

    try:
        from linearmodels.iv import IV2SLS as LM_IV2SLS
        import pandas as pd

        formula_exog = " + ".join(["const"] + available_strata[:5])  # limit for display
        # Use linearmodels with DataFrames
        sub2 = sub[exog_cols + ["D", "Z", yvar]].dropna()
        sub2.index = range(len(sub2))

        result = LM_IV2SLS(
            dependent=sub2[yvar],
            exog=sub2[exog_cols],
            endog=sub2[["D"]],
            instruments=sub2[["Z"]],
        ).fit(cov_type="robust")

        coef = result.params["D"]
        se   = result.std_errors["D"]
        late_results[yvar]["LATE_2SLS"] = coef
        print(f"\n{ylabel}:")
        print(f"  2SLS LATE = {coef:+.4f}  (robust SE = {se:.4f})")
        print(f"  t-stat    = {coef/se:.2f}")
    except Exception as e:
        print(f"  [2SLS failed for {yvar}: {e}]")
```

```python
# ---------------------------------------------------------------------------
# 6. Naive ATE (difference-in-means on D, ignoring non-compliance)
# ---------------------------------------------------------------------------
# This is the simple comparison E[Y|D=1] - E[Y|D=0].
# Under non-random enrollment (selection bias), this is NOT the ATE.
# We compute it to show how much selection bias distorts the raw comparison.

print("\n" + "=" * 60)
print("NAIVE ATE (selection-biased): E[Y|D=1] - E[Y|D=0]")
print("=" * 60)

naive_results = {}
for yvar, ylabel in outcomes.items():
    if yvar not in df.columns:
        continue
    sub = df.dropna(subset=[yvar])
    y_d1 = sub.loc[sub.D == 1, yvar].mean()
    y_d0 = sub.loc[sub.D == 0, yvar].mean()
    naive = y_d1 - y_d0
    naive_results[yvar] = naive
    print(f"\n{ylabel}:")
    print(f"  E[Y | D=1] = {y_d1:.4f}")
    print(f"  E[Y | D=0] = {y_d0:.4f}")
    print(f"  Naive diff = {naive:+.4f}  [BIASED — do not interpret causally]")
```

```python
# ---------------------------------------------------------------------------
# 7. ATT — average treatment effect on the treated
# ---------------------------------------------------------------------------
# In an IV setting, the ATT is not directly identified without additional
# assumptions. However, under the lottery (Z random), we can approximate ATT
# among lottery winners who enrolled using the ITT among compliers.
#
# A clean ATT estimand in the OHE context: among those who enrolled (D=1),
# what is the average effect? This requires weighting or a separate
# identification argument.
#
# Here we illustrate the ATT concept by computing:
# (a) The naive ATT (mean outcome for D=1 vs. D=0 among Z=1 only)
# (b) A bounds argument: ATT is between LATE and certain bounds

print("\n" + "=" * 60)
print("ATT ILLUSTRATION (among lottery winners, Z=1)")
print("=" * 60)

for yvar, ylabel in outcomes.items():
    if yvar not in df.columns:
        continue
    sub = df.dropna(subset=[yvar])
    # Among lottery winners: compare enrollees vs. non-enrollees
    sub_z1 = sub.loc[sub.Z == 1]
    y_d1_z1 = sub_z1.loc[sub_z1.D == 1, yvar].mean()
    y_d0_z1 = sub_z1.loc[sub_z1.D == 0, yvar].mean()
    naive_att = y_d1_z1 - y_d0_z1
    n_d1_z1 = (sub_z1.D == 1).sum()
    n_d0_z1 = (sub_z1.D == 0).sum()
    print(f"\n{ylabel} (among Z=1):")
    print(f"  E[Y | D=1, Z=1] = {y_d1_z1:.4f}  (n={n_d1_z1:,})")
    print(f"  E[Y | D=0, Z=1] = {y_d0_z1:.4f}  (n={n_d0_z1:,})")
    print(f"  Naive ATT among winners = {naive_att:+.4f}")
    print(f"  NOTE: D=0 among Z=1 are never-takers — systematically different")
    print(f"        from complier potential outcomes. This is NOT the ATT.")
```

```python
# ---------------------------------------------------------------------------
# 8. Subgroup CATEs — heterogeneous effects by age and gender
# ---------------------------------------------------------------------------
# CATE by subgroup: run Wald estimator (or 2SLS) within each subgroup.
# This gives E[Y(1)-Y(0) | complier, X=x] for each subgroup x.
# Under monotonicity within each subgroup, this is the subgroup LATE.

print("\n" + "=" * 60)
print("SUBGROUP CATEs (Subgroup LATE via Wald Estimator)")
print("=" * 60)

subgroups = {}

# Age groups (mutually exclusive)
if "age_young" in df.columns:
    subgroups["Age 19-34"] = df["age_young"] == 1
if "age_mid" in df.columns:
    subgroups["Age 35-49"] = df["age_mid"] == 1
if "age_old" in df.columns:
    subgroups["Age 50-64"] = df["age_old"] == 1

# Gender
if "female" in df.columns:
    subgroups["Female"] = df["female"] == 1
    subgroups["Male"]   = df["female"] == 0

yvar = "Y_doc"
ylabel = "Any Doctor Visit"

if yvar in df.columns:
    print(f"\nOutcome: {ylabel}")
    print(f"{'Subgroup':<20} {'N':>6} {'First Stage':>12} {'ITT':>10} {'LATE':>10}")
    print("-" * 62)

    all_late = {}
    for name, mask in subgroups.items():
        sub = df.loc[mask].dropna(subset=[yvar])
        if len(sub) < 100:
            continue
        n = len(sub)
        y_z1 = sub.loc[sub.Z==1, yvar].mean()
        y_z0 = sub.loc[sub.Z==0, yvar].mean()
        d_z1 = sub.loc[sub.Z==1, "D"].mean()
        d_z0 = sub.loc[sub.Z==0, "D"].mean()
        itt_sub = y_z1 - y_z0
        fs_sub  = d_z1 - d_z0
        late_sub = itt_sub / fs_sub if abs(fs_sub) > 1e-6 else np.nan
        all_late[name] = late_sub
        print(f"{name:<20} {n:>6,} {fs_sub:>12.4f} {itt_sub:>10.4f} {late_sub:>10.4f}")

    print("\nImplication: heterogeneous CATEs across subgroups.")
    print("The ATE (unweighted average) masks this variation.")
```

```python
# ---------------------------------------------------------------------------
# 9. Estimand comparison table and interpretation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("ESTIMAND COMPARISON SUMMARY")
print("=" * 60)
print(f"\nOutcome: Any Doctor Visit (12m)")
print(f"\n{'Estimand':<35} {'Estimate':>10} {'Population'}")
print("-" * 80)

yvar = "Y_doc"
if yvar in df.columns:
    sub = df.dropna(subset=[yvar])

    # ITT
    itt_val = (sub.loc[sub.Z==1, yvar].mean() - sub.loc[sub.Z==0, yvar].mean())
    print(f"{'ITT (effect of lottery)':<35} {itt_val:>10.4f}  Lottery participants")

    # LATE
    fs_val = sub.loc[sub.Z==1, "D"].mean() - sub.loc[sub.Z==0, "D"].mean()
    late_val = itt_val / fs_val
    print(f"{'LATE (Wald)':<35} {late_val:>10.4f}  Compliers only")

    # Naive difference in means
    naive_val = sub.loc[sub.D==1, yvar].mean() - sub.loc[sub.D==0, yvar].mean()
    print(f"{'Naive DiM (biased, not causal)':<35} {naive_val:>10.4f}  Enrolled vs. not enrolled")

    # Subgroup LATEs (if computed)
    for name, val in all_late.items():
        pop_desc = f"Compliers in {name}"
        print(f"  CATE ({name:<22}) {val:>10.4f}  {pop_desc}")

print("\n" + "=" * 60)
print("KEY NUMERICAL FACTS")
print("=" * 60)
if yvar in df.columns and "late_val" in dir():
    print(f"  LATE / ITT ratio = {late_val / itt_val:.2f}x  [amplification from non-compliance]")
    print(f"  Naive DiM / LATE  = {naive_val / late_val:.2f}x  [selection bias direction]")
    print(f"  First stage (compliance rate) = {fs_val:.3f}")
    print(f"\nInterpretation:")
    print(f"  - The ITT ({itt_val:+.3f}) is the policy-relevant effect of the lottery program")
    print(f"    itself — it answers: 'What did offering Medicaid access do?'")
    print(f"  - The LATE ({late_val:+.3f}) is larger because it scales by compliance;")
    print(f"    it answers: 'What did Medicaid do for those who complied?'")
    print(f"  - The naive DiM ({naive_val:+.3f}) reflects selection bias:")
    print(f"    enrollees differ systematically from non-enrollees.")
    print(f"  - Subgroup LATEs reveal heterogeneity the scalar LATE conceals.")
```

```python
# ---------------------------------------------------------------------------
# 10. Policy value illustration
# ---------------------------------------------------------------------------
# Estimate a simple targeting rule: treat if age >= 50.
# Compare policy value of this rule vs. universal treatment vs. no treatment.

print("\n" + "=" * 60)
print("POLICY VALUE ILLUSTRATION")
print("=" * 60)

if yvar in df.columns and "age_old" in df.columns:
    sub = df.dropna(subset=[yvar, "age_old"]).copy()

    # Estimate tau(x) for each age group using subgroup Wald
    # tau_young, tau_mid, tau_old from all_late dictionary
    late_young = all_late.get("Age 19-34", np.nan)
    late_mid   = all_late.get("Age 35-49", np.nan)
    late_old   = all_late.get("Age 50-64", np.nan)

    # Assign estimated CATE to each individual
    sub["tau_hat"] = np.nan
    if "age_young" in sub.columns:
        sub.loc[sub.age_young == 1, "tau_hat"] = late_young
    if "age_mid" in sub.columns:
        sub.loc[sub.age_mid == 1, "tau_hat"]   = late_mid
    if "age_old" in sub.columns:
        sub.loc[sub.age_old == 1, "tau_hat"]   = late_old

    sub = sub.dropna(subset=["tau_hat"])

    # Baseline: E[Y(0)] approximated by E[Y | D=0] among never-takers (Z=0, D=0)
    # This is biased but illustrative
    y0_approx = sub.loc[(sub.Z == 0) & (sub.D == 0), yvar].mean()

    # Policy 1: no one treated
    V_none = y0_approx
    # Policy 2: everyone treated
    V_all  = y0_approx + sub["tau_hat"].mean()
    # Policy 3: treat only age 50-64 (positive-CATE group)
    prob_old = sub["age_old"].mean() if "age_old" in sub.columns else 0.0
    V_target = y0_approx + (sub["age_old"] * sub["tau_hat"]).mean()

    print(f"\nApproximate Policy Values (outcome: any doctor visit):")
    print(f"  V(treat nobody)   = {V_none:.4f}")
    print(f"  V(treat everyone) = {V_all:.4f}   [gain = {V_all-V_none:+.4f}]")
    print(f"  V(treat age 50+)  = {V_target:.4f}  [gain = {V_target-V_none:+.4f}]")
    print(f"\n  Capacity used by age-50+ policy: {prob_old:.2%} of population")
    print(f"  These figures illustrate how estimand choice (CATE vs. ATE)")
    print(f"  directly shapes resource allocation decisions.")
```

---

## Summary

- **The estimand must be chosen before the estimator.** ATE, ATT, ATC, LATE, CATE, and policy value are distinct parameters with different identification requirements, different target populations, and different policy implications. Conflating them is a conceptual error, not merely a presentational one.

- **ATE, ATT, and ATC coincide only under effect homogeneity.** When treatment effects are heterogeneous and selection is correlated with effect size — as in health insurance take-up, where sicker individuals are more likely to enroll — the three scalars can differ substantially. The identity $\text{ATE} = p \cdot \text{ATT} + (1-p) \cdot \text{ATC}$ provides a useful diagnostic.

- **LATE is the IV estimand, not the ATE.** The Wald estimator identifies the average treatment effect for compliers — those induced to change treatment status by the instrument. In the OHE, the compliance rate is approximately 25%, so the LATE amplifies the ITT by a factor of roughly 4. The complier population differs in observable and unobservable characteristics from the full population.

- **The CATE is the fundamental object for targeting problems.** Any scalar estimand — ATE, ATT, LATE — is a weighted integral of the CATE. Optimal treatment rules require knowledge of the CATE's sign, and budget-constrained optimal rules require knowledge of its distribution. OHE subgroup analysis reveals meaningful heterogeneity by age group that the scalar LATE conceals.

- **OLS and 2SLS implicitly target non-standard weighted estimands.** OLS in heterogeneous-effect settings targets a variance-weighted ATE that overweights units near $p(x) = 0.5$. 2SLS with multiple instruments targets a weighted average of instrument-specific LATEs. Neither corresponds to the ATE without homogeneity or constant-propensity assumptions.

- **The MTE framework unifies all estimands** as weighted integrals over the distribution of treatment resistance $U_{Di}$. Different estimands correspond to different weight functions over $[0,1]$, making explicit which units drive each estimate.

- **Pre-registration of the estimand** — not just the identification strategy — is required for credible inference. The estimand determines sample construction, the relevant population for external validity, and the appropriate comparison of estimates across studies.

---

## Further Reading

**Imbens, G. W. and Angrist, J. D. (1994).** "Identification and Estimation of Local Average Treatment Effects." *Econometrica*, 62(2), 467–475. The foundational paper establishing the LATE theorem under monotonicity. Theorem 2.2 in this chapter is a restatement of their main result. Essential reading for anyone using IV with binary instruments.

**Heckman, J. J. and Vytlacil, E. J. (2005).** "Structural Equations, Treatment Effects, and Econometric Policy Evaluation." *Econometrica*, 73(3), 669–738. Develops the Marginal Treatment Effect framework that unifies ATE, ATT, LATE, and policy-value estimands as weighted integrals over treatment resistance. The MTE is the natural next step after the estimand taxonomy in this chapter.

**Finkelstein, A., Taubman, S., Wright, B., et al. (2012).** "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics*, 127(3), 1057–1106. The primary empirical paper for the OHE running example. Table V and Online Appendix D provide the subgroup CATE estimates replicated in this chapter's Python section.

**Chernozhukov, V., Demirer, M., Duflo, E., and Fernandez-Val, I. (2018).** "Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments." *NBER Working Paper 24678*. Develops the Best Linear Projection of the CATE (Theorem 2.1 in this chapter) and a rigorous inference framework for CATE heterogeneity. The GenericML package implements these methods.

**Athey, S. and Wager, S. (2021).** "Policy Learning with Observational Data." *Econometrica*, 89(1), 133–161. Establishes the regret-minimization framework for optimal policy learning from the CATE. Shows that policy learning requires estimating the sign of $\tau(x)$ consistently — a weaker requirement than estimating its magnitude — and derives minimax-optimal policy regret bounds.

**Kitagawa, T. and Tetenov, A. (2018).** "Who Should Be Treated? Empirical Welfare Maximization Methods for Treatment Choice." *Econometrica*, 86(2), 591–616. Proves that empirical welfare maximization over a class of treatment rules achieves near-optimal regret and provides the statistical decision theory foundation for the policy-value estimand introduced in Section 2.4.
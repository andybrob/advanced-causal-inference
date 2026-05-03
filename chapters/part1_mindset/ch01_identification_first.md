# Chapter 1: The Identification-First Principle

Empirical economists spend enormous effort choosing between fixed effects and random effects, debating instrument strength, and optimizing standard error estimators. This effort is frequently misallocated. The choice between a robust sandwich estimator and a clustered covariance matrix is consequential only if the underlying quantity being estimated has a causal interpretation in the first place. Identification — the mapping from a probability distribution over observables to a causal parameter of interest — is logically prior to every estimation decision. A consistent estimator of a non-identified parameter is consistently estimating the wrong thing.

The Oregon Health Insurance Experiment (OHE) is one of the cleanest natural experiments in health economics. In 2008, Oregon used a lottery to allocate limited Medicaid slots among roughly 90,000 low-income adults on a waiting list. Approximately 30,000 were selected, creating random variation in insurance access that is unconfounded by individual health status, risk preferences, or financial sophistication. The naive empirical question — do insured people have better health outcomes than uninsured people? — has an obvious answer and an obvious problem: people who obtain insurance differ systematically from those who do not. The lottery resolves this problem, but understanding *why* it resolves it, and precisely *what* causal parameter it identifies, requires the conceptual architecture this chapter develops.

---

## 1.1 The Fundamental Problem and Potential Outcomes

Let $i \in \{1, \ldots, n\}$ index units. Define the binary treatment $D_i \in \{0, 1\}$ where $D_i = 1$ denotes Medicaid enrollment. For each unit, posit two **potential outcomes**: $Y_i(1)$, the outcome that would obtain under insurance, and $Y_i(0)$, the outcome under no insurance. The **individual treatment effect** is $\tau_i = Y_i(1) - Y_i(0)$.

The fundamental problem of causal inference (Holland, 1986) is that we observe at most one potential outcome per unit. Define the observed outcome:

$$Y_i^{obs} = D_i Y_i(1) + (1 - D_i) Y_i(0)$$

The missing potential outcome $Y_i(1 - D_i)$ is a **counterfactual** — it describes what would have happened under the treatment the unit did not receive. No amount of data collection can recover it for a given unit at a given point in time. Causal inference is fundamentally about imputing counterfactuals, and the credibility of any causal claim rests entirely on the plausibility of those imputations.

### Stable Unit Treatment Value Assumption

The potential outcomes framework requires **SUTVA** (Rubin, 1980), comprising two conditions:

**No interference:** Unit $i$'s potential outcomes do not depend on the treatment assignments of other units. Formally, if $\mathbf{D} = (D_1, \ldots, D_n)$ is the vector of all assignments, then:

$$Y_i(\mathbf{D}) = Y_i(D_i)$$

**No hidden versions of treatment:** The treatment is well-defined; there is a single version of $D_i = 1$ (insurance enrollment) such that $Y_i(1)$ is unambiguous.

SUTVA is not innocuous. No-interference fails when treatment of one unit affects outcomes of others — vaccination programs create herd immunity effects, making $Y_i(0)$ depend on others' vaccination rates. In OHE, partial equilibrium arguments support the assumption: the lottery affected a small fraction of low-income Oregonians, so provider availability and health system capacity were unlikely to shift materially. For national insurance expansions (e.g., the ACA), interference through labor market and insurance market equilibria is a genuine concern.

The no-hidden-versions condition requires that "enrolled in Medicaid" is a coherent, singular treatment. Oregon Medicaid in 2008 had a specific benefit structure; this is more defensible than treatments like "education" or "poverty."

**Consistency** follows from SUTVA: for each unit, $Y_i^{obs} = Y_i(D_i)$. This links the potential outcomes framework to data we can actually observe.

### Estimands

The individual treatment effect $\tau_i$ is never identified without additional assumptions. Population-level estimands aggregate over the distribution of $\tau_i$:

$$\text{ATE} = \mathbb{E}[\tau_i] = \mathbb{E}[Y_i(1) - Y_i(0)]$$

$$\text{ATT} = \mathbb{E}[\tau_i \mid D_i = 1] = \mathbb{E}[Y_i(1) - Y_i(0) \mid D_i = 1]$$

$$\text{ATU} = \mathbb{E}[\tau_i \mid D_i = 0] = \mathbb{E}[Y_i(1) - Y_i(0) \mid D_i = 0]$$

These estimands differ whenever treatment effects are heterogeneous *and* treatment selection is correlated with effect size. The ATE requires information on $Y_i(0)$ for the treated and $Y_i(1)$ for the untreated — both counterfactual. The ATT requires only $Y_i(0)$ for the treated to be imputed; the ATU requires $Y_i(1)$ for the untreated to be imputed.

A further estimand, central to IV settings, is the **Local Average Treatment Effect (LATE)** or Complier Average Causal Effect (CACE):

$$\text{LATE} = \mathbb{E}[\tau_i \mid \text{Complier}_i]$$

where compliers are units whose treatment status responds to the instrument. We develop this formally in Chapter 3; for now, note that different identification strategies recover different estimands, and conflating them is a source of many unproductive empirical debates.

---

## 1.2 Structural Equations and the Pearl Framework

The potential outcomes framework is concise for treatment effect estimation but cumbersome for reasoning about complex systems with many interacting variables. Pearl's (2000) **Structural Causal Model (SCM)** framework provides complementary machinery.

An SCM $\mathcal{M} = (U, V, F, P_U)$ consists of:
- **Exogenous variables** $U = \{U_1, \ldots, U_k\}$ with joint distribution $P_U$, representing unobserved background factors
- **Endogenous variables** $V = \{V_1, \ldots, V_m\}$
- **Structural equations** $F = \{f_1, \ldots, f_m\}$ where $V_i = f_i(\text{Pa}(V_i), U_i)$ and $\text{Pa}(V_i) \subseteq V \setminus \{V_i\}$ denotes parents of $V_i$
- The induced **Directed Acyclic Graph (DAG)** $\mathcal{G}$, where an edge $V_j \to V_i$ exists if $V_j \in \text{Pa}(V_i)$

The **do-operator** formalizes intervention. The quantity $P(Y \mid do(D = d))$ is defined by the **mutilated model** $\mathcal{M}_d$, obtained by replacing the structural equation for $D$ with $D = d$ and leaving all other equations intact. This is the distribution of $Y$ in a world where $D$ is externally set to $d$, eliminating any back-door paths through common causes of $D$ and $Y$.

The connection to potential outcomes: under regularity conditions, $Y_i(d)$ has the same distribution as $Y$ in $\mathcal{M}_d$. Pearl's **causal hierarchy** (the ladder of causation) distinguishes:

1. **Association**: $P(Y \mid D = d)$ — observational conditional distributions
2. **Intervention**: $P(Y \mid do(D = d))$ — post-intervention distributions
3. **Counterfactual**: $P(Y_i(d') \mid D_i = d, Y_i = y)$ — retrospective unit-level counterfactuals

Levels 2 and 3 cannot be computed from level 1 data alone without causal assumptions encoded in the model structure. The identification problem is: under what conditions does $P(Y \mid do(D = d))$ equal some functional of the observed distribution $P(Y, D, X)$?

**The Back-Door Criterion** (Pearl, 2000): A set of observed covariates $X$ satisfies the back-door criterion relative to $(D, Y)$ in DAG $\mathcal{G}$ if:
1. No node in $X$ is a descendant of $D$
2. $X$ blocks every back-door path between $D$ and $Y$ (paths with an arrow into $D$)

If $X$ satisfies the back-door criterion:

$$P(Y = y \mid do(D = d)) = \sum_x P(Y = y \mid D = d, X = x) P(X = x)$$

This is the **adjustment formula**, and it provides the foundation for regression-based causal inference when sufficient controls are available. The key structural question — whether a proposed control set satisfies back-door — cannot be answered from data; it requires domain knowledge encoded in the DAG.

In the OHE context: lottery selection $Z_i$ is assigned by a random draw from a waiting list, stratified by household size. The path $Z_i \to D_i \to Y_i$ is the front-door (causal) path. The path $Z_i \leftarrow \text{(confounders)} \to Y_i$ does not exist by design — randomization severs any back-door from $Z$ to $Y$ through unobservables. This is the structural content of the lottery's identifying power.

---

## 1.3 The Identification-Estimation-Inference Triad

Graduate econometrics training typically emphasizes the **estimation** and **inference** stages, treating identification as either obvious or someone else's problem. This is an error in priority ordering. The correct sequence is:

**Identification**: Does there exist a mapping $\phi: \mathcal{P} \to \mathbb{R}$ such that $\phi(P)$ equals the causal parameter $\theta^*$ for all data-generating processes $P$ in a specified model class? If not, $\theta^*$ is not identified and no estimator — however sophisticated — recovers it consistently.

**Estimation**: Given that $\theta^* = \phi(P)$, how do we construct a feasible estimator $\hat{\theta}_n$ from finite data such that $\hat{\theta}_n \xrightarrow{p} \phi(P_0)$ where $P_0$ is the true DGP?

**Inference**: How do we characterize the sampling distribution of $\hat{\theta}_n - \theta^*$ to form valid confidence intervals and tests?

A result is **identified** if and only if two data-generating processes that are observationally equivalent (same $P(Y^{obs}, D, X)$) imply the same value of the causal parameter. Formally, let $\Theta$ be the parameter space. The parameter $\theta$ is identified in model $\mathcal{M}$ if the mapping $\theta \mapsto P_\theta$ (from parameter values to induced observational distributions) is injective on $\mathcal{M}$.

**Partial identification** (Manski, 1990, 2003) is the systematic approach when the identification condition fails: instead of point identification, derive the tightest possible bounds on $\theta^*$ consistent with the observed data and maintained assumptions. We return to partial identification in Chapter 8; here the point is that non-identification does not mean "we know nothing" — it means the data constrain the parameter to a set rather than a point.

### What Graduate Training Leaves Underspecified

Standard econometrics courses introduce instrumental variables with the exclusion restriction and relevance condition as maintained assumptions. What they typically underspecify:

1. **Which estimand IV recovers.** Under treatment effect heterogeneity, 2SLS estimates the LATE for compliers, not the ATE. Whether compliers are policy-relevant is a substantive question that depends on context.

2. **The distinction between testable and untestable assumptions.** Instrument relevance ($\text{Cov}(Z, D) \neq 0$) is testable; the exclusion restriction ($Z \perp Y \mid D, U$) is not. Conflating these leads to overconfidence in identification.

3. **Sensitivity to SUTVA.** If treatment of one unit affects outcomes of others, potential outcomes are indexed over assignment vectors, not scalars, and standard estimators are generally inconsistent for any standard estimand.

4. **The target population.** An ATT estimated from a convenience sample may not transport to the policy-relevant population. External validity is part of the identification problem.

---

## 1.4 The Naive Estimator and Its Decomposition

The observational estimand $\mathbb{E}[Y^{obs} \mid D = 1] - \mathbb{E}[Y^{obs} \mid D = 0]$ is what OLS on $Y_i = \alpha + \tau D_i + \varepsilon_i$ recovers (without covariates). Decompose this:

$$\mathbb{E}[Y^{obs} \mid D = 1] - \mathbb{E}[Y^{obs} \mid D = 0]$$

$$= \mathbb{E}[Y_i(1) \mid D_i = 1] - \mathbb{E}[Y_i(0) \mid D_i = 0]$$

Add and subtract $\mathbb{E}[Y_i(0) \mid D_i = 1]$:

$$= \underbrace{\mathbb{E}[Y_i(1) - Y_i(0) \mid D_i = 1]}_{\text{ATT}} + \underbrace{\mathbb{E}[Y_i(0) \mid D_i = 1] - \mathbb{E}[Y_i(0) \mid D_i = 0]}_{\text{Selection Bias}}$$

The selection bias term is the difference in baseline potential outcomes between the treated and control groups. In the OHE context: people who obtain insurance (in the absence of the lottery) are not a random draw. They may be sicker (adverse selection), more health-conscious, or more financially sophisticated. Any of these channels can generate $\mathbb{E}[Y_i(0) \mid D_i = 1] \neq \mathbb{E}[Y_i(0) \mid D_i = 0]$, rendering the naive estimator inconsistent for the ATT.

To expose the further gap between ATT and ATE, use $\text{ATE} = p \cdot \text{ATT} + (1-p) \cdot \text{ATU}$ where $p = P(D_i = 1)$:

$$\text{Naive} - \text{ATE} = \text{Selection Bias} + (1-p)(\text{ATT} - \text{ATU})$$

The second term is **heterogeneity bias**: it is nonzero when treatment effects differ between those who select into treatment and those who do not. Even with zero selection bias (as in a randomized experiment where $\mathbb{E}[Y_i(0) \mid D_i = 1] = \mathbb{E}[Y_i(0) \mid D_i = 0]$), the naive estimator recovers the ATE only if treatment effects are constant ($\tau_i = \tau$ for all $i$) or treatment is fully randomized (so $p$ can be set but the estimand is still the ATE).

**Theorem 1.1 (Identification under Random Assignment).** Suppose $D_i \perp (Y_i(0), Y_i(1))$ (independence, as in a completely randomized experiment). Then:

$$\mathbb{E}[Y^{obs} \mid D = 1] - \mathbb{E}[Y^{obs} \mid D = 0] = \text{ATE}$$

*Proof.* Under independence:
$$\mathbb{E}[Y_i(1) \mid D_i = 1] = \mathbb{E}[Y_i(1)] \quad \text{and} \quad \mathbb{E}[Y_i(0) \mid D_i = 0] = \mathbb{E}[Y_i(0)]$$

Therefore the selection bias term is $\mathbb{E}[Y_i(0) \mid D_i = 1] - \mathbb{E}[Y_i(0) \mid D_i = 0] = \mathbb{E}[Y_i(0)] - \mathbb{E}[Y_i(0)] = 0$, and the naive estimator equals $\mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)] = \text{ATE}$. $\square$

The lottery in OHE generates something close to this. The instrument $Z_i$ (lottery selection) is randomly assigned conditional on household size (the stratification variable). If we condition on household size strata, $Z_i \perp (Y_i(0), Y_i(1))$ by design. This means the **intention-to-treat (ITT) effect** $\mathbb{E}[Y^{obs} \mid Z_i = 1] - \mathbb{E}[Y^{obs} \mid Z_i = 0]$ is identified as the causal effect of *lottery selection* on outcomes. Scaling by the first-stage compliance rate recovers the LATE, as we develop in Chapter 3.

---

## 1.5 The Credibility Revolution and Asymmetric Confounding

The **credibility revolution** (Angrist and Pischke, 2010) in empirical economics shifted the discipline's attention from estimation sophistication to identification rigor. The core claim: a study's contribution to knowledge is bounded by the weakest link in its identification chain, not by the sophistication of its econometric methods.

Three principles characterize credible causal research:

**1. Transparent identification.** State the identifying assumptions explicitly and argue for their plausibility. An assumption like "lottery selection is independent of potential outcomes, conditional on household size strata" is falsifiable in its observable implications (covariate balance across lottery winners and losers) even if the core independence claim is not directly testable.

**2. Design-based inference.** Where possible, derive uncertainty from randomization rather than asymptotic approximations that may perform poorly in finite samples. In OHE, randomization justifies finite-sample exact inference on the ITT using permutation distributions.

**3. Sensitivity analysis.** Quantify how conclusions change as assumptions are relaxed. If the exclusion restriction holds only approximately — e.g., lottery selection affects health behaviors directly, not only through insurance — how large must this direct effect be to overturn the main finding?

### The Asymmetry Between Ruling In and Ruling Out

A critical epistemological point: confounding can be ruled *out* by design (randomization eliminates all confounders simultaneously) but can never be definitively ruled *in* from observational data. No set of covariates, however rich, guarantees unconfoundedness in observational studies. The researcher who controls for 50 demographic variables has not eliminated omitted variable bias; they have merely shifted the burden of confounding to unobserved variables that are uncorrelated with the 50 included controls.

This asymmetry has a precise formal statement. Define the **sharp null** $H_0: Y_i(1) = Y_i(0)$ for all $i$. Randomization implies the distribution of any test statistic under $H_0$ is known exactly (it is the permutation distribution). A covariate adjustment strategy implies no such thing — the null distribution depends on the correctness of the outcome model, which is untested.

The practical implication: a finding from a randomized design is more credible than an equally-sized finding from an observational study with the same point estimate and standard error, because the randomized finding has a known, assumption-free identification strategy. Effect size and statistical significance are not sufficient statistics for evidential weight.

**Manski's Worst-Case Bounds.** Without any assumptions beyond the observed data range, the ATE is bounded as:

$$\text{ATE} \in \left[\mathbb{E}[Y^{obs} \mid D=1] p - y_{max}(1-p) + y_{min} \cdot 0, \ldots\right]$$

More precisely, if $Y_i \in [y_{min}, y_{max}]$, then:

$$\text{ATE} \in \left[p \cdot \mathbb{E}[Y^{obs}|D=1] + (1-p) \cdot y_{min} - \{p \cdot y_{max} + (1-p) \cdot \mathbb{E}[Y^{obs}|D=0]\},\right.$$
$$\left. p \cdot \mathbb{E}[Y^{obs}|D=1] + (1-p) \cdot y_{max} - \{p \cdot y_{min} + (1-p) \cdot \mathbb{E}[Y^{obs}|D=0]\}\right]$$

Simplifying: the width of the Manski bounds is $(y_{max} - y_{min})$, regardless of sample size. More data narrows confidence sets around the bounds but does not shrink the bounds themselves. Only assumptions shrink bounds.

---

## Python: Loading the OHE Data and Decomposing Identification Failure

```python
"""
Chapter 1: Oregon Health Insurance Experiment — Identification Failure Demo

This script:
1. Loads the OHE data from the NBER replication files
2. Computes the naive OLS estimator (insured vs. uninsured comparison)
3. Decomposes the bias using the bias decomposition formula
4. Demonstrates covariate balance by lottery status (validating randomization)
5. Computes Manski sharp bounds on the ATE for a binary outcome

Requirements:
    pip install pandas numpy scipy statsmodels requests pyreadstat
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
# The Oregon Health Insurance Experiment data is available from NBER.
# We use the 12-month survey file and the administrative enrollment data.
# Baicker et al. (2013) replication files: stata .dta format.
#
# Primary file: oregonhie_survey12m_vars.dta (12-month survey outcomes)
# We merge with: oregonhie_descriptive_vars.dta (demographic controls)
#
# For reproducibility, we load from the NBER public archive.

import urllib.request
import os
import tempfile

DATA_DIR = tempfile.mkdtemp(prefix="ohie_")

def download_nber_file(filename: str, data_dir: str) -> str:
    """Download a single OHE data file from NBER if not already cached."""
    base_url = "https://data.nber.org/oregon/data/"
    local_path = os.path.join(data_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(base_url + filename, local_path)
    return local_path

# Download the core files
try:
    path_survey = download_nber_file("oregonhie_survey12m_vars.dta", DATA_DIR)
    path_descr  = download_nber_file("oregonhie_descriptive_vars.dta", DATA_DIR)
    path_admin  = download_nber_file("oregonhie_stateprograms_vars.dta", DATA_DIR)

    survey  = pd.read_stata(path_survey)
    descr   = pd.read_stata(path_descr)
    admin   = pd.read_stata(path_admin)
    df = survey.merge(descr, on="person_id", how="inner") \
               .merge(admin, on="person_id", how="inner")
    print(f"Merged dataset: {df.shape[0]:,} observations, {df.shape[1]} columns")

except Exception as e:
    # ---------------------------------------------------------------------------
    # Fallback: construct a synthetic dataset that replicates the key
    # moments from Finkelstein et al. (2012, QJE) Table 1 and Table 5.
    # This is for illustration when the NBER server is unavailable.
    # All parameters are drawn from the published paper.
    # ---------------------------------------------------------------------------
    print(f"NBER download failed ({e}). Using calibrated synthetic data.")
    np.random.seed(2008)   # Oregon lottery year

    n = 23_741              # approximate analytic sample size in Finkelstein et al.

    # Household size distribution (lottery strata)
    numhh_probs = [0.60, 0.25, 0.15]
    numhh = np.random.choice([1, 2, 3], size=n, p=numhh_probs)

    # Lottery selection probability varies by household size in OHE
    # (larger households had higher selection rates)
    p_select = {1: 0.25, 2: 0.30, 3: 0.35}
    selected = np.array([np.random.binomial(1, p_select[h]) for h in numhh])

    # Unobserved health propensity (confounder)
    # Low health_u → worse health, also → more likely to want insurance
    health_u = np.random.normal(0, 1, n)

    # Treatment: Medicaid enrollment
    # Compliers respond to lottery; always-takers enroll regardless; never-takers don't
    unit_type = np.random.choice(
        ['complier', 'always_taker', 'never_taker'],
        size=n, p=[0.25, 0.10, 0.65]
    )
    enrolled = np.where(
        unit_type == 'always_taker', 1,
        np.where(unit_type == 'complier', selected, 0)
    )

    # Outcomes
    # Calibrated to Finkelstein et al. Table 5:
    # Baseline doc_any_12m ~ 57% uninsured, ITT ~ 2.1pp, FS ~ 24.7pp → LATE ~ 8.5pp
    # Adverse selection: sicker people more likely to want insurance
    doc_any_baseline = 0.57
    adverse_selection_effect = -0.12    # sicker (lower health_u) → more likely enrolled
    treatment_effect_complier = 0.085   # LATE from paper

    # Potential outcomes
    Y0 = (np.random.uniform(size=n) <
          (doc_any_baseline + adverse_selection_effect * (-health_u)
           - 0.05 * (unit_type == 'always_taker'))).astype(float)
    Y1 = np.minimum(1.0, Y0 + treatment_effect_complier +
                    np.random.normal(0, 0.02, n))

    doc_any_12m = enrolled * Y1 + (1 - enrolled) * Y0

    # Catastrophic expense: Finkelstein et al. find ~5.1pp reduction for compliers
    # Adverse selection: sicker AND poorer → more catastrophic expense
    cat_baseline = 0.06
    Y0_cat = (np.random.uniform(size=n) <
              (cat_baseline - 0.03 * (-health_u))).astype(float)
    Y1_cat = np.maximum(0.0, Y0_cat - 0.051 + np.random.normal(0, 0.01, n))
    catastrophic_exp_inp = enrolled * Y1_cat + (1 - enrolled) * Y0_cat

    # Demographics
    age = np.random.normal(40, 10, n).clip(19, 64)
    female = np.random.binomial(1, 0.55, n)

    df = pd.DataFrame({
        'person_id':             np.arange(n),
        'selected':              selected,
        'ohp_all_ever_admin':    enrolled,
        'doc_any_12m':           doc_any_12m,
        'catastrophic_exp_inp':  catastrophic_exp_inp,
        'numhh_list':            numhh,
        'age_inp':               age,
        'female_inp':            female,
        '_Y0_doc':               Y0,     # true potential outcome (oracle)
        '_Y1_doc':               Y1,     # true potential outcome (oracle)
        '_unit_type':            unit_type,
    })

    print(f"Synthetic dataset: {df.shape[0]:,} observations")

# ---------------------------------------------------------------------------
# 2. Variable Standardization
# ---------------------------------------------------------------------------
KEY_VARS = ['selected', 'ohp_all_ever_admin', 'doc_any_12m',
            'catastrophic_exp_inp', 'numhh_list']

# Coerce to numeric; OHE .dta files encode some outcomes as labeled categoricals
for v in KEY_VARS:
    if v in df.columns:
        df[v] = pd.to_numeric(df[v], errors='coerce')

df = df.dropna(subset=['selected', 'ohp_all_ever_admin',
                        'doc_any_12m', 'numhh_list'])
print(f"Analysis sample after dropping missing: {len(df):,}")

# Create household size dummies for conditional randomization adjustment
df = pd.get_dummies(df, columns=['numhh_list'], prefix='hh', drop_first=False)
hh_dummies = [c for c in df.columns if c.startswith('hh_')]

# ---------------------------------------------------------------------------
# 3. Sample Descriptives
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TABLE 1: Sample Means by Treatment Status")
print("="*60)

desc_vars = ['doc_any_12m', 'catastrophic_exp_inp',
             'ohp_all_ever_admin', 'selected']

desc = df.groupby('ohp_all_ever_admin')[desc_vars].agg(['mean', 'std'])
print(desc.round(4))

# Enrollment rate overall
enroll_rate = df['ohp_all_ever_admin'].mean()
select_rate  = df['selected'].mean()
compliance   = df.groupby('selected')['ohp_all_ever_admin'].mean()
print(f"\nOverall enrollment rate:      {enroll_rate:.3f}")
print(f"Overall lottery selection:     {select_rate:.3f}")
print(f"Enrollment | not selected:     {compliance[0]:.3f}")
print(f"Enrollment | selected:         {compliance[1]:.3f}")
print(f"First-stage (compliance gap):  {compliance[1] - compliance[0]:.3f}")

# ---------------------------------------------------------------------------
# 4. Naive OLS: The Biased Estimator
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TABLE 2: Naive OLS — Enrolled vs. Not Enrolled")
print("="*60)

for outcome in ['doc_any_12m', 'catastrophic_exp_inp']:
    ols = smf.ols(f'{outcome} ~ ohp_all_ever_admin', data=df).fit(
        cov_type='HC2'
    )
    coef = ols.params['ohp_all_ever_admin']
    se   = ols.bse['ohp_all_ever_admin']
    pval = ols.pvalues['ohp_all_ever_admin']
    print(f"\n  Outcome: {outcome}")
    print(f"  Naive OLS coef: {coef:+.4f}  SE: {se:.4f}  p-val: {pval:.4f}")
    print(f"  Interpretation: naive gap between insured and uninsured")

# ---------------------------------------------------------------------------
# 5. Bias Decomposition (Oracle, using synthetic data)
# ---------------------------------------------------------------------------
# With real data we cannot compute this; with synthetic data we have Y0, Y1.
# This illustrates *why* the naive estimator fails.

if '_Y0_doc' in df.columns:
    print("\n" + "="*60)
    print("TABLE 3: Oracle Bias Decomposition (Synthetic Data Only)")
    print("="*60)

    D = df['ohp_all_ever_admin'].values
    Y0 = df['_Y0_doc'].values
    Y1 = df['_Y1_doc'].values
    Y_obs = df['doc_any_12m'].values

    # True ATE and ATT
    ATE = (Y1 - Y0).mean()
    ATT = (Y1[D == 1] - Y0[D == 1]).mean()
    ATU = (Y1[D == 0] - Y0[D == 0]).mean()
    p   = D.mean()

    # Naive estimator
    naive = Y_obs[D == 1].mean() - Y_obs[D == 0].mean()

    # Selection bias: E[Y0 | D=1] - E[Y0 | D=0]
    sel_bias = Y0[D == 1].mean() - Y0[D == 0].mean()

    # Heterogeneity bias: (1-p)(ATT - ATU)
    het_bias = (1 - p) * (ATT - ATU)

    print(f"\n  True ATE:                {ATE:+.4f}")
    print(f"  True ATT:                {ATT:+.4f}")
    print(f"  True ATU:                {ATU:+.4f}")
    print(f"\n  Naive estimator:         {naive:+.4f}")
    print(f"  = ATT                    {ATT:+.4f}")
    print(f"  + Selection bias         {sel_bias:+.4f}")
    print(f"\n  Naive - ATE:             {naive - ATE:+.4f}")
    print(f"  = Selection bias         {sel_bias:+.4f}")
    print(f"  + Heterogeneity bias     {het_bias:+.4f}")
    print(f"  (Check: sum)             {sel_bias + het_bias:+.4f}")

    print(f"""
  Interpretation:
    The naive gap ({naive:+.4f}) overstates the true ATE ({ATE:+.4f})
    by {naive - ATE:+.4f} pp. Selection bias accounts for {sel_bias:+.4f} pp
    (sicker people select into insurance, raising baseline doc visit rates).
    Heterogeneity bias adds {het_bias:+.4f} pp from differential treatment effects
    between the enrolled and non-enrolled populations.
""")

# ---------------------------------------------------------------------------
# 6. Covariate Balance: Validating Randomization
# ---------------------------------------------------------------------------
print("="*60)
print("TABLE 4: Covariate Balance by Lottery Selection (Z)")
print("="*60)
print("(Validates identifying assumption: Z independent of pre-treatment vars)")
print()

balance_vars = ['doc_any_12m', 'catastrophic_exp_inp']
if 'age_inp' in df.columns:
    balance_vars += ['age_inp', 'female_inp']

bal_rows = []
for v in balance_vars:
    if v not in df.columns:
        continue
    g0 = df.loc[df['selected'] == 0, v].dropna()
    g1 = df.loc[df['selected'] == 1, v].dropna()
    diff = g1.mean() - g0.mean()
    tstat, pval = stats.ttest_ind(g1, g0)
    # Normalized difference (Imbens & Rubin 2015: |norm_diff| > 0.25 is concerning)
    norm_diff = diff / np.sqrt((g0.var() + g1.var()) / 2)
    bal_rows.append({
        'Variable': v,
        'Mean (Z=0)': g0.mean(),
        'Mean (Z=1)': g1.mean(),
        'Difference': diff,
        'Norm. Diff.': norm_diff,
        'p-value': pval
    })

bal_df = pd.DataFrame(bal_rows).set_index('Variable')
print(bal_df.round(4).to_string())
print()
print("Note: All p-values should be >> 0.05 for pre-lottery outcomes if Z is")
print("      conditionally randomly assigned. Post-lottery outcomes (doc_any_12m)")
print("      will show imbalance — that is the treatment effect, not a failure.")

# ---------------------------------------------------------------------------
# 7. Intention-to-Treat Estimate (Identified)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TABLE 5: Intention-to-Treat Effect of Lottery Selection")
print("="*60)
print("(Replacing D with Z gives identified estimator under randomization)")
print()

for outcome in ['doc_any_12m', 'catastrophic_exp_inp']:
    # Condition on household size strata (lottery was stratified)
    formula = f'{outcome} ~ selected + ' + ' + '.join(hh_dummies)
    itt_model = smf.ols(formula, data=df).fit(cov_type='HC2')
    coef = itt_model.params['selected']
    se   = itt_model.bse['selected']
    ci   = itt_model.conf_int().loc['selected']
    print(f"  Outcome: {outcome}")
    print(f"  ITT:  {coef:+.4f}  SE: {se:.4f}  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print()

# ---------------------------------------------------------------------------
# 8. Manski Worst-Case Bounds (No Assumptions)
# ---------------------------------------------------------------------------
print("="*60)
print("TABLE 6: Manski Sharp Bounds on ATE (Binary Outcome)")
print("="*60)
print("(No assumptions beyond support of Y: bounds width = y_max - y_min = 1)")
print()

for outcome in ['doc_any_12m', 'catastrophic_exp_inp']:
    p_treat = df['ohp_all_ever_admin'].mean()
    E_Y_D1 = df.loc[df['ohp_all_ever_admin'] == 1, outcome].mean()
    E_Y_D0 = df.loc[df['ohp_all_ever_admin'] == 0, outcome].mean()
    y_min, y_max = 0.0, 1.0

    lb = (p_treat * E_Y_D1 + (1 - p_treat) * y_min
          - p_treat * y_max - (1 - p_treat) * E_Y_D0)
    ub = (p_treat * E_Y_D1 + (1 - p_treat) * y_max
          - p_treat * y_min - (1 - p_treat) * E_Y_D0)

    print(f"  Outcome: {outcome}")
    print(f"  Manski bounds: [{lb:.4f}, {ub:.4f}]  (width = {ub - lb:.4f})")
    print(f"  Naive estimate: {E_Y_D1 - E_Y_D0:+.4f} — point in [{lb:.4f}, {ub:.4f}]? "
          f"{'Yes' if lb <= E_Y_D1 - E_Y_D0 <= ub else 'No'}")
    print()

print("""
Interpretation:
  Without any identification assumptions, we can only say the ATE for
  doc_any_12m lies somewhere in a unit-width interval. The naive OLS
  estimate lies within these bounds (as it must), but the bounds are
  so wide as to be scientifically uninformative. Every credible
  identification strategy is a set of assumptions that narrows these
  bounds — ideally to a point. The lottery narrows them dramatically.
""")
```

---

## Summary

- **Identification is logically prior to estimation.** A consistent estimator of a non-identified parameter consistently estimates the wrong quantity. Choosing between clustered and robust standard errors without first establishing identification is premature optimization.

- **The fundamental problem is the missing counterfactual.** We observe $Y_i(D_i)$ but not $Y_i(1 - D_i)$. Causal inference is the principled imputation of missing potential outcomes under explicit assumptions.

- **SUTVA is load-bearing.** No-interference and no-hidden-versions are both required for the potential outcomes notation to be well-defined. Both are contestable in practice and must be argued for, not assumed silently.

- **The naive estimator conflates ATE, selection bias, and heterogeneity bias.** The decomposition $\text{Naive} = \text{ATT} + \text{Selection Bias}$ and $\text{Naive} - \text{ATE} = \text{Selection Bias} + (1-p)(\text{ATT} - \text{ATU})$ shows that two distinct failure modes are present in observational comparisons, not one.

- **The potential outcomes and SCM frameworks are complementary, not competing.** Potential outcomes notation is sharp for defining estimands; Pearl's do-calculus is powerful for reasoning about identification in complex systems with many variables and mediated effects.

- **Randomization identifies the ATE by eliminating selection bias.** The OHE lottery conditionally randomizes insurance access, making the ITT effect identified without functional form assumptions. The gap between the naive estimate and the ITT estimate is a direct measure of selection bias in the Oregon data.

- **Without assumptions, only wide Manski bounds are available.** Every identification strategy is an assumption that shrinks these bounds. The scientific question is always whether the assumptions are credible, not whether they produce tighter bounds.

---

## Further Reading

**Rubin, D.B. (1974).** "Estimating causal effects of treatments in randomized and nonrandomized studies." *Journal of Educational Psychology*, 66(5), 688–701. The foundational paper for the potential outcomes framework; introduces the notation $Y_i(t)$ and the concept of assignment mechanisms.

**Pearl, J. (2000).** *Causality: Models, Reasoning, and Inference*. Cambridge University Press. Full development of the SCM framework, do-calculus, and graphical identification criteria including the back-door and front-door criteria. Chapters 1–3 are essential for anyone reasoning about identification in multi-variable systems.

**Finkelstein, A., Taubman, S., Wright, B., et al. (2012).** "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics*, 127(3), 1057–1106. The primary paper using OHE data; introduces the lottery design, constructs the ITT and IV estimates, and provides the canonical benchmark for all subsequent OHE analyses.

**Manski, C.F. (1990).** "Nonparametric Bounds on Treatment Effects." *American Economic Review Papers and Proceedings*, 80(2), 319–323. Derives the sharp bounds on treatment effects identified under minimal assumptions; foundational for the partial identification literature.

**Angrist, J.D. and Pischke, J.-S. (2010).** "The Credibility Revolution in Empirical Economics: How Better Research Design is Taking the Con out of Econometrics." *Journal of Economic Perspectives*, 24(2), 3–30. A lucid manifesto for design-based identification; provides historical context for the shift from structural modeling to quasi-experimental methods and argues for transparent, assumption-light designs.

**Imbens, G.W. and Rubin, D.B. (2015).** *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. The most comprehensive modern treatment of the potential outcomes framework; Chapters 1–4 develop SUTVA, randomization, and the fundamental identification problem at a level of rigor appropriate for this book's audience.
# Chapter 36: External Validity and Transportability

## 36.1 The Problem of External Validity

Every causal estimate is an estimate *somewhere*. The Oregon Health Insurance Experiment recovered a local average treatment effect for Medicaid lottery compliers in 2008 Oregon — a specific, well-defined population. The question of whether that number means anything for a different state, a later year, or a policy expansion that reaches non-compliers is a *separate* empirical question, not a logical consequence of the internal identification result.

This distinction — between **internal validity** (correct identification in the study sample) and **external validity** (relevance of the estimate for a target population) — is frequently acknowledged and rarely formalized. The goal of this chapter is to make it precise.

Two related but distinct problems arise in practice:

**Generalizability** (also called *external validity* in the narrow sense): the study population is a non-representative sample of a well-defined target population. You want to recover the target population ATE from a study whose participants were selected non-randomly from that same population. The Oregon lottery enrolled participants from Medicaid waitlist registrants — a subset of low-income Oregonians.

**Transportability**: the study population and the target population are *structurally distinct* — different states, different time periods, different health systems. You want to use estimates from one domain to inform causal claims in another. Transporting OHE estimates to California or to the 2014 ACA expansion population is a transportability problem.

Both require explicit assumptions about *which* variables mediate the difference between populations, and both admit graphical formalization via selection diagrams.

## 36.2 Selection Diagrams and S-Variables

The formal framework for transportability is due to Bareinboim and Pearl. A **selection diagram** augments the causal DAG with **selection variables** $S_i$, drawn as square nodes with no parents, that index which domain a unit belongs to. An arrow $S \to V$ in the selection diagram indicates that the distribution of $V$ differs across domains — not necessarily its structural equation, but its marginal or conditional distribution.

Let $\Pi$ be a target domain and $\pi^*$ be the study domain. A selection diagram $\mathcal{D}$ over variables $(V_1, \ldots, V_k, S)$ encodes the claim that for any node $V_j$ with no $S$-arrow, the conditional $P(V_j | \text{pa}(V_j))$ is invariant across domains; where $S \to V_j$ appears, that conditional may differ.

**Definition 36.1 (S-admissible set).** A set $X$ of covariates is *S-admissible* for transporting $E[Y(d)]$ from $\pi^*$ to $\Pi$ if, in the selection diagram $\mathcal{D}$, conditioning on $X$ blocks all paths from $S$ to $Y$ that do not pass through $D$. Formally, $(Y(d) \perp\!\!\!\perp S \mid X)$ in $\mathcal{D}_{\overline{D}}$, the diagram with incoming arrows to $D$ removed.

When an S-admissible set exists, the transport formula gives identification.

**Theorem 36.1 (Transport Formula).** Suppose $X$ is S-admissible for transporting $E_\Pi[Y(d)]$ from $\pi^*$ to $\Pi$. Then:

$$E_\Pi[Y(d)] = \sum_x E_{\pi^*}[Y \mid D=d, X=x] \cdot P_\Pi(X=x)$$

where $E_{\pi^*}[\cdot]$ denotes expectations computed in the study population and $P_\Pi(\cdot)$ denotes covariate distributions in the target population.

*Proof sketch.* By S-admissibility, $Y(d) \perp\!\!\!\perp S \mid X$. This means:

$$P_\Pi(Y(d) = y \mid X=x) = P_{\pi^*}(Y(d) = y \mid X=x)$$

The potential outcome conditional on $X$ is the same in both domains. The transport formula then follows from:

$$E_\Pi[Y(d)] = \sum_x E_\Pi[Y(d) \mid X=x] \cdot P_\Pi(X=x) = \sum_x E_{\pi^*}[Y(d) \mid X=x] \cdot P_\Pi(X=x)$$

Under ignorability of $D$ given $X$ within $\pi^*$, or instrument validity within $\pi^*$, the inner expectation $E_{\pi^*}[Y(d) \mid X=x]$ is identified from study data. $\square$

The theorem has a clean interpretation: the structural relationship between treatment and outcome, conditional on the right covariates, is portable. What changes across populations is the covariate distribution, and that is directly observed in the target.

Note the asymmetry: we borrow *causal relationships* from the study and *covariate distributions* from the target. This is precisely backwards from naive extrapolation, which often borrows the aggregate effect rather than the conditional one.

## 36.3 Inverse Probability of Sampling Weights

The transport formula has a natural weighting implementation. Define a combined sample $\mathcal{S} = \mathcal{S}_{\pi^*} \cup \mathcal{S}_\Pi$ where $\mathcal{S}_\Pi$ consists of covariate-only observations from the target population (e.g., census records, survey respondents). Introduce the sampling indicator $S_i = 1$ if unit $i$ is from the study, $S_i = 0$ from the target.

The **inverse probability of sampling weight (IPSW)** for unit $i$ in the study is:

$$w_i = \frac{P_\Pi(X_i)}{P_{\pi^*}(X_i)} = \frac{P(S=0 \mid X_i)}{P(S=1 \mid X_i)} \cdot \frac{P(S=1)}{P(S=0)}$$

The second form follows from Bayes' theorem and makes the estimator operational: fit a classifier to predict $S$ from $X$ in the combined sample, then reweight study units by their odds of being in the target.

The **IPSW transported ATE estimator** is:

$$\hat{\tau}_\Pi^{\text{IPSW}} = \frac{\sum_{i \in \mathcal{S}_{\pi^*}} w_i \hat{\tau}(X_i)}{\sum_{i \in \mathcal{S}_{\pi^*}} w_i}$$

where $\hat{\tau}(X_i)$ is a CATE estimate. In the IV setting with binary $D$, $\hat{\tau}(X_i)$ is the stratum-specific LATE, and the transported quantity is a weighted average of stratum-specific LATEs under the target covariate distribution.

**Overlap requirement.** The IPSW estimator is consistent only if $P_\Pi(X=x) > 0 \Rightarrow P_{\pi^*}(X=x) > 0$ for all $x$ in the support of the target. Covariate values present in the target but absent from the study produce infinite weights and the transport formula breaks down — not just numerically but conceptually. There is no study evidence about those units.

**Variance of IPSW estimator.** Under standard regularity conditions, $\hat{\tau}_\Pi^{\text{IPSW}}$ is $\sqrt{n}$-consistent with asymptotic variance that has two components: variance of the CATE estimator (inflated by the weights) and variance from estimating the propensity score $P(S=1 \mid X)$. Bootstrap is the practical approach; the sandwich estimator accounting for first-stage weight estimation is exact but rarely implemented.

**Stabilized weights.** Raw IPSW weights can be highly variable when study and target distributions differ substantially. Truncating weights at the 95th or 99th percentile trades variance reduction for bias; alternatively, entropy balancing minimizes weight dispersion subject to moment constraints. Both are preferable to raw IPSW when the effective sample size $n_\text{eff} = (\sum w_i)^2 / \sum w_i^2$ falls well below $n_{\pi^*}$.

## 36.4 Generalizability vs. Transportability in Formal Terms

The transport formula (Theorem 36.1) covers both generalizability and transportability but with different selection diagram structures.

**Generalizability** (single population, biased sample): the selection variable $S$ encodes enrollment in the study. Arrows $S \to X_j$ for covariates $X_j$ that predict study participation capture the sampling mechanism. No $S \to Y$ arrow directly — all selection bias operates through measured covariates. This is the standard "missing at random" structure for potential outcomes.

**Transportability** (two distinct domains): the selection variable $S$ indexes *domain membership*, and arrows $S \to V_j$ may appear on any node, including unmeasured effect modifiers. If $S \to U$ for an unmeasured variable $U$ that moderates the treatment effect, then no observed $X$ is S-admissible and the transport formula does not apply — transportability fails.

This gives a precise account of when transportability fails: when domain differences operate through variables that moderate the causal effect and are not observed in both domains. The selection diagram makes this a testable (or at least explicit) assumption rather than a vague appeal to "similar populations."

**Partial transportability.** When full transportability fails, it may still be possible to transport conditional effects for a subset of the population where overlap holds, or to bound the transported ATE using the methods of Chapter 33. Bareinboim and Pearl also establish results for *partial identification* under incomplete selection diagrams — the transported quantity is set-identified rather than point-identified.

## 36.5 MTE-Based Reweighting and the Policy-Relevant Treatment Effect

The marginal treatment effect framework (Chapter 22) provides a complementary approach that is particularly natural for transporting IV estimates. Recall:

$$\text{MTE}(u_D) = E[Y(1) - Y(0) \mid U_D = u_D]$$

where $U_D \sim \text{Uniform}(0,1)$ is the latent resistance to treatment. The LATE identified by instrument $Z$ recovers:

$$\text{LATE} = \int_0^1 \text{MTE}(u) \cdot \mathbf{1}[p(Z=0) \leq u \leq p(Z=1)] \, du / [p(Z=1) - p(Z=0)]$$

that is, an average over the complier range of $u_D$.

The **Policy-Relevant Treatment Effect (PRTE)** for a target policy $\pi$ that induces propensity score $p_\pi(X)$ in the target population is:

$$\text{PRTE} = \frac{\int \text{MTE}(u_D) [P_\pi(U_D \leq u_D) - P_{\pi^*}(U_D \leq u_D)] \, du_D}{P_\pi(D=1) - P_{\pi^*}(D=1)}$$

This is a reweighting of the MTE by the change in the propensity score distribution induced by the policy change. The numerator integrates MTE against the difference in CDF of treatment propensities between the policy and the status quo.

**Connection to transportability.** The PRTE via MTE is a form of transport: we are asking what the effect would be in a population where the selection-into-treatment mechanism has changed. If $\text{MTE}(u_D)$ is constant (no selection on gains), PRTE equals ATE regardless of the policy, and IV estimates transport freely. If MTE is declining in $u_D$ (those most likely to comply benefit most — positive selection), then expanding coverage to higher-resistance units yields smaller effects, and the OHE LATE overstates the PRTE for the ACA expansion.

**Estimation of PRTE.** The MTE is identified from local IV, $\partial E[Y \mid Z=z] / \partial p(z)$ evaluated at the propensity score. In practice, estimate $\hat{p}(Z)$ by first-stage regression, then run a partially linear regression of $Y$ on $D$ and a flexible function of $\hat{p}$; the derivative of the flexible component with respect to $\hat{p}$ estimates MTE. The target distribution $P_\pi(U_D)$ is constructed from census or survey marginals.

## 36.6 Manski Bounds for Non-Identified Transport

When the S-admissibility condition fails, or when the support of $P_\Pi(X)$ extends beyond the study support, the transported ATE is not point-identified. Chapter 33 developed sharp bounds under no assumptions (Manski) and under monotone treatment response. Those bounds apply directly here.

Let $\mathcal{X}_\Pi$ partition into $\mathcal{X}_\text{overlap} = \{x : P_{\pi^*}(X=x) > 0\}$ and $\mathcal{X}_\text{no-overlap} = \{x : P_{\pi^*}(X=x) = 0\}$.

For $x \in \mathcal{X}_\text{overlap}$, the transport formula identifies $E_{\pi^*}[Y(d) \mid X=x]$. For $x \in \mathcal{X}_\text{no-overlap}$, $Y(d)$ is bounded by $[y_{\min}, y_{\max}]$ (or tighter under MTR/MTS). The transported ATE bounds are:

$$\tau_\Pi^\text{lo} = \sum_{x \in \mathcal{X}_\text{overlap}} \hat{\tau}(x) P_\Pi(x) + y_{\min} \cdot P_\Pi(\mathcal{X}_\text{no-overlap})$$

$$\tau_\Pi^\text{hi} = \sum_{x \in \mathcal{X}_\text{overlap}} \hat{\tau}(x) P_\Pi(x) + y_{\max} \cdot P_\Pi(\mathcal{X}_\text{no-overlap})$$

The bound width equals $(y_{\max} - y_{\min}) \cdot P_\Pi(\mathcal{X}_\text{no-overlap})$, which is the probability mass in the target assigned to covariate values with no study representation. This makes the cost of non-overlap concrete: if 20% of the target falls outside study support and the outcome range is $[0,1]$, the bounds have width 0.20 regardless of how good the study is.

## Python: Transporting OHE LATE to the Oregon General Population

The following implementation estimates the OHE LATE, constructs a synthetic target population from census-calibrated marginals, estimates IPSW weights, computes the transported ATE, derives the PRTE via a local IV polynomial estimator, and reports Manski bounds for the non-overlap region.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load OHE data ──────────────────────────────────────────────────────────
# Download from https://data.nber.org/oregon/
# Here we load the 12-month survey data merged with administrative enrollment
ohe = pd.read_stata(
    "~/data/oregon/oregonhie_survey12m_vars.dta",
    columns=[
        "person_id",
        "selected",          # Z: lottery instrument (1=selected for Medicaid)
        "ohp_all_ever_admin",# D: ever enrolled in OHC (administrative)
        "doc_any_12m",       # Y1: any doctor visit in past 12 months
        "catastrophic_exp_inp",# Y2: catastrophic medical expenditure
        "numhh_list",        # strata: household size at time of lottery
        "age_inp",           # covariate
        "female_inp",        # covariate
        "english_inp",       # covariate
        "self_list",         # self-enrolled (primary lottery applicant)
    ]
).dropna()

ohe["Z"] = ohe["selected"].astype(float)
ohe["D"] = ohe["ohp_all_ever_admin"].astype(float)
ohe["Y1"] = ohe["doc_any_12m"].astype(float)
ohe["Y2"] = ohe["catastrophic_exp_inp"].astype(float)

# Household-size dummies for strata (OHE protocol)
ohe = pd.get_dummies(ohe, columns=["numhh_list"], drop_first=True, dtype=float)
strata_cols = [c for c in ohe.columns if "numhh_list" in c]

cov_cols = ["age_inp", "female_inp", "english_inp"] + strata_cols
X_study = ohe[cov_cols].values
Z = ohe["Z"].values
D = ohe["D"].values
Y1 = ohe["Y1"].values
Y2 = ohe["Y2"].values

n_study = len(ohe)
print(f"Study n = {n_study:,}")

# ── 2. LATE via 2SLS (internal) ───────────────────────────────────────────────
# First stage: P(D=1 | Z, X)
from numpy.linalg import lstsq

def tsls(Y, D, Z, X):
    """2SLS with controls X, instrument Z, treatment D."""
    ZX = np.column_stack([Z, X, np.ones(len(Y))])
    DX = np.column_stack([D, X, np.ones(len(Y))])
    # First stage
    gamma, *_ = lstsq(ZX, D, rcond=None)
    D_hat = ZX @ gamma
    # Second stage: replace D with D_hat
    DX_hat = np.column_stack([D_hat, X, np.ones(len(Y))])
    beta, *_ = lstsq(DX_hat, Y, rcond=None)
    return beta[0]  # coefficient on D

late_y1 = tsls(Y1, D, Z, X_study)
late_y2 = tsls(Y2, D, Z, X_study)
print(f"OHE LATE (doc visit):        {late_y1:.4f}")
print(f"OHE LATE (catastrophic exp): {late_y2:.4f}")

# ── 3. Simulate target population from census marginals ───────────────────────
# Oregon low-income population (below 138% FPL) marginals:
#   age:     mean=38, sd=10, truncated [18,65]
#   female:  55%
#   english: 72%
#   numhh:   multinomial [1,2,3+] = [0.45, 0.32, 0.23]
rng = np.random.default_rng(42)
n_target = 50_000

age_t = np.clip(rng.normal(38, 10, n_target), 18, 65)
female_t = rng.binomial(1, 0.55, n_target).astype(float)
english_t = rng.binomial(1, 0.72, n_target).astype(float)
numhh_t = rng.choice([1, 2, 3], size=n_target, p=[0.45, 0.32, 0.23])

# Match strata dummy coding to study
target_df = pd.DataFrame({
    "age_inp": age_t,
    "female_inp": female_t,
    "english_inp": english_t,
    "numhh_list": numhh_t.astype(str),
})
target_df = pd.get_dummies(target_df, columns=["numhh_list"], drop_first=True, dtype=float)
# Align columns
for c in strata_cols:
    if c not in target_df.columns:
        target_df[c] = 0.0
target_df = target_df[[c for c in cov_cols if c in target_df.columns]].reindex(
    columns=cov_cols, fill_value=0.0
)
X_target = target_df.values

print(f"\nTarget population n = {n_target:,}")
print(f"Age: study mean={X_study[:,0].mean():.1f}, target mean={X_target[:,0].mean():.1f}")
print(f"Female: study={X_study[:,1].mean():.2f}, target={X_target[:,1].mean():.2f}")

# ── 4. IPSW weights ───────────────────────────────────────────────────────────
# Pool study (S=1) and target (S=0); fit classifier P(S=1 | X)
X_pool = np.vstack([X_study, X_target])
S_pool = np.concatenate([np.ones(n_study), np.zeros(n_target)])

scaler = StandardScaler()
X_pool_scaled = scaler.fit_transform(X_pool)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
lr.fit(X_pool_scaled, S_pool)

# P(S=1 | X) for study units
X_study_scaled = scaler.transform(X_study)
ps1 = lr.predict_proba(X_study_scaled)[:, 1]   # P(S=1 | X_i)
ps0 = 1 - ps1                                  # P(S=0 | X_i)

# IPSW = P(S=0|X)/P(S=1|X) * P(S=1)/P(S=0)
p1 = n_study / (n_study + n_target)
p0 = n_target / (n_study + n_target)
raw_weights = (ps0 / ps1) * (p1 / p0)

# Clip at 99th percentile for stability
w_clip = np.clip(raw_weights, 0, np.percentile(raw_weights, 99))
w_norm = w_clip / w_clip.mean()   # normalize

ess = (w_norm.sum())**2 / (w_norm**2).sum()
print(f"\nIPSW summary:")
print(f"  Raw weight range: [{raw_weights.min():.2f}, {raw_weights.max():.2f}]")
print(f"  Effective sample size: {ess:.0f} / {n_study} ({100*ess/n_study:.0f}%)")

# ── 5. Stratum-specific LATE and IPSW-transported ATE ────────────────────────
from sklearn.linear_model import LinearRegression

def stratum_late(Y, D, Z, X, weights=None):
    """
    Compute weighted 2SLS. Returns scalar transported effect.
    For IPSW: pass study-to-target weights as `weights`.
    """
    n = len(Y)
    w = weights if weights is not None else np.ones(n)
    w = w / w.mean()

    # Weighted first stage: D ~ Z + X
    ZX = np.column_stack([Z, X, np.ones(n)])
    Wsqrt = np.sqrt(w)
    gamma, *_ = lstsq((ZX.T * w).T, w * D, rcond=None)
    D_hat = ZX @ gamma

    # Weighted second stage: Y ~ D_hat + X
    DX_hat = np.column_stack([D_hat, X, np.ones(n)])
    beta, *_ = lstsq((DX_hat.T * w).T, w * Y, rcond=None)
    return beta[0]

late_transported_y1 = stratum_late(Y1, D, Z, X_study, weights=w_norm)
late_transported_y2 = stratum_late(Y2, D, Z, X_study, weights=w_norm)

print(f"\nTransported ATE estimates:")
print(f"  Unweighted LATE (doc visit):        {late_y1:.4f}")
print(f"  IPSW-transported ATE (doc visit):   {late_transported_y1:.4f}")
print(f"  Unweighted LATE (catastrophic):     {late_y2:.4f}")
print(f"  IPSW-transported ATE (catastrophic):{late_transported_y2:.4f}")

# ── 6. Bootstrap CI for transported ATE ──────────────────────────────────────
B = 500
boot_y1 = np.empty(B)
boot_y2 = np.empty(B)

for b in range(B):
    idx = rng.integers(0, n_study, n_study)
    Yb1, Yb2, Db, Zb = Y1[idx], Y2[idx], D[idx], Z[idx]
    Xb = X_study[idx]
    wb = w_norm[idx]
    try:
        boot_y1[b] = stratum_late(Yb1, Db, Zb, Xb, weights=wb)
        boot_y2[b] = stratum_late(Yb2, Db, Zb, Xb, weights=wb)
    except Exception:
        boot_y1[b] = np.nan
        boot_y2[b] = np.nan

ci_y1 = np.nanpercentile(boot_y1, [2.5, 97.5])
ci_y2 = np.nanpercentile(boot_y2, [2.5, 97.5])
print(f"\n95% bootstrap CI (transported, doc visit): [{ci_y1[0]:.4f}, {ci_y1[1]:.4f}]")
print(f"95% bootstrap CI (transported, catastrophic): [{ci_y2[0]:.4f}, {ci_y2[1]:.4f}]")

# ── 7. MTE estimation and PRTE ───────────────────────────────────────────────
# Estimate P(D=1 | Z, X) for each unit (propensity score for treatment)
from sklearn.linear_model import LogisticRegression as LR

lr_first = LR(C=1.0, max_iter=500, random_state=1)
lr_first.fit(np.column_stack([Z, X_study]), D)
p_hat = lr_first.predict_proba(np.column_stack([Z, X_study]))[:, 1]

# Estimate MTE via polynomial local IV
# Regress Y on D, p_hat, and poly(p_hat, degree=3)
from numpy.polynomial import polynomial as P

def mte_polynomial(Y, D, p, degree=3):
    """
    Local IV polynomial estimator of MTE.
    Returns: (mte_fn, coef) where mte_fn(u) evaluates MTE at u in [0,1].
    """
    poly_p = np.column_stack([p**k for k in range(1, degree+1)])
    X_reg = np.column_stack([D, poly_p, np.ones(len(Y))])
    coef, *_ = lstsq(X_reg, Y, rcond=None)

    # MTE(u) = dE[Y|p]/dp = sum_k k*b_k * u^{k-1}  (derivative of polynomial in p)
    # coef[0] = beta_D, coef[1:degree+1] = beta_poly, coef[-1] = intercept
    poly_coefs = coef[1:degree+1]  # coefficients on p, p^2, ..., p^degree

    def mte_fn(u):
        # derivative of sum_k beta_k * u^k wrt u
        return sum(poly_coefs[k-1] * k * u**(k-1) for k in range(1, degree+1))

    return mte_fn, coef

mte_fn_y1, _ = mte_polynomial(Y1, D, p_hat, degree=3)
mte_fn_y2, _ = mte_polynomial(Y2, D, p_hat, degree=3)

# PRTE: reweight MTE by target vs study propensity distribution
# Study propensity: p_hat distribution in study (instrument at Z=0 vs Z=1)
# Target policy: full coverage p_pi = 0.8 (ACA expansion-like)
# ACA expansion targets everyone below 138% FPL → p_pi(X) ≈ 0.8

p_study_mean = p_hat.mean()   # ~0.25 in Oregon lottery
p_target_policy = 0.80        # ACA-like expansion target

u_grid = np.linspace(0.01, 0.99, 200)
mte_y1 = np.array([mte_fn_y1(u) for u in u_grid])
mte_y2 = np.array([mte_fn_y2(u) for u in u_grid])

# Weight = I[u <= p_target] - I[u <= p_study_mean]  (difference of CDFs)
# PRTE numerator = integral MTE * (F_target(u) - F_study(u)) du
# For point masses at p_target_policy and p_study_mean:
# Delta CDF = 1(u <= p_target_policy) - 1(u <= p_study_mean)
delta_cdf = (u_grid <= p_target_policy).astype(float) - (u_grid <= p_study_mean).astype(float)
du = u_grid[1] - u_grid[0]

prte_y1 = np.trapz(mte_y1 * delta_cdf, u_grid) / (p_target_policy - p_study_mean)
prte_y2 = np.trapz(mte_y2 * delta_cdf, u_grid) / (p_target_policy - p_study_mean)

print(f"\nMTE-based PRTE (policy: p = {p_target_policy}):")
print(f"  PRTE (doc visit):        {prte_y1:.4f}")
print(f"  PRTE (catastrophic exp): {prte_y2:.4f}")
print(f"  (vs LATE doc visit: {late_y1:.4f}, LATE catastrophic: {late_y2:.4f})")

# MTE shape diagnostic
print(f"\nMTE shape (doc visit) at quantiles of p_hat:")
for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
    u = np.quantile(p_hat, q)
    print(f"  u = {u:.3f}: MTE = {mte_fn_y1(u):.4f}")

# ── 8. Manski bounds for non-overlap region ──────────────────────────────────
# Identify study support via convex hull proxy: percentile ranges per covariate
# A target unit is "out of support" if any covariate exceeds study range
support_mask = np.ones(n_target, dtype=bool)
for j in range(X_target.shape[1]):
    lo, hi = X_study[:, j].min(), X_study[:, j].max()
    support_mask &= (X_target[:, j] >= lo) & (X_target[:, j] <= hi)

frac_no_overlap = 1 - support_mask.mean()
print(f"\nNon-overlap fraction of target: {100*frac_no_overlap:.1f}%")

# Y1 ∈ {0,1}, so y_min=0, y_max=1
y_min, y_max = 0.0, 1.0

# Transported estimate uses overlap region only
late_overlap = stratum_late(Y1, D, Z, X_study, weights=w_norm)  # already computed

tau_lo = late_overlap * (1 - frac_no_overlap) + y_min * frac_no_overlap
tau_hi = late_overlap * (1 - frac_no_overlap) + y_max * frac_no_overlap
bound_width = tau_hi - tau_lo

print(f"Manski bounds for transported ATE (doc visit):")
print(f"  Lower: {tau_lo:.4f}")
print(f"  Upper: {tau_hi:.4f}")
print(f"  Width: {bound_width:.4f} (= y_range × P(no overlap) = 1.0 × {frac_no_overlap:.3f})")

# ── 9. Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Estimand':<40} {'Doc Visit':>10} {'Catastrophic':>12}")
print("="*65)
print(f"{'OHE LATE (internal)':<40} {late_y1:>10.4f} {late_y2:>12.4f}")
print(f"{'IPSW transported ATE':<40} {late_transported_y1:>10.4f} {late_transported_y2:>12.4f}")
print(f"{'MTE-based PRTE (ACA-like expansion)':<40} {prte_y1:>10.4f} {prte_y2:>12.4f}")
print(f"{'Manski lower (doc visit)':<40} {tau_lo:>10.4f} {'':>12}")
print(f"{'Manski upper (doc visit)':<40} {tau_hi:>10.4f} {'':>12}")
print("="*65)
```

The output table separates three conceptually distinct estimands. The OHE LATE is the effect on compliers in the 2008 lottery sample. The IPSW transported ATE reweights that estimate to the Oregon low-income population marginals. The MTE-based PRTE asks what the effect would be for units drawn into coverage by an ACA-like policy — those with higher resistance to enrollment, which the MTE shape will typically show have smaller effects if selection on gains is positive. The Manski bounds quantify the irreducible uncertainty from target units with covariate profiles unseen in the study.

## 36.7 Sensitivity Analysis for Transport Assumptions

The S-admissibility assumption is fundamentally untestable from study data alone: it asserts that there are no unmeasured variables both causing domain membership and moderating the treatment effect. Sensitivity analysis asks how large such violations would need to be to overturn the conclusion.

**Calibrated sensitivity parameter.** Following the approach of Zhao, Small, and Bhattacharya, define a sensitivity parameter $\Gamma$ such that the true transport weights satisfy $w_i / \Gamma \leq w_i^* \leq w_i \cdot \Gamma$ for all $i$. When $\Gamma = 1$, S-admissibility holds. The range of transported ATEs over all weight perturbations bounded by $\Gamma$ gives sensitivity bounds. These are computable via linear programming over the weight perturbation set.

**Benchmarking against observed covariates.** A more interpretable approach: drop one covariate at a time from the IPSW model and measure how much the transported estimate changes. If dropping $X_j$ shifts the estimate by $\Delta_j$, then an unmeasured confounder with a selection association comparable to $X_j$ would shift the estimate by approximately $\Delta_j$. This benchmarks the sensitivity parameter against the observed covariates, making it substantively interpretable.

In the OHE application, age and household size are the strongest predictors of study participation. If an unmeasured variable (e.g., chronic illness status) has a selection association of similar magnitude, the benchmarked sensitivity analysis quantifies the resulting bias in the transported estimate.

## Summary

- **Internal validity** establishes that a causal effect is correctly estimated in the study sample; **external validity** concerns whether that estimate is relevant for a different population. These are logically independent claims.
- **Selection diagrams** formalize the sources of cross-domain heterogeneity via $S$-variables. An S-admissible set $X$ satisfies $(Y(d) \perp\!\!\!\perp S \mid X)$ and supports the transport formula $E_\Pi[Y(d)] = \sum_x E_{\pi^*}[Y|D=d,X=x]P_\Pi(X=x)$.
- **IPSW** operationalizes the transport formula by fitting a classifier to distinguish study from target units and reweighting study observations by the resulting odds ratio. Effective sample size diagnostics and weight stabilization are essential.
- **The MTE-based PRTE** provides a richer decomposition: it identifies which marginal compliers are added by a policy change and weights the MTE accordingly. Positive selection on gains implies LATE overstates the PRTE for coverage expansions.
- **Transportability fails** when domain differences operate through unmeasured effect modifiers — a claim not verifiable from data but made precise by the selection diagram structure.
- **Manski bounds** recover partial identification when the target covariate support exceeds the study support. Bound width equals $(y_{\max} - y_{\min}) \times P_\Pi(\mathcal{X}_\text{no-overlap})$, making the cost of non-overlap explicit.
- **Sensitivity analysis** via weight perturbation bounds or covariate-benchmarking assesses the robustness of transported estimates to violations of S-admissibility.

## Further Reading

1. **Bareinboim, E. and Pearl, J. (2016).** "Causal inference and the data-fusion problem." *PNAS* 113(27): 7345–7352. The definitive statement of the transportability framework, selection diagrams, and the do-calculus conditions for cross-domain identification. Read alongside the 2012 UAI paper for the original transportability theorems.

2. **Stuart, E. A., Cole, S. R., Bradshaw, C. P., and Leaf, P. J. (2011).** "The use of propensity scores to assess the generalizability of results from randomized trials." *Journal of the Royal Statistical Society: Series A* 174(2): 369–386. Practical IPSW methodology for generalizing RCT results to target populations, with careful attention to overlap diagnostics and effective sample size.

3. **Heckman, J. J., Urzua, S., and Vytlacil, E. (2006).** "Understanding instrumental variables in models with essential heterogeneity." *Review of Economics and Statistics* 88(3): 389–432. Full development of the MTE framework with PRTE derivations; shows how different estimands (ATE, ATT, LATE, PRTE) relate as different weighted averages of MTE.

4. **Finkelstein, A., Taubman, S., Wright, B., et al. (2012).** "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics* 127(3): 1057–1106. Primary OHE results paper. Essential reading for understanding the complier population, lottery design, and outcome measurement that grounds the running example.

5. **Dahabreh, I. J. and Hernán, M. A. (2019).** "Extending inferences from a randomized trial to a new target population." *Statistics in Medicine* 38(20): 3838–3851. Counterfactual framework for generalizability with doubly-robust estimators; bridges the Pearl selection-diagram language and the Rubin-model IPSW literature.

6. **Zhao, Q., Small, D. S., and Bhattacharya, B. B. (2019).** "Sensitivity analysis for inverse probability weighting estimators via the percentile bootstrap." *Journal of the Royal Statistical Society: Series B* 81(4): 735–761. Provides the calibrated sensitivity analysis for IPSW estimators used in Section 36.7; the percentile bootstrap approach is directly implementable with the code in this chapter.
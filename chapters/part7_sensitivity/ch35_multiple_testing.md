# Chapter 35: Multiple Testing, Researcher Degrees of Freedom, and Specification Search

---

A credible causal study reports not one outcome but many: multiple endpoints, multiple subgroups, multiple specifications. Each additional test is an opportunity for a false positive. Each unreported specification is a researcher choice that could have gone another way. This chapter develops the statistical machinery for disciplined inference under multiplicity, and the diagnostic tools—specification curves and multiverse analysis—that expose how much researcher flexibility drives reported results.

---

## 35.1 The Multiplicity Problem

Begin with the simplest case. You have $m$ independent null hypotheses $H_1, \ldots, H_m$, each tested at level $\alpha = 0.05$. If all nulls are true, the probability that at least one is falsely rejected is:

$$\Pr(\text{at least one false rejection}) = 1 - (1-\alpha)^m$$

For $m = 5$ this is $0.226$; for $m = 20$ it is $0.64$. The Oregon Health Insurance Experiment reports outcomes across at least eight domain categories. If we test one hypothesis per domain without adjustment, we should expect roughly one spurious rejection even if the intervention has no effect on anything.

**Definition 35.1 (Family-Wise Error Rate).** The *family-wise error rate* (FWER) of a testing procedure applied to a family $\mathcal{H} = \{H_1, \ldots, H_m\}$ is:

$$\text{FWER} = \Pr\!\left(\text{reject at least one } H_k \mid H_k \text{ true for all } k\right)$$

Strong FWER control requires $\text{FWER} \leq \alpha$ regardless of which subset of nulls are true. Weak control requires it only when all nulls are true.

**Definition 35.2 (False Discovery Rate).** Let $V$ be the number of false rejections and $R$ the total number of rejections. The *false discovery rate* is:

$$\text{FDR} = \mathbb{E}\!\left[\frac{V}{\max(R,1)}\right]$$

FDR $\leq$ FWER always; under the global null, FDR = FWER. FDR is the right criterion when discoveries are the unit of inference and a small proportion of false positives is tolerable—genomics is the canonical example. Causal economics studies, where each false positive may inform policy, often warrant the stricter FWER criterion.

---

## 35.2 Bonferroni and Holm Corrections

**Theorem 35.1 (Bonferroni).** Reject $H_k$ iff $p_k \leq \alpha/m$. This procedure controls FWER $\leq \alpha$ strongly.

*Proof.* Under the global null, by Boole's inequality:

$$\text{FWER} = \Pr\!\left(\bigcup_{k=1}^m \{p_k \leq \alpha/m\}\right) \leq \sum_{k=1}^m \Pr(p_k \leq \alpha/m) = m \cdot \frac{\alpha}{m} = \alpha$$

Under a partial null (only $m_0 \leq m$ nulls true), the bound still holds because we sum over fewer than $m$ events. $\square$

Bonferroni is exact when tests are positively correlated (Simes' inequality) and conservative otherwise. Its power penalty grows with $m$.

**Theorem 35.2 (Holm Stepdown).** Order the $p$-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$. Let $k^* = \min\{k : p_{(k)} > \alpha/(m-k+1)\}$. Reject $H_{(1)}, \ldots, H_{(k^*-1)}$. This procedure controls FWER $\leq \alpha$ strongly and is uniformly more powerful than Bonferroni.

*Proof sketch.* Let $\mathcal{H}_0 \subseteq \mathcal{H}$ be the set of true nulls with $|\mathcal{H}_0| = m_0$. A false rejection occurs only if some $H_{(k)} \in \mathcal{H}_0$ is rejected. At the step where we consider $H_{(k)}$, at most $m_0$ true nulls remain, so the threshold $\alpha/(m-k+1) \geq \alpha/m_0$. By Bonferroni applied to the surviving true nulls, the probability of any false rejection is $\leq \alpha$. $\square$

In the OHE context, Holm is uniformly preferable to Bonferroni: it applies a less severe correction to tests ranked lower in significance, improving power for secondary endpoints that are genuinely affected.

---

## 35.3 Benjamini-Hochberg FDR Control

**Theorem 35.3 (Benjamini-Hochberg, 1995).** Order $p_{(1)} \leq \cdots \leq p_{(m)}$. Let:

$$k^* = \max\!\left\{k : p_{(k)} \leq \frac{k}{m}\alpha\right\}$$

Reject $H_{(1)}, \ldots, H_{(k^*)}$ (reject none if the set is empty). Under independence of test statistics, this procedure controls FDR $\leq (m_0/m)\alpha \leq \alpha$.

*Proof sketch.* The proof by Benjamini and Hochberg (1995) decomposes the expected false discovery proportion. Let $V$ be false rejections and $R$ total rejections. Writing $\mathbb{E}[V/R]$ as a sum over true nulls and applying the independence assumption to bound each term yields $\text{FDR} = (m_0/m)\alpha$. The full argument uses the representation of $V/R$ in terms of the empirical process of $p$-values under the null. $\square$

Under positive regression dependence (PRDS), BH controls FDR $\leq \alpha$ exactly; under arbitrary dependence, the Benjamini-Yekutieli (2001) procedure adds a $\log(m)$ correction.

The **q-value** is the BH-adjusted $p$-value: $q_{(k)} = \min_{j \geq k}(m/j) p_{(j)}$. A q-value of $0.10$ means that in the list of all discoveries up to this one, 10% are expected false positives.

In OHE, FDR control is appropriate for exploratory analysis of many biomarkers; FWER control is appropriate for the pre-specified primary outcomes where a false positive could influence Medicaid policy.

---

## 35.4 Romano-Wolf Stepdown with Resampling

Bonferroni and BH treat tests as independent or only loosely coupled. The OHE outcome variables—inpatient use, outpatient use, ED visits, financial strain—are correlated because they share the same units and the same instrument. Ignoring correlation leads to over-conservative FWER correction.

The Romano-Wolf procedure (Romano and Wolf, 2005) exploits the joint distribution of test statistics via resampling to achieve FWER control that is exact (not conservative) under correlation.

**Algorithm 35.1 (Romano-Wolf Stepdown).**

1. Compute test statistics $t_1, \ldots, t_m$ from the original data.
2. Draw $B$ bootstrap samples. For each bootstrap draw $b$, compute statistics $t_1^{(b)}, \ldots, t_m^{(b)}$ and center them: $\tilde{t}_k^{(b)} = t_k^{(b)} - t_k$.
3. At step $s$ of the stepdown, let $\mathcal{R}_s$ be the indices not yet rejected. Compute the step-$s$ critical value:

$$c_s(\alpha) = \inf\!\left\{c : \Pr\!\left(\max_{k \in \mathcal{R}_s} |\tilde{t}_k^{(b)}| \leq c\right) \geq 1-\alpha\right\}$$

4. Reject all $H_k$ with $|t_k| \geq c_s(\alpha)$. Remove them from $\mathcal{R}_s$ to form $\mathcal{R}_{s+1}$. Repeat until no new rejections.

**Theorem 35.4 (Romano-Wolf FWER Control).** Under the assumption that the joint distribution of $(t_1, \ldots, t_m)$ is consistently estimated by the bootstrap, Algorithm 35.1 controls FWER $\leq \alpha$ in large samples and achieves asymptotic exactness.

The centering in step 2 is critical: it shifts bootstrap statistics to satisfy the null hypothesis, a technique analogous to the Rademacher multiplier bootstrap used in wild bootstrap IV inference.

In practice, the wild bootstrap is preferred for IV settings (where errors are heteroskedastic) and for OHE (clustered by lottery list). The `wildboottest` package in Python implements this for 2SLS coefficient tests; for multi-outcome testing one draws common Rademacher weights across all outcome regressions simultaneously to preserve the joint distribution.

---

## 35.5 Researcher Degrees of Freedom

Beyond reported outcomes, researchers face choices at every stage of analysis:

- **Sample restrictions**: drop non-compliers? restrict to single-member households?
- **Covariate adjustment**: baseline lottery-list controls only? add demographics? add baseline health?
- **Outcome definition**: ever hospitalized vs. number of hospitalizations; binary financial hardship vs. continuous expenditure.
- **Estimator**: ITT OLS vs. IV/LATE; linear probability model vs. probit; with/without clustering.
- **Subgroup analyses**: by age, by chronic illness, by urban/rural.

Simmons, Nelson, and Simonsohn (2011) show that with four binary researcher choices, the false positive rate at nominal $\alpha = 0.05$ exceeds 0.60. This is not fraud; it is the accumulated probability mass of unreported analytic paths.

**Definition 35.3 (Researcher Degrees of Freedom).** A researcher degree of freedom is any analysis choice, not pre-specified and not disclosed, that is correlated with the final reported test statistic.

The econometrics literature distinguishes two regimes:

1. **Pre-registered specification**: the analysis plan commits to outcomes, sample restrictions, and estimator before seeing data. FWER/FDR corrections are applied to the registered family. All other analyses are exploratory.

2. **Unregistered analysis**: results are selected post-hoc. The effective number of tests $m$ is unobserved. Inference is unreliable regardless of reported $p$-values.

Pre-registration constrains researcher degrees of freedom at the cost of flexibility. The specification curve and multiverse analysis are diagnostic tools for the unregistered regime.

---

## 35.6 Specification Curves

**Definition 35.4 (Specification Curve).** A specification curve plots the point estimate (and confidence interval) for a target parameter $\tau$ across all "reasonable" specifications, ordered by estimate magnitude. Each specification is a node defined by a set of binary choices $(d_1, \ldots, d_K)$ where $d_k \in \{0,1\}$ indexes, e.g., covariate inclusion, sample restriction, outcome coding.

With $K$ binary choices the curve has up to $2^K$ points. Simonsohn, Simmons, and Nelson (2020) define "reasonable" as specifications that (a) are defensible a priori, (b) test the hypothesis of interest, and (c) are mutually exclusive in their choices.

The specification curve communicates:

1. **Sign consistency**: are all or nearly all estimates in the same direction?
2. **Magnitude consistency**: do estimates cluster or span a wide range?
3. **Estimate distribution**: where does the median estimate fall?

**Inference on the Specification Curve.** To test the global null $H_0: \tau = 0$, Simonsohn et al. propose a permutation test:

1. Define a test statistic $T$ of the specification curve (e.g., the median estimate, or the share of significant estimates).
2. Permute the treatment indicator $B$ times, recomputing the full specification curve each time. Under $H_0$, permuted and observed curves have the same distribution.
3. The $p$-value is the fraction of permuted $T^{(b)} \geq T_{\text{obs}}$.

This procedure requires computing $2^K \times B$ regressions—computationally intensive but tractable for $K \leq 10$ and $B = 500$.

---

## 35.7 Multiverse Analysis

Steegen et al. (2016) generalize the specification curve to a **multiverse analysis**: rather than fixing the outcome and varying specifications, one varies both specifications and outcomes. Each leaf of the multiverse tree is a (specification, outcome) pair.

The multiverse representation makes the following explicit:

- **Outcome researcher d.f.**: which of the 8 OHE domains to report
- **Sample researcher d.f.**: full sample vs. survey respondents vs. single-member households
- **Covariate researcher d.f.**: number of household members only (lottery strata) vs. full demographic controls

A multiverse analysis does not resolve which specification is correct. Its purpose is diagnostic: if effect estimates are robust across the full multiverse, the finding is credible; if they cluster around zero or span positive and negative values, the reported result is specification-sensitive.

**Formal connection to FWER.** If the analyst pre-commits to reporting the full multiverse and applies Holm correction across all (specification, outcome) cells, FWER is controlled. The cost is power; the benefit is that no selective reporting is possible by construction.

---

## 35.8 Application to the OHE

The OHE reports outcomes in eight domains across two waves (12-month and 24-month). We focus on 12-month outcomes: `doc_any_12m` (any doctor visit), `er_any_12m` (any ED visit), `hosp_any_12m` (any hospitalization), `rx_any_12m` (any prescription), `bp_dx_12m` (blood pressure diagnosis), `dia_dx_12m` (diabetes diagnosis), `dep_screen_12m` (depression screen positive), `catastrophic_exp_inp` (catastrophic financial expenditure).

For each outcome we estimate the IV/LATE of Medicaid coverage (instrumented by lottery selection, stratified by household size):

$$Y_i = \alpha + \tau D_i + \beta' X_i + \varepsilon_i, \quad D_i = \pi_0 + \pi_1 Z_i + \gamma' X_i + \nu_i$$

where $Z_i$ is `selected`, $D_i$ is `ohp_all_ever_admin`, and $X_i$ includes `numhh_list` dummies (mandatory randomization strata).

The specification curve varies: (a) inclusion of demographic covariates, (b) sample restriction to single-adult households, (c) binary vs. count outcomes where applicable. With three binary choices we have $2^3 = 8$ specifications per outcome and $8 \times 8 = 64$ total (specification, outcome) cells.

---

## Python: Multiple Testing Corrections and Specification Curve for the Oregon Health Insurance Experiment

```python
"""
Chapter 35: Multiple testing corrections and specification curve analysis
Oregon Health Insurance Experiment (12-month outcomes)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

warnings.filterwarnings("ignore")

# ── 1. Load OHE data ──────────────────────────────────────────────────────────

OHE_URL = "https://data.nber.org/oregon/oregonhie_descriptive_vars.dta"
CACHE = Path("/tmp/ohe_descriptive.dta")

if not CACHE.exists():
    print("Downloading OHE data...")
    urllib.request.urlretrieve(OHE_URL, CACHE)

df_raw = pd.read_stata(CACHE)

# Keep 12-month survey respondents with non-missing instrument
OUTCOMES_12M = [
    "doc_any_12m",   # any doctor visit
    "er_any_12m",    # any ED visit
    "hosp_any_12m",  # any hospitalization
    "rx_any_12m",    # any prescription fill
    "bp_dx_12m",     # blood pressure diagnosis
    "dia_dx_12m",    # diabetes diagnosis
    "dep_screen_12m",# PHQ depression screen positive
    "catastrophic_exp_inp",  # catastrophic expenditure
]

OUTCOME_LABELS = [
    "Doctor visit", "ED visit", "Hospitalization", "Prescription",
    "BP diagnosis", "Diabetes dx", "Depression screen", "Catastrophic expenditure"
]

CONTROLS_BASE = ["numhh_list"]          # mandatory strata
CONTROLS_DEMO = [                        # optional demographics
    "age_19_34_inp", "age_35_49_inp", "age_50_64_inp",
    "female_inp", "english_list",
]

df = df_raw.copy()
df["numhh_list"] = df["numhh_list"].astype("category")

# Drop if instrument missing
df = df.dropna(subset=["selected"])
df["selected"] = df["selected"].astype(float)
df["ohp_all_ever_admin"] = df["ohp_all_ever_admin"].astype(float)

# ── 2. IV estimation for each outcome ─────────────────────────────────────────

def estimate_iv(outcome: str, data: pd.DataFrame, extra_controls: list[str]) -> dict:
    """
    IV/2SLS of Medicaid coverage on outcome, instrumented by lottery.
    Strata (numhh_list) always included. Returns coef, se, p-value.
    """
    controls_all = CONTROLS_BASE + extra_controls
    # Build dummies for numhh_list
    dummies = pd.get_dummies(data["numhh_list"], prefix="hh", drop_first=True)
    extra_cols = [c for c in extra_controls if c in data.columns]
    
    cols_needed = [outcome, "ohp_all_ever_admin", "selected"] + extra_cols
    sub = data[cols_needed].dropna().copy()
    sub = pd.concat([sub, dummies.reindex(sub.index)], axis=1)
    
    hh_cols = [c for c in dummies.columns]
    exog_vars = hh_cols + extra_cols
    
    # Construct matrices for linearmodels
    y = sub[outcome]
    endog = sub["ohp_all_ever_admin"]
    instruments = sub["selected"]
    
    if exog_vars:
        exog = sm_add_const(sub[exog_vars])
    else:
        exog = sm_add_const(pd.DataFrame(index=sub.index))
    
    try:
        res = IV2SLS(
            dependent=y,
            exog=exog,
            endog=endog,
            instruments=instruments
        ).fit(cov_type="robust")
        coef = res.params["ohp_all_ever_admin"]
        se   = res.std_errors["ohp_all_ever_admin"]
        pval = res.pvalues["ohp_all_ever_admin"]
        return {"coef": coef, "se": se, "pval": pval, "n": len(sub), "converged": True}
    except Exception as e:
        return {"coef": np.nan, "se": np.nan, "pval": np.nan, "n": 0, "converged": False}


def sm_add_const(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add constant column."""
    out = df_in.copy()
    out.insert(0, "const", 1.0)
    return out


# Base specification: strata only, full survey sample
results_base = []
for outcome, label in zip(OUTCOMES_12M, OUTCOME_LABELS):
    if outcome not in df.columns:
        results_base.append({"outcome": label, "coef": np.nan, "se": np.nan, "pval": np.nan})
        continue
    r = estimate_iv(outcome, df, extra_controls=[])
    r["outcome"] = label
    results_base.append(r)

results_df = pd.DataFrame(results_base).dropna(subset=["pval"])
pvals = results_df["pval"].values
labels = results_df["outcome"].values
coefs  = results_df["coef"].values
ses    = results_df["se"].values

# ── 3. Multiple testing corrections ───────────────────────────────────────────

def apply_corrections(pvals: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Apply Bonferroni, Holm, and BH corrections. Returns summary DataFrame."""
    _, p_bonf, _, _   = multipletests(pvals, alpha=alpha, method="bonferroni")
    _, p_holm, _, _   = multipletests(pvals, alpha=alpha, method="holm")
    _, p_bh,   _, _   = multipletests(pvals, alpha=alpha, method="fdr_bh")
    
    reject_bonf = (p_bonf <= alpha)
    reject_holm = (p_holm <= alpha)
    reject_bh   = (p_bh   <= alpha)
    
    return pd.DataFrame({
        "p_raw":    pvals,
        "p_bonf":   p_bonf,
        "p_holm":   p_holm,
        "q_bh":     p_bh,
        "rej_bonf": reject_bonf,
        "rej_holm": reject_holm,
        "rej_bh":   reject_bh,
    })


corr_df = apply_corrections(pvals)
corr_df.index = labels

print("=" * 72)
print("Table 35.1  IV Estimates and Multiple Testing Corrections (OHE 12m)")
print("=" * 72)
header = f"{'Outcome':<28} {'Coef':>7} {'SE':>7} {'p_raw':>7} {'p_Holm':>7} {'q_BH':>7} {'Rej?':>5}"
print(header)
print("-" * 72)
for i, lbl in enumerate(labels):
    rej = "*" if corr_df.loc[lbl, "rej_holm"] else (" †" if corr_df.loc[lbl, "rej_bh"] else "  ")
    print(
        f"{lbl:<28} {coefs[i]:>7.3f} {ses[i]:>7.3f} "
        f"{pvals[i]:>7.3f} {corr_df.loc[lbl,'p_holm']:>7.3f} "
        f"{corr_df.loc[lbl,'q_bh']:>7.3f} {rej:>5}"
    )
print("-" * 72)
print("* Holm-significant at α=0.05   † BH-significant at FDR=0.05")
print()

# ── 4. Romano-Wolf stepdown via wild bootstrap ────────────────────────────────

def romano_wolf_stepdown(
    outcomes: list[str],
    data: pd.DataFrame,
    extra_controls: list[str],
    B: int = 499,
    alpha: float = 0.05,
    rng_seed: int = 42
) -> np.ndarray:
    """
    Romano-Wolf FWER-controlling stepdown for IV estimates.
    Uses Rademacher wild bootstrap with common weights across outcomes.
    Returns array of adjusted p-values.
    """
    rng = np.random.default_rng(rng_seed)
    m = len(outcomes)
    
    # Step 1: observed t-statistics
    t_obs = np.zeros(m)
    for j, out in enumerate(outcomes):
        if out not in data.columns:
            t_obs[j] = 0.0
            continue
        r = estimate_iv(out, data, extra_controls)
        t_obs[j] = abs(r["coef"] / r["se"]) if r["se"] > 0 else 0.0
    
    # Step 2: bootstrap distribution with COMMON Rademacher weights
    # For each bootstrap draw, re-weight observations by Rademacher ±1
    # and recompute IV using weighted 2SLS (approximation).
    # Full wild bootstrap for 2SLS re-runs first stage; we use the
    # score bootstrap approximation for tractability.
    
    # Build common dataset for all outcomes
    extra_cols = [c for c in extra_controls if c in data.columns]
    dummies = pd.get_dummies(data["numhh_list"], prefix="hh", drop_first=True)
    
    base_cols = [c for c in outcomes if c in data.columns] + ["ohp_all_ever_admin", "selected"] + extra_cols
    base_cols = list(dict.fromkeys(base_cols))
    sub = data[base_cols].dropna().copy()
    sub = pd.concat([sub, dummies.reindex(sub.index).fillna(0)], axis=1)
    n_sub = len(sub)
    hh_cols = list(dummies.columns)
    
    t_boot = np.zeros((B, m))
    
    for b in range(B):
        # Common Rademacher weights
        weights = rng.choice([-1.0, 1.0], size=n_sub)
        for j, out in enumerate(outcomes):
            if out not in sub.columns:
                t_boot[b, j] = 0.0
                continue
            y_w  = sub[out].values * weights
            d_w  = sub["ohp_all_ever_admin"].values
            z_w  = sub["selected"].values
            # Construct weighted exog
            Xcols = hh_cols + extra_cols
            if Xcols:
                X = np.column_stack([np.ones(n_sub)] + [sub[c].values for c in Xcols])
            else:
                X = np.ones((n_sub, 1))
            
            try:
                # 2SLS score bootstrap: regress weighted outcome on instruments/controls
                # First stage
                first_stage = np.linalg.lstsq(
                    np.column_stack([X, z_w]), d_w, rcond=None
                )[0]
                D_hat = np.column_stack([X, z_w]) @ first_stage
                
                # Second stage with weighted outcome
                X2 = np.column_stack([X, D_hat])
                coef_w = np.linalg.lstsq(X2, y_w, rcond=None)[0]
                resid_w = y_w - X2 @ coef_w
                # Robust SE
                bread = np.linalg.pinv(X2.T @ X2)
                meat  = X2.T @ np.diag(resid_w**2) @ X2
                vcov  = bread @ meat @ bread
                se_w  = np.sqrt(max(vcov[-1, -1], 1e-12))
                t_boot[b, j] = abs(coef_w[-1] / se_w)
            except Exception:
                t_boot[b, j] = 0.0
        
        # Center: subtract observed (null-centering)
        t_boot[b] -= t_obs
    
    # Absolute-value stepdown
    t_boot_abs = np.abs(t_boot)
    
    # Stepdown algorithm
    active = list(range(m))
    p_rw = np.ones(m)
    order = np.argsort(t_obs)[::-1]  # descending by |t_obs|
    
    for step_idx, k in enumerate(order):
        if k not in active:
            continue
        current_active = [i for i in order[step_idx:] if i in active]
        # Max statistic over active set
        max_boot = t_boot_abs[:, current_active].max(axis=1)
        p_rw[k] = np.mean(max_boot >= t_obs[k])
        active.remove(k)
    
    # Enforce monotonicity (stepdown requirement)
    for s in range(1, len(order)):
        k_prev, k_curr = order[s-1], order[s]
        p_rw[k_curr] = max(p_rw[k_curr], p_rw[k_prev])
    
    return p_rw


print("Computing Romano-Wolf adjusted p-values (B=499)...")
outcomes_present = [o for o in OUTCOMES_12M if o in df.columns]
labels_present   = [OUTCOME_LABELS[i] for i, o in enumerate(OUTCOMES_12M) if o in df.columns]

p_rw = romano_wolf_stepdown(outcomes_present, df, extra_controls=[], B=499)

print("\nTable 35.2  Romano-Wolf vs Holm Adjusted p-Values")
print(f"{'Outcome':<28} {'p_Holm':>8} {'p_RW':>8}")
print("-" * 46)
for lbl, p_h, p_r in zip(labels_present, corr_df.loc[labels_present, "p_holm"].values, p_rw):
    print(f"{lbl:<28} {p_h:>8.3f} {p_r:>8.3f}")
print()

# ── 5. Specification curve ────────────────────────────────────────────────────

# Three binary choices:
#   A: add demographic controls (0=no, 1=yes)
#   B: restrict to single-member households (0=no, 1=yes)
#   C: outcome is doc_any_12m (fix one outcome for the curve)

SPEC_OUTCOME = "doc_any_12m"  # fix outcome: doctor visit

spec_records = []

df_single = df[df["numhh_list"] == "1"].copy() if "numhh_list" in df.columns else df.copy()

for (use_demo, single_hh) in product([0, 1], [0, 1]):
    extra = CONTROLS_DEMO if use_demo else []
    data_spec = df_single if single_hh else df
    
    if SPEC_OUTCOME not in data_spec.columns:
        continue
    
    r = estimate_iv(SPEC_OUTCOME, data_spec, extra_controls=extra)
    if not r["converged"]:
        continue
    
    spec_records.append({
        "coef": r["coef"],
        "se":   r["se"],
        "pval": r["pval"],
        "demo_controls": bool(use_demo),
        "single_hh": bool(single_hh),
        "n": r["n"],
    })

spec_df = pd.DataFrame(spec_records).sort_values("coef").reset_index(drop=True)
spec_df["ci_lo"] = spec_df["coef"] - 1.96 * spec_df["se"]
spec_df["ci_hi"] = spec_df["coef"] + 1.96 * spec_df["se"]
spec_df["significant"] = spec_df["pval"] < 0.05

# ── 6. Plot specification curve ───────────────────────────────────────────────

fig = plt.figure(figsize=(9, 7))
gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)

ax_main  = fig.add_subplot(gs[0])
ax_demo  = fig.add_subplot(gs[1], sharex=ax_main)
ax_singl = fig.add_subplot(gs[2], sharex=ax_main)

x = np.arange(len(spec_df))
colors = ["#e41a1c" if sig else "#377eb8" for sig in spec_df["significant"]]

ax_main.bar(x, spec_df["coef"], color=colors, width=0.6, alpha=0.8, label="IV estimate")
ax_main.errorbar(x, spec_df["coef"],
                 yerr=np.array([spec_df["coef"] - spec_df["ci_lo"],
                                spec_df["ci_hi"] - spec_df["coef"]]),
                 fmt="none", color="black", linewidth=1.2, capsize=3)
ax_main.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_main.set_ylabel("IV Estimate (LATE)", fontsize=10)
ax_main.set_title(
    "Figure 35.1  Specification Curve: Effect of Medicaid on Doctor Visit (12m)",
    fontsize=10, fontweight="bold"
)
ax_main.tick_params(labelbottom=False)

# Annotation for legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e41a1c", label="p < 0.05"),
    Patch(facecolor="#377eb8", label="p ≥ 0.05"),
]
ax_main.legend(handles=legend_elements, fontsize=8, loc="upper left")

# Indicator panels
for ax_ind, col, label in [
    (ax_demo,  "demo_controls", "Demo\ncontrols"),
    (ax_singl, "single_hh",    "Single\nhousehold"),
]:
    ax_ind.set_yticks([])
    ax_ind.set_ylabel(label, fontsize=8, rotation=0, labelpad=45, va="center")
    for xi, val in enumerate(spec_df[col]):
        marker = "■" if val else "□"
        color  = "#333333" if val else "#aaaaaa"
        ax_ind.text(xi, 0.5, marker, ha="center", va="center",
                    color=color, fontsize=10)
    ax_ind.set_ylim(0, 1)
    ax_ind.set_xlim(-0.5, len(spec_df) - 0.5)
    if ax_ind == ax_singl:
        ax_ind.set_xlabel("Specification (sorted by estimate)", fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/ch35_spec_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Specification curve saved to /tmp/ch35_spec_curve.png")

# ── 7. Permutation test for specification curve ───────────────────────────────

def spec_curve_permutation_test(
    outcome: str,
    data: pd.DataFrame,
    extra_options: list[list],
    B: int = 199,
    rng_seed: int = 0,
) -> float:
    """
    Permutation test for specification curve null H0: tau = 0.
    Test statistic: median IV estimate across specifications.
    """
    rng = np.random.default_rng(rng_seed)
    
    def compute_median_estimate(df_perm: pd.DataFrame) -> float:
        ests = []
        for (use_demo, single_hh) in product([0, 1], [0, 1]):
            extra = CONTROLS_DEMO if use_demo else []
            d_use = df_perm[df_perm["numhh_list"] == "1"] if single_hh else df_perm
            r = estimate_iv(outcome, d_use, extra)
            if r["converged"]:
                ests.append(r["coef"])
        return np.median(ests) if ests else 0.0
    
    T_obs = compute_median_estimate(data)
    
    T_perm = []
    df_perm = data.copy()
    for _ in range(B):
        df_perm["selected"] = rng.permutation(df_perm["selected"].values)
        T_perm.append(compute_median_estimate(df_perm))
    
    p_perm = np.mean(np.abs(np.array(T_perm)) >= np.abs(T_obs))
    return p_perm, T_obs, T_perm


print("Running permutation test for specification curve (B=199)...")
p_perm, T_obs, T_perm_vals = spec_curve_permutation_test(
    SPEC_OUTCOME, df, extra_options=[], B=199
)

print(f"\nSpecification curve permutation test (outcome: doctor visit 12m)")
print(f"  Observed median estimate : {T_obs:.4f}")
print(f"  Permutation p-value      : {p_perm:.3f}")
print()

# ── 8. Summary table across all corrections ───────────────────────────────────

print("=" * 72)
print("Table 35.3  Comparison of Correction Methods (OHE, 8 Outcomes)")
print("=" * 72)
print(f"{'Method':<35} {'Rejections at α=0.05':>20}")
print("-" * 57)
print(f"{'Uncorrected':<35} {int((pvals < 0.05).sum()):>20}")
print(f"{'Bonferroni':<35} {int(corr_df['rej_bonf'].sum()):>20}")
print(f"{'Holm (stepdown)':<35} {int(corr_df['rej_holm'].sum()):>20}")
print(f"{'Benjamini-Hochberg (FDR)':<35} {int(corr_df['rej_bh'].sum()):>20}")
print(f"{'Romano-Wolf':<35} {int((p_rw < 0.05).sum()):>20}")
print("=" * 72)
```

---

## Summary

- **FWER** controls the probability of any false rejection; **FDR** controls the expected fraction of rejections that are false. FWER is appropriate when individual false positives are costly (policy-relevant outcomes); FDR is appropriate when aggregating discoveries (exploratory biomarker analysis).

- **Bonferroni** ($p_k \leq \alpha/m$) is simple and exact under positive dependence but conservative under positive correlation. **Holm** dominates Bonferroni uniformly: it rejects at least as many hypotheses by using a tighter threshold after removing the most significant outcomes.

- **Benjamini-Hochberg** provides FDR $\leq (m_0/m)\alpha$ under independence and PRDS. The **q-value** is the BH-adjusted $p$-value interpreted as the expected false discovery rate among all discoveries at that threshold.

- **Romano-Wolf** achieves FWER control that is asymptotically exact (not conservative) for correlated tests by using bootstrap resampling with common weights to estimate the joint null distribution. It uniformly dominates Bonferroni and Holm in power under realistic dependence structures.

- **Researcher degrees of freedom** — unreported choices in outcomes, samples, covariates, and estimators — inflate effective type I error even without explicit $p$-hacking. Pre-registration controls this by making the family of tests explicit before data analysis.

- **Specification curves** expose the sensitivity of a finding to analytic choices. The **permutation test** for the curve's median estimate provides a valid $p$-value for the global null that accounts for the full search over specifications.

- **Multiverse analysis** extends the specification curve to simultaneous variation in outcomes and specifications. Pre-committing to reporting the full multiverse with Holm correction provides FWER control at the cost of power, while providing maximal transparency about sensitivity.

---

## Further Reading

1. **Benjamini, Y. and Hochberg, Y. (1995).** "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society Series B*, 57(1), 289–300. The foundational paper proving FDR control under independence and PRDS, introducing the BH algorithm.

2. **Romano, J. P. and Wolf, M. (2005).** "Stepwise multiple testing as formalized data snooping." *Econometrica*, 73(4), 1237–1282. Develops the resampling-based stepdown procedure; proves asymptotic FWER control under stationarity; explicitly connects to data snooping in financial econometrics.

3. **Simonsohn, U., Simmons, J. P. and Nelson, L. D. (2020).** "Specification curve analysis." *Nature Human Behaviour*, 4, 1208–1214. Defines the specification curve, provides the permutation test, and discusses practical criteria for which specifications belong in the curve.

4. **Anderson, M. L. (2008).** "Multiple inference and gender differences in the effects of early intervention: a reevaluation of the Abecedarian, Perry Preschool, and Early Training Projects." *Journal of the American Statistical Association*, 103(484), 1481–1495. Applies FWER corrections (Holm) to reanalyze early childhood intervention RCTs; the canonical applied econometrics reference for multiple testing in program evaluation.

5. **Finkelstein, A. et al. (2012).** "The Oregon Health Insurance Experiment: evidence from the first year." *Quarterly Journal of Economics*, 127(3), 1057–1106. Primary OHE paper; reports 8-domain outcome structure and discusses inference strategy; basis for the numerical examples in this chapter.

6. **Clarke, D., Romano, J. P. and Wolf, M. (2020).** "The Romano-Wolf multiple-hypothesis correction in Stata." *Stata Journal*, 20(4), 812–843. Practical implementation guide for Romano-Wolf with Stata code and discussion of wild bootstrap for clustered/heteroskedastic settings; Python analogs are direct translations of the algorithms described here.
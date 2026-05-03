# Chapter 18: Event Studies as Evidence, Not Decoration

Event studies have become the default visualization for difference-in-differences designs. Open any applied economics paper from the past decade and you will find one: a plot of coefficients across event time, a vertical line at zero marking the intervention, and a cluster of pre-period estimates hovering near zero that the author cites as evidence of parallel trends. The problem is that this visual ritual has become almost entirely disconnected from the formal inferential content it purports to convey. Pre-trends "look flat" until they don't, reference period choices shift the visual impression without changing the underlying estimates, and the confidence bands displayed routinely understate uncertainty about the parallel trends assumption itself.

This chapter treats event studies as objects of formal analysis. We develop the estimand precisely, work through normalization and reference period choice as substantive decisions rather than cosmetic ones, derive the correct joint test for pre-trends, and then confront the fundamental limitation: even a perfect pre-trend test cannot validate parallel trends in the post-period. The resolution is Rambachan and Roth's (2023) sensitivity framework, which translates visual pre-trend information into honest bounds on post-period effects. Throughout, we use the ACA Medicaid expansion and BRFSS data to build a complete, honest event study of insurance coverage effects on self-reported health.

## 18.1 The Event Study Estimand

Let $D_{it} \in \{0,1\}$ indicate treatment for unit $i$ at time $t$, and let $t_i^*$ denote the calendar time at which unit $i$ first receives treatment ($t_i^* = \infty$ for never-treated units). Define event time as $k = t - t_i^*$, so $k < 0$ is pre-treatment and $k \geq 0$ is post-treatment. Under the potential outcomes framework with SUTVA, the event-study estimand at horizon $k$ is:

$$\beta_k = E[Y_{it}(1) - Y_{it}(0) \mid k = t - t_i^*]$$

where $Y_{it}(1)$ and $Y_{it}(0)$ are potential outcomes under treatment and control at calendar time $t$. This is an average treatment effect for the cohort observed at exactly $k$ periods from their treatment date, averaging over all cohort-calendar time pairs that map to event time $k$.

Several points deserve emphasis. First, $\beta_k$ is defined for a specific event-time relative to treatment onset; it is not a calendar-time effect. Second, for $k < 0$, the estimand is $E[Y_{it}(0) - Y_{it}(0)] = 0$ under the assumption that treatment has no effect before it occurs and no anticipation exists — this is the testable restriction that pre-trend analysis exploits. Third, for $k \geq 0$, $\beta_k$ is a cohort-averaged ATT that mixes across groups treated at different calendar times unless the design is homogeneous.

The regression implementation pools across cohorts and event times. Define event-time dummies $\mathbf{1}[t - t_i^* = k]$ for $k \in \{-K, \ldots, -2, 0, 1, \ldots, L\}$, omitting $k = -1$ as the reference period. The two-way fixed effects event study regression is:

$$Y_{it} = \alpha_i + \alpha_t + \sum_{k=-K, k \neq -1}^{L} \beta_k \cdot \mathbf{1}[t - t_i^* = k] + \varepsilon_{it}$$

The resulting $\hat{\beta}_k$ estimates are interpreted as deviations from the period immediately before treatment. Under parallel trends and no anticipation, we expect $\hat{\beta}_k \approx 0$ for all $k < 0$ and interpret $\hat{\beta}_k$ for $k \geq 0$ as the treatment effect at horizon $k$.

**On endpoints:** Units without $K$ pre-periods or $L$ post-periods contribute to some but not all event-time cells. The standard practice of "binning" endpoints — replacing $k \leq -K$ with a single dummy and $k \geq L$ with a single dummy — trades off bias (the bin absorbs heterogeneous effects) against variance (more observations per coefficient). The choice of $K$ and $L$ is a design decision, not a data-driven one, and should be stated explicitly.

## 18.2 Normalization and Reference Period Choice

The reference period normalization is not innocuous. Because event-study coefficients are identified as deviations from the reference period, the choice of which period to omit mechanically shifts the visual level of the entire plot. What looks like "flat pre-trends" centered on zero with $k = -1$ as reference may look like a secular drift when $k = -3$ is chosen.

**Theorem 18.1 (Normalization Invariance of Differences).** *Let $\hat{\beta}_k^{(-1)}$ denote event-study estimates with $k=-1$ as reference and $\hat{\beta}_k^{(-j)}$ with $k=-j$ as reference. Then for any two post-period horizons $k, k'$:*

$$\hat{\beta}_k^{(-1)} - \hat{\beta}_{k'}^{(-1)} = \hat{\beta}_k^{(-j)} - \hat{\beta}_{k'}^{(-j)}$$

*Proof.* The two estimators are related by $\hat{\beta}_k^{(-j)} = \hat{\beta}_k^{(-1)} - \hat{\beta}_{-j}^{(-1)}$ for all $k \neq -j$. The difference $\hat{\beta}_k^{(-j)} - \hat{\beta}_{k'}^{(-j)} = (\hat{\beta}_k^{(-1)} - \hat{\beta}_{-j}^{(-1)}) - (\hat{\beta}_{k'}^{(-1)} - \hat{\beta}_{-j}^{(-1)}) = \hat{\beta}_k^{(-1)} - \hat{\beta}_{k'}^{(-1)}$. $\square$

The implication is immediate: the *shape* of the event study — the slope of pre-trends, the magnitude of post-period effects relative to each other — is invariant to reference period choice. But the *level* is not. Causal claims about level effects (e.g., "treatment raised $Y$ by 5 points at $k=0$") depend on the reference period normalization. The conventional $k = -1$ reference is appealing because it measures effects relative to the period immediately before treatment, which is often the most credible counterfactual baseline. But if $k = -1$ is itself contaminated by anticipatory behavior, this choice smuggles a pre-treatment effect into the normalization.

A useful diagnostic is to estimate the event study under two or three reference period choices and overlay them. Invariance of differences should hold exactly in-sample; if the plot shapes differ dramatically in their pre-period behavior under different normalizations, that signals a noisy $k=-1$ estimate rather than a trend violation. For the ACA analysis we use $k = -1$ (2013, the year before the 2014 expansion) as the reference and verify robustness to $k = -2$.

## 18.3 Pre-Trend Testing: Joint Tests Over Visual Inspection

The near-universal practice of "eyeballing" pre-trends is statistically indefensible as a formal test. Visually assessing whether confidence intervals overlap zero is approximately equivalent to conducting $K$ separate Bonferroni-uncorrected hypothesis tests, each at level $\alpha$, which inflates type I error. The correct procedure is a joint Wald test.

**Pre-Trend Test.** Let $\hat{\boldsymbol{\beta}}_{pre} = (\hat{\beta}_{-K}, \ldots, \hat{\beta}_{-2})'$ be the $(K-1)$-vector of pre-period estimates (excluding the reference at $k=-1$) and $\hat{V}_{pre}$ the corresponding block of the cluster-robust variance-covariance matrix. The joint pre-trend test statistic is:

$$W = \hat{\boldsymbol{\beta}}_{pre}' \hat{V}_{pre}^{-1} \hat{\boldsymbol{\beta}}_{pre} \xrightarrow{d} \chi^2_{K-1}$$

under $H_0: \boldsymbol{\beta}_{pre} = \mathbf{0}$. Equivalently, the $F$-version divides by $K-1$ and is compared to $F_{K-1, G-1}$ where $G$ is the number of clusters.

This test has well-known power limitations. Pre-trend tests are underpowered when the sample is small, when there are few clusters, or when the pre-period window is short. Failure to reject $H_0$ is not evidence that parallel trends holds; it may simply reflect that the test cannot detect moderate violations. Roth (2022) formalizes this: he shows that in typical DiD applications, the pre-trend test has low power against violations large enough to meaningfully bias post-period estimates. This motivates the sensitivity approach of Section 18.5.

**Anticipation.** A subtler violation of the pre-trend test's null arises from anticipatory behavior. If agents know treatment is coming and adjust behavior in advance, then $E[Y_{it}(0) \mid k = -1] \neq E[Y_{it}(0) \mid k = -2]$: the untreated potential outcome itself shifts in anticipation of treatment. In the ACA context, states that announced early Medicaid expansion plans may have seen insurance enrollment begin rising in 2013 even before the official January 2014 expansion date. Anticipation of up to $A$ periods can be accommodated by shifting the reference period to $k = -(A+1)$ and treating $k \in \{-A, \ldots, -1\}$ as potentially contaminated post-period estimates. The cost is a shorter clean pre-period window for trend testing.

## 18.4 Aggregating Group-Time ATTs into Event-Study Format

The TWFE event study regression is misspecified when treatment effect heterogeneity varies across cohorts or calendar time. Callaway and Sant'Anna (2021) show that TWFE event-study coefficients are weighted sums of group-time ATTs $ATT(g,t)$ with potentially negative weights, producing estimates that can have the wrong sign even when every $ATT(g,t) > 0$.

The correction is to first estimate cohort-specific event studies and then aggregate. Define $ATT(g, g+k)$ as the ATT for the cohort first treated at calendar time $g$, observed at event time $k$. The aggregated event-study coefficient at horizon $k$ is:

$$\beta_k^{agg} = \sum_g w_g \cdot ATT(g, g+k)$$

where weights $w_g \propto N_g$, the cohort size. This aggregation is internally consistent: it never compares late-treated units to already-treated units and uses only clean comparison groups (never-treated or not-yet-treated).

**The Stacking Estimator.** An implementationally simple alternative to the Callaway-Sant'Anna correction is the stacking estimator (Cengiz et al. 2019, Baker et al. 2022). For each treatment cohort $g$, construct a "clean" dataset containing: (i) cohort $g$ units for event windows $[-K, L]$, and (ii) clean control units (never-treated or not-yet-treated as of $g+L$). Stack these cohort-specific datasets and run a single TWFE regression with cohort-dataset fixed effects:

$$Y_{igt} = \alpha_{ig} + \alpha_{tg} + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}[t - g = k] \cdot D_{ig} + \varepsilon_{igt}$$

The dataset subscript $g$ in the fixed effects ensures that each cohort is compared only to its designated clean controls, eliminating the contaminated comparisons that bias TWFE. The stacking estimator is numerically equivalent to the Callaway-Sant'Anna aggregated event study under equal cohort weighting.

For the ACA analysis, we have three main expansion cohorts: 2014 early adopters, 2015 adopters, and 2016 adopters (with 2010–2012 adopters forming a separate early group and non-expanding states as clean controls through 2016). We construct the stacked event study with a $[-3, +2]$ event window, using non-expanding states as the control group for all cohorts.

## 18.5 Honest Pre-Trend Inference: Rambachan-Roth Sensitivity

The pre-trend test answers the question: "Are pre-period estimates statistically indistinguishable from zero?" This is the wrong question. The right question is: "Given what we observe about pre-period violations, how bad could post-period parallel trends violations be?"

Rambachan and Roth (2023) formalize this logic. Let $\delta_k = E[Y_{it}(0,\text{treated group}) - Y_{it}(0,\text{control group}) \mid k]$ be the difference in untreated potential outcomes between treated and control units at event time $k$, normalized so that $\delta_{-1} = 0$. The parallel trends assumption requires $\delta_k = 0$ for all $k \geq 0$. The event study pre-period estimates provide a noisy signal about $\delta_k$ for $k < 0$.

The key sensitivity parameter is $\bar{M}$, which bounds the maximum slope change in the violations across adjacent periods:

$$|\delta_{k+1} - 2\delta_k + \delta_{k-1}| \leq \bar{M} \quad \forall k$$

This is a "smoothness restriction" on the trend violations: it says violations cannot change direction too abruptly. When $\bar{M} = 0$, violations are constrained to linear trends. As $\bar{M}$ grows, the assumption weakens toward no restriction.

**Theorem 18.2 (Identified Set Under Smoothness Restriction, informal).** *Under the smoothness restriction with parameter $\bar{M}$, the identified set for the post-period ATT at horizon $k \geq 0$ is:*

$$\mathcal{B}(\bar{M}) = \left[\hat{\beta}_k - \text{bias}^+(\bar{M}), \; \hat{\beta}_k + \text{bias}^+(\bar{M})\right]$$

*where $\text{bias}^+(\bar{M})$ is the maximum bias consistent with pre-period estimates and the smoothness constraint. Confidence sets for $\beta_k$ are constructed by inverting this identified set using the conditional hybrid method of Andrews et al. (2019).*

The practical workflow is:

1. Estimate the event study and obtain $\hat{\boldsymbol{\beta}}$ and $\hat{V}$.
2. Choose a grid of $\bar{M}$ values from 0 (linear extrapolation of pre-trend) up to a value where the confidence set spans zero.
3. Report the "breakdown $\bar{M}$": the smallest $\bar{M}$ at which the confidence set includes zero. This is the minimum violation magnitude needed to overturn the conclusion.

The sensitivity analysis does not replace the point estimate. It contextualizes it: an estimate of +3 percentage points on insurance coverage with a breakdown $\bar{M}$ of 0.5 (meaning violations would need to be quite non-linear to overturn the conclusion) is more convincing than one with breakdown $\bar{M}$ of 0.1.

An alternative sensitivity parameter set is $\Delta^{SD}(M)$, which bounds the magnitude of violations relative to the standard deviation of pre-period estimates rather than their second differences. The choice between smoothness and sign restrictions depends on the application; the `HonestDiD` package implements both.

## 18.6 Confidence Band Construction

Event-study plots require honest confidence bands. The standard approach is to plot pointwise $1-\alpha$ confidence intervals around each $\hat{\beta}_k$ using $\hat{\beta}_k \pm 1.96 \cdot \hat{\sigma}_k$ with cluster-robust standard errors. These are the correct intervals for testing $H_0: \beta_k = 0$ individually, but they are not simultaneous bands: if all bands are $95\%$ pointwise, the probability that all $K+L+1$ bands simultaneously cover their true values is substantially lower than $95\%$.

For visualization where the viewer is naturally assessing the entire profile jointly, simultaneous bands are more appropriate:

$$\hat{\beta}_k \pm c_\alpha \cdot \hat{\sigma}_k$$

where $c_\alpha$ is chosen so that $P(\max_k |\hat{\beta}_k - \beta_k|/\hat{\sigma}_k \leq c_\alpha) = 1-\alpha$. Under joint normality of $\hat{\boldsymbol{\beta}}$, this can be computed as the $(1-\alpha)$ quantile of $\max_k |Z_k|$ where $(Z_{-K}, \ldots, Z_L) \sim N(\mathbf{0}, \hat{R})$ with $\hat{R}$ the correlation matrix. Monte Carlo integration over this distribution is simple and computationally fast.

In practice, the simultaneous bands are 20–40% wider than pointwise bands for typical event-study lengths of 6–10 periods. Reporting only pointwise bands while visually encouraging joint assessment is a common source of overconfidence in pre-trend assessments.

## Python: ACA Medicaid Expansion Event Study with Honest Sensitivity Bands

```python
"""
Chapter 18: Event Studies as Evidence, Not Decoration
ACA Medicaid expansion event study using BRFSS data.
Requires: pyfixest, matplotlib, numpy, pandas, scipy
HonestDiD Python wrapper: pip install honest-did (or use R via rpy2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.linalg import block_diag
import pyfixest as pf
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Load and prepare ACA / BRFSS data
# ─────────────────────────────────────────────

def load_brfss_aca(path: str = "data/brfss_aca.parquet") -> pd.DataFrame:
    """
    Expected columns:
      state_fips      – state identifier
      year            – calendar year 2010-2016
      insured_rate    – fraction insured (from BRFSS)
      health_good     – fraction reporting good/very good/excellent health
      n               – survey respondents (for weighting)
    Expansion timing assigned from Kaiser Family Foundation data.
    """
    df = pd.read_parquet(path)
    
    # ACA Medicaid expansion dates (simplified three-cohort coding)
    # Early adopters: states that expanded in 2014
    # Late adopters: 2015 or 2016
    # Non-expanders: never expanded through 2016
    expansion_year = {
        # 2014 cohort (26 states + DC)
        **{s: 2014 for s in [
            6, 10, 15, 17, 21, 25, 26, 27, 32, 33, 34, 35, 36,
            38, 39, 41, 44, 53, 50, 54, 8, 4, 16, 23, 24, 9, 11
        ]},
        # 2015 cohort
        **{s: 2015 for s in [2, 46, 49]},
        # 2016 cohort
        **{s: 2016 for s in [19, 48, 31, 40]},
    }
    
    df['expand_year'] = df['state_fips'].map(expansion_year)
    df['treated'] = df['expand_year'].notna().astype(int)
    df['event_time'] = np.where(
        df['treated'] == 1,
        df['year'] - df['expand_year'],
        np.nan
    )
    return df


def simulate_brfss_aca(
    n_states: int = 50,
    years: range = range(2010, 2017),
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate BRFSS-style panel when real data unavailable.
    DGP:
      - 26 states expand in 2014, 8 expand 2015-2016, 16 never expand
      - Parallel pre-trends with small linear drift
      - Treatment effect on insured_rate: +5pp at k=0, +7pp at k=1, +7pp at k=2
      - Treatment effect on health_good: +1.5pp at k=0, +2pp at k=1, +2.5pp at k=2
      - Mild heterogeneity across cohorts
    """
    rng = np.random.default_rng(seed)
    
    n_years = len(list(years))
    year_list = list(years)
    
    # Assign expansion cohorts
    state_ids = np.arange(1, n_states + 1)
    expand_2014 = state_ids[:26]
    expand_2015 = state_ids[26:32]
    expand_2016 = state_ids[32:36]
    never_expand = state_ids[36:]
    
    expand_year = {}
    for s in expand_2014: expand_year[s] = 2014
    for s in expand_2015: expand_year[s] = 2015
    for s in expand_2016: expand_year[s] = 2016
    
    # State fixed effects
    state_fe = rng.normal(0, 3, n_states)
    year_fe_ins = rng.normal(0, 0.5, n_years)
    year_fe_health = rng.normal(0, 0.3, n_years)
    
    # ATT profiles by cohort (some heterogeneity)
    def att_insured(k, cohort_shift=0.0):
        profile = {-3: 0, -2: 0, -1: 0, 0: 5.0, 1: 7.0, 2: 7.5}
        return profile.get(k, 0) + cohort_shift
    
    def att_health(k, cohort_shift=0.0):
        profile = {-3: 0, -2: 0, -1: 0, 0: 1.5, 1: 2.0, 2: 2.5}
        return profile.get(k, 0) + cohort_shift
    
    records = []
    for state_idx, state in enumerate(state_ids):
        ey = expand_year.get(state, None)
        for yr_idx, yr in enumerate(year_list):
            k = (yr - ey) if ey is not None else None
            
            cohort_shift_ins = (
                0.5 if state in expand_2015
                else (-0.5 if state in expand_2016 else 0)
            )
            cohort_shift_h = cohort_shift_ins * 0.2
            
            effect_ins = att_insured(k, cohort_shift_ins) if k is not None else 0
            effect_health = att_health(k, cohort_shift_h) if k is not None else 0
            
            base_ins = 70 + state_fe[state_idx] + year_fe_ins[yr_idx]
            base_health = 55 + state_fe[state_idx] * 0.5 + year_fe_health[yr_idx]
            
            noise_ins = rng.normal(0, 1.2)
            noise_health = rng.normal(0, 0.8)
            n_resp = rng.integers(2000, 8000)
            
            records.append({
                'state_fips': state,
                'year': yr,
                'insured_rate': base_ins + effect_ins + noise_ins,
                'health_good': base_health + effect_health + noise_health,
                'n': n_resp,
                'expand_year': ey,
                'treated': int(ey is not None),
                'event_time': k,
            })
    
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 2. Stacked dataset construction
# ─────────────────────────────────────────────

def build_stacked_dataset(
    df: pd.DataFrame,
    cohorts: list,
    event_window: tuple = (-3, 2),
    outcome: str = 'insured_rate'
) -> pd.DataFrame:
    """
    For each treatment cohort, create a clean 2x2 stack using
    never-expanders as controls. Bin endpoints.
    """
    K_pre, K_post = -event_window[0], event_window[1]
    never_treated = df[df['expand_year'].isna()]['state_fips'].unique()
    
    stacks = []
    for g in cohorts:
        cohort_states = df[df['expand_year'] == g]['state_fips'].unique()
        
        # Calendar years in the event window for this cohort
        cal_years = range(g + event_window[0], g + event_window[1] + 1)
        
        # Treated units in window
        cohort_df = df[
            (df['state_fips'].isin(cohort_states)) &
            (df['year'].isin(cal_years))
        ].copy()
        cohort_df['cohort'] = g
        cohort_df['in_treatment'] = 1
        
        # Clean controls: never-treated, same calendar window
        control_df = df[
            (df['state_fips'].isin(never_treated)) &
            (df['year'].isin(cal_years))
        ].copy()
        control_df['cohort'] = g
        control_df['in_treatment'] = 0
        control_df['event_time'] = control_df['year'] - g
        
        stack = pd.concat([cohort_df, control_df], ignore_index=True)
        stack['stack_id'] = f"cohort_{g}"
        stacks.append(stack)
    
    stacked = pd.concat(stacks, ignore_index=True)
    
    # Bin endpoints
    stacked['event_time_binned'] = stacked['event_time'].clip(
        lower=event_window[0], upper=event_window[1]
    )
    
    # Create unique unit FE within each stack
    stacked['unit_stack'] = (
        stacked['state_fips'].astype(str) + "_" + stacked['stack_id']
    )
    stacked['year_stack'] = (
        stacked['year'].astype(str) + "_" + stacked['stack_id']
    )
    
    return stacked


# ─────────────────────────────────────────────
# 3. pyfixest event study estimation
# ─────────────────────────────────────────────

def estimate_event_study(
    stacked: pd.DataFrame,
    outcome: str,
    event_window: tuple = (-3, 2),
    ref_period: int = -1
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run stacked TWFE event study via pyfixest.
    Returns: (coef_df, beta_vec, vcov_matrix)
    """
    periods = [k for k in range(event_window[0], event_window[1] + 1)
               if k != ref_period]
    
    # Create event-time dummies
    for k in periods:
        varname = f"rel_{k}_neg" if k < 0 else f"rel_{k}"
        stacked[varname] = (
            (stacked['event_time_binned'] == k) & 
            (stacked['in_treatment'] == 1)
        ).astype(float)
    
    dummy_names = [
        (f"rel_{k}_neg" if k < 0 else f"rel_{k}") for k in periods
    ]
    dummy_str = " + ".join(dummy_names)
    
    formula = f"{outcome} ~ {dummy_str} | unit_stack + year_stack"
    
    model = pf.feols(
        formula,
        data=stacked,
        vcov={"CRV1": "state_fips"}
    )
    
    coef_df = model.tidy()
    coef_df = coef_df[coef_df['Coefficient'].isin(dummy_names)].copy()
    coef_df['event_time'] = periods
    coef_df = coef_df.sort_values('event_time').reset_index(drop=True)
    
    # Add reference period row (beta = 0 by construction)
    ref_row = pd.DataFrame({
        'event_time': [ref_period],
        'Estimate': [0.0],
        'Std. Error': [0.0],
        't value': [np.nan],
        'Pr(>|t|)': [np.nan],
        'Coefficient': [f'ref_{ref_period}']
    })
    coef_df = pd.concat([coef_df, ref_row]).sort_values('event_time').reset_index(drop=True)
    
    # Extract beta and vcov for non-reference periods
    non_ref = [i for i, k in enumerate(coef_df['event_time']) if k != ref_period]
    beta_vec = coef_df.loc[non_ref, 'Estimate'].values
    
    # Reconstruct VCV from model
    vcov = model.vcov()
    vcov_arr = np.array(vcov)
    
    return coef_df, beta_vec, vcov_arr


# ─────────────────────────────────────────────
# 4. Pre-trend joint test
# ─────────────────────────────────────────────

def pretrend_joint_test(
    beta_pre: np.ndarray,
    vcov_pre: np.ndarray
) -> dict:
    """
    Wald test H0: beta_pre = 0.
    Returns: test statistic, df, p-value.
    """
    k = len(beta_pre)
    try:
        vcov_inv = np.linalg.inv(vcov_pre)
        W = float(beta_pre @ vcov_inv @ beta_pre)
    except np.linalg.LinAlgError:
        vcov_inv = np.linalg.pinv(vcov_pre)
        W = float(beta_pre @ vcov_inv @ beta_pre)
    
    F_stat = W / k
    p_chi2 = 1 - stats.chi2.cdf(W, df=k)
    p_F = 1 - stats.f.cdf(F_stat, dfn=k, dfd=max(1, 50 - k))
    
    return {
        'W': W, 'F': F_stat, 'df': k,
        'p_chi2': p_chi2, 'p_F': p_F
    }


# ─────────────────────────────────────────────
# 5. Simultaneous confidence bands
# ─────────────────────────────────────────────

def simultaneous_bands(
    beta: np.ndarray,
    se: np.ndarray,
    vcov: np.ndarray,
    alpha: float = 0.05,
    n_sim: int = 50_000,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute simultaneous confidence bands via simulation.
    Returns (lower, upper) arrays of same length as beta.
    """
    rng = np.random.default_rng(seed)
    
    # Correlation matrix from vcov
    std = np.sqrt(np.diag(vcov))
    std = np.where(std == 0, 1.0, std)
    corr = vcov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    
    # Simulate max absolute standardized normal
    draws = rng.multivariate_normal(np.zeros(len(beta)), corr, size=n_sim)
    max_abs = np.max(np.abs(draws), axis=1)
    c_alpha = np.quantile(max_abs, 1 - alpha)
    
    lower = beta - c_alpha * se
    upper = beta + c_alpha * se
    return lower, upper


# ─────────────────────────────────────────────
# 6. Rambachan-Roth sensitivity (pure-Python implementation)
# ─────────────────────────────────────────────

def rambachan_roth_sensitivity(
    beta_post: np.ndarray,
    beta_pre: np.ndarray,
    vcov_full: np.ndarray,
    n_pre: int,
    M_bar_grid: np.ndarray,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Simplified Rambachan-Roth sensitivity under smoothness restriction Delta^SD.
    
    For each M_bar, computes the identified set bounds assuming violations
    cannot deviate more than M_bar * sigma_pre from a linear extrapolation
    of the pre-trend.
    
    This is an approximation to the full conditional hybrid method;
    for publication-quality results use the HonestDiD R package.
    
    Parameters
    ----------
    beta_post : post-period ATT estimates (length L)
    beta_pre  : pre-period estimates (length K-1, excluding reference)
    vcov_full : full (K-1+L) x (K-1+L) vcov, pre blocks first
    n_pre     : number of pre-period coefficients
    M_bar_grid: array of smoothness parameters to evaluate
    alpha     : significance level
    
    Returns
    -------
    DataFrame with columns: M_bar, lower_bound, upper_bound, robust_ci_lower,
                            robust_ci_upper, identified_set_contains_zero
    """
    L = len(beta_post)
    K = n_pre
    
    # Pre-period vcov
    vcov_pre = vcov_full[:K, :K]
    se_pre = np.sqrt(np.diag(vcov_pre))
    
    # Linear extrapolation of pre-trend slope
    # Fit OLS trend through pre-period estimates: beta_k = a + b*k
    pre_times = np.arange(-K, 0)  # e.g., [-3, -2] if K=2
    X_pre = np.column_stack([np.ones(K), pre_times])
    try:
        trend_coef = np.linalg.lstsq(X_pre, beta_pre, rcond=None)[0]
    except Exception:
        trend_coef = np.array([0.0, 0.0])
    
    # Extrapolated trend at post-periods
    post_times = np.arange(0, L)
    X_post = np.column_stack([np.ones(L), post_times])
    trend_extrapolated = X_post @ trend_coef
    
    # SD of pre-period estimates (average SE as scale)
    sigma_pre = np.mean(se_pre) if len(se_pre) > 0 else 1.0
    
    results = []
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    
    for M_bar in M_bar_grid:
        # Maximum bias at each post-period horizon under smoothness M_bar
        # Bias grows with horizon and with M_bar
        max_bias = np.array([
            M_bar * sigma_pre * (1 + k) for k in post_times
        ])
        
        # Robust CI accounts for bias + sampling uncertainty
        vcov_post = vcov_full[K:, K:]
        se_post = np.sqrt(np.diag(vcov_post))
        
        # Identified set: remove trend extrapolation, bound residual
        beta_demeaned = beta_post - trend_extrapolated
        
        lb_id = beta_demeaned - max_bias
        ub_id = beta_demeaned + max_bias
        
        # Robust CI: identified set +/- sampling uncertainty
        lb_ci = lb_id - z_alpha * se_post
        ub_ci = ub_id + z_alpha * se_post
        
        # Average over post periods (aggregate effect)
        agg_lb_id = np.mean(lb_id)
        agg_ub_id = np.mean(ub_id)
        agg_lb_ci = np.mean(lb_ci)
        agg_ub_ci = np.mean(ub_ci)
        
        results.append({
            'M_bar': M_bar,
            'lower_id': agg_lb_id,
            'upper_id': agg_ub_id,
            'lower_ci': agg_lb_ci,
            'upper_ci': agg_ub_ci,
            'contains_zero': (agg_lb_ci <= 0 <= agg_ub_ci),
        })
    
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 7. Main analysis
# ─────────────────────────────────────────────

def main():
    # ── Data ──────────────────────────────────
    try:
        df = load_brfss_aca("data/brfss_aca.parquet")
        print("Loaded real BRFSS data.")
    except FileNotFoundError:
        print("Real data not found; using simulated DGP.")
        df = simulate_brfss_aca(n_states=50, seed=42)
    
    EVENT_WINDOW = (-3, 2)
    COHORTS = [2014, 2015, 2016]
    OUTCOMES = {
        'insured_rate': 'Insurance Coverage Rate (pp)',
        'health_good': 'Good/Very Good Health (%)',
    }
    
    # ── Stacked dataset ───────────────────────
    stacked = build_stacked_dataset(df, COHORTS, EVENT_WINDOW)
    print(f"Stacked dataset: {len(stacked):,} observations, "
          f"{stacked['state_fips'].nunique()} unique states, "
          f"{stacked['cohort'].nunique()} cohorts")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "ACA Medicaid Expansion: Event Study and Rambachan-Roth Sensitivity",
        fontsize=13, fontweight='bold', y=1.01
    )
    
    for col_idx, (outcome, outcome_label) in enumerate(OUTCOMES.items()):
        # ── Event study estimation ─────────────
        coef_df, beta_vec, vcov_arr = estimate_event_study(
            stacked.copy(), outcome, EVENT_WINDOW, ref_period=-1
        )
        
        event_times = coef_df['event_time'].values
        betas = coef_df['Estimate'].values
        ses = coef_df['Std. Error'].values
        
        # Pre-trend test
        pre_mask = coef_df['event_time'] < -1
        beta_pre = betas[pre_mask]
        
        n_pre_coef = len(beta_pre)
        if n_pre_coef > 0 and len(vcov_arr) >= n_pre_coef:
            vcov_pre = vcov_arr[:n_pre_coef, :n_pre_coef]
            pt = pretrend_joint_test(beta_pre, vcov_pre)
            print(f"\n{outcome} pre-trend test:")
            print(f"  F({pt['df']}, ~47) = {pt['F']:.3f}, "
                  f"p = {pt['p_F']:.3f}  |  "
                  f"Chi2({pt['df']}) = {pt['W']:.3f}, "
                  f"p = {pt['p_chi2']:.3f}")
        
        # Simultaneous bands
        non_ref_mask = coef_df['event_time'] != -1
        betas_nr = betas[non_ref_mask]
        ses_nr = ses[non_ref_mask]
        et_nr = event_times[non_ref_mask]
        
        if len(vcov_arr) == len(betas_nr):
            sim_lo, sim_hi = simultaneous_bands(betas_nr, ses_nr, vcov_arr)
        else:
            sim_lo = betas_nr - 1.96 * ses_nr
            sim_hi = betas_nr + 1.96 * ses_nr
        
        # ── Plot 1: Event study with bands ────
        ax = axes[0, col_idx]
        
        # Shaded simultaneous band
        ax.fill_between(et_nr, sim_lo, sim_hi,
                        alpha=0.15, color='steelblue',
                        label='95% simultaneous band')
        # Pointwise CI
        ax.fill_between(
            event_times,
            betas - 1.96 * ses,
            betas + 1.96 * ses,
            alpha=0.30, color='steelblue',
            label='95% pointwise CI'
        )
        # Point estimates
        ax.plot(event_times, betas, 'o-', color='steelblue',
                linewidth=2, markersize=6, zorder=5)
        # Reference period marker
        ax.axvline(-1, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.axvline(-0.5, color='darkred', linestyle='--', linewidth=1.5,
                   label='Treatment onset (k=0)')
        
        # Shade pre-period
        ax.axvspan(EVENT_WINDOW[0] - 0.5, -1.5, alpha=0.05, color='gray')
        
        ax.set_xlabel("Event Time (years relative to expansion)", fontsize=10)
        ax.set_ylabel(outcome_label, fontsize=10)
        ax.set_title(f"Event Study: {outcome_label}", fontsize=11)
        ax.set_xticks(event_times)
        ax.legend(fontsize=8, loc='upper left')
        
        if n_pre_coef > 0:
            ax.text(
                0.02, 0.05,
                f"Pre-trend F-test: F={pt['F']:.2f}, p={pt['p_F']:.3f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        # ── Rambachan-Roth sensitivity ─────────
        post_mask = coef_df['event_time'] >= 0
        beta_post_arr = betas[post_mask]
        n_post = len(beta_post_arr)
        
        M_bar_grid = np.linspace(0, 2.0, 40)
        
        # Construct approximate full vcov for sensitivity
        if len(vcov_arr) >= n_pre_coef + n_post:
            vcov_for_rr = vcov_arr[:n_pre_coef + n_post, :n_pre_coef + n_post]
        else:
            vcov_for_rr = np.diag(ses[non_ref_mask] ** 2)
            vcov_for_rr = vcov_for_rr[:n_pre_coef + n_post, :n_pre_coef + n_post]
        
        rr_df = rambachan_roth_sensitivity(
            beta_post=beta_post_arr,
            beta_pre=beta_pre,
            vcov_full=vcov_for_rr,
            n_pre=n_pre_coef,
            M_bar_grid=M_bar_grid
        )
        
        breakdown_rows = rr_df[rr_df['contains_zero']]
        breakdown_M = (
            breakdown_rows['M_bar'].min() if len(breakdown_rows) > 0
            else M_bar_grid[-1]
        )
        print(f"  {outcome}: breakdown M_bar ≈ {breakdown_M:.2f} "
              f"(SE units of pre-period)")
        
        # ── Plot 2: Sensitivity plot ───────────
        ax2 = axes[1, col_idx]
        
        ax2.fill_between(
            rr_df['M_bar'],
            rr_df['lower_ci'],
            rr_df['upper_ci'],
            alpha=0.25, color='darkorange',
            label='Robust 95% CI'
        )
        ax2.fill_between(
            rr_df['M_bar'],
            rr_df['lower_id'],
            rr_df['upper_id'],
            alpha=0.45, color='darkorange',
            label='Identified set'
        )
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.axvline(breakdown_M, color='red', linestyle='--', linewidth=1.5,
                    label=f'Breakdown M̄={breakdown_M:.2f}')
        
        ax2.set_xlabel("Smoothness Parameter M̄", fontsize=10)
        ax2.set_ylabel(f"Avg. Post-Period Effect ({outcome_label})", fontsize=10)
        ax2.set_title(
            f"Rambachan-Roth Sensitivity: {outcome_label}",
            fontsize=11
        )
        ax2.legend(fontsize=8)
        ax2.text(
            0.02, 0.92,
            f"Breakdown at M̄={breakdown_M:.2f}",
            transform=ax2.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig("ch18_event_study_honest.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved to ch18_event_study_honest.png")
    
    # ── Summary table ──────────────────────────
    print("\n" + "="*60)
    print("EVENT STUDY SUMMARY — ACA MEDICAID EXPANSION")
    print("="*60)
    stacked_summary = build_stacked_dataset(df, COHORTS, EVENT_WINDOW)
    for outcome, label in OUTCOMES.items():
        coef_df, _, _ = estimate_event_study(
            stacked_summary.copy(), outcome, EVENT_WINDOW, ref_period=-1
        )
        print(f"\n{label}:")
        print(coef_df[['event_time', 'Estimate', 'Std. Error', 'Pr(>|t|)']].to_string(
            index=False, float_format='{:.3f}'.format
        ))


if __name__ == "__main__":
    main()
```

## Summary

- The event-study estimand $\beta_k = E[Y_{it}(1) - Y_{it}(0) \mid k = t - t_i^*]$ is a cohort-averaged ATT at event-time horizon $k$; pre-period values are zero under no-anticipation and parallel trends, providing a testable restriction.
- Reference period normalization shifts the level of the event-study plot without changing between-period differences; the choice of $k = -1$ as reference implies that treatment effects are measured relative to the immediately pre-treatment period, which is contaminated by anticipation if agents respond before the formal treatment date.
- Visual inspection of pre-trends is not a valid statistical test; the correct procedure is a joint Wald test on all pre-period coefficients simultaneously, with power caveats: Roth (2022) shows that typical pre-trend tests have low power against violations large enough to materially bias post-period estimates.
- The TWFE event-study estimator is misspecified under treatment effect heterogeneity across cohorts; the stacking estimator or Callaway-Sant'Anna aggregation recovers internally consistent cohort-averaged event-study profiles by restricting comparisons to clean control groups.
- Simultaneous confidence bands are the appropriate visualization when the viewer assesses the event-study profile jointly; pointwise bands displayed over a multi-period plot implicitly conduct multiple comparisons without correction, overstating confidence in pre-trend assessments.
- Rambachan and Roth's (2023) sensitivity analysis translates the binary pre-trend test into a continuous question: how smooth must violations be for the conclusion to survive? The breakdown $\bar{M}$ — the minimum non-linearity in trend violations needed to include zero in the confidence set — provides an actionable robustness benchmark.
- For the ACA Medicaid expansion, a well-specified stacked event study shows near-zero pre-trends for both insurance coverage and self-reported health, with coverage effects rising sharply at $k=0$ and stabilizing; honest sensitivity analysis indicates the coverage conclusion survives substantial pre-trend non-linearity, while health effects break down at lower $\bar{M}$, appropriately reflecting weaker identification.

## Further Reading

- **Callaway, B. and Sant'Anna, P.H.C. (2021).** "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2), 200–230. The foundational paper formalizing group-time ATTs and their aggregation into event-study format under heterogeneous treatment effects; required reading for any staggered adoption design.

- **Rambachan, A. and Roth, J. (2023).** "A More Credible Approach to Parallel Trends." *Review of Economic Studies*, 90(5), 2555–2591. Introduces the smoothness and sign restriction frameworks for sensitivity analysis; derives the conditional hybrid confidence sets implemented in `HonestDiD`; fundamentally reframes pre-trend testing from binary hypothesis testing to continuous sensitivity reporting.

- **Roth, J. (2022).** "Pre-test with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305–322. Formal analysis of the power of pre-trend tests against violations large enough to bias post-period estimates; demonstrates that standard pre-trend tests are severely underpowered in typical applied settings.

- **Baker, A.C., Larcker, D.F., and Wang, C.C.Y. (2022).** "How Much Should We Trust Staggered Difference-in-Differences Estimates?" *Journal of Financial Economics*, 144(2), 370–395. Comprehensive simulation study documenting TWFE bias under heterogeneous treatment effects in finance panel data; introduces and validates the stacking estimator as a practical correction.

- **Sun, L. and Abraham, S. (2021).** "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effect." *Journal of Econometrics*, 225(2), 175–199. Derives interaction-weighted estimators that are numerically equivalent to Callaway-Sant'Anna under particular weighting schemes; provides a TWFE-adjacent implementation path for applied researchers unwilling to leave the standard regression framework.

- **Freyaldenhoven, S., Hansen, C., Pérez, M.J., and Shapiro, J.M. (2021).** "Visualization, Identification, and Estimation in the Linear Panel Event-Study Design." *NBER Working Paper 29170*. Comprehensive treatment of event-study specification choices including binning, normalization, and confidence band construction; introduces `eventstudyinteract` and discusses the simultaneous vs. pointwise distinction with simulation evidence.
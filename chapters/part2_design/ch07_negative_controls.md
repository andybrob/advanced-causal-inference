# Chapter 7: Negative Controls, Placebos, and Falsification Tests

Causal identification rests on assumptions that, by definition, cannot be directly verified from observed data. A researcher claiming that lottery selection in the Oregon Health Experiment is as-good-as-random, or that pre-expansion trends in Medicaid-eligible and ineligible states were parallel, is making a claim about a counterfactual world. The honest response to this uncomfortable fact is not silence but systematic probing: construct tests whose results would be inconsistent with the identifying assumptions if those assumptions failed, and report what you find.

This chapter develops the formal toolkit for such probing. Negative control outcomes and exposures, placebo tests, and pre-trend falsification are not decorative appendages to a causal analysis—they are the primary empirical evidence that an identification strategy is credible. We will see that this toolkit has a unified logical structure (Section 7.1), that the structure generates testable moment conditions with known distributions under the null (Sections 7.2–7.4), and that it can be extended into a constructive identification strategy when both a negative control outcome and a negative control exposure are available (Section 7.5).

---

## 7.1 The Logic of Falsification

The standard vocabulary distinguishes "identification assumptions" from "testable implications." This distinction is fuzzier than it appears. Most untestable assumptions imply testable moment restrictions when combined with auxiliary structure; the art of falsification is finding those restrictions and computing their power.

**Definition 7.1 (Negative Control Outcome).** A random variable $Y_\text{NCO}$ is a *negative control outcome* for treatment $D$ and identification strategy $\mathcal{S}$ if, under the causal model implied by $\mathcal{S}$, the potential outcomes satisfy $Y_\text{NCO}(1) = Y_\text{NCO}(0)$ almost surely. Equivalently, $D$ has no causal path to $Y_\text{NCO}$.

**Definition 7.2 (Negative Control Exposure).** A random variable $Z_\text{NCE}$ is a *negative control exposure* for outcome $Y$ and identification strategy $\mathcal{S}$ if, under the causal model implied by $\mathcal{S}$, $Z_\text{NCE}$ has no causal path to $Y$: $E[Y(d, Z_\text{NCE} = z)] = E[Y(d)]$ for all $d$ and $z$.

The key point is that both definitions are *conditional* on the identification strategy. Whether pre-lottery health status is a valid NCO for the OHE lottery depends on whether lottery assignment was truly randomized after conditioning on household size strata. The NCO is not a property of the outcome in isolation but a joint claim about the outcome and the assumed data-generating process.

**Proposition 7.1 (NCO Testability).** Let $\mathcal{S}$ be an identification strategy implying unconfoundedness: $D \perp\!\!\!\perp Y(d) \mid X$ for all $d$. If $Y_\text{NCO}$ is a valid NCO under $\mathcal{S}$, then:

$$E[Y_\text{NCO} \mid D = 1, X] = E[Y_\text{NCO} \mid D = 0, X] \quad \text{almost surely.}$$

*Proof.* Under unconfoundedness, $D \perp\!\!\!\perp Y_\text{NCO}(d) \mid X$ for all $d$. Since $Y_\text{NCO}$ is a valid NCO, $Y_\text{NCO}(1) = Y_\text{NCO}(0)$ a.s., call this common value $Y_\text{NCO}^*$. Then:
$$E[Y_\text{NCO} \mid D = 1, X] = E[Y_\text{NCO}^* \mid D = 1, X] = E[Y_\text{NCO}^* \mid X] = E[Y_\text{NCO}^* \mid D = 0, X] = E[Y_\text{NCO} \mid D = 0, X],$$
where the second equality uses $D \perp\!\!\!\perp Y_\text{NCO}^* \mid X$. $\square$

The proposition converts an untestable assumption (unconfoundedness) into a testable moment condition. Rejection of the moment condition refutes at least one of: unconfoundedness, the claim that $Y_\text{NCO}$ has no causal relationship to $D$, or a combination. Non-rejection does not confirm unconfoundedness—it is consistent with it.

**The asymmetry of evidence.** This asymmetry deserves emphasis. A battery of negative control tests that all pass constitutes weak positive evidence: the data are *consistent* with the identification strategy. A single well-powered test that fails constitutes strong negative evidence: the identification strategy is *inconsistent* with the data under the auxiliary assumption that the NCO is valid. Researchers who report only passing tests, or who report failing tests without discussing what went wrong, are not doing science.

---

## 7.2 Negative Control Outcomes in the Oregon Health Experiment

The OHE lottery assigned Medicaid eligibility by household. Individuals on the household waitlist were selected with probability proportional to household size (hence the need to condition on `numhh_list`). Within strata, selection is random, so the ideal is:

$$Z \perp\!\!\!\perp Y_\text{pre} \mid \text{numhh\_list},$$

where $Z$ is lottery selection and $Y_\text{pre}$ is any pre-lottery characteristic. This is identical in form to Proposition 7.1, with $D$ replaced by $Z$ (the instrumental variable) and $X$ replaced by strata dummies.

**Baseline balance as NCO testing.** A pre-lottery health outcome cannot be caused by post-lottery events. If the lottery is properly randomized within strata, selected and non-selected individuals should have identical distributions of pre-lottery outcomes after conditioning on strata. Any imbalance is direct evidence of either flawed randomization or differential survey response (itself a form of selection confounding).

The standard balance test regresses each pre-lottery outcome on treatment indicator and strata dummies:

$$Y_\text{pre,i} = \alpha + \beta Z_i + \gamma' \mathbf{s}_i + \varepsilon_i,$$

where $\mathbf{s}_i$ is a vector of strata dummies and the null $H_0: \beta = 0$ is the NCO condition. Reporting $\hat{\beta}$, its standard error, and the omnibus $F$-statistic across all pre-lottery outcomes is standard practice. The OHE paper (Finkelstein et al., 2012) reports this in their Appendix Table 1.

**Permutation inference for balance.** Classical $t$-tests assume the sampling distribution is well-approximated asymptotically. For small strata or non-normal outcomes, a permutation test is more reliable. Under $H_0$, the assignment mechanism is known: within each stratum, treatment is randomly assigned with known probabilities. Permuting assignments within strata preserves this structure and generates the exact null distribution.

---

## 7.3 Pre-Trend Placebo Tests in Difference-in-Differences

The ACA Medicaid expansion provides a canonical DiD setting. Approximately half of U.S. states expanded Medicaid in 2014, with a handful expanding earlier (2010–2013) and later (2015–2016). The identifying assumption is parallel trends: absent expansion, health outcomes would have evolved identically in expansion and non-expansion states.

**Pre-trend tests as NCO tests.** Pre-period outcomes are negative control outcomes for a treatment that has not yet occurred. The placebo regression is:

$$Y_{st} = \alpha_s + \lambda_t + \sum_{\ell = -L}^{-1} \delta_\ell \cdot \mathbf{1}[t - T_s^* = \ell] + \varepsilon_{st},$$

where $\alpha_s$ are state fixed effects, $\lambda_t$ are time fixed effects, $T_s^*$ is the expansion year for state $s$ (undefined for never-expanders, who are the control group), and $\ell$ indexes event-time leads. Under parallel trends, $\delta_\ell = 0$ for all $\ell < 0$.

**Theorem 7.1 (Pre-trend test power).** Let $\delta_\ell^* = \beta_\ell$ denote the true pre-period differential trend per period, and let $\hat{\delta}_\ell \sim N(\beta_\ell, \sigma_\ell^2)$ asymptotically. The power of a level-$\alpha$ test against the local alternative $\beta_\ell = \Delta$ is:

$$\pi(\Delta) = \Phi\!\left(-z_{\alpha/2} + \frac{|\Delta|}{\sigma_\ell}\right) + \Phi\!\left(-z_{\alpha/2} - \frac{|\Delta|}{\sigma_\ell}\right),$$

where $\Phi$ is the standard normal CDF. The minimum detectable effect at power $1-\kappa$ is $\text{MDE} = (z_{\alpha/2} + z_\kappa) \cdot \sigma_\ell$.

*Proof.* Standard: under $H_0: \beta_\ell = 0$, reject when $|\hat{\delta}_\ell| > z_{\alpha/2} \sigma_\ell$. Under $H_a: \beta_\ell = \Delta$, $P(|\hat{\delta}_\ell| > z_{\alpha/2} \sigma_\ell) = P(\hat{\delta}_\ell > z_{\alpha/2}\sigma_\ell) + P(\hat{\delta}_\ell < -z_{\alpha/2}\sigma_\ell)$. Standardizing gives the result. $\square$

The MDE is the critical quantity for interpretation. A researcher who reports "pre-trends look fine" without reporting $\sigma_\ell$ is hiding the fact that the test may be wildly underpowered against economically meaningful violations. If $\text{MDE} = 5$ percentage points and the estimated treatment effect is $3$ percentage points, a passing pre-trend test is nearly worthless as validation.

**Placebo outcomes: injury hospitalizations.** Medicaid expansion should not immediately affect injury-related hospitalizations—these are determined by accidents, not insurance coverage, at least in the short run. Using injury hospitalization rates as a placebo outcome tests whether expansion states were on differential trends for *any* health outcome. If the DiD estimate for injury hospitalizations is large and significant, something other than the Medicaid expansion is driving differential trends, and the causal interpretation of the main estimate is in jeopardy.

More formally, let $Y^\text{inj}$ be injury hospitalization rate. Under parallel trends plus no immediate effect:

$$E[Y^\text{inj}_{st} \mid s \in \text{Expansion}, t > T^*] - E[Y^\text{inj}_{st} \mid s \in \text{Expansion}, t < T^*]$$
$$- \left(E[Y^\text{inj}_{st} \mid s \notin \text{Expansion}, t > T^*] - E[Y^\text{inj}_{st} \mid s \notin \text{Expansion}, t < T^*]\right) = 0.$$

A non-zero estimate falsifies either parallel trends or the assumption that expansion has no short-run effect on injuries—and the latter assumption is difficult to challenge credibly.

---

## 7.4 Negative Control Exposures

NCEs are less commonly used than NCOs but are powerful when available. The logic reverses: instead of finding an outcome that treatment cannot affect, find an exposure that the outcome cannot be affected by.

**Definition restated.** $Z_\text{NCE}$ is a valid NCE for outcome $Y$ if the structural equation for $Y$ does not contain $Z_\text{NCE}$, even though $Z_\text{NCE}$ and the true treatment $D$ may be associated (e.g., because they share common causes).

**Example: geography as NCE.** In the ACA setting, a state's latitude has no plausible causal effect on individual health outcomes (conditional on baseline covariates). But if there is unmeasured confounding—say, political culture correlates with both expansion timing and health investment—then latitude may be associated with both expansion timing and outcomes through that confounder. Testing whether latitude predicts health outcomes (beyond its association with expansion) can reveal confounding.

**Moment condition.** Under correct identification of $\tau = E[Y(1) - Y(0)]$ via strategy $\mathcal{S}$, the residualized outcome $\tilde{Y} = Y - \hat{\tau} D$ should be mean-independent of any valid NCE:

$$E[\tilde{Y} \mid Z_\text{NCE}] = E[\tilde{Y}]. \tag{7.1}$$

If $(7.1)$ fails, either $\hat{\tau}$ is wrong or $Z_\text{NCE}$ is not a valid NCE. The test thus operates on the residual, after subtracting the estimated causal effect, making it sensitive to both the sign and magnitude of any confounding.

---

## 7.5 The Proximal Causal Inference Framework

NCOs and NCEs can be combined into a constructive identification strategy, not merely a diagnostic one. This is the insight of Miao, Geng, and Tchetgen Tchetgen (2018) and the subsequent proximal causal inference literature.

**Setup.** Let $U$ be unmeasured confounding. Suppose we observe:
- $W$: a negative control exposure (affected by $U$ but not causally by $Y$)
- $Z$: a negative control outcome variable (affected by $U$ but not causally by $D$)

The key distinction from the diagnostic uses above is that here $W$ and $Z$ are proxies for $U$, not simply variables that happen to satisfy the NCE/NCO conditions. Both are "leaky" in that they imperfectly capture $U$.

**Theorem 7.2 (Proximal Identification, Miao-Tchetgen-Tchetgen).** Under the proximal identification assumptions—(i) $D \perp\!\!\!\perp Z \mid U, X$; (ii) $W \perp\!\!\!\perp Y(d) \mid U, X$ for all $d$; (iii) completeness conditions on $(W, U)$ and $(Z, U)$—the average treatment effect is identified by:

$$E[Y(d)] = \int E[Y \mid D = d, W = w, X = x] \cdot h(w, d, x) \, dw \, dP(x), \tag{7.2}$$

where $h(\cdot)$ is the solution to the integral equation:

$$E[Z \mid D = d, W = w, X] = \int E[Z \mid U = u, X] \cdot h(w, d, X) \, dP(U \mid W = w, X) \, dw. \tag{7.3}$$

*Proof sketch.* The integral equation $(7.3)$ is a Fredholm equation of the first kind. Under the completeness condition on $(Z, U)$, the operator $\mathcal{T}: h \mapsto E[Z \mid D, W, X]$ is injective, so $h$ is identified as the solution. Given $h$, $(7.2)$ follows from iterated expectations: $E[Y(d)] = E_X E_{U \mid X} E[Y \mid D=d, U, X]$, which can be rewritten using $h$ to eliminate $U$ in favor of the observable $W$. Full proof in Miao, Geng, and Tchetgen Tchetgen (2018), Theorem 1. $\square$

The practical importance of Theorem 7.2 is that it moves negative controls from a diagnostic tool (checking assumptions) to an identification tool (recovering causal effects in the presence of unmeasured confounding). The price is two completeness conditions, which are not directly testable, and the computational burden of solving the integral equation.

**OHE illustration.** In the OHE, suppose we are concerned that lottery compliance (actually enrolling in Medicaid conditional on winning) is confounded by unmeasured health motivation $U$. A pre-lottery health behavior (e.g., whether the person previously sought emergency care) might serve as $W$ (a proxy for $U$ that doesn't cause post-lottery outcomes directly), and a post-lottery outcome that enrollment cannot plausibly affect (e.g., dental visits, if the Medicaid plan had no dental coverage) might serve as $Z$. The proximal framework then yields an identification formula for the LATE that is robust to this specific form of confounding.

---

## 7.6 Formal Relationship Between NCO Tests and Sensitivity Analysis

Negative control tests and sensitivity analyses are dual approaches to the same problem. A sensitivity analysis asks: how large must confounding be to explain away the estimated effect? An NCO test asks: is there any evidence that confounding of that magnitude exists?

**Proposition 7.2 (Duality).** Let $\Gamma$ parameterize the degree of unmeasured confounding (Rosenbaum's $\Gamma$-sensitivity). Let $\delta_\text{NCO}$ be the observed NCO imbalance and $\sigma_\text{NCO}$ its standard error. If the effect of confounding on $Y_\text{NCO}$ is proportional to its effect on $Y$ by factor $\rho$, then the NCO test rejects at level $\alpha$ if and only if:

$$\frac{|\delta_\text{NCO}|}{\sigma_\text{NCO}} > z_{\alpha/2} \iff \Gamma > \exp\!\left(\frac{\rho \cdot |\tau|}{\text{SD}(Y)}\right), \tag{7.4}$$

*approximately*, under a logistic sensitivity model.

This means that a passing NCO test rules out confounding above a threshold $\Gamma^*$ only if the test has sufficient power—that is, only if $\rho$ is not too small. A pre-lottery health variable that is only weakly related to the unmeasured confounder (small $\rho$) provides a weak test. Researchers should seek NCOs that are closely related to the suspected confounding mechanism.

---

## Python: Negative Controls and Placebo Tests

```python
"""
Chapter 7: Negative Controls, Placebos, and Falsification Tests
Oregon Health Experiment + ACA BRFSS panel
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── 0. Data loading helpers ──────────────────────────────────────────────────

def load_ohp_data() -> pd.DataFrame:
    """
    Load OHE person-level data.
    Expects ohp_allmatch_completed.dta from https://data.nber.org/oregon/
    Falls back to a synthetic dataset with identical schema if not found.
    """
    try:
        df = pd.read_stata(
            "data/ohp_allmatch_completed.dta",
            columns=[
                "treatment",      # Z: lottery winner
                "ohp_all_ever_admin",  # D: ever enrolled
                "doc_any_12m",    # Y: any doctor visit
                "catastrophic_exp_inp",  # Y: catastrophic inpatient expense
                "numhh_list",     # strata
                # Pre-lottery NCOs
                "er_any_pre_lottery",       # any ER visit before lottery
                "health_poor_pre_lottery",  # self-reported poor health pre
                "cost_any_oop_pre",         # any out-of-pocket cost pre
            ],
        )
        df = df.rename(columns={"treatment": "Z", "ohp_all_ever_admin": "D"})
        df = df.dropna(subset=["Z", "numhh_list"])
        print(f"Loaded OHE data: n={len(df):,}")
        return df
    except FileNotFoundError:
        return _synthetic_ohp(n=20_745, seed=42)


def _synthetic_ohp(n: int = 20_745, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHE-schema data with known balance properties."""
    rng = np.random.default_rng(seed)
    # Strata: household size 1, 2, 3+
    numhh = rng.choice([1, 2, 3], size=n, p=[0.5, 0.35, 0.15])
    # Selection probability varies by strata (as in real OHE)
    p_select = np.where(numhh == 1, 0.30, np.where(numhh == 2, 0.45, 0.55))
    Z = rng.binomial(1, p_select)
    # Compliance: LATE structure
    U = rng.normal(0, 1, n)  # unmeasured health motivation
    D = ((Z == 1) & (rng.uniform(size=n) < 0.40)).astype(int)
    # Pre-lottery NCOs: should be balanced given Z
    er_pre = rng.binomial(1, 0.15 + 0.05 * (U > 1))
    health_poor_pre = rng.binomial(1, 0.20 + 0.05 * (U > 1))
    cost_oop_pre = rng.binomial(1, 0.35 + 0.05 * (U > 1))
    # Post-lottery outcomes
    doc_any = rng.binomial(1, 0.55 + 0.12 * D + 0.05 * (U > 0))
    catastrophic = rng.binomial(1, 0.08 - 0.04 * D + 0.02 * (U > 1))

    return pd.DataFrame({
        "Z": Z, "D": D,
        "doc_any_12m": doc_any,
        "catastrophic_exp_inp": catastrophic,
        "numhh_list": numhh,
        "er_any_pre_lottery": er_pre,
        "health_poor_pre_lottery": health_poor_pre,
        "cost_any_oop_pre": cost_oop_pre,
    })


def load_aca_panel() -> pd.DataFrame:
    """
    Synthetic state-year BRFSS panel, 2008-2016.
    Mimics ACA Medicaid expansion staggered DiD structure.
    """
    rng = np.random.default_rng(7)
    states = [f"S{i:02d}" for i in range(1, 51)]
    years = list(range(2008, 2017))
    # Expansion timing
    expansion_year = {}
    for s in states[:25]:   # expanders
        expansion_year[s] = rng.choice([2014, 2014, 2014, 2015, 2016])
    for s in states[25:]:   # non-expanders
        expansion_year[s] = 9999

    rows = []
    for s in states:
        state_trend = rng.normal(0, 0.3)
        state_fe = rng.normal(0, 1)
        injury_fe = rng.normal(0, 0.5)
        for y in years:
            expanded = int(y >= expansion_year[s])
            treat_effect = 2.5 * expanded  # pct-point on health_good
            injury_effect = 0.0            # true null for placebo outcome
            health_good = (
                50 + state_fe + state_trend * (y - 2008)
                + treat_effect + rng.normal(0, 2)
            )
            injury_rate = (
                8 + injury_fe + 0.05 * (y - 2008)
                + injury_effect * expanded + rng.normal(0, 0.8)
            )
            rows.append({
                "state": s, "year": y,
                "expanded": expanded,
                "expansion_year": expansion_year[s],
                "health_good_pct": health_good,
                "injury_hosp_rate": injury_rate,
                "event_time": y - expansion_year[s] if expansion_year[s] < 9999 else np.nan,
            })
    return pd.DataFrame(rows)


# ── 1. OHE: NCO balance test with permutation inference ─────────────────────

def ohp_nco_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regress each pre-lottery NCO on Z + strata dummies.
    Report coefficient, SE, p-value (OLS and permutation).
    """
    nco_vars = [c for c in df.columns if "pre" in c]
    strata_dummies = pd.get_dummies(df["numhh_list"], prefix="hh", drop_first=True)
    X_ctrl = pd.concat([strata_dummies], axis=1).astype(float)
    results = []

    for var in nco_vars:
        y = df[var].values.astype(float)
        mask = ~np.isnan(y)
        y_, Z_ = y[mask], df["Z"].values[mask]
        X_ = X_ctrl[mask].values
        X_full = np.column_stack([np.ones(mask.sum()), Z_, X_])

        coef, _, _, _ = lstsq(X_full, y_)
        beta = coef[1]
        resid = y_ - X_full @ coef
        n, k = X_full.shape
        sigma2 = (resid @ resid) / (n - k)
        XtXinv = np.linalg.inv(X_full.T @ X_full)
        se_beta = np.sqrt(sigma2 * XtXinv[1, 1])
        t_stat = beta / se_beta
        p_ols = 2 * stats.t.sf(abs(t_stat), df=n - k)

        # Permutation test: permute Z within strata
        t_perm = []
        strata_ = df["numhh_list"].values[mask]
        for _ in range(2000):
            Z_perm = Z_.copy()
            for s in np.unique(strata_):
                idx = np.where(strata_ == s)[0]
                Z_perm[idx] = rng_perm.permutation(Z_[idx])
            X_p = np.column_stack([np.ones(n), Z_perm, X_])
            c_p, _, _, _ = lstsq(X_p, y_)
            r_p = y_ - X_p @ c_p
            s2_p = (r_p @ r_p) / (n - k)
            se_p = np.sqrt(s2_p * np.linalg.inv(X_p.T @ X_p)[1, 1])
            t_perm.append(c_p[1] / se_p)
        p_perm = np.mean(np.abs(t_perm) >= abs(t_stat))

        results.append({
            "NCO": var, "beta": beta, "SE": se_beta,
            "t": t_stat, "p_OLS": p_ols, "p_permutation": p_perm,
        })

    return pd.DataFrame(results)


rng_perm = np.random.default_rng(99)


# ── 2. MDE computation for pre-trend test ────────────────────────────────────

def compute_mde(sigma: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable effect for pre-trend coefficient at given power."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_kappa = stats.norm.ppf(power)
    return (z_alpha + z_kappa) * sigma


def pre_trend_power_curve(sigmas: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Power as function of true pre-trend slope Delta, for given SE."""
    deltas = np.linspace(0, 5, 200)
    records = []
    for sigma in sigmas:
        z_a = stats.norm.ppf(1 - alpha / 2)
        power = stats.norm.sf(z_a - deltas / sigma) + stats.norm.cdf(-z_a - deltas / sigma)
        for d, pw in zip(deltas, power):
            records.append({"sigma": sigma, "delta": d, "power": pw})
    return pd.DataFrame(records)


# ── 3. ACA panel: event-study + placebo outcome ──────────────────────────────

def event_study_estimates(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Event-study regression: leads/lags around expansion year.
    Comparison group: never-expanders + pre-period expanders.
    """
    df = panel.copy()
    expanders = df[df["expansion_year"] < 9999].copy()
    never = df[df["expansion_year"] == 9999].copy()

    # Construct event-time dummies for expanders, clip at [-4, 4]
    expanders["et"] = expanders["event_time"].clip(-4, 4).astype(int)
    # Drop et == -1 as reference
    et_dummies = pd.get_dummies(expanders["et"], prefix="et")
    et_dummies = et_dummies.drop(columns=["et_-1"], errors="ignore")

    # Stack never-expanders with NaN event-time → all dummies = 0
    never_dummies = pd.DataFrame(
        0, index=never.index, columns=et_dummies.columns
    )
    combined = pd.concat([expanders, never], ignore_index=True)
    dummy_combined = pd.concat([et_dummies, never_dummies], ignore_index=True)
    combined = pd.concat([combined.reset_index(drop=True), dummy_combined], axis=1)

    # Add state and year FE via dummies
    state_fe = pd.get_dummies(combined["state"], drop_first=True, prefix="st")
    year_fe = pd.get_dummies(combined["year"], drop_first=True, prefix="yr")
    y = combined[outcome].values
    X = np.column_stack([
        np.ones(len(combined)),
        dummy_combined.values,
        state_fe.values,
        year_fe.values,
    ])
    coef, _, _, _ = lstsq(X, y)
    resid = y - X @ coef
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    # Clustered SE approximation: HC1
    meat = (X.T * resid**2) @ X
    bread = np.linalg.inv(X.T @ X)
    hc1 = (n / (n - k)) * bread @ meat @ bread
    n_et = dummy_combined.shape[1]
    betas = coef[1:1+n_et]
    ses = np.sqrt(np.diag(hc1)[1:1+n_et])
    et_labels = [c.replace("et_", "") for c in dummy_combined.columns]
    return pd.DataFrame({
        "event_time": [int(e) for e in et_labels],
        "beta": betas, "se": ses,
        "ci_lo": betas - 1.96 * ses,
        "ci_hi": betas + 1.96 * ses,
        "outcome": outcome,
    })


# ── 4. Proximal causal inference illustration ────────────────────────────────

def simulate_proximal(n: int = 5000, seed: int = 123) -> dict:
    """
    Simulate the proximal setup:
      U ~ N(0,1) unmeasured confounder
      W ~ N(U, 1)  negative control exposure (proxy for U)
      D = 1[alpha*U + gamma*Z + noise > 0], Z is instrument
      Z ~ N(0,1) (instrument, excluded from Y)
      Y = tau*D + beta*U + eps   true tau = 2.0
      NCO: Q ~ N(U, 1), not affected by D  (negative control outcome)

    Naive OLS ignores U → biased. Proximal method recovers tau.
    """
    rng = np.random.default_rng(seed)
    true_tau = 2.0
    U = rng.normal(0, 1, n)
    Z = rng.normal(0, 1, n)
    W = U + rng.normal(0, 1, n)       # NCE proxy
    Q = U + rng.normal(0, 1, n)       # NCO proxy
    D_star = 0.8 * U + 1.2 * Z + rng.normal(0, 1, n)
    D = (D_star > 0).astype(float)
    Y = true_tau * D + 1.5 * U + rng.normal(0, 1, n)

    # Naive OLS: omits U
    X_naive = np.column_stack([np.ones(n), D])
    c_naive, _, _, _ = lstsq(X_naive, Y)
    tau_naive = c_naive[1]

    # Oracle OLS: includes U
    X_oracle = np.column_stack([np.ones(n), D, U])
    c_oracle, _, _, _ = lstsq(X_oracle, Y)
    tau_oracle = c_oracle[1]

    # Proximal estimator: two-stage using W and Q
    # Stage 1: regress D on Z, W (bridging function for D)
    X1 = np.column_stack([np.ones(n), Z, W])
    c1, _, _, _ = lstsq(X1, D)
    D_hat = X1 @ c1  # E[D | Z, W]
    # Stage 2: regress Y on D_hat, Q (outcome bridge)
    X2 = np.column_stack([np.ones(n), D_hat, Q])
    c2, _, _, _ = lstsq(X2, Y)
    tau_proximal = c2[1]

    return {
        "true_tau": true_tau,
        "tau_naive": tau_naive,
        "tau_oracle": tau_oracle,
        "tau_proximal": tau_proximal,
        "n": n,
    }


# ── 5. Visualization ─────────────────────────────────────────────────────────

def plot_chapter7(
    balance_df: pd.DataFrame,
    es_main: pd.DataFrame,
    es_placebo: pd.DataFrame,
    power_df: pd.DataFrame,
    proximal_results: dict,
):
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Panel A: NCO balance coefficients + 95% CI
    ax_a = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(balance_df))
    ax_a.barh(y_pos, balance_df["beta"], xerr=1.96 * balance_df["SE"],
              color=["#d73027" if p < 0.1 else "#4575b4"
                     for p in balance_df["p_permutation"]],
              capsize=4, height=0.5)
    ax_a.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(
        [v.replace("_pre_lottery", "").replace("_pre", "")
         for v in balance_df["NCO"]], fontsize=8
    )
    ax_a.set_xlabel("Coefficient on Z (winner)")
    ax_a.set_title("A: NCO Balance (OHE)", fontweight="bold")
    ax_a.text(0.98, 0.04, "Blue = p≥0.10 (perm)", transform=ax_a.transAxes,
              ha="right", fontsize=7)

    # Panel B: Pre-trend power curves
    ax_b = fig.add_subplot(gs[0, 1])
    for sigma, grp in power_df.groupby("sigma"):
        ax_b.plot(grp["delta"], grp["power"], label=f"SE={sigma:.1f}")
    ax_b.axhline(0.80, color="gray", linestyle=":", linewidth=0.9)
    ax_b.set_xlabel("True pre-trend slope Δ (pct-pts)")
    ax_b.set_ylabel("Power")
    ax_b.set_title("B: Pre-trend Test Power", fontweight="bold")
    ax_b.legend(fontsize=8)

    # Panel C: Event study — main outcome
    ax_c = fig.add_subplot(gs[0, 2])
    es_main_s = es_main.sort_values("event_time")
    ax_c.errorbar(
        es_main_s["event_time"], es_main_s["beta"],
        yerr=1.96 * es_main_s["se"], fmt="o-", color="#1a9850", capsize=4
    )
    ax_c.axvline(-0.5, color="black", linewidth=0.8, linestyle="--")
    ax_c.axhline(0, color="gray", linewidth=0.6)
    ax_c.set_xlabel("Event time (years to expansion)")
    ax_c.set_ylabel("Coefficient (pct-pts)")
    ax_c.set_title("C: Event Study – Health Good %", fontweight="bold")

    # Panel D: Event study — placebo outcome
    ax_d = fig.add_subplot(gs[1, 0])
    es_pl_s = es_placebo.sort_values("event_time")
    ax_d.errorbar(
        es_pl_s["event_time"], es_pl_s["beta"],
        yerr=1.96 * es_pl_s["se"], fmt="s--", color="#d73027", capsize=4
    )
    ax_d.axvline(-0.5, color="black", linewidth=0.8, linestyle="--")
    ax_d.axhline(0, color="gray", linewidth=0.6)
    ax_d.set_xlabel("Event time (years to expansion)")
    ax_d.set_ylabel("Coefficient (per 100k)")
    ax_d.set_title("D: Placebo – Injury Hospitalization", fontweight="bold")

    # Panel E: Proximal estimator comparison
    ax_e = fig.add_subplot(gs[1, 1])
    labels = ["True τ", "Naive OLS", "Oracle OLS", "Proximal"]
    vals = [
        proximal_results["true_tau"],
        proximal_results["tau_naive"],
        proximal_results["tau_oracle"],
        proximal_results["tau_proximal"],
    ]
    colors = ["#2166ac", "#d73027", "#1a9850", "#762a83"]
    bars = ax_e.bar(labels, vals, color=colors, width=0.5)
    ax_e.axhline(proximal_results["true_tau"], color="#2166ac",
                 linestyle="--", linewidth=0.9)
    ax_e.set_ylabel("Estimated τ")
    ax_e.set_title("E: Proximal vs. Naive vs. Oracle", fontweight="bold")
    for bar, v in zip(bars, vals):
        ax_e.text(bar.get_x() + bar.get_width()/2, v + 0.04,
                  f"{v:.2f}", ha="center", fontsize=8)

    # Panel F: NCO imbalance → sensitivity Gamma linkage
    ax_f = fig.add_subplot(gs[1, 2])
    gammas = np.linspace(1.0, 3.0, 200)
    rho_vals = [0.3, 0.6, 1.0]
    tau_est = 0.10   # illustrative estimated ATE on 0-1 outcome
    sd_y = 0.30
    for rho in rho_vals:
        # NCO imbalance needed to reject if confounder is at level Gamma
        delta_nco = rho * tau_est * np.log(gammas) / sd_y
        ax_f.plot(gammas, delta_nco, label=f"ρ={rho}")
    ax_f.set_xlabel("Sensitivity parameter Γ")
    ax_f.set_ylabel("NCO imbalance to detect (std units)")
    ax_f.set_title("F: NCO Test vs. Sensitivity Γ", fontweight="bold")
    ax_f.legend(fontsize=8)
    ax_f.axhline(0.10, color="gray", linestyle=":", linewidth=0.9,
                 label="observed imbalance")

    fig.suptitle(
        "Chapter 7: Negative Controls, Placebos, and Falsification Tests",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.savefig("chapter7_figures.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved chapter7_figures.pdf")


# ── 6. Main execution ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 7: Negative Controls and Falsification Tests")
    print("=" * 60)

    # Load data
    ohp = load_ohp_data()
    aca = load_aca_panel()

    # 1. NCO balance in OHE
    print("\n--- OHE: Pre-lottery NCO Balance ---")
    balance = ohp_nco_balance(ohp)
    print(balance.to_string(index=False, float_format="{:.4f}".format))

    # 2. Pre-trend power curves
    print("\n--- Pre-trend Test Power ---")
    sigmas = np.array([0.5, 1.0, 2.0])
    power_df = pre_trend_power_curve(sigmas)
    for sigma in sigmas:
        mde = compute_mde(sigma)
        print(f"  SE={sigma:.1f}: MDE at 80% power = {mde:.2f} pct-pts")

    # 3. ACA event studies
    print("\n--- ACA Event Study: Health Good % ---")
    es_health = event_study_estimates(aca, "health_good_pct")
    print(es_health[["event_time","beta","se","ci_lo","ci_hi"]].to_string(index=False))

    print("\n--- ACA Event Study: Injury Hosp (Placebo) ---")
    es_injury = event_study_estimates(aca, "injury_hosp_rate")
    print(es_injury[["event_time","beta","se","ci_lo","ci_hi"]].to_string(index=False))

    # 4. Proximal causal inference illustration
    print("\n--- Proximal Causal Inference ---")
    proximal = simulate_proximal(n=5000)
    print(f"  True τ       : {proximal['true_tau']:.2f}")
    print(f"  Naive OLS    : {proximal['tau_naive']:.2f}  (biased upward)")
    print(f"  Oracle OLS   : {proximal['tau_oracle']:.2f}  (infeasible benchmark)")
    print(f"  Proximal est.: {proximal['tau_proximal']:.2f}  (uses NCO + NCE)")

    # 5. Produce figures
    plot_chapter7(balance, es_health, es_injury, power_df, proximal)
```

---

## Summary

- **Negative control outcomes** (NCOs) are variables that cannot be causally affected by treatment; under correct identification, any association between treatment and an NCO is evidence of confounding or identification failure.

- **The asymmetry principle** governs falsification: a failed NCO test strongly refutes the identification strategy; a passed NCO test only fails to refute it. Power analysis, via the minimum detectable effect formula $\text{MDE} = (z_{\alpha/2} + z_\kappa)\sigma$, is essential for interpreting null results.

- **Pre-trend tests in DiD** are a special case: pre-period outcomes are NCOs for a treatment that has not yet occurred, and event-study coefficient plots with confidence intervals are the standard display.

- **Placebo outcomes**—outcomes that treatment cannot plausibly affect, such as injury hospitalizations in the ACA context—extend the NCO logic to the post-period and test whether observed treatment-outcome associations are specific to the proposed mechanism.

- **Negative control exposures** (NCEs) test the residual outcome after removing the estimated treatment effect; rejection implies either estimation error or invalidity of the NCE, providing a check orthogonal to NCO tests.

- **Proximal causal inference** (Miao-Tchetgen-Tchetgen) converts the NCO/NCE toolkit from a diagnostic to a constructive identification strategy: under completeness conditions, the joint availability of an NCO and NCE proxy for unmeasured confounding enables point identification of the ATE via a Fredholm integral equation.

- **The sensitivity-NCO duality** (Proposition 7.2) links the observable NCO imbalance to the sensitivity parameter $\Gamma$: a high-$\rho$ NCO that passes a powerful balance test rules out confounding above a quantifiable threshold, directly tightening the conclusions of a Rosenbaum-style sensitivity analysis.

---

## Further Reading

1. **Miao, W., Geng, Z., & Tchetgen Tchetgen, E.J. (2018).** "Identifying causal effects with proxy variables of an unmeasured confounder." *Biometrika*, 105(4), 987–993. — The foundational paper for proximal identification; proves Theorem 7.2 in full generality.

2. **Finkelstein, A., Taubman, S., Wright, B., Bernstein, M., Gruber, J., Newhouse, J.P., ... & Baicker, K. (2012).** "The Oregon health insurance experiment: Evidence from the first year." *Quarterly Journal of Economics*, 127(3), 1057–1106. — Primary OHE paper; Appendix Table 1 is the canonical example of NCO balance reporting.

3. **Roth, J. (2022).** "Pre-test with caution: Event-study estimates after testing for parallel trends." *American Economic Review: Insights*, 4(3), 305–322. — Shows that pre-testing for parallel trends and then running the main regression distorts inference; derives corrections; essential reading before reporting event-study estimates.

4. **Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010).** "Negative controls: A tool for detecting confounding and bias in observational studies." *Epidemiology*, 21(3), 383–388. — Accessible introduction to NCO and NCE logic from the epidemiology perspective; useful for building intuition before the formal treatment.

5. **Callaway, B., & Sant'Anna, P.H.C. (2021).** "Difference-in-differences with multiple time periods." *Journal of Econometrics*, 225(2), 200–230. — Addresses the staggered adoption setting of the ACA panel; pre-trend tests under heterogeneous treatment timing require the group-time ATT framework developed here.

6. **Rosenbaum, P.R. (2002).** *Observational Studies* (2nd ed.), Springer, Chapters 4–5. — The canonical reference for sensitivity analysis and its formal $\Gamma$-parameterization; Proposition 7.2 in this chapter bridges this framework to the NCO literature.
# Chapter 49: Choosing the Right Method Under Constraints

## 49.1 The Method Selection Problem

Every causal inference project begins with a question and ends—if successful—with a credible answer. Between those two points lies a sequence of decisions whose quality determines whether the final estimate is trustworthy or merely plausible. This chapter provides a systematic framework for that decision sequence.

The naive view treats method selection as matching a technique to a dataset: "I have panel data, so I'll use difference-in-differences." The mature view treats method selection as a constrained optimization problem. The objective is minimizing the mean squared error of a causal estimand under a set of constraints—data availability, institutional knowledge about the assignment mechanism, plausibility of identifying assumptions, and the robustness requirements imposed by the research question itself.

Define the setting formally. We observe data $\mathcal{D} = \{(Y_i, D_i, X_i, Z_i, \text{pre}_{it})\}$ where $Y_i$ is the outcome, $D_i$ the treatment, $X_i$ covariates, $Z_i$ a candidate instrument, and $\text{pre}_{it}$ pre-treatment panel observations when available. The target estimand is $\tau = E[Y_i(1) - Y_i(0)]$ or a local or conditional variant. The identified set for $\tau$ under assumptions $\mathcal{A}$ is $\mathcal{I}(\tau, \mathcal{A}, \mathcal{D})$.

The core tension in method selection: stronger assumptions shrink $\mathcal{I}(\tau, \mathcal{A}, \mathcal{D})$ toward a point but increase the damage from assumption failure. Weaker assumptions protect against misspecification but yield wide identified sets or require larger samples to achieve precision. This is the **honesty-credibility frontier**, formalized in Section 49.4.

## 49.2 A Taxonomy of Identifying Assumptions

Before building the decision framework, we need a common language for what each method requires. Table 49.1 organizes the major methods by their identifying assumption category and data requirement.

Each method family rests on a primary assumption that cannot be tested with the observed data and a set of auxiliary assumptions that can be partially assessed.

**Selection on observables** (IPW, AIPW, matching) requires that $D_i \perp\!\!\!\perp (Y_i(0), Y_i(1)) \mid X_i$. The identifying power is entirely in the richness of $X_i$ relative to the omitted confounders. The assumption is untestable but partially assessable through covariate balance and sensitivity analysis (Chapter 47).

**Difference-in-differences** requires $E[Y_{it}(0) - Y_{it-1}(0) \mid D_i = 1] = E[Y_{it}(0) - Y_{it-1}(0) \mid D_i = 0]$—the parallel trends (PT) assumption. This is an assumption about counterfactual dynamics, not observed dynamics. Pre-trend tests assess plausibility of PT but do not test it directly.

**Instrumental variables** requires (i) relevance: $\text{Cov}(Z_i, D_i) \neq 0$; (ii) exclusion: $Z_i \perp\!\!\!\perp Y_i(d)$ for all $d$; (iii) independence: $Z_i \perp\!\!\!\perp (Y_i(0), Y_i(1), D_i(0), D_i(1))$. Relevance is testable; exclusion and independence are not.

**Regression discontinuity** requires continuity of $E[Y_i(0) \mid X_i = x]$ and $E[Y_i(1) \mid X_i = x]$ at the cutoff. It is the most credible design when the forcing variable is continuous and agents cannot precisely manipulate their value.

**Synthetic control** and **matrix completion** require low-rank structure in the untreated potential outcome matrix, with the donor pool providing the span needed to reconstruct the treated unit's counterfactual.

## 49.3 The Decision Framework

The decision framework operates as a sequence of gates. At each gate, a question about the research setting either opens a path or closes it. The output is a partial ordering of methods by their feasibility and credibility given the specific constraints.

**Gate 1: Is the assignment mechanism known or strongly credible?**

If yes (randomized experiment, lottery, sharp threshold), the design dominates. Use it. The lottery in the Oregon Health Insurance Experiment means IV/LATE is the primary estimator and selection-on-observables methods are secondary at best.

**Gate 2: Is panel data available with pre-treatment periods?**

If yes and if the treatment is staggered or absorbing, DiD and its extensions (Chapter 28, Callaway-Sant'Anna, Sun-Abraham) are feasible. The key question is not "do I have panel data?" but "is parallel trends defensible given the assignment process?" In the ACA Medicaid expansion, states did not expand randomly—expansion was correlated with prior health outcomes and political economy, creating PT concerns that must be addressed via pre-trend tests and synthetic control cross-validation.

**Gate 3: Is there a valid instrument?**

A valid instrument is rare. The Oregon lottery is a genuine instrument. Geographic distance to providers, policy eligibility cutoffs, and judge/examiner assignment are candidates elsewhere. If an instrument is available, ask: is the first stage strong ($F > 104.7$ by the Montiel-Olea–Pflueger robust threshold, not the old $F > 10$ heuristic)? Is the exclusion restriction defensible—and if violated, how badly does LATE distort?

**Gate 4: What is the sample size and clustering structure?**

Methods that require nonparametric estimation (regression discontinuity, kernel-based) need sufficient density near the point of interest. Methods that rely on asymptotics (GMM, 2SLS) require enough clusters for cluster-robust inference to be valid. The OHE has household-level randomization with a strata variable (`numhh_list`) that must be accounted for in standard errors.

**Gate 5: What is the target estimand?**

IV identifies LATE for compliers—a policy-relevant parameter only when the complier population is the population of interest. DiD under parallel trends identifies ATT for the treated. Selection-on-observables methods can target ATE or ATT depending on the weighting scheme. If the policy question is about the population effect of universal coverage, LATE from an IV based on lottery winners is the wrong target, and this matters for the combination weights in Section 49.5.

## 49.4 The Honesty-Credibility Frontier

**Definition 49.1 (Semiparametric Efficiency Bound).** Under a nonparametric model with assumptions $\mathcal{A}$, the semiparametric efficiency bound $V^*(\mathcal{A})$ is the minimum asymptotic variance achievable by any regular estimator of $\tau$ that is consistent under $\mathcal{A}$.

The efficient influence function (EIF) for ATE under unconfoundedness is:

$$\psi_i^{ATE} = \frac{D_i(Y_i - \mu_1(X_i))}{e(X_i)} - \frac{(1-D_i)(Y_i - \mu_0(X_i))}{1-e(X_i)} + \mu_1(X_i) - \mu_0(X_i) - \tau$$

where $e(X_i) = P(D_i = 1 \mid X_i)$ and $\mu_d(X_i) = E[Y_i \mid D_i = d, X_i]$. The AIPW estimator achieves $V^* = E[(\psi_i^{ATE})^2]$.

The critical insight is that $V^*(\mathcal{A})$ is a common floor: under the correct assumptions, all sufficiently flexible estimators converge to the same variance bound. Methods differ only in which assumptions they require to reach that floor.

**Theorem 49.1 (Assumption Violation Bias).** Let $\hat{\tau}$ be an estimator identified under assumption $\mathcal{A}_0$, and suppose $\mathcal{A}_0$ fails with violation magnitude $\delta$. Then:

$$\text{Bias}(\hat{\tau}) = B(\delta, \mathcal{A}_0, \hat{\tau}) + o(\delta)$$

where the leading bias term $B$ is method-specific. For the main methods:

*Unconfoundedness violation:* If there exists omitted $U_i$ with $E[Y_i(0) \mid D_i, X_i] = \alpha U_i + g(X_i)$ and $\text{Cov}(D_i, U_i \mid X_i) = \delta$, then $\text{Bias}(\hat{\tau}_{OLS}) \approx \alpha \delta / \text{Var}(D_i \mid X_i)$.

*Parallel trends violation:* If treated and control have trending divergence $E[Y_{it}(0) \mid D_i=1] - E[Y_{it}(0) \mid D_i=0] = \gamma t$, then $\text{Bias}(\hat{\tau}_{DiD}) \approx \gamma$.

*Exclusion restriction violation:* If $Z_i$ has direct effect $\rho$ on $Y_i$, then:

$$\text{Bias}(\hat{\tau}_{IV}) = \frac{\rho}{\pi}$$

where $\pi = E[D_i(1) - D_i(0)]$ is the first-stage compliance rate. The bias amplification factor $1/\pi$ makes IV estimates highly sensitive to exclusion violations when compliance is low—a critical consideration when the first stage is weak.

This gives us the key insight: **IV bias from exclusion violations is amplified by the inverse of the first-stage coefficient**. A 10% exclusion violation with 20% compliance is a 50% bias in the IV estimate. Selection-on-observables estimators do not have this amplification property.

## 49.5 The Weakest Link Principle

Identification chains are only as strong as their weakest assumption. In the OHE, the causal chain is:

$$Z_i \xrightarrow{\text{relevance}} D_i \xrightarrow{\text{exclusion + unconfounded}} Y_i$$

Each arrow requires at least one assumption. The credibility of the entire chain is bounded by the credibility of the least credible link. This principle has practical consequences:

1. **Adding instruments does not strengthen identification if the common weakest link persists.** Combining two instruments, each with marginal exclusion validity, does not create a valid combined instrument.

2. **Conditioning on colliders can introduce new weak links.** Controlling for a post-treatment mediator to address unconfoundedness creates a new collider bias link even if all original assumptions hold.

3. **Sensitivity to the weakest link should drive robustness checks.** The priority for sensitivity analysis (Chapter 47) is not the assumption you are most confident about but the assumption that, if violated even slightly, produces the largest bias given the first-order bias formula.

**Corollary 49.1 (Weakest Link Bound).** Let $\delta^*_k$ be the violation magnitude of assumption $k$ that would render $\hat{\tau}$ outside a confidence interval of width $2 \cdot 1.96 \cdot \text{SE}(\hat{\tau})$. The effective credibility of the estimator is $\min_k \delta^*_k$ across all identifying assumptions.

This bound provides a single scalar measure of identification robustness that can be compared across competing estimators. The estimator with the largest $\min_k \delta^*_k$ is the most robust, holding precision constant.

## 49.6 Combining Estimators: Minimum MSE Combination

When multiple identification strategies are available and defensible—as in the insurance question, where we have both the lottery and panel variation from state expansion—combining estimates can reduce MSE relative to using either alone.

**Definition 49.2 (Combined Estimator).** Given two consistent estimators $\hat{\tau}_1$ and $\hat{\tau}_2$ of the same estimand $\tau$, the minimum-MSE linear combination is:

$$\hat{\tau}_{comb} = \lambda \hat{\tau}_1 + (1-\lambda) \hat{\tau}_2$$

with optimal weight:

$$\lambda^* = \frac{\text{Var}(\hat{\tau}_2) - \text{Cov}(\hat{\tau}_1, \hat{\tau}_2)}{\text{Var}(\hat{\tau}_1) + \text{Var}(\hat{\tau}_2) - 2\text{Cov}(\hat{\tau}_1, \hat{\tau}_2)}$$

This is the generalized ridge combination familiar from forecast combination (Bates-Granger 1969). Several features are worth emphasizing.

First, $\lambda^*$ only minimizes variance when both estimators are consistent—i.e., when both identifying assumptions hold. If one estimator is biased, the optimal weight in the MSE sense depends on the bias-variance decomposition:

$$\text{MSE}(\hat{\tau}_{comb}) = \lambda^2 \text{Var}(\hat{\tau}_1) + (1-\lambda)^2 \text{Var}(\hat{\tau}_2) + 2\lambda(1-\lambda)\text{Cov}(\hat{\tau}_1, \hat{\tau}_2) + [(1-\lambda)B_2 + \lambda B_1]^2$$

where $B_k = \text{Bias}(\hat{\tau}_k)$. When both estimators are unbiased ($B_1 = B_2 = 0$), the last term vanishes and the variance-minimizing $\lambda^*$ is also MSE-minimizing.

Second, when $\hat{\tau}_1$ and $\hat{\tau}_2$ target different estimands—LATE from IV versus ATT from DiD—combination is only valid if the two target parameters are believed to be equal or if the combination is explicitly interpreted as a weighted average of two parameters. In the insurance application, the IV estimates LATE for lottery compliers; DiD estimates the ATT for states that expanded Medicaid. These populations overlap but are not identical, and the combined estimator should be interpreted accordingly.

Third, estimated weights $\hat{\lambda}^*$ introduce additional sampling variance. The plug-in estimator uses sample variances and covariance from bootstrap resampling. When standard errors are large, shrinking toward $\lambda = 0.5$ (equal weighting) often reduces realized MSE.

## 49.7 Applying the Framework to the Insurance Question

We now apply the decision framework to the health insurance question under three scenarios, illustrating how different data constraints change the method selection.

**Scenario A: Cross-Sectional Data Only**

Suppose we observe only a single cross-section of individuals, with no lottery information and no pre-treatment data. Gate 1 fails (no known assignment mechanism). Gate 2 fails (no panel). Gate 3 requires finding an instrument from the cross-section—perhaps distance to the nearest Medicaid office or presence of outreach workers. Gate 4 depends on sample size.

The available methods are: (a) OLS with rich covariate adjustment; (b) IPW/AIPW if propensity score is credible; (c) IV if a valid instrument exists in the cross-section. The credibility ranking is (c) $\succ$ (b) $\succ$ (a) in terms of robustness to selection, but (a) $\succ$ (b) $\succ$ (c) in terms of precision if all assumptions hold.

The recommendation is AIPW with thorough sensitivity analysis (Rosenbaum bounds, Cinelli-Hazlett partial $R^2$ decomposition) as the primary estimator, supplemented by IV if a candidate instrument can be defended.

**Scenario B: State-Level Panel with Staggered Adoption**

The ACA Medicaid expansion scenario. Gate 1 fails (state adoption was not random). Gate 2 opens (we have annual BRFSS data by state from 2010 onward, pre-expansion). Gate 3 is borderline—there is no clean lottery instrument at the state level, though border discontinuities have been used.

The primary method is staggered DiD using Callaway-Sant'Anna (2021) or the imputation estimator of Borusyak-Jaravel-Spiess (2024), which avoids contamination of already-treated units as controls. The critical assumption check is pre-trends: we need at least three pre-periods per adopting cohort to assess trend divergence. If pre-trends are present, we need either (a) a trend-adjustment (Rambachan-Roth honest DiD); (b) a synthetic control for plausibility; or (c) a conditional PT argument based on covariate adjustment.

**Scenario C: Lottery Instrument Available**

This is the OHE scenario. Gate 1 opens immediately. The lottery (`selected`) is the instrument, `ohp_all_ever_admin` is the endogenous treatment, and we use 2SLS with strata fixed effects for `numhh_list`.

The estimand is LATE for compliers—individuals who enrolled because they won the lottery and would not have enrolled otherwise. The first-stage $F$-statistic is well above 10 (approximately 800 in the published data), so weak instrument concerns are minimal. The primary threat is exclusion: does lottery selection affect health outcomes through channels other than insurance enrollment? Behavioral effects of winning a lottery (income effects, reduced stress from financial protection even before enrollment) are a credible exclusion concern and warrant sensitivity analysis under partial exclusion failure.

## Python: Method Selection Flowchart, Bias Simulation, and Combination Estimator

```python
"""
Chapter 49: Method Selection Framework
- Decision flowchart via graphviz
- Bias under assumption violation simulation
- Minimum-MSE combination estimator on OHE + simulated ACA
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# ── Optional graphviz ────────────────────────────────────────────────────────
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("graphviz not installed; skipping flowchart rendering.")

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Decision Tree Flowchart
# ═══════════════════════════════════════════════════════════════════════════

def build_decision_flowchart():
    """Build method-selection decision tree as a graphviz Digraph."""
    if not HAS_GRAPHVIZ:
        return None

    dot = graphviz.Digraph(
        "method_selection",
        comment="Causal Method Selection Framework",
        graph_attr={"rankdir": "TB", "fontsize": "11", "splines": "ortho"},
        node_attr={"shape": "box", "style": "rounded,filled", "fontsize": "10"},
    )

    # Decision nodes
    decisions = {
        "Q1": "Randomised / lottery\nassignment?",
        "Q2": "Panel data with\npre-treatment periods?",
        "Q3": "Valid instrument\navailable?",
        "Q4": "Sharp / fuzzy\nthreshold in forcing var?",
        "Q5": "Rich covariates,\nno unmeasured confounding?",
        "Q6": "Parallel trends\ndefensible?",
        "Q7": "Strong first stage\n(F > 100)?",
    }
    for k, label in decisions.items():
        dot.node(k, label, fillcolor="#AED6F1")

    # Terminal nodes
    terminals = {
        "T_RCT":    ("ITT / IV-LATE\n(randomization-based SE)",  "#A9DFBF"),
        "T_DiD":    ("Staggered DiD\n(CS-DiD / BJS imputation)",  "#A9DFBF"),
        "T_SC":     ("Synthetic Control\n+ SC-DiD",               "#A9DFBF"),
        "T_RDD":    ("RDD (local linear,\nCCT bandwidth)",        "#A9DFBF"),
        "T_IV":     ("2SLS / LIML\n+ sensitivity to excl.",      "#A9DFBF"),
        "T_AIPW":   ("AIPW / DR-learner\n+ Rosenbaum bounds",    "#A9DFBF"),
        "T_PARTIAL":("Partial identification\n+ sensitivity",    "#F9E79F"),
        "T_COMBINE":("Combination estimator\n(min-MSE weights)", "#F0B27A"),
    }
    for k, (label, color) in terminals.items():
        dot.node(k, label, fillcolor=color, shape="box")

    # Edges
    edges = [
        ("Q1", "T_RCT",    "Yes"),
        ("Q1", "Q2",       "No"),
        ("Q2", "Q6",       "Yes"),
        ("Q2", "Q3",       "No"),
        ("Q6", "T_DiD",    "Yes"),
        ("Q6", "T_SC",     "No / uncertain"),
        ("Q3", "Q7",       "Yes"),
        ("Q3", "Q4",       "No"),
        ("Q7", "T_IV",     "Yes"),
        ("Q7", "T_AIPW",   "No (weak)"),
        ("Q4", "T_RDD",    "Yes"),
        ("Q4", "Q5",       "No"),
        ("Q5", "T_AIPW",   "Yes"),
        ("Q5", "T_PARTIAL","No"),
    ]
    for src, dst, label in edges:
        dot.edge(src, dst, label=label)

    # Combination overlay
    dot.edge("T_DiD", "T_COMBINE", style="dashed", color="gray",
             label="+ IV available")
    dot.edge("T_IV",  "T_COMBINE", style="dashed", color="gray")

    dot.render("ch49_method_flowchart", format="pdf", cleanup=True)
    print("Flowchart saved to ch49_method_flowchart.pdf")
    return dot


flowchart = build_decision_flowchart()


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Assumption Violation Bias Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_bias_under_violations(
    n: int = 5_000,
    n_sims: int = 400,
    delta_grid: np.ndarray = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each violation magnitude delta, simulate bias of:
      - OLS (omitted variable U)
      - DiD (parallel trends violation)
      - IV  (exclusion restriction violation)
    True ATT = 0.3 in all DGPs.
    """
    rng = np.random.default_rng(seed)
    if delta_grid is None:
        delta_grid = np.linspace(0.0, 0.5, 21)

    TRUE_TAU = 0.3
    results = []

    for delta in delta_grid:
        ols_biases, did_biases, iv_biases = [], [], []

        for _ in range(n_sims):
            # ── Shared latent structure ──────────────────────────────────
            X  = rng.normal(0, 1, n)
            U  = rng.normal(0, 1, n)           # unmeasured confounder
            Z  = (rng.normal(0, 1, n) > 0).astype(float)  # lottery IV

            # First stage: compliance ~ 50%
            pi = 0.5
            D_z1 = (rng.uniform(size=n) < 0.5 + 0.5).astype(float)  # always-taker
            D    = Z * (rng.uniform(size=n) < pi).astype(float)
            D   += (1 - Z) * (rng.uniform(size=n) < 0.0).astype(float)

            # ── OLS DGP: omitted variable bias ───────────────────────────
            # D correlated with U through selection
            D_obs = (0.5 * X + delta * U + rng.normal(0, 1, n) > 0).astype(float)
            Y_ols = TRUE_TAU * D_obs + 0.5 * X + 1.0 * U + rng.normal(0, 0.5, n)

            design = np.column_stack([np.ones(n), D_obs, X])
            coef   = np.linalg.lstsq(design, Y_ols, rcond=None)[0]
            ols_biases.append(coef[1] - TRUE_TAU)

            # ── DiD DGP: parallel trends violation ────────────────────────
            n_pre, n_post = 2, 2
            t_vals  = np.arange(n_pre + n_post)
            n_units = n // (n_pre + n_post)
            G  = (rng.uniform(size=n_units) < 0.5).astype(float)  # treated group
            trend_diff = delta  # diverging trend per period

            rows = []
            for unit in range(n_units):
                for t in t_vals:
                    post  = float(t >= n_pre)
                    trend = trend_diff * G[unit] * t
                    y     = (0.4 * G[unit]
                             + TRUE_TAU * G[unit] * post
                             + trend
                             + rng.normal(0, 0.3))
                    rows.append({"unit": unit, "t": t, "G": G[unit],
                                 "post": post, "Y": y})
            df_panel = pd.DataFrame(rows)
            df_post  = df_panel[df_panel["post"] == 1].groupby(["unit","G"])["Y"].mean().reset_index()
            df_pre   = df_panel[df_panel["post"] == 0].groupby(["unit","G"])["Y"].mean().reset_index()
            df_diff  = df_post.copy()
            df_diff["Y"] -= df_pre["Y"].values
            att_did = (df_diff[df_diff["G"]==1]["Y"].mean()
                       - df_diff[df_diff["G"]==0]["Y"].mean())
            did_biases.append(att_did - TRUE_TAU)

            # ── IV DGP: exclusion violation ───────────────────────────────
            # Z has direct effect rho = delta on Y
            rho  = delta
            eps  = rng.normal(0, 1, n)
            Y_iv = TRUE_TAU * D + rho * Z + 0.5 * X + eps

            # 2SLS by hand
            X_iv   = np.column_stack([np.ones(n), D, X])
            Z_iv   = np.column_stack([np.ones(n), Z, X])
            PZ     = Z_iv @ np.linalg.solve(Z_iv.T @ Z_iv, Z_iv.T)
            D_hat  = PZ @ X_iv[:, 1]
            X_2sls = X_iv.copy(); X_2sls[:, 1] = D_hat
            coef_iv = np.linalg.lstsq(X_2sls, Y_iv, rcond=None)[0]
            iv_biases.append(coef_iv[1] - TRUE_TAU)

        results.append({
            "delta":     delta,
            "OLS_bias":  np.mean(ols_biases),
            "DiD_bias":  np.mean(did_biases),
            "IV_bias":   np.mean(iv_biases),
            "OLS_rmse":  np.sqrt(np.mean(np.array(ols_biases)**2)),
            "DiD_rmse":  np.sqrt(np.mean(np.array(did_biases)**2)),
            "IV_rmse":   np.sqrt(np.mean(np.array(iv_biases)**2)),
        })

    return pd.DataFrame(results)


print("\nSimulating bias under assumption violations...")
bias_df = simulate_bias_under_violations(n=3_000, n_sims=200)

print("\n── Bias under assumption violations (selected delta values) ──")
summary_cols = ["delta", "OLS_bias", "DiD_bias", "IV_bias"]
print(bias_df[summary_cols].iloc[::5].round(4).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: Minimum-MSE Combination Estimator on OHE
# ═══════════════════════════════════════════════════════════════════════════

def load_ohe_data(path: str = "data/oregon_data.csv") -> pd.DataFrame | None:
    """Load Oregon Health Insurance Experiment data if available."""
    try:
        df = pd.read_csv(path)
        needed = ["selected", "ohp_all_ever_admin", "doc_any_12m", "numhh_list"]
        if all(c in df.columns for c in needed):
            return df.dropna(subset=needed)
    except FileNotFoundError:
        pass
    return None


def simulate_ohe_like(n: int = 15_000, seed: int = 7) -> pd.DataFrame:
    """
    Simulate data matching OHE structure for reproducible demonstration.
    selected     : lottery instrument (Z)
    ohp_all_ever : treatment (D), compliance ~29% from intention-to-treat
    doc_any_12m  : had any doctor visit (Y1)
    catastrophic : catastrophic out-of-pocket expense (Y2)
    numhh_list   : household size strata (1, 2, 3+)
    """
    rng = np.random.default_rng(seed)
    numhh = rng.choice([1, 2, 3], size=n, p=[0.6, 0.25, 0.15])
    selected = (rng.uniform(size=n) < 0.30 + 0.05 * (numhh == 1)).astype(int)

    # Compliance: selected -> enrolled with prob 0.29
    enrolled = np.zeros(n, dtype=int)
    enrolled[selected == 1] = (rng.uniform(size=selected.sum()) < 0.29).astype(int)

    U = rng.normal(0, 1, n)  # unobserved health baseline
    doc     = (0.45 + 0.12 * enrolled - 0.05 * U + rng.normal(0, 0.3, n) > 0.3).astype(int)
    catast  = (0.15 - 0.04 * enrolled + 0.06 * U + rng.normal(0, 0.2, n) > 0.2).astype(int)
    income  = rng.normal(20_000, 8_000, n)
    age     = rng.randint(19, 65, n)

    return pd.DataFrame({
        "selected": selected,
        "ohp_all_ever_admin": enrolled,
        "doc_any_12m": doc,
        "catastrophic_exp_inp": catast,
        "numhh_list": numhh,
        "income": income,
        "age": age,
        "U": U,
    })


def tsls_estimate(df: pd.DataFrame, Y_col: str, D_col: str,
                  Z_col: str, strata_col: str, n_boot: int = 500,
                  seed: int = 0) -> dict:
    """
    2SLS with strata fixed effects, bootstrap SE.
    Returns dict: {estimate, se, ci_lo, ci_hi, F_stat}.
    """
    rng = np.random.default_rng(seed)

    # Demean within strata
    def demean(s: pd.Series, groups) -> np.ndarray:
        out = s.values.astype(float).copy()
        for g in groups.unique():
            mask = groups == g
            out[mask] -= out[mask].mean()
        return out

    Y  = demean(df[Y_col],   df[strata_col])
    D  = demean(df[D_col],   df[strata_col])
    Z  = demean(df[Z_col],   df[strata_col])
    n  = len(Y)

    # First stage
    fs_cov = np.cov(Z, D)
    pi_hat = np.cov(Z, D)[0, 1] / np.var(Z)
    D_hat  = pi_hat * Z
    resid_fs = D - D_hat
    F_stat = (pi_hat**2 * np.var(Z)) / (np.var(resid_fs) / (n - 2))

    # Second stage
    tau_iv = np.cov(Z, Y)[0, 1] / np.cov(Z, D)[0, 1]

    # Bootstrap
    boots = []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx   = rng.choice(idx_all, size=n, replace=True)
        pi_b  = np.cov(Z[idx], D[idx])[0, 1] / np.var(Z[idx])
        tau_b = np.cov(Z[idx], Y[idx])[0, 1] / np.cov(Z[idx], D[idx])[0, 1]
        boots.append(tau_b)

    se    = np.std(boots)
    ci_lo = tau_iv - 1.96 * se
    ci_hi = tau_iv + 1.96 * se

    return {"estimate": tau_iv, "se": se, "ci_lo": ci_lo,
            "ci_hi": ci_hi, "F_stat": F_stat, "boots": np.array(boots)}


def ols_aipw_estimate(df: pd.DataFrame, Y_col: str, D_col: str,
                      covariates: list, n_boot: int = 500, seed: int = 1) -> dict:
    """
    AIPW (doubly robust) estimate with logistic propensity score,
    linear outcome models, bootstrap SE.
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression

    rng    = np.random.default_rng(seed)
    Y      = df[Y_col].values.astype(float)
    D      = df[D_col].values.astype(float)
    Xmat   = df[covariates].values.astype(float)
    n      = len(Y)

    def _aipw(Y, D, Xmat):
        ps_model = LogisticRegression(max_iter=500).fit(Xmat, D)
        e_hat    = np.clip(ps_model.predict_proba(Xmat)[:, 1], 0.05, 0.95)
        mu1 = LinearRegression().fit(Xmat[D==1], Y[D==1]).predict(Xmat)
        mu0 = LinearRegression().fit(Xmat[D==0], Y[D==0]).predict(Xmat)
        eif = (D*(Y - mu1)/e_hat - (1-D)*(Y - mu0)/(1-e_hat) + mu1 - mu0)
        return eif.mean()

    tau_dr = _aipw(Y, D, Xmat)
    boots  = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(_aipw(Y[idx], D[idx], Xmat[idx]))

    se    = np.std(boots)
    return {"estimate": tau_dr, "se": se,
            "ci_lo": tau_dr - 1.96*se,
            "ci_hi": tau_dr + 1.96*se,
            "boots": np.array(boots)}


def optimal_combination(est1: dict, est2: dict) -> dict:
    """
    Minimum-MSE linear combination using bootstrap covariance.
    Assumes both estimators target the same estimand.
    """
    b1 = est1["boots"]
    b2 = est2["boots"]
    n  = min(len(b1), len(b2))
    b1, b2 = b1[:n], b2[:n]

    var1   = np.var(b1)
    var2   = np.var(b2)
    cov12  = np.cov(b1, b2)[0, 1]

    denom  = var1 + var2 - 2 * cov12
    lam    = (var2 - cov12) / denom if abs(denom) > 1e-12 else 0.5
    lam    = np.clip(lam, 0.0, 1.0)   # constrain to convex combination

    tau_c  = lam * est1["estimate"] + (1 - lam) * est2["estimate"]
    # Variance of combination
    var_c  = (lam**2 * var1 + (1-lam)**2 * var2 + 2*lam*(1-lam)*cov12)
    se_c   = np.sqrt(var_c)

    return {
        "lambda": lam,
        "estimate": tau_c,
        "se": se_c,
        "ci_lo": tau_c - 1.96 * se_c,
        "ci_hi": tau_c + 1.96 * se_c,
        "var_reduction_pct": 100 * (1 - var_c / min(var1, var2)),
    }


# ── Load or simulate OHE data ────────────────────────────────────────────
df_ohe = load_ohe_data()
if df_ohe is None:
    print("\nOHE data not found — using simulated OHE-like data.")
    df_ohe = simulate_ohe_like(n=15_000)

print(f"\nSample size: {len(df_ohe):,}")
print(f"Lottery selection rate: {df_ohe['selected'].mean():.3f}")
print(f"Enrollment rate (overall): {df_ohe['ohp_all_ever_admin'].mean():.3f}")
print(f"Enrollment rate (selected=1): "
      f"{df_ohe.loc[df_ohe['selected']==1,'ohp_all_ever_admin'].mean():.3f}")

# ── IV estimate (2SLS): effect on doctor visits ──────────────────────────
print("\n── IV (2SLS) estimate: doc_any_12m ──")
iv_est = tsls_estimate(
    df_ohe, Y_col="doc_any_12m", D_col="ohp_all_ever_admin",
    Z_col="selected", strata_col="numhh_list", n_boot=500
)
print(f"  LATE (IV):  {iv_est['estimate']:.4f}  "
      f"SE: {iv_est['se']:.4f}  "
      f"95% CI: [{iv_est['ci_lo']:.4f}, {iv_est['ci_hi']:.4f}]")
print(f"  First-stage F: {iv_est['F_stat']:.1f}")

# ── AIPW estimate ────────────────────────────────────────────────────────
print("\n── AIPW (doubly robust) estimate: doc_any_12m ──")
covs = ["numhh_list", "income", "age"]
aipw_est = ols_aipw_estimate(
    df_ohe, Y_col="doc_any_12m", D_col="ohp_all_ever_admin",
    covariates=covs, n_boot=500
)
print(f"  ATE (AIPW): {aipw_est['estimate']:.4f}  "
      f"SE: {aipw_est['se']:.4f}  "
      f"95% CI: [{aipw_est['ci_lo']:.4f}, {aipw_est['ci_hi']:.4f}]")

# ── Combination estimator ─────────────────────────────────────────────────
print("\n── Minimum-MSE combination ──")
comb = optimal_combination(iv_est, aipw_est)
print(f"  Optimal lambda (weight on IV): {comb['lambda']:.4f}")
print(f"  Combined estimate: {comb['estimate']:.4f}  "
      f"SE: {comb['se']:.4f}  "
      f"95% CI: [{comb['ci_lo']:.4f}, {comb['ci_hi']:.4f}]")
print(f"  Variance reduction vs. best single: {comb['var_reduction_pct']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: Assumption Severity Table
# ═══════════════════════════════════════════════════════════════════════════

def print_assumption_table():
    rows = [
        ("RCT / Lottery",         "SUTVA, no attrition",    "✓",  "★★★★★"),
        ("IV (2SLS)",             "Exclusion, monotonicity", "✗",  "★★★★☆"),
        ("RDD",                   "Continuity at cutoff",    "~",  "★★★★☆"),
        ("DiD / Staggered DiD",   "Parallel trends",         "~",  "★★★☆☆"),
        ("Synthetic Control",     "Convex hull / span",      "~",  "★★★☆☆"),
        ("AIPW / IPW",            "Unconfoundedness",        "✗",  "★★☆☆☆"),
        ("OLS (adjusted)",        "No omitted variables",    "✗",  "★☆☆☆☆"),
    ]
    header = f"{'Method':<26}{'Primary Assumption':<30}{'Partially Testable':<22}{'Credibility'}"
    print("\n── Table 49.1: Assumption severity ranking ──")
    print(header)
    print("─" * len(header))
    for method, assumption, testable, stars in rows:
        print(f"{method:<26}{assumption:<30}{testable:<22}{stars}")

print_assumption_table()


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: Honesty–Credibility Frontier
# ═══════════════════════════════════════════════════════════════════════════

def honesty_credibility_frontier(
    delta_grid: np.ndarray = None,
    var_floor: float = 0.001,
    first_stage_pi: float = 0.29,
) -> pd.DataFrame:
    """
    Compute (variance, bias^2) frontier for each method as violation delta
    increases. Variance decreases with stronger assumptions; bias^2 increases.
    """
    if delta_grid is None:
        delta_grid = np.linspace(0.0, 0.5, 50)

    records = []
    for delta in delta_grid:
        # OLS: low variance (strong parametric assumptions), amplified bias
        records.append({
            "method": "OLS",
            "delta": delta,
            "variance": var_floor * 1.0,
            "bias_sq": (1.5 * delta) ** 2,
        })
        # AIPW: moderate variance, moderate bias
        records.append({
            "method": "AIPW",
            "delta": delta,
            "variance": var_floor * 2.5,
            "bias_sq": (0.8 * delta) ** 2,
        })
        # DiD: moderate variance, linear bias from PT violation
        records.append({
            "method": "DiD",
            "delta": delta,
            "variance": var_floor * 3.0,
            "bias_sq": delta ** 2,
        })
        # IV: highest variance, amplified bias via 1/pi
        records.append({
            "method": "IV",
            "delta": delta,
            "variance": var_floor * 4.0,
            "bias_sq": (delta / first_stage_pi) ** 2,
        })
        # Partial ID: no point identified, bounded MSE
        records.append({
            "method": "Partial ID",
            "delta": delta,
            "variance": var_floor * 6.0,
            "bias_sq": 0.0,   # no assumption => no bias from violation
        })

    df = pd.DataFrame(records)
    df["MSE"] = df["variance"] + df["bias_sq"]
    return df


frontier_df = honesty_credibility_frontier()

print("\n── MSE by method at delta = 0.0 (no violation) ──")
d0 = frontier_df[frontier_df["delta"] == 0.0][["method", "variance", "bias_sq", "MSE"]]
print(d0.to_string(index=False))

print("\n── MSE by method at delta = 0.20 (moderate violation) ──")
d2 = frontier_df[(frontier_df["delta"] - 0.20).abs() < 0.005][
    ["method", "variance", "bias_sq", "MSE"]
]
print(d2.to_string(index=False))

# Crossover: at what delta does IV become worse than DiD in MSE?
iv_mse  = frontier_df[frontier_df["method"] == "IV"].set_index("delta")["MSE"]
did_mse = frontier_df[frontier_df["method"] == "DiD"].set_index("delta")["MSE"]
crossover = (iv_mse - did_mse).abs().idxmin()
print(f"\nIV vs DiD MSE crossover at delta ≈ {crossover:.3f}")
print("(Above this exclusion violation, DiD has lower MSE than IV "
      "if parallel trends holds perfectly.)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6: Scenario summary table
# ═══════════════════════════════════════════════════════════════════════════

def print_scenario_summary():
    scenarios = [
        ("A: Cross-section only",
         "No panel, no lottery",
         "AIPW + Rosenbaum bounds",
         "ATE (unconfounded)",
         "Unconfoundedness"),
        ("B: Staggered panel (ACA)",
         "State panel, no lottery",
         "CS-DiD / BJS imputation",
         "ATT (treated cohorts)",
         "Parallel trends"),
        ("C: Lottery available (OHE)",
         "Lottery + cross-section",
         "2SLS + AIPW (combined)",
         "LATE (compliers)",
         "Exclusion restriction"),
    ]
    print("\n── Table 49.2: Method selection by scenario (insurance question) ──")
    header = (f"{'Scenario':<28}{'Data Constraints':<28}"
              f"{'Primary Method':<24}{'Estimand':<26}{'Key Assumption'}")
    print(header)
    print("─" * (len(header) + 5))
    for row in scenarios:
        print(f"{row[0]:<28}{row[1]:<28}{row[2]:<24}{row[3]:<26}{row[4]}")


print_scenario_summary()

print("\nChapter 49 analysis complete.")
```

## Summary

- Method selection is a constrained optimization over MSE, not a lookup table. The five-gate decision framework—(1) known assignment, (2) panel, (3) instrument, (4) threshold, (5) confounding richness—provides a sequenced path that prunes infeasible methods before comparing feasible ones.

- All correct estimators share the semiparametric efficiency floor $V^*(\mathcal{A})$; methods differ only in which assumptions they need to reach that floor, not in their limiting precision given those assumptions.

- Assumption violation bias is method-specific and, critically, not monotone in precision. IV has amplified bias under exclusion violations by the factor $1/\pi$ (inverse compliance rate), making it more vulnerable than OLS or DiD to small exclusion failures when compliance is low.

- The weakest link principle bounds identification credibility by the assumption with the smallest breakdown value $\delta^*_k$. Sensitivity analysis should prioritize the weakest link, not the strongest.

- The minimum-MSE combination estimator $\hat{\tau}_{comb} = \lambda^* \hat{\tau}_{IV} + (1-\lambda^*) \hat{\tau}_{AIPW}$ achieves variance reduction when the two estimators have positive covariance and both are consistent, with optimal weight derived from bootstrap variance-covariance of the bootstrap distributions.

- Applied to the Oregon Health Insurance Experiment, the lottery instrument has a first-stage $F$-statistic far exceeding robust thresholds, but the exclusion restriction (lottery selection → outcomes only through enrollment) warrants sensitivity analysis because winning the lottery may reduce financial anxiety independently of coverage.

- The honesty-credibility frontier formalizes the bias-variance tradeoff across methods: at zero violation, OLS has lowest MSE; at moderate violations, AIPW and DiD dominate IV due to bias amplification; at large violations, partial identification, while imprecise, has the best MSE guarantee.

## Further Reading

1. **Imbens, G. W. (2010)**. "Better LATE than nothing: Some comments on Deaton (2009) and Heckman and Urzua (2009)." *Journal of Economic Literature*, 48(2), 399–423. The definitive statement of when LATE is and is not the right estimand, with direct relevance to the combination problem when IV and DiD target different populations.

2. **Callaway, B., & Sant'Anna, P. H. C. (2021)**. "Difference-in-differences with multiple time periods." *Journal of Econometrics*, 225(2), 200–230. Derives the assumption-specific efficiency bound for staggered DiD and shows when existing estimators achieve it; essential background for Scenario B.

3. **Cinelli, C., & Hazlett, C. (2020)**. "Making sense of sensitivity: Extending omitted variable bias." *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67. Provides the partial $R^2$ sensitivity framework that operationalizes the $\delta^*_k$ breakdown values needed for the weakest link bound.

4. **Rambachan, A., & Roth, J. (2023)**. "A more credible approach to parallel trends." *Review of Economic Studies*, 90(5), 2555–2591. Formalizes the honesty-credibility tradeoff for DiD by introducing restrictions on the magnitude of pre-trend extrapolation, directly connecting to Section 49.4's frontier.

5. **Bates, J. M., & Granger, C. W. J. (1969)**. "The combination of forecasts." *Operational Research Quarterly*, 20(4), 451–468. The original derivation of the minimum-MSE linear combination weights; the causal combination estimator in Section 49.6 is this result applied to consistent causal estimators.

6. **Tamer, E. (2010)**. "Partial identification in econometrics." *Annual Review of Economics*, 2(1), 167–195. Reviews the partial identification literature that provides the baseline comparison—what can be learned with no assumptions beyond sign restrictions—necessary to calibrate how much identifying assumptions are buying in practice.
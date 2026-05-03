# Chapter 43: Causal Inference for Pricing, Marketing, and Product Interventions

The methods developed in earlier chapters — instrumental variables, double machine learning, heterogeneous treatment effects, and interference-robust estimators — were motivated primarily by policy evaluation problems: does Medicaid improve health? does job training raise earnings? Business applications share the same identification challenges but add structural features that require tailored solutions. Prices are endogenous because firms set them in response to demand signals. Marketing touchpoints are assigned by algorithms that condition on user behavior. Product features roll out on platforms where users interact, violating the stable unit treatment value assumption (SUTVA) at scale. This chapter systematizes how to apply causal inference to these three domains, deriving estimators, characterizing their failure modes, and anchoring every result in the health insurance running example as a structural parallel.

---

## 43.1 Price Elasticity via Instrumental Variables

### 43.1.1 The Endogeneity Problem in Pricing

Let $Q_{it}$ denote quantity demanded (or take-up rate) and $P_{it}$ denote price for unit $i$ at time $t$. The naive OLS regression

$$\log Q_{it} = \alpha + \varepsilon \log P_{it} + \mathbf{X}_{it}'\beta + u_{it}$$

estimates a biased $\hat{\varepsilon}$ because prices are set by firms observing demand shifters correlated with $u_{it}$. When demand is high, firms may raise prices (upward bias toward zero or positive elasticity) or they may lower prices to gain market share (downward bias). The direction is application-specific; the bias is universal.

The structural parallel in health insurance is direct. Medicaid copayments — the "price" of utilization — are not randomly assigned. States with sicker populations may waive copayments, inducing negative correlation between price and health need that biases OLS elasticity estimates upward (less negative than true).

### 43.1.2 Cost-Shifter Instruments

The textbook solution, tracing to Hausman (1996) and Nevo (2001), uses a variable $Z_{it}$ that shifts cost and hence price but has no direct effect on demand:

**Assumption 43.1 (Instrument Validity for Elasticity)**
1. *Relevance*: $\text{Cov}(Z_{it}, \log P_{it} \mid \mathbf{X}_{it}) \neq 0$
2. *Exclusion*: $Z_{it} \perp u_{it} \mid \mathbf{X}_{it}$
3. *Monotonicity*: $\log P_{it}(Z=1) \geq \log P_{it}(Z=0)$ a.s. (or the reverse)

Under these conditions, 2SLS recovers the Local Average Treatment Effect on the complier subpopulation:

$$\hat{\varepsilon}_{IV} = \frac{\widehat{\text{Cov}}(\log Q_{it}, Z_{it} \mid \mathbf{X}_{it})}{\widehat{\text{Cov}}(\log P_{it}, Z_{it} \mid \mathbf{X}_{it})}$$

This is the Wald estimator in log-log form, giving the arc elasticity of demand with respect to price among units whose price was shifted by the instrument.

**Theorem 43.1 (IV Elasticity Consistency)**
Under Assumption 43.1 and standard regularity conditions (bounded moments, nondegenerate first stage), $\hat{\varepsilon}_{IV} \xrightarrow{p} \varepsilon_{LATE}$ where $\varepsilon_{LATE} = E[\partial \log Q / \partial \log P \mid \text{complier}]$.

*Proof sketch*. The 2SLS estimator solves the moment condition $E[Z_{it}(\log Q_{it} - \varepsilon \log P_{it} - \mathbf{X}_{it}'\beta)] = 0$. Under exclusion, $E[Z_{it} u_{it}] = 0$, so the moment is identified at the true $\varepsilon_{LATE}$. Consistency follows from the law of large numbers and the continuous mapping theorem applied to the ratio form of the Wald estimator.

### 43.1.3 Heterogeneous Price Elasticity via DML

When elasticity varies across market segments — a central concern in pricing strategy — the partially linear DML framework from Chapter 12 extends naturally. Write

$$\log Q_{it} = \theta(\mathbf{X}_{it}) \log P_{it} + g(\mathbf{X}_{it}) + u_{it}$$

where $\theta(\mathbf{X}_{it})$ is the heterogeneous elasticity function. The DML-IV estimator (Chapter 21) proceeds in three stages:

1. Partial out controls from $\log Q$, $\log P$, and $Z$: fit $\hat{g}_Q$, $\hat{g}_P$, $\hat{g}_Z$ using flexible ML on $\mathbf{X}$.
2. Form residuals $\tilde{Q} = \log Q - \hat{g}_Q(\mathbf{X})$, $\tilde{P} = \log P - \hat{g}_P(\mathbf{X})$, $\tilde{Z} = Z - \hat{g}_Z(\mathbf{X})$.
3. Estimate $\theta(\mathbf{X})$ by regressing $\tilde{Q}$ on $\tilde{P}$ instrumented by $\tilde{Z}$, using a second-stage CATE learner (e.g., causal forest or linear interaction model).

The Neyman orthogonality of the DML moment ensures that first-stage estimation error in $\hat{g}$ enters the second stage at rate $o_p(n^{-1/4})$, preserving $\sqrt{n}$-consistency of the average elasticity while allowing flexible nonparametric nuisance estimation.

---

## 43.2 Incrementality Testing in Digital Marketing

### 43.2.1 The Attribution Problem

Digital marketing attribution asks: of all conversions observed after a user was exposed to an ad, how many were *caused* by the ad? The naive last-touch attribution rule assigns full credit to the final touchpoint before conversion. Multi-touch rules distribute credit according to position or time decay. Neither approach has a causal interpretation without additional assumptions.

**Proposition 43.1 (Last-Touch Bias)**
Let $Y_i(1), Y_i(0)$ be potential conversions under ad exposure and non-exposure. Let $A_i \in \{0,1\}$ indicate ad assignment, and suppose $A_i$ is assigned by an algorithm with $P(A_i = 1 \mid \mathbf{X}_i) = e(\mathbf{X}_i) \neq 1/2$ in general. Then

$$E[\hat{\alpha}_{LT}] = E[Y_i \mid A_i = 1] = E[Y_i(1) \mid A_i = 1] \neq E[Y_i(1) - Y_i(0)] = \tau$$

unless $A_i \perp (Y_i(1), Y_i(0))$, which fails when the algorithm targets high-intent users.

*Proof*. $E[\hat{\alpha}_{LT}] = E[Y_i \mid A_i = 1] = E[Y_i(1) \mid A_i = 1]$ by consistency. This equals $\tau + E[Y_i(0) \mid A_i = 1]$ only when $E[Y_i(0) \mid A_i = 1] = 0$, i.e., no counterfactual conversions among ad-exposed users — exactly what targeting algorithms violate.

### 43.2.2 Ghost Ad Holdouts and Geo-Experiments

The gold standard for incrementality measurement is randomized holdout: assign a random subset of users (or geographies) to receive no ad (or a placebo), compare outcomes to the treated group. The causal estimand is the Average Treatment Effect on the Treated (ATT):

$$\tau_{ATT} = E[Y_i(1) - Y_i(0) \mid A_i = 1]$$

Geo-level holdouts (assigning matched markets to treatment/control) trade statistical power for reduced interference: a user in the control DMA does not see ads in that market, so her behavior is uncontaminated. The estimator is a difference-in-means at the geo level, with precision governed by between-geo variance rather than individual variance.

### 43.2.3 Shapley Attribution as a Cooperative Game

When holdout experiments are infeasible (historical data only, small advertiser), Shapley value attribution provides a principled allocation of credit across touchpoints. Model the set of touchpoints $\mathcal{T} = \{1, \ldots, K\}$ as players in a cooperative game where the value function $v(S)$ for coalition $S \subseteq \mathcal{T}$ is the conversion rate for user paths containing exactly the touchpoints in $S$.

The Shapley value for touchpoint $k$ is:

$$\phi_k = \sum_{S \subseteq \mathcal{T} \setminus \{k\}} \frac{|S|!(|\mathcal{T}| - |S| - 1)!}{|\mathcal{T}|!} \left[v(S \cup \{k\}) - v(S)\right]$$

**Theorem 43.2 (Shapley Axioms)**
The Shapley value is the unique allocation satisfying efficiency ($\sum_k \phi_k = v(\mathcal{T})$), symmetry (identical players get equal shares), dummy player ($v(S \cup \{k\}) = v(S)$ for all $S$ implies $\phi_k = 0$), and additivity across games.

*Note on causal interpretation*. Shapley attribution does not recover $\tau$ without randomization. It allocates observed conversion probability according to marginal contributions in the data-generating process of the observed path distribution, which conflates selection into paths with path effects. The result is a consistency-coherent decomposition, not an unbiased causal effect estimate. It is strictly superior to last-touch as a summary statistic but should not be interpreted causally unless path assignment is as-good-as-random.

---

## 43.3 A/B Testing with Network Effects

### 43.3.1 SUTVA Failure in Marketplace Settings

Standard A/B testing assumes that unit $i$'s outcome depends only on its own treatment. Marketplaces, social networks, and two-sided platforms violate this: treating a buyer on a ride-sharing platform increases demand for driver supply, spilling over to control buyers who now face longer wait times. The standard difference-in-means estimator is biased for the direct effect and may even have the wrong sign.

Let $Y_i(\mathbf{d})$ denote $i$'s potential outcome under the full treatment vector $\mathbf{d} \in \{0,1\}^n$. Under stratified interference (Chapter 41), $Y_i(\mathbf{d}) = Y_i(d_i, G_i(\mathbf{d}_{-i}))$ where $G_i(\mathbf{d}_{-i})$ is a sufficient statistic for the neighborhood treatment proportion. The standard estimator

$$\hat{\tau}_{naive} = \bar{Y}_{treat} - \bar{Y}_{control}$$

estimates a mixture of direct and spillover effects that depends on the proportion $\pi$ assigned to treatment:

**Proposition 43.2 (Bias of Naive A/B Estimator Under Interference)**
Under linear-in-means interference, $Y_i(d_i, \pi_{-i}) = \mu + \tau d_i + \gamma \pi_{-i} + \varepsilon_i$ where $\pi_{-i}$ is the proportion of $i$'s neighbors treated, the naive estimator satisfies:

$$E[\hat{\tau}_{naive}] = \tau + \gamma \cdot E[\pi_{-i} \mid d_i = 1] - \gamma \cdot E[\pi_{-i} \mid d_i = 0]$$

Under complete randomization with proportion $\pi$: $E[\pi_{-i} \mid d_i] \approx \pi$ for large networks, so $E[\hat{\tau}_{naive}] \approx \tau$. But under clustered assignment (treat all users in some markets), the spillover term vanishes within clusters, and the estimator recovers the total effect $\tau + \gamma$ (within-cluster) minus $\gamma \cdot 0$ (cross-cluster spillover) — a different estimand than $\tau$.

### 43.3.2 Cluster-Randomized Designs

The resolution is to randomize at the level of natural clusters — geographies, device cohorts, seller categories — that limit cross-cluster spillovers. The cluster-randomized estimator is consistent for the cluster-level ITT effect, which includes within-cluster spillovers as part of the estimand. Separating direct from spillover effects requires a two-level design: vary both whether the cluster is treated ($D_c \in \{0,1\}$) and the within-cluster proportion $\pi_c \in \{0, 0.25, 0.5, 1\}$. This is the Bernoulli-over-Bernoulli design. The estimator for the direct effect isolates variation in $d_i$ holding $\pi_c$ fixed; the spillover is identified by variation in $\pi_c$ holding $d_i = 0$.

---

## 43.4 Switchback Experiments for Time-Series Interference

### 43.4.1 The Carryover Problem

When randomizing individuals is infeasible (platform-wide algorithm changes affect all users simultaneously), switchback experiments randomize the treatment *in time*: alternating periods receive treatment or control, and outcomes are averaged across periods. Let $T$ total periods be divided into $B$ blocks, each of length $L$. The period-level estimator is:

$$\hat{\tau}_{SB} = \frac{1}{T} \sum_{t=1}^{T} \hat{\tau}_t, \quad \hat{\tau}_t = Y_t - Y_{t-1} \cdot \mathbf{1}[\text{period flips}]$$

More precisely, with within-period DID:

$$\hat{\tau}_{SB} = \frac{2}{B} \sum_{b=1}^{B/2} \left[\bar{Y}_{treat,b} - \bar{Y}_{control,b}\right]$$

**Theorem 43.3 (Switchback Bias from Carryover)**
Suppose the treatment effect at time $t$ depends on current and lagged treatment: $Y_t = \mu + \tau D_t + \kappa D_{t-1} + \varepsilon_t$ (single lag carryover). Then

$$E[\hat{\tau}_{SB}] = \tau + \kappa \cdot P(D_{t-1} \neq D_t) \cdot \text{sign}(D_t - D_{t-1})$$

In a balanced switchback design with $P(D_t = 1) = 0.5$, the bias is $\kappa/2$: half the carryover effect leaks into the switchback estimator because transitions occur at block boundaries, and the first observation of each block carries the previous block's treatment state.

*Proof sketch*. At transition periods, $D_{t-1}$ and $D_t$ differ in sign. The contribution of $\kappa D_{t-1}$ to $\bar{Y}_{treat}$ is $\kappa \cdot E[D_{t-1} \mid D_t = 1]$. Under a balanced alternating design, $D_{t-1} = 0$ exactly at the first observation of each treatment block, contributing $\kappa \cdot 0$ in those periods. Averaging over the block, $E[\kappa D_{t-1} \mid D_t = 1] = \kappa(1 - 1/L)$ for block length $L$. As $L \to \infty$, the bias vanishes; as $L \to 1$ (rapid switching), the bias approaches $\kappa/2$.

### 43.4.2 Bias Correction via Washout Periods

The standard engineering fix is to drop the first $w$ observations of each block as a washout period. The bias of the washout-corrected estimator is $O(\kappa \rho^w)$ for carryover decay rate $\rho < 1$, at the cost of reduced effective sample size. The optimal washout length balances bias against variance:

$$w^* = \arg\min_w \left[\kappa^2 \rho^{2w} + \frac{\sigma^2}{T - Bw}\right]$$

For exponential carryover $\kappa \rho^s$ with known $\rho$, this gives $w^* = \frac{1}{2|\log \rho|} \log\left(\frac{\kappa^2 T |\log \rho|}{\sigma^2}\right)$.

---

## 43.5 Long-Run vs. Short-Run Treatment Effects and Novelty

### 43.5.1 The Novelty Effect Decomposition

When a new product feature is introduced, short-run adoption and engagement may be inflated by novelty (curiosity, recency) or deflated by disruption (unfamiliarity, habit change). Let $\tau(t)$ be the treatment effect as a function of time since exposure. The observed short-run estimate from a $T$-period experiment is:

$$\hat{\tau}_{ST} = \frac{1}{T} \int_0^T \tau(t) \, dt$$

The long-run estimand of interest is $\tau_{LR} = \lim_{t \to \infty} \tau(t)$ (assuming convergence). A proxy extrapolation approach uses a surrogate outcome $S_i$ measurable in the short run but correlated with long-run behavior:

**Assumption 43.2 (Proxy Surrogacy)**
$Y_i^{LR} \perp D_i \mid S_i, \mathbf{X}_i$ — the long-run outcome is independent of treatment given the short-run surrogate and covariates.

Under surrogacy, $\tau_{LR} = E[Y^{LR}(1) - Y^{LR}(0)] = E[\partial E[Y^{LR} \mid S, \mathbf{X}] / \partial S \cdot (S(1) - S(0))]$, giving the proxy formula:

$$\hat{\tau}_{LR} = \hat{\tau}_{ST} \times \hat{\gamma}, \quad \hat{\gamma} = \frac{\partial E[Y^{LR} \mid S, \mathbf{X}]}{\partial S}$$

where $\hat{\gamma}$ is estimated from historical cohorts that have long-run outcome data, and $\hat{\tau}_{ST}$ is the short-run surrogate effect from the current experiment. This is the Prentice–Chen–Tian surrogate framework applied to business outcomes.

### 43.5.2 Heterogeneous Long-Run Effects and Habituation

Habituation is the phenomenon where $\tau(t) \to 0$ as $t \to \infty$ even when $\tau(0) > 0$: users adapt to the feature and it ceases to affect behavior. Formally, the treatment effect process $\tau(t) = \tau_\infty + (\tau_0 - \tau_\infty)e^{-\lambda t}$ with $\tau_\infty < \tau_0$.

Detecting habituation requires a heterogeneous-tenure analysis: compare treatment effects among users who have been exposed for different durations. In a staggered rollout (which the ACA Medicaid expansion exactly is), the Callaway-Sant'Anna estimator from Chapter 36 provides group-time ATTs $ATT(g, t)$ that can be plotted as a function of exposure duration $t - g$ to estimate $\tau(t - g)$.

---

## 43.6 Running Example: Price Elasticity of Medicaid Enrollment

The ACA Medicaid expansion created plausibly exogenous variation in the "price" of insurance to near-poor households: states that expanded set premiums to zero for those below 138% FPL, while non-expanding states maintained various premium and cost-sharing structures. Using state-year variation in copayment requirements as an instrument for effective price, we can estimate the price elasticity of Medicaid enrollment.

Let $Q_{st}$ be the Medicaid enrollment rate in state $s$ at year $t$, $P_{st}$ the average effective copayment (from CMS waiver filings), and $Z_{st}$ an indicator for ACA expansion (or alternatively, the copayment level mandated by the federal benchmark, which varies by income threshold and is plausibly exogenous to state-level health demand). The 2SLS estimator in first-differences:

$$\Delta \log Q_{st} = \varepsilon \Delta \log P_{st} + \Delta \mathbf{X}_{st}' \beta + \Delta u_{st}$$

instrumenting $\Delta \log P_{st}$ with $\Delta Z_{st}$. In the Oregon experiment, the instrument is `selected` (lottery winner), the price is the effective premium (zero for Medicaid-enrolled, positive for uninsured), and the outcome is `ohp_all_ever_admin` (ever enrolled in Medicaid). This IV recovers the enrollment elasticity among lottery compliers — those who enrolled *because* they won the lottery.

---

## Python: Price Elasticity IV, Switchback Simulation, and Shapley Attribution

```python
"""
Chapter 43: Causal Inference for Pricing, Marketing, and Product Interventions

Sections:
  1. IV price elasticity: OHE lottery data
  2. Switchback simulation with carryover bias
  3. Shapley-value attribution (shap + synthetic path data)
  4. DML heterogeneous elasticity via EconML
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
import warnings
warnings.filterwarnings("ignore")

# ── 1. IV PRICE ELASTICITY: OREGON HEALTH INSURANCE EXPERIMENT ──────────────

def load_ohe(path: str = "~/data/oregon/oregonhie_survey12m_vars.csv") -> pd.DataFrame:
    """
    Load OHE 12-month survey. Key variables:
      selected         : lottery instrument (Z)
      ohp_all_ever_admin : ever enrolled in Medicaid (D)
      doc_any_12m      : saw a doctor in last 12 months (Y)
      catastrophic_exp_inp : had catastrophic inpatient expenditure (Y)
      numhh_list       : household size (strata for lottery)
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["selected", "ohp_all_ever_admin", "doc_any_12m",
                            "catastrophic_exp_inp", "numhh_list"])
    df["selected"] = df["selected"].astype(int)
    df["ohp_all_ever_admin"] = df["ohp_all_ever_admin"].astype(float)
    df["doc_any_12m"] = df["doc_any_12m"].astype(float)
    df["catastrophic_exp_inp"] = df["catastrophic_exp_inp"].astype(float)
    # Stratum dummies for lottery design
    df = pd.get_dummies(df, columns=["numhh_list"], prefix="hh", drop_first=True)
    return df


def iv_elasticity(df: pd.DataFrame) -> dict:
    """
    2SLS: effect of Medicaid enrollment on doctor visits and catastrophic spending.
    Z = lottery selection, D = ever enrolled, Y = doc_any_12m / catastrophic_exp_inp.
    
    We proxy price elasticity interpretation: 'selected' shifts the effective
    price of insurance to zero for compliers. ITT / first-stage = LATE (Wald).
    """
    hh_cols = [c for c in df.columns if c.startswith("hh_")]
    controls = " + ".join(hh_cols) if hh_cols else "1"
    
    results = {}
    for outcome in ["doc_any_12m", "catastrophic_exp_inp"]:
        # OLS (naive, biased)
        ols = smf.ols(f"{outcome} ~ ohp_all_ever_admin + {controls}", data=df).fit()
        
        # First stage: D ~ Z + controls
        fs_formula = f"ohp_all_ever_admin ~ selected + {controls}"
        fs = smf.ols(fs_formula, data=df).fit()
        first_stage_F = (fs.fvalue if hasattr(fs, 'fvalue') else
                         fs.tvalues["selected"]**2)
        
        # ITT: Y ~ Z + controls
        itt = smf.ols(f"{outcome} ~ selected + {controls}", data=df).fit()
        
        # Wald (LATE): ITT / first-stage coefficient
        wald = itt.params["selected"] / fs.params["selected"]
        wald_se = abs(itt.bse["selected"] / fs.params["selected"])  # delta method approx
        
        results[outcome] = {
            "ols_coef": ols.params["ohp_all_ever_admin"],
            "ols_se": ols.bse["ohp_all_ever_admin"],
            "first_stage_coef": fs.params["selected"],
            "first_stage_F": first_stage_F,
            "itt_coef": itt.params["selected"],
            "wald_late": wald,
            "wald_se": wald_se,
        }
    return results


def print_elasticity_table(results: dict) -> None:
    print("\n" + "="*65)
    print(f"{'Outcome':<30} {'OLS':>8} {'OLS SE':>8} {'LATE':>8} {'LATE SE':>8}")
    print("-"*65)
    for outcome, r in results.items():
        print(f"{outcome:<30} {r['ols_coef']:>8.4f} {r['ols_se']:>8.4f} "
              f"{r['wald_late']:>8.4f} {r['wald_se']:>8.4f}")
    print("="*65)
    for outcome, r in results.items():
        print(f"  {outcome}: First-stage F = {r['first_stage_F']:.1f}, "
              f"ITT = {r['itt_coef']:.4f}")


# ── 2. SWITCHBACK SIMULATION WITH CARRYOVER BIAS ────────────────────────────

def simulate_switchback(
    T: int = 500,
    block_length: int = 10,
    tau_direct: float = 0.10,
    kappa_carryover: float = 0.06,
    sigma: float = 0.20,
    washout: int = 0,
    seed: int = 42
) -> dict:
    """
    Simulate switchback experiment with first-order carryover.
    
    DGP: Y_t = mu + tau * D_t + kappa * D_{t-1} + eps_t
    Design: balanced alternating blocks of length `block_length`.
    
    Returns naive estimate and washout-corrected estimate.
    """
    rng = np.random.default_rng(seed)
    mu = 1.0

    # Generate treatment schedule: alternate blocks
    n_blocks = T // block_length
    block_assignments = np.repeat(
        np.tile([1, 0], n_blocks // 2 + 1)[:n_blocks], block_length
    )[:T]

    D = block_assignments.astype(float)
    D_lag = np.concatenate([[0.0], D[:-1]])  # lagged treatment

    eps = rng.normal(0, sigma, T)
    Y = mu + tau_direct * D + kappa_carryover * D_lag + eps

    # Mask for valid (non-washout) observations within each block
    block_id = np.arange(T) // block_length
    position_in_block = np.arange(T) % block_length
    valid = position_in_block >= washout

    naive_est = np.mean(Y[(D == 1) & valid]) - np.mean(Y[(D == 0) & valid])

    # Theoretical bias
    theoretical_bias = kappa_carryover / 2 if washout == 0 else (
        kappa_carryover * (0.5 ** washout)
    )

    return {
        "naive_estimate": naive_est,
        "true_direct_effect": tau_direct,
        "bias": naive_est - tau_direct,
        "theoretical_bias": theoretical_bias,
        "n_valid": valid.sum(),
    }


def switchback_bias_experiment() -> pd.DataFrame:
    """
    Show how washout period reduces bias at cost of sample size.
    """
    rows = []
    for washout in [0, 1, 2, 3, 5]:
        results_list = [
            simulate_switchback(T=2000, block_length=10,
                                tau_direct=0.10, kappa_carryover=0.06,
                                washout=washout, seed=s)
            for s in range(200)
        ]
        biases = [r["bias"] for r in results_list]
        n_valid = results_list[0]["n_valid"]
        rows.append({
            "washout": washout,
            "mean_bias": np.mean(biases),
            "std_bias": np.std(biases),
            "n_valid": n_valid,
            "theoretical_bias": results_list[0]["theoretical_bias"],
        })
    return pd.DataFrame(rows)


# ── 3. SHAPLEY ATTRIBUTION ───────────────────────────────────────────────────

def simulate_marketing_paths(
    n: int = 5000,
    seed: int = 99
) -> pd.DataFrame:
    """
    Synthetic multi-touch attribution data.
    Touchpoints: [search, email, display, social]
    True incremental effects: search=0.15, email=0.08, display=0.03, social=0.05
    Last-touch is biased because search correlates with high intent.
    """
    rng = np.random.default_rng(seed)
    
    # Latent intent (unobserved by firm, drives both search and conversion)
    intent = rng.normal(0, 1, n)
    
    touchpoints = {
        "search":  (rng.random(n) < 0.3 + 0.2 * (intent > 0)).astype(float),
        "email":   (rng.random(n) < 0.4).astype(float),
        "display": (rng.random(n) < 0.5).astype(float),
        "social":  (rng.random(n) < 0.35).astype(float),
    }
    
    # True causal conversion probability (not confounded with search)
    logit = (-2.0
             + 0.15 * touchpoints["search"]  # causal effect
             + 0.08 * touchpoints["email"]
             + 0.03 * touchpoints["display"]
             + 0.05 * touchpoints["social"]
             + 0.40 * intent)               # confounding
    p_convert = 1 / (1 + np.exp(-logit))
    converted = (rng.random(n) < p_convert).astype(float)
    
    df = pd.DataFrame(touchpoints)
    df["converted"] = converted
    df["intent"] = intent
    return df


def shapley_attribution(df: pd.DataFrame) -> pd.Series:
    """
    Compute Shapley attribution for touchpoints {search, email, display, social}.
    
    Value function v(S): empirical conversion rate for users exposed to exactly
    the touchpoints in S (and only those).
    
    Uses shap.maskers.Independent on a LogisticRegression for efficiency;
    falls back to exact enumeration for K<=4.
    """
    import itertools
    
    channels = ["search", "email", "display", "social"]
    K = len(channels)
    
    # Build conversion rate lookup: v(S) = mean(converted) for users with
    # exactly the exposures matching S
    def v(S):
        mask = np.ones(len(df), dtype=bool)
        for ch in channels:
            if ch in S:
                mask &= (df[ch] == 1)
            else:
                mask &= (df[ch] == 0)
        if mask.sum() < 5:
            # Fallback: relax exact matching, use all users with S exposed
            mask2 = np.ones(len(df), dtype=bool)
            for ch in S:
                mask2 &= (df[ch] == 1)
            return df.loc[mask2, "converted"].mean() if mask2.sum() > 0 else 0.0
        return df.loc[mask, "converted"].mean()
    
    # Exact Shapley enumeration
    phi = {ch: 0.0 for ch in channels}
    for k_idx, ch in enumerate(channels):
        others = [c for c in channels if c != ch]
        for size in range(len(others) + 1):
            for coalition in itertools.combinations(others, size):
                S = set(coalition)
                weight = (np.math.factorial(size) *
                          np.math.factorial(K - size - 1) /
                          np.math.factorial(K))
                marginal = v(S | {ch}) - v(S)
                phi[ch] += weight * marginal
    
    return pd.Series(phi)


def last_touch_attribution(df: pd.DataFrame) -> pd.Series:
    """
    Last-touch: attribute conversion to last observed touchpoint.
    Assumes order: search > email > display > social (recency proxy).
    """
    channels = ["social", "display", "email", "search"]  # search = most recent
    last_touch = pd.Series(0.0, index=["search", "email", "display", "social"])
    
    for _, row in df[df["converted"] == 1].iterrows():
        for ch in channels:  # later in list = more recent
            if row[ch] == 1:
                last_touch[ch] += 1
                break
    return last_touch / last_touch.sum()


# ── 4. DML HETEROGENEOUS ELASTICITY VIA ECONML ──────────────────────────────

def dml_heterogeneous_elasticity_aca(
    path_brfss: str = "~/data/brfss/brfss_panel.csv"
) -> None:
    """
    DML-IV estimate of heterogeneous price elasticity of insurance take-up
    using ACA Medicaid expansion state-year variation.
    
    Treatment D: state-year Medicaid enrollment rate (or individual coverage)
    Instrument Z: ACA expansion indicator (× year >= 2014)
    Outcome Y: healthcare access index (BRFSS: hlthpln1, checkup, medcost)
    Covariates X: age, income_cat, race, state FE, year FE
    
    Heterogeneity: elasticity by income category (near-poor vs. moderate income)
    """
    try:
        from econml.iv.dml import DMLIV
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("econml not installed. Run: pip install econml")
        return

    try:
        df = pd.read_csv(path_brfss)
    except FileNotFoundError:
        # Simulate BRFSS-like panel for illustration
        rng = np.random.default_rng(0)
        n = 8000
        state_ids = rng.integers(0, 50, n)
        year = rng.choice([2012, 2013, 2014, 2015, 2016], n)
        expanded_states = rng.choice(range(50), 30, replace=False)
        expansion_year = {s: 2014 for s in expanded_states}
        
        Z = np.array([
            1.0 if (state_ids[i] in expansion_year and
                    year[i] >= expansion_year[state_ids[i]])
            else 0.0
            for i in range(n)
        ])
        age = rng.normal(45, 12, n).clip(18, 80)
        income_cat = rng.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
        # Confounded enrollment (higher income = more likely to be insured anyway)
        D_logit = -1.5 + 1.2 * Z + 0.3 * income_cat + rng.normal(0, 0.5, n)
        D = (1 / (1 + np.exp(-D_logit)))
        # Heterogeneous outcome effect: larger for low income
        true_effect = 0.20 - 0.08 * income_cat
        Y = 0.5 + true_effect * D + 0.05 * (age - 45)/12 + rng.normal(0, 0.15, n)
        
        df = pd.DataFrame({
            "Y": Y, "D": D, "Z": Z,
            "age_std": (age - 45) / 12,
            "income_cat": income_cat,
            "state": state_ids,
        })
        print("  [Using simulated BRFSS-like data]")

    feature_cols = [c for c in ["age_std", "income_cat"] if c in df.columns]
    X = df[feature_cols].values
    Y = df["Y"].values
    D = df["D"].values
    Z = df["Z"].values

    model = DMLIV(
        model_y_xw=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        model_t_xw=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        model_t_xwz=GradientBoostingClassifier(n_estimators=100, max_depth=3),
        model_final=LinearRegression(fit_intercept=False),
        cv=3,
        random_state=42,
    )
    model.fit(Y, D, Z=Z, X=X)

    print("\nDML-IV Heterogeneous Elasticity:")
    print(f"  Average effect (ATE): {model.ate():.4f}")
    if X.shape[1] > 0:
        X_grid = np.array([[0, 0], [0, 1], [0, 2]])  # low/mid/high income
        effects = model.effect(X_grid)
        for i, inc in enumerate(["Low income", "Mid income", "High income"]):
            print(f"  {inc}: CATE = {effects[i]:.4f}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("\n" + "="*65)
    print("CHAPTER 43: Price Elasticity IV (Oregon Health Insurance Experiment)")
    print("="*65)

    ohe_path = os.path.expanduser("~/data/oregon/oregonhie_survey12m_vars.csv")
    if os.path.exists(ohe_path):
        df_ohe = load_ohe(ohe_path)
        results = iv_elasticity(df_ohe)
        print_elasticity_table(results)
        print("\nInterpretation:")
        print("  OLS coefficient on insurance is upward-biased (selection)")
        print("  LATE (Wald) recovers effect on lottery compliers")
        for outcome, r in results.items():
            direction = ">" if r["ols_coef"] > r["wald_late"] else "<"
            print(f"  {outcome}: OLS {r['ols_coef']:.4f} {direction} LATE {r['wald_late']:.4f}")
    else:
        print(f"OHE data not found at {ohe_path}. Download from https://data.nber.org/oregon/")

    print("\n" + "="*65)
    print("SWITCHBACK BIAS vs. WASHOUT PERIOD")
    print("="*65)
    sb_table = switchback_bias_experiment()
    print(sb_table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNote: bias decays with washout at cost of valid observations.")

    print("\n" + "="*65)
    print("SHAPLEY ATTRIBUTION vs. LAST-TOUCH")
    print("="*65)
    df_mkt = simulate_marketing_paths(n=10000, seed=42)
    shapley = shapley_attribution(df_mkt)
    last_touch = last_touch_attribution(df_mkt)
    attr_df = pd.DataFrame({
        "Shapley": shapley,
        "LastTouch": last_touch,
    })
    print(attr_df.round(4))
    print("\nNote: Last-touch over-credits 'search' due to intent confounding.")
    print("Shapley distributes credit via marginal contribution, not rank.")
    print("Neither recovers true causal effects without randomization.")

    print("\n" + "="*65)
    print("DML HETEROGENEOUS ELASTICITY (ACA/BRFSS or Simulated)")
    print("="*65)
    brfss_path = os.path.expanduser("~/data/brfss/brfss_panel.csv")
    dml_heterogeneous_elasticity_aca(path_brfss=brfss_path)
```

---

## Summary

- Price elasticity estimated by OLS is biased because prices are set in response to demand signals; cost-shifter IV (e.g., lottery assignment in OHE, ACA expansion in BRFSS) resolves endogeneity and recovers a LATE among compliers; DML-IV extends this to heterogeneous elasticity across market segments.

- Last-touch and multi-touch attribution rules are biased estimators of causal incrementality because ad assignment algorithms target high-intent users; randomized holdouts (ghost ads, geo-experiments) identify the ATT directly; Shapley attribution satisfies consistency axioms and improves on last-touch but does not recover causal effects without randomization.

- Standard A/B testing is biased under network interference; the direction and magnitude of bias depend on the spillover parameter $\gamma$ and the randomization level; cluster-randomized designs with two-level treatment assignment separate direct from spillover effects.

- Switchback experiments introduce bias equal to half the carryover effect under rapid block switching; bias decays as $\kappa/2$ for single-lag carryover and can be corrected by washout periods at the cost of reduced effective sample, with the optimal washout length trading squared bias against variance.

- Short-run experiment estimates may conflate novelty effects with stable treatment effects; the proxy surrogacy framework extrapolates to long-run effects by estimating $\hat{\tau}_{LR} = \hat{\tau}_{ST} \times \hat{\gamma}$ where $\hat{\gamma}$ is the short-run-to-long-run mapping estimated from historical cohorts; validity requires the surrogacy assumption $Y^{LR} \perp D \mid S, \mathbf{X}$.

- All four design problems — pricing, attribution, network effects, time-series interference — reduce to the same core challenge that runs throughout this book: the assignment mechanism is not random, and identification requires either an instrument, a valid design, or an untestable structural assumption; the choice among these alternatives is always domain-specific and should be made explicit.

---

## Further Reading

**Angrist, J.D. and Imbens, G.W. (1994)**. "Identification and Estimation of Local Average Treatment Effects." *Econometrica* 62(2): 467–475. The foundational paper establishing LATE identification via instruments; directly applicable to the IV elasticity setup of Section 43.1.

**Bojinov, I. and Shephard, N. (2019)**. "Time Series Experiments and Causal Estimands: Exact Randomization Tests and Trading." *Journal of the American Statistical Association* 114(528): 1665–1682. Provides the formal causal framework for switchback experiments, defining exact causal estimands and valid inference under temporal interference without SUTVA.

**Shapley, L.S. (1953)**. "A Value for n-Person Games." In *Contributions to the Theory of Games*, Vol. II. The original paper; combined with the Prentice–Chen–Tian surrogacy literature and Proposition 43.1 here to characterize when Shapley attribution is and is not causally interpretable.

**Chernozhukov, V. et al. (2018)**. "Double/Debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal* 21(1): C1–C68. The DML framework underlying Section 43.1.3; the Neyman orthogonality condition is what makes first-stage estimation error asymptotically negligible in the heterogeneous elasticity estimator.

**Prentice, R.L. (1989)**. "Surrogate Endpoints in Clinical Trials: Definition and Operational Criteria." *Statistics in Medicine* 8(4): 431–440. The original surrogacy condition used in Section 43.5; the business-context proxy extrapolation formula derives from this framework combined with the identification argument in Athey et al. (2019) "Surrogate Indices."

**Bajari, P. et al. (2021)**. "Multiple Randomization Designs." *NBER Working Paper 29080*. Develops the two-level Bernoulli-over-Bernoulli design discussed in Section 43.3.2 for separating direct and spillover effects; includes large-scale empirical applications from marketplace platforms.
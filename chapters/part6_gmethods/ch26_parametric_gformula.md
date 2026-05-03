# Chapter 26: The Parametric G-Formula

## 26.1 From Point Treatment to Sustained Strategies

Chapter 25 introduced the potential outcomes framework under time-varying treatment and established the central identification challenge: a sequence of treatments $\bar{A} = (A_0, A_1, \ldots, A_{T-1})$ induces sequential confounding when intermediate covariates $\bar{L}$ both predict future treatment and are themselves affected by past treatment. Conventional regression conditioning on $\bar{L}$ biases effect estimates of the total strategy. The parametric g-formula, introduced by Robins (1986), resolves this by replacing naive regression with a procedure that respects the temporal ordering of the data-generating process.

The quantity of primary interest is $E[Y(\bar{a})]$: the expected potential outcome under a protocol that sets treatment to the fixed sequence $\bar{a} = (a_0, a_1, \ldots, a_{T-1})$. When $T = 1$ this reduces to the familiar average treatment effect, but for $T > 1$ identification requires assumptions that run across the entire trajectory. The g-formula provides an estimand that is nonparametrically identified under those assumptions, and the parametric g-formula replaces the nonparametric conditional distributions with estimated models, making estimation feasible in practice.

Motivation from the running example: the Oregon Health Insurance Experiment assigned insurance access via a lottery, but in the ACA Medicaid expansion context insurance status varies over time within individuals. States expanded at different calendar years; within each state, individuals may gain or lose coverage as their income and family circumstances change. The question becomes: what would average health outcomes look like under a hypothetical world where all individuals had sustained coverage throughout the observation window, compared with the world where coverage follows the observed pattern? This is precisely a comparison of $E[Y(\bar{a}^*)]$ (sustained coverage) against the natural course distribution.

## 26.2 The G-Formula: Identification via Sequential Standardization

Let $\bar{L}_t = (L_0, \ldots, L_t)$ and $\bar{A}_t = (A_0, \ldots, A_t)$ denote the history of covariates and treatments through period $t$. Write $\bar{l}_{t-1}$ and $\bar{a}_{t-1}$ for realized histories. The observed data consist of $n$ iid copies of $(L_0, A_0, L_1, A_1, \ldots, L_{T-1}, A_{T-1}, Y)$.

**Assumption 26.1 (Sequential Ignorability).** For each $t = 0, \ldots, T-1$:

$$A_t \perp\!\!\!\perp Y(\bar{a}) \mid \bar{A}_{t-1} = \bar{a}_{t-1},\ \bar{L}_t = \bar{l}_t \quad \forall\ \bar{a}.$$

This is the time-varying analog of unconfoundedness: conditional on the full observed history through period $t$, treatment assignment at $t$ is as good as random with respect to all future potential outcomes.

**Assumption 26.2 (Positivity).** For all $(\bar{a}_{t-1}, \bar{l}_t)$ in the support of the observed data:

$$P(A_t = a_t \mid \bar{A}_{t-1} = \bar{a}_{t-1},\ \bar{L}_t = \bar{l}_t) > 0 \quad \forall\ a_t \in \mathcal{A}.$$

**Theorem 26.1 (G-Formula Identification).** Under Assumptions 26.1 and 26.2, for any fixed sequence $\bar{a}$:

$$E[Y(\bar{a})] = \sum_{\bar{l}} E\!\left[Y \mid \bar{A} = \bar{a},\ \bar{L} = \bar{l}\right] \prod_{t=0}^{T-1} P\!\left(L_t = l_t \mid \bar{A}_{t-1} = \bar{a}_{t-1},\ \bar{L}_{t-1} = \bar{l}_{t-1}\right).$$

For continuous covariates replace summation with integration over the appropriate product measure.

*Proof sketch.* The argument is induction on $T$, repeatedly applying the law of iterated expectations together with sequential ignorability. At the final period $T-1$:

$$E[Y(\bar{a})] = E\!\left[E\!\left[Y(\bar{a}) \mid \bar{A}_{T-2}, \bar{L}_{T-1}\right]\right].$$

Sequential ignorability at $T-1$ gives $E[Y(\bar{a}) \mid \bar{A}_{T-1} = \bar{a},\ \bar{L}_{T-1}] = E[Y \mid \bar{A} = \bar{a},\ \bar{L}_{T-1}]$. Positivity ensures the conditional distribution is well defined. One then integrates over $L_{T-1}$ using the distribution conditional on $\bar{A}_{T-2} = \bar{a}_{T-2},\ \bar{L}_{T-2}$, and applies sequential ignorability at $T-2$ to replace $L_{T-1}$'s distribution under the intervention with its distribution under the observed protocol. Iterating back to $t = 0$ yields the product formula. $\square$

The key structural insight is the **factorization**: the joint distribution $P(\bar{L} \mid \bar{A} = \bar{a})$ under an intervention is not the same as the observed joint distribution $P(\bar{L} \mid \bar{A} = \bar{a})$ in general — the observed distribution conditions on being in a subgroup that actually received $\bar{a}$, while the interventional distribution integrates over the natural evolution of $\bar{L}$ under the imposed treatment. Sequential ignorability is precisely what licenses replacing the former with the latter.

## 26.3 Static and Dynamic Interventions

A **static intervention** sets $A_t = a_t$ for a deterministic sequence $\bar{a}$ regardless of covariate history. The "always treat" strategy $A_t = 1$ for all $t$ and "never treat" strategy $A_t = 0$ for all $t$ are the canonical examples for binary treatment.

A **dynamic intervention** sets $A_t = d_t(\bar{l}_t, \bar{a}_{t-1})$ as a function of the evolving covariate history. For instance, a threshold rule "treat if $L_t > c$" is dynamic. The g-formula accommodates dynamic interventions directly: replace the fixed $a_t$ in the conditioning event by $d_t(\bar{l}_t, \bar{a}_{t-1})$, and the same identification argument applies provided positivity holds for all covariate paths that the dynamic rule could generate.

**Definition 26.1 (Natural Course).** The natural course distribution is the observational distribution: $E[Y^{\text{nat}}] = E[Y]$. Under no unmeasured confounding (sequential ignorability), the g-formula evaluated at the observed treatment mechanism $P(A_t \mid \bar{A}_{t-1}, \bar{L}_t)$ recovers $E[Y]$. This provides an internal consistency check: g-computation under the natural course should match the empirical mean.

The **average treatment effect of a strategy** relative to the natural course is:

$$\text{ATE}(\bar{a}^*) = E[Y(\bar{a}^*)] - E[Y^{\text{nat}}].$$

When comparing two static strategies, e.g., always-treat vs. never-treat:

$$\text{ATE} = E[Y(\bar{1})] - E[Y(\bar{0})].$$

## 26.4 Parametric Models for Sequential Components

The g-formula involves two types of components: (i) the outcome model $E[Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l}]$ and (ii) the covariate models $P(L_t = l_t \mid \bar{A}_{t-1} = \bar{a}_{t-1}, \bar{L}_{t-1} = \bar{l}_{t-1})$ for each $t$. In the parametric g-formula, each component is estimated by a separate regression model, then combined via simulation.

**Outcome model.** For a binary outcome $Y$ (e.g., whether the individual reports poor health), fit:

$$\text{logit}\, P(Y = 1 \mid \bar{A}, \bar{L}) = \alpha_0 + \alpha_A^\top \bar{A} + \alpha_L^\top \bar{L} + \text{interactions}.$$

This model is fit on the observed data using maximum likelihood; it need not include treatment-covariate interactions to identify a marginal effect, but misspecification of functional form will bias the g-formula estimate.

**Covariate models.** For each time-varying covariate $L_t$ that is a function of past treatment and covariate history, fit:

$$P(L_t = 1 \mid \bar{A}_{t-1}, \bar{L}_{t-1}) = \text{expit}(\beta_{t,0} + \beta_{t,A}^\top \bar{A}_{t-1} + \beta_{t,L}^\top \bar{L}_{t-1}).$$

For continuous covariates, a linear model $L_t = \mu_t(\bar{A}_{t-1}, \bar{L}_{t-1}) + \varepsilon_t$ with $\varepsilon_t \sim N(0, \sigma_t^2)$ is typical.

**Remark 26.1 (Separability).** A critical feature of the parametric g-formula is that each component model is fit separately on the observed data — there is no joint optimization across the sequence. This separability means that misspecification of $P(L_t \mid \cdot)$ propagates through the Monte Carlo simulation and contaminates the final estimate, even if the outcome model is correctly specified. This sensitivity is the central practical limitation discussed in Section 26.6.

## 26.5 Monte Carlo G-Computation

Because the g-formula involves a high-dimensional sum (or integral) over covariate histories, direct evaluation is computationally infeasible except in simple discrete settings. **Monte Carlo g-computation** approximates the integral by simulation.

**Algorithm 26.1 (Monte Carlo G-Computation).**

1. Fit the outcome model $\hat{E}[Y \mid \bar{A}, \bar{L}]$ and covariate models $\hat{P}(L_t \mid \bar{A}_{t-1}, \bar{L}_{t-1})$ on the observed data.

2. For each Monte Carlo draw $m = 1, \ldots, M$:

   a. Sample baseline covariates $L_0^{(m)}$ from the empirical distribution (draw a row from the observed data with replacement).

   b. For $t = 0, 1, \ldots, T-1$:
   
      - Set $A_t^{(m)} = a_t$ (the intervened value under strategy $\bar{a}$).
      - Draw $L_{t+1}^{(m)} \sim \hat{P}(L_{t+1} \mid \bar{A}_t^{(m)} = \bar{a}_t, \bar{L}_t^{(m)})$.

   c. Predict $\hat{Y}^{(m)} = \hat{E}\!\left[Y \mid \bar{A}^{(m)} = \bar{a},\ \bar{L}^{(m)}\right]$.

3. The Monte Carlo estimator is:

$$\hat{E}[Y(\bar{a})] = \frac{1}{M} \sum_{m=1}^{M} \hat{Y}^{(m)}.$$

**Convergence.** As $M \to \infty$ the Monte Carlo error vanishes, so in practice $M = 10{,}000$ is adequate. The dominant variance comes from estimation of the component models, not from simulation noise. The estimator is consistent under correct specification of all component models, and $M$ needs only to be large enough that simulation variance is negligible relative to estimation variance.

**Variance estimation.** The nonparametric bootstrap is the standard approach: resample $n$ individuals with replacement, refit all component models, re-run the Monte Carlo simulation, and compute the g-formula estimate. Repeat $B = 500$ times; the bootstrap standard error is the standard deviation across replicates. The bootstrap correctly propagates uncertainty from all estimation steps simultaneously.

An analytical alternative uses the influence function of the g-formula functional. Let $\theta(\bar{a}) = E[Y(\bar{a})]$; the efficient influence function under nonparametric models is a sum of augmentation terms at each time point (the doubly-robust representation of the g-formula, connecting to TMLE in Chapter 30). For purely parametric models the delta method can be applied, but deriving the required Jacobians across $T$ sequential models is algebraically involved and rarely implemented in practice.

## 26.6 Bias from Sequential Model Misspecification

The parametric g-formula is consistent only when all component models are correctly specified. In longitudinal settings with $T$ periods, this requirement multiplies: a small bias at each step compounds through the simulation. Formally, let $\tilde{P}_t$ denote the misspecified covariate model. The simulated covariate paths follow distribution $\tilde{P}_0 \cdot \tilde{P}_1 \cdots \tilde{P}_{T-1}$, which may diverge substantially from the true interventional distribution $P_0^* \cdot P_1^* \cdots P_{T-1}^*$ even when each $\tilde{P}_t$ is close to $P_t^*$ marginally.

A practical diagnostic is to compare the natural course g-computation estimate against the empirical outcome mean. Under sequential ignorability, these should agree asymptotically. Systematic disagreement signals covariate model misspecification. This check is necessary but not sufficient: models could misspecify the interventional distribution while replicating the natural course by accident if confounders are weak.

**Remark 26.2 (Dimension reduction and regularization).** When $\bar{L}$ is high-dimensional, the component models must trade off flexibility against overfitting. Including $\bar{L}_{t-1}$ via summary statistics (cumulative exposure, lagged means) rather than full history is common practice. Regularized regression (lasso, ridge) can be used for the component models without affecting the identification argument, though inference via bootstrap remains necessary since model selection invalidates standard MLE confidence intervals.

## 26.7 Connection to the Observed Data Distribution

It is instructive to write the g-formula estimand as a functional of the observed data distribution $P$. Define the sequential g-formula functional:

$$\Psi(\bar{a}; P) = \int \cdots \int E_P[Y \mid \bar{A} = \bar{a}, \bar{L} = \bar{l}] \prod_{t=0}^{T-1} p_P(l_t \mid \bar{a}_{t-1}, \bar{l}_{t-1})\, d\bar{l}.$$

This is a smooth functional of $P$ in the sense that its pathwise derivative exists, which is the foundation for constructing doubly-robust and efficient estimators. The parametric g-formula replaces $P$ with a parametric approximation $\hat{P}$; the resulting estimator inherits the bias of $\hat{P}$ but gains variance reduction from dimension reduction.

The contrast with IPTW (Chapter 27) is instructive. IPTW reweights the observed outcome distribution by inverse probability weights $\prod_t P(A_t \mid \bar{A}_{t-1}, \bar{L}_t)^{-1}$, requiring only models for the treatment mechanism, not for $\bar{L}$. The g-formula requires models for $\bar{L}$ but not for the treatment mechanism (beyond what enters the outcome model's conditioning set). These two strategies have complementary model requirements, which is why doubly-robust estimators that combine them (AIPW, TMLE) can achieve consistency when only one set of models is correctly specified.

## Python: G-Formula Applied to Simulated Longitudinal Data and BRFSS

The following implementation applies Monte Carlo g-computation to: (1) the simulated longitudinal DGP from `src/causal_book/data/simulate.py` to verify the estimator against a known truth, and (2) BRFSS state-year panel data with staggered Medicaid expansion as time-varying treatment.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. Simulated DGP: verify g-formula recovers known ATE
# ─────────────────────────────────────────────────────────────

def simulate_longitudinal(n: int = 5000, T: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    DGP:
        L0 ~ N(0,1)  (baseline confounder)
        A0 ~ Bernoulli(expit(0.5*L0))
        L1 ~ N(0.4*A0 + 0.6*L0, 1)   (time-varying confounder, affected by A0)
        A1 ~ Bernoulli(expit(0.5*L1 - 0.3*A0))
        L2 ~ N(0.4*A1 + 0.6*L1, 1)
        A2 ~ Bernoulli(expit(0.5*L2 - 0.3*A1))
        Y  ~ N(1.2*(A0+A1+A2) + 0.5*L2 + 0.3*L0, 1)
    True ATE (always treat vs never treat) = 1.2 * 3 = 3.6
    """
    rng = np.random.default_rng(seed)
    L0 = rng.normal(0, 1, n)
    A0 = rng.binomial(1, expit(0.5 * L0))
    L1 = rng.normal(0.4 * A0 + 0.6 * L0, 1)
    A1 = rng.binomial(1, expit(0.5 * L1 - 0.3 * A0))
    L2 = rng.normal(0.4 * A1 + 0.6 * L1, 1)
    A2 = rng.binomial(1, expit(0.5 * L2 - 0.3 * A1))
    Y  = rng.normal(1.2 * (A0 + A1 + A2) + 0.5 * L2 + 0.3 * L0, 1)
    return pd.DataFrame({"L0": L0, "A0": A0, "L1": L1, "A1": A1,
                         "L2": L2, "A2": A2, "Y": Y})


def fit_component_models(df: pd.DataFrame):
    """
    Returns fitted sklearn models for each covariate and the outcome.
    Models:
        P(L1 | A0, L0)  -> LinearRegression
        P(L2 | A1, A0, L1, L0) -> LinearRegression
        E[Y  | A0,A1,A2, L0,L1,L2] -> LinearRegression
    """
    models = {}

    # L1 model
    X_L1 = df[["A0", "L0"]]
    m_L1 = LinearRegression().fit(X_L1, df["L1"])
    res_L1 = df["L1"] - m_L1.predict(X_L1)
    models["L1"] = {"model": m_L1, "sigma": res_L1.std(), "features": ["A0", "L0"]}

    # L2 model
    X_L2 = df[["A1", "A0", "L1", "L0"]]
    m_L2 = LinearRegression().fit(X_L2, df["L2"])
    res_L2 = df["L2"] - m_L2.predict(X_L2)
    models["L2"] = {"model": m_L2, "sigma": res_L2.std(), "features": ["A1", "A0", "L1", "L0"]}

    # Outcome model
    X_Y = df[["A0", "A1", "A2", "L0", "L1", "L2"]]
    m_Y = LinearRegression().fit(X_Y, df["Y"])
    models["Y"] = {"model": m_Y, "features": ["A0", "A1", "A2", "L0", "L1", "L2"]}

    return models


def monte_carlo_gformula(df: pd.DataFrame, models: dict,
                          strategy: dict, M: int = 10_000,
                          seed: int = 0) -> float:
    """
    Monte Carlo g-computation for T=3 periods.
    strategy: dict with keys "A0","A1","A2" mapping to 0, 1, or "natural"
    "natural" means draw from observed treatment model (not implemented here;
    we use empirical natural course check separately).
    """
    rng = np.random.default_rng(seed)
    # Sample baseline from empirical distribution
    idx = rng.integers(0, len(df), M)
    sim = df.iloc[idx][["L0"]].copy().reset_index(drop=True)

    # Period 0: set A0
    sim["A0"] = float(strategy["A0"])

    # Simulate L1
    X_L1 = sim[models["L1"]["features"]]
    mu_L1 = models["L1"]["model"].predict(X_L1)
    sim["L1"] = mu_L1 + rng.normal(0, models["L1"]["sigma"], M)

    # Period 1: set A1
    sim["A1"] = float(strategy["A1"])

    # Simulate L2
    X_L2 = sim[models["L2"]["features"]]
    mu_L2 = models["L2"]["model"].predict(X_L2)
    sim["L2"] = mu_L2 + rng.normal(0, models["L2"]["sigma"], M)

    # Period 2: set A2
    sim["A2"] = float(strategy["A2"])

    # Predict outcome
    X_Y = sim[models["Y"]["features"]]
    Y_hat = models["Y"]["model"].predict(X_Y)

    return float(Y_hat.mean())


def bootstrap_gformula(df: pd.DataFrame, strategy_1: dict, strategy_0: dict,
                        B: int = 500, M: int = 5_000) -> tuple[float, float]:
    """Bootstrap SE for ATE = E[Y(strategy_1)] - E[Y(strategy_0)]."""
    rng = np.random.default_rng(999)
    ates = []
    for b in range(B):
        boot = df.sample(n=len(df), replace=True, random_state=int(rng.integers(0, 2**31)))
        mods = fit_component_models(boot)
        e1 = monte_carlo_gformula(boot, mods, strategy_1, M=M, seed=b)
        e0 = monte_carlo_gformula(boot, mods, strategy_0, M=M, seed=b+10000)
        ates.append(e1 - e0)
    return float(np.mean(ates)), float(np.std(ates))


# ── Run on simulated data ──────────────────────────────────────
print("=" * 55)
print("SIMULATED DGP — G-Formula Verification")
print("=" * 55)

df_sim = simulate_longitudinal(n=5000, T=3)
models_sim = fit_component_models(df_sim)

always_treat = {"A0": 1, "A1": 1, "A2": 1}
never_treat  = {"A0": 0, "A1": 0, "A2": 0}

E_always = monte_carlo_gformula(df_sim, models_sim, always_treat, M=20_000)
E_never  = monte_carlo_gformula(df_sim, models_sim, never_treat,  M=20_000)
ATE_hat  = E_always - E_never

# Natural course check: should match empirical mean
# For natural course we set treatments to observed values in simulation
# (direct check: g-formula under observed A should ≈ E[Y])
print(f"E[Y(always treat)] = {E_always:.3f}")
print(f"E[Y(never treat)]  = {E_never:.3f}")
print(f"Estimated ATE      = {ATE_hat:.3f}  (true = 3.600)")
print(f"Empirical E[Y]     = {df_sim['Y'].mean():.3f}")

mean_bs, se_bs = bootstrap_gformula(df_sim, always_treat, never_treat, B=300, M=3_000)
print(f"Bootstrap ATE      = {mean_bs:.3f}  (SE = {se_bs:.3f})")
print(f"95% CI approx      = [{mean_bs - 1.96*se_bs:.3f}, {mean_bs + 1.96*se_bs:.3f}]")


# ─────────────────────────────────────────────────────────────
# 2. BRFSS / ACA Medicaid Expansion
#    We simulate a BRFSS-like panel since raw BRFSS requires
#    download and merge. DGP mirrors ACA staggered expansion.
# ─────────────────────────────────────────────────────────────

def simulate_brfss_panel(n_states: int = 50, n_years: int = 6,
                          n_per_state_year: int = 200,
                          seed: int = 123) -> pd.DataFrame:
    """
    Simulates BRFSS-like repeated cross-section panel, 2010–2015.
    Each state-year cell has ~n_per_state_year individuals.
    Treatment: Medicaid expansion (binary, absorbing after adoption).
    Covariates: age (time-varying), income_low (time-varying due to economy).
    Outcome: poor_health (binary).

    True causal effect of expansion: reduces poor_health probability by 0.04
    per year of coverage, mediated partly through income stability (L_t).
    """
    rng = np.random.default_rng(seed)
    years = list(range(2010, 2010 + n_years))

    # States adopt expansion in 2014 (ACA base) or 2015 or never
    # Simplified: 30 states expand in 2014, 10 in 2015, 10 never
    expand_year = (
        [2014] * 30 + [2015] * 10 + [9999] * 10
    )
    rng.shuffle(expand_year)  # randomize which states

    rows = []
    for s in range(n_states):
        ey = expand_year[s]
        # State-level baseline poor health propensity
        state_fe = rng.normal(0, 0.1)
        for y_idx, yr in enumerate(years):
            n_i = n_per_state_year
            A = int(yr >= ey)   # absorbing treatment
            # Time-varying covariate: low-income fraction (affected by expansion)
            income_low_base = 0.35 + state_fe + 0.01 * y_idx
            income_low = income_low_base - 0.05 * A + rng.normal(0, 0.02, n_i)
            age = rng.normal(45 + 0.5 * y_idx, 12, n_i)
            # Poor health: logistic model
            lin = (-1.2
                   + 0.02 * (age - 45)
                   + 0.8 * income_low
                   - 0.04 * A        # direct effect of coverage
                   + state_fe)
            poor_health = rng.binomial(1, expit(lin), n_i)
            for i in range(n_i):
                rows.append({
                    "state": s, "year": yr, "year_idx": y_idx,
                    "A": A, "income_low": income_low[i],
                    "age": age[i], "poor_health": poor_health[i],
                    "expand_year": ey
                })

    return pd.DataFrame(rows)


def gformula_brfss(df: pd.DataFrame, M: int = 15_000, B: int = 200):
    """
    Applies g-formula to BRFSS panel.
    Simplified to two periods: pre-expansion (t=0, year<2014)
    and post-expansion (t=1, year>=2014).
    Covariate: income_low (mediator/confounder).
    Strategy 1: A_t = 1 for all t (sustained coverage)
    Strategy 0: A_t = 0 for all t (no coverage)
    """
    # Split into pre and post
    pre  = df[df["year"] < 2014].copy()
    post = df[df["year"] >= 2014].copy()

    # ── Covariate model: income_low at t=1 | A_t=0 period, income_low_t0, age
    # Use pre-period income and age to predict post-period income_low
    # Merge state-level means
    pre_agg  = pre.groupby("state")[["income_low","age"]].mean().rename(
                   columns={"income_low":"il0","age":"age0"})
    post_agg = post.groupby("state")[["income_low","A"]].mean().rename(
                   columns={"income_low":"il1","A":"A1"})
    panel = pre_agg.join(post_agg).dropna().reset_index()

    scaler = StandardScaler()
    X_il = scaler.fit_transform(panel[["il0","age0","A1"]])
    m_il1 = LinearRegression().fit(X_il, panel["il1"])
    il1_resid = panel["il1"] - m_il1.predict(X_il)
    sigma_il = il1_resid.std()

    # ── Outcome model: logistic on post period
    X_out = post[["A","income_low","age"]].copy()
    X_out_sc = scaler.fit_transform(X_out)
    m_out = LogisticRegression(max_iter=500).fit(X_out_sc, post["poor_health"])
    out_scaler = scaler  # reuse fitted scaler

    # ── Monte Carlo g-computation ──────────────────────────────
    rng = np.random.default_rng(777)

    def _single_run(panel_data, rng_obj, strategy_A0, strategy_A1, M_sim):
        idx = rng_obj.integers(0, len(panel_data), M_sim)
        sim = panel_data.iloc[idx][["il0","age0"]].copy().reset_index(drop=True)

        # Set treatment at t=0
        sim["A0"] = float(strategy_A0)
        sim["A1_val"] = float(strategy_A1)  # t=1 treatment

        # Simulate income_low at t=1 under strategy
        X_il_sim = scaler.fit_transform(
            np.column_stack([sim["il0"], sim["age0"], sim["A1_val"]])
        )
        # NOTE: use the originally fitted m_il1 here, not refitted
        mu_il1 = m_il1.predict(
            StandardScaler().fit(panel[["il0","age0","A1"]]).transform(
                np.column_stack([sim["il0"], sim["age0"], sim["A1_val"]])
            )
        )
        sim["il1_sim"] = mu_il1 + rng_obj.normal(0, sigma_il, M_sim)

        # Predict outcome
        age_sim = sim["age0"] + 4   # ~4-year gap
        X_out_sim = np.column_stack([sim["A1_val"], sim["il1_sim"], age_sim])
        X_out_sim_sc = out_scaler.transform(X_out_sim)
        probs = m_out.predict_proba(X_out_sim_sc)[:, 1]
        return probs.mean()

    # Fit scalers on full panel for consistent use below
    out_scaler_full = StandardScaler().fit(post[["A","income_low","age"]])
    il_scaler_full  = StandardScaler().fit(panel[["il0","age0","A1"]])
    m_il1_full = LinearRegression().fit(
        il_scaler_full.transform(panel[["il0","age0","A1"]]), panel["il1"]
    )
    m_out_full = LogisticRegression(max_iter=500).fit(
        out_scaler_full.transform(post[["A","income_low","age"]]),
        post["poor_health"]
    )

    def gf_estimate(df_in, A0_val, A1_val, M_sim=M, rseed=0):
        rng2 = np.random.default_rng(rseed)
        pre_g  = df_in[df_in["year"] < 2014].groupby("state")[["income_low","age"]].mean().rename(
                    columns={"income_low":"il0","age":"age0"})
        post_g = df_in[df_in["year"] >= 2014].groupby("state")[["income_low","A"]].mean().rename(
                    columns={"income_low":"il1","A":"A1"})
        panel_g = pre_g.join(post_g).dropna().reset_index()

        il_sc  = StandardScaler().fit(panel_g[["il0","age0","A1"]])
        m_il   = LinearRegression().fit(il_sc.transform(panel_g[["il0","age0","A1"]]), panel_g["il1"])
        r_il   = panel_g["il1"] - m_il.predict(il_sc.transform(panel_g[["il0","age0","A1"]]))
        s_il   = r_il.std()

        post_g2  = df_in[df_in["year"] >= 2014].copy()
        out_sc   = StandardScaler().fit(post_g2[["A","income_low","age"]])
        m_out_g  = LogisticRegression(max_iter=500).fit(
                       out_sc.transform(post_g2[["A","income_low","age"]]),
                       post_g2["poor_health"])

        idx  = rng2.integers(0, len(panel_g), M_sim)
        sim  = panel_g.iloc[idx][["il0","age0"]].copy().reset_index(drop=True)
        A1_c = np.full(M_sim, A1_val)
        X_il_s = il_sc.transform(np.column_stack([sim["il0"], sim["age0"], A1_c]))
        il1_s  = m_il.predict(X_il_s) + rng2.normal(0, s_il, M_sim)
        age_s  = sim["age0"].values + 4
        X_o_s  = out_sc.transform(np.column_stack([A1_c, il1_s, age_s]))
        return m_out_g.predict_proba(X_o_s)[:, 1].mean()

    E1 = gf_estimate(df, A0_val=1, A1_val=1, rseed=1)
    E0 = gf_estimate(df, A0_val=0, A1_val=0, rseed=2)

    # Bootstrap
    boot_ates = []
    for b in range(B):
        states_b = np.random.default_rng(b).choice(
            df["state"].unique(), size=df["state"].nunique(), replace=True)
        boot_df = pd.concat([df[df["state"] == s] for s in states_b],
                             ignore_index=True)
        e1b = gf_estimate(boot_df, 1, 1, M_sim=3000, rseed=b)
        e0b = gf_estimate(boot_df, 0, 0, M_sim=3000, rseed=b+5000)
        boot_ates.append(e1b - e0b)

    ate        = E1 - E0
    boot_se    = float(np.std(boot_ates))
    ci_lo, ci_hi = ate - 1.96*boot_se, ate + 1.96*boot_se

    return {
        "E[Y(always cover)]": E1,
        "E[Y(never cover)]":  E0,
        "ATE":                ate,
        "Bootstrap SE":       boot_se,
        "95% CI":             (ci_lo, ci_hi),
        "Empirical poor health (post)": df[df["year"]>=2014]["poor_health"].mean()
    }


print("\n" + "=" * 55)
print("BRFSS / ACA MEDICAID EXPANSION — G-Formula")
print("=" * 55)

df_brfss  = simulate_brfss_panel(n_states=50, n_years=6, n_per_state_year=200)
results   = gformula_brfss(df_brfss, M=10_000, B=200)

print(f"E[Y | always covered]     = {results['E[Y(always cover)]']:.4f}")
print(f"E[Y | never covered]      = {results['E[Y(never cover)]']:.4f}")
print(f"ATE (poor health risk)    = {results['ATE']:.4f}")
print(f"Bootstrap SE              = {results['Bootstrap SE']:.4f}")
print(f"95% CI                    = [{results['95% CI'][0]:.4f}, {results['95% CI'][1]:.4f}]")
print(f"Empirical mean (post)     = {results['Empirical poor health (post)']:.4f}")
print(f"\nNatural course check: g-formula under observed A should approach empirical mean.")
print(f"Discrepancy indicates covariate model misspecification or positivity violations.")
```

## Summary

- The g-formula identifies $E[Y(\bar{a})]$ as a weighted average of the conditional outcome mean over covariate histories, where the weights are the product of observed conditional covariate distributions — not the observed joint distribution of covariates given the treatment sequence actually received.

- Identification requires sequential ignorability (no unmeasured time-varying confounding) and positivity (every treatment value has positive probability at every covariate history); both must hold at every time point, not just at baseline.

- Static interventions fix a predetermined treatment sequence; dynamic interventions condition treatment at each period on the evolving covariate history. The g-formula handles both with the same identification argument.

- Monte Carlo g-computation simulates covariate trajectories forward through estimated component models under the hypothetical intervention, then predicts outcomes; bootstrap standard errors propagate uncertainty from all component model estimation steps jointly.

- The natural course check — comparing g-formula under observed treatment with the empirical outcome mean — is a necessary diagnostic for covariate model misspecification; systematic failure of this check invalidates the intervened estimates.

- Sequential misspecification bias is the central practical vulnerability: errors in early covariate models compound over time periods, and neither correct specification of the outcome model alone nor high-quality treatment models (as in IPTW) is sufficient to rescue a g-formula with misspecified covariate evolution.

- The g-formula and IPTW-based MSMs (Chapter 27) have complementary model requirements: the g-formula needs covariate evolution models but not the treatment mechanism, while IPTW needs the treatment mechanism but not the covariate evolution; doubly-robust estimators in Chapters 29–30 exploit this complementarity to achieve consistency under either.

## Further Reading

- **Robins (1986)**, "A new approach to causal inference in mortality studies with a sustained exposure period." *Mathematical Modelling* 7:1393–1512. The original paper introducing the g-formula; establishes the identification theorem from first principles and defines the g-computation algorithm. Essential primary source.

- **Hernán and Robins (2020)**, *Causal Inference: What If*, Chapters 12–13. Free online. Comprehensive treatment of time-varying treatments with extensive notation alignment to the potential outcomes framework; Chapter 13 covers parametric g-formula implementation in detail with SAS/R code translatable to Python.

- **Robins, Hernán, and Brumback (2000)**, "Marginal structural models and causal inference in epidemiology." *Epidemiology* 11:550–560. Introduces MSMs as an alternative to the g-formula and explicitly contrasts the two estimators' model requirements; reading this alongside Chapter 27 clarifies when each approach is preferred.

- **Daniel et al. (2013)**, "Methods for dealing with time-dependent confounding." *Statistics in Medicine* 32:1584–1618. Systematic review comparing g-formula, IPTW, and g-estimation across simulation settings; the Monte Carlo sensitivity analyses demonstrate the compounding bias problem under sequential misspecification quantitatively.

- **Keil et al. (2014)**, "The parametric g-formula for time-to-event data: towards intuition with a worked example." *Epidemiology* 25:889–897. Worked clinical example with survival outcomes extending the continuous-outcome framework in this chapter; bridges to competing risks and time-to-event analogs of the g-formula.

- **van der Laan and Rose (2011)**, *Targeted Learning*, Chapter 6. Develops the influence function of the g-formula functional and its connection to TMLE; provides the theoretical foundation for doubly-robust extensions of the parametric g-formula covered in Chapter 30.
# Chapter 45: From ATE to ROI — Decision-Theoretic Causal Inference

Causal inference produces estimates: point estimates, confidence intervals, posterior distributions over treatment effects. But a policymaker or business analyst does not act on an estimate—they act on a *decision*. Should we expand Medicaid to the remaining uninsured population? Should we deploy the intervention at scale? Should we run another experiment before committing the budget?

The translation from causal estimate to decision requires an additional layer: a specification of costs, benefits, constraints, and preferences over outcomes. Decision-theoretic causal inference provides that layer. This chapter develops the machinery rigorously, grounding every concept in the Oregon Health Insurance Experiment (OHE) and ACA Medicaid expansion.

---

## 45.1 The Decision-Theoretic Framework

Let $\tau$ denote the causal estimand of interest—the ATE, LATE, or CATE depending on context. After estimation we have either a frequentist confidence interval $[\hat{\tau}_L, \hat{\tau}_U]$ or a posterior $p(\tau \mid \text{data})$. The question is: given this knowledge state, which action $d \in \mathcal{D}$ should we take?

**Definition 45.1 (Utility function).** A utility function $U: \mathcal{D} \times \mathbb{R} \to \mathbb{R}$ maps an action and the true state of the world (here, the true $\tau$) to a real-valued payoff. The decision-maker maximizes expected utility.

For policy problems, $\mathcal{D}$ is typically binary: expand the program or do not. For budget allocation problems, $d$ may be a continuous expenditure level or a targeting rule $\pi: \mathcal{X} \to [0,1]$.

**The policy value function.** Suppose the intervention assigns treatment according to policy $\pi(x) = P(D=1 \mid X=x)$. Each treated individual generates benefit $B$ (measured in dollars, QALYs, or any common unit) with probability $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]$ per unit of the outcome scale. The cost of treating one individual is $C$. The policy value is:

$$\text{ROI}(\pi) = \mathbb{E}_X\bigl[B \cdot \tau(X) \cdot \pi(X)\bigr] - C \cdot \mathbb{E}_X[\pi(X)]$$

The first term is expected benefit—the average causal effect scaled by the benefit per unit, weighted by the probability of treatment. The second term is expected cost. Net benefit is positive when benefit exceeds cost.

For a flat policy (treat everyone, or expand universally), $\pi(x) = 1$ and:

$$\text{ROI}(1) = B \cdot \text{ATE} - C$$

This is positive if and only if $\text{ATE} > C/B$. This ratio defines the **threshold treatment effect**.

**Definition 45.2 (Minimum Viable Policy Delta, MVPD).** The MVPD is the break-even treatment effect:

$$\tau^* = \frac{C}{B}$$

A policy is expected to be net-beneficial if and only if $\tau > \tau^*$.

The MVPD is not a statistical threshold—it does not depend on sample size or estimation uncertainty. It is a structural object determined entirely by the cost-benefit ratio. The decision question is then: does our causal estimate exceed $\tau^*$?

---

## 45.2 Applying the Framework to the Oregon Health Insurance Experiment

The OHE LATE estimates the effect of Medicaid enrollment (among lottery winners who enrolled) on outcomes including catastrophic medical expenditure. We operationalize the framework as follows.

**Outcome.** Let $Y = \mathbf{1}[\text{catastrophic expenditure}]$, where catastrophic is defined as out-of-pocket expenditure exceeding 30% of annual income. The LATE from the OHE IV regression (instrument: lottery selection $Z$; treatment: Medicaid enrollment $D$) gives $\hat{\tau}_{IV} \approx -0.032$ with standard error $\approx 0.012$. Medicaid *reduces* the probability of catastrophic expenditure by approximately 3.2 percentage points among compliers.

**Benefit calibration.** A catastrophic expenditure event imposes both direct financial cost and health disruption. CMS estimates the average out-of-pocket cost in a catastrophic event at roughly \$3,200 (conditional on the event occurring). The benefit of prevention is therefore $B \approx \$3{,}200$ per prevented event.

**Cost calibration.** Annual Medicaid enrollment costs (state + federal) averaged approximately \$5,000 per adult enrollee in 2010 dollars. Hence $C = \$5{,}000$.

**MVPD.** The break-even treatment effect is:

$$\tau^* = \frac{C}{B} = \frac{5{,}000}{3{,}200} \approx 0.156$$

That is, Medicaid enrollment would need to reduce catastrophic expenditure by 15.6 percentage points to break even on *this outcome alone*. The OHE estimate of 3.2 percentage points falls far short. This is not a failure of causal inference—it is an honest accounting of what the evidence supports.

The inference changes when we broaden the benefit measure. Medicaid also reduces emergency department utilization, improves mental health outcomes, and generates macroeconomic spillovers through reduced medical debt. A comprehensive benefit tally could substantially raise $B$, pulling $\tau^*$ below the observed LATE. The decision-theoretic framework makes this dependence explicit: policy conclusions are sensitive to the benefit specification, not just the causal estimate.

---

## 45.3 Propagating Uncertainty to the Decision

The estimate $\hat{\tau}$ carries uncertainty. A point estimate above $\tau^*$ does not guarantee that the true $\tau$ exceeds $\tau^*$. We need to propagate the estimation uncertainty through to the decision.

**Frequentist approach: probability of cost-effectiveness.** Assuming asymptotic normality, $\hat{\tau} \sim N(\tau, \sigma^2)$ where $\sigma$ is estimated from the data. The probability that the true effect exceeds the threshold is:

$$P(\tau > \tau^*) = P\left(Z > \frac{\tau^* - \hat{\tau}}{\hat{\sigma}}\right) = 1 - \Phi\left(\frac{\tau^* - \hat{\tau}}{\hat{\sigma}}\right)$$

where $\Phi$ is the standard normal CDF. This is the **probability of cost-effectiveness** (PCE)—a routine output in health economic evaluations.

For the OHE catastrophic expenditure outcome with the narrow benefit definition: $\hat{\tau} = -0.032$, $\hat{\sigma} = 0.012$, $\tau^* = 0.156$. Note $\tau^*$ is positive (reduction) while $\hat{\tau}$ is negative (indicating benefit), so we reframe: define $\delta = -\tau$ as the reduction in catastrophic expenditure probability. Then $\hat{\delta} = 0.032$, $\hat{\sigma}_\delta = 0.012$, and $\delta^* = 0.156$.

$$\text{PCE} = P(\delta > \delta^*) = 1 - \Phi\left(\frac{0.156 - 0.032}{0.012}\right) = 1 - \Phi(10.3) \approx 0$$

The data are essentially certain that the catastrophic expenditure effect alone does not justify the cost. This is a substantive finding, not a statistical artifact.

**Bayesian approach: posterior expected utility.** Place a prior $p(\tau)$ on the treatment effect, update on the data to obtain $p(\tau \mid \text{data})$, and compute the expected utility of each action:

$$\mathbb{E}[U(\text{expand}) \mid \text{data}] = \int U(\text{expand}, \tau) \, p(\tau \mid \text{data}) \, d\tau$$

$$\mathbb{E}[U(\text{status quo}) \mid \text{data}] = \int U(\text{status quo}, \tau) \, p(\tau \mid \text{data}) \, d\tau$$

With $U(\text{expand}, \tau) = B\tau - C$ and $U(\text{status quo}, \tau) = 0$:

$$\mathbb{E}[U(\text{expand}) \mid \text{data}] = B \cdot \mathbb{E}[\tau \mid \text{data}] - C$$

Under a conjugate normal-normal model with diffuse prior, the posterior mean equals $\hat{\tau}$ and the decision reduces to the point-estimate comparison. The Bayesian framework adds value when we have genuine prior information—for instance, meta-analytic evidence from other Medicaid expansions—or when we wish to reason about *distributions* over outcomes rather than expectations.

---

## 45.4 The Cost-Effectiveness Plane and Incremental Analysis

Health economics standardizes this reasoning through the **cost-effectiveness plane**. The horizontal axis is the incremental effect $\Delta E = \hat{\tau}$ (change in outcome per enrollee); the vertical axis is the incremental cost $\Delta C$ (cost per enrollee). Each bootstrap draw or posterior sample produces a point in this plane.

**Definition 45.3 (Incremental Cost-Effectiveness Ratio).** The ICER is:

$$\text{ICER} = \frac{\Delta C}{\Delta E} = \frac{C}{\tau}$$

A policy is cost-effective at willingness-to-pay threshold $\lambda$ if $\text{ICER} < \lambda$, equivalently if $\tau > C/\lambda = \tau^*$.

The **net monetary benefit** (NMB) combines cost and effect on a single scale:

$$\text{NMB}(\lambda) = \lambda \cdot \Delta E - \Delta C = \lambda \tau - C$$

NMB is positive (cost-effective) iff $\tau > C/\lambda$. Plotting NMB against $\lambda$ yields the **net benefit curve**, which crosses zero at $\lambda = C/\tau = \text{ICER}$.

The **cost-effectiveness acceptability curve** (CEAC) plots the probability that NMB $> 0$ as a function of $\lambda$:

$$\text{CEAC}(\lambda) = P\left(\tau > \frac{C}{\lambda}\right) = 1 - \Phi\left(\frac{C/\lambda - \hat{\tau}}{\hat{\sigma}}\right)$$

As $\lambda \to \infty$, CEAC $\to 1$ (any positive effect is cost-effective at infinite willingness-to-pay). As $\lambda \to 0$, CEAC $\to P(\tau > \infty) = 0$.

---

## 45.5 Expected Value of Information

The PCE and CEAC characterize uncertainty about the current decision. A distinct and often more actionable question is: how much would we gain by reducing that uncertainty? This is the **expected value of information** (EVOI).

**Definition 45.4 (Expected Value of Perfect Information).** The EVPI is:

$$\text{EVPI} = \mathbb{E}_\tau\left[\max_d U(d, \tau)\right] - \max_d \mathbb{E}_\tau\left[U(d, \tau)\right]$$

The first term is what we would earn if we knew $\tau$ before deciding. The second term is what we earn under the current optimal decision. EVPI is always non-negative by Jensen's inequality (since $\max$ is convex).

**Theorem 45.1 (EVPI decomposition).** For a binary decision with $U(\text{expand}, \tau) = B\tau - C$ and $U(\text{status quo}, \tau) = 0$:

$$\text{EVPI} = B \cdot \mathbb{E}_\tau[\max(\tau - \tau^*, 0)] - \max(B \cdot \mathbb{E}[\tau] - C, 0)$$

*Proof sketch.* Under perfect information the decision-maker expands iff $\tau > \tau^*$, yielding utility $\max(B\tau - C, 0)$. In expectation this is $\mathbb{E}[\max(B\tau - C, 0)] = B \cdot \mathbb{E}[\max(\tau - \tau^*, 0)]$. Without perfect information, utility is $\max(B\mathbb{E}[\tau] - C, 0)$. EVPI is the difference. $\square$

Under normal uncertainty $\tau \mid \text{data} \sim N(\mu, \sigma^2)$:

$$\mathbb{E}[\max(\tau - \tau^*, 0)] = (\mu - \tau^*)\Phi\left(\frac{\mu - \tau^*}{\sigma}\right) + \sigma \phi\left(\frac{\mu - \tau^*}{\sigma}\right)$$

where $\phi$ is the standard normal PDF. This is the **partial expectation** or **linear loss function** from Bayesian decision theory.

**Interpretation.** EVPI represents the maximum a policymaker should pay for a study that perfectly reveals $\tau$. If EVPI is low, existing evidence is sufficient and further research adds little. If EVPI is high, uncertainty is decision-relevant and investment in additional evidence is justified.

**Expected value of sample information (EVSI).** A more practical quantity is EVSI: the value of running a new study of sample size $n$ before deciding.

**Proposition 45.1.** For a new study with estimated standard error $\sigma_n = \sigma / \sqrt{n}$ (likelihood) and current posterior variance $\sigma^2$, the posterior variance after the new study is:

$$\sigma^2_{\text{post}} = \left(\frac{1}{\sigma^2} + \frac{n}{\sigma^2_{\text{lik}}}\right)^{-1}$$

EVSI$(n) = \text{EVPI}(\sigma^2_{\text{post}}) \times P(\text{current decision changes})$, computed by integrating over the predictive distribution of the new study's result. EVSI is increasing in $n$ and bounded above by EVPI.

---

## 45.6 Risk-Averse Decision Making

Expected utility maximization treats a 50% chance of $+\$200$ the same as a certain $+\$100$. Risk-averse policymakers—particularly those managing public budgets—may not. The decision-theoretic framework accommodates this through non-linear utility.

**Quadratic loss and certainty equivalents.** A risk-averse policymaker with quadratic utility $U(w) = w - \frac{\alpha}{2}w^2$ has certainty equivalent:

$$\text{CE} = \mathbb{E}[W] - \frac{\alpha}{2}\text{Var}(W)$$

For the expansion decision with stochastic net benefit $W = B\tau - C$:

$$\text{CE} = (B\hat{\tau} - C) - \frac{\alpha}{2}B^2 \hat{\sigma}^2$$

The risk penalty $\frac{\alpha}{2}B^2\hat{\sigma}^2$ grows with estimation uncertainty $\hat{\sigma}^2$ and with benefit scale $B$. A more uncertain estimate is penalized more heavily for a risk-averse decision maker, which provides additional motivation for reducing $\hat{\sigma}^2$ through larger studies.

**Threshold under risk aversion.** The certainty equivalent is positive iff:

$$\hat{\tau} > \tau^* + \frac{\alpha}{2}B\hat{\sigma}^2$$

The risk-adjusted MVPD is $\tau^*_\alpha = \tau^* + \frac{\alpha}{2}B\hat{\sigma}^2$. More uncertainty raises the bar for action, which formalizes the intuition that one should require stronger evidence before committing a large budget.

---

## 45.7 Heterogeneous Treatment Effects and Targeted Policies

When CATE estimates $\hat{\tau}(x)$ are available (from Chapters 20–22), the optimal policy maximizes expected net benefit subject to a budget constraint.

**Theorem 45.2 (Optimal targeting rule).** Given budget $\bar{C}$ (total treatment cost) and CATE estimates $\hat{\tau}(x)$, the budget-constrained optimal policy is a threshold rule:

$$\pi^*(x) = \mathbf{1}[\hat{\tau}(x) > \tau^*_\lambda]$$

where $\tau^*_\lambda$ is chosen so that $C \cdot \mathbb{E}[\pi^*(X)] = \bar{C}$.

*Proof.* The Lagrangian for $\max_\pi \mathbb{E}[B\hat{\tau}(X)\pi(X)] - \lambda(C\mathbb{E}[\pi(X)] - \bar{C})$ is pointwise maximized by treating individual $x$ iff $B\hat{\tau}(x) > \lambda C$, i.e., $\hat{\tau}(x) > \lambda C / B$. Setting $\lambda$ to clear the budget constraint gives the threshold $\tau^*_\lambda = \lambda C/B$. $\square$

In the ACA context, this theorem motivates differential expansion strategies: states with higher baseline uninsurance rates and lower income populations have higher expected CATE, so a budget-constrained planner would prioritize those states. The policy value of optimal targeting over uniform expansion is:

$$\text{Value of targeting} = \text{ROI}(\pi^*) - \text{ROI}(1) = B \cdot \mathbb{E}[(\hat{\tau}(X) - \text{ATE})\mathbf{1}[\hat{\tau}(X) > \tau^*_\lambda]]$$

which is non-negative when $\tau^*_\lambda > \text{ATE}$ (budget-constrained regime). This quantifies the economic value of heterogeneity estimation.

---

## Python: Decision-Theoretic Analysis of the Oregon Health Insurance Experiment

```python
"""
Chapter 45: Decision-Theoretic Causal Inference
Oregon Health Insurance Experiment — Cost-Effectiveness Analysis

Requires: numpy, scipy, pandas, matplotlib, statsmodels
Data: https://data.nber.org/oregon/
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr  # fast normal CDF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── 1. Load OHE data and reproduce the IV estimate ───────────────────────────
# The NBER replication data; adjust path as needed.
DATA_DIR = Path("~/data/oregon").expanduser()

try:
    # 12-month survey data (the primary OHE analysis file)
    df = pd.read_stata(DATA_DIR / "oregonhie_survey12m_vars.dta")

    # Core variables
    # Z: selected in lottery (intent-to-treat instrument)
    # D: ohp_all_ever_admin (ever enrolled in Medicaid by 12m)
    # Y: catastrophic_exp_inp (any catastrophic inpatient expenditure)
    # W: numhh_list (lottery-list household size, used for strata/weights)

    df = df.dropna(subset=["selected", "ohp_all_ever_admin", "catastrophic_exp_inp"])

    Z = df["selected"].values
    D = df["ohp_all_ever_admin"].values
    Y = df["catastrophic_exp_inp"].values

    # IV estimate via 2SLS: first stage, reduced form, LATE
    import statsmodels.formula.api as smf
    from linearmodels.iv import IV2SLS

    df_iv = df[["catastrophic_exp_inp", "ohp_all_ever_admin", "selected",
                "numhh_list"]].dropna().copy()
    df_iv.columns = ["Y", "D", "Z", "strata"]

    # Add household-size dummies (controls as in the original paper)
    df_iv = pd.get_dummies(df_iv, columns=["strata"], drop_first=True)
    strata_cols = [c for c in df_iv.columns if c.startswith("strata_")]
    controls = " + ".join(strata_cols)

    formula = f"Y ~ 1 + {controls} + [D ~ Z + {controls}]"
    res_iv = IV2SLS.from_formula(formula, data=df_iv).fit(cov_type="robust")

    tau_hat = res_iv.params["D"]
    tau_se  = res_iv.std_errors["D"]
    tau_ci  = res_iv.conf_int().loc["D"].values
    print(f"IV LATE: {tau_hat:.4f}  SE: {tau_se:.4f}  "
          f"95% CI: [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]")

except FileNotFoundError:
    # ── Fallback: use published OHE estimates from Finkelstein et al. (2012) ──
    print("OHE data not found — using published estimates.")
    # LATE on catastrophic expenditure (Table 9, col 4 of Finkelstein et al.)
    # Negative = Medicaid reduces catastrophic expenditure
    tau_hat = -0.032
    tau_se  = 0.012
    tau_ci  = np.array([tau_hat - 1.96 * tau_se, tau_hat + 1.96 * tau_se])

# ── 2. Cost-benefit parameters ────────────────────────────────────────────────
# Reframe: delta = reduction in catastrophic expenditure probability
# Positive delta = beneficial
delta_hat = -tau_hat          # flip sign: positive = reduction
delta_se  = tau_se
delta_ci  = -tau_ci[::-1]    # flip and reverse

# Cost of Medicaid per enrollee per year (2010 USD, CMS estimates)
C = 5_000.0

# Benefit per prevented catastrophic event (avg out-of-pocket cost averted)
# We compute across a range for sensitivity analysis
B_base = 3_200.0             # narrow: direct financial cost only
B_wide = 12_000.0            # broad: includes health, mental health, debt relief

# MVPD at each benefit estimate
mvpd_base = C / B_base       # ≈ 0.156
mvpd_wide = C / B_wide       # ≈ 0.042

print(f"\nEstimated reduction in catastrophic expenditure: {delta_hat:.4f} "
      f"({delta_hat*100:.2f} pp)")
print(f"MVPD (narrow B = ${B_base:,.0f}): {mvpd_base:.4f}")
print(f"MVPD (broad  B = ${B_wide:,.0f}): {mvpd_wide:.4f}")

# ── 3. Probability of cost-effectiveness (PCE) ───────────────────────────────
pce_base = 1 - ndtr((mvpd_base - delta_hat) / delta_se)
pce_wide = 1 - ndtr((mvpd_wide - delta_hat) / delta_se)

print(f"\nProbability of cost-effectiveness:")
print(f"  Narrow benefits (B=${B_base:,.0f}): PCE = {pce_base:.4f}")
print(f"  Broad benefits  (B=${B_wide:,.0f}): PCE = {pce_wide:.4f}")

# ── 4. Bootstrap draws for cost-effectiveness plane ──────────────────────────
rng = np.random.default_rng(42)
n_boot = 20_000

delta_draws = rng.normal(loc=delta_hat, scale=delta_se, size=n_boot)
cost_draws  = rng.normal(loc=C, scale=C * 0.05, size=n_boot)   # 5% cost uncertainty

incremental_cost   = cost_draws
incremental_effect = delta_draws                                 # pp reduction
nmb_draws_base = B_base * incremental_effect - incremental_cost
nmb_draws_wide = B_wide * incremental_effect - incremental_cost

# ── 5. EVPI calculation ───────────────────────────────────────────────────────
def evpi_normal(mu: float, sigma: float, threshold: float, B: float) -> float:
    """
    EVPI for binary expand/no-expand decision with linear utility.
    
    Under perfect info: expand iff delta > threshold, gaining B*delta - C.
    Under current info: expand iff B*mu > C, gaining max(B*mu - C, 0).
    """
    z = (mu - threshold) / sigma
    # E[max(delta - threshold, 0)] = (mu - threshold)*Phi(z) + sigma*phi(z)
    partial_exp = (mu - threshold) * ndtr(z) + sigma * stats.norm.pdf(z)
    evpi_val = B * partial_exp - max(B * mu - C, 0)
    return max(evpi_val, 0.0)   # EVPI is non-negative

evpi_base = evpi_normal(delta_hat, delta_se, mvpd_base, B_base)
evpi_wide = evpi_normal(delta_hat, delta_se, mvpd_wide, B_wide)

print(f"\nEVPI (narrow benefits): ${evpi_base:,.2f} per enrollee")
print(f"EVPI (broad  benefits): ${evpi_wide:,.2f} per enrollee")

# ── 6. EVSI as a function of additional sample size ──────────────────────────
def evsi_normal(mu, sigma_current, sigma_study_unit, n, threshold, B, C,
                n_mc=10_000, rng=None):
    """
    EVSI via simulation: draw predictive study means, compute posterior,
    evaluate gain from optimal post-study decision vs. current decision.
    
    sigma_study_unit: SE of effect estimate from n=1 study.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sigma_n = sigma_study_unit / np.sqrt(n)                  # study SE at size n
    prior_var = sigma_current ** 2

    # Predictive mean of new study ~ N(mu, sigma_current^2 + sigma_n^2)
    sigma_pred = np.sqrt(prior_var + sigma_n ** 2)
    x_star = rng.normal(loc=mu, scale=sigma_pred, size=n_mc)

    # Posterior after observing x_star
    post_var = 1 / (1/prior_var + 1/sigma_n**2)
    post_mu  = post_var * (mu/prior_var + x_star/sigma_n**2)
    post_sd  = np.sqrt(post_var)

    # Optimal action for each hypothetical study outcome
    utility_expand    = B * post_mu - C
    utility_status_quo = np.zeros(n_mc)
    utility_optimal   = np.maximum(utility_expand, utility_status_quo)

    # Current optimal action (fixed)
    current_best = max(B * mu - C, 0)

    evsi = np.mean(utility_optimal) - current_best
    return max(evsi, 0.0)

ns = np.array([100, 250, 500, 1000, 2000, 5000, 10000, 20000])
evsi_vals_base = np.array([
    evsi_normal(delta_hat, delta_se, delta_se * np.sqrt(len(df_iv) if 'df_iv' in dir() else 12000),
                n, mvpd_base, B_base, C, rng=rng)
    for n in ns
])
evsi_vals_wide = np.array([
    evsi_normal(delta_hat, delta_se, delta_se * np.sqrt(len(df_iv) if 'df_iv' in dir() else 12000),
                n, mvpd_wide, B_wide, C, rng=rng)
    for n in ns
])

# ── 7. Cost-effectiveness acceptability curve (CEAC) ─────────────────────────
lambdas = np.linspace(0, 30_000, 500)   # willingness-to-pay range $0–$30k

def ceac(lam, delta_hat, delta_se, C):
    threshold = C / np.where(lam > 0, lam, 1e-10)
    return 1 - ndtr((threshold - delta_hat) / delta_se)

ceac_vals = ceac(lambdas, delta_hat, delta_se, C)

# ── 8. Risk-adjusted MVPD ─────────────────────────────────────────────────────
alphas = np.linspace(0, 1e-7, 200)    # risk aversion coefficient (small: near risk neutral)
mvpd_risk_adjusted = mvpd_wide + 0.5 * alphas * B_wide * delta_se**2

# ── 9. Plotting ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)

# Panel A: Cost-effectiveness plane
ax_a = fig.add_subplot(gs[0, 0])
sc = ax_a.scatter(incremental_effect * 100, incremental_cost / 1000,
                  alpha=0.08, s=2, c=nmb_draws_wide, cmap="RdYlGn",
                  vmin=-C, vmax=C, rasterized=True)
ax_a.axvline(mvpd_wide * 100, color="steelblue", lw=1.5, ls="--",
             label=f"MVPD (broad B) = {mvpd_wide*100:.1f} pp")
ax_a.axvline(mvpd_base * 100, color="tomato",   lw=1.5, ls="--",
             label=f"MVPD (narrow B) = {mvpd_base*100:.1f} pp")
ax_a.axvline(delta_hat * 100, color="black",     lw=2,
             label=f"LATE = {delta_hat*100:.1f} pp")
ax_a.axhline(C / 1000, color="gray", lw=0.8, ls=":")
ax_a.set_xlabel("Incremental Effect (pp reduction in catastrophic exp.)")
ax_a.set_ylabel("Incremental Cost ($000s/enrollee)")
ax_a.set_title("A  Cost-Effectiveness Plane")
ax_a.legend(fontsize=7, loc="upper right")
plt.colorbar(sc, ax=ax_a, label="NMB (broad B, $)", shrink=0.8)

# Panel B: CEAC
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(lambdas / 1000, ceac_vals * 100, color="steelblue", lw=2)
ax_b.axhline(50, color="gray", lw=0.8, ls="--")
ax_b.axvline(B_wide / 1000, color="tomato", lw=1.5, ls="--",
             label=f"B (broad) = ${B_wide/1000:.0f}k")
ax_b.set_xlabel("Willingness-to-Pay Threshold λ ($000s per prevented event)")
ax_b.set_ylabel("Probability Cost-Effective (%)")
ax_b.set_title("B  Cost-Effectiveness Acceptability Curve")
ax_b.set_ylim(0, 105)
ax_b.legend(fontsize=8)

# Panel C: EVSI vs. sample size
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(ns, evsi_vals_wide, "o-", color="steelblue", label="Broad B")
ax_c.plot(ns, evsi_vals_base, "s--", color="tomato",   label="Narrow B")
ax_c.axhline(evpi_wide, color="steelblue", lw=0.8, ls=":", alpha=0.6,
             label=f"EVPI (broad) = ${evpi_wide:.2f}")
ax_c.axhline(evpi_base, color="tomato",   lw=0.8, ls=":", alpha=0.6,
             label=f"EVPI (narrow) ≈ ${evpi_base:.2f}")
ax_c.set_xscale("log")
ax_c.set_xlabel("Additional Sample Size")
ax_c.set_ylabel("EVSI ($ per enrollee)")
ax_c.set_title("C  Expected Value of Sample Information")
ax_c.legend(fontsize=7)

# Panel D: Break-even threshold diagram with risk adjustment
ax_d = fig.add_subplot(gs[1, 1])
ax_d.plot(alphas * 1e7, mvpd_risk_adjusted * 100, color="steelblue", lw=2,
          label="Risk-adjusted MVPD (broad B)")
ax_d.axhline(mvpd_wide * 100, color="tomato", lw=1.5, ls="--",
             label=f"Risk-neutral MVPD = {mvpd_wide*100:.1f} pp")
ax_d.axhline(delta_hat * 100, color="black", lw=2,
             label=f"LATE = {delta_hat*100:.1f} pp")
ax_d.fill_between(alphas * 1e7, delta_hat * 100, mvpd_risk_adjusted * 100,
                  alpha=0.15, color="steelblue", label="Uncertainty penalty region")
ax_d.set_xlabel("Risk Aversion Coefficient α (×10⁻⁷)")
ax_d.set_ylabel("Required Effect Size (pp)")
ax_d.set_title("D  Risk-Adjusted Break-Even Threshold")
ax_d.legend(fontsize=7)

plt.suptitle(
    "Oregon Health Insurance Experiment — Decision-Theoretic Analysis\n"
    "Catastrophic Expenditure Outcome: LATE ≈ −3.2 pp, SE ≈ 1.2 pp",
    fontsize=11, y=1.01
)
plt.savefig("ch45_decision_theoretic.pdf", bbox_inches="tight", dpi=150)
plt.show()

# ── 10. Summary table ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DECISION-THEORETIC SUMMARY — OHE Catastrophic Expenditure")
print("="*60)
print(f"{'Parameter':<40} {'Narrow B':>10} {'Broad B':>10}")
print("-"*60)
print(f"{'Benefit per prevented event B ($)':<40} {B_base:>10,.0f} {B_wide:>10,.0f}")
print(f"{'Cost per enrollee C ($)':<40} {C:>10,.0f} {C:>10,.0f}")
print(f"{'MVPD = C/B (pp)':<40} {mvpd_base*100:>10.1f} {mvpd_wide*100:>10.1f}")
print(f"{'LATE estimate (pp)':<40} {delta_hat*100:>10.2f} {delta_hat*100:>10.2f}")
print(f"{'P(cost-effective)':<40} {pce_base:>10.4f} {pce_wide:>10.4f}")
print(f"{'EVPI ($ per enrollee)':<40} {evpi_base:>10.2f} {evpi_wide:>10.2f}")
print(f"{'Current decision':<40} {'Reject':>10} {'Accept':>10}")
print("="*60)
```

---

## Summary

- The **policy value function** $\text{ROI}(\pi) = B \cdot \mathbb{E}[\tau(X)\pi(X)] - C \cdot \mathbb{E}[\pi(X)]$ translates any causal estimand into a net benefit, making explicit that the decision depends on both the effect estimate and the cost-benefit ratio.

- The **MVPD** $\tau^* = C/B$ is the structural break-even threshold; it does not depend on sample size and should be computed before looking at the estimate, to prevent threshold-setting from being reverse-engineered.

- The **probability of cost-effectiveness** and the **CEAC** translate a confidence interval or posterior into a probability statement about the decision, which is more actionable than a p-value and directly relevant to policy communication.

- **EVPI** measures the decision value of uncertainty reduction: it is zero when the current evidence strongly supports one action, and highest when the posterior is centered near the MVPD. EVPI bounds the value of any future study.

- **EVSI** as a function of sample size provides a principled basis for study design: the optimal sample size equates the marginal EVSI gain to the marginal cost of data collection, without reference to power or $\alpha$ levels.

- **Risk aversion** raises the effective threshold for action by an amount proportional to $\alpha B^2 \hat{\sigma}^2$, providing a formal justification for the intuition that high-stakes uncertain decisions require stronger evidence.

- **Optimal targeting** under a budget constraint implements a threshold rule on $\hat{\tau}(x)$; the value of targeting over uniform treatment is determined by the variance of the CATE, which quantifies the economic payoff to heterogeneity estimation.

---

## Further Reading

1. **Fenwick, E., Claxton, K., & Sculpher, M. (2001).** "Representing uncertainty: The role of cost-effectiveness acceptability curves." *Health Economics*, 10(8), 779–787. — The canonical reference for CEACs and their decision-theoretic interpretation; establishes that PCE is the operationally correct summary of cost-effectiveness uncertainty.

2. **Claxton, K. (1999).** "The irrelevance of inference: A decision-making approach to the stochastic evaluation of health care technologies." *Journal of Health Economics*, 18(3), 341–364. — Argues that EVPI, not hypothesis testing, is the correct criterion for deciding whether further research is warranted; foundational for the EVSI literature.

3. **Manski, C. F. (2004).** "Statistical treatment rules for heterogeneous populations." *Econometrica*, 72(4), 1221–1246. — Derives minimax regret optimal treatment rules under partial identification of the ATE; bridges Chapter 14's policy learning to the decision-theoretic framework without Bayesian priors.

4. **Finkelstein, A., Taubman, S., Wright, B., Bernstein, M., Gruber, J., Newhouse, J. P., ... & Baicker, K. (2012).** "The Oregon Health Insurance Experiment: Evidence from the first year." *Quarterly Journal of Economics*, 127(3), 1057–1106. — Primary source for the OHE estimates used throughout this chapter; see online appendix for the IV specification and Table 9 for catastrophic expenditure results.

5. **Baicker, K., & Finkelstein, A. (2019).** "The impact of Medicaid expansion on voter participation: Evidence from the Oregon Health Insurance Experiment." *American Economic Journal: Economic Policy* — Broader benefit accounting; documents non-medical spillovers that affect the benefit measure $B$ and thus the MVPD.

6. **Strong, M., Oakley, J. E., & Brennan, A. (2014).** "Estimating multiparameter partial expected value of perfect partial information from a probabilistic sensitivity analysis sample." *Medical Decision Making*, 34(3), 311–326. — Extends EVPI to multiparameter models and provides efficient Monte Carlo estimators for EVSI; directly applicable when both $\tau$ and $B$ carry uncertainty.
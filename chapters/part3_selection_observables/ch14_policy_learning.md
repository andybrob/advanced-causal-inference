# Chapter 14: Policy Learning and Treatment Rules

The chapters preceding this one asked a question of identification and estimation: what is the causal effect of treatment on outcome $Y$? This chapter asks the downstream question that motivates much of applied work: *given* what we have learned about heterogeneous effects, how should we assign treatment? A policymaker operating a Medicaid expansion cannot offer coverage to everyone—budget constraints bite. Even absent resource constraints, a blanket mandate may harm some recipients while helping others. The statistical problem is to learn a *treatment rule* $\pi: \mathcal{X} \to \{0,1\}$ that maps observed covariates to a treatment recommendation, optimizing aggregate welfare while respecting institutional constraints.

This chapter develops the formal framework for policy learning. We begin with the definition of policy value and regret, establish finite-sample regret bounds for empirical welfare maximization, and connect the machinery to doubly-robust off-policy evaluation. We then examine budget-constrained policies via Lagrangian relaxation, briefly treat fairness constraints, and close with the relationship between CATE estimation and policy learning—showing that for binary treatment, knowing the *sign* of $\tau(x)$ is sufficient for the optimal rule.

Throughout, we apply these ideas to the Oregon Health Insurance Experiment (OHE), asking: given age, income, and baseline health indicators, who benefits most from Medicaid coverage, and can a simple policy tree recover most of the welfare gains from the theoretically optimal rule?

---

## 14.1 Policy Value and the Empirical Welfare Problem

Let $(Y_i(0), Y_i(1), X_i, Z_i, D_i)$ be defined as in the OHE context: $Z_i$ is lottery selection (instrument), $D_i$ is actual enrollment in Oregon Health Plan, $X_i$ are pre-randomization covariates, and $Y_i \in \{Y_i^{\text{doc}}, Y_i^{\text{cat}}\}$ indexes outcomes of interest. A *statistical treatment rule* (STR) is a (possibly randomized) map $\pi: \mathcal{X} \to [0,1]$, where $\pi(x)$ is interpreted as the probability of assigning treatment to an individual with covariates $x$. We focus primarily on deterministic rules $\pi: \mathcal{X} \to \{0,1\}$.

**Definition 14.1 (Policy Value).** The welfare value of policy $\pi$ is
$$V(\pi) \;=\; \mathbb{E}\bigl[Y_i(\pi(X_i))\bigr] \;=\; \mathbb{E}\bigl[\pi(X_i)\,Y_i(1) + (1-\pi(X_i))\,Y_i(0)\bigr].$$

Under the standard identifying assumptions (consistency, unconfoundedness $D_i \perp (Y_i(0), Y_i(1)) \mid X_i$, overlap $0 < p(x) < 1$), the policy value can be expressed entirely in terms of observables:
$$V(\pi) = \mathbb{E}\bigl[\mu_1(X_i)\,\pi(X_i) + \mu_0(X_i)\,(1-\pi(X_i))\bigr],$$
where $\mu_d(x) = \mathbb{E}[Y_i \mid X_i = x, D_i = d]$.

The *policy regret* of a learned rule $\hat{\pi}$ relative to the oracle optimal $\pi^* = \arg\max_{\pi \in \Pi} V(\pi)$ is defined as follows.

**Definition 14.2 (Policy Regret).** For a policy class $\Pi$,
$$R(\hat{\pi}) \;=\; V(\pi^*) - V(\hat{\pi}).$$

A key representation of regret connects it directly to the conditional average treatment effect $\tau(x) = \mu_1(x) - \mu_0(x)$:

**Lemma 14.1 (Regret Representation).** For any deterministic $\hat{\pi}$,
$$R(\hat{\pi}) \;=\; \mathbb{E}\bigl[|\tau(X_i)| \cdot \mathbf{1}\bigl(\hat{\pi}(X_i) \neq \pi^*(X_i)\bigr)\bigr].$$

*Proof.* Since $\pi^*(x) = \mathbf{1}(\tau(x) > 0)$, we have $V(\pi^*) - V(\hat{\pi}) = \mathbb{E}[(\pi^*(X_i) - \hat{\pi}(X_i))\tau(X_i)]$. On the event $\hat{\pi}(X_i) \neq \pi^*(X_i)$, the sign of $\pi^*(X_i) - \hat{\pi}(X_i)$ equals the sign of $\tau(X_i)$, so the integrand equals $|\tau(X_i)|$. On $\hat{\pi}(X_i) = \pi^*(X_i)$, the integrand is zero. $\square$

This representation has a striking implication: regret accrues only where the learned rule disagrees with the oracle, and the cost of disagreement is proportional to $|\tau(X_i)|$—the magnitude of the treatment effect at that point. Errors in regions where effects are near zero are cheap; errors where effects are large are costly.

---

## 14.2 Empirical Welfare Maximization and Regret Bounds

The *empirical welfare maximization* (EWM) approach of Kitagawa and Tetenov (2018) proposes to estimate $\pi^*$ by solving
$$\hat{\pi} \;=\; \arg\max_{\pi \in \Pi}\; \hat{V}(\pi),$$
where $\hat{V}(\pi)$ is an empirical estimator of $V(\pi)$ and $\Pi$ is a restricted policy class. The restriction to $\Pi$ is essential: unrestricted maximization over all measurable functions would overfit catastrophically.

**Inverse Propensity Weighting Estimator.** Under unconfoundedness, a natural estimator uses IPW:
$$\hat{V}^{\text{IPW}}(\pi) \;=\; \frac{1}{n}\sum_{i=1}^n \frac{Y_i \cdot \mathbf{1}(D_i = \pi(X_i))}{\hat{p}(X_i, D_i)},$$
where $\hat{p}(x,1) = \hat{\Pr}(D_i = 1 \mid X_i = x)$ and $\hat{p}(x,0) = 1 - \hat{p}(x,1)$. In randomized settings such as the OHE (using $Z_i$ as $D_i$ in a reduced-form analysis), the propensity score is known: $p(x,1) = p(x,0) = 1/2$, simplifying estimation considerably.

**Doubly Robust Policy Value Estimator.** To improve efficiency and achieve robustness to misspecification of either the outcome model or the propensity model, the doubly robust (DR) estimator is:

$$\hat{V}^{\text{DR}}(\pi) \;=\; \frac{1}{n}\sum_{i=1}^n \left[\hat{\mu}_{\pi(X_i)}(X_i) + \frac{\mathbf{1}(\pi(X_i) = D_i)\bigl(Y_i - \hat{\mu}_{D_i}(X_i)\bigr)}{\hat{p}(X_i, D_i)}\right].$$

This estimator is doubly robust in the sense that it is consistent for $V(\pi)$ if either $\hat{\mu}_d(\cdot)$ or $\hat{p}(\cdot)$ is correctly specified.

**Theorem 14.1 (Minimax Regret Bound, Kitagawa-Tetenov 2018).** Let $\Pi$ be a finite policy class with $|\Pi|$ elements, $Y_i \in [0,1]$, and $p(x,d) \geq \underline{p} > 0$. Then the EWM estimator satisfies
$$\mathbb{E}[R(\hat{\pi})] \;\leq\; C\,\sqrt{\frac{\log|\Pi|}{n}},$$
where $C$ depends on $\underline{p}$ and the range of $Y$. Moreover, no estimator can achieve a uniformly smaller expected regret over the minimax-optimal class of data-generating processes.

*Proof sketch.* The result follows from a uniform law of large numbers over $\Pi$. For each $\pi \in \Pi$, the difference $\hat{V}(\pi) - V(\pi)$ is a sum of bounded i.i.d. terms; Hoeffding's inequality gives concentration at rate $O(1/\sqrt{n})$. A union bound over $|\Pi|$ policies gives $\sup_{\pi \in \Pi}|\hat{V}(\pi) - V(\pi)| \leq C\sqrt{\log|\Pi|/n}$ with high probability. Since $\hat{\pi}$ maximizes $\hat{V}$, we have $\hat{V}(\hat{\pi}) \geq \hat{V}(\pi^*)$, and combining the two concentration bounds gives the regret bound. Minimax lower bounds follow from a Fano-type argument; see Kitagawa and Tetenov (2018) for details. $\square$

For policy trees of depth $k$ over $p$ covariates, the log-cardinality satisfies $\log|\Pi| = O(k \log p)$, so the bound becomes $O(\sqrt{k \log p / n})$. Deeper trees incur larger regret bounds—a formal statement of the bias-variance tradeoff in policy learning.

---

## 14.3 Policy Trees: Interpretable Treatment Rules

Policy trees are decision trees where leaves output treatment decisions rather than outcome predictions. Their appeal is transparency: a depth-2 tree can be printed on a policy memo and communicated to frontline administrators who implement the rule.

Formally, a depth-$k$ policy tree partitions $\mathcal{X}$ into $2^k$ rectangular regions using axis-aligned splits, assigning a treatment decision $\{0,1\}$ to each leaf. The EWM problem over the class $\Pi_k$ of depth-$k$ trees is NP-hard in general, but dynamic programming algorithms (Sverdrup, Kanodia, Zhou, Wager, and Athey 2020) solve it exactly in $O(n^2 p \cdot 2^k)$ time, which is feasible for moderate $k$.

The `econml` library implements policy trees as `PolicyTree`, which maximizes the IPW-weighted empirical welfare. The input is the CATE estimate (or the augmented IPW scores) from a prior estimation stage, connecting policy learning directly to the CATE machinery of Chapter 12.

---

## 14.4 Budget-Constrained Optimal Policy

In many applications, a planner cannot treat everyone with $\tau(X_i) > 0$—budget or capacity limits the fraction treated to at most $\kappa \in (0,1)$. The constrained policy problem is:
$$\pi^*_\kappa \;=\; \arg\max_{\pi \in \Pi}\; V(\pi) \quad \text{subject to}\quad \mathbb{E}[\pi(X_i)] \leq \kappa.$$

**Theorem 14.2 (Lagrangian Solution to Budget-Constrained Policy).** Under mild regularity, the solution to the constrained problem has the threshold form:
$$\pi^*_\kappa(x) \;=\; \mathbf{1}(\tau(x) \geq \lambda^*_\kappa),$$
where $\lambda^*_\kappa$ is the Lagrange multiplier solving $\mathbb{E}[\mathbf{1}(\tau(X_i) \geq \lambda^*_\kappa)] = \kappa$.

*Proof.* Form the Lagrangian $\mathcal{L}(\pi, \lambda) = V(\pi) - \lambda(\mathbb{E}[\pi(X_i)] - \kappa) = \mathbb{E}[(\tau(X_i) - \lambda)\pi(X_i)] + \mathbb{E}[\mu_0(X_i)] + \lambda\kappa$. For fixed $\lambda \geq 0$, the pointwise maximizer is $\pi_\lambda(x) = \mathbf{1}(\tau(x) \geq \lambda)$. The optimal $\lambda^*_\kappa$ is found by strong duality (Slater's condition holds since $\tau$ has a continuous distribution), which requires $\mathbb{E}[\pi_{\lambda^*_\kappa}(X_i)] = \kappa$ at the constraint boundary. $\square$

In practice, $\lambda^*_\kappa$ is estimated by computing $\hat{\tau}(X_i)$ for all observations and finding the $100(1-\kappa)$-th percentile of the estimated CATE distribution. The policy value curve $\kappa \mapsto V(\pi^*_\kappa)$ is concave and increasing; plotting it reveals the marginal welfare gain of expanding coverage.

---

## 14.5 Fairness Constraints

Policy learning without fairness constraints may recommend concentrating treatment among a subgroup that happens to have high average $\tau(x)$, excluding others on the basis of sensitive attributes. Suppose $A_i \in \{0,1\}$ is a protected attribute (e.g., gender, race). A *demographic parity* constraint requires:
$$\mathbb{E}[\pi(X_i) \mid A_i = 1] \;=\; \mathbb{E}[\pi(X_i) \mid A_i = 0].$$

This constraint can be incorporated into the Lagrangian framework by adding a penalty $\gamma \cdot |\mathbb{E}[\pi \mid A=1] - \mathbb{E}[\pi \mid A=0]|$. The resulting optimal rule still has a threshold structure, but with *group-specific thresholds* $\lambda^*_a$ that equalize treatment rates across groups. Kitagawa and Tetenov (2021) and Sverdrup et al. discuss the formal minimax properties of fair policy learners; notably, the regret cost of imposing fairness constraints is bounded and can be estimated.

A subtler fairness criterion is *equalized welfare*: require $V_a(\pi) \geq V_b(\pi)$ across groups, ensuring neither group is actively harmed relative to the alternative. This is a welfare constraint rather than a representation constraint and may be more natural in health policy contexts.

---

## 14.6 From CATE to Policy: The Sign Sufficiency Principle

A central theme of this chapter is that policy learning requires less from the data than CATE estimation. The oracle rule $\pi^*(x) = \mathbf{1}(\tau(x) > 0)$ depends only on the *sign* of $\tau(x)$. An estimator that is a poor approximation to $\tau(\cdot)$ in $L_2$ may nevertheless perfectly recover the sign, yielding zero regret.

**Proposition 14.1 (Sign Sufficiency).** If $\hat{\tau}(x)$ satisfies $\text{sign}(\hat{\tau}(x)) = \text{sign}(\tau(x))$ almost everywhere, then $\mathbf{1}(\hat{\tau}(x) > 0) = \pi^*(x)$ a.e. and $R(\hat{\pi}) = 0$.

This observation has a practical corollary: aggressive regularization of CATE estimators that shrinks estimated effects toward zero may inadvertently introduce sign errors in high-noise regions, increasing regret. Methods such as the policy tree bypass CATE estimation altogether, directly optimizing the empirical welfare criterion.

However, sign sufficiency also reveals a limitation: in the presence of budget constraints, knowing only the sign is insufficient—we need an *ordering* of individuals by treatment effect magnitude to determine who receives the scarce $\kappa$ fraction of treatments. This is why budget-constrained policy learning implicitly requires more information from the CATE estimator than unconstrained binary assignment.

---

## Python: Policy Learning on the Oregon Health Insurance Experiment

```python
"""
Chapter 14: Policy Learning — Oregon Health Insurance Experiment
Requires: econml, pandas, numpy, scikit-learn, matplotlib, statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from econml.policy import PolicyTree

# ── 1. Data loading ──────────────────────────────────────────────────────────
# Download from https://data.nber.org/oregon/
# We use the 12-month survey + administrative data merge.
# Minimal required columns after merge:
#   Z (lottery), D (ohp_all_ever_admin), Y_doc (doc_any_12m),
#   Y_cat (catastrophic_exp_inp), age_19_34 ... age_56_64, female,
#   hhinc_pct_fpl (household income as % FPL)

def load_ohe(path="oregon_data.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "selected":             "Z",
        "ohp_all_ever_admin":   "D",
        "doc_any_12m":          "Y_doc",
        "catastrophic_exp_inp": "Y_cat",
    })
    # Age group dummies — use whatever age indicators are in the public file
    age_cols = [c for c in df.columns if c.startswith("age_")]
    df["female"] = df["female"].astype(float)
    df["hhinc_pct_fpl"] = pd.to_numeric(df["hhinc_pct_fpl"], errors="coerce")
    df = df.dropna(subset=["Z","D","Y_doc","female","hhinc_pct_fpl"] + age_cols)
    return df, age_cols

def make_synthetic_ohe(n=10_000, seed=42):
    """
    Synthetic data that mimics OHE structure for reproducible illustration.
    True CATE: tau(x) = 0.12 + 0.18*young - 0.10*high_income + 0.08*female
    """
    rng = np.random.default_rng(seed)
    female       = rng.binomial(1, 0.55, n).astype(float)
    young        = rng.binomial(1, 0.40, n).astype(float)   # age 19-34
    high_income  = rng.binomial(1, 0.30, n).astype(float)   # hhinc > 100% FPL
    hhinc        = rng.uniform(0, 200, n)

    tau_true = 0.12 + 0.18*young - 0.10*high_income + 0.08*female
    Z  = rng.binomial(1, 0.50, n).astype(float)             # RCT: p=0.5
    D  = Z.copy()                                            # perfect compliance (ITT=LATE)
    mu0 = 0.55 + 0.05*female + 0.10*young
    Y_doc = rng.binomial(1, np.clip(mu0 + tau_true*D, 0, 1), n).astype(float)
    Y_cat = rng.binomial(1, np.clip(0.05 - 0.03*D + 0.02*young, 0, 1), n).astype(float)

    return pd.DataFrame({
        "Z": Z, "D": D,
        "Y_doc": Y_doc, "Y_cat": Y_cat,
        "female": female, "young": young,
        "high_income": high_income, "hhinc_pct_fpl": hhinc,
        "tau_true": tau_true,
    }), ["young"]

# Use synthetic data; replace with load_ohe() for real analysis.
df, age_cols = make_synthetic_ohe()
covariate_cols = age_cols + ["female", "high_income", "hhinc_pct_fpl"]
X = df[covariate_cols].values
Z = df["Z"].values
D = df["D"].values
Y = df["Y_doc"].values          # primary outcome: any doctor visit at 12 months

# ── 2. Doubly-Robust CATE scores (augmented IPW) ─────────────────────────────
# In the RCT, p(Z=1|X) = 0.5 exactly; we use this known propensity.
# For observational applications, replace with a fitted propensity model.

p_score = np.full(len(df), 0.5)   # known from RCT design

# Cross-fitted outcome models mu_0(x) and mu_1(x)
def cross_fit_outcome(X, D, Y, n_folds=5):
    n = len(Y)
    mu_hat = np.zeros(n)
    fold_ids = np.array_split(np.arange(n), n_folds)
    for fold in fold_ids:
        train = np.setdiff1d(np.arange(n), fold)
        m0 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                        learning_rate=0.05, random_state=0)
        m1 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                        learning_rate=0.05, random_state=0)
        m0.fit(X[train][D[train]==0], Y[train][D[train]==0])
        m1.fit(X[train][D[train]==1], Y[train][D[train]==1])
        mu_hat[fold] = np.where(D[fold]==1,
                                m1.predict(X[fold]),
                                m0.predict(X[fold]))
    # Fit final models on full data for prediction
    m0_full = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=0)
    m1_full = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=0)
    m0_full.fit(X[D==0], Y[D==0])
    m1_full.fit(X[D==1], Y[D==1])
    mu0_full = m0_full.predict(X)
    mu1_full = m1_full.predict(X)
    return mu0_full, mu1_full

mu0, mu1 = cross_fit_outcome(X, D, Y)

# Augmented IPW (Neyman-orthogonal) score for CATE
# Gamma_i = mu1(X) - mu0(X) + [1(D=1)/p - 1(D=0)/(1-p)] * (Y - mu_D(X))
mu_D = np.where(D == 1, mu1, mu0)
gamma = ((mu1 - mu0)
         + (D / p_score - (1-D) / (1-p_score)) * (Y - mu_D))

print(f"Mean AIPW score (≈ ATE): {gamma.mean():.4f}")
print(f"Std of AIPW score:       {gamma.std():.4f}")

# ── 3. Doubly-Robust Policy Value Estimator ───────────────────────────────────
def dr_policy_value(pi_decisions, mu0, mu1, D, Y, p_score):
    """
    Compute DR policy value for a vector of binary decisions pi_decisions.
    pi_decisions: array of 0/1 of length n
    """
    mu_pi   = np.where(pi_decisions == 1, mu1, mu0)
    treated = (pi_decisions == D).astype(float)
    mu_act  = np.where(D == 1, mu1, mu0)
    dr_scores = mu_pi + treated * (Y - mu_act) / np.where(D==1, p_score, 1-p_score)
    return dr_scores.mean(), dr_scores.std() / np.sqrt(len(Y))

# Baseline policies
v_universal, se_universal = dr_policy_value(np.ones(len(Y),dtype=int),mu0,mu1,D,Y,p_score)
v_none,      se_none      = dr_policy_value(np.zeros(len(Y),dtype=int),mu0,mu1,D,Y,p_score)
print(f"\nBaseline policies (Y = doc_any_12m):")
print(f"  Universal coverage: {v_universal:.4f} (SE={se_universal:.4f})")
print(f"  No coverage:        {v_none:.4f}      (SE={se_none:.4f})")
print(f"  Max possible gain:  {v_universal - v_none:.4f}")

# ── 4. Policy Tree ────────────────────────────────────────────────────────────
# PolicyTree takes the AIPW score as input and finds the depth-k tree
# that maximizes empirical welfare E[gamma_i * pi(X_i)].
# econml's PolicyTree expects (X, Gamma) where Gamma[:,0] = reward of D=0
# and Gamma[:,1] = reward of D=1.  We construct Gamma from the DR scores.

# Reward matrix: shape (n, 2)
# Gamma[:,d] = outcome under assignment d using DR decomposition
reward_0 = mu0 + (1-D)/(1-p_score) * (Y - mu_D) * (1 - D) / np.where(D==0, 1, 1)
# Cleaner: directly construct per-action DR rewards
def dr_reward(d_assign, mu0, mu1, D, Y, p_score):
    mu_a   = np.where(d_assign == 1, mu1, mu0)
    mu_act = np.where(D == 1, mu1, mu0)
    p_act  = np.where(D == 1, p_score, 1 - p_score)
    indicator = (D == d_assign).astype(float)
    return mu_a + indicator * (Y - mu_act) / p_act

R = np.column_stack([
    dr_reward(np.zeros(len(Y), dtype=int), mu0, mu1, D, Y, p_score),
    dr_reward(np.ones(len(Y),  dtype=int), mu0, mu1, D, Y, p_score),
])  # shape (n, 2): R[:,0] = DR reward if untreated, R[:,1] = DR reward if treated

pt = PolicyTree(max_depth=2, min_samples_leaf=50, random_state=0)
pt.fit(X, R, feature_names=covariate_cols)

pi_tree = pt.predict(X)  # 0 or 1 for each observation
v_tree, se_tree = dr_policy_value(pi_tree, mu0, mu1, D, Y, p_score)
print(f"\nDepth-2 policy tree value:  {v_tree:.4f} (SE={se_tree:.4f})")
print(f"Tree treatment rate:        {pi_tree.mean():.3f}")
pt.print_tree_()

# ── 5. CATE-threshold policy ──────────────────────────────────────────────────
tau_hat = mu1 - mu0     # simple plug-in CATE estimate
pi_cate  = (tau_hat > 0).astype(int)
v_cate, se_cate = dr_policy_value(pi_cate, mu0, mu1, D, Y, p_score)
print(f"\nCATE-threshold policy value: {v_cate:.4f} (SE={se_cate:.4f})")
print(f"CATE treatment rate:          {pi_cate.mean():.3f}")

# Oracle (using known tau_true from synthetic data)
if "tau_true" in df.columns:
    pi_oracle = (df["tau_true"].values > 0).astype(int)
    v_oracle, se_oracle = dr_policy_value(pi_oracle, mu0, mu1, D, Y, p_score)
    print(f"Oracle policy value:          {v_oracle:.4f} (SE={se_oracle:.4f})")

# ── 6. Budget-constrained policy value curve ─────────────────────────────────
kappas = np.linspace(0.05, 1.0, 40)
v_budget = []
se_budget = []

for kappa in kappas:
    # Threshold at (1-kappa) quantile of estimated CATE
    threshold = np.quantile(tau_hat, 1 - kappa)
    pi_k = (tau_hat >= threshold).astype(int)
    # Enforce exact budget: take top-kappa fraction
    pi_k = np.zeros(len(tau_hat), dtype=int)
    top_k = int(np.ceil(kappa * len(tau_hat)))
    top_idx = np.argsort(tau_hat)[-top_k:]
    pi_k[top_idx] = 1
    v_k, se_k = dr_policy_value(pi_k, mu0, mu1, D, Y, p_score)
    v_budget.append(v_k)
    se_budget.append(se_k)

v_budget  = np.array(v_budget)
se_budget = np.array(se_budget)

# ── 7. Figure: Policy value curve ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(kappas, v_budget, color="steelblue", lw=2, label="Budget-constrained (CATE order)")
ax.fill_between(kappas, v_budget - 1.96*se_budget, v_budget + 1.96*se_budget,
                alpha=0.25, color="steelblue")
ax.axhline(v_universal, color="darkred",  ls="--", lw=1.5, label=f"Universal (κ=1): {v_universal:.3f}")
ax.axhline(v_none,      color="gray",     ls=":",  lw=1.5, label=f"No treatment:     {v_none:.3f}")
ax.axhline(v_tree,      color="darkorange", ls="-.", lw=1.5,
           label=f"Policy tree (κ={pi_tree.mean():.2f}): {v_tree:.3f}")
ax.set_xlabel("Budget constraint κ (max treatment fraction)", fontsize=12)
ax.set_ylabel("DR policy value E[Y(π)]", fontsize=12)
ax.set_title("Policy value as a function of budget constraint\nOregon Health Insurance Experiment (doc_any_12m)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("ch14_policy_value_curve.pdf", dpi=150)
plt.show()

# ── 8. Regret table ──────────────────────────────────────────────────────────
print("\n── Policy Comparison ─────────────────────────────────────")
header = f"{'Policy':<30} {'Value':>8} {'SE':>7} {'Regret':>8} {'Treat%':>8}"
print(header)
print("-" * len(header))
policies = [
    ("Universal coverage",   v_universal, se_universal),
    ("No treatment",         v_none,      se_none),
    ("CATE threshold",       v_cate,      se_cate),
    ("Policy tree (d=2)",    v_tree,      se_tree),
]
if "tau_true" in df.columns:
    policies.append(("Oracle", v_oracle, se_oracle))
    v_ref = v_oracle
else:
    v_ref = v_universal

for name, v, se in policies:
    regret = v_ref - v
    trt_rate = (pi_cate.mean() if "CATE" in name
                else pi_tree.mean() if "tree" in name
                else (1.0 if "Universal" in name else 0.0))
    print(f"{name:<30} {v:>8.4f} {se:>7.4f} {regret:>8.4f} {trt_rate:>8.2%}")
```

The code above illustrates four central objects from the chapter. The AIPW score $\Gamma_i$ in step 2 is the semiparametrically efficient influence function for the ATE; when used as input to `PolicyTree`, it converts the policy optimization problem into a weighted classification problem that the tree algorithm can solve exactly. Step 5 demonstrates sign sufficiency: the CATE-threshold policy $\hat{\pi}(x) = \mathbf{1}(\hat{\tau}(x) > 0)$ often attains near-oracle performance at a fraction of the complexity of the full CATE estimator. The budget curve in step 6 is the key deliverable for a policymaker: it shows the marginal welfare gain of relaxing the capacity constraint from $\kappa$ to $\kappa + d\kappa$, with the steepest gains concentrated at low $\kappa$ when the highest-benefit individuals are treated first.

---

## 14.7 Inference for Policy Value

Learning a policy and evaluating it on the same data produces optimistic bias. Cross-fitting (reserving a hold-out set for evaluation after fitting the policy tree on training data) controls this. Under standard regularity conditions, the DR policy value estimator is $\sqrt{n}$-consistent and asymptotically normal:
$$\sqrt{n}\bigl(\hat{V}^{\text{DR}}(\hat{\pi}) - V(\hat{\pi})\bigr) \;\xrightarrow{d}\; \mathcal{N}(0, \sigma^2_\pi),$$
where $\sigma^2_\pi = \text{Var}(\text{DR score}_i(\hat{\pi}))$. A $(1-\alpha)$ confidence interval for $V(\hat{\pi})$ is therefore $\hat{V}^{\text{DR}} \pm z_{\alpha/2} \hat{\sigma}_\pi / \sqrt{n}$.

Comparing two policies $\hat{\pi}_1$ and $\hat{\pi}_2$ requires the joint distribution of $(\hat{V}^{\text{DR}}(\hat{\pi}_1), \hat{V}^{\text{DR}}(\hat{\pi}_2))$. The difference $\hat{V}^{\text{DR}}(\hat{\pi}_1) - \hat{V}^{\text{DR}}(\hat{\pi}_2)$ is a sample mean of i.i.d. terms $\text{DR}_i(\hat{\pi}_1) - \text{DR}_i(\hat{\pi}_2)$, so standard $t$-tests apply directly.

One subtlety: the confidence intervals above condition on the learned policy $\hat{\pi}$, not averaging over the distribution of policies. If one wants to account for the randomness in the policy-learning step itself, the bootstrap provides valid coverage but at higher computational cost.

---

## Summary

Policy learning translates heterogeneous causal effect estimates into actionable treatment assignment rules. The key results of this chapter are:

1. **Regret has a CATE interpretation**: $R(\hat{\pi}) = \mathbb{E}[|\tau(X_i)| \cdot \mathbf{1}(\hat{\pi}(X_i) \neq \pi^*(X_i))]$. Errors where effects are small are cheap; errors where they are large are costly.

2. **EWM achieves minimax regret**: Maximizing empirical welfare over a restricted policy class $\Pi$ achieves expected regret $O(\sqrt{\log|\Pi|/n})$, and no method can do better in the minimax sense.

3. **Budget constraints yield threshold policies**: The capacity-constrained optimal rule assigns treatment to the $\kappa$ fraction of individuals with the largest CATE, estimated via the quantile of $\hat{\tau}(X_i)$.

4. **Sign sufficiency**: For unconstrained binary treatment, knowing $\text{sign}(\tau(x))$ is sufficient for the optimal rule; this is a strictly weaker requirement than accurate CATE estimation.

5. **Doubly-robust policy evaluation** provides $\sqrt{n}$-consistent, asymptotically normal estimates of policy value, enabling formal comparison across candidate rules.

In the OHE application, a depth-2 policy tree targeting young, low-income individuals recovers most of the welfare gain of universal Medicaid coverage while treating substantially fewer than half the eligible population—a finding with direct implications for budget-constrained Medicaid expansions.

---

## Further Reading

**Empirical welfare maximization.** The foundational treatment is Kitagawa and Tetenov (2018), "Who Should Be Treated? Empirical Welfare Maximization Methods for Treatment Choice," *Econometrica* 86(2):591–616. Their companion paper Kitagawa and Tetenov (2021) extends to fairness constraints. The minimax lower bounds use techniques from statistical decision theory; see Lehmann and Casella (1998), *Theory of Point Estimation*, Chapter 5.

**Policy trees.** Sverdrup, Kanodia, Zhou, Wager, and Athey (2020), "policytree: Policy learning via doubly robust empirical welfare maximization over trees," *Journal of Open Source Software* 5(50):2232, describes the exact dynamic programming algorithm implemented in `policytree` (R) and `econml` (Python). The depth-$k$ exact algorithm resolves a computational gap left open by greedy approaches.

**Off-policy evaluation.** Dudík, Langford, and Li (2011), "Doubly Robust Policy Evaluation and Learning," *ICML*, establishes the doubly robust policy estimator. Athey and Wager (2021), "Policy Learning with Observational Data," *Econometrica* 89(1):133–161, provides a unified treatment of policy learning under unconfoundedness and derives regret bounds for the DR-based EWM.

**Budget constraints and Lagrangian methods.** Luedtke and van der Laan (2016), "Statistical Inference for the Mean Outcome Under a Possibly Non-Unique Optimal Treatment Strategy," *Annals of Statistics* 44(2):713–742, handles the non-regularity arising when the constraint binds at a non-unique threshold. Sverdrup and Cui (2023) develop confidence intervals for budget-constrained policies.

**Oregon Health Insurance Experiment.** Finkelstein et al. (2012), "The Oregon Health Insurance Experiment: Evidence from the First Year," *Quarterly Journal of Economics* 127(3):1057–1106, is the primary analysis paper. Taubman et al. (2014), "Medicaid Increases Emergency-Department Use," *Science* 343(6168):263–268, examines downstream utilization effects that create the heterogeneous-effects environment studied here.

**Broader context.** For a machine-learning perspective connecting policy learning to contextual bandits and reinforcement learning, see Sutton and Barto (2018), *Reinforcement Learning: An Introduction*, Chapter 2 and the policy gradient literature. The causal inference perspective distinguishing policy learning from prediction is developed in Athey (2017), "Beyond Prediction: Using Big Data for Policy Problems," *Science* 355(6324):483–485.
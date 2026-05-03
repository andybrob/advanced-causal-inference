The simulate.py path doesn't exist yet. I'll write the chapter using the simulated longitudinal DGP as described in the blueprint, constructing the simulation inline.
I have enough context. The simulate.py file doesn't exist yet, so I'll write the chapter with self-contained simulation code inline.

# Chapter 29: Dynamic Treatment Regimes and Reinforcement Learning Connections

## 29.1 From Static to Dynamic Treatment Rules

All preceding chapters have asked a single-shot causal question: what is the effect of a fixed treatment $A$ on outcome $Y$? Real policy problems are sequential. A clinician decides whether to prescribe Medicaid-covered services at month 1, observes the patient's response, and decides again at month 6. An insurance eligibility administrator decides each open-enrollment period. The optimal action at each stage generally depends on everything that has happened before — health status, prior coverage, cost shocks. A **dynamic treatment regime** (DTR) formalizes this as a sequence of decision rules that maps individual history to treatment recommendation.

**Definition 29.1 (Dynamic Treatment Regime).** Let $T$ denote the number of decision stages. For stage $t \in \{1,\ldots,T\}$ let $H_t = (L_1, A_1, \ldots, L_{t-1}, A_{t-1}, L_t)$ denote the history up to stage $t$, where $L_t$ is the vector of time-varying covariates measured at stage $t$ and $A_t \in \{0,1\}$ is the binary treatment. A DTR $d = (d_1, \ldots, d_T)$ is a sequence of functions $d_t : \mathcal{H}_t \to \{0,1\}$.

Under regime $d$, each subject's treatment at stage $t$ is $A_t = d_t(H_t)$. Write $\bar{A}_t = (A_1,\ldots,A_t)$ and $\bar{d}(H_t) = (d_1(H_1),\ldots,d_t(H_t))$ for the regime-induced treatment history.

**Definition 29.2 (Potential Outcomes under a Regime).** $Y^d$ denotes the potential outcome that would be observed if a subject followed regime $d$ throughout. The **value** of regime $d$ is:
$$V^d = E[Y^d]$$
The goal is to find $d^* = \arg\max_d V^d$ within some class of regimes $\mathcal{D}$.

This formulation requires extending the consistency, positivity, and sequential ignorability assumptions from Chapters 25–27.

**Assumption 29.1 (Sequential Ignorability / SUTVA).** For each $t$:
$$Y^{\bar{a}} \perp\!\!\!\perp A_t \mid H_t, \quad \text{for all } \bar{a}$$
and $P(A_t = a_t \mid H_t) > 0$ a.s. for all $a_t$ with positive probability under any candidate regime.

Sequential ignorability says that conditional on everything observed up to stage $t$, treatment assignment is as good as random. This holds by design in a **SMART** (Sequential Multiple Assignment Randomized Trial) and requires untestable assumptions in observational data such as the BRFSS panel.

## 29.2 The Q-Function and Bellman Recursion

The workhorse for DTR estimation is the **Q-function**, the expected cumulative outcome given current history and current action, under the assumption that optimal actions are taken thereafter.

**Definition 29.3 (Q-function).** For a fixed regime $d$ and stage $t$:
$$Q_t^d(h_t, a_t) = E\!\left[Y^d \,\big|\, H_t = h_t,\, A_t = a_t\right]$$

where the expectation integrates over future randomness assuming regime $d$ is followed from stage $t+1$ onward.

The connection to dynamic programming is immediate. At the final stage $T$:
$$Q_T^d(h_T, a_T) = E[Y \mid H_T = h_T, A_T = a_T]$$

which is just the conditional mean outcome. Moving backward one stage:

**Theorem 29.1 (Bellman Recursion for DTRs).** Under Assumption 29.1:
$$Q_{t}^d(h_t, a_t) = E\!\left[\max_{a_{t+1}} Q_{t+1}^d(H_{t+1},\, d_{t+1}(H_{t+1}))\,\big|\, H_t = h_t,\, A_t = a_t\right]$$

For the **optimal regime** $d^*$, the Bellman optimality equation becomes:
$$Q_t^*(h_t, a_t) = E\!\left[\max_{a} Q_{t+1}^*(H_{t+1}, a)\,\big|\, H_t = h_t,\, A_t = a_t\right]$$

and the optimal decision at stage $t$ is:
$$d_t^*(h_t) = \arg\max_{a \in \{0,1\}} Q_t^*(h_t, a)$$

*Proof sketch.* By iterated expectations and sequential ignorability, $V^d = E[Q_1^d(H_1, d_1(H_1))]$. Optimizing stage by stage from $T$ back to $1$ — each stage's choice affects only downstream potential outcomes — yields the separability that makes backward induction valid. The full proof follows by induction on $T - t$ using the tower property and sequential ignorability. $\square$

The value of the optimal regime is then recovered as:
$$V^{d^*} = E[\max_{a} Q_1^*(H_1, a)]$$

## 29.3 Q-Learning: Estimation by Backward Induction

**Q-learning** translates Theorem 29.1 into a regression algorithm. The key insight: at each stage we can construct a regression pseudo-outcome that encodes the value of future optimal decisions.

**Algorithm 29.1 (Q-Learning for DTRs).**

1. *Stage $T$.* Regress $Y$ on $(H_T, A_T)$ to obtain $\hat{Q}_T(h_T, a_T)$.

2. *Stage $t < T$.* Define the pseudo-outcome:
   $$\tilde{Y}_t^{(i)} = \max_{a \in \{0,1\}} \hat{Q}_{t+1}(H_{t+1}^{(i)}, a)$$
   Regress $\tilde{Y}_t^{(i)}$ on $(H_t^{(i)}, A_t^{(i)})$ to obtain $\hat{Q}_t$.

3. *Optimal regime.*
   $$\hat{d}_t^*(h_t) = \arg\max_{a} \hat{Q}_t(h_t, a)$$

4. *Value estimate.*
   $$\hat{V}^{d^*} = \frac{1}{n}\sum_{i=1}^n \max_a \hat{Q}_1(H_1^{(i)}, a)$$

The regression at each stage can use any flexible learner — linear models, random forests, gradient boosting. In the linear case with main effects and treatment interactions, the treatment-covariate interaction coefficient at stage $t$ determines which patients benefit from $A_t = 1$.

**Remark on Bias Propagation.** Q-learning error compounds across stages. If $\hat{Q}_T$ has approximation error $\epsilon_T$, then $\hat{Q}_{T-1}$ inherits $\epsilon_T$ plus its own estimation error. This is the "curse of horizon" in offline policy optimization. Doubly robust methods (Section 29.5) partially address this.

## 29.4 SMART Designs and Identification

In observational data, sequential ignorability is an assumption. The **SMART** design provides experimental identification of the optimal DTR without it.

A SMART is a multi-stage randomized experiment where randomization occurs at each stage, potentially conditional on intermediate response. Participants are randomized to initial treatment; responders and non-responders are then separately re-randomized at stage 2.

**Identification in a SMART.** Let $e_t(a_t \mid h_t) = P(A_t = a_t \mid H_t = h_t)$ denote the known treatment probability at stage $t$ (from the design). The **inverse probability weighted** (IPW) estimator of $V^d$ is:

$$\hat{V}^d_{\text{IPW}} = \frac{1}{n}\sum_{i=1}^n \frac{\mathbf{1}(\bar{A}_i = \bar{d}(H_i))}{\prod_{t=1}^T e_t(A_{it} \mid H_{it})} Y_i$$

The indicator $\mathbf{1}(\bar{A}_i = \bar{d}(H_i))$ selects subjects whose observed treatment history is consistent with regime $d$ at every stage. The product of propensity scores reweights for the probability of following that history under the design.

**Theorem 29.2 (IPW Consistency).** Under Assumption 29.1 and positivity, $\hat{V}^d_{\text{IPW}} \xrightarrow{p} V^d$.

*Proof.* $E\!\left[\frac{\mathbf{1}(\bar{A}_i = \bar{d}(H_i))}{\prod_t e_t(A_{it}\mid H_{it})} Y_i\right] = E[Y^d]$ by sequential ignorability and iterated expectations, using the factorization of the joint treatment probability. $\square$

The IPW estimator has high variance when regimes are rarely followed or propensity scores are extreme, motivating doubly robust alternatives.

## 29.5 Doubly Robust DTR Estimation

The doubly robust (DR) estimator for DTRs combines outcome regression and propensity score weighting so that it is consistent if either model is correctly specified at each stage.

**Theorem 29.3 (Doubly Robust Value Estimator).** Define:
$$\hat{V}^d_{\text{DR}} = \frac{1}{n}\sum_{i=1}^n \left[\frac{\mathbf{1}(\bar{A}_i = \bar{d}(H_i))}{\prod_t \hat{e}_t(A_{it}\mid H_{it})} Y_i + \sum_{t=1}^T \frac{\mathbf{1}(\bar{A}_i^{t-1} = \bar{d}^{t-1}(H_i^{t-1}))}{\prod_{s=1}^{t-1} \hat{e}_s(A_{is}\mid H_{is})} \left(\hat{Q}_t(H_{it}, d_t(H_{it})) - \hat{Q}_t(H_{it}, A_{it})\right)\right]$$

where $\hat{Q}_t$ is fitted via backward induction and $\hat{e}_t$ are fitted propensity scores. Then $\hat{V}^d_{\text{DR}}$ is consistent if for each $t$ either $\hat{Q}_t \to Q_t$ or $\hat{e}_t \to e_t$.

The augmentation term is an IPW-weighted correction that adds back the difference between what the regime would predict and what was actually done. It has mean zero when either model is correct, providing the double-robustness. The estimator's semiparametric efficiency bound is achieved when both models are consistently estimated at the appropriate rates (Luedtke and van der Laan, 2016).

**Variance Estimation.** The influence function of $\hat{V}^d_{\text{DR}}$ is the summand minus $V^d$. Bootstrap or analytic influence-function-based standard errors are both valid; the bootstrap is simpler but computationally expensive for large $T$.

## 29.6 Connections to Reinforcement Learning

The vocabulary of DTRs maps directly onto reinforcement learning (RL). The correspondence is:

| Causal DTR | RL |
|---|---|
| Stage $t$ | Time step $t$ |
| History $H_t$ | State $S_t$ |
| Treatment $A_t$ | Action $A_t$ |
| Regime $d$ | Policy $\pi$ |
| Value $V^d$ | Expected return $V^\pi$ |
| Q-function $Q_t^d$ | Action-value function $Q^\pi$ |
| Bellman recursion | Bellman equation |
| Backward induction | Value iteration |

In RL terms, a DTR is a **finite-horizon Markov Decision Process** (MDP) with a deterministic policy. The key difference from standard RL is that causal inference operates **offline** (from fixed observational or experimental data) rather than online (with active environment interaction), and it demands **identification** — the connection between observed data and potential outcomes — as a first step.

**Policy Gradient Connection.** The policy gradient theorem gives:
$$\nabla_\theta V^{d_\theta} = E\!\left[\sum_{t=1}^T \nabla_\theta \log d_\theta(A_t \mid H_t) \cdot Q_t^{d_\theta}(H_t, A_t)\right]$$

where $d_\theta$ is a parameterized stochastic policy. In the causal DTR literature, this corresponds to AIPW-based gradient estimators for value optimization over a policy class. The doubly robust estimator of Section 29.5 can be seen as a one-step debiased gradient.

**Fitted Q-Iteration.** The RL algorithm **fitted Q-iteration** (Ernst et al., 2005) is precisely Q-learning applied with flexible function approximators. The backward induction in Algorithm 29.1 is a finite-horizon version. The infinite-horizon case introduces discount factors $\gamma < 1$ and requires iterating the Bellman operator to convergence rather than running backward a fixed number of steps.

**Off-Policy Evaluation.** The IPW and DR estimators of Sections 29.4–29.5 are exactly the **off-policy evaluation** (OPE) estimators studied in the RL literature, under the name doubly robust off-policy evaluation (Jiang and Li, 2016). The causal inference literature derived these independently with greater attention to semiparametric efficiency.

## 29.7 Optimal Regime Estimation and Inference

Estimating the optimal regime introduces statistical complications absent from estimating the value of a fixed regime. The optimal rule involves a $\max$ operation, creating non-smooth objective functions and non-standard asymptotics near the boundary where the two arms are nearly equally good.

**Near-Boundary Problem.** Let $\Delta_t(h_t) = Q_t^*(h_t, 1) - Q_t^*(h_t, 0)$ be the blip function. The optimal decision is $d_t^*(h_t) = \mathbf{1}(\Delta_t(h_t) > 0)$. Near $\Delta_t = 0$, estimation uncertainty about which action is better leads to misclassification of subjects, and the distribution of $\hat{V}^{d^*} - V^{d^*}$ is non-normal — it has the distribution of a maximum of Gaussian random variables. Standard bootstrap confidence intervals are invalid.

**Theorem 29.4 (Non-regularity of Optimal Value Estimation, Laber et al. 2014).** Let $\hat{V}^{d^*}$ be the plug-in value estimator based on estimated Q-functions. Then $\sqrt{n}(\hat{V}^{d^*} - V^{d^*})$ does not converge to a Gaussian distribution in general; the asymptotic distribution depends on the probability mass near the indifference boundary $\{\Delta_t = 0\}$.

**Remedies.** Three approaches are used in practice:

1. **Hard-threshold regime classes.** Restrict $\mathcal{D}$ to regimes indexed by a finite-dimensional parameter $\eta$ (e.g., linear threshold $d_t(h_t) = \mathbf{1}(\beta_t^\top h_t > 0)$). The value function $V(\eta)$ is smooth in $\eta$ away from degenerate configurations, enabling regular estimation by outcome-weighted learning.

2. **Smooth approximation.** Replace $\max(a, b)$ with a softmax approximation $\text{softmax}(a, b; \lambda)$ for small $\lambda$, reducing boundary non-regularity at the cost of bias.

3. **m-out-of-n bootstrap.** Uses bootstrap samples of size $m \ll n$; this is consistent for the non-regular case but requires careful selection of $m$.

## Python: Q-Learning and Doubly Robust DTR Estimation on Simulated Longitudinal Data

The following code simulates a three-period longitudinal dataset motivated by the BRFSS panel structure — individuals with annual insurance and health status observations — and implements Q-learning backward induction, an IPW value estimator for comparison, and a doubly robust value estimator. We then compare the value of the optimal estimated DTR against static "always insure" and "never insure" regimes.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

# -------------------------------------------------------------------------
# 1. Simulate a 3-period longitudinal DGP
#    Motivation: annual BRFSS-like panel, binary insurance coverage A_t,
#    time-varying health status L_t, terminal outcome Y (financial hardship).
#
#    DGP:
#      L_1 ~ Bernoulli(0.4)          baseline poor health
#      A_t | H_t: observational, correlated with L_t
#      L_{t+1} = f(L_t, A_t) + noise  health evolves
#      Y = g(L_3, A_3, cumulative coverage) + noise
# -------------------------------------------------------------------------

N = 5000
T = 3

def simulate_longitudinal(n, seed=42):
    rng = np.random.default_rng(seed)

    # Baseline covariates
    age    = rng.normal(45, 12, n).clip(18, 80)
    income = rng.normal(35000, 15000, n).clip(5000, 150000)
    L      = np.zeros((n, T+1))  # health status: 1 = poor health
    A      = np.zeros((n, T), dtype=int)
    ps     = np.zeros((n, T))    # true propensity scores

    # t=1 baseline health
    L[:, 0] = (rng.normal(-0.3 + 0.4*(income < 20000), 1, n) > 0).astype(float)

    for t in range(T):
        # Observational treatment: sicker people more likely to seek insurance
        # but lower income reduces access
        log_odds = (0.5 * L[:, t]
                    - 0.8 * (income < 20000)
                    + 0.3 * (age > 60)
                    + 0.5 * (t == 0)   # first-year enrollment bump
                    + rng.normal(0, 0.3, n))
        ps[:, t] = 1 / (1 + np.exp(-log_odds))
        A[:, t]  = (rng.uniform(size=n) < ps[:, t]).astype(int)

        if t < T - 1:
            # Health transition: insurance improves health (reduces poor-health prob)
            # Poor health persists without treatment
            health_logit = (1.2 * L[:, t]
                            - 0.9 * A[:, t]
                            + 0.4 * (income < 20000)
                            + rng.normal(0, 0.5, n))
            L[:, t+1] = (rng.uniform(size=n) < 1/(1+np.exp(-health_logit))).astype(float)

    # Terminal poor-health state
    L[:, T] = L[:, T-1]

    # Outcome: catastrophic financial expenditure (continuous, higher = worse)
    # True optimal: insure when L_t=1 (poor health), don't otherwise
    cumulative_coverage = A.sum(axis=1)
    Y = (3.0 * L[:, T]
         - 1.5 * cumulative_coverage
         + 0.8 * L[:, T] * (1 - A[:, T-1])   # uninsured + sick = worst outcome
         + 0.5 * (income < 20000)
         + rng.normal(0, 1.5, n))

    return {
        'L': L,          # (n, T+1): health at each stage + final
        'A': A,          # (n, T): treatment at each stage
        'ps': ps,        # (n, T): true propensity scores
        'age': age,
        'income': income,
        'Y': Y,
    }

data = simulate_longitudinal(N)
L, A, ps_true, Y = data['L'], data['A'], data['ps'], data['Y']
age, income = data['age'], data['income']

print(f"Mean Y: {Y.mean():.3f}  |  Insured all 3 periods: {(A.sum(1)==3).mean():.3f}")
print(f"Average treatment rates: {A.mean(0)}")

# -------------------------------------------------------------------------
# 2. Build history features for each stage
# -------------------------------------------------------------------------

def make_history(t, L, A, age, income):
    """Concatenate all observed information up to stage t."""
    cols = [age.reshape(-1,1), income.reshape(-1,1), L[:, 0].reshape(-1,1)]
    for s in range(t):
        cols.append(L[:, s+1].reshape(-1,1))
        cols.append(A[:, s].reshape(-1,1))
    cols.append(L[:, t].reshape(-1,1))   # current health
    return np.hstack(cols)

H = {t: make_history(t, L, A, age, income) for t in range(T)}

# -------------------------------------------------------------------------
# 3. Q-Learning: backward induction
# -------------------------------------------------------------------------

Q_hat  = {}   # Q_hat[t] is a fitted model
opt_d  = {}   # opt_d[t] is a function h -> {0,1}

# Stage T-1 = 2 (0-indexed): regress Y on H_2, A_2
# Include treatment interaction by augmenting features with A*H features

def augment_with_treatment(H_feat, A_vec):
    """Stack H, A, and H*A interaction features."""
    A_col = A_vec.reshape(-1, 1)
    return np.hstack([H_feat, A_col, H_feat * A_col])

pseudo_outcome = Y.copy()

for t in reversed(range(T)):
    X = augment_with_treatment(H[t], A[:, t])
    model = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    model.fit(X, pseudo_outcome)
    Q_hat[t] = model

    # Optimal action at this stage: compare predicted Q for a=1 vs a=0
    X1 = augment_with_treatment(H[t], np.ones(N))
    X0 = augment_with_treatment(H[t], np.zeros(N))
    opt_d[t] = (model.predict(X1) > model.predict(X0)).astype(int)

    # Pseudo-outcome for stage t-1: max_a Q_t(H_t, a)
    pseudo_outcome = np.maximum(model.predict(X1), model.predict(X0))

# -------------------------------------------------------------------------
# 4. Estimate value of optimal DTR (Q-learning plug-in)
# -------------------------------------------------------------------------

V_optimal_ql = pseudo_outcome.mean()   # pseudo_outcome at t=0 = max_a Q_1(H_1,a)

print(f"\nQ-learning value estimate (optimal DTR): {V_optimal_ql:.3f}")

# -------------------------------------------------------------------------
# 5. IPW value estimator for the optimal DTR
# -------------------------------------------------------------------------

# Fit propensity scores (we don't know true ones in practice)
ps_hat = np.zeros((N, T))
for t in range(T):
    X_ps = np.hstack([H[t], np.ones((N, 1))])
    ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.05, random_state=42)
    ps_model.fit(H[t], A[:, t])
    ps_hat[:, t] = ps_model.predict_proba(H[t])[:, 1]

def ipw_value(opt_d, ps_hat, A, Y):
    """IPW estimator for value of regime defined by opt_d."""
    follows = np.ones(N, dtype=bool)
    weight  = np.ones(N)
    for t in range(T):
        d_t = opt_d[t]
        follows &= (A[:, t] == d_t)
        # propensity of the action actually taken
        prob_t = np.where(A[:, t] == 1, ps_hat[:, t], 1 - ps_hat[:, t])
        prob_t = np.clip(prob_t, 0.01, 0.99)
        weight *= prob_t
    return (follows * Y / weight).sum() / N

V_optimal_ipw = ipw_value(opt_d, ps_hat, A, Y)
print(f"IPW value estimate (optimal DTR):        {V_optimal_ipw:.3f}")

# -------------------------------------------------------------------------
# 6. Doubly Robust value estimator
# -------------------------------------------------------------------------

def dr_value(opt_d, Q_hat, ps_hat, A, H, Y):
    """
    Doubly robust estimator of V^d.
    Augmentation: for each stage t, add the IPW-weighted blip correction.
    """
    # Full IPW term
    follows = np.ones(N, dtype=bool)
    cum_weight = np.ones(N)
    for t in range(T):
        d_t = opt_d[t]
        follows &= (A[:, t] == d_t)
        prob_t = np.where(A[:, t] == 1, ps_hat[:, t], 1 - ps_hat[:, t])
        prob_t = np.clip(prob_t, 0.01, 0.99)
        cum_weight *= prob_t

    ipw_term = (follows * Y / cum_weight)

    # Augmentation terms: stage-specific corrections
    aug = np.zeros(N)
    cum_weight_t = np.ones(N)
    follows_tminus1 = np.ones(N, dtype=bool)

    for t in range(T):
        d_t = opt_d[t]

        # Q predictions under regime and under observed
        X_d  = augment_with_treatment(H[t], d_t.astype(float))
        X_obs = augment_with_treatment(H[t], A[:, t].astype(float))
        blip  = Q_hat[t].predict(X_d) - Q_hat[t].predict(X_obs)

        # Weight = product of propensities for stages 0..t-1
        weight_tminus1 = np.clip(cum_weight_t, 0.01, None)
        aug += follows_tminus1 * blip / weight_tminus1

        # Update for next stage
        follows_tminus1 &= (A[:, t] == d_t)
        prob_t = np.where(A[:, t] == 1, ps_hat[:, t], 1 - ps_hat[:, t])
        prob_t = np.clip(prob_t, 0.01, 0.99)
        cum_weight_t *= prob_t

    return (ipw_term + aug).mean()

V_optimal_dr = dr_value(opt_d, Q_hat, ps_hat, A, H, Y)
print(f"DR value estimate (optimal DTR):         {V_optimal_dr:.3f}")

# -------------------------------------------------------------------------
# 7. Static regime comparisons
# -------------------------------------------------------------------------

def static_value(a_val, Q_hat, ps_hat, A, H, Y, method='dr'):
    """Value of static regime: always a_val at every stage."""
    static_d = {t: np.full(N, a_val, dtype=int) for t in range(T)}
    if method == 'ipw':
        return ipw_value(static_d, ps_hat, A, Y)
    return dr_value(static_d, Q_hat, ps_hat, A, H, Y)

V_always_insure  = static_value(1, Q_hat, ps_hat, A, H, Y)
V_never_insure   = static_value(0, Q_hat, ps_hat, A, H, Y)

print(f"\nDR value (always insure):                {V_always_insure:.3f}")
print(f"DR value (never insure):                 {V_never_insure:.3f}")
print(f"Gain of optimal DTR over always insure:  {V_optimal_dr - V_always_insure:.3f}")
print(f"Gain of optimal DTR over never insure:   {V_optimal_dr - V_never_insure:.3f}")

# -------------------------------------------------------------------------
# 8. Describe the estimated optimal regime
# -------------------------------------------------------------------------

print("\n--- Optimal DTR description ---")
for t in range(T):
    insure_rate = opt_d[t].mean()
    # Compare insure rate by current health status
    poor_health_insure = opt_d[t][L[:, t] == 1].mean()
    good_health_insure = opt_d[t][L[:, t] == 0].mean()
    print(f"Stage {t+1}: insure {insure_rate:.3f} overall | "
          f"poor health: {poor_health_insure:.3f} | "
          f"good health: {good_health_insure:.3f}")

# -------------------------------------------------------------------------
# 9. Bootstrap confidence intervals for DR value estimates
# -------------------------------------------------------------------------

n_boot = 200
boot_values = {'optimal': [], 'always': [], 'never': []}

for b in range(n_boot):
    idx = rng.integers(0, N, N)
    L_b    = L[idx]; A_b = A[idx]; Y_b = Y[idx]
    age_b  = age[idx]; income_b = income[idx]

    H_b = {t: make_history(t, L_b, A_b, age_b, income_b) for t in range(T)}
    ps_b = np.zeros((N, T))
    for t in range(T):
        pm = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                        learning_rate=0.05, random_state=b)
        pm.fit(H_b[t], A_b[:, t])
        ps_b[:, t] = pm.predict_proba(H_b[t])[:, 1]

    Q_b = {}; od_b = {}
    po = Y_b.copy()
    for t in reversed(range(T)):
        Xb = augment_with_treatment(H_b[t], A_b[:, t])
        mb = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                       learning_rate=0.05, random_state=b)
        mb.fit(Xb, po)
        Q_b[t] = mb
        X1b = augment_with_treatment(H_b[t], np.ones(N))
        X0b = augment_with_treatment(H_b[t], np.zeros(N))
        od_b[t] = (mb.predict(X1b) > mb.predict(X0b)).astype(int)
        po = np.maximum(mb.predict(X1b), mb.predict(X0b))

    boot_values['optimal'].append(dr_value(od_b, Q_b, ps_b, A_b, H_b, Y_b))
    boot_values['always'].append(static_value(1, Q_b, ps_b, A_b, H_b, Y_b))
    boot_values['never'].append(static_value(0, Q_b, ps_b, A_b, H_b, Y_b))

for k, vals in boot_values.items():
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"95% CI  {k:8s}: [{lo:.3f}, {hi:.3f}]")
```

**Interpreting the output.** The optimal DTR description at each stage should reveal the health-state-contingent structure built into the DGP: the regime preferentially insures subjects in poor health ($L_t = 1$), where marginal benefit is largest, and may recommend against coverage for healthy subjects at low financial risk. The gain over "always insure" is modest but real — it reflects the savings from not insuring subjects who would not benefit and the concentrated benefit for high-risk subgroups. The DR and Q-learning value estimates should be close; large divergence indicates either model misspecification or positivity violations.

## Summary

- A **dynamic treatment regime** is a sequence of decision rules $d = (d_1, \ldots, d_T)$ mapping history to treatment. Its value $V^d = E[Y^d]$ is the population mean potential outcome under the regime.
- The **Q-function** $Q_t(h_t, a_t)$ is the expected outcome conditional on history and current action, assuming optimal behavior thereafter. The **Bellman recursion** connects Q-functions across stages via iterated expectation, enabling backward induction.
- **Q-learning** estimates the optimal regime by backward regression: fit $Q_T$ by regressing $Y$ on $(H_T, A_T)$, construct pseudo-outcomes $\max_a \hat{Q}_{t+1}(H_{t+1}, a)$, and regress those at stage $t$.
- **SMART designs** provide experimental identification of DTR value via known randomization probabilities; observational estimation requires sequential ignorability.
- The **doubly robust DTR estimator** augments IPW with a blip-function correction, achieving consistency if either the outcome model or the propensity model is correct at each stage, and attaining the semiparametric efficiency bound when both are correct.
- DTRs are finite-horizon MDPs with deterministic policies; Q-learning, IPW, and doubly robust OPE map directly to RL concepts, with the key addition of identification from causal assumptions.
- Estimating the **optimal** (rather than fixed) regime introduces non-regularity near the indifference boundary $Q_t(h_t, 1) \approx Q_t(h_t, 0)$; standard bootstrap is inconsistent there and restricted policy classes or $m$-out-of-$n$ bootstrap are required.
- Error propagates multiplicatively across stages via the product of propensity scores and additively via Q-function approximation error; doubly robust methods reduce but do not eliminate horizon-length sensitivity.

## Further Reading

- **Murphy, S.A. (2003).** "Optimal dynamic treatment regimes." *Journal of the Royal Statistical Society B*, 65(2): 331–366. The foundational paper establishing DTR estimation as a semiparametric problem and introducing G-estimation for blip functions. Essential for understanding the connection to structural nested mean models.

- **Schulte, P.J., Tsiatis, A.A., Laber, E.B., Davidian, M. (2014).** "Q- and A-learning methods for estimating optimal dynamic treatment regimes." *Statistical Science*, 29(4): 640–661. Side-by-side comparison of Q-learning and A-learning (G-estimation of blip parameters); covers the non-regularity problem and variance estimation in detail.

- **Laber, E.B., Lizotte, D.J., Qian, M., Pelham, W.E., Murphy, S.A. (2014).** "Dynamic treatment regimes: Technical challenges and applications." *Electronic Journal of Statistics*, 8: 1225–1272. Covers inference challenges near the indifference boundary and the m-out-of-n bootstrap remedy; connects to the ADHD SMART trial.

- **Luedtke, A.R., van der Laan, M.J. (2016).** "Statistical inference for the mean outcome under a possibly non-unique optimal treatment strategy." *Annals of Statistics*, 44(2): 713–742. Establishes the semiparametric efficiency theory for doubly robust optimal DTR value estimation and characterizes the non-regular asymptotic distribution.

- **Jiang, N., Li, L. (2016).** "Doubly robust off-policy value evaluation for reinforcement learning." *ICML 2016*, PMLR 48: 652–661. Derives the DR-OPE estimator from the RL side, providing the bridge to the causal inference literature's augmented IPW estimator.

- **Linn, K.A., Laber, E.B., Stefanski, L.A. (2017).** "iqLearn: Interactive Q-learning in R." *Journal of Statistical Software*, 82(1). Practical implementation reference for the interactive Q-learning procedure that handles the non-regularity via smooth value-function approximations; translates directly to Python implementations.
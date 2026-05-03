# Chapter 19: Synthetic Control and Matrix Completion

## 19.1 The Core Problem: Few Treated Units

The methods developed in Chapters 15–17 assumed enough treated units to estimate average treatment effects with classical asymptotics. Difference-in-differences requires parallel trends to hold on average across a heterogeneous donor pool; event-study estimators average over many events. What happens when you have exactly one treated unit — or a handful?

This is not a pathological case. It is the norm in comparative policy analysis. A single country adopts a carbon tax. One state implements a minimum wage increase before others. Oregon runs a Medicaid lottery and later adopts ACA expansion in 2014. In each case the analyst confronts an $N = 1$ problem on the treatment side, with a potentially large donor pool of control units observed over time.

The fundamental challenge is counterfactual construction. You observe $Y_{1t}$ for the treated unit up to the treatment date $T_0$, and need $Y_{1t}^{(0)}$ — what the treated unit would have experienced absent treatment — for $t > T_0$. Parallel trends says $Y_{1t}^{(0)} - Y_{1,t-1}^{(0)} = \bar{Y}_{J,t} - \bar{Y}_{J,t-1}$ for some average over donors $j = 2, \ldots, J+1$. But if the treated unit's pre-period trajectory diverges from the equal-weighted donor average, this is a bad approximation.

Synthetic control replaces the equal-weighted average with an optimally weighted convex combination — weights chosen to make the pre-period synthetic control track the treated unit as closely as possible. The intuition is clean: if a weighted combination of California and Texas closely tracks Oregon's uninsurance rate from 2000–2013, that same combination is a credible counterfactual for 2014 onward.

## 19.2 The Abadie-Diamond-Hainmueller Estimator

Let $i = 1$ denote the treated unit (Oregon), $i = 2, \ldots, J+1$ the donor states, $t = 1, \ldots, T$ time periods with treatment beginning at $T_0 + 1$. Define the pre-period $\mathcal{T}_0 = \{1, \ldots, T_0\}$ and post-period $\mathcal{T}_1 = \{T_0 + 1, \ldots, T\}$.

**Definition 19.1 (Synthetic Control Weights).** The synthetic control weight vector $\hat{W} = (\hat{w}_2, \ldots, \hat{w}_{J+1})$ solves:

$$\hat{W} = \arg\min_{W \in \Delta^J} \sum_{t \in \mathcal{T}_0} \left( Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt} \right)^2$$

where $\Delta^J = \{W : w_j \geq 0, \sum_j w_j = 1\}$ is the unit simplex.

In the original Abadie, Diamond, and Hainmueller (2010) formulation, the objective is a weighted distance in a space that includes both outcomes and pre-period covariates $X$:

$$\hat{W} = \arg\min_{W \in \Delta^J} (X_1 - X_W)' V (X_1 - X_W)$$

where $X_W = \sum_j w_j X_j$ and $V$ is a positive semidefinite matrix of covariate importance weights, itself chosen to minimize out-of-sample prediction error over the pre-period. In practice, when pre-period outcome lags are included in $X$, this reduces approximately to minimizing prediction mean-squared error in pre-period outcomes, which is the form we work with throughout.

**Definition 19.2 (SC Estimand).** For each post-period $t \in \mathcal{T}_1$:

$$\hat{\tau}_t^{SC} = Y_{1t} - \hat{Y}_{1t}^{SC}, \quad \text{where } \hat{Y}_{1t}^{SC} = \sum_{j=2}^{J+1} \hat{w}_j Y_{jt}$$

An average effect over the post-period is $\hat{\tau}^{SC} = (T - T_0)^{-1} \sum_{t > T_0} \hat{\tau}_t^{SC}$.

The convexity constraint is what distinguishes SC from regression-based matching. Restricting to the simplex means the synthetic control is an interpolation within the convex hull of donor outcomes, not an extrapolation. This provides a safeguard against the synthetic control being driven by extreme weights on donors with very different outcome levels.

**Theorem 19.1 (Consistency of SC).** Suppose the data-generating process follows a factor model:

$$Y_{it}^{(0)} = \mu_i + \lambda_t + \theta_t' F_i + \epsilon_{it}$$

where $F_i \in \mathbb{R}^r$ are unit-specific factor loadings, $\theta_t \in \mathbb{R}^r$ are time-varying factor weights, and $\epsilon_{it}$ are mean-zero idiosyncratic shocks. If $\hat{W}$ achieves exact pre-period fit — meaning $\sum_j \hat{w}_j Y_{jt} = Y_{1t}$ for all $t \in \mathcal{T}_0$ — and if the pre-period is long relative to the number of factors ($T_0 \geq r$), then:

$$Y_{1t}^{(0)} - \sum_{j=2}^{J+1} \hat{w}_j Y_{jt} = O_p\left(\sqrt{\frac{T_0}{T_0^2}} \right) \to 0$$

as $T_0 \to \infty$, under suitable moment conditions on $\epsilon_{it}$.

*Proof sketch.* If $\sum_j \hat{w}_j Y_{jt} = Y_{1t}$ for all pre-period $t$, then expanding:

$$\sum_j \hat{w}_j (\mu_j + \lambda_t + \theta_t' F_j + \epsilon_{jt}) = \mu_1 + \lambda_t + \theta_t' F_1 + \epsilon_{1t}$$

The $\lambda_t$ terms cancel since $\sum_j \hat{w}_j = 1$. For this to hold for $T_0$ periods with $r$ factors, we need $\sum_j \hat{w}_j F_j = F_1$ (factor balance) and $\sum_j \hat{w}_j \mu_j = \mu_1$ (level balance). Factor balance is achievable when $T_0 \geq r$ and donors span $F_1$ in the convex hull. Post-period, the synthetic outcome tracks $Y_{1t}^{(0)}$ up to a term $\sum_j \hat{w}_j \epsilon_{jt} - \epsilon_{1t}$, which is $O_p(T_0^{-1/2})$ by averaging over pre-period to pin down the weights, yielding the result. $\square$

The critical assumption is that $F_1$ lies in the convex hull of $\{F_j\}_{j \geq 2}$ — the treated unit's factor loadings are interpolable from the donors. This is the **interpolation condition**, and it is empirically verifiable through pre-period fit quality. Poor pre-period fit signals either that the convexity constraint is binding (extrapolation would be needed) or that the factor model structure is misspecified.

## 19.3 Inference via Placebo Tests

Standard asymptotic inference fails here: there is one treated unit, so there is no central limit theorem operating on the treatment side. The solution, introduced by Abadie et al., is exact permutation inference.

**Definition 19.3 (In-Space Placebo).** Iteratively apply the SC procedure treating each donor $j$ as if it were the treated unit, using all remaining units (including $i = 1$) as its donor pool. Compute:

$$\hat{\tau}_t^{(j)} = Y_{jt} - \sum_{k \neq j} \hat{w}_k^{(j)} Y_{kt}, \quad t > T_0$$

For the true treated unit, $\hat{\tau}_t^{(1)}$ is defined analogously.

**Definition 19.4 (Post/Pre RMSPE Ratio).** For each unit $j$, define:

$$R_j = \frac{\sqrt{(T - T_0)^{-1} \sum_{t > T_0} (\hat{\tau}_t^{(j)})^2}}{\sqrt{T_0^{-1} \sum_{t \leq T_0} (\hat{\tau}_t^{(j)})^2}}$$

Units with poor pre-period fit (large pre-RMSPE) are unreliable placebos. Standard practice filters out placebo units with pre-RMSPE exceeding some multiple (e.g., 2×) of the treated unit's pre-RMSPE.

**Theorem 19.2 (Permutation p-value).** Under the sharp null $H_0: Y_{1t}^{(1)} = Y_{1t}^{(0)}$ for all $t > T_0$, the rank of $R_1$ among $\{R_j\}_{j=1}^{J+1}$ is uniformly distributed over $\{1, \ldots, J+1\}$. The one-sided p-value $\hat{p} = \Pr(R_j \geq R_1)$ is exactly valid.

This is a finite-sample result: exchangeability of treatment assignment (by the lottery or as-if random policy adoption) ensures that the true treated unit's label is uniformly distributed under the null, giving exact randomization inference.

A practical concern: when the donor pool is small ($J + 1 < 20$), the minimum achievable p-value is $1/(J+1)$, which may be too coarse. This motivates supplementing space-placebo tests with **time placebos** — applying SC to artificial treatment dates $T_0' < T_0$ within the pre-period, where the true effect is known to be zero, and checking that $\hat{\tau}^{SC}$ for these false treatment dates is small.

## 19.4 Synthetic Difference-in-Differences

The SC estimator uses unit weights to balance pre-period trajectories but applies them symmetrically across all post-period periods. Arkhangelsky et al. (2021) propose Synthetic DiD (SDID), which additionally applies **time weights** to balance pre-period and post-period time structure.

**Definition 19.5 (SDID Estimator).** The SDID estimator solves:

$$(\hat{\tau}^{SDID}, \hat{\alpha}, \hat{\beta}, \hat{\delta}) = \arg\min_{\tau, \alpha, \beta, \delta} \sum_{i=1}^{N} \sum_{t=1}^{T} \hat{\omega}_i \hat{\lambda}_t \left( Y_{it} - \alpha_i - \beta_t - \tau W_{it} - \delta \right)^2$$

where $W_{it} = \mathbf{1}[i = 1, t > T_0]$ is the treatment indicator, and:

- **Unit weights** $\hat{\omega} = (\hat{\omega}_1, \hat{\omega}_2, \ldots, \hat{\omega}_{J+1})$ with $\hat{\omega}_1 = 1$ and $(\hat{\omega}_2, \ldots, \hat{\omega}_{J+1}) \in \Delta^J$ minimize pre-period outcome discrepancy between treated and donor weighted average
- **Time weights** $\hat{\lambda} = (\hat{\lambda}_1, \ldots, \hat{\lambda}_T)$ with $\hat{\lambda}_t = 0$ for $t > T_0$ and $(\hat{\lambda}_1, \ldots, \hat{\lambda}_{T_0}) \in \Delta^{T_0}$ weight pre-period time periods to resemble the post-period outcome distribution

The time weights specifically solve:

$$\hat{\lambda}^{pre} = \arg\min_{\lambda \in \Delta^{T_0}} \left\| \frac{1}{J} \sum_{j=2}^{J+1} Y_{jt^{post}} - \sum_{t \leq T_0} \lambda_t \frac{1}{J} \sum_{j=2}^{J+1} Y_{jt} \right\|^2 + \zeta^2 T_0 \|\lambda\|^2$$

where $t^{post}$ denotes the average post-period outcome of donors and $\zeta$ is a ridge regularization parameter. The regularization avoids degenerate solutions that concentrate all weight on a single pre-period date.

**Theorem 19.3 (SDID as Doubly Robust).** Under a factor model $Y_{it}^{(0)} = \mu_i + \nu_t + \epsilon_{it}$ with $\mathbb{E}[\epsilon_{it}] = 0$, the SDID estimator $\hat{\tau}^{SDID}$ is consistent for $\tau$ if either:

1. The unit weights $\hat{\omega}$ satisfy $\sum_j \hat{\omega}_j Y_{jt} = Y_{1t}$ for $t \leq T_0$ (SC balance), or
2. The time weights $\hat{\lambda}$ satisfy parallel trends in mean: $\mathbb{E}[Y_{it}^{(0)}]$ follows the two-way FE structure

If both conditions hold simultaneously, $\hat{\tau}^{SDID}$ achieves lower variance than either SC or DiD alone.

This double robustness — one set of weights can fail as long as the other succeeds — gives SDID an efficiency advantage analogous to augmented IPW estimators (Chapter 11). When the panel is long and donors are many, SDID approaches the efficiency bound for this class of estimators.

The SDID estimate of the average post-period treatment effect on the treated is the coefficient $\hat{\tau}$ on the treatment indicator $W_{it}$ in the weighted two-way FE regression.

## 19.5 Matrix Completion and Nuclear Norm Minimization

Both SC and SDID rely on explicit parameterization: weights over donors. Matrix completion approaches the same problem from the perspective of low-rank matrix recovery. Think of the $N \times T$ outcome matrix $Y$ as having a low-rank signal plus noise. The counterfactual outcomes for the treated unit in the post-period are simply missing entries to be imputed.

**Definition 19.6 (Observation Operator).** Let $\Omega \subseteq [N] \times [T]$ denote the set of observed $(i, t)$ pairs. For the standard SC setup, $\Omega$ excludes the treated unit in the post-period: $\Omega = ([N] \times [T]) \setminus \{(1, t) : t > T_0\}$.

Define the projection $P_\Omega(M)_{it} = M_{it} \cdot \mathbf{1}[(i,t) \in \Omega]$.

**Definition 19.7 (MC-NNM Estimator).** The nuclear norm minimization estimator (Athey et al., 2021) solves:

$$\hat{L} = \arg\min_{L \in \mathbb{R}^{N \times T}} \frac{1}{|\Omega|} \|P_\Omega(Y - L)\|_F^2 + \lambda \|L\|_*$$

where $\|L\|_* = \sum_k \sigma_k(L)$ is the nuclear norm (sum of singular values), a convex proxy for rank.

The treatment effect estimates are $\hat{\tau}_{1t} = Y_{1t} - \hat{L}_{1t}$ for $t > T_0$.

The nuclear norm penalty encourages $\hat{L}$ to have low effective rank. If the true outcome matrix has rank $r \ll \min(N, T)$ — as implied by a factor model with $r$ factors — then nuclear norm minimization recovers the true $L$ with high probability when the observed entries satisfy an incoherence condition.

**Theorem 19.4 (MC-NNM Recovery Guarantee, informal).** Let $L^* = \mu \mathbf{1}' + \mathbf{1}\nu' + F\Theta'$ be a rank-$r$ matrix where $F \in \mathbb{R}^{N \times r}$ contains unit factor loadings and $\Theta \in \mathbb{R}^{T \times r}$ contains time factor loadings. Assume the standard incoherence conditions $\|F_i\| \leq \mu_0 \sqrt{r/N}$, $\|\Theta_t\| \leq \mu_0 \sqrt{r/T}$, and that $\Omega$ is sampled uniformly at random with $|\Omega| \geq C \mu_0^2 r (N + T) \log^2(\max(N,T))$ observed entries. Then with high probability, setting $\lambda \propto \sqrt{(N+T)/|\Omega|}$:

$$\frac{1}{NT}\|\hat{L} - L^*\|_F^2 \leq C \mu_0^2 r \frac{N + T}{|\Omega|}$$

*Proof sketch.* The result follows from the restricted isometry property (RIP) of the sampling operator $P_\Omega$ combined with the fact that for low-rank matrices, nuclear norm and Frobenius norm are equivalent up to the rank factor. The incoherence condition prevents the low-rank signal from being concentrated in a small subset of entries, ensuring each observed entry carries information about the global structure. Standard arguments for nuclear norm minimization (Candès and Recht, 2009) then give the Frobenius error bound. $\square$

The crucial difference from SC is that MC-NNM does not restrict the counterfactual to lie in the convex hull of donors. The optimization is over a matrix space, and the regularization is implicit through the nuclear norm rather than explicit through the simplex constraint. This allows MC-NNM to extrapolate when necessary, at the cost of requiring distributional assumptions on the missing pattern.

Athey et al. (2021) extend MC-NNM to include unit and time fixed effects explicitly — decomposing $L = \alpha \mathbf{1}' + \mathbf{1}\beta' + M$ and applying nuclear norm only to $M$. This is the **MC-NNM with fixed effects** variant, which tends to perform better in panels where level heterogeneity is large.

## 19.6 Connections and Comparisons

The three estimators inhabit different positions in a bias-variance tradeoff space. SC is low-variance when pre-period fit is excellent but can be biased if the convexity constraint binds. SDID adds the time-weighting dimension and achieves a doubly-robust property. MC-NNM is most flexible but requires the incoherence condition and a tuning parameter $\lambda$.

Under the two-way FE model $Y_{it}^{(0)} = \alpha_i + \beta_t + \epsilon_{it}$, all three are consistent. Under richer factor models, SC and MC-NNM both exploit factor structure, while standard DiD (TWFE) does not. The empirical literature increasingly uses all three as a robustness suite rather than committing to one.

---

## Python: Synthetic Control, MC-NNM, and SDID for ACA Medicaid Expansion

```python
"""
Chapter 19: Synthetic Control, MC-NNM, and SDID
Running example: Oregon ACA Medicaid expansion 2014
Donor pool: non-expansion states from BRFSS-style panel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Simulate ACA-style state panel data
#    (In production: load BRFSS state-year uninsured rates 2010-2019)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

STATES = [
    'Oregon',        # treated: expanded 2014
    'California', 'Washington', 'Colorado', 'Nevada', 'New Mexico',   # donors
    'Arizona', 'Minnesota', 'Michigan', 'Illinois', 'Ohio',
    'Pennsylvania', 'Delaware', 'Rhode Island', 'Connecticut',
    'Maryland', 'Massachusetts', 'Vermont', 'New York', 'New Jersey',
]

N = len(STATES)
T = 10          # 2010-2019
T0 = 4          # last pre-period year index (2013 = index 3, treatment 2014 = index 4)
YEARS = list(range(2010, 2020))

# Simulate latent factor model: 2 factors
r = 2
F = np.random.randn(N, r) * 0.5         # unit loadings
Theta = np.random.randn(T, r) * 0.3     # time loadings

alpha = np.random.uniform(10, 25, N)    # state-level uninsured baseline (%)
beta = np.linspace(0, -3, T)            # secular trend (declining uninsurance)
noise = np.random.randn(N, T) * 0.8

# True outcome matrix
Y = (alpha[:, None]
     + beta[None, :]
     + F @ Theta.T
     + noise)

# Oregon treatment effect: expansion reduces uninsured rate
# Starts 2014 (t=4), grows to -5pp by 2019
tau_true = np.array([0, 0, 0, 0, -3.0, -3.8, -4.2, -4.5, -4.8, -5.0])
Y[0, :] += tau_true      # Oregon is index 0

# Build dataframe
rows = []
for i, state in enumerate(STATES):
    for t_idx, year in enumerate(YEARS):
        rows.append({
            'state': state,
            'year': year,
            'uninsured_rate': Y[i, t_idx],
            'treated': int(state == 'Oregon' and year >= 2014),
        })
df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Control via constrained optimization
# ─────────────────────────────────────────────────────────────────────────────

def fit_synthetic_control(Y_matrix, treated_idx=0, T0=4):
    """
    Fit SC weights for treated unit.
    Y_matrix: (N x T) numpy array
    Returns: weight vector of length N-1 (donors)
    """
    Y_treated_pre = Y_matrix[treated_idx, :T0]
    Y_donors_pre  = np.delete(Y_matrix, treated_idx, axis=0)[:, :T0]  # (J x T0)

    J = Y_donors_pre.shape[0]

    def objective(w):
        synth_pre = w @ Y_donors_pre    # (T0,)
        return np.sum((Y_treated_pre - synth_pre) ** 2)

    def grad(w):
        synth_pre = w @ Y_donors_pre
        resid = synth_pre - Y_treated_pre
        return 2 * (Y_donors_pre @ resid)

    w0 = np.ones(J) / J
    bounds = Bounds(lb=0.0, ub=1.0)
    constraint = LinearConstraint(np.ones((1, J)), lb=1.0, ub=1.0)

    result = minimize(
        objective, w0,
        jac=grad,
        method='trust-constr',
        bounds=bounds,
        constraints=constraint,
        options={'gtol': 1e-8, 'maxiter': 2000}
    )
    return result.x


# Fit SC for Oregon
Y_mat = df.pivot(index='state', columns='year', values='uninsured_rate').values
state_order = df.pivot(index='state', columns='year', values='uninsured_rate').index.tolist()
oregon_idx = state_order.index('Oregon')

sc_weights = fit_synthetic_control(Y_mat, treated_idx=oregon_idx, T0=T0)

# Synthetic control time series
Y_donors = np.delete(Y_mat, oregon_idx, axis=0)
Y_synth  = sc_weights @ Y_donors                         # (T,)
Y_oregon = Y_mat[oregon_idx, :]

sc_effects = Y_oregon - Y_synth

# Pre-period RMSPE
pre_rmspe = np.sqrt(np.mean(sc_effects[:T0] ** 2))
post_rmspe = np.sqrt(np.mean(sc_effects[T0:] ** 2))
print(f"SC pre-RMSPE:  {pre_rmspe:.3f} pp")
print(f"SC post-RMSPE: {post_rmspe:.3f} pp")
print(f"SC avg effect (2014-2019): {sc_effects[T0:].mean():.3f} pp")

# Top donor weights
donor_names = [s for s in state_order if s != 'Oregon']
top_donors = sorted(zip(donor_names, sc_weights), key=lambda x: -x[1])[:5]
print("\nTop SC donor weights:")
for name, w in top_donors:
    print(f"  {name}: {w:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Placebo-in-Space inference
# ─────────────────────────────────────────────────────────────────────────────

def rmspe_ratio(effects, T0):
    pre  = np.sqrt(np.mean(effects[:T0]**2))
    post = np.sqrt(np.mean(effects[T0:]**2))
    if pre < 1e-6:
        return np.nan
    return post / pre


placebo_effects = {}
for i, state in enumerate(state_order):
    # Use all other states (including Oregon) as donors
    Y_donors_pl = np.delete(Y_mat, i, axis=0)
    Y_treated_pl = Y_mat[i, :]
    try:
        w_pl = fit_synthetic_control(
            np.vstack([Y_mat[i:i+1, :], Y_donors_pl]),
            treated_idx=0, T0=T0
        )
        synth_pl = w_pl @ Y_donors_pl
        placebo_effects[state] = Y_treated_pl - synth_pl
    except Exception:
        placebo_effects[state] = np.zeros(T)

ratios = {s: rmspe_ratio(e, T0) for s, e in placebo_effects.items()}
ratios_clean = {s: r for s, r in ratios.items() if r is not np.nan
                and rmspe_ratio(placebo_effects[s], T0) <= 5 * pre_rmspe}

oregon_ratio = ratios['Oregon']
pval = np.mean([r >= oregon_ratio for r in ratios_clean.values()])
print(f"\nPermutation p-value (one-sided): {pval:.3f}")
print(f"Oregon RMSPE ratio: {oregon_ratio:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MC-NNM (Nuclear Norm Minimization)
# ─────────────────────────────────────────────────────────────────────────────

def mc_nnm(Y_obs, mask, lam, max_iter=500, tol=1e-6):
    """
    Nuclear norm minimization via soft-thresholding of singular values.
    Uses proximal gradient descent.

    Y_obs: (N x T) matrix, missing entries can be 0 (ignored by mask)
    mask:  (N x T) binary; 1 = observed
    lam:   regularization parameter
    """
    N_, T_ = Y_obs.shape
    L = Y_obs.copy()

    # Estimate step size (Lipschitz constant = 1 for Frobenius loss with mask)
    step = 1.0

    for iteration in range(max_iter):
        L_old = L.copy()

        # Gradient step: only update at observed entries
        grad = mask * (L - Y_obs)
        L_grad = L - step * grad

        # Proximal step: soft-threshold singular values
        U, s, Vt = svd(L_grad, full_matrices=False)
        s_thresh = np.maximum(s - step * lam, 0)
        L = U @ np.diag(s_thresh) @ Vt

        change = np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-12)
        if change < tol:
            break

    return L


# Build observation mask: treat post-period Oregon as missing
mask = np.ones((N, T))
mask[oregon_idx, T0:] = 0

# Cross-validate lambda over pre-period holdout
def cv_lambda(Y_mat, mask, lambdas, oregon_idx, T0, n_folds=3):
    """Hold out random pre-period treated observations to tune lambda."""
    pre_obs = list(range(T0))
    fold_size = len(pre_obs) // n_folds
    cv_errors = np.zeros(len(lambdas))

    for fold in range(n_folds):
        val_t = pre_obs[fold * fold_size:(fold + 1) * fold_size]
        mask_cv = mask.copy()
        mask_cv[oregon_idx, val_t] = 0

        for k, lam in enumerate(lambdas):
            L_hat = mc_nnm(Y_mat * mask_cv, mask_cv, lam)
            cv_errors[k] += np.mean((Y_mat[oregon_idx, val_t]
                                     - L_hat[oregon_idx, val_t]) ** 2)

    return lambdas[np.argmin(cv_errors)]


lambdas = np.logspace(-2, 1, 15)
best_lam = cv_lambda(Y_mat, mask, lambdas, oregon_idx, T0)
print(f"\nMC-NNM best lambda (CV): {best_lam:.4f}")

L_hat = mc_nnm(Y_mat * mask, mask, lam=best_lam)

mc_effects = Y_oregon - L_hat[oregon_idx, :]
print(f"MC-NNM avg effect (2014-2019): {mc_effects[T0:].mean():.3f} pp")
print(f"MC-NNM recovered rank: {np.sum(np.linalg.svd(L_hat, compute_uv=False) > 0.1)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SDID Estimator
# ─────────────────────────────────────────────────────────────────────────────

def compute_sdid_time_weights(Y_donors, T0, zeta=None):
    """
    Compute time weights lambda that make the weighted pre-period donor
    average resemble the post-period donor average.
    """
    J, T = Y_donors.shape
    donor_mean_pre  = Y_donors[:, :T0].mean(axis=0)   # (T0,)
    donor_mean_post = Y_donors[:, T0:].mean(axis=0)   # (T - T0,)
    target = donor_mean_post.mean()                    # scalar: avg post outcome

    if zeta is None:
        # Default regularization from Arkhangelsky et al.
        zeta = max(1e-6, np.std(Y_donors[:, :T0]) * (J * T0) ** (1/4))

    def obj(lam):
        pred = lam @ donor_mean_pre
        return (pred - target) ** 2 + zeta**2 * T0 * np.sum(lam**2)

    def grad_lam(lam):
        pred = lam @ donor_mean_pre
        return 2 * (pred - target) * donor_mean_pre + 2 * zeta**2 * T0 * lam

    lam0 = np.ones(T0) / T0
    bounds = Bounds(lb=0.0, ub=1.0)
    constraint = LinearConstraint(np.ones((1, T0)), lb=1.0, ub=1.0)
    result = minimize(obj, lam0, jac=grad_lam, method='trust-constr',
                      bounds=bounds, constraints=constraint)
    return result.x


def sdid_estimate(Y_mat, treated_idx, T0):
    """
    SDID point estimate following Arkhangelsky et al. (2021).
    Returns (tau_hat, unit_weights, time_weights).
    """
    N_, T_ = Y_mat.shape
    Y_donors_mat = np.delete(Y_mat, treated_idx, axis=0)  # (J x T)
    Y_treated    = Y_mat[treated_idx, :]                   # (T,)
    J = Y_donors_mat.shape[0]

    # Step 1: unit weights (SC-style, pre-period)
    omega = fit_synthetic_control(
        np.vstack([Y_mat[treated_idx:treated_idx+1, :], Y_donors_mat]),
        treated_idx=0, T0=T0
    )

    # Step 2: time weights
    lam = compute_sdid_time_weights(Y_donors_mat, T0)
    # Extend to full T: pre-period weights lam, post-period weight = 0
    lam_full = np.zeros(T_)
    lam_full[:T0] = lam

    # Step 3: TWFE regression with weights
    # Stack observations: treated unit + donors
    # Weight: treated unit gets omega=1; donor i gets omega[i-1]
    omega_full = np.concatenate([[1.0], omega])   # (N,)

    rows_list = []
    for i_idx in range(N_):
        for t_idx in range(T_):
            w_unit = omega_full[i_idx]
            w_time = lam_full[t_idx] if t_idx < T0 else 1.0
            treated_flag = int(i_idx == 0 and t_idx >= T0)
            rows_list.append({
                'i': i_idx, 't': t_idx,
                'Y': Y_mat[treated_idx, t_idx] if i_idx == 0 else Y_donors_mat[i_idx-1, t_idx],
                'W': treated_flag,
                'weight': w_unit * w_time,
            })

    reg_df = pd.DataFrame(rows_list)
    reg_df = reg_df[reg_df['weight'] > 1e-10]

    # Demean by unit and time FE (weighted)
    unit_fe  = reg_df.groupby('i').apply(
        lambda g: np.average(g['Y'], weights=g['weight'])).to_dict()
    time_fe  = reg_df.groupby('t').apply(
        lambda g: np.average(g['Y'], weights=g['weight'])).to_dict()
    grand_mean = np.average(reg_df['Y'], weights=reg_df['weight'])

    reg_df['Y_dm'] = (reg_df['Y']
                      - reg_df['i'].map(unit_fe)
                      - reg_df['t'].map(time_fe)
                      + grand_mean)

    # WLS: regress Y_dm on W_dm
    reg_df['W_dm'] = (reg_df['W']
                      - reg_df['i'].map(
                          reg_df.groupby('i').apply(
                              lambda g: np.average(g['W'], weights=g['weight'])).to_dict())
                      - reg_df['t'].map(
                          reg_df.groupby('t').apply(
                              lambda g: np.average(g['W'], weights=g['weight'])).to_dict())
                      + np.average(reg_df['W'], weights=reg_df['weight']))

    num = np.sum(reg_df['weight'] * reg_df['W_dm'] * reg_df['Y_dm'])
    den = np.sum(reg_df['weight'] * reg_df['W_dm'] ** 2)
    tau_hat = num / den if den > 1e-10 else np.nan

    return tau_hat, omega, lam


tau_sdid, omega_sdid, lam_sdid = sdid_estimate(Y_mat, oregon_idx, T0)
print(f"\nSDID estimate: {tau_sdid:.3f} pp")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TWFE (naive) for comparison
# ─────────────────────────────────────────────────────────────────────────────

df['post'] = (df['year'] >= 2014).astype(int)
df['oregon'] = (df['state'] == 'Oregon').astype(int)
df['treated_post'] = df['oregon'] * df['post']

# Demean: subtract state mean and year mean, add grand mean
grand = df['uninsured_rate'].mean()
state_means = df.groupby('state')['uninsured_rate'].transform('mean')
year_means  = df.groupby('year')['uninsured_rate'].transform('mean')
df['Y_dm']  = df['uninsured_rate'] - state_means - year_means + grand
W_dm_grand  = df['treated_post'].mean()
state_W_dm  = df.groupby('state')['treated_post'].transform('mean')
year_W_dm   = df.groupby('year')['treated_post'].transform('mean')
df['W_dm']  = df['treated_post'] - state_W_dm - year_W_dm + W_dm_grand

twfe_tau = (df['W_dm'] * df['Y_dm']).sum() / (df['W_dm']**2).sum()
print(f"TWFE estimate:  {twfe_tau:.3f} pp")

# True average effect
true_avg = tau_true[T0:].mean()
print(f"True avg effect: {true_avg:.3f} pp")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Results table and plots
# ─────────────────────────────────────────────────────────────────────────────

results = pd.DataFrame({
    'Estimator': ['SC', 'MC-NNM', 'SDID', 'TWFE'],
    'Avg Effect (pp)': [
        sc_effects[T0:].mean(),
        mc_effects[T0:].mean(),
        tau_sdid,
        twfe_tau
    ],
    'Bias vs Truth': [
        sc_effects[T0:].mean() - true_avg,
        mc_effects[T0:].mean() - true_avg,
        tau_sdid - true_avg,
        twfe_tau - true_avg,
    ]
})
print("\n─── Estimator Comparison ────────────────────────────────")
print(results.to_string(index=False, float_format='{:.3f}'.format))


# ── Figure 1: SC plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(YEARS, Y_oregon, 'k-o', label='Oregon (observed)', linewidth=2)
ax.plot(YEARS, Y_synth,  'b--s', label='Synthetic Oregon (SC)', linewidth=2)
ax.axvline(x=2013.5, color='red', linestyle=':', linewidth=1.5, label='Treatment (2014)')
ax.fill_between(YEARS[T0:],
                Y_synth[T0:],
                Y_oregon[T0:],
                alpha=0.2, color='blue', label='Estimated effect')
ax.set_xlabel('Year')
ax.set_ylabel('Uninsured Rate (%)')
ax.set_title('Synthetic Control: Oregon Medicaid Expansion')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Figure 2: Placebo distribution ─────────────────────────────────────────
ax2 = axes[1]
placebo_post_effects = {}
for state, eff in placebo_effects.items():
    # only plot placebos with decent pre-period fit
    if (rmspe_ratio(eff, T0) is not np.nan and
            np.sqrt(np.mean(eff[:T0]**2)) < 5 * pre_rmspe):
        placebo_post_effects[state] = eff[T0:]

for state, eff in placebo_post_effects.items():
    color = 'red' if state == 'Oregon' else 'gray'
    lw    = 2.5 if state == 'Oregon' else 0.5
    alpha = 1.0 if state == 'Oregon' else 0.4
    zord  = 10  if state == 'Oregon' else 1
    ax2.plot(YEARS[T0:], eff, color=color, linewidth=lw,
             alpha=alpha, zorder=zord)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_xlabel('Year')
ax2.set_ylabel('Estimated Effect (pp)')
ax2.set_title(f'Placebo-in-Space  |  p-value = {pval:.2f}')
ax2.text(2016, ax2.get_ylim()[1]*0.85, 'Oregon', color='red', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch19_synthetic_control.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: ch19_synthetic_control.png")
```

---

## Summary

- Synthetic control constructs a counterfactual for a single treated unit as a convex combination of donor units, with weights chosen to minimize pre-period prediction error; the convexity constraint ensures interpolation, not extrapolation.
- Identification rests on a factor model assumption: if the synthetic control achieves good pre-period fit, it tracks latent factor loadings of the treated unit and provides a valid counterfactual post-treatment.
- Inference uses in-space placebo tests — applying the same SC procedure to each donor as if it were treated — and computes a p-value as the rank of the treated unit's post/pre RMSPE ratio.
- Synthetic DiD (SDID) augments SC with time weights that balance pre-period and post-period outcome distributions among donors, achieving a doubly robust property: consistency if either unit weights or parallel trends holds, with efficiency gains when both hold.
- Matrix completion (MC-NNM) treats post-period treated outcomes as missing matrix entries and recovers them via nuclear norm minimization, exploiting low-rank structure without the convexity constraint of SC; a cross-validated regularization parameter controls effective rank.
- All three methods outperform TWFE when the treated unit is a poor match for the equal-weighted donor average; in the ACA expansion setting, SC and SDID recover effects close to the true simulated DGP while TWFE is biased by donor heterogeneity.
- The methods form a robustness suite rather than competitors: good pre-period SC fit validates the counterfactual, SDID provides variance reduction, and MC-NNM tests sensitivity to the convexity assumption.

---

## Further Reading

1. **Abadie, A., Diamond, A., and Hainmueller, J. (2010).** "Synthetic Control Methods for Comparative Case Studies." *Journal of the American Statistical Association*, 105(490), 493–505. The foundational paper; proves identification under the factor model and introduces the RMSPE-ratio placebo test. Essential reading.

2. **Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G., and Wager, S. (2021).** "Synthetic Difference-in-Differences." *American Economic Review*, 111(12), 4088–4118. Introduces SDID, proves the doubly robust efficiency property, and provides bootstrap variance estimation. Supplants SC as the default method when $N$ is moderate.

3. **Athey, S., Bayati, M., Doudchenko, N., Imbens, G., and Khosravi, K. (2021).** "Matrix Completion Methods for Causal Panel Data Models." *Journal of the American Statistical Association*, 116(536), 1716–1730. Derives MC-NNM with fixed effects, establishes the recovery guarantee under incoherence, and compares to SC on multiple empirical applications including Proposition 99 and German reunification.

4. **Candès, E. and Recht, B. (2009).** "Exact Matrix Completion via Convex Optimization." *Foundations of Computational Mathematics*, 9(6), 717–772. The mathematical foundation for nuclear norm minimization and the RIP argument underlying Theorem 19.4.

5. **Ferman, B. and Pinto, C. (2021).** "Synthetic Controls with Imperfect Pretreatment Fit." *Quantitative Economics*, 12(4), 1197–1221. Analyzes the behavior of SC when exact pre-period fit is not achievable — the generic case — and derives asymptotic bias bounds; essential for applied practitioners who cannot achieve RMSPE near zero.

6. **Ben-Michael, E., Feller, A., and Rothstein, J. (2021).** "The Augmented Synthetic Control Method." *Journal of the American Statistical Association*, 116(536), 1789–1803. Introduces augmented SC (ASC), which adds an outcome model bias correction analogous to AIPW, providing a bridge between SC and doubly robust methods and connecting directly to the ideas in Chapter 11.
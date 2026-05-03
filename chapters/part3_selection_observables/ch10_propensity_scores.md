# Chapter 10: Propensity Scores Without Ritual

Propensity score methods occupy an unusual place in applied causal inference. They are among the most frequently used tools in observational studies, yet also among the most frequently misused. The rituals surrounding them—estimate a logistic regression, match on nearest neighbors, report a Love plot, declare success—are often performed without understanding what the propensity score actually accomplishes mathematically, when it helps, and when it actively misleads. This chapter strips away the ritual and examines the propensity score as a mathematical object, with precise conditions under which it delivers valid causal estimates and equally precise conditions under which it fails.

We work throughout with the Oregon Health Experiment (OHE), a 2008 Medicaid lottery in which selected individuals were invited to apply for Oregon's Medicaid program. The lottery selection $Z$ serves as an instrument for actual enrollment $D$, and our running example asks: what is the effect of Medicaid enrollment on healthcare utilization and financial protection? In this chapter we treat the problem as an observational study—ignoring the lottery and asking whether we can recover causal effects by conditioning on pre-treatment covariates. This sets up a deliberate contrast: at the end of the chapter we compare propensity-score-based estimates of the ATE to the IV-based LATE from Chapter 9, and the difference tells us something important about estimands.

## 10.1 The Balancing Score Theorem

We begin with potential outcomes. For each unit $i$, let $Y_i(1)$ denote the outcome under treatment ($D_i = 1$, Medicaid enrollment) and $Y_i(0)$ denote the outcome under control. We observe $Y_i = D_i Y_i(1) + (1-D_i) Y_i(0)$ and a vector of pre-treatment covariates $X_i$.

The fundamental identifying assumption in observational studies is **unconfoundedness** (also called conditional ignorability or selection on observables):

$$
(Y_i(0), Y_i(1)) \perp D_i \mid X_i \tag{10.1}
$$

Under (10.1), naive regression of $Y$ on $D$ conditional on $X$ recovers the average treatment effect. The propensity score is defined as the conditional probability of treatment given covariates:

$$
p(x) = \Pr(D_i = 1 \mid X_i = x) \tag{10.2}
$$

The central result of Rosenbaum and Rubin (1983) is that the propensity score is a *balancing score*—a coarsening of $X$ that is sufficient for removing confounding.

**Theorem 10.1** (Rosenbaum-Rubin Balancing Score Theorem). *Under unconfoundedness (10.1) and the positivity condition $0 < p(x) < 1$ for all $x$ in the support of $X$, we have:*

$$(Y_i(0), Y_i(1)) \perp D_i \mid p(X_i) \tag{10.3}$$

*Proof.* It suffices to show $\Pr(D_i = 1 \mid Y_i(0), Y_i(1), p(X_i)) = \Pr(D_i = 1 \mid p(X_i)) = p(X_i)$.

By the law of iterated expectations:
$$
\Pr(D_i = 1 \mid Y_i(0), Y_i(1), p(X_i)) = E[D_i \mid Y_i(0), Y_i(1), p(X_i)]
$$
$$
= E\left[ E[D_i \mid Y_i(0), Y_i(1), X_i] \mid Y_i(0), Y_i(1), p(X_i) \right]
$$

By unconfoundedness (10.1), $E[D_i \mid Y_i(0), Y_i(1), X_i] = E[D_i \mid X_i] = p(X_i)$. Thus:
$$
= E[p(X_i) \mid Y_i(0), Y_i(1), p(X_i)] = p(X_i)
$$

The last equality holds because $p(X_i)$ is already determined. $\blacksquare$

The theorem says something remarkable: if unconfoundedness holds conditional on the full covariate vector $X$ (which may be high-dimensional), it also holds conditional on the scalar propensity score $p(X)$. This is a *dimension reduction* result. Matching or weighting on a single number—rather than a high-dimensional vector—is sufficient to remove confounding, provided the propensity score model is correctly specified.

Notice what the theorem does *not* say. It does not say that estimating $\hat{p}(X)$ by logistic regression and then conditioning on $\hat{p}$ removes confounding. The theorem is about the *true* propensity score. Estimation error in $\hat{p}$ introduces bias; the direction and magnitude of that bias depend on the functional form assumptions embedded in the estimation procedure. This point—often glossed over in applied work—motivates the doubly robust estimators we discuss in Chapter 11.

## 10.2 Overlap and Positivity: When Propensity Score Methods Fail Completely

The positivity condition $0 < p(x) < 1$ is not merely a technical regularity condition. It is a genuine empirical constraint with serious consequences when violated.

**Definition 10.1** (Overlap). The *overlap* condition holds if $0 < p(x) < 1$ for all $x$ in the support of $X$. A region of covariate space where $p(x) \approx 0$ or $p(x) \approx 1$ is called a region of *limited overlap* or *near-violation of positivity*.

When overlap fails for some subgroup, the ATE for that subgroup is not identified by any method that relies on unconfoundedness—not regression, not matching, not IPW. This is a fundamental identification failure, not a statistical efficiency problem.

The consequences for different estimators are asymmetric. Regression extrapolates through regions of non-overlap using the parametric model; the extrapolation may or may not be trustworthy. IPW, by contrast, assigns extreme weights $1/\hat{p}(x)$ to treated observations in regions where $p(x) \approx 0$, causing the estimator to blow up in finite samples. This is often called the *weight instability* problem.

**Definition 10.2** (Estimand Shift Under Trimming). Let $\mathcal{T}_\alpha = \{i : \alpha \leq \hat{p}(X_i) \leq 1-\alpha\}$ be the set of units satisfying a trimming rule with threshold $\alpha$. The trimmed estimator recovers not the ATE but the ATE for the *trimmed population*:
$$
\tau_\alpha = E[Y_i(1) - Y_i(0) \mid p(X_i) \in [\alpha, 1-\alpha]]
$$

This is not the ATE, not the ATT, and generally not a pre-specified target estimand. Trimming resolves the numerical problem at the cost of silently changing what is being estimated. The applied researcher must decide: is the trimmed estimand the relevant policy question? In the OHE context, trimming units with very high or very low predicted enrollment probability means we estimate the effect of Medicaid for those in the middle of the enrollment propensity distribution—arguably not the most policy-relevant group.

## 10.3 Matching Estimators

Matching estimators construct a counterfactual for each unit by finding similar units on the other side of the treatment boundary. The key insight from Theorem 10.1 is that similarity can be measured using the scalar propensity score rather than the full covariate vector.

**Definition 10.3** (Nearest-Neighbor Matching). The nearest-neighbor matching estimator with $k$ matches assigns to each treated unit $i$ a matched set $\mathcal{M}(i)$ of $k$ control units minimizing $|\hat{p}(X_i) - \hat{p}(X_j)|$ subject to $D_j = 0$. The ATT estimator is:
$$
\hat{\tau}_\text{NNM} = \frac{1}{n_1} \sum_{i: D_i=1} \left( Y_i - \frac{1}{k}\sum_{j \in \mathcal{M}(i)} Y_j \right) \tag{10.4}
$$

Note that (10.4) estimates the ATT, not the ATE—treated units are matched to controls, not the reverse. If the estimand of interest is the ATE, one must also match each control unit to treated units and average, or use a different method.

**Caliper matching** imposes a maximum distance constraint: $|\hat{p}(X_i) - \hat{p}(X_j)| \leq c$. The caliper $c$ is typically chosen as a fraction of the standard deviation of the log-odds: Cochran and Rubin (1973) suggest $c = 0.2\sigma_{\log\text{odds}}$. Units without a match within the caliper are dropped—again changing the estimand.

**Mahalanobis distance matching** bypasses the propensity score entirely and matches on the covariate vector directly:
$$
d_M(i,j) = (X_i - X_j)^\top \hat{\Sigma}^{-1} (X_i - X_j)
$$
where $\hat{\Sigma}$ is the sample covariance of $X$. This avoids propensity score misspecification but returns the curse of dimensionality for high-dimensional $X$.

A critical practical issue: matching with replacement reduces bias by allowing better matches but introduces a correlation structure that must be accounted for in variance estimation. Standard errors that ignore the matching weights are incorrect.

## 10.4 Inverse Probability Weighting

IPW estimators reweight the observed data to create a pseudo-population in which treatment is independent of covariates. The Horvitz-Thompson (HT) estimator is:

$$
\hat{\tau}_\text{HT} = \frac{1}{n}\sum_{i=1}^n \left( \frac{D_i Y_i}{\hat{p}(X_i)} - \frac{(1-D_i)Y_i}{1-\hat{p}(X_i)} \right) \tag{10.5}
$$

The Hájek estimator normalizes the weights within each treatment arm:

$$
\hat{\tau}_\text{Hajek} = \frac{\sum_i D_i Y_i / \hat{p}(X_i)}{\sum_i D_i / \hat{p}(X_i)} - \frac{\sum_i (1-D_i) Y_i / (1-\hat{p}(X_i))}{\sum_i (1-D_i) / (1-\hat{p}(X_i))} \tag{10.6}
$$

**Proposition 10.1** (Consistency of HT Estimator). *Under unconfoundedness and positivity, and assuming the propensity score is known (or consistently estimated at $\sqrt{n}$ rate), $\hat{\tau}_\text{HT} \xrightarrow{p} \tau_\text{ATE}$.*

The Hájek estimator is generally preferred in practice. Its self-normalization makes it more stable when propensity scores are near zero or one, because the extreme weights in numerator and denominator partially cancel. The HT estimator is unbiased in theory but can have enormous variance when weights are extreme; the Hájek estimator trades a small bias for substantial variance reduction.

**Stabilized weights** provide a related improvement. Instead of $w_i = D_i / p(X_i) + (1-D_i)/(1-p(X_i))$, use:
$$
\tilde{w}_i = D_i \cdot \frac{\Pr(D_i=1)}{p(X_i)} + (1-D_i) \cdot \frac{\Pr(D_i=0)}{1-p(X_i)}
$$

The stabilizing factor $\Pr(D=d)$ is estimated by the marginal treatment rate. Stabilized weights have the same mean ($\approx 1$) for treated and controls separately, which aids diagnostics.

## 10.5 Why Flexible Propensity Score Models Can Hurt

There is a widely-held intuition that estimating the propensity score with a more flexible model—LASSO, random forest, neural network—should improve causal estimates. This intuition is wrong in general, and the reason is subtle.

The propensity score's role is *not* prediction. A perfectly predicted propensity score with $R^2 \approx 1$ is a signal of overlap failure, not success. What matters for causal estimation is not that $\hat{p}(X)$ approximates $p(X)$ well in prediction loss, but that the resulting weights or matches achieve covariate balance.

More concretely: a highly flexible propensity score model can overfit to noise in the covariates, creating extreme weights for idiosyncratic reasons unrelated to the confounding structure. Hirano and Imbens (2001) show that the semiparametrically efficient IPW estimator uses propensity scores estimated in a specific way—not simply "as flexibly as possible." The efficiency bound depends on the *outcome* model, not just the treatment model. When the propensity score is estimated too flexibly, IPW variance can actually *increase* relative to a correctly-specified parametric model.

This is the key insight often missed: **propensity score estimation is not a prediction problem**. The right criterion for a propensity score model is not cross-validated AUC; it is the resulting covariate balance in the weighted or matched sample. This motivates the covariate-balancing approaches in the next section.

## 10.6 Entropy Balancing and Covariate Balancing Propensity Score

Entropy balancing (Hainmueller 2012) inverts the logic of propensity score methods. Rather than estimating a propensity score and hoping it produces balance, entropy balancing directly solves for observation weights that achieve exact covariate balance, subject to a minimum-entropy constraint.

**Definition 10.4** (Entropy Balancing Weights). Let $q_i = 1/n_0$ for $i$ in the control group (uniform baseline weights). The entropy balancing weights solve:

$$
\min_{w} \sum_{i: D_i=0} w_i \log\left(\frac{w_i}{q_i}\right) \tag{10.7}
$$

subject to:
$$
\sum_{i: D_i=0} w_i c_r(X_i) = m_r, \quad r = 1, \ldots, R \tag{10.8}
$$
$$
\sum_{i: D_i=0} w_i = 1, \quad w_i \geq 0 \tag{10.9}
$$

where $c_r(X_i)$ are balance constraints (typically $c_r(X_i) = X_{ir}$ for mean balance, or $c_r(X_i) = X_{ir}^2$ for variance balance) and $m_r = \frac{1}{n_1}\sum_{i: D_i=1} c_r(X_i)$ are the corresponding treated-group moments.

The objective (10.7) is the KL divergence from the uniform distribution; minimizing it finds the weights closest to uniform that satisfy the balance constraints. This makes the reweighted control distribution as close as possible to the treated distribution in covariate moments, without imposing any functional form on how covariates relate to treatment.

**Proposition 10.2** (Dual Representation). *The primal problem (10.7)-(10.9) has the dual:*
$$
\max_{\lambda, \gamma} - \sum_{i: D_i=0} q_i \exp\left(\sum_r \lambda_r c_r(X_i) + \gamma\right) - \sum_r \lambda_r m_r - \gamma \tag{10.10}
$$

*The optimal primal weights recover as $w_i^* \propto q_i \exp(\sum_r \lambda_r^* c_r(X_i))$.*

This dual representation reveals that entropy balancing weights have the form of an exponential family tilting of the base distribution—a fact that connects entropy balancing to exponential tilting in statistics and to the maximum entropy principle in information theory.

The **Covariate Balancing Propensity Score (CBPS)** of Imai and Ratkovic (2014) takes a different approach: it estimates a logistic regression model for the propensity score, but chooses the coefficients to optimize covariate balance rather than likelihood. CBPS adds moment conditions of the form $E[X_i(D_i - p(X_i; \beta))] = 0$ to the logistic score equations and solves via GMM. The result is a propensity score model that, by construction, produces a balanced weighted sample.

## 10.7 Diagnostics: Love Plots and Standardized Mean Differences

No propensity score analysis is complete without balance diagnostics. The standardized mean difference (SMD) for covariate $k$ is:

$$
\text{SMD}_k = \frac{\bar{X}_{k,1} - \bar{X}_{k,0}^w}{s_k} \tag{10.11}
$$

where $\bar{X}_{k,1}$ is the treated mean, $\bar{X}_{k,0}^w$ is the (possibly weighted or matched) control mean, and $s_k = \sqrt{(s_{k,1}^2 + s_{k,0}^2)/2}$ is the pooled standard deviation. A conventional threshold is $|\text{SMD}_k| < 0.1$, though this threshold has no theoretical justification—it is a rule of thumb.

A **Love plot** displays the SMD for each covariate before and after weighting or matching, allowing visual assessment of balance improvement. The Love plot is a diagnostic, not a validity certificate: good Love plots are necessary but not sufficient for valid causal inference, because they only assess balance on observed covariates.

## Python: Propensity Scores on the Oregon Health Experiment

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import optimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load OHE Data ──────────────────────────────────────────────────────────
# Download from https://data.nber.org/oregon/
# Files used: oregonhie_descriptive_vars.dta, oregonhie_survey12m_vars.dta

try:
    import pyreadstat
    desc, _ = pyreadstat.read_dta("oregonhie_descriptive_vars.dta")
    surv, _ = pyreadstat.read_dta("oregonhie_survey12m_vars.dta")
except Exception:
    # Simulate representative data with OHE-like structure for illustration
    rng = np.random.default_rng(42)
    n = 23741
    
    # Covariates: household size categories (from numhh_list strata)
    numhh = rng.choice([1, 2, 3], size=n, p=[0.60, 0.28, 0.12])
    
    # Age (self-reported in survey; approximate distribution)
    age = rng.normal(40, 12, n).clip(19, 64).astype(int)
    female = rng.binomial(1, 0.56, n)
    english = rng.binomial(1, 0.85, n)
    
    # Lottery selection probability depends on household size (design feature)
    base_select = np.where(numhh == 1, 0.30, np.where(numhh == 2, 0.45, 0.60))
    selected = rng.binomial(1, base_select, n)
    
    # Enrollment (treatment): selected individuals more likely to enroll
    enroll_prob = 0.05 + 0.25 * selected + 0.02 * female - 0.001 * (age - 40)
    enroll_prob = np.clip(enroll_prob, 0.01, 0.95)
    enrolled = rng.binomial(1, enroll_prob, n)
    
    # Outcomes
    # doc_any_12m: any doctor visit in 12 months
    doc_prob = 0.55 + 0.10 * enrolled + 0.05 * female + 0.003 * (age - 40)
    doc_prob = np.clip(doc_prob, 0, 1)
    doc_any = rng.binomial(1, doc_prob, n)
    
    # catastrophic_exp_inp: catastrophic inpatient expenditure
    cat_prob = 0.08 - 0.03 * enrolled + 0.01 * (age - 40) / 10
    cat_prob = np.clip(cat_prob, 0, 1)
    cat_exp = rng.binomial(1, cat_prob, n)
    
    df = pd.DataFrame({
        'person_id':        np.arange(n),
        'selected':         selected,
        'ohp_all_ever_admin': enrolled,
        'numhh_list':       numhh,
        'age':              age,
        'female':           female,
        'english':          english,
        'doc_any_12m':      doc_any,
        'catastrophic_exp_inp': cat_exp
    })
    df = df.dropna(subset=['ohp_all_ever_admin','doc_any_12m'])
    print(f"Using simulated OHE-like data  n={len(df)}")

# ── 2. Prepare Analysis Dataset ───────────────────────────────────────────────
# Treatment: Medicaid enrollment
D = df['ohp_all_ever_admin'].values.astype(float)
Y_doc = df['doc_any_12m'].values.astype(float)
Y_cat = df['catastrophic_exp_inp'].values.astype(float)

# Covariates for propensity score
cov_cols = ['numhh_list', 'age', 'female', 'english']
X_raw = df[cov_cols].copy()

# Add squared terms and an interaction
X_raw['age_sq']       = X_raw['age'] ** 2
X_raw['female_age']   = X_raw['female'] * X_raw['age']
X_raw['numhh_female'] = X_raw['numhh_list'] * X_raw['female']

feature_names = X_raw.columns.tolist()
X = X_raw.values.astype(float)

# Standardize for logistic regression
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

print(f"\nSample: n={len(D)},  treated={D.mean():.3f},  "
      f"doc_any={Y_doc.mean():.3f},  cat_exp={Y_cat.mean():.3f}")

# ── 3. Estimate Propensity Score ──────────────────────────────────────────────
lr = LogisticRegression(C=1.0, max_iter=500, random_state=0)
lr.fit(X_sc, D)
ps = lr.predict_proba(X_sc)[:, 1]

print(f"\nPropensity score: min={ps.min():.4f}, "
      f"mean={ps.mean():.4f}, max={ps.max():.4f}")
print(f"Near-zero (<0.05): {(ps<0.05).mean():.3f}  "
      f"Near-one (>0.95): {(ps>0.95).mean():.3f}")

# ── 4. IPW and Hájek Estimators ───────────────────────────────────────────────
TRIM = 0.01  # symmetric trimming threshold

def hajek_ate(D, Y, ps, trim=TRIM):
    """Hájek (self-normalized IPW) ATE estimator with optional trimming."""
    mask = (ps >= trim) & (ps <= 1 - trim)
    D_, Y_, ps_ = D[mask], Y[mask], ps[mask]
    
    w1 = D_ / ps_
    w0 = (1 - D_) / (1 - ps_)
    
    mu1 = np.sum(w1 * Y_) / np.sum(w1)
    mu0 = np.sum(w0 * Y_) / np.sum(w0)
    return mu1 - mu0, mu1, mu0, mask.sum()

def ht_ate(D, Y, ps, trim=TRIM):
    """Horvitz-Thompson (unnormalized IPW) ATE estimator."""
    mask = (ps >= trim) & (ps <= 1 - trim)
    D_, Y_, ps_ = D[mask], Y[mask], ps[mask]
    n = len(D_)
    
    tau = (1/n) * np.sum(D_ * Y_ / ps_ - (1 - D_) * Y_ / (1 - ps_))
    return tau

tau_hajek_doc, mu1_doc, mu0_doc, n_trim = hajek_ate(D, Y_doc, ps)
tau_hajek_cat, mu1_cat, mu0_cat, _      = hajek_ate(D, Y_cat, ps)
tau_ht_doc  = ht_ate(D, Y_doc, ps)
tau_ht_cat  = ht_ate(D, Y_cat, ps)

print(f"\n── IPW Estimates ──")
print(f"{'':30s}  {'Doc visit':>10}  {'Cat. exp.':>10}")
print(f"{'Hájek ATE':30s}  {tau_hajek_doc:>+10.4f}  {tau_hajek_cat:>+10.4f}")
print(f"{'HT ATE':30s}  {tau_ht_doc:>+10.4f}  {tau_ht_cat:>+10.4f}")
print(f"Trimmed n: {n_trim} ({n_trim/len(D):.1%} of sample)")

# ── 5. Nearest-Neighbor Matching (ATT) ───────────────────────────────────────
def nn_match_att(D, Y, ps, k=1, caliper=None):
    """
    Nearest-neighbor propensity-score matching, ATT.
    Returns (tau_att, matched_pairs) where matched_pairs is an array
    of (treated_idx, control_idx) tuples.
    """
    treated_idx  = np.where(D == 1)[0]
    control_idx  = np.where(D == 0)[0]
    ps_ctrl      = ps[control_idx]
    
    diffs  = []
    pairs  = []
    
    for ti in treated_idx:
        distances = np.abs(ps[ti] - ps_ctrl)
        order     = np.argsort(distances)[:k]
        
        if caliper is not None and distances[order[0]] > caliper:
            continue
        
        matches = control_idx[order]
        pairs.append((ti, matches))
        diffs.append(Y[ti] - Y[matches].mean())
    
    tau_att = np.mean(diffs)
    return tau_att, pairs

sigma_logodds = np.std(np.log(ps / (1 - ps)))
caliper = 0.2 * sigma_logodds  # Cochran-Rubin rule

tau_nnm_doc, pairs_doc = nn_match_att(D, Y_doc, ps, k=1, caliper=caliper)
tau_nnm_cat, pairs_cat = nn_match_att(D, Y_cat, ps, k=1, caliper=caliper)

print(f"\n── Nearest-Neighbor Matching (ATT, caliper={caliper:.4f}) ──")
print(f"Matched pairs: {len(pairs_doc)}")
print(f"ATT doc_any_12m: {tau_nnm_doc:+.4f}")
print(f"ATT cat_exp_inp: {tau_nnm_cat:+.4f}")

# ── 6. Entropy Balancing ──────────────────────────────────────────────────────
def entropy_balancing(X_ctrl, target_means, base_weights=None):
    """
    Solve the entropy balancing problem (Hainmueller 2012).
    Minimizes KL(w || q) subject to X_ctrl.T @ w = target_means
    and sum(w) = 1.

    Uses the dual: w_i* ∝ q_i * exp(lambda @ X_ctrl[i])
    and optimizes lambda via L-BFGS-B on the dual objective.
    """
    n, p = X_ctrl.shape
    if base_weights is None:
        q = np.ones(n) / n
    else:
        q = base_weights / base_weights.sum()

    def dual_obj_grad(lam):
        log_w = np.log(q) + X_ctrl @ lam
        # Stabilize by subtracting max
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        
        obj  = -np.dot(lam, target_means) - np.log((q * np.exp(X_ctrl @ lam)).sum())
        grad = X_ctrl.T @ w - target_means
        return -obj, -grad          # we minimize, so negate

    lam0   = np.zeros(p)
    result = optimize.minimize(dual_obj_grad, lam0,
                               method='L-BFGS-B', jac=True,
                               options={'maxiter': 2000, 'ftol': 1e-12})
    
    lam_star = result.x
    log_w    = np.log(q) + X_ctrl @ lam_star
    log_w   -= log_w.max()
    w        = np.exp(log_w)
    w       /= w.sum()
    return w, result

# Use raw (unstandardized) covariates for interpretable balance constraints
X_eb = X.copy()
treated_mask  = D == 1
control_mask  = D == 0

target_means = X_eb[treated_mask].mean(axis=0)  # treated covariate means
X_ctrl_eb    = X_eb[control_mask]

eb_weights, eb_result = entropy_balancing(X_ctrl_eb, target_means)
print(f"\n── Entropy Balancing ──")
print(f"Optimization success: {eb_result.success}  "
      f"Max |balance error|: {np.abs(X_ctrl_eb.T @ eb_weights - target_means).max():.2e}")

# ATT from entropy-balanced weights (treated at 1/n_1, controls at eb_weights)
n1 = treated_mask.sum()
mu1_eb = Y_doc[treated_mask].mean()
mu0_eb_doc = np.dot(eb_weights, Y_doc[control_mask])
tau_eb_doc = mu1_eb - mu0_eb_doc

mu0_eb_cat = np.dot(eb_weights, Y_cat[control_mask])
tau_eb_cat = Y_cat[treated_mask].mean() - mu0_eb_cat

print(f"ATT doc_any_12m : {tau_eb_doc:+.4f}")
print(f"ATT cat_exp_inp  : {tau_eb_cat:+.4f}")

# ── 7. Naïve OLS (for comparison) ─────────────────────────────────────────────
from numpy.linalg import lstsq

def ols_coef(Y, D, X):
    """OLS coefficient on D after partialing out X (Frisch-Waugh)."""
    Xd = np.column_stack([np.ones(len(D)), X, D])
    coef, _, _, _ = lstsq(Xd, Y, rcond=None)
    return coef[-1]

tau_ols_doc = ols_coef(Y_doc, D, X)
tau_ols_cat = ols_coef(Y_cat, D, X)

# ── 8. Standardized Mean Differences (Love Plot) ──────────────────────────────
def compute_smd(X, D, weights_ctrl=None):
    """
    Compute SMD for each covariate.
    weights_ctrl: observation weights for control group (used for entropy balancing).
    If None, uses uniform weights (unweighted or IPW).
    """
    smds = []
    for k in range(X.shape[1]):
        x1 = X[D == 1, k]
        x0 = X[D == 0, k]
        
        mean1 = x1.mean()
        
        if weights_ctrl is not None:
            mean0 = np.dot(weights_ctrl, x0) / weights_ctrl.sum()
            var0  = np.dot(weights_ctrl, (x0 - mean0)**2) / weights_ctrl.sum()
        else:
            mean0 = x0.mean()
            var0  = x0.var()
        
        pooled_sd = np.sqrt((x1.var() + var0) / 2)
        smd = (mean1 - mean0) / (pooled_sd + 1e-12)
        smds.append(smd)
    return np.array(smds)

# IPW weights for SMD calculation (Hájek-style, controls only)
ipw_w_ctrl = (1 - ps[control_mask]) ** (-1)   # 1 / (1-p) for controls

smd_raw   = compute_smd(X, D)
smd_ipw   = compute_smd(X, D, weights_ctrl=ipw_w_ctrl)
smd_eb    = compute_smd(X, D, weights_ctrl=eb_weights)

# ── Love Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

y_pos    = np.arange(len(feature_names))
offset   = 0.22

ax.scatter(smd_raw, y_pos + offset,  marker='o', s=60,
           color='#d62728', label='Unadjusted', zorder=3)
ax.scatter(smd_ipw, y_pos,            marker='s', s=60,
           color='#1f77b4', label='IPW (Hájek)', zorder=3)
ax.scatter(smd_eb, y_pos - offset,   marker='^', s=60,
           color='#2ca02c', label='Entropy Balancing', zorder=3)

ax.axvline(0,    color='black', lw=0.8)
ax.axvline( 0.1, color='gray',  lw=0.8, ls='--', alpha=0.7)
ax.axvline(-0.1, color='gray',  lw=0.8, ls='--', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names, fontsize=9)
ax.set_xlabel('Standardized Mean Difference', fontsize=11)
ax.set_title('Love Plot: Covariate Balance\n(OHE — Medicaid Enrollment)', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(-0.6, 0.6)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('love_plot_ohe.png', dpi=150)
plt.show()
print("Love plot saved: love_plot_ohe.png")

# ── 9. Compare All Estimators ─────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Estimator':35s}  {'doc_any':>8}  {'cat_exp':>8}  {'Estimand':>6}")
print("="*60)
print(f"{'Naïve OLS':35s}  {tau_ols_doc:>+8.4f}  {tau_ols_cat:>+8.4f}  {'ATE':>6}")
print(f"{'Hájek IPW':35s}  {tau_hajek_doc:>+8.4f}  {tau_hajek_cat:>+8.4f}  {'ATE':>6}")
print(f"{'HT IPW':35s}  {tau_ht_doc:>+8.4f}  {tau_ht_cat:>+8.4f}  {'ATE':>6}")
print(f"{'NN Matching (caliper)':35s}  {tau_nnm_doc:>+8.4f}  {tau_nnm_cat:>+8.4f}  {'ATT':>6}")
print(f"{'Entropy Balancing':35s}  {tau_eb_doc:>+8.4f}  {tau_eb_cat:>+8.4f}  {'ATT':>6}")
print("="*60)

# ── 10. PS Overlap Plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, (label, mask) in zip(axes, [('Control (D=0)', D==0), ('Treated (D=1)', D==1)]):
    ax.hist(ps[mask], bins=50, density=True, alpha=0.75,
            color='#1f77b4' if 'Control' in label else '#d62728',
            edgecolor='white', linewidth=0.3)
    ax.axvline(TRIM,     color='black', ls='--', lw=1.2, label=f'Trim={TRIM}')
    ax.axvline(1-TRIM,   color='black', ls='--', lw=1.2)
    ax.set_title(f'PS Distribution — {label}', fontsize=11)
    ax.set_xlabel('Estimated Propensity Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle('Overlap Check: Oregon Health Experiment', fontsize=12)
plt.tight_layout()
plt.savefig('ps_overlap_ohe.png', dpi=150)
plt.show()
print("Overlap plot saved: ps_overlap_ohe.png")
```

## Summary

The propensity score theorem guarantees that conditioning on a scalar—the probability of treatment—is sufficient to remove confounding, provided unconfoundedness holds and the propensity score is correctly specified. This dimension reduction is the core theoretical contribution of Rosenbaum and Rubin (1983).

Four practical lessons emerge from the analysis of the OHE data.

First, estimand clarity is not optional. Matching estimators recover the ATT; IPW estimators recover the ATE (or the trimmed-sample ATE when trimming is applied). These are different parameters and they differ numerically in the OHE application. The question "does Medicaid improve outcomes for people who enrolled?" (ATT) is distinct from "does Medicaid improve outcomes for a randomly selected individual?" (ATE). Both are legitimate, but they must be distinguished before estimation, not after.

Second, the Love plot is a diagnostic, not a conclusion. Good balance on observed covariates says nothing about balance on unobserved covariates. The OHE is special precisely because it has a valid instrument (the lottery) that identifies the LATE under weaker assumptions than unconfoundedness. Comparing the IV-based LATE from Chapter 9 to the propensity-score-based ATE reveals not merely numerical differences but fundamental differences in what the two methods require: IV requires exclusion restriction and monotonicity; propensity score methods require no unmeasured confounders. In the OHE, the IV estimate should be treated as the benchmark; the observational estimates' deviation from it measures the residual confounding not captured by the available covariates.

Third, entropy balancing dominates logistic-regression-based IPW in the Love plot because it targets balance directly rather than via an intermediate propensity score model. Its weights are the unique minimum-entropy solution that achieves exact moment balance, and they degrade gracefully when balance constraints are tightened.

Fourth, flexible propensity score estimation is not a free lunch. Overfitting the propensity score to sample noise creates extreme weights with no corresponding reduction in confounding bias. The right criterion for evaluating a propensity score model is post-weighting covariate balance, not held-out prediction accuracy.

## Further Reading

**Foundational papers**. Rosenbaum and Rubin (1983) "The Central Role of the Propensity Score in Observational Studies for Causal Effects" (*Biometrika*) is the original theorem. Rubin (1974) "Estimating Causal Effects of Treatments in Randomized and Non-Randomized Studies" (*Journal of Educational Psychology*) establishes the potential outcomes framework.

**IPW and efficiency**. Hirano, Imbens, and Ridder (2003) "Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score" (*Econometrica*) derives the efficiency bound and shows that nonparametric estimation of the propensity score achieves it. Hájek (1971) provides the classical result on self-normalized estimators.

**Entropy balancing and CBPS**. Hainmueller (2012) "Entropy Balancing for Causal Effects" (*Political Analysis*) introduces entropy balancing with the dual derivation. Imai and Ratkovic (2014) "Covariate Balancing Propensity Score" (*Journal of the Royal Statistical Society B*) introduces CBPS.

**Overlap and trimming**. Crump, Hotz, Imbens, and Mitnik (2009) "Dealing with Limited Overlap in Estimation of Average Treatment Effects" (*Biometrika*) provides formal rules for optimal trimming thresholds and characterizes the resulting estimand.

**Flexible PS estimation risks**. Kang and Schafer (2007) "Demystifying Double Robustness" (*Statistical Science*) with discussion contains the notorious simulation showing that flexible propensity score models can badly worsen IPW estimates when outcomes are simply modeled. King and Nielsen (2019) "Why Propensity Scores Should Not Be Used for Matching" (*Political Analysis*) is a provocative critique of PS matching specifically, arguing Mahalanobis matching dominates in finite samples.

**OHE references**. Finkelstein et al. (2012) "The Oregon Health Insurance Experiment: Evidence from the First Year" (*Quarterly Journal of Economics*) is the primary study. Baicker et al. (2013) "The Oregon Experiment—Effects of Medicaid on Clinical Outcomes" (*New England Journal of Medicine*) covers the two-year clinical outcomes.
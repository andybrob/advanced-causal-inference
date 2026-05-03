# Chapter 12: Double/Debiased Machine Learning

Classical econometrics proceeds by specifying a parametric model for the full joint distribution of outcomes, treatments, and covariates, then estimating all parameters simultaneously. This approach has a fatal flaw in high-dimensional settings: regularization bias. When the number of potential controls $p$ is large relative to the sample size $n$, ordinary least squares is inconsistent, and regularized estimators like Lasso introduce shrinkage bias that contaminates inference on the parameter of interest $\theta$. The bias from estimating nuisance functions—the conditional expectation functions $E[Y|X]$ and $E[D|X]$—does not vanish fast enough to permit $\sqrt{n}$-consistent inference on $\theta$ by naive plugin.

Double/Debiased Machine Learning (DML), developed by Chernozhukov et al. (2018), resolves this with two ideas that work in concert. First, **Neyman orthogonality**: design the estimating equation so that its sensitivity to first-order perturbations in the nuisance functions is zero. Second, **cross-fitting**: use sample splitting to prevent overfitting from polluting the score. Together, these reduce the requirement on nuisance estimation from $n^{-1/2}$ to $n^{-1/4}$, opening the door to any modern ML method as a nuisance estimator while preserving valid frequentist inference on $\theta$.

## 12.1 The Problem with Naive Plugin

Consider the partially linear model (PLM):

$$Y_i = \theta_0 D_i + g_0(X_i) + \varepsilon_i, \quad E[\varepsilon_i \mid D_i, X_i] = 0$$

where $Y_i$ is an outcome, $D_i$ is a treatment (binary or continuous), $X_i \in \mathbb{R}^p$ is a high-dimensional covariate vector, $g_0$ is an unknown function, and $\theta_0$ is the scalar causal effect of interest.

The naive approach estimates $g_0$ by regularized regression of $Y$ on $X$ (ignoring $D$), then regresses the residual on $D$. Let $\hat{g}$ be this estimator. The plugin score is:

$$\hat{\theta}_{\text{naive}} = \frac{\sum_i D_i (Y_i - \hat{g}(X_i))}{\sum_i D_i^2}$$

The bias of this estimator depends on the estimation error $\hat{g} - g_0$. Writing $\ell_0(X_i) = E[D_i \mid X_i]$, a simple calculation gives:

$$\sqrt{n}(\hat{\theta}_{\text{naive}} - \theta_0) = \frac{\frac{1}{\sqrt{n}}\sum_i D_i \varepsilon_i}{\frac{1}{n}\sum_i D_i^2} - \frac{\frac{1}{\sqrt{n}}\sum_i D_i (\hat{g}(X_i) - g_0(X_i))}{\frac{1}{n}\sum_i D_i^2}$$

The second term is a bias term that involves $\sum_i D_i (\hat{g}(X_i) - g_0(X_i))$. By Cauchy-Schwarz:

$$\left|\frac{1}{n}\sum_i D_i (\hat{g}(X_i) - g_0(X_i))\right| \leq \|\hat{g} - g_0\|_2 \cdot \left(\frac{1}{n}\sum_i D_i^2\right)^{1/2}$$

For $\sqrt{n}$-consistency we need $\sqrt{n}\|\hat{g} - g_0\|_2 \to 0$, i.e., $\|\hat{g} - g_0\|_2 = o(n^{-1/2})$. This is the classical parametric rate—precisely what Lasso and other regularized estimators cannot achieve when the true $g_0$ is not sparse or when $p$ grows with $n$. Lasso achieves rate $n^{-1/4}$ under approximate sparsity conditions; random forests and gradient boosting achieve similar rates under smoothness conditions. None reaches $n^{-1/2}$.

The naive plugin fails because the score function $\psi_{\text{naive}}(W; \theta, g) = D(Y - \theta D - g(X))$ is *not* Neyman orthogonal: its Gateaux derivative with respect to $g$ in the direction $h$ is $-E[D \cdot h(X)]$, which is generically nonzero.

## 12.2 Neyman Orthogonality

**Definition 12.1 (Neyman Orthogonality).** Let $W = (Y, D, X)$, $\theta \in \Theta \subset \mathbb{R}^d$, and $\eta = (g, \ell) \in \mathcal{T}$ a nuisance parameter. A score function $\psi(W; \theta, \eta)$ satisfies **Neyman orthogonality** at $(\theta_0, \eta_0)$ if:

1. $E[\psi(W; \theta_0, \eta_0)] = 0$ (moment condition)
2. $\partial_\eta E[\psi(W; \theta_0, \eta_0)][h] = 0$ for all directions $h$ in the tangent set (orthogonality)

where $\partial_\eta[\cdot][h]$ denotes the Gateaux derivative in direction $h$.

The intuition is a second-order Taylor expansion. If $\hat{\eta}$ is an estimator of $\eta_0$, then:

$$E[\psi(W; \theta_0, \hat{\eta})] \approx E[\psi(W; \theta_0, \eta_0)] + \underbrace{\partial_\eta E[\psi(W; \theta_0, \eta_0)][\hat{\eta} - \eta_0]}_{= 0 \text{ by orthogonality}} + O(\|\hat{\eta} - \eta_0\|^2)$$

The first-order bias term vanishes by construction, leaving only a second-order remainder proportional to $\|\hat{\eta} - \eta_0\|^2$. If $\|\hat{\eta} - \eta_0\|_2 = O_p(n^{-1/4})$, then $n^{-1/2}\|\hat{\eta} - \eta_0\|^2 = O_p(n^{-1/2}) \cdot O_p(n^{-1/2}) = o_p(n^{-1/2} \cdot \sqrt{n}) = o_p(1)$. This is the key rate relaxation.

**Constructing an Orthogonal Score for the PLM.** The Frisch-Waugh-Lovell theorem suggests the fix: partial out $X$ from both $Y$ and $D$. Define:

$$v_0(X) = D - \ell_0(X), \quad \tilde{Y} = Y - g_0(X), \quad \tilde{D} = D - \ell_0(X)$$

The **partialling-out score** is:

$$\psi(W; \theta, g, \ell) = (Y - \theta D - g(X))(D - \ell(X))$$

**Proposition 12.1.** The partialling-out score satisfies Neyman orthogonality at $(\theta_0, g_0, \ell_0)$.

*Proof.* We verify the two conditions. For the moment condition:

$$E[\psi(W; \theta_0, g_0, \ell_0)] = E[(Y - \theta_0 D - g_0(X))(D - \ell_0(X))] = E[\varepsilon_i \cdot v_0(X_i)] = 0$$

since $E[\varepsilon_i \mid X_i] = 0$ and $v_0(X_i)$ is $X_i$-measurable.

For orthogonality with respect to $g$ in direction $h_g$:

$$\partial_g E[\psi(W; \theta_0, g_0, \ell_0)][h_g] = -E[h_g(X)(D - \ell_0(X))] = -E[h_g(X) \cdot E[D - \ell_0(X) \mid X]] = 0$$

since $E[D \mid X] = \ell_0(X)$ by definition. For orthogonality with respect to $\ell$ in direction $h_\ell$:

$$\partial_\ell E[\psi(W; \theta_0, g_0, \ell_0)][h_\ell] = -E[(Y - \theta_0 D - g_0(X)) h_\ell(X)] = -E[\varepsilon_i h_\ell(X_i)] = 0$$

by the conditional mean independence assumption. $\square$

## 12.3 Cross-Fitting

Neyman orthogonality alone is insufficient. When $\hat{g}$ and $\hat{\ell}$ are fit on the same data used to evaluate the score, empirical process terms arise that can reintroduce $n^{-1/2}$ rates on the nuisance error. Cross-fitting eliminates these terms by ensuring estimation and evaluation occur on independent samples.

**Algorithm 12.1 (DML2 with $K$-Fold Cross-Fitting).**

1. Partition $\{1, \ldots, n\}$ into $K$ folds $I_1, \ldots, I_K$ of approximately equal size.
2. For each fold $k = 1, \ldots, K$:
   - Let $I_k^c = \{1, \ldots, n\} \setminus I_k$ be the complement.
   - Fit $\hat{g}^{(k)}$ by regressing $Y$ on $X$ using observations in $I_k^c$.
   - Fit $\hat{\ell}^{(k)}$ by regressing $D$ on $X$ using observations in $I_k^c$.
   - Compute residuals for $i \in I_k$: $\tilde{Y}_i = Y_i - \hat{g}^{(k)}(X_i)$, $\tilde{D}_i = D_i - \hat{\ell}^{(k)}(X_i)$.
3. Pool all residuals and compute:

$$\hat{\theta}_{\text{DML2}} = \frac{\sum_{k=1}^K \sum_{i \in I_k} \tilde{D}_i \tilde{Y}_i}{\sum_{k=1}^K \sum_{i \in I_k} \tilde{D}_i^2}$$

The DML2 estimator aggregates numerator and denominator across all folds before taking the ratio. The **DML1** alternative computes $\hat{\theta}^{(k)}$ within each fold and averages:

$$\hat{\theta}_{\text{DML1}} = \frac{1}{K} \sum_{k=1}^K \hat{\theta}^{(k)}, \quad \hat{\theta}^{(k)} = \frac{\sum_{i \in I_k} \tilde{D}_i \tilde{Y}_i}{\sum_{i \in I_k} \tilde{D}_i^2}$$

DML2 is preferred in practice because it is more numerically stable when individual fold denominators are small and because its theoretical analysis is cleaner. DML1 can exhibit higher variance when fold-level second stage regressions are poorly conditioned.

**Theorem 12.1 (DML Asymptotic Normality).** Under regularity conditions including: (i) Neyman orthogonality of $\psi$, (ii) $K$-fold cross-fitting with fixed $K$, (iii) nuisance estimation rates $\|\hat{g}^{(k)} - g_0\|_2 \cdot \|\hat{\ell}^{(k)} - \ell_0\|_2 = o_p(n^{-1/2})$, and (iv) bounded moments, the DML2 estimator satisfies:

$$\sqrt{n}(\hat{\theta}_{\text{DML2}} - \theta_0) \xrightarrow{d} \mathcal{N}(0, V_0)$$

where $V_0 = J_0^{-2} E[\psi(W; \theta_0, \eta_0)^2]$ and $J_0 = E[\partial_\theta \psi(W; \theta_0, \eta_0)] = E[\tilde{D}_i^2]$.

*Proof sketch.* The cross-fit score evaluated at the true $\theta_0$ decomposes as:

$$\frac{1}{\sqrt{n}} \sum_i \psi(W_i; \theta_0, \hat{\eta}^{(k(i))}) = \frac{1}{\sqrt{n}} \sum_i \psi(W_i; \theta_0, \eta_0) + R_n$$

The remainder $R_n$ involves two types of terms: (a) the Gateaux derivative term, which vanishes by Neyman orthogonality, and (b) a quadratic remainder $O(\|\hat{\eta} - \eta_0\|^2)$. Under condition (iii), $\sqrt{n} \cdot \|\hat{g} - g_0\|_2 \|\hat{\ell} - \ell_0\|_2 = o_p(1)$, so $R_n = o_p(1)$. Cross-fitting ensures the leading term satisfies the CLT by independence between $\hat{\eta}^{(k)}$ and observations in fold $k$, eliminating the need for Donsker-class conditions on the nuisance estimators. The result follows by the standard M-estimator delta method. $\square$

**Remark.** Condition (iii) requires the *product* of nuisance rates to be $o(n^{-1/2})$. If both $\hat{g}$ and $\hat{\ell}$ achieve rate $n^{-1/4}$, then the product is $n^{-1/2}$, satisfying the condition. This is the key rate relaxation: each nuisance only needs $n^{-1/4}$, not $n^{-1/2}$.

## 12.4 Variance Estimation and Inference

The sandwich variance estimator for DML2 is:

$$\hat{V} = \hat{J}^{-2} \cdot \frac{1}{n} \sum_{k=1}^K \sum_{i \in I_k} \hat{\psi}_i^2$$

where $\hat{\psi}_i = \tilde{Y}_i - \hat{\theta} \tilde{D}_i$ are the second-stage residuals (using out-of-fold predictions throughout) and $\hat{J} = \frac{1}{n} \sum_i \tilde{D}_i^2$. A $(1-\alpha)$ confidence interval is:

$$\hat{\theta} \pm z_{\alpha/2} \sqrt{\hat{V}/n}$$

This variance estimator is consistent under the same conditions as Theorem 12.1. The crucial point is that $\hat{\psi}_i$ must use out-of-fold predictions $\hat{g}^{(k(i))}$ and $\hat{\ell}^{(k(i))}$—using in-sample predictions would underestimate variance.

## 12.5 The Interactive Regression Model

For binary treatments, the partially linear model may be restrictive if treatment effect heterogeneity is substantial. The **interactive regression model** (IRM) relaxes this:

$$Y_i = g_0(D_i, X_i) + \varepsilon_i, \quad D_i = 1\{f_0(X_i) \geq U_i\}$$

where $E[\varepsilon_i \mid D_i, X_i] = 0$ and $U_i \perp D_i \mid X_i$. The average treatment effect is:

$$\theta_0^{\text{ATE}} = E[g_0(1, X_i) - g_0(0, X_i)]$$

The orthogonal score for the ATE in the IRM is the AIPW (augmented inverse propensity weighting) score from Chapter 11:

$$\psi^{\text{ATE}}(W; \theta, \mu_1, \mu_0, \pi) = \mu_1(X) - \mu_0(X) + \frac{D(Y - \mu_1(X))}{\pi(X)} - \frac{(1-D)(Y - \mu_0(X))}{1 - \pi(X)} - \theta$$

where $\mu_d(X) = E[Y \mid D=d, X]$ and $\pi(X) = P(D=1 \mid X)$. One can verify that the Gateaux derivatives of $E[\psi^{\text{ATE}}]$ with respect to $\mu_1$, $\mu_0$, and $\pi$ all vanish at the true values—the AIPW score achieves double robustness precisely because it is Neyman orthogonal. DML applied to the IRM is therefore equivalent to cross-fit AIPW.

## 12.6 High-Dimensional Controls: When OLS Fails and DML Succeeds

To build intuition for why high-dimensional controls break OLS, consider the PLM with $p > n$. OLS is not even defined. With $p < n$ but $p$ growing, the OLS estimator of $g_0$ has estimation error $\|\hat{g}_{\text{OLS}} - g_0\|_2 = O_p(\sqrt{p/n})$. The naive plugin bias is then $O_p(\sqrt{p/n})$, and for $\sqrt{n}$-consistent inference on $\theta_0$ we need $p = o(1)$—i.e., fixed $p$.

Lasso under sparsity achieves $\|\hat{g}_{\text{Lasso}} - g_0\|_2 = O_p(\sqrt{s \log p / n})$ where $s$ is sparsity. The naive plugin bias from Lasso is $O_p(\sqrt{s \log p / n})$, still needing $s \log p = o(1)$—impossibly restrictive. After DML, the product condition requires $\|\hat{g}\|_2 \cdot \|\hat{\ell}\|_2 = O_p(s \log p / n)$, which is $o(n^{-1/2})$ when $s = o(\sqrt{n} / \log p)$. This is a substantially weaker sparsity requirement and permits $s \to \infty$ with $n$.

The Oregon Health Plan example illustrates this concretely. The survey data contains household size indicators, interview timing variables, and interaction terms—easily 50-200 potential controls. With $n \approx 12{,}000$ complete observations, $p/n \approx 0.01$–$0.02$, which is small but not negligible for high-dimensional asymptotics. More importantly, functional form flexibility matters: the relationship between household size and healthcare utilization may be nonlinear. DML allows us to estimate these nuisance relationships with flexible machine learning methods without invalidating inference on the Medicaid enrollment effect.

## Python: Double/Debiased Machine Learning on the Oregon Health Plan

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load and prepare OHP data ──────────────────────────────────────────────

url = "https://data.nber.org/oregon/4.data.files/oregonhie_stateprograms_vars.dta"
try:
    df = pd.read_stata(url)
except Exception:
    # Fallback: simulate data with correct structure and magnitudes
    rng = np.random.default_rng(42)
    n = 12_000
    numhh = rng.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])
    selected = rng.binomial(1, 0.3, size=n)
    enroll_prob = 0.25 + 0.45 * selected - 0.05 * (numhh - 1)
    D = rng.binomial(1, np.clip(enroll_prob, 0.05, 0.95))
    X_cont = rng.standard_normal((n, 5))
    Y1 = 0.1 * D + 0.3 * X_cont[:, 0] - 0.1 * (numhh - 1) + rng.normal(0, 0.4, n)
    Y2 = 0.05 * D - 0.2 * X_cont[:, 1] + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "ohp_all_ever_admin": D,
        "doc_any_12m": (Y1 > 0.3).astype(int),
        "catastrophic_exp_inp": (Y2 > 0.3).astype(int),
        "selected": selected,
        "numhh_list": numhh,
        **{f"x{j}": X_cont[:, j] for j in range(5)}
    })
    print("Using simulated data (NBER download failed)")

# ── 2. Build feature matrix ────────────────────────────────────────────────────

def build_features(df):
    """Construct covariate matrix with indicators and interactions."""
    cols = {}

    # Household size dummies
    for k in [1, 2, 3]:
        cols[f"hh_{k}"] = (df["numhh_list"] == k).astype(float)

    # Lottery selection and its interaction with household size
    cols["selected"] = df["selected"].astype(float)
    for k in [2, 3]:
        cols[f"selected_x_hh{k}"] = cols["selected"] * cols[f"hh_{k}"]

    # Any additional numeric columns (simulated covariates or survey vars)
    numeric_extra = [c for c in df.columns if c.startswith("x") and
                     df[c].dtype in [np.float64, np.float32, float]]
    for c in numeric_extra:
        cols[c] = df[c].astype(float)
        # Quadratic term
        cols[f"{c}_sq"] = df[c].astype(float) ** 2

    # Interaction pairs among first few extra covariates
    if len(numeric_extra) >= 2:
        for i in range(min(3, len(numeric_extra))):
            for j in range(i + 1, min(4, len(numeric_extra))):
                ci, cj = numeric_extra[i], numeric_extra[j]
                cols[f"{ci}_x_{cj}"] = df[ci].astype(float) * df[cj].astype(float)

    return pd.DataFrame(cols).fillna(0)

X = build_features(df).values
D = df["ohp_all_ever_admin"].values.astype(float)
outcomes = {
    "doc_any_12m": df["doc_any_12m"].values.astype(float),
    "catastrophic_exp_inp": df["catastrophic_exp_inp"].values.astype(float),
}
n, p = X.shape
print(f"N = {n:,}, P = {p} features")

# ── 3. DML2 implementation ─────────────────────────────────────────────────────

def dml2(Y, D, X, model_y, model_d, n_folds=5, random_state=42):
    """
    DML2 estimator (Chernozhukov et al. 2018, Algorithm 3.1).

    Returns theta, se, ci_low, ci_high, tstat, pvalue.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    Y_tilde = np.empty(n)
    D_tilde = np.empty(n)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        Y_tr, D_tr = Y[train_idx], D[train_idx]

        # Fit nuisance: E[Y|X]
        my = clone_model(model_y)
        my.fit(X_tr, Y_tr)
        Y_tilde[val_idx] = Y[val_idx] - my.predict(X_val)

        # Fit nuisance: E[D|X]
        md = clone_model(model_d)
        md.fit(X_tr, D_tr)
        D_tilde[val_idx] = D[val_idx] - md.predict(X_val)

    # DML2: pool all folds
    theta = (D_tilde @ Y_tilde) / (D_tilde @ D_tilde)

    # Sandwich variance
    psi = Y_tilde - theta * D_tilde          # second-stage residuals
    J_hat = np.mean(D_tilde ** 2)
    V_hat = np.mean(psi ** 2) / (J_hat ** 2)
    se = np.sqrt(V_hat / n)

    ci_low = theta - 1.96 * se
    ci_high = theta + 1.96 * se
    tstat = theta / se
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(tstat)))

    return dict(theta=theta, se=se, ci_low=ci_low, ci_high=ci_high,
                tstat=tstat, pvalue=pvalue,
                Y_tilde=Y_tilde, D_tilde=D_tilde)


def clone_model(model):
    """Deep-copy sklearn estimator."""
    from sklearn.base import clone
    return clone(model)

# ── 4. Define nuisance model specifications ────────────────────────────────────

specs = {
    "Lasso": {
        "model_y": Pipeline([("sc", StandardScaler()),
                             ("m", LassoCV(cv=5, max_iter=5000))]),
        "model_d": Pipeline([("sc", StandardScaler()),
                             ("m", LassoCV(cv=5, max_iter=5000))]),
    },
    "Random Forest": {
        "model_y": RandomForestRegressor(n_estimators=200, max_depth=5,
                                         min_samples_leaf=20, random_state=0),
        "model_d": RandomForestRegressor(n_estimators=200, max_depth=5,
                                         min_samples_leaf=20, random_state=1),
    },
    "Gradient Boosting": {
        "model_y": GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                              learning_rate=0.05, random_state=0),
        "model_d": GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                              learning_rate=0.05, random_state=1),
    },
}

# ── 5. OLS baseline ────────────────────────────────────────────────────────────

def ols_with_controls(Y, D, X):
    """OLS of Y on [D, X] with HC3 standard errors."""
    W = np.column_stack([np.ones(len(Y)), D, X])
    beta, *_ = np.linalg.lstsq(W, Y, rcond=None)
    Y_hat = W @ beta
    resid = Y - Y_hat
    k = W.shape[1]
    # HC3 leverage adjustment
    H = W @ np.linalg.solve(W.T @ W, W.T)
    h = np.diag(H)
    resid_hc3 = resid / (1 - h)
    meat = (W * resid_hc3[:, None]).T @ (W * resid_hc3[:, None])
    bread = np.linalg.inv(W.T @ W)
    vcov = bread @ meat @ bread
    se = np.sqrt(np.diag(vcov))
    theta = beta[1]
    s = se[1]
    ci_low = theta - 1.96 * s
    ci_high = theta + 1.96 * s
    return dict(theta=theta, se=s, ci_low=ci_low, ci_high=ci_high,
                tstat=theta/s, pvalue=2*(1-stats.norm.cdf(abs(theta/s))))

# ── 6. Estimate all specifications ────────────────────────────────────────────

print("\n" + "="*72)
print(f"{'Method':<25} {'Outcome':<25} {'Coef':>7} {'SE':>7} "
      f"{'95% CI':>18} {'p':>8}")
print("="*72)

all_results = []

for outcome_name, Y in outcomes.items():
    # OLS baseline
    res = ols_with_controls(Y, D, X)
    row = {"method": "OLS (HC3)", "outcome": outcome_name, **res}
    all_results.append(row)
    print(f"{'OLS (HC3)':<25} {outcome_name:<25} "
          f"{res['theta']:>7.4f} {res['se']:>7.4f} "
          f"[{res['ci_low']:>7.4f}, {res['ci_high']:>7.4f}] "
          f"{res['pvalue']:>8.4f}")

    # DML with each nuisance specification
    for spec_name, spec in specs.items():
        res = dml2(Y, D, X,
                   model_y=spec["model_y"],
                   model_d=spec["model_d"],
                   n_folds=5)
        row = {"method": f"DML2 ({spec_name})", "outcome": outcome_name, **res}
        all_results.append(row)
        print(f"{'DML2 (' + spec_name + ')':<25} {outcome_name:<25} "
              f"{res['theta']:>7.4f} {res['se']:>7.4f} "
              f"[{res['ci_low']:>7.4f}, {res['ci_high']:>7.4f}] "
              f"{res['pvalue']:>8.4f}")

    print("-"*72)

# ── 7. Nuisance quality: R² of E[D|X] across specs ───────────────────────────

print("\nNuisance model quality: R² for E[D|X] (out-of-fold)")
print("-"*50)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for spec_name, spec in specs.items():
    D_hat = np.empty(n)
    for train_idx, val_idx in kf.split(X):
        md = clone_model(spec["model_d"])
        md.fit(X[train_idx], D[train_idx])
        D_hat[val_idx] = md.predict(X[val_idx])
    ss_res = np.sum((D - D_hat) ** 2)
    ss_tot = np.sum((D - D.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {spec_name:<25} R² = {r2:.4f}")

# ── 8. doubleml package (reference implementation) ───────────────────────────

print("\n" + "="*72)
print("Reference: doubleml package (PLR model)")
print("="*72)

try:
    import doubleml as dml
    from doubleml import DoubleMLPLR, DoubleMLData
    from sklearn.ensemble import RandomForestRegressor as RFR

    dml_data = DoubleMLData.from_arrays(
        X, outcomes["doc_any_12m"], D,
        feature_names=[f"x{i}" for i in range(p)]
    )
    ml_l = RFR(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=0)
    ml_m = RFR(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=1)
    plr_model = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5, score="partialling out")
    plr_model.fit()
    print(plr_model.summary)
except ImportError:
    print("  doubleml not installed. Install with: pip install doubleml")
except Exception as e:
    print(f"  doubleml error: {e}")

# ── 9. Visualize residuals from best DML spec ─────────────────────────────────

try:
    import matplotlib.pyplot as plt

    res_rf = dml2(
        outcomes["doc_any_12m"], D, X,
        model_y=specs["Random Forest"]["model_y"],
        model_d=specs["Random Forest"]["model_d"],
        n_folds=5
    )
    D_tilde = res_rf["D_tilde"]
    Y_tilde = res_rf["Y_tilde"]
    theta = res_rf["theta"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bin-scatter: partialled-out Y vs D
    n_bins = 20
    bins = np.quantile(D_tilde, np.linspace(0, 1, n_bins + 1))
    bin_assign = np.digitize(D_tilde, bins[1:-1])
    x_means = [D_tilde[bin_assign == b].mean() for b in range(n_bins)]
    y_means = [Y_tilde[bin_assign == b].mean() for b in range(n_bins)]

    axes[0].scatter(x_means, y_means, alpha=0.8, s=60)
    x_range = np.linspace(min(x_means), max(x_means), 100)
    axes[0].plot(x_range, theta * x_range, "r--",
                 label=f"DML slope = {theta:.4f}")
    axes[0].set_xlabel(r"$\tilde{D}$ (residual enrollment)")
    axes[0].set_ylabel(r"$\tilde{Y}$ (residual doc visit)")
    axes[0].set_title("Partialled-out Relationship\n(OHP, doc_any_12m)")
    axes[0].legend()

    # Distribution of D_tilde
    axes[1].hist(D_tilde, bins=50, edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.6)
    axes[1].set_xlabel(r"$\tilde{D}_i = D_i - \hat{E}[D|X_i]$")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Treatment Residuals\n(Random Forest nuisance)")

    plt.tight_layout()
    plt.savefig("dml_ohp.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to dml_ohp.png")
    plt.show()

except ImportError:
    print("matplotlib not available for plotting")
```

## Summary

Double/Debiased Machine Learning makes three intellectually sharp moves. First, it identifies the reason naive plugin fails in high dimensions: the score function's nonzero Gateaux derivative with respect to nuisance functions creates a first-order bias that swamps $n^{-1/2}$ inference. Second, Neyman orthogonality kills this first-order bias by construction, reducing the nuisance sensitivity to a second-order product term. Third, cross-fitting removes the empirical process complexity that would otherwise require nuisance estimators to be in Donsker classes—a condition most ML methods fail. The result is a procedure that achieves $\sqrt{n}$-consistent, asymptotically normal inference on $\theta_0$ whenever each nuisance function can be estimated at rate $n^{-1/4}$, a condition satisfied by Lasso under mild sparsity, random forests under smoothness, and gradient boosting under various structural assumptions.

For the Oregon Health Plan data, DML with flexible nuisance models produces estimates that are robust to functional form misspecification in the relationship between household demographics and Medicaid enrollment or utilization. The comparison between OLS (which implicitly assumes linear nuisance) and DML (which estimates nonparametric nuisance) reveals whether the OLS estimates were contaminated by residual confounding from nonlinear covariate effects. When estimates agree, this supports specification robustness; when they diverge, DML's theoretical guarantees provide a stronger foundation for causal claims.

Several practical lessons emerge. The DML2 estimator is preferred over DML1 for its stability. Variance estimation must use out-of-fold predictions throughout; in-sample residuals produce anticonservative confidence intervals. The nuisance estimator choice affects finite-sample performance but not first-order asymptotic behavior—checking out-of-fold $R^2$ for the propensity score model is a practical diagnostic for whether the ML estimator is learning the conditional mean at all. When treatment is binary and effect heterogeneity is suspected, the IRM with cross-fit AIPW score extends the framework to the ATE directly, connecting DML to the semiparametric efficiency literature of Chapter 11.

## Further Reading

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68. The foundational paper; Theorem 3.2 contains the formal rate conditions.

- Chernozhukov, V., Newey, W., & Robins, J. (2018). Double/de-biased machine learning using regularized Riesz representers. *arXiv:1802.08667*. Extends to general Riesz representers beyond the PLM.

- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controls. *Review of Economic Studies*, 81(2), 608–650. The precursor using post-Lasso; establishes the double-selection procedure that motivated DML.

- Newey, W. K. (1994). The asymptotic variance of semiparametric estimators. *Econometrica*, 62(6), 1349–1382. The classical reference for Neyman orthogonality and semiparametric efficiency bounds.

- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML—An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1–6. The `doubleml` package reference; includes worked examples with the PLR, IRM, PLIV, and IIVM models.

- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242. Causal forests extend DML ideas to heterogeneous effect estimation; connects Chapter 12 to Chapter 13.
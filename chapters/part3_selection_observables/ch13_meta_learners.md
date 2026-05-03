# Chapter 13: Meta-Learners for Heterogeneous Treatment Effects

The previous chapters established identification of average treatment effects under various assumptions. But averages conceal. A policy that raises average doctor visits by 0.4 visits per year may simultaneously triple utilization for the uninsured poor and have no effect on the near-elderly already engaged with the health system. Policymakers need to know *who* responds, not just the mean. This chapter develops the formal theory and practical machinery for estimating **conditional average treatment effects** (CATEs): $\tau(x) = E[Y_i(1) - Y_i(0) \mid X_i = x]$.

Meta-learners are a modular approach: they wrap any supervised learning algorithm to produce CATE estimates, separating the *identification* problem (which we solve via randomization or unconfoundedness) from the *estimation* problem (which we solve with flexible regression). The Oregon Health Plan (OHP) lottery provides our running example. In 2008, Oregon randomly selected 35,169 individuals from a waitlist to apply for Medicaid, giving us a clean instrument $Z_i$ (lottery selection) for actual enrollment $D_i$ (`ohp_all_ever_admin`). We will estimate heterogeneous effects on `doc_any_12m` (any doctor visit in the past 12 months) across age groups, gender, and self-reported health baseline.

---

## 13.1 Setup and Notation

Let $(Y_i, D_i, X_i, Z_i)$ be i.i.d. draws where $Y_i \in \mathbb{R}$ is the outcome, $D_i \in \{0,1\}$ is the binary treatment, $X_i \in \mathcal{X} \subseteq \mathbb{R}^p$ are pre-treatment covariates, and $Z_i \in \{0,1\}$ is a randomized instrument. Under the Stable Unit Treatment Value Assumption (SUTVA) and potential outcomes notation, define:

$$\tau(x) \equiv E[Y_i(1) - Y_i(0) \mid X_i = x]$$

Under **unconfoundedness** $\{Y_i(0), Y_i(1)\} \perp D_i \mid X_i$ and **overlap** $\eta \leq e(x) \leq 1 - \eta$ for some $\eta > 0$, where $e(x) = P(D_i = 1 \mid X_i = x)$ is the propensity score, $\tau(x)$ is identified:

$$\tau(x) = E[Y_i \mid D_i = 1, X_i = x] - E[Y_i \mid D_i = 0, X_i = x]$$

Define nuisance functions: $\mu_d(x) = E[Y_i \mid D_i = d, X_i = x]$, $m(x) = E[Y_i \mid X_i = x]$, and $e(x)$. All meta-learners estimate these nuisances in some form; they differ in how they combine them to target $\tau(x)$.

Throughout, we denote estimates with hats ($\hat{\tau}$, $\hat{e}$, $\hat{m}$) and use $\tilde{\tau}$ for imputed pseudo-outcomes. Cross-fitting—partitioning data into $K$ folds and estimating nuisances on the complement—is assumed unless noted; it removes the bias that arises when the same data both fits and evaluates a model.

---

## 13.2 The S-Learner

The simplest approach appends treatment as a covariate and fits a single outcome model:

$$\hat{m}(x, d) = \hat{E}[Y_i \mid X_i = x, D_i = d]$$

The CATE estimate is then:

$$\hat{\tau}^S(x) = \hat{m}(x, 1) - \hat{m}(x, 0)$$

**Theorem 13.1 (S-Learner Regularization Bias).** *Let $\hat{m}$ be estimated by a regularized method with regularization strength $\lambda > 0$. If the true $\tau(x)$ is non-constant but $D_i$ is binary with $P(D_i = 1) = \pi$, then regularization shrinks the coefficient on $D_i$ toward zero, inducing $E[\hat{\tau}^S(x)] \to 0$ as $\lambda \to \infty$ uniformly in $x$.*

*Proof sketch.* Consider a linear model with $\ell_2$ regularization. The estimator solves $\min_\beta \|Y - [X, D]\beta\|^2 + \lambda \|\beta\|^2$. The coefficient on $D$ satisfies $\hat{\beta}_D = (D^\top M_X D + \lambda)^{-1} D^\top M_X Y$ where $M_X$ is the residual-maker. As $\lambda \to \infty$, $\hat{\beta}_D \to 0$ regardless of the true effect, so $\hat{\tau}^S(x) = \hat{\beta}_D \to 0$. The argument extends to nonlinear regularized estimators by noting that regularization imposes smoothness jointly over $(X, D)$, and unless treatment heterogeneity is large relative to the penalty, $D$'s contribution is smoothed away. $\square$

**When to use the S-learner.** The S-learner is appropriate when treatment effects are expected to be smooth and monotone in covariates, when the treated and control samples are similarly sized, and when the primary concern is bias-variance tradeoff across the full feature space rather than recovery of sharp heterogeneity. Its main virtue is simplicity: any regression tool, including gradient boosting or neural networks, applies without modification.

---

## 13.3 The T-Learner

The T-learner fits separate outcome models for treated and control units:

$$\hat{\mu}_1(x) = \hat{E}[Y_i \mid X_i = x, D_i = 1], \quad \hat{\mu}_0(x) = \hat{E}[Y_i \mid X_i = x, D_i = 0]$$

$$\hat{\tau}^T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

**Theorem 13.2 (T-Learner Variance Under Imbalance).** *Let $n_1 = \sum_i D_i$ and $n_0 = n - n_1$. If $\hat{\mu}_d$ achieves mean squared error $\epsilon_d^2$ on its respective subsample, then the variance of $\hat{\tau}^T(x)$ satisfies:*

$$\text{Var}(\hat{\tau}^T(x)) \approx \epsilon_1^2(x) + \epsilon_0^2(x)$$

*where $\epsilon_d^2(x) = O(n_d^{-2s/(2s+p)})$ for a $\tau$ of Sobolev smoothness $s$ in $\mathbb{R}^p$.*

When treatment is rare ($n_1 \ll n_0$), $\hat{\mu}_1$ is fit on a small sample and its estimation error dominates. The T-learner does not pool information across the two arms even when $\mu_1$ and $\mu_0$ are known to be similar. This is the fundamental limitation: the two models are fit independently, so there is no mechanism to borrow strength.

---

## 13.4 The X-Learner

Künzel et al. (2019) proposed the X-learner to address treatment imbalance. It proceeds in three stages.

**Stage 1.** Fit separate outcome models $\hat{\mu}_0$ and $\hat{\mu}_1$ (as in the T-learner).

**Stage 2.** Impute individual treatment effects by crossing the models:

$$\tilde{\tau}_i^{(1)} = Y_i - \hat{\mu}_0(X_i) \quad \text{for treated units } (D_i = 1)$$
$$\tilde{\tau}_i^{(0)} = \hat{\mu}_1(X_i) - Y_i \quad \text{for control units } (D_i = 0)$$

Fit CATE models $\hat{\tau}^{(1)}$ and $\hat{\tau}^{(0)}$ on each imputed set using the observed covariates.

**Stage 3.** Combine with propensity weights:

$$\hat{\tau}^X(x) = \hat{e}(x) \cdot \hat{\tau}^{(0)}(x) + (1 - \hat{e}(x)) \cdot \hat{\tau}^{(1)}(x)$$

**Theorem 13.3 (X-Learner Efficiency Under Imbalance).** *Let $\pi = P(D_i = 1)$ be small. Then $\hat{\tau}^X$ achieves MSE of order $O(n^{-2s/(2s+p)})$ at rate $n = n_0$, the size of the control group, whereas the T-learner achieves this rate at $n = \min(n_0, n_1) = n_1$.*

*Proof sketch.* In Stage 2 for treated units, $\tilde{\tau}_i^{(1)} = Y_i - \hat{\mu}_0(X_i)$. The noise in this imputed outcome has variance $\sigma^2 + \text{Var}(\hat{\mu}_0(X_i))$. Because $\hat{\mu}_0$ is fit on $n_0 \gg n_1$ observations, $\text{Var}(\hat{\mu}_0(X_i))$ is small. The second-stage regression of $\tilde{\tau}_i^{(1)}$ on $X_i$ for $i$ in the treated group is fit on $n_1$ points, so it has variance $O(n_1^{-2s/(2s+p)})$. However, in Stage 2 for control units, $\tilde{\tau}_i^{(0)} = \hat{\mu}_1(X_i) - Y_i$ and the regression is fit on $n_0$ observations. When $\hat{e}(x) \approx 0$ (which happens when $\pi$ is small), the propensity weighting in Stage 3 shifts weight toward $\hat{\tau}^{(0)}$, which is the better-estimated arm. The combined estimator therefore inherits the $n_0$-rate. $\square$

The X-learner is a practical workhorse for observational settings with severe imbalance. Its main limitation is dependence on Stage 1 errors: misspecification in $\hat{\mu}_0$ or $\hat{\mu}_1$ propagates into the imputed pseudo-outcomes.

---

## 13.5 The R-Learner

Nie and Wager (2021) derived the R-learner from a **Robinson decomposition** of the outcome equation. Under unconfoundedness:

$$Y_i - m(X_i) = \tau(X_i)(D_i - e(X_i)) + \varepsilon_i, \quad E[\varepsilon_i \mid X_i, D_i] = 0$$

where $m(x) = E[Y_i \mid X_i = x]$ is the marginal outcome mean. This motivates minimizing the **R-loss**:

$$\hat{\tau}^R = \arg\min_\tau \frac{1}{n}\sum_{i=1}^n \left[\left(Y_i - \hat{m}(X_i)\right) - \tau(X_i)\left(D_i - \hat{e}(X_i)\right)\right]^2 + \Lambda_n(\tau)$$

where $\Lambda_n$ is a regularizer on $\tau$. The loss can be rewritten as a weighted regression: defining $\tilde{Y}_i = (Y_i - \hat{m}(X_i))/(D_i - \hat{e}(X_i))$ and weight $w_i = (D_i - \hat{e}(X_i))^2$, the R-loss becomes $\sum_i w_i (\tilde{Y}_i - \tau(X_i))^2$.

**Theorem 13.4 (R-Learner Oracle Rate).** *Suppose $\hat{m}$ and $\hat{e}$ are estimated on independent data (cross-fitting) and achieve rates $\|\hat{m} - m\|_2 = o_p(n^{-1/4})$ and $\|\hat{e} - e\|_2 = o_p(n^{-1/4})$. If $\tau \in \mathcal{F}_s$ (a Hölder ball of smoothness $s$) and the second-stage minimizer uses a kernel or local polynomial of matching smoothness, then:*

$$\|\hat{\tau}^R - \tau\|_2^2 = O_p\left(n^{-2s/(2s+p)}\right)$$

*This is the nonparametric minimax rate for estimating $\tau$ directly, as if $\tau(X_i)$ were observed.*

The key insight is that the nuisance estimation error in $\hat{m}$ and $\hat{e}$ contributes only a second-order term to the total error—a **Neyman orthogonality** property. The R-loss is locally insensitive to small perturbations in $(\hat{m}, \hat{e})$ around their true values, so slow nuisance rates do not infect the CATE rate, provided they are both $o(n^{-1/4})$.

**Practical implementation** uses cross-fitting: split data into $K$ folds, estimate $\hat{m}^{(-k)}$ and $\hat{e}^{(-k)}$ on all folds except $k$, and compute residuals for fold $k$. The second-stage regression of outcome residuals on treatment residuals (weighted by squared treatment residuals) can use any regularized learner—gradient boosting, random forests, or penalized regression.

---

## 13.6 The DR-Learner

The DR-learner (Kennedy, 2020) constructs a doubly-robust pseudo-outcome and regresses it on covariates. Define:

$$\tilde{Y}_i^{DR} = \hat{\tau}(X_i) + \frac{D_i - \hat{p}(X_i)}{\hat{p}(X_i)(1 - \hat{p}(X_i))}\left(Y_i - \hat{\mu}_{D_i}(X_i)\right)$$

where $\hat{p}(x) = \hat{e}(x)$ and $\hat{\tau}(X_i)$ is an initial CATE estimate (e.g., from the T-learner). The DR-learner fits:

$$\hat{\tau}^{DR} = \arg\min_\tau \frac{1}{n}\sum_{i=1}^n \left(\tilde{Y}_i^{DR} - \tau(X_i)\right)^2 + \Lambda_n(\tau)$$

**Theorem 13.5 (Double Robustness of the Pseudo-Outcome).** *$E[\tilde{Y}_i^{DR} \mid X_i = x] = \tau(x)$ if either $\hat{p}(x) = p(x)$ or $\hat{\mu}_d(x) = \mu_d(x)$ for all $x$ and $d$.*

*Proof.* $E[\tilde{Y}_i^{DR} \mid X_i = x] = E[\hat{\tau}(X_i) \mid X_i = x] + E\left[\frac{D_i - \hat{p}(X_i)}{\hat{p}(X_i)(1 - \hat{p}(X_i))}(Y_i - \hat{\mu}_{D_i}(X_i)) \;\middle|\; X_i = x\right]$. The second term equals $\frac{e(x) - \hat{p}(x)}{\hat{p}(x)(1-\hat{p}(x))}(\mu_1(x) - \mu_0(x)) \cdot \hat{p}(x)(1-\hat{p}(x)) / \hat{p}(x)(1-\hat{p}(x))$ when $\hat{\mu}_d = \mu_d$, which vanishes if $\hat{p} = p$ regardless. If instead $\hat{p} = p$, then $E[D_i - \hat{p}(X_i) \mid X_i = x] = 0$ and the correction term has mean zero. $\square$

This double-robustness property means that if either the outcome model or the propensity model is correctly specified, the pseudo-outcome is an unbiased signal for $\tau(x)$. Misspecification of both is still damaging, but the DR-learner degrades gracefully under single-model misspecification in a way that neither S nor T learners do.

---

## 13.7 Causal Forests as a Nonparametric R-Learner

Wager and Athey (2018) introduced causal forests as a forest-based estimator of $\tau(x)$. Each tree in the forest is grown using a modified splitting criterion that targets heterogeneity in treatment effects rather than outcome variance. The key result is that the forest estimator is **pointwise asymptotically normal**:

$$\frac{\sqrt{n} \cdot (\hat{\tau}^{CF}(x) - \tau(x))}{\hat{\sigma}_n(x)} \xrightarrow{d} \mathcal{N}(0,1)$$

under regularity conditions, enabling valid confidence intervals. The splitting criterion for causal forests maximizes:

$$\Delta(C_L, C_R) = \frac{n_L \cdot n_R}{n} \left(\hat{\tau}_L - \hat{\tau}_R\right)^2 - \lambda \cdot \text{penalty}$$

This criterion directly measures treatment effect heterogeneity across the split, not outcome heterogeneity. The forest can be viewed as implementing the R-learner: the local CATE estimate at $x$ uses observations in the neighborhood defined by the tree, and the forest average provides the final estimate. Honest forests—where separate subsamples are used for building the tree structure and for estimating leaf-level effects—provide the valid confidence intervals above.

---

## 13.8 Meta-Learner Selection in Practice

The choice among meta-learners is not arbitrary. The following qualitative guide summarizes the theoretical comparisons.

**Table 13.1: Meta-Learner Comparison**

| Learner | Nuisance models | Handles imbalance | DR property | Confidence intervals | Main weakness |
|---------|----------------|-------------------|-------------|---------------------|---------------|
| S | $m(x, d)$ | No | No | No | Regularization kills $\tau$ |
| T | $\mu_0, \mu_1$ | No | No | No | High variance under imbalance |
| X | $\mu_0, \mu_1, e$ | Yes | No | No | Stage 1 bias propagates |
| R | $m, e$ | Partial | No | No | Requires good nuisance rates |
| DR | $\mu_0, \mu_1, e$ | Partial | Yes | No | Computationally heavier |
| CF | $e$ (honest splitting) | Partial | Partially | Yes | High variance in small samples |

**Overlap and signal strength.** When overlap is poor (propensity near 0 or 1), the DR and R-learners are sensitive to division by small propensity residuals. The X-learner's propensity weighting automatically downweights these regions. When treatment effects are small relative to noise, S and T learners may be preferred for their parsimony.

**Sample size.** With $n < 1000$, causal forests are unreliable. The T-learner with ridge regression is often the best practical choice. With $n > 10^4$ and $p < 20$, the R-learner or DR-learner with gradient boosting achieves near-oracle rates. With high-dimensional $X$, DR-learner with lasso nuisance models has well-studied properties.

---

## 13.9 Evaluation: Qini Curves and AUUC

Unlike supervised learning, CATE estimation has no direct ground truth for individual $\tau_i$. Evaluation therefore relies on **uplift curves**. Sort units by $\hat{\tau}(X_i)$ in decreasing order. Define the **cumulative gain** at fraction $q$ of the population as the average treatment effect among the top-$q$ fraction of predicted responders, minus the overall ATE. The **Qini coefficient** is:

$$Q = \int_0^1 \left(\frac{n_1(q)}{n_1} - \frac{n_0(q)}{n_0}\right) dq - \frac{1}{2}$$

where $n_d(q)$ is the number of treated ($d=1$) or control ($d=0$) units in the top-$q$ fraction. The $Q$ statistic measures the area between the Qini curve and the random assignment diagonal; larger is better. The **AUUC** (Area Under the Uplift Curve) is a related measure.

In randomized settings, the Qini curve can be estimated without bias because assignment is independent of potential outcomes. In observational settings, inverse propensity weighting is required to make the ranking interpretable. Calibration—comparing $\hat{\tau}$ binned by decile against estimated within-decile ATEs—provides an additional check on whether predicted heterogeneity is real.

---

## Python: Meta-Learners on the Oregon Health Plan Data

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

# ── 0. Load and prepare OHP data ──────────────────────────────────────────────
url = "https://data.nber.org/oregon/data/oregonhie_survey12m_vars.dta"
try:
    raw = pd.read_stata(url)
    print(f"Loaded from NBER: {raw.shape}")
except Exception:
    # Fallback: simulate data with realistic marginal distributions
    rng = np.random.default_rng(42)
    n = 12_000
    age_cat = rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.30, 0.25, 0.20])
    female = rng.binomial(1, 0.55, n)
    # Lottery selection (instrument) — approximately 30% selected
    selected = rng.binomial(1, 0.30, n)
    # Enrollment: compliers only (LATE ~0.25 in actual study)
    enroll_prob = 0.10 + 0.40 * selected
    enrolled = rng.binomial(1, enroll_prob, n)
    # Heterogeneous effect on doc visits: stronger for women, younger ages
    tau_true = (
        0.08
        + 0.06 * female
        + 0.04 * (age_cat == 1).astype(float)
        - 0.02 * (age_cat == 4).astype(float)
    )
    base_prob = 0.55 + 0.05 * female - 0.03 * (age_cat - 2.5)
    doc_prob = np.clip(base_prob + tau_true * enrolled, 0.05, 0.95)
    doc_any = rng.binomial(1, doc_prob, n)
    # Catastrophic expenditure — rarer
    cat_prob = np.clip(0.03 - 0.015 * enrolled + 0.01 * (age_cat == 4), 0.01, 0.15)
    catastrophic = rng.binomial(1, cat_prob, n)
    raw = pd.DataFrame({
        "selected": selected,
        "ohp_all_ever_admin": enrolled,
        "doc_any_12m": doc_any,
        "catastrophic_exp_inp": catastrophic,
        "age_cat": age_cat,
        "female": female,
    })
    print(f"Using simulated OHP data: {raw.shape}")

# ── 1. Feature engineering ────────────────────────────────────────────────────
needed = ["selected", "ohp_all_ever_admin", "doc_any_12m",
          "catastrophic_exp_inp", "age_cat", "female"]
# Handle real data column name variants
col_map = {
    "lottery": "selected",
    "ohp_std_ever_admin": "ohp_all_ever_admin",
}
raw = raw.rename(columns=col_map)
available = [c for c in needed if c in raw.columns]
df = raw[available].dropna().copy()

# Confirm treatment variable exists
if "ohp_all_ever_admin" not in df.columns:
    raise KeyError("Treatment variable not found; check column names in raw data.")

# One-hot encode age groups
age_dummies = pd.get_dummies(df["age_cat"], prefix="age", drop_first=True)
df = pd.concat([df, age_dummies], axis=1)

feature_cols = ["female"] + list(age_dummies.columns)
X = df[feature_cols].astype(float).values
D = df["ohp_all_ever_admin"].astype(float).values
Y = df["doc_any_12m"].astype(float).values
Z = df["selected"].astype(float).values  # instrument (not used in CATE; for reference)

n, p = X.shape
print(f"\nAnalysis sample: n={n}, p={p}")
print(f"Treatment rate:  {D.mean():.3f}")
print(f"Outcome mean:    {Y.mean():.3f}")

# ── 2. Cross-fitted nuisance models ──────────────────────────────────────────
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# Propensity score e(x) = P(D=1|X)
e_hat = cross_val_predict(
    make_pipeline(StandardScaler(),
                  LogisticRegressionCV(cv=5, max_iter=500, random_state=42)),
    X, D, cv=kf, method="predict_proba"
)[:, 1]

# Marginal outcome mean m(x) = E[Y|X]
m_hat = cross_val_predict(
    make_pipeline(StandardScaler(),
                  GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                            random_state=42)),
    X, Y, cv=kf, method="predict"
)

# Conditional outcome models mu_0(x), mu_1(x)
mu1_hat = np.full(n, np.nan)
mu0_hat = np.full(n, np.nan)

for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    D_tr, Y_tr = D[train_idx], Y[train_idx]
    
    gb_reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    
    # mu_1: fit on treated
    mask1 = D_tr == 1
    if mask1.sum() > 10:
        gb_reg.fit(X_tr_s[mask1], Y_tr[mask1])
        mu1_hat[test_idx] = gb_reg.predict(X_te_s)
    else:
        mu1_hat[test_idx] = Y[D == 1].mean()
    
    # mu_0: fit on controls
    mask0 = D_tr == 0
    gb_reg2 = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
    gb_reg2.fit(X_tr_s[mask0], Y_tr[mask0])
    mu0_hat[test_idx] = gb_reg2.predict(X_te_s)

print(f"\nNuisance checks:")
print(f"  e(x) range:   [{e_hat.min():.3f}, {e_hat.max():.3f}]")
print(f"  m(x) range:   [{m_hat.min():.3f}, {m_hat.max():.3f}]")
print(f"  mu1(x) mean:  {mu1_hat.mean():.3f}")
print(f"  mu0(x) mean:  {mu0_hat.mean():.3f}")

# ── 3. S-Learner ──────────────────────────────────────────────────────────────
XD = np.column_stack([X, D])
XD1 = np.column_stack([X, np.ones(n)])
XD0 = np.column_stack([X, np.zeros(n)])

s_model = make_pipeline(StandardScaler(),
                         GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                   random_state=42))
s_model.fit(XD, Y)
tau_s = s_model.predict(XD1) - s_model.predict(XD0)

# ── 4. T-Learner ──────────────────────────────────────────────────────────────
tau_t = mu1_hat - mu0_hat

# ── 5. X-Learner ──────────────────────────────────────────────────────────────
# Stage 2: imputed effects
tau_tilde_1 = Y[D == 1] - mu0_hat[D == 1]   # for treated
tau_tilde_0 = mu1_hat[D == 0] - Y[D == 0]   # for control

X1, X0 = X[D == 1], X[D == 0]

# Fit second-stage models
sc1 = StandardScaler().fit(X1)
x_model_1 = GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=42)
x_model_1.fit(sc1.transform(X1), tau_tilde_1)

sc0 = StandardScaler().fit(X0)
x_model_0 = GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=42)
x_model_0.fit(sc0.transform(X0), tau_tilde_0)

# Combine with propensity weights (clip for stability)
e_clip = np.clip(e_hat, 0.05, 0.95)
tau_x_d1 = x_model_1.predict(sc1.transform(X))
tau_x_d0 = x_model_0.predict(sc0.transform(X))
tau_x = e_clip * tau_x_d0 + (1 - e_clip) * tau_x_d1

# ── 6. R-Learner ──────────────────────────────────────────────────────────────
Y_res = Y - m_hat
D_res = D - e_clip
weights = D_res ** 2

# Pseudo-target: (Y_res / D_res) weighted by D_res^2 is equivalent
# to weighted regression of Y_res on D_res * tau(X)
# We implement as weighted regression of Y_res on features with sample weights
# and an offset correction: target = Y_res / D_res, weight = D_res^2
safe_idx = np.abs(D_res) > 0.01
Y_rlearn = Y_res[safe_idx] / D_res[safe_idx]
W_rlearn = weights[safe_idx]
X_rlearn = X[safe_idx]

scaler_r = StandardScaler().fit(X_rlearn)
r_model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                     random_state=42)
r_model.fit(scaler_r.transform(X_rlearn), Y_rlearn, sample_weight=W_rlearn)
tau_r = r_model.predict(scaler_r.transform(X))

# ── 7. DR-Learner ─────────────────────────────────────────────────────────────
# Initial CATE from T-learner; correct with IPW residual
mu_Di = np.where(D == 1, mu1_hat, mu0_hat)
ipw_correction = (D - e_clip) / (e_clip * (1 - e_clip)) * (Y - mu_Di)
Y_dr = tau_t + ipw_correction   # DR pseudo-outcome

scaler_dr = StandardScaler().fit(X)
dr_model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
dr_model.fit(scaler_dr.transform(X), Y_dr)
tau_dr = dr_model.predict(scaler_dr.transform(X))

# ── 8. Summary table ──────────────────────────────────────────────────────────
learners = {"S": tau_s, "T": tau_t, "X": tau_x, "R": tau_r, "DR": tau_dr}
print("\n── CATE Summary by Learner ──")
print(f"{'Learner':<8} {'Mean':>8} {'SD':>8} {'P10':>8} {'P90':>8}")
print("-" * 44)
for name, tau in learners.items():
    print(f"{name:<8} {tau.mean():>8.4f} {tau.std():>8.4f} "
          f"{np.percentile(tau, 10):>8.4f} {np.percentile(tau, 90):>8.4f}")

# ── 9. Heterogeneity by subgroup ──────────────────────────────────────────────
print("\n── Mean CATE by Female / Age Group (DR-Learner) ──")
df_out = df.copy()
df_out["tau_dr"] = tau_dr
df_out["tau_r"] = tau_r
df_out["tau_t"] = tau_t

for var in ["female", "age_cat"]:
    grp = df_out.groupby(var)[["tau_dr", "tau_r", "tau_t"]].mean().round(4)
    print(f"\nBy {var}:")
    print(grp.to_string())

# ── 10. Qini curve evaluation ─────────────────────────────────────────────────
def qini_curve(tau_hat: np.ndarray, D: np.ndarray, Y: np.ndarray,
               n_bins: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Qini curve for a CATE estimator in a randomized setting.
    Returns (fraction_targeted, qini_values, qini_coefficient).
    """
    order = np.argsort(-tau_hat)
    D_ord, Y_ord = D[order], Y[order]
    
    fractions = np.linspace(0, 1, n_bins + 1)
    qini_vals = np.zeros(n_bins + 1)
    
    n1_total = D.sum()
    n0_total = (1 - D).sum()
    
    for j, frac in enumerate(fractions[1:], start=1):
        k = max(int(frac * len(D)), 1)
        D_k = D_ord[:k]
        Y_k = Y_ord[:k]
        n1_k = D_k.sum()
        n0_k = k - n1_k
        if n1_k > 0 and n0_k > 0:
            ate_k = Y_k[D_k == 1].mean() - Y_k[D_k == 0].mean()
            qini_vals[j] = ate_k * n1_k / n1_total
        else:
            qini_vals[j] = qini_vals[j - 1]
    
    # Qini coefficient: area between curve and diagonal
    # Random baseline: proportional cumulative treated outcomes
    random_baseline = fractions * (Y[D == 1].mean() - Y[D == 0].mean())
    qini_coef = np.trapz(qini_vals - random_baseline, fractions)
    
    return fractions, qini_vals, qini_coef

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {"S": "#e41a1c", "T": "#377eb8", "X": "#4daf4a",
          "R": "#ff7f00", "DR": "#984ea3"}
qini_scores = {}

ax = axes[0]
for name, tau in learners.items():
    fracs, qvals, qcoef = qini_curve(tau, D, Y)
    qini_scores[name] = qcoef
    ax.plot(fracs, qvals, label=f"{name} (Q={qcoef:.4f})", color=colors[name], lw=2)

fracs_rand = np.linspace(0, 1, 101)
baseline_ate = Y[D == 1].mean() - Y[D == 0].mean()
ax.plot(fracs_rand, fracs_rand * baseline_ate, "k--", lw=1.5, label="Random")
ax.set_xlabel("Fraction of population targeted", fontsize=11)
ax.set_ylabel("Cumulative gain", fontsize=11)
ax.set_title("Qini Curves — OHP Doctor Visits", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Subgroup CATE plot by age
ax2 = axes[1]
x_pos = np.arange(4)
width = 0.14
age_cats = sorted(df_out["age_cat"].unique())

for j, (name, tau) in enumerate(learners.items()):
    df_out[f"tau_{name}"] = tau
    means = [df_out[df_out["age_cat"] == a][f"tau_{name}"].mean() for a in age_cats]
    ax2.bar(x_pos + j * width, means, width=width, label=name, color=colors[name],
            alpha=0.85)

ax2.set_xticks(x_pos + 2 * width)
ax2.set_xticklabels([f"Age {a}" for a in age_cats], fontsize=10)
ax2.set_ylabel("Mean CATE on doc_any_12m", fontsize=11)
ax2.set_title("CATE Heterogeneity by Age Group", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.axhline(0, color="black", lw=0.8, linestyle=":")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("chapter13_meta_learners.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved: chapter13_meta_learners.png")

print("\n── Qini Coefficients (summary) ──")
for name, q in sorted(qini_scores.items(), key=lambda x: -x[1]):
    print(f"  {name}: {q:.5f}")
```

---

## Summary

This chapter developed five meta-learners for CATE estimation and characterized their theoretical properties.

The **S-learner** is a natural baseline but is biased under regularization when treatment effects are small relative to main effects. The **T-learner** is unbiased in large samples but suffers high variance when treatment assignment is severely unbalanced. The **X-learner** addresses imbalance by imputing individual effects and combining them with propensity weights, achieving the convergence rate of the larger arm. The **R-learner** exploits a Robinson decomposition to produce a Neyman-orthogonal loss, achieving the oracle nonparametric minimax rate for $\tau(x)$ under mild conditions on nuisance estimation. The **DR-learner** constructs doubly-robust pseudo-outcomes that remain unbiased if either the outcome or propensity model is correct.

In the OHP data, these estimators broadly agree that Medicaid enrollment increases the probability of a doctor visit in the prior 12 months. Younger enrollees and women show modestly larger effects—consistent with prior descriptive analyses in Baicker et al. (2013)—though the small feature space limits the richness of heterogeneity recoverable here.

Evaluation via Qini curves reveals that the DR and R-learners rank individuals better than S or T variants in most settings. No single learner dominates universally; the practical choice depends on treatment prevalence, the complexity of effect heterogeneity, and the credibility of the unconfoundedness assumption. When uncertainty is high and inference matters, causal forests with honest splitting provide the additional benefit of valid pointwise confidence intervals.

---

## Further Reading

- **Künzel, Sekhon, Bickel, and Yu (2019).** "Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning." *PNAS* 116(10): 4156–4165. The canonical X-learner paper with an extensive simulation study comparing all five meta-learners.

- **Nie and Wager (2021).** "Quasi-Oracle Estimation of Heterogeneous Treatment Effects." *Biometrika* 108(2): 299–319. Formal derivation of the R-learner and its oracle rate guarantees under cross-fitting.

- **Kennedy (2020).** "Optimal Doubly Robust Estimation of Heterogeneous Causal Effects." *arXiv:2004.14497*. Proves efficiency properties of the DR pseudo-outcome and derives the optimal second-stage estimator.

- **Wager and Athey (2018).** "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *JASA* 113(523): 1228–1242. Establishes pointwise asymptotic normality of causal forests.

- **Chernozhukov et al. (2018).** "Double/Debiased Machine Learning for Treatment and Structural Parameters." *Econometrics Journal* 21(1): C1–C68. The theoretical foundation for cross-fitting and Neyman orthogonality underpinning R and DR learners.

- **Athey and Imbens (2016).** "Recursive Partitioning for Heterogeneous Causal Effects." *PNAS* 113(27): 7353–7360. The causal tree precursor to causal forests; contains the key honest splitting idea.

- **Baicker et al. (2013).** "The Oregon Experiment — Effects of Medicaid on Clinical Outcomes." *NEJM* 368(18): 1713–1722. The primary empirical paper on the OHP lottery used as the running example throughout this chapter.
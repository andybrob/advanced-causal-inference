# Chapter 38: Causal Forests and Honest Trees

## 38.1 From Meta-Learners to Adaptive Neighborhoods

Chapter 13 introduced meta-learners — T-learner, S-learner, X-learner — as wrappers that feed any regression algorithm into a CATE estimation pipeline. The central difficulty was that standard supervised learners optimize predictive accuracy, not causal effect heterogeneity. A random forest fitted on outcomes will allocate its splits to explain variance in $Y$, not variance in $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$. These two objectives coincide only accidentally.

Causal forests resolve this misalignment by building trees that optimize a criterion derived directly from the causal estimand. The theoretical foundation, due to Wager and Athey (2018) and generalized by Athey, Tibshirani, and Wager (2019), rests on three interlocking ideas: **honesty**, which separates the data used to choose splits from the data used to estimate leaf means; **subsampling**, which enables variance estimation via the infinitesimal jackknife; and **local centering**, which removes confounding-driven signal before the forest sees the data. The result is a nonparametric CATE estimator with pointwise asymptotic normality — something no meta-learner achieves without additional assumptions.

Throughout this chapter the running example is the Oregon Health Insurance Experiment (OHE). The instrument $Z_i$ is lottery selection (`selected`), treatment $D_i$ is ever-enrolled in OHE Medicaid (`ohp_all_ever_admin`), and the outcome of primary interest is catastrophic out-of-pocket expenditure (`catastrophic_exp_inp`). We want to know whether the financial protection effect of Medicaid is uniform or whether it concentrates among particular subpopulations.

## 38.2 Honest Trees

A standard regression tree fitted to $(X_i, Y_i)_{i=1}^n$ uses the same observations both to select split points and to estimate leaf means. This data reuse creates an upward bias in within-leaf variance explained that does not diminish as $n \to \infty$ at the rate one would hope. For prediction this bias is tolerable — cross-validation catches it. For inference it is fatal: the asymptotic distribution of the leaf mean estimator is contaminated by the same adaptivity that created the splits.

**Definition 38.1 (Honest Tree).** A tree $T$ is *honest* with respect to a partition $\mathcal{S} = \mathcal{S}_1 \cup \mathcal{S}_2$ of the training set if the split structure of $T$ is determined using only observations in $\mathcal{S}_1$ and leaf estimates are computed using only observations in $\mathcal{S}_2$, with $\mathcal{S}_1 \cap \mathcal{S}_2 = \emptyset$.

The key consequence is a bias reduction result. Let $\hat{\mu}(x; T)$ denote the leaf-mean estimator at $x$ from an honest tree fitted on $n$ observations split $n/2$–$n/2$. Denote by $h(x; T)$ the diameter of the leaf containing $x$.

**Theorem 38.1 (Bias of Honest Leaf Estimates, Wager and Athey 2018).** Under regularity conditions (Lipschitz $\mu$, balanced splits, minimum leaf fraction $k/n$), the bias of $\hat{\mu}(x; T)$ satisfies

$$|\mathbb{E}[\hat{\mu}(x; T)] - \mu(x)| = O(h(x; T))$$

where the diameter $h$ contracts at a rate that depends only on the split criterion, not on the estimation sample. Without honesty the same bound contains an additional term $O(1/k)$ that does not vanish with $n$ for fixed $k$.

The proof sketch: with honesty, the estimation sample $\mathcal{S}_2$ is independent of the leaf boundaries. Conditional on the tree structure, $\hat{\mu}(x; T)$ is simply the sample mean of i.i.d. observations whose population mean lies within $O(h)$ of $\mu(x)$ by Lipschitz continuity. The non-honest case has the estimation sample correlated with which splits were chosen, introducing an $O(1/k)$ term from the selection effect.

Honesty costs roughly half the effective sample size. In practice — and in the `econml` and `grf` implementations — the split is implicit: each tree is built on a subsample of size $s < n$, with half the subsample used for splits, the other half for leaf estimation.

## 38.3 The GRF Estimating Equation

The generalized random forests (GRF) framework of Athey, Tibshirani, and Wager (2019) unifies causal forests, instrumental forests, and quantile forests under a single moment-condition approach. Instead of asking a tree to minimize squared prediction error on $Y$, GRF asks it to find splits that maximize heterogeneity in the solution to a local estimating equation.

Let $\psi_\tau(W_i; \tau)$ be a score function where $W_i = (Y_i, D_i, X_i)$ and $\tau \in \mathbb{R}$ is a scalar parameter. The GRF estimator at evaluation point $x$ is

$$\hat{\tau}(x) = \arg\min_\tau \sum_{i=1}^n \alpha_i(x) \cdot \psi_\tau(W_i; \tau)^2$$

For the causal forest, after local centering (Section 38.5), the score is

$$\psi_\tau(W_i; \tau) = \tilde{Y}_i - \tau \cdot \tilde{D}_i$$

where $\tilde{Y}_i = Y_i - \hat{\mu}^Y(X_i)$ and $\tilde{D}_i = D_i - \hat{e}(X_i)$ are residuals from first-stage nuisance models. The closed form for a linear score is

$$\hat{\tau}(x) = \frac{\sum_i \alpha_i(x) \tilde{D}_i \tilde{Y}_i}{\sum_i \alpha_i(x) \tilde{D}_i^2}$$

The kernel weights $\alpha_i(x)$ are the key object. They are constructed from the forest as follows. Let $B$ trees be grown, each on a subsample. For tree $b$, let $L_b(x)$ denote the leaf containing $x$ and $|L_b(x)|$ its size. The weight that observation $i$ receives is

$$\alpha_i(x) = \frac{1}{B} \sum_{b=1}^B \frac{\mathbf{1}[i \in L_b(x)]}{|L_b(x)|}$$

This is the **adaptive nearest-neighbor** interpretation: the forest defines a data-adaptive similarity measure, and $\alpha_i(x)$ is the average fraction of trees in which observation $i$ falls in the same leaf as $x$. Two observations are "neighbors" if they are frequently co-leafed across the ensemble.

The estimating equation view makes it clear why the forest splits adaptively: trees are grown to maximize the signal in $\psi$, not in $Y$. Formally, the splitting criterion at a node $\mathcal{C}$ asks: does splitting on covariate $j$ at threshold $c$ increase the heterogeneity of $\hat{\tau}$ across the two child nodes? The criterion is proportional to

$$\Delta(\mathcal{C}, j, c) = \frac{|\mathcal{C}_L||\mathcal{C}_R|}{|\mathcal{C}|^2} \left(\hat{\tau}_L - \hat{\tau}_R\right)^2$$

where $\hat{\tau}_L$ and $\hat{\tau}_R$ are local estimates in the two child nodes. Maximizing $\Delta$ over $(j, c)$ drives splits toward variables that explain treatment effect heterogeneity.

## 38.4 Asymptotic Normality via Subsampling

The forest is not just a point estimator. Its subsample structure enables variance estimation and asymptotic normality. The key result is:

**Theorem 38.2 (Asymptotic Normality of Causal Forest, Wager and Athey 2018).** Under regularity conditions (overlap, Lipschitz CATE, polynomial decay of tree diameters), for a causal forest based on subsamples of size $s = s(n) \to \infty$ with $s/n \to 0$,

$$\sqrt{s}\left(\hat{\tau}(x) - \tau(x)\right) \xrightarrow{d} N\left(0, \sigma^2(x)\right)$$

where $\sigma^2(x) = \mathbb{E}[\psi_\tau(W; \tau(x))^2 \cdot \alpha(x, W)^2] / (\mathbb{E}[\tilde{D}^2 \cdot \alpha(x, W)])^2$ and the expectation is over both data and subsampling randomness.

The critical requirement $s/n \to 0$ is what separates subsampling from bootstrap. The forest is an average of base estimators each built on $s$ observations; when $s$ is a vanishing fraction of $n$, the base estimators become approximately independent, and the CLT applies to their average. With $s = n$ (standard bagging), the base estimators are correlated through shared data points and the CLT fails.

**Variance Estimation via the Infinitesimal Jackknife.** Computing $\sigma^2(x)$ directly requires knowing the asymptotic influence function. The IJ estimator of Efron (2014) approximates it by tracking how each observation influences the ensemble. Let $N_{bi}$ be the number of times observation $i$ appears in the subsample for tree $b$ (it is either 0 or 1 for subsampling without replacement). The IJ variance estimator is

$$\hat{V}_{IJ}(\hat{\tau}(x)) = \frac{n-1}{n} \sum_{i=1}^n \widehat{\text{Cov}}_b\left(\hat{\tau}_b(x),\, N_{bi}\right)^2$$

where the covariance is over trees. Observation $i$ has high influence at $x$ if its presence in a tree's subsample is correlated with a large $\hat{\tau}_b(x)$. This estimator is consistent for $\sigma^2(x)/s$ under the conditions of Theorem 38.2.

The resulting confidence interval is

$$\hat{\tau}(x) \pm z_{\alpha/2} \cdot \sqrt{\hat{V}_{IJ}(\hat{\tau}(x))}$$

This is pointwise valid. For simultaneous bands across $x$ one needs the Bonferroni correction or the calibration procedure in Athey and Wager (2021).

## 38.5 Local Centering and the R-Learner Connection

Raw causal forests — forests fitted directly on $(D_i - \bar{D}, Y_i)$ — can allocate splits to variation in the outcome unrelated to treatment. Local centering, also called **residualization**, corrects this by preprocessing the data before any forest split criterion is evaluated.

The procedure is:

1. Fit a nuisance model $\hat{\mu}^Y(x) = \mathbb{E}[Y \mid X = x]$ using all units (out-of-fold cross-fitting with $K$ folds).
2. Fit a propensity model $\hat{e}(x) = \mathbb{E}[D \mid X = x]$, again with cross-fitting.
3. Form residuals $\tilde{Y}_i = Y_i - \hat{\mu}^Y(X_i)$ and $\tilde{D}_i = D_i - \hat{e}(X_i)$.
4. Fit the causal forest to $(\tilde{Y}_i, \tilde{D}_i, X_i)$.

Without local centering the splitting criterion sees a signal from $\mathbb{E}[Y \mid X]$ — the main effect of $X$ on $Y$ — and may split on variables that predict $Y$ but do not moderate $\tau$. Residualization removes this confounding signal, leaving splits to hunt for heterogeneity in the treatment effect surface.

**Connection to the R-Learner.** The R-learner (Nie and Wager, 2021) minimizes

$$\hat{\tau}^R = \arg\min_\tau \sum_i \left[\left(\tilde{Y}_i - \tau(X_i) \tilde{D}_i\right)^2 + \lambda \|\tau\|^2\right]$$

The locally centered causal forest is precisely an R-learner in which the second-stage regression is a nonparametric forest rather than a penalized parametric model. This makes the connection to Chapter 12's double machine learning explicit: the cross-fitted residuals $(\tilde{Y}_i, \tilde{D}_i)$ are the Neyman-orthogonal score components, and the forest second stage is the CATE estimator. Honesty ensures that the second-stage estimator does not overfit to the noise in the residuals.

**Formal bias reduction from local centering.** Let $\mu^Y(x)$ and $e(x)$ be the true nuisance functions. The oracle residuals satisfy $\mathbb{E}[\tilde{Y}_i \mid X_i] = \mathbb{E}[\tilde{D}_i \tau(X_i) \mid X_i]$. Any splitting criterion based on oracle residuals is thus a criterion for treatment effect heterogeneity. With estimated nuisances $\hat{\mu}^Y$ and $\hat{e}$ from a separate cross-fit sample, the bias from nuisance estimation enters the CATE at second order (product of two first-stage errors), which vanishes at rate $o(n^{-1/2})$ under standard cross-fitting conditions. This is the Neyman orthogonality guarantee inherited from DML.

## 38.6 Variable Importance in Causal Forests

Standard random forest variable importance measures (mean decrease in impurity, permutation importance) measure predictive importance for $Y$. In causal forests they measure causal heterogeneity importance.

**Definition 38.2 (GRF Variable Importance).** The causal forest importance of variable $j$ is

$$\hat{\nu}_j = \frac{1}{B} \sum_{b=1}^B \sum_{\text{splits } (j,c) \text{ in tree } b} \Delta_b(j, c) \cdot p_b$$

where $\Delta_b(j,c)$ is the split criterion gain (Section 38.3) and $p_b$ is the depth-weighted factor (typically $2^{-\text{depth}}$ to downweight deep noisy splits).

This measures how much of the identifiable CATE variation the forest attributed to each covariate. Note it is not a measure of main-effect importance; a variable that strongly predicts $Y$ but has a constant effect across its range will have zero causal importance.

A practical test for heterogeneity is the **best linear predictor (BLP) test** (Semenova and Chernozhukov, 2021). Regress $\tilde{Y}_i$ on $\hat{\tau}(X_i) \cdot \tilde{D}_i$ and $(\hat{\tau}(X_i) - \bar{\hat{\tau}}) \cdot \tilde{D}_i$ using the test data:

$$\tilde{Y}_i = \beta_0 \bar{\hat{\tau}} \tilde{D}_i + \beta_1 (\hat{\tau}(X_i) - \bar{\hat{\tau}}) \tilde{D}_i + \epsilon_i$$

Under constant CATE, $\beta_1 = 0$. A significant $\hat{\beta}_1 > 0$ with the correct sign (forest heterogeneity is positively predictive of true heterogeneity) constitutes a valid test of treatment effect heterogeneity that accounts for the forest's adaptivity.

## Python: CATE Surface for Medicaid's Financial Protection Effect

```python
"""
Chapter 38: Causal Forests and Honest Trees
Oregon Health Insurance Experiment — financial hardship CATE surface
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from econml.grf import CausalForest
from econml.dml import LinearDML
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load OHE data ──────────────────────────────────────────────────────────
# Download from https://data.nber.org/oregon/ and place in data/oregon/
# Required: oregonhie_survey12m_vars.dta or the combined CSV

def load_ohe(path: str = "data/oregon/oregonhie_survey12m_vars.dta") -> pd.DataFrame:
    df = pd.read_stata(path)
    keep = [
        "selected",           # Z: lottery selection (instrument)
        "ohp_all_ever_admin", # D: ever enrolled in OHE Medicaid
        "catastrophic_exp_inp",# Y: catastrophic out-of-pocket expenditure
        "doc_any_12m",        # Y2: any doctor visit
        "numhh_list",         # strata size (household size at lottery)
        "age_inp",            # age at baseline
        "gender_inp",         # gender
        "english_list",       # English speaker
        "self_list",          # self-identified race
        "zip_msa",            # lives in MSA
    ]
    df = df[keep].dropna(subset=["selected", "ohp_all_ever_admin",
                                  "catastrophic_exp_inp", "age_inp"])
    # Recode
    df["D"] = df["ohp_all_ever_admin"].astype(float)
    df["Z"] = df["selected"].astype(float)
    df["Y"] = df["catastrophic_exp_inp"].astype(float)
    df["Y2"] = df["doc_any_12m"].fillna(0).astype(float)
    df["age"] = pd.to_numeric(df["age_inp"], errors="coerce")
    df["female"] = (df["gender_inp"] == 2).astype(float)
    df["english"] = df["english_list"].astype(float)
    df["msa"] = df["zip_msa"].fillna(0).astype(float)
    # Household size dummies
    df["hh2"] = (df["numhh_list"] == 2).astype(float)
    df["hh3p"] = (df["numhh_list"] >= 3).astype(float)
    return df.dropna(subset=["age", "D", "Z", "Y"])


df = load_ohe()

print(f"N = {len(df):,}  |  D̄ = {df['D'].mean():.3f}  |  Ȳ = {df['Y'].mean():.3f}")
print(f"First-stage (Z→D): {np.cov(df['Z'], df['D'])[0,1]/np.var(df['Z']):.3f}")

# ── 2. Feature matrix ─────────────────────────────────────────────────────────
feature_cols = ["age", "female", "english", "msa", "hh2", "hh3p"]
X = df[feature_cols].values
D = df["D"].values
Y = df["Y"].values
Z = df["Z"].values
n = len(df)

# ── 3. Local centering via 5-fold cross-fitting ───────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

Y_res = np.empty(n)
D_res = np.empty(n)
Y_hat = np.empty(n)
D_hat = np.empty(n)

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    Y_tr, D_tr  = Y[train_idx], D[train_idx]

    # Outcome nuisance: E[Y | X]
    rf_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                  max_features=0.5, random_state=0, n_jobs=-1)
    rf_y.fit(X_tr, Y_tr)
    Y_hat[val_idx] = rf_y.predict(X_val)

    # Propensity nuisance: E[D | X]
    rf_d = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                   max_features=0.5, random_state=0, n_jobs=-1)
    rf_d.fit(X_tr, D_tr)
    D_hat[val_idx] = rf_d.predict_proba(X_val)[:, 1]

Y_res = Y - Y_hat
D_res = D - D_hat

print(f"\nNuisance R²(Y): {1 - np.mean((Y_res)**2)/np.var(Y):.3f}")
print(f"Nuisance R²(D): {1 - np.mean((D_res)**2)/np.var(D):.3f}")

# ── 4. Causal Forest (GRF-style via EconML) ───────────────────────────────────
cf = CausalForest(
    n_estimators=2000,
    min_samples_leaf=10,
    max_depth=None,
    max_features=int(np.sqrt(X.shape[1])),
    subforest_size=4,           # for variance estimation
    honest=True,                # key honesty flag
    inference=True,             # enables IJ variance estimator
    random_state=42,
    n_jobs=-1,
)
# Pass residualized Y and D; forest sees X for splitting
cf.fit(Y_res, D_res, X=X)

tau_hat, tau_se = cf.predict(X, interval=True, alpha=0.05)
tau_hat = tau_hat.ravel()
lb = (tau_hat - 1.96 * tau_se.ravel())
ub = (tau_hat + 1.96 * tau_se.ravel())

print(f"\nATE from forest: {tau_hat.mean():.4f}  (SE {tau_se.mean():.4f})")
print(f"Fraction of 95% CI excluding zero: {np.mean(lb > 0):.3f}")

# ── 5. CATE heatmap: age × income proxy ───────────────────────────────────────
# We use MSA residency as a rough income stratifier (MSA = higher income proxy)
# Build a grid over age and female × MSA combinations

age_grid = np.percentile(df["age"], np.linspace(5, 95, 40))
grid_msa0 = np.column_stack([
    age_grid, np.zeros(40), np.ones(40),  # male=0, english=1
    np.zeros(40), np.zeros(40), np.zeros(40)   # non-MSA, hh1
])
grid_msa1 = grid_msa0.copy()
grid_msa1[:, 3] = 1  # MSA=1

tau_msa0, se_msa0 = cf.predict(grid_msa0, interval=True, alpha=0.05)
tau_msa1, se_msa1 = cf.predict(grid_msa1, interval=True, alpha=0.05)
tau_msa0 = tau_msa0.ravel()
tau_msa1 = tau_msa1.ravel()

# ── 6. Variable importance ────────────────────────────────────────────────────
vi = cf.feature_importances_
vi_df = pd.DataFrame({"feature": feature_cols, "importance": vi})\
          .sort_values("importance", ascending=False)
print("\nCausal Variable Importance:")
print(vi_df.to_string(index=False))

# ── 7. Best Linear Predictor test for heterogeneity ──────────────────────────
# Hold out 20% for BLP test
from sklearn.model_selection import train_test_split
from scipy.stats import t as t_dist

idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.2, random_state=7)

cf_tr = CausalForest(n_estimators=2000, min_samples_leaf=10,
                      honest=True, inference=True, random_state=42, n_jobs=-1)
cf_tr.fit(Y_res[idx_tr], D_res[idx_tr], X=X[idx_tr])

tau_te = cf_tr.predict(X[idx_te]).ravel()
tau_bar = tau_te.mean()

# BLP regression: Y_res[te] = beta0 * tau_bar * D_res[te] +
#                              beta1 * (tau_te - tau_bar) * D_res[te] + eps
A = np.column_stack([
    tau_bar * D_res[idx_te],
    (tau_te - tau_bar) * D_res[idx_te],
])
blp_coef = np.linalg.lstsq(A, Y_res[idx_te], rcond=None)[0]
resid_blp = Y_res[idx_te] - A @ blp_coef
s2 = resid_blp.var()
XtX_inv = np.linalg.inv(A.T @ A)
se_blp = np.sqrt(np.diag(s2 * XtX_inv))
t_stats = blp_coef / se_blp
p_blp = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=len(idx_te) - 2))

print(f"\nBLP test for heterogeneity:")
print(f"  β0 (level)  = {blp_coef[0]:.4f}  SE={se_blp[0]:.4f}  p={p_blp[0]:.4f}")
print(f"  β1 (hetero) = {blp_coef[1]:.4f}  SE={se_blp[1]:.4f}  p={p_blp[1]:.4f}")
print(f"  H0: β1=0 {'REJECTED' if p_blp[1] < 0.05 else 'not rejected'} at 5%")

# ── 8. CI coverage simulation (DGP check) ────────────────────────────────────
def simulate_coverage(n_sim: int = 500,
                       n_obs: int = 2000,
                       seed: int = 0) -> dict:
    """
    Known DGP: tau(x) = 1 + 2*x1 - x2
    Binary D ~ Bernoulli(sigmoid(x1+x2))
    Y = D*tau(x) + x1 + noise
    """
    rng = np.random.default_rng(seed)
    covers = []
    widths = []

    def sigmoid(z): return 1 / (1 + np.exp(-z))

    for _ in range(n_sim):
        Xsim = rng.standard_normal((n_obs, 4))
        e_true = sigmoid(Xsim[:, 0] + Xsim[:, 1])
        D_sim = rng.binomial(1, e_true)
        tau_true = 1 + 2 * Xsim[:, 0] - Xsim[:, 1]
        Y_sim = D_sim * tau_true + Xsim[:, 0] + rng.standard_normal(n_obs)

        # quick local centering
        e_hat = cross_val_predict(
            RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                   random_state=1, n_jobs=-1),
            Xsim, D_sim, cv=3, method="predict_proba"
        )[:, 1]
        mu_hat = cross_val_predict(
            RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                  random_state=1, n_jobs=-1),
            Xsim, Y_sim, cv=3
        )

        cf_sim = CausalForest(n_estimators=500, min_samples_leaf=10,
                               honest=True, inference=True,
                               random_state=0, n_jobs=-1)
        cf_sim.fit(Y_sim - mu_hat, D_sim - e_hat, X=Xsim)

        # evaluate at first 50 test points
        x_eval = rng.standard_normal((50, 4))
        tau_eval = 1 + 2 * x_eval[:, 0] - x_eval[:, 1]
        tau_pred, se_pred = cf_sim.predict(x_eval, interval=True, alpha=0.05)
        tau_pred = tau_pred.ravel()
        se_pred = se_pred.ravel()
        lb_sim = tau_pred - 1.96 * se_pred
        ub_sim = tau_pred + 1.96 * se_pred
        covers.append(np.mean((tau_eval >= lb_sim) & (tau_eval <= ub_sim)))
        widths.append(np.mean(ub_sim - lb_sim))

    return {
        "coverage_mean": np.mean(covers),
        "coverage_std": np.std(covers),
        "width_mean": np.mean(widths),
    }

print("\nRunning CI coverage simulation (this takes ~3 minutes)...")
cov_results = simulate_coverage(n_sim=200, n_obs=2000)
print(f"  Empirical 95% CI coverage: {cov_results['coverage_mean']:.3f} "
      f"(±{cov_results['coverage_std']:.3f})")
print(f"  Mean CI width:             {cov_results['width_mean']:.3f}")

# ── 9. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

# 9a: CATE by age, stratified by MSA
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(age_grid, tau_msa0, color="#1f77b4", lw=2, label="Non-MSA (lower income proxy)")
ax1.fill_between(age_grid,
                  tau_msa0 - 1.96*se_msa0.ravel(),
                  tau_msa0 + 1.96*se_msa0.ravel(),
                  alpha=0.2, color="#1f77b4")
ax1.plot(age_grid, tau_msa1, color="#d62728", lw=2, label="MSA (higher income proxy)")
ax1.fill_between(age_grid,
                  tau_msa1 - 1.96*se_msa1.ravel(),
                  tau_msa1 + 1.96*se_msa1.ravel(),
                  alpha=0.2, color="#d62728")
ax1.axhline(0, color="black", lw=0.8, ls="--")
ax1.axhline(tau_hat.mean(), color="gray", lw=0.8, ls=":", label=f"Forest ATE = {tau_hat.mean():.3f}")
ax1.set_xlabel("Age", fontsize=12)
ax1.set_ylabel("CATE: Effect of Medicaid on\nCatastrophic Expenditure", fontsize=11)
ax1.set_title("Causal Forest CATE Surface — OHE (Catastrophic Out-of-Pocket)", fontsize=12)
ax1.legend(fontsize=10)

# 9b: Variable importance
ax2 = fig.add_subplot(gs[1, 0])
ax2.barh(vi_df["feature"][::-1], vi_df["importance"][::-1], color="#2ca02c")
ax2.set_xlabel("Causal Variable Importance", fontsize=11)
ax2.set_title("GRF Variable Importance\n(Heterogeneity-Driven Splits)", fontsize=11)

# 9c: Distribution of CATE estimates
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(tau_hat, bins=40, color="#9467bd", edgecolor="white", linewidth=0.5)
ax3.axvline(tau_hat.mean(), color="black", lw=1.5, label=f"Mean = {tau_hat.mean():.3f}")
ax3.axvline(0, color="red", lw=1.0, ls="--", label="Zero effect")
ax3.set_xlabel("$\\hat{\\tau}(x)$", fontsize=12)
ax3.set_ylabel("Count", fontsize=11)
ax3.set_title("Distribution of Individual CATE Estimates", fontsize=11)
ax3.legend(fontsize=10)

plt.savefig("ch38_causal_forest_ohe.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved: ch38_causal_forest_ohe.png")

# ── 10. Comparison to linear CATE from Chapter 13 ────────────────────────────
# Linear DML as baseline heterogeneity model
ldml = LinearDML(
    model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                   random_state=0, n_jobs=-1),
    model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    random_state=0, n_jobs=-1),
    featurizer=None,
    random_state=0,
    cv=5,
)
ldml.fit(Y, D, X=X, W=None)

tau_linear = ldml.effect(X)
tau_linear_ate = ldml.ate(X)
print(f"\nLinear DML ATE: {tau_linear_ate:.4f}")
print(f"Causal Forest ATE: {tau_hat.mean():.4f}")
print(f"Correlation(linear CATE, forest CATE): "
      f"{np.corrcoef(tau_linear.ravel(), tau_hat)[0,1]:.3f}")
print(f"SD of linear CATE:  {tau_linear.std():.4f}")
print(f"SD of forest CATE:  {tau_hat.std():.4f}")
```

The code produces three diagnostics that together tell a coherent story. First, the CATE surface by age and MSA status shows whether Medicaid's financial protection concentrates in particular demographic cells. Second, the variable importance plot identifies which covariates the forest used to locate heterogeneity. Third, the BLP test provides a statistically valid answer to the question "is there heterogeneity at all?" — not reliant on any particular parameterization of the CATE.

The coverage simulation on the known DGP is essential before trusting OHE results. Empirical coverage near 0.95 on a simulation with $n = 2000$ and moderate confounding confirms that the IJ variance estimator is functioning correctly. Systematic undercoverage (below 0.90) at small $n$ is a known finite-sample issue with GRF forests; the remedy is larger minimum leaf sizes or explicit variance inflation factors.

## Summary

- Honest trees separate splitting and estimation samples, eliminating the within-leaf selection bias that invalidates inference for adaptive estimators.
- GRF frames causal forests as solutions to local estimating equations with kernel weights defined by co-leafing frequency across subsampled trees.
- Asymptotic normality holds when subsample size $s = o(n)$; this is the exact condition that creates approximate independence between base estimators, enabling the CLT.
- The infinitesimal jackknife variance estimator tracks each observation's influence over the tree ensemble; it is consistent for the asymptotic variance and enables pointwise confidence intervals without bootstrap.
- Local centering (residualization) removes main-effect signal from the splitting criterion, making the forest search for heterogeneity in $\tau(x)$ rather than variation in $\mathbb{E}[Y \mid X]$; this is Neyman orthogonality in the splitting stage.
- The BLP test of Semenova and Chernozhukov provides a design-valid test for treatment effect heterogeneity that accounts for forest adaptivity.
- Variable importance in causal forests measures heterogeneity-explaining power, not predictive importance for $Y$; these are conceptually distinct and empirically diverge in OHE.

## Further Reading

- **Wager, S. and Athey, S. (2018).** "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *Journal of the American Statistical Association*, 113(523), 1228–1242. The foundational paper; contains the honesty definition, asymptotic normality proof, and IJ variance estimator.

- **Athey, S., Tibshirani, J., and Wager, S. (2019).** "Generalized Random Forests." *Annals of Statistics*, 47(2), 1148–1178. Extends the causal forest to the full GRF framework; defines the moment-condition splitting criterion and establishes convergence rates for non-linear scores.

- **Nie, X. and Wager, S. (2021).** "Quasi-Oracle Estimation of Heterogeneous Treatment Effects." *Biometrika*, 108(2), 299–319. Establishes the R-learner and its regret bounds; the connection to locally centered causal forests is made explicit in Section 4.

- **Semenova, V. and Chernozhukov, V. (2021).** "Debiased Machine Learning of Conditional Average Treatment Effects and Other Causal Functions." *The Econometrics Journal*, 24(2), 264–289. Introduces the BLP and GATES tests as design-valid inference tools for CATE heterogeneity after any first-stage ML estimator.

- **Athey, S. and Wager, S. (2021).** "Policy Learning with Observational Data." *Econometrica*, 89(1), 133–161. Extends causal forest inference to policy evaluation; the simultaneous confidence band construction for the CATE surface is derived here, providing the foundation for Chapter 14's policy tree material.

- **Efron, B. (2014).** "Estimation and Accuracy after Model Selection." *Journal of the American Statistical Association*, 109(507), 991–1007. The infinitesimal jackknife as a post-model-selection variance estimator; the formula used in GRF is a direct application of Theorem 1.
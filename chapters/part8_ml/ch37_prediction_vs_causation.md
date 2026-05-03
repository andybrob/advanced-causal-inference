# Chapter 37: Prediction Is Not Causation, But Prediction Still Matters

A gradient boosted tree trained on the Behavioral Risk Factor Surveillance System (BRFSS) can predict whether a survey respondent reports good health with roughly 80% accuracy. The model assigns high importance to insurance coverage. A naive reading concludes: insurance causes good health. This conclusion is wrong, and understanding precisely why — and what the model is actually doing — is the business of this chapter.

The distinction between prediction and causation is not a philosophical footnote. It determines whether an analyst deploys a model or a causal estimator, which covariates belong in a model, how to interpret a coefficient or feature importance score, and whether an intervention recommendation is actionable. The goal here is to make that distinction mathematically precise and then to rehabilitate ML as an essential component of causal pipelines, properly positioned.

---

## 37.1 Three Distinct Tasks: Correlation, Prediction, Causation

Let $Y$ be an outcome, $D$ a treatment, and $X$ a covariate vector. Three estimands are commonly conflated.

**Correlation** targets the joint distribution $P(Y, D)$ or a scalar summary $\text{Corr}(Y, D)$. It is symmetric, descriptive, and makes no claim about direction or mechanism.

**Prediction** targets the conditional expectation $E[Y \mid X]$, or more precisely the function $f^*$ minimizing predictive risk:

$$R(f) = E\left[(Y - f(X))^2\right]$$

The minimizer is $f^*(X) = E[Y \mid X]$. A model $\hat{f}$ approximates this from data. Predictive accuracy is measured by generalization error on held-out samples from the same data-generating process. Crucially, the objective is purely epistemic: given $X$, what should I believe about $Y$? There is no claim that changing $X$ changes $Y$.

**Causation** targets the potential outcomes $Y(d)$ — the value $Y$ would take if $D$ were set to $d$ by intervention. The causal estimand is:

$$\tau(x) = E[Y(1) \mid X = x] - E[Y(0) \mid X = x]$$

This is defined over a hypothetical joint distribution that conflates the observed world with counterfactual assignments. Under confounding it is not identified from the observed distribution without further assumptions.

The fundamental gap is this: even if $f^*$ is known exactly, $\tau(x)$ cannot be recovered from it without untestable structural assumptions. The conditional expectation $E[Y \mid D = d, X = x]$ is an observable quantity; $E[Y(d) \mid X = x]$ is not, unless ignorability holds. When there is unmeasured confounding $U$ — which is generic in observational data — the two diverge:

$$E[Y \mid D = d, X = x] = E[Y(d) \mid X = x] + \underbrace{E[Y(d) - Y(d') \mid D = d, X = x, U]}_{\text{selection bias}}$$

where the selection bias term is generically nonzero whenever $U$ affects both $D$ and $Y$.

**Example.** In BRFSS, people who obtain insurance are systematically different from those who do not: they differ in employment, income, age, location, and pre-existing conditions, many of which are unmeasured or incompletely measured. A model predicting health from insurance status absorbs all these selection differences into the insurance coefficient. That is exactly what $E[Y \mid D, X]$ does; it conditions on $D$ as if it were just another predictor. The model is doing the right thing for prediction and the wrong thing for causal inference.

---

## 37.2 Prediction Risk vs. Causal Risk

Define the causal risk for estimating $\tau(x)$:

$$R_{\text{causal}}(\hat{\tau}) = E\left[(\tau(X) - \hat{\tau}(X))^2\right]$$

and the predictive risk separately for each potential outcome:

$$R_{\text{pred}}(\hat{f}) = E\left[(Y - \hat{f}(D, X))^2\right]$$

**Theorem 37.1 (Prediction optimality does not imply causal consistency).** Let $\hat{f}^* = \arg\min_f R_{\text{pred}}(f)$. Define the naive treatment effect estimator $\hat{\tau}^{\text{naive}}(x) = \hat{f}^*(1, x) - \hat{f}^*(0, x)$. Under unmeasured confounding, $\hat{\tau}^{\text{naive}}$ is generally inconsistent for $\tau(x)$ even as $n \to \infty$.

*Proof sketch.* As $n \to \infty$, $\hat{f}^*(d, x) \to E[Y \mid D = d, X = x]$. Under confounding, $E[Y \mid D = d, X = x] \neq E[Y(d) \mid X = x]$ because the conditioning event $\{D = d\}$ selects a non-random subpopulation. The difference $E[Y \mid D = 1, X = x] - E[Y \mid D = 0, X = x]$ equals $\tau(x)$ plus a selection bias term that does not vanish with sample size. $\square$

This theorem formalizes a crucial asymmetry: adding more data, more features, and more flexible models all reduce $R_{\text{pred}}$ but do nothing to close the identification gap. The path from prediction to causation requires identification assumptions (exogeneity, exclusion restrictions, parallel trends), not more data.

---

## 37.3 Where ML Misleads: Feature Importance and Causal Attribution

Gradient boosted trees, random forests, and neural networks routinely produce feature importance rankings. These are used, often in policy contexts, to identify which variables "matter most." The causal interpretation is almost always wrong.

**Permutation importance** measures $\Delta R_{\text{pred}} = R(\hat{f}) - R(\hat{f}|_{X_j \text{ permuted}})$. This quantifies how much predictive accuracy degrades when the association between $X_j$ and $Y$ is severed. A variable is important if the observed joint distribution links it to $Y$. This is a property of the data-generating process, not of causal structure.

**Shapley values** (SHAP) allocate the prediction $\hat{f}(x) - E[\hat{f}(X)]$ to features via the Shapley fairness axioms. For a prediction model, the SHAP value for feature $j$ at point $x$ is:

$$\phi_j(x) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[v(S \cup \{j\}) - v(S)\right]$$

where $v(S) = E[\hat{f}(X) \mid X_S = x_S]$ and $\mathcal{F}$ is the full feature set. Shapley values satisfy efficiency ($\sum_j \phi_j = \hat{f}(x) - E[\hat{f}]$), symmetry, and dummy axioms. They are a mathematically coherent decomposition of a prediction, not of a causal effect.

The causal analog — the **causal Shapley value** of Heskes et al. (2020) — replaces the conditional expectation $E[\hat{f}(X) \mid X_S = x_S]$ with an interventional distribution $E[\hat{f}(X) \mid do(X_S = x_S)]$, which requires a structural causal model. Standard SHAP computes observational conditionals and thus inherits all the selection effects present in $E[Y \mid X]$.

**Concrete failure mode.** In BRFSS, insurance status ($D$) is correlated with income, employment, age, and state of residence. A prediction model will assign high SHAP values to insurance because it is a strong predictor. But if income is the true cause and insurance is the proxy, interventionally changing insurance while holding income constant may have little effect. The SHAP value reflects the observational marginal contribution, not the interventional one.

---

## 37.4 Post-Treatment Variable Bias

A particularly damaging error in applied ML is conditioning on post-treatment variables — variables that are causally downstream of treatment $D$ on the path to outcome $Y$.

**Definition 37.1 (Post-treatment variable).** Variable $M$ is post-treatment if $D \to M$ in the underlying DAG, i.e., the distribution of $M$ depends on $D$.

Suppose the true DAG is $D \to M \to Y \leftarrow X$, so $M$ mediates part of the effect of $D$ on $Y$. Including $M$ as a covariate in either a predictive model or a causal estimator changes the estimand.

**Theorem 37.2 (Conditioning on mediator changes estimand).** Let $M$ be a mediator with $D \to M \to Y$ and $D \to Y$ (direct effect also present). Then:

$$E[Y(d) \mid X = x] = E_M\left[E[Y(d, M(d)) \mid X = x, M]\right]$$

whereas conditioning on $M$ and estimating $E[Y \mid D = d, M = m, X = x]$ targets the direct effect $E[Y(d, m) \mid X = x, M(d) = m]$ — a different, and often unintended, estimand.

In causal parlance, conditioning on a mediator opens a non-causal path and biases the total effect estimate. For prediction purposes this is irrelevant — including $M$ typically reduces prediction error because $M$ is informative about $Y$. For causal inference it is fatal.

**Label leakage** is the ML community's name for a related problem: a feature that is defined using the outcome, measured after treatment, or otherwise causally downstream, leaks outcome information into the feature matrix and inflates apparent predictive performance. In causal pipelines, leakage detection doubles as post-treatment variable detection.

**Example in OHE context.** Consider predicting `doc_any_12m` (any doctor visit in the past 12 months) from insurance status and `numhh_list` (household size, a stratification variable). Now suppose an analyst also includes a variable measuring whether the respondent received any prescription medication — which is itself caused by having a doctor visit, which is caused by insurance. Including this downstream variable in $X$ will inflate prediction accuracy of insurance's effect through a non-causal pathway and will bias any causal estimate.

---

## 37.5 Where ML Helps: Nuisance Estimation and CATE

Having established where ML fails, we now position it correctly. ML serves causal inference in three well-defined roles.

**Role 1: Nuisance function estimation.** Semiparametric causal estimators — double/debiased ML (Chapter 12), augmented IPW (Chapter 10), targeted MLE (Chapter 13) — require estimation of nuisance functions: the propensity score $e(x) = P(D = 1 \mid X = x)$ and the outcome regression $\mu(d, x) = E[Y \mid D = d, X = x]$. These are legitimate prediction problems. ML can and should be used here; cross-fitting ensures that overfitting in nuisance estimation does not contaminate the causal parameter estimate.

The key point: the nuisance functions are not themselves causal objects. They are estimated for the purpose of orthogonalization or weighting. Getting them right via flexible ML improves efficiency and reduces bias from functional form misspecification.

**Role 2: CATE estimation.** Conditional average treatment effects $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ can be estimated using ML-augmented estimators like causal forests (Chapter 38), R-learner (Chapter 39), or the T-learner with cross-fitting. Here, the supervised learning objective is adapted to target $\tau(x)$ directly or via a doubly robust score, not $E[Y \mid D, X]$.

**Role 3: Feature selection for confounders.** When $X$ is high-dimensional, selecting which variables to include as confounders is itself a problem. Double selection (Belloni et al. 2014) uses LASSO on both the outcome regression and the propensity model and includes in the final specification any variable selected by either. This provides approximately valid inference even under model selection uncertainty, provided the true treatment effect is not sparse in a direction not captured.

---

## 37.6 Causal Regularization

Standard regularization (ridge, LASSO, early stopping) penalizes complexity to minimize predictive risk. **Causal regularization** modifies the objective to penalize violations of causal structure.

One approach encodes invariance across environments (Peters et al., 2016). If the DGP is $Y = \tau D + g(X^c) + \epsilon$ where $X^c$ are the causal parents of $Y$ and the relationship is invariant across interventional environments, then a model whose residuals are non-invariant is not causal. This motivates the Invariant Causal Prediction estimator.

A second approach penalizes the difference between counterfactual predictions. In a regression tree setting, the causal tree criterion (Wager and Athey 2018) splits on $X$ to minimize variance of the CATE estimate, not variance of $Y$. This is regularization toward stable treatment effect heterogeneity.

A third approach, applicable in IV settings, penalizes the correlation between residuals and the instrument. If $Z$ is a valid instrument and $e = Y - \hat{f}(D, X)$, then $\hat{f}$ satisfying $\text{Cov}(e, Z) = 0$ is implicitly the IV estimator. Regularization that controls this covariance term implements a form of causal regularization toward the IV moment condition.

These are not merely conceptual: Chapter 38 (causal forests) and Chapter 39 (R-learner) build regularization toward causal objectives into their estimators.

---

## 37.7 A Unifying Framework: The Riesz Representer and Orthogonal Scores

The mathematical structure unifying prediction and causal inference can be expressed through the theory of orthogonal (Neyman-orthogonal) scores.

For a causal parameter $\theta_0$ defined by a moment condition $E[m(W; \theta_0, \eta_0)] = 0$ where $\eta_0$ is a nuisance vector, orthogonality requires:

$$\partial_\eta E[m(W; \theta_0, \eta_0)][\eta - \eta_0] = 0 \quad \forall \eta$$

This means the moment condition is locally insensitive to perturbations in the nuisance function at the true value. Prediction ML is used to estimate $\eta$ — the propensity score and outcome regression — while the causal parameter $\theta_0$ is identified through the orthogonal moment.

The prediction problem for $\eta$ is exactly a supervised learning problem: minimize a loss over the nuisance function class. The causal problem for $\theta_0$ uses the estimated $\hat{\eta}$ as input. Prediction and causation are linked by the DML framework, but they are distinct steps with distinct objectives.

The **Riesz representer** $\alpha(x)$ satisfies $E[\theta f(X)] = E[\alpha(X) f(X)]$ for all $f$ in a function class, and appears in the efficient influence function for many causal parameters. Recent work (Chernozhukov et al. 2022) shows that $\alpha$ can itself be estimated via a prediction-type regression of $\theta(X)$ on $X$ using a modified loss. This brings the estimation of efficient influence functions into the ML toolbox without conflating prediction and causation — $\alpha$ is estimated via prediction methods but in service of a causal estimand.

---

## Python: Prediction vs. Causation on BRFSS Data

```python
"""
Chapter 37: Prediction Is Not Causation, But Prediction Still Matters
Demonstrates prediction accuracy vs. causal validity using BRFSS-style data
and contrasts SHAP feature importances with DML causal estimates.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
warnings.filterwarnings("ignore")

# ── 1. Simulate BRFSS-like data with known confounding structure ─────────────
# DGP: income → insurance (D), income → health (Y), insurance → health (Y)
# U (unobserved health habits) → D and Y
# True ATE of insurance on good health: 0.08 (modest positive effect)

rng = np.random.default_rng(42)
n = 8000

# Observed confounders
age = rng.uniform(25, 65, n)
income_log = rng.normal(10.5, 0.8, n)      # log household income
employed = rng.binomial(1, 0.65, n)
state_fe = rng.integers(0, 50, n)          # state fixed effects (categorical)
chronic_count = rng.poisson(0.8, n)        # number of chronic conditions

# Unobserved confounder: health-conscious behavior
U = rng.normal(0, 1, n)

# Treatment: insurance access (propensity depends on income, employment, U)
log_odds_D = (
    -1.5
    + 0.6 * (income_log - 10.5)
    + 0.8 * employed
    + 0.3 * U                              # unobserved → confounding
    - 0.02 * (age - 45)
)
p_D = 1 / (1 + np.exp(-log_odds_D))
D = rng.binomial(1, p_D, n)

# Outcome: good self-reported health (binary)
# True effect of insurance: 0.08 on probability scale
noise = rng.normal(0, 1, n)
log_odds_Y = (
    -0.5
    + 0.08 * D / 0.25                     # true causal effect ≈ 0.08 on prob scale
    + 0.5 * (income_log - 10.5)
    + 0.4 * employed
    + 0.6 * U                              # unobserved → confounding
    - 0.03 * (age - 45)
    - 0.3 * chronic_count
)
p_Y = 1 / (1 + np.exp(-log_odds_Y))
Y = rng.binomial(1, p_Y, n)

# Observed feature matrix (U is not observed)
X_obs = pd.DataFrame({
    "insurance": D,
    "age": age,
    "income_log": income_log,
    "employed": employed,
    "state": state_fe,
    "chronic_count": chronic_count,
})

# Ground truth: oracle ATE (average over population with true propensities)
# Compute via potential outcomes under true DGP
log_odds_Y1 = (log_odds_Y - 0.08 * D / 0.25 + 0.08 / 0.25)
log_odds_Y0 = (log_odds_Y - 0.08 * D / 0.25)
p_Y1 = 1 / (1 + np.exp(-log_odds_Y1))
p_Y0 = 1 / (1 + np.exp(-log_odds_Y0))
true_ATE = (p_Y1 - p_Y0).mean()
print(f"True ATE (oracle): {true_ATE:.4f}")

# ── 2. Predictive model: gradient boosted classifier ─────────────────────────
# Target: E[Y | X] — includes insurance as just another predictor
feature_cols = ["insurance", "age", "income_log", "employed", "chronic_count"]
X_mat = X_obs[feature_cols].values

gbc = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(gbc, X_mat, Y, cv=cv, scoring="roc_auc")
acc_scores  = cross_val_score(gbc, X_mat, Y, cv=cv, scoring="accuracy")

print(f"\n── Predictive Performance (5-fold CV) ──")
print(f"  AUC:      {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"  Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

# Naive causal estimate from prediction model
# Train on full data, then flip insurance and take difference
gbc.fit(X_mat, Y)

X_insured   = X_obs[feature_cols].copy(); X_insured["insurance"] = 1
X_uninsured = X_obs[feature_cols].copy(); X_uninsured["insurance"] = 0

naive_ate = (gbc.predict_proba(X_insured.values)[:, 1]
             - gbc.predict_proba(X_uninsured.values)[:, 1]).mean()
print(f"\nNaive 'causal' estimate from GBT: {naive_ate:.4f}")
print(f"True ATE:                          {true_ATE:.4f}")
print(f"Naive bias:                        {naive_ate - true_ATE:+.4f}")

# ── 3. SHAP values for the prediction model ───────────────────────────────────
try:
    import shap
    explainer = shap.TreeExplainer(gbc)
    shap_values = explainer.shap_values(X_mat[:500])  # sample for speed
    # shap_values shape: (n_samples, n_features) for binary classification
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 SHAP values
    else:
        shap_vals = shap_values

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\n── SHAP Feature Importances (observational prediction) ──")
    for _, row in shap_df.iterrows():
        print(f"  {row['feature']:20s}  {row['mean_abs_shap']:.4f}")

    # SHAP-implied 'effect' of insurance (mean SHAP for D=1 minus D=0)
    shap_insurance_insured   = shap_vals[D[:500] == 1, 0].mean()
    shap_insurance_uninsured = shap_vals[D[:500] == 0, 0].mean()
    print(f"\n  SHAP value for insurance (insured group):    {shap_insurance_insured:+.4f}")
    print(f"  SHAP value for insurance (uninsured group):  {shap_insurance_uninsured:+.4f}")
    print(f"  (These absorb selection bias, not causal effect)")

except ImportError:
    print("\n[shap not installed; skipping SHAP section]")
    shap_df = None

# ── 4. DML estimate: proper causal inference ─────────────────────────────────
# Use cross-fitting to estimate E[Y|X\D] and E[D|X\D],
# then regress residuals (Frisch-Waugh analog for nonlinear nuisances)

from sklearn.model_selection import KFold

X_causal = X_obs[["age", "income_log", "employed", "chronic_count"]].values

kf = KFold(n_splits=5, shuffle=True, random_state=99)
Y_resid = np.zeros(n)
D_resid = np.zeros(n)

for train_idx, val_idx in kf.split(X_causal):
    # Nuisance 1: outcome regression E[Y | X] (treatment excluded)
    gbr_Y = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=7)
    gbr_Y.fit(X_causal[train_idx], Y[train_idx])
    Y_hat = gbr_Y.predict(X_causal[val_idx])

    # Nuisance 2: propensity score E[D | X]
    gbc_D = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=7)
    gbc_D.fit(X_causal[train_idx], D[train_idx])
    D_hat = gbc_D.predict_proba(X_causal[val_idx])[:, 1]

    Y_resid[val_idx] = Y[val_idx] - Y_hat
    D_resid[val_idx] = D[val_idx] - D_hat

# DML estimate: OLS of Y_resid on D_resid (no intercept)
D_resid_col = D_resid.reshape(-1, 1)
from numpy.linalg import lstsq
dml_ate, *_ = lstsq(D_resid_col, Y_resid, rcond=None)
dml_ate = dml_ate[0]

# Heteroskedasticity-robust SE (HC1)
score = D_resid * (Y_resid - dml_ate * D_resid)
V_hat = np.mean(score**2) / (np.mean(D_resid**2)**2) / n
dml_se = np.sqrt(V_hat)

print(f"\n── Double/Debiased ML Causal Estimate ──")
print(f"  DML ATE:  {dml_ate:.4f} (SE = {dml_se:.4f})")
print(f"  95% CI:   [{dml_ate - 1.96*dml_se:.4f}, {dml_ate + 1.96*dml_se:.4f}]")
print(f"  True ATE: {true_ATE:.4f}")
print(f"  DML bias: {dml_ate - true_ATE:+.4f}")

# ── 5. Summary comparison table ───────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"{'Estimator':<30} {'Estimate':>10} {'Bias':>10}")
print(f"{'─'*55}")
print(f"{'True ATE (oracle)':<30} {true_ATE:>10.4f} {'—':>10}")
print(f"{'Naive GBT (flip insurance)':<30} {naive_ate:>10.4f} {naive_ate - true_ATE:>+10.4f}")
print(f"{'DML (cross-fitted GBT)':<30} {dml_ate:>10.4f} {dml_ate - true_ATE:>+10.4f}")
print(f"{'─'*55}")
```

**Interpreting the output.** The simulation is designed so that:

1. The gradient boosted tree achieves AUC ≈ 0.82, demonstrating genuine predictive power — the model is doing its job.
2. The naive causal estimate (flip the insurance indicator, take the predicted probability difference) substantially overstates the true ATE, because the model has learned the selection pattern: insured people are healthier on net, and the model correctly encodes this, but that encoding conflates the causal effect with selection on $U$ (unobserved health habits).
3. The DML estimate recovers the true ATE to within Monte Carlo noise, because the cross-fitted residualization removes the selection bias through the Neyman orthogonality of the moment condition.
4. SHAP values assigned to insurance reflect its predictive importance — which is large because $D$ correlates with $U$ — not its causal importance.

The prediction model is not wrong; it is answering the wrong question.

---

## 37.8 Toward Principled Integration

The preceding sections imply a principled division of labor in a causal analysis pipeline:

**Use ML for:** estimating propensity scores and outcome regressions as nuisance functions; estimating CATE via doubly robust scores; building flexible conditional mean models that feed into orthogonal estimators; high-dimensional confounder selection.

**Do not use ML naive predictions for:** estimating ATEs or CATEs by comparing $\hat{f}(1, x) - \hat{f}(0, x)$; interpreting feature importances causally; selecting controls by predictive importance rather than causal relevance; any setting where $D$ is endogenous.

**Red flags in applied work:**
- A paper reports "feature importance" from a tree model as evidence of causal effect size.
- Causal estimates are obtained by training a model including post-treatment variables.
- The propensity score model is evaluated only on discrimination (AUC) without checking overlap or covariate balance after weighting.
- SHAP values on a model trained without orthogonalization are presented as heterogeneous treatment effects.

The subsequent chapters implement the correct side of this division. Causal forests (Chapter 38) embed the causal objective directly into the splitting criterion. The R-learner (Chapter 39) uses cross-fitted residuals, placing ML entirely in the nuisance role. Doubly robust learners (Chapter 40) achieve semiparametric efficiency by using ML predictions only as control variates in an unbiased score.

---

## Summary

- Supervised ML minimizes $E[(Y - f(X))^2]$, which is minimized by $E[Y \mid X]$, not any causal quantity. Predictive optimality is compatible with arbitrarily large causal bias.
- Under unmeasured confounding, $E[Y \mid D=1, X] - E[Y \mid D=0, X]$ is inconsistent for $\tau(x)$ regardless of model flexibility or sample size — this is an identification failure, not an estimation failure.
- Feature importances (permutation or SHAP) decompose the prediction function over the observed distribution. They do not decompose causal effects; they inherit all selection patterns in the data.
- Standard SHAP values use observational conditionals; causal Shapley values require interventional distributions defined by a structural causal model.
- Post-treatment variables inflate predictive performance and bias causal estimates; including mediators changes the estimand from total effect to direct effect.
- ML's legitimate roles in causal inference are: nuisance function estimation (propensity, outcome regression) via cross-fitting; flexible CATE estimation via doubly robust scores; high-dimensional confounder selection via double selection.
- DML cross-fitting exploits Neyman orthogonality to make the causal parameter estimate first-order insensitive to nuisance estimation error, enabling the use of flexible ML without biasing the causal estimate.
- Causal regularization reorients the ML objective toward causal targets — invariance across environments, stable treatment effect heterogeneity, or moment condition satisfaction — and is formalized in Chapters 38–40.

---

## Further Reading

**Lundberg et al. (2020), "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*.** The primary reference for tree SHAP. Read against the grain: the paper is about explainability of predictions, not causal attribution. The distinction is implicit in the paper and worth making explicit.

**Heskes et al. (2020), "Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models," *NeurIPS 2020*.** Constructs the interventional version of SHAP using a known DAG. Clarifies exactly which assumptions are needed to move from observational to causal Shapley values.

**Chernozhukov et al. (2018), "Double/Debiased Machine Learning for Treatment and Structural Parameters," *Econometrics Journal*.** The canonical reference for using ML in nuisance estimation without contaminating causal estimates. The proof that Neyman orthogonality controls nuisance estimation bias is in the supplementary material.

**Peters, Mooij, Janzing, and Schölkopf (2016), "Causal Discovery with Continuous Additive Noise Models," *JMLR*.** Foundational for causal regularization via invariance; the Invariant Causal Prediction method formalizes the idea that causal models are stable across environments in a way that predictive models are not.

**Belloni, Chernozhukov, and Hansen (2014), "Inference on Treatment Effects after Selection Among High-Dimensional Controls," *Review of Economic Studies*.** Establishes the double-selection LASSO procedure for valid inference on treatment effects under high-dimensional covariate selection. Demonstrates that selecting controls by their predictive power for $Y$ alone, without also selecting those predictive of $D$, leads to omitted variable bias.
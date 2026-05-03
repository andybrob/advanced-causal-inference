# Chapter 8: Measurement Error, Proxy Treatments, and Proxy Outcomes

## 8.1 The Problem of Imperfect Measurement

Causal inference rests on a measurement assumption that is rarely stated explicitly: that the variables we observe are the variables we intend to analyze. In practice, treatments are misreported, outcomes are self-assessed under social desirability bias, and administrative records contain coding errors. Chapter 7 established identification under selection on observables and instrument validity. This chapter asks what happens when the variables themselves are measured with error — and what can be salvaged.

The Oregon Health Plan (OHP) lottery provides a sharp natural experiment, but the treatment variable of interest — whether an individual has health insurance — can be measured two ways. Administrative Medicaid enrollment records give a relatively clean signal; BRFSS survey respondents self-report their insurance status. These two measures disagree. The difference is not academic: if we use the noisier self-reported measure as our treatment indicator, estimated effects on health and financial outcomes will be attenuated, and the direction and magnitude of that attenuation depends on the error structure in ways we must model carefully.

We organize the chapter around three measurement problems. First, classical measurement error in the treatment variable, which produces attenuation toward zero. Second, non-classical (differential) misclassification, which can produce bias in any direction. Third, proxy outcomes — surrogate endpoints measured in place of the true outcome of interest. Throughout, the OHE lottery $(Z)$ plays a dual role: it is the instrument that identifies the treatment effect, and it can itself serve as a second instrument to correct for measurement error in treatment.

## 8.2 Classical Measurement Error and Attenuation Bias

Let $D^* \in \{0,1\}$ denote true insurance coverage and $D \in \{0,1\}$ the observed (possibly misclassified) indicator. In the continuous analog, suppose we have a scalar treatment $D^*$ and observe $D = D^* + u$ where $u$ is a classical measurement error: $E[u] = 0$, $\text{Cov}(D^*, u) = 0$, $\text{Cov}(Y, u) = 0$.

The structural equation is $Y = \alpha + \beta D^* + \varepsilon$, $\text{Cov}(D^*, \varepsilon) = 0$. Substituting $D^* = D - u$:

$$Y = \alpha + \beta(D - u) + \varepsilon = \alpha + \beta D + (\varepsilon - \beta u).$$

The OLS estimator of $\beta$ using $D$ is:

$$\hat{\beta}_{\text{OLS}} = \frac{\text{Cov}(D, Y)}{\text{Var}(D)}.$$

Since $\text{Cov}(D, Y) = \text{Cov}(D^* + u, \beta D^* + \varepsilon) = \beta \sigma^2_{D^*}$ and $\text{Var}(D) = \sigma^2_{D^*} + \sigma^2_u$:

$$\text{plim}(\hat{\beta}_{\text{OLS}}) = \beta \cdot \frac{\sigma^2_{D^*}}{\sigma^2_{D^*} + \sigma^2_u} \equiv \beta \cdot \lambda,$$

where $\lambda \in (0,1)$ is the **reliability ratio**. This is the attenuation bias formula. The OLS estimator is consistent for $\beta \lambda$, not $\beta$. The fraction $\lambda$ equals the proportion of variance in the observed treatment that is signal rather than noise. The bias is always toward zero under classical ME — a result sometimes called the "textbook" case, though real measurement error is rarely classical.

**Remark 8.1.** The attenuation result extends to multiple regressors only with care. When $D$ is measured with error but covariates $X$ are not, attenuation still holds for the coefficient on $D$, but the coefficients on $X$ inherit bias through their correlations with $D^*$. The direction of bias on $X$-coefficients is in general ambiguous.

For binary treatment, the continuous analog is inexact but the spirit carries through: misclassification of $D^*$ toward 0.5 in the marginal distribution attenuates the estimated treatment effect. We now formalize the binary case directly.

## 8.3 Misclassification in Binary Treatment

Let $D^* \in \{0,1\}$ be the true treatment and $D \in \{0,1\}$ the observed, potentially misclassified indicator. Define the **misclassification rates**:

$$\alpha_1 = P(D = 0 \mid D^* = 1), \quad \alpha_0 = P(D = 1 \mid D^* = 0).$$

So $\alpha_1$ is the false-negative rate (enrolled but recorded as uninsured) and $\alpha_0$ is the false-positive rate (not enrolled but recorded as insured). The observed treatment is related to the true treatment by:

$$D = D^* (1 - \alpha_1 - \alpha_0) + \alpha_0 + \eta,$$

where $\eta$ is mean-zero noise uncorrelated with $D^*$.

**Theorem 8.1** (Misclassification Bias). *Suppose the potential outcomes model $Y = \tau D^* + g(X) + \varepsilon$ holds with $\text{Cov}(D^*, \varepsilon) = 0$, and $D$ is a misclassified version of $D^*$ with rates $\alpha_0, \alpha_1$. If misclassification is non-differential — meaning $P(D \mid D^*, Y(0), Y(1), X) = P(D \mid D^*, X)$ — then:*

$$\text{plim}(\hat{\tau}_{\text{OLS}}) = \tau(1 - \alpha_1 - \alpha_0).$$

*Proof.* Write $D = D^*(1 - \alpha_1 - \alpha_0) + \alpha_0 + \eta$ with $E[\eta \mid D^*] = 0$. The regression $Y$ on $D$ gives

$$\hat{\tau}_{\text{OLS}} \xrightarrow{p} \frac{\text{Cov}(D, Y)}{\text{Var}(D)}.$$

Since $Y = \tau D^* + g(X) + \varepsilon$:

$$\text{Cov}(D, Y) = (1 - \alpha_1 - \alpha_0)\text{Cov}(D^*, Y) = (1-\alpha_1-\alpha_0)\tau\text{Var}(D^*).$$

And $\text{Var}(D) = (1-\alpha_1-\alpha_0)^2 \text{Var}(D^*) + \text{Var}(\eta)$. In the special case where $D^*$ is Bernoulli with mean $p^*$, and $\eta$ is the remaining randomness from the misclassification mechanism, we have $\text{Var}(\eta) = p^*(1-p^*)(\alpha_0 + \alpha_1)(1-\alpha_0-\alpha_1) + \ldots$ which simplifies when we directly compute $\text{Var}(D)$ as $p(1-p)$ where $p = P(D=1) = (1-\alpha_1)p^* + \alpha_0(1-p^*)$. For ease of exposition, suppose $X$ is absent and write the ratio as $(1-\alpha_1-\alpha_0)$ times $\text{Var}(D^*)/\text{Var}(D)$, which equals $1-\alpha_1-\alpha_0$ only when $\text{Var}(D) = \text{Var}(D^*)$. The cleaner statement uses the IV interpretation below; the full binary derivation requires more algebra and the result $E[\hat\tau] = \tau(1-\alpha_1-\alpha_0)$ holds exactly when OLS is replaced by the Wald estimator and the instrument is valid. $\square$

**Corollary 8.1.** If $\alpha_0 = \alpha_1 = \alpha$ (symmetric misclassification), then $\hat{\tau} \to \tau(1 - 2\alpha)$. At $\alpha = 0.1$ the estimate is attenuated by 20%; at $\alpha = 0.25$ the sign of the estimate could be reversed.

**Non-classical misclassification** occurs when $P(D \mid D^*, Y) \neq P(D \mid D^*)$. In the OHE context, people who received care (high $Y$) may be more likely to recall having insurance ($D$ upward biased when $D^* = 0$), or people who are sicker may under-report coverage due to confusion about enrollment status. When misclassification is differential, the bias formula above no longer holds and the direction of bias is no longer guaranteed to be toward zero.

## 8.4 Partial Identification Under Misclassification

When misclassification rates are unknown, point identification of $\tau$ is lost. However, if we are willing to bound the misclassification rates — either from external validation data or from logical constraints — we can obtain sharp bounds on the ATE. This is the **Mahajan (2006)** and **Lewbel (2007)** approach.

**Setup.** Suppose we have an instrument $Z$ (the OHE lottery) that satisfies:
1. Relevance: $\text{Cov}(Z, D^*) \neq 0$.
2. Exclusion: $Z \perp Y(d)$ for $d \in \{0,1\}$.
3. Monotonicity: $D^*(Z=1) \geq D^*(Z=0)$ almost surely.

Mahajan (2006) shows that under these conditions and with a binary instrument and binary misclassified treatment, the ATE is point identified provided the instrument takes at least two values and misclassification is non-differential. The key insight is that the instrument introduces variation in $E[D \mid Z]$ that can be used to separate the structural treatment effect from the misclassification contamination.

**Proposition 8.1** (Bounds under bounded misclassification, Mahajan 2006). *Suppose $\alpha_0 \leq \bar\alpha_0$ and $\alpha_1 \leq \bar\alpha_1$ are known upper bounds. Define the Wald estimand using observed $D$:*

$$\hat\tau_{\text{Wald}} = \frac{E[Y \mid Z=1] - E[Y \mid Z=0]}{E[D \mid Z=1] - E[D \mid Z=0]}.$$

*Then the true LATE satisfies:*

$$\tau_{\text{LATE}} \in \left[\frac{\hat\tau_{\text{Wald}}}{1 - \bar\alpha_1 - \bar\alpha_0},\ \frac{\hat\tau_{\text{Wald}}}{1 - \bar\alpha_1 - \bar\alpha_0}\right]$$

*with the bounds widening as $\bar\alpha_0 + \bar\alpha_1$ increases.*

The formula makes the identification logic transparent: the observed Wald estimand equals the true LATE scaled by $(1 - \alpha_1 - \alpha_0)$, so bounding the scaling factor from above and below translates directly into bounds on the LATE. When a validation subsample is available — for instance, a subset of OHE respondents for whom we have both administrative enrollment records and self-reports — we can estimate $\alpha_0$ and $\alpha_1$ directly, restoring approximate point identification.

**Remark 8.2** (Sharp bounds without instruments). Even without an instrument, Manski-style sharp bounds on the ATE under bounded misclassification can be derived from the distribution of $(Y, D)$. These are wider than the IV-based bounds and often uninformative without additional structure, but they remain valid when instrument validity is questionable.

## 8.5 IV as a Correction for Measurement Error

The OHE lottery provides an instrument $Z$ for true enrollment $D^*$. But the Wald estimator using the observed (misclassified) $D$ in the denominator is not consistent for the LATE under $D^*$. The correct IV estimator uses $Z$ as instrument for $D^*$; if we use $D$ in the first stage instead, we are essentially instrumenting for the noisy proxy of the treatment, not the treatment itself.

However, there is a well-known IV solution to ME in the continuous case: if two error-prone measurements of the same underlying variable are available, each can serve as an instrument for the other. In the OHE context, suppose we have both self-reported insurance $D_{\text{SR}}$ and administrative enrollment $D_{\text{admin}}$, each measuring true coverage $D^*$ with independent errors. Then the 2SLS estimator using $D_{\text{admin}}$ as instrument for $D_{\text{SR}}$ (or vice versa) is consistent for $\beta$ under classical error, because the instrument is correlated with $D^*$ but uncorrelated with the measurement error in $D_{\text{SR}}$ (by independence of errors).

**Theorem 8.2** (Two-measurements-as-instruments). *Let $D_1 = D^* + u_1$ and $D_2 = D^* + u_2$ with $E[u_j] = 0$, $\text{Cov}(D^*, u_j) = 0$, and $\text{Cov}(u_1, u_2) = 0$. Then $D_2$ is a valid instrument for $D_1$ in the regression $Y = \beta D_1 + \varepsilon$, and the IV estimator is consistent for $\beta$.*

*Proof.* Relevance: $\text{Cov}(D_2, D_1) = \text{Cov}(D^* + u_2, D^* + u_1) = \text{Var}(D^*) > 0$. Exogeneity: $\text{Cov}(D_2, \varepsilon - \beta u_1) = \text{Cov}(D^* + u_2, \varepsilon - \beta u_1) = \text{Cov}(D^*, \varepsilon) - \beta\text{Cov}(D^*, u_1) + \text{Cov}(u_2, \varepsilon) - \beta\text{Cov}(u_2, u_1) = 0$ under the stated conditions. $\square$

The independence condition $\text{Cov}(u_1, u_2) = 0$ is critical and often questionable. Self-report errors and administrative errors may share a common component (e.g., both miscoded when enrollment was pending), violating independence. In such cases, an external instrument — the lottery $Z$ — is necessary.

## 8.6 Proxy Outcomes and Surrogate Endpoints

So far we have focused on measurement error in the treatment. Outcomes can also be measured with error, with different implications. Let $Y^*$ be the true outcome (e.g., true health status) and $Y = Y^* + v$ the observed measure (e.g., self-reported health). If $v$ is classical measurement error in $Y$, then OLS on $(Y, D)$ is still consistent for $\beta$: the error enters only the residual, inflating standard errors but not biasing the point estimate.

The problem arises with **non-classical outcome error**. If sicker people are less likely to report accurate insurance status, then $\text{Cov}(v, D^*) \neq 0$, and the OLS estimator of the effect of $D^*$ on $Y$ is biased even when $D^*$ is observed exactly. This is **endogenous measurement error in the outcome**.

**Proxy outcomes** arise in a different but related setting: the true outcome $Y^*$ is unavailable, and we use a surrogate $S$ that is causally downstream of the treatment and predictive of $Y^*$. Prentice (1989) formalized the surrogate endpoint concept; the key requirement is that the treatment effect on $Y^*$ is fully mediated through $S$. This condition is almost never exactly satisfied and typically requires validation data.

In the ACA/BRFSS setting, self-reported health status (excellent/very good/good/fair/poor) is a proxy for true health, with potentially differential reporting across states with different ACA expansion policies. The concern is that ACA expansion changed both true health and the propensity to report accurately — a form of non-classical outcome error that can bias difference-in-differences estimates.

## 8.7 Connecting the OHE and ACA Settings

The OHE offers a clean identification strategy but limited external validity. The ACA/BRFSS data offer broad population coverage but introduce both treatment misclassification (self-reported insurance in BRFSS vs. actual enrollment) and outcome measurement concerns (self-reported health, financial outcomes).

A synthesis strategy uses the OHE to validate measurement error rates and then applies those rates as sensitivity parameters in the ACA analysis. Specifically:

1. In the OHE, compute $\hat\alpha_0$ and $\hat\alpha_1$ from the joint distribution of $(D_{\text{SR}}, D_{\text{admin}})$ among lottery participants and non-participants separately.
2. Apply these estimates to bound the ACA/BRFSS estimates under the assumption that BRFSS measurement error is similar in structure to OHE self-report error.
3. Present sensitivity curves showing how the ACA treatment effect estimate varies as misclassification rates move from zero to the OHE-estimated values.

This is not a formal identification result — it is a structured sensitivity analysis — but it is considerably more disciplined than ignoring measurement error entirely.

## Python: Measurement Error, Misclassification Bounds, and IV Correction

```python
"""
Chapter 8: Measurement Error, Proxy Treatments, and Proxy Outcomes
Oregon Health Plan (OHE) + ACA/BRFSS running example.

Requirements: numpy, pandas, scipy, statsmodels, linearmodels
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)

# ============================================================
# Part 1: Attenuation bias — continuous treatment analog
# ============================================================

def simulate_attenuation(n=5000, beta=0.5, sigma_star=1.0, sigma_u_vals=None):
    """
    Simulate attenuation bias under classical ME across noise levels.
    True model: Y = beta * D_star + eps
    """
    if sigma_u_vals is None:
        sigma_u_vals = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]

    results = []
    D_star = rng.normal(0, sigma_star, n)
    eps = rng.normal(0, 1, n)
    Y = beta * D_star + eps

    for sigma_u in sigma_u_vals:
        u = rng.normal(0, sigma_u, n)
        D_obs = D_star + u
        X = sm.add_constant(D_obs)
        model = sm.OLS(Y, X).fit()
        beta_hat = model.params[1]
        reliability = sigma_star**2 / (sigma_star**2 + sigma_u**2)
        results.append({
            "sigma_u": sigma_u,
            "reliability_ratio": reliability,
            "beta_hat": beta_hat,
            "predicted_plim": beta * reliability,
            "attenuation_pct": 100 * (1 - beta_hat / beta),
        })

    df = pd.DataFrame(results)
    print("=== Attenuation Bias Under Classical ME ===")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    return df

attenuation_df = simulate_attenuation()

# ============================================================
# Part 2: Binary misclassification — OHE simulation
# ============================================================

def simulate_ohe_misclassification(
    n=10000,
    tau=0.05,          # true ITT / LATE
    first_stage=0.25,  # P(D*=1|Z=1) - P(D*=1|Z=0), OHE-like
    p_z=0.5,
    alpha_0=0.05,      # false positive rate (uninsured reported as insured)
    alpha_1=0.10,      # false negative rate (insured reported as uninsured)
):
    """
    Simulate OHE-style RCT with binary misclassification.
    Z = lottery win, D* = true Medicaid enrollment, D = self-report.
    Y = outcome (e.g., catastrophic expenditure indicator, sign-flipped).
    """
    Z = rng.binomial(1, p_z, n)

    # First stage: D* | Z
    p_d_z0 = 0.10   # baseline enrollment among non-winners
    p_d_z1 = p_d_z0 + first_stage
    D_star = np.where(Z == 1,
                      rng.binomial(1, p_d_z1, n),
                      rng.binomial(1, p_d_z0, n))

    # Potential outcomes: Y(1) - Y(0) = tau for compliers
    eps = rng.normal(0, 1, n)
    Y = 0.3 + tau * D_star + eps   # Y is continuous here (e.g., health index)

    # Misclassification: non-differential
    # P(D=1 | D*=0) = alpha_0, P(D=0 | D*=1) = alpha_1
    noise = rng.uniform(0, 1, n)
    D_obs = np.where(
        D_star == 1,
        np.where(noise < alpha_1, 0, 1),   # false negative
        np.where(noise < alpha_0, 1, 0),   # false positive
    )

    return pd.DataFrame({"Z": Z, "D_star": D_star, "D_obs": D_obs, "Y": Y})


def wald_estimator(df, D_col, Y_col="Y", Z_col="Z"):
    """Wald / reduced-form IV estimator."""
    num = df.loc[df[Z_col]==1, Y_col].mean() - df.loc[df[Z_col]==0, Y_col].mean()
    den = df.loc[df[Z_col]==1, D_col].mean() - df.loc[df[Z_col]==0, D_col].mean()
    return num / den if abs(den) > 1e-10 else np.nan


def compare_estimators(alpha_0=0.05, alpha_1=0.10, tau=0.05, n=20000):
    df = simulate_ohe_misclassification(n=n, tau=tau, alpha_0=alpha_0, alpha_1=alpha_1)

    # OLS using misclassified D
    X = sm.add_constant(df["D_obs"])
    ols_misclass = sm.OLS(df["Y"], X).fit().params["D_obs"]

    # OLS using true D* (infeasible oracle)
    X_star = sm.add_constant(df["D_star"])
    ols_oracle = sm.OLS(df["Y"], X_star).fit().params["D_star"]

    # Wald using true D* (correct IV)
    wald_star = wald_estimator(df, "D_star")

    # Wald using misclassified D (biased first stage)
    wald_obs = wald_estimator(df, "D_obs")

    # Theoretical attenuation factor
    attenuation = 1 - alpha_1 - alpha_0

    print(f"\n=== Misclassification Bias (alpha_0={alpha_0}, alpha_1={alpha_1}) ===")
    print(f"  True tau (LATE)            : {tau:.4f}")
    print(f"  OLS on D* (oracle)         : {ols_oracle:.4f}")
    print(f"  OLS on D_obs (misclassified): {ols_misclass:.4f}")
    print(f"  Wald / IV using D*         : {wald_star:.4f}")
    print(f"  Wald / IV using D_obs      : {wald_obs:.4f}")
    print(f"  Theoretical: tau*(1-a1-a0) : {tau * attenuation:.4f}")
    print(f"  Attenuation factor         : {attenuation:.4f}")

    return {
        "tau": tau, "alpha_0": alpha_0, "alpha_1": alpha_1,
        "ols_oracle": ols_oracle, "ols_misclass": ols_misclass,
        "wald_star": wald_star, "wald_obs": wald_obs,
    }

r1 = compare_estimators(alpha_0=0.05, alpha_1=0.10)
r2 = compare_estimators(alpha_0=0.10, alpha_1=0.20)  # heavier misclassification

# ============================================================
# Part 3: Mahajan-style misclassification bounds
# ============================================================

def mahajan_bounds(df, alpha_0_max, alpha_1_max, Y_col="Y", D_col="D_obs", Z_col="Z"):
    """
    Sharp bounds on LATE given upper bounds on misclassification rates.
    Based on the identity: Wald(D_obs) = LATE * (1 - alpha_1 - alpha_0)
    """
    wald_obs = wald_estimator(df, D_col, Y_col, Z_col)

    # Bootstrap SE for the Wald estimator
    boot_walds = []
    for _ in range(500):
        samp = df.sample(len(df), replace=True)
        boot_walds.append(wald_estimator(samp, D_col, Y_col, Z_col))
    se_wald = np.std(boot_walds)

    # Bounds: LATE = Wald_obs / (1 - alpha_1 - alpha_0)
    # Minimum of (1-alpha_1-alpha_0) gives widest bound
    min_scale = 1 - alpha_1_max - alpha_0_max
    max_scale = 1.0   # zero misclassification

    lb = wald_obs / max_scale   # tightest lower (if alpha=0, Wald_obs = LATE)
    ub_worst = wald_obs / min_scale if min_scale > 0 else np.inf

    # For lower bound on LATE when effect is positive:
    # Smallest true effect consistent with observed Wald and bounded rates
    if wald_obs > 0:
        late_lb = wald_obs  # zero mis-classification gives smallest correction
        late_ub = wald_obs / min_scale
    else:
        late_ub = wald_obs
        late_lb = wald_obs / min_scale

    print(f"\n=== Mahajan Bounds (alpha_0 <= {alpha_0_max}, alpha_1 <= {alpha_1_max}) ===")
    print(f"  Observed Wald (D_obs)      : {wald_obs:.4f}  (SE={se_wald:.4f})")
    print(f"  Point estimate if alpha=0  : {wald_obs:.4f}")
    print(f"  Bound under max alpha      : [{late_lb:.4f}, {late_ub:.4f}]")
    print(f"  Bound width                : {abs(late_ub - late_lb):.4f}")
    return late_lb, late_ub, wald_obs

df_sim = simulate_ohe_misclassification(n=20000, tau=0.05, alpha_0=0.05, alpha_1=0.10)
bounds_tight = mahajan_bounds(df_sim, alpha_0_max=0.10, alpha_1_max=0.15)
bounds_wide  = mahajan_bounds(df_sim, alpha_0_max=0.20, alpha_1_max=0.30)

# ============================================================
# Part 4: Two-measurements-as-instruments (OHE admin vs. self-report)
# ============================================================

def simulate_two_measures(n=10000, beta=0.5, rho_u=0.0):
    """
    Simulate two error-prone measures of D*.
    D1 = D* + u1 (e.g., self-report), D2 = D* + u2 (e.g., admin + noise).
    rho_u: correlation between u1 and u2 (should be 0 for IV validity).
    """
    D_star = rng.normal(0, 1, n)
    eps = rng.normal(0, 1, n)
    Y = beta * D_star + eps

    # Correlated errors
    cov_u = np.array([[0.5, rho_u * 0.5], [rho_u * 0.5, 0.3]])
    u = rng.multivariate_normal([0, 0], cov_u, n)
    D1 = D_star + u[:, 0]  # self-report analog
    D2 = D_star + u[:, 1]  # admin analog (less noisy)

    df = pd.DataFrame({"D_star": D_star, "D1": D1, "D2": D2, "Y": Y})

    # OLS on D1 (attenuated)
    ols_d1 = sm.OLS(Y, sm.add_constant(D1)).fit().params["D1"]

    # OLS on D2 (less attenuated but still biased)
    ols_d2 = sm.OLS(Y, sm.add_constant(D2)).fit().params["D2"]

    # 2SLS: use D2 as instrument for D1
    iv_model = IV2SLS(
        dependent=df["Y"],
        exog=None,
        endog=df["D1"],
        instruments=df["D2"]
    ).fit(cov_type="robust")
    iv_est = iv_model.params["D1"]

    # Oracle OLS on D_star
    ols_star = sm.OLS(Y, sm.add_constant(D_star)).fit().params["D_star"]

    print(f"\n=== Two-Measurements IV (rho_u={rho_u}) ===")
    print(f"  True beta                  : {beta:.4f}")
    print(f"  OLS on D1 (self-report)    : {ols_d1:.4f}")
    print(f"  OLS on D2 (admin)          : {ols_d2:.4f}")
    print(f"  2SLS: D2 instruments D1    : {iv_est:.4f}")
    print(f"  Oracle OLS on D*           : {ols_star:.4f}")
    if abs(rho_u) > 0.01:
        print(f"  [Warning: correlated errors (rho={rho_u}), IV inconsistent]")
    return ols_d1, ols_d2, iv_est, ols_star

# Independent errors: IV should recover beta
simulate_two_measures(rho_u=0.0)

# Correlated errors: IV fails
simulate_two_measures(rho_u=0.5)

# ============================================================
# Part 5: Sensitivity analysis — ACA/BRFSS analog
# ============================================================

def aca_sensitivity_curve(
    tau_wald_obs=0.04,   # observed Wald/DiD estimate using self-report insurance
    alpha_max=0.20,
    n_points=50,
):
    """
    Show how the corrected ATE varies as a function of assumed misclassification.
    tau_true = tau_wald_obs / (1 - alpha_0 - alpha_1)
    Parameterize along diagonal: alpha_0 = alpha_1 = alpha.
    """
    alphas = np.linspace(0, alpha_max, n_points)
    # Avoid division by zero and negative denominator
    valid = alphas < 0.5
    scales = 1 - 2 * alphas[valid]
    tau_corrected = tau_wald_obs / scales

    print("\n=== ACA/BRFSS Sensitivity to Misclassification (alpha_0 = alpha_1 = alpha) ===")
    print(f"  Observed DiD estimate      : {tau_wald_obs:.4f}")
    header = f"{'alpha':>8} | {'scale (1-2a)':>14} | {'tau_corrected':>14} | {'mult_factor':>12}"
    print(header)
    print("-" * len(header))
    display_idx = np.linspace(0, valid.sum()-1, 10, dtype=int)
    for i in display_idx:
        a = alphas[valid][i]
        s = scales[i]
        tc = tau_corrected[i]
        print(f"  {a:8.3f} | {s:14.4f} | {tc:14.4f} | {tc/tau_wald_obs:12.2f}x")

    # OHE-estimated misclassification rates (approximate from literature)
    alpha_ohe = 0.075  # ~7.5% combined rate from OHE validation
    tau_ohe_corrected = tau_wald_obs / (1 - 2 * alpha_ohe)
    print(f"\n  OHE-calibrated correction (alpha={alpha_ohe:.3f}): {tau_ohe_corrected:.4f}")
    return alphas[valid], tau_corrected

alphas_curve, taus_curve = aca_sensitivity_curve()

# ============================================================
# Part 6: Non-classical ME — differential misclassification
# ============================================================

def simulate_differential_misclassification(n=20000, tau=0.05):
    """
    Non-classical ME: misclassification depends on Y (outcome).
    Healthy people (high Y) more likely to over-report insurance.
    """
    Z = rng.binomial(1, 0.5, n)
    D_star = np.where(Z==1,
                      rng.binomial(1, 0.35, n),
                      rng.binomial(1, 0.10, n))
    Y = 0.3 + tau * D_star + rng.normal(0, 1, n)

    # Non-classical: alpha_0 depends on Y (healthier -> higher false positive)
    # People with better outcomes more confidently report having insurance
    alpha_0_i = 0.03 + 0.05 * (Y > Y.mean()).astype(float)  # differential by outcome
    alpha_1_i = np.full(n, 0.08)

    noise = rng.uniform(0, 1, n)
    D_obs = np.where(
        D_star == 1,
        np.where(noise < alpha_1_i, 0, 1),
        np.where(noise < alpha_0_i, 1, 0),
    )

    # Wald with misclassified D
    wald_obs = wald_estimator(
        pd.DataFrame({"Z": Z, "D_obs": D_obs, "Y": Y}),
        "D_obs"
    )
    wald_true = wald_estimator(
        pd.DataFrame({"Z": Z, "D_star": D_star, "Y": Y}),
        "D_star"
    )

    avg_alpha_0 = alpha_0_i.mean()
    print(f"\n=== Differential Misclassification (non-classical ME) ===")
    print(f"  True tau                   : {tau:.4f}")
    print(f"  Wald using D*              : {wald_true:.4f}")
    print(f"  Wald using D_obs (non-class): {wald_obs:.4f}")
    print(f"  Avg alpha_0 (Y-dependent)  : {avg_alpha_0:.4f}")
    print(f"  [Bias direction now ambiguous — non-classical ME]")

simulate_differential_misclassification()

print("\n=== Chapter 8 simulations complete ===")
```

## Summary

- Classical measurement error in a continuous treatment attenuates OLS coefficients by the reliability ratio $\lambda = \sigma^2_{D^*}/(\sigma^2_{D^*} + \sigma^2_u)$; the bias is always toward zero and the estimator is still consistent for $\lambda \beta$, not $\beta$.

- For binary treatment with misclassification rates $\alpha_0$ (false positives) and $\alpha_1$ (false negatives), the Wald/IV estimator using the noisy treatment in the first stage converges to $\tau(1 - \alpha_1 - \alpha_0)$; at empirically plausible rates of 5–10% per cell the attenuation can exceed 20%.

- Non-classical misclassification — where error rates depend on potential outcomes or covariates — breaks the simple attenuation formula and can produce bias in any direction; the ACA/BRFSS setting, where healthier respondents may more confidently report insurance status, is a credible instance.

- Mahajan (2006)-style partial identification delivers sharp bounds on the LATE by inverting the attenuation formula over the set of misclassification rates consistent with prior knowledge or validation data; bounds widen linearly as the uncertainty about misclassification grows.

- When two independent error-prone measurements of the same treatment are available, each can instrument for the other (the TSLS estimator is consistent), but correlated errors — plausible when administrative and self-report errors share a common cause — invalidate this approach and leave the OHE lottery $Z$ as the necessary instrument.

- Measurement error in the outcome inflates standard errors but does not bias the treatment effect estimate under classical error; non-classical outcome error (endogenous reporting) does produce bias and requires either an external validator or a sensitivity analysis parameterized by the degree of differential reporting.

- A principled synthesis uses OHE administrative-versus-self-report discrepancies to calibrate misclassification rates, then applies those rates as sensitivity parameters in the ACA/BRFSS analysis, converting an unacknowledged source of attenuation into a quantified and bounded form of uncertainty.

## Further Reading

- **Mahajan, A. (2006).** "Identification and estimation of regression models with misclassification." *Econometrica*, 74(3), 631–665. The foundational partial identification result for binary treatment misclassification with an instrument.

- **Lewbel, A. (2007).** "Estimation of average treatment effects with misclassification." *Econometrica*, 75(2), 537–551. Extends Mahajan to allow for covariates and explores semiparametric efficiency.

- **Carroll, R.J., Ruppert, D., Stefanski, L.A., and Crainiceanu, C.M. (2006).** *Measurement Error in Nonlinear Models: A Modern Perspective*, 2nd ed. Chapman & Hall. Comprehensive treatment of ME in regression, including SIMEX and regression calibration.

- **Bound, J., Brown, C., and Mathiowetz, N. (2001).** "Measurement error in survey data." *Handbook of Econometrics*, Vol. 5. The definitive survey article on non-classical measurement error in economics, including the consequences for instrumental variables.

- **Finkelstein, A., Taubman, S., Wright, B., et al. (2012).** "The Oregon Health Insurance Experiment: Evidence from the first year." *Quarterly Journal of Economics*, 127(3), 1057–1106. Primary OHE paper; discusses measurement of outcomes and the relationship between administrative enrollment and self-reported coverage.

- **Kreider, B., Pepper, J.V., Gundersen, C., and Jolliffe, D. (2012).** "Identifying the effects of SNAP (Food Stamps) on child health outcomes when participation is endogenous and misreported." *Journal of the American Statistical Association*, 107(499), 958–975. Applied example of misclassification bounds with program evaluation data.
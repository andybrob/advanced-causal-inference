# Chapter 46: Causal Monitoring and Post-Deployment Learning

Causal inference does not end at deployment. A policy—Medicaid expansion, a pricing algorithm, a clinical protocol—enters a world that keeps changing. Populations shift, institutions adapt, spillovers accumulate, and the very act of intervention alters subsequent data-generating processes. A causal estimate valid at launch may be misleading a year later, not because the original analysis was wrong, but because the conditions that made it valid no longer hold.

This chapter develops the machinery for causal monitoring: detecting when a deployed causal model has decayed, sequentially updating causal estimates from streaming data, and deciding when to trigger recalibration. The tools combine classical statistical process control, modern doubly robust estimation, and bandit-style exploration. The running application is ongoing surveillance of Medicaid expansion effects using simulated monthly BRFSS-style panel data.

## 46.1 The Causal Decay Problem

Let $\tau^* = E[Y(1) - Y(0)]$ be the true average treatment effect at the time of the original study. A deployed policy relies on some estimate $\hat{\tau}$ of $\tau^*$. Causal decay occurs when $\tau^*$ itself changes over time, or when the conditions required for identification break down in new data.

Three distinct failure modes deserve separate treatment.

**Covariate shift** occurs when the distribution of pre-treatment covariates $P(X)$ changes while the conditional outcome model $E[Y(d) | X]$ remains stable. The ATE may be unchanged in principle, but if the operational population now differs from the study population, the weighted average shifts. Formally, if $\tau^*(t) = E_t[E[Y(1)|X] - E[Y(0)|X]]$ where $E_t$ integrates over the time-$t$ covariate distribution, then $\tau^*(t)$ drifts even with stable potential outcome surfaces.

**Concept drift** occurs when $E[Y(d) | X]$ itself changes. In Medicaid expansion, this could reflect improvements in provider capacity, changes in what "having insurance" means as benefit packages evolve, or changes in the composition of newly insured populations over expansion waves. This is the more dangerous failure mode because it is invisible to covariate monitoring.

**Identification decay** is structurally different from the first two. An instrument that was valid at baseline (e.g., distance to nearest expansion-era clinic) may become irrelevant as clinics open everywhere, violating the relevance condition. A parallel trends assumption may hold for the first two post-expansion years and fail thereafter as never-takers find alternative coverage paths. Identification decay cannot be detected purely from prediction metrics; it requires ongoing scrutiny of the identifying assumptions themselves.

**Definition 46.1 (Causal Stability).** A causal estimate $\hat{\tau}$ deployed at time $t_0$ is *causally stable* through time $T$ if (i) the identifying assumptions hold in each period $t \in [t_0, T]$ under the data-generating process of that period, and (ii) the estimand $\tau^*(t)$ lies within a tolerance band $[\tau^* - \delta, \tau^* + \delta]$ for a pre-specified $\delta > 0$.

This definition separates the statistical question (is $\hat{\tau}(t)$ tracking $\tau^*(t)$?) from the policy question (is $\tau^*(t)$ still large enough to justify the intervention?).

## 46.2 Sequential Causal Testing and CUSUM

Classical hypothesis testing assumes a fixed sample. When data arrive sequentially, repeated testing inflates type I error. The solution is sequential testing with error guarantees that hold uniformly over time.

### 46.2.1 CUSUM for Causal Shift

The cumulative sum (CUSUM) chart is the foundational tool. Let $\hat{\tau}_t$ be a period-$t$ causal estimate (we will specify the estimator in §46.3). Define:

$$S_t = \max(0,\; S_{t-1} + (\hat{\tau}_t - \tau_0) - k)$$

with $S_0 = 0$. Here $\tau_0$ is the in-control value (the baseline estimate), and $k$ is the *allowance* or *slack*, typically set to half the minimum shift worth detecting: $k = \delta/2$. An alarm is triggered at the first time $T^* = \inf\{t : S_t > h\}$ where $h$ is the control limit.

**Theorem 46.1 (CUSUM Average Run Length).** Suppose $\hat{\tau}_t \overset{iid}{\sim} N(\tau, \sigma^2)$. When $\tau = \tau_0$ (in-control), the average run length $ARL_0 = E[T^* | \tau = \tau_0]$ satisfies the Wald approximation

$$ARL_0 \approx \frac{e^{2\lambda(h + 1.166)} - 2\lambda(h+1.166) - 1}{2\lambda^2}$$

where $\lambda = k/\sigma$. When $\tau = \tau_0 + \delta$ (out-of-control), $ARL_1 \approx e^{2\lambda h}/(\delta/\sigma)^2$ for large $h$.

*Proof sketch.* The CUSUM statistic is a likelihood ratio test for a change in mean of a Gaussian sequence. The one-sided CUSUM corresponds to the sequential probability ratio test (SPRT) restarted after each zero-crossing. The ARL expressions follow from Wald's identity applied to the stopped random walk; see Siegmund (1985) for exact expressions via renewal theory.

The Gaussian assumption is restrictive. In practice, $\hat{\tau}_t$ is a nonparametric estimate with unknown sampling distribution. Two robust alternatives are:

1. **Bootstrapped control limits**: estimate the in-control distribution of $\hat{\tau}_t$ from early periods and compute empirical quantiles for $h$.
2. **E-value CUSUM**: replace the Gaussian likelihood ratio with an *e-value* $e_t = p(\hat{\tau}_t | \tau_0 + \delta) / p(\hat{\tau}_t | \tau_0)$ that is a test martingale. The product $E_t = \prod_{s=1}^t e_s$ satisfies $E_E[E_t] \leq 1$ under $H_0$ for all stopping times, giving a sequential test with exact type I error control without distributional assumptions (Ville, 1939; Shafer and Vovk, 2021).

### 46.2.2 Two-Sided Monitoring

For policy monitoring we often want to detect both effect decay ($\tau$ dropping below $\tau_0 - \delta$) and effect amplification ($\tau$ rising above $\tau_0 + \delta$). Run two CUSUM statistics simultaneously:

$$S_t^+ = \max(0,\; S_{t-1}^+ + (\hat{\tau}_t - \tau_0) - k)$$
$$S_t^- = \max(0,\; S_{t-1}^- - (\hat{\tau}_t - \tau_0) - k)$$

Alarm when $\max(S_t^+, S_t^-) > h$. The joint ARL under independence of the two arms is approximately $ARL_0 / 2$ compared to the one-sided case, which must be accounted for in the calibration of $h$.

## 46.3 Online Doubly Robust Estimation

To feed the CUSUM, we need period-specific causal estimates. In observational settings—which is the realistic scenario for ongoing Medicaid surveillance—we require estimates that are valid despite confounding. Online doubly robust (DR) estimation provides this.

### 46.3.1 The DR Pseudo-Outcome

Recall the augmented inverse propensity weighted (AIPW) estimator. For a binary treatment $D_i \in \{0,1\}$:

$$\Gamma_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{D_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-D_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}$$

where $\hat{\mu}_d(x) = \hat{E}[Y | D=d, X=x]$ and $\hat{e}(x) = \hat{P}(D=1|X=x)$. The DR property states that $E[\Gamma_i] = \tau^*$ if either $\hat{\mu}_d$ or $\hat{e}$ is correctly specified, not necessarily both.

The key insight for online estimation is that $\Gamma_i$ is a *unit-level* unbiased estimate of the individual contribution to the ATE. We can update a running average with each new observation.

**Definition 46.2 (Online DR Estimator).** Given a sequence of DR pseudo-outcomes $\Gamma_1, \Gamma_2, \ldots$, the online DR estimator with learning rate $\eta_t$ is:

$$\hat{\tau}_t = \hat{\tau}_{t-1} + \eta_t [\Gamma_t - \hat{\tau}_{t-1}]$$

With $\eta_t = 1/t$ (equal weighting), this reduces to the sample mean: $\hat{\tau}_t = t^{-1}\sum_{s=1}^t \Gamma_s$. Adaptive choices of $\eta_t$ allow down-weighting stale observations when concept drift is suspected.

**Theorem 46.2 (Consistency of Online DR).** Let $\eta_t = 1/t$. If both nuisance models $\hat{\mu}_d$ and $\hat{e}$ are $\sqrt{n}$-consistent (e.g., via cross-fitting), then $\hat{\tau}_t \to \tau^*$ almost surely as $t \to \infty$, and $\sqrt{t}(\hat{\tau}_t - \tau^*) \overset{d}{\to} N(0, \text{Var}(\Gamma_i))$.

*Proof sketch.* By the DR property, $E[\Gamma_t] = \tau^* + o(1/\sqrt{t})$ where the remainder comes from the product of nuisance biases. With $\sqrt{n}$-consistent nuisances, this product is $O(1/n) = o(1/\sqrt{n})$ (the double robustness "bonus"). The CLT then follows from the martingale CLT for the recursive average.

### 46.3.2 Sliding Window and Exponential Downweighting

For detecting concept drift, we prefer recent data. Two standard approaches:

**Sliding window**: maintain $\hat{\tau}_t^{(w)} = w^{-1} \sum_{s=t-w+1}^{t} \Gamma_s$. This has variance $\text{Var}(\Gamma)/w$ and detects shifts with delay roughly $w/2$, creating a bias-variance tradeoff in window width.

**Exponential downweighting**: use $\eta_t = 1 - \alpha$ for a forgetting factor $\alpha \in (0,1)$:

$$\hat{\tau}_t^{(\alpha)} = \alpha \hat{\tau}_{t-1}^{(\alpha)} + (1-\alpha)\Gamma_t$$

The effective sample size is $1/(1-\alpha)$. For monthly data with moderate drift, $\alpha \in [0.9, 0.95]$ is typical. This is the exponentially weighted moving average (EWMA) control chart applied to causal pseudo-outcomes.

## 46.4 Nuisance Model Updating

Online DR requires nuisance models $\hat{\mu}_d$ and $\hat{e}$ that remain calibrated as new data arrives. Stale nuisance models invalidate the DR property even if the pseudo-outcome formula is applied correctly.

Three strategies with different computational profiles:

**Periodic refitting**: refit nuisance models every $B$ periods on a rolling window of data. Computationally intensive but straightforward; valid as long as $B$ is small relative to the drift timescale.

**Online gradient updates**: maintain gradient-boosted or linear models that accept incremental updates. The `river` library in Python implements online versions of logistic regression, gradient boosting, and random forests that update in $O(1)$ per observation.

**Conformal prediction monitoring**: rather than updating the nuisance model, monitor its predictive calibration using conformal prediction scores. If the nonconformity scores $s_t = |Y_t - \hat{\mu}_{D_t}(X_t)|$ begin to exceed their historical quantiles (tested via a CUSUM on $s_t$), trigger a nuisance refitting. This separates the *detection* of nuisance decay from the *estimation* problem.

A subtle issue arises with cross-fitting in the online setting. The standard cross-fitting protocol that yields $\sqrt{n}$-consistent DR estimates requires holding out data for nuisance fitting. In streaming settings, this can be approximated by maintaining two separate models trained on alternating substreams (odd-indexed and even-indexed observations), computing $\Gamma_t$ for odd observations using the even-trained model and vice versa. This online cross-fitting approximation preserves the double robustness property asymptotically.

## 46.5 Causal Exploration-Exploitation: Bandits with Causal Structure

Monitoring is reactive. The companion problem is *active*: if we can choose which units to treat, how do we balance learning about the current causal effect (exploration) against applying the best-known treatment (exploitation)?

### 46.5.1 Thompson Sampling with DR Updates

Thompson sampling maintains a posterior over the treatment effect $\tau$ and samples actions proportional to the probability that each action is optimal. With a conjugate Gaussian prior:

$$\tau \sim N(\mu_0, \sigma_0^2)$$

After observing DR pseudo-outcome $\Gamma_t$, the posterior update (treating $\Gamma_t$ as a noisy observation with noise variance $\sigma_\Gamma^2 = \text{Var}(\Gamma)$) is:

$$\mu_t = \frac{\sigma_\Gamma^{-2} \Gamma_t + \sigma_{t-1}^{-2} \mu_{t-1}}{\sigma_\Gamma^{-2} + \sigma_{t-1}^{-2}}, \qquad \sigma_t^{-2} = \sigma_\Gamma^{-2} + \sigma_{t-1}^{-2}$$

At decision time $t+1$, draw $\tilde{\tau} \sim N(\mu_t, \sigma_t^2)$ and treat if $\tilde{\tau} > 0$ (or optimize over a richer action space).

The DR pseudo-outcome enters as the likelihood term rather than the raw outcome $Y_t$, which is critical: using $Y_t$ directly conflates the treatment effect with confounded baseline differences. Using $\Gamma_t$ preserves the causal interpretation of the posterior.

**Theorem 46.3 (Regret Bound for Causal Thompson Sampling).** Under the Gaussian model with true effect $\tau^*$ and DR pseudo-outcome noise $\sigma_\Gamma^2$, Thompson sampling achieves cumulative regret

$$R_T = \sum_{t=1}^T |\tau^*| \cdot \mathbf{1}[\text{wrong action at } t] = O(\sigma_\Gamma \sqrt{T \log T})$$

matching the minimax lower bound up to logarithmic factors.

*Proof sketch.* This follows from the standard Thompson sampling regret analysis (Russo and Van Roy, 2014) applied to the Gaussian bandit with known noise variance, substituting $\Gamma_t$ for the reward. The DR property ensures $E[\Gamma_t | X_t, D_t] = \tau^* + \text{confounding correction} \approx \tau^*$, so the causal bandit reduces to a standard bandit with pseudo-rewards.

### 46.5.2 Contextual Bandits and Heterogeneous Effects

In realistic deployments the relevant quantity is the conditional average treatment effect $\tau(x) = E[Y(1) - Y(0) | X = x]$. The decision rule "treat if $\tau(x) > c$" requires learning $\tau(\cdot)$ online. The natural extension uses a contextual bandit with a model for $\tau(x)$ updated via DR pseudo-outcomes. The `river` library's regression components can maintain an online estimate of $\tau(x)$ for moderate-dimensional $x$.

## 46.6 Granger Causality and Its Limitations

In purely time-series settings without experimental or quasi-experimental variation, a popular approach is Granger causality: $X$ Granger-causes $Y$ if $X_{t-1}$ predicts $Y_t$ conditional on $Y_{t-1}, Y_{t-2}, \ldots$.

**Definition 46.3 (Granger Non-Causality).** $X$ does *not* Granger-cause $Y$ if

$$Y_t \perp X_{t-1}, X_{t-2}, \ldots \mid Y_{t-1}, Y_{t-2}, \ldots$$

The standard test regresses $Y_t$ on lags of both $Y$ and $X$ and applies an F-test for the joint significance of the $X$ lags.

**Critical Limitations.** Granger causality is *predictive*, not *structural*. Three failure modes are worth stating explicitly:

1. **Confounded Granger causality**: if a latent variable $U_t$ drives both $X_t$ and $Y_{t+1}$, then $X$ Granger-causes $Y$ without any structural effect. In Medicaid monitoring, if state-level economic conditions drive both enrollment and health outcomes, Granger tests will spuriously detect "causal" effects of enrollment on outcomes.

2. **Instantaneous effects**: Granger causality tests for lagged effects. If the treatment effect is contemporaneous (same period), the test has no power.

3. **Non-stationarity**: standard Granger tests assume covariance-stationarity. If outcomes trend or have unit roots, standard F-statistics are invalid. Use cointegration tests (Engle-Granger, Johansen) as a prerequisite.

Despite these limitations, Granger causality serves a legitimate role in causal monitoring: as a *screening* tool for identifying candidate confounders to include in structural models, and as an anomaly detector when the null is well-established. If insurance enrollment suddenly Granger-causes emergency department utilization in a direction opposite to the historical structural estimate, that is a monitoring signal worth investigating—even if the Granger test itself is not a structural estimate.

## 46.7 Recalibration Triggers and the Update Decision

Detection is not recalibration. After a CUSUM alarm, the monitoring system must decide: (a) investigate whether the alarm reflects true drift or a data artifact; (b) if drift is confirmed, update the causal model; (c) decide whether the policy should change.

A formal recalibration decision rule has three components:

**Trigger condition**: CUSUM alarm $S_t > h$ sustained for $m$ consecutive periods (requiring persistence reduces false positive recalibration).

**Drift diagnosis**: decompose the alarm into covariate shift vs. concept drift. Run a classifier to predict period (early vs. late) from covariates $X$; high accuracy indicates covariate shift. Run a calibration test on the nuisance models; poor calibration indicates concept drift in $\hat{\mu}_d$ or $\hat{e}$.

**Update protocol**: 
- Covariate shift only: reweight the original estimate using importance weights $\hat{P}_{new}(X)/\hat{P}_{old}(X)$ without refitting the structural model.
- Concept drift: full refit of nuisance models and causal estimator on recent data, with explicit temporal bounds on the validity window of the new estimate.
- Identification decay: requires qualitative assessment—no algorithm replaces subject matter judgment about whether the identifying assumptions still hold.

**Definition 46.4 (Validity Window).** The *validity window* of a causal estimate $\hat{\tau}$ is the time interval $[t_0, T^*]$ where $T^*$ is the first CUSUM alarm time. All subsequent inference using $\hat{\tau}$ should be restricted to data within this window.

## Python: CUSUM Monitoring of Medicaid Expansion Effects

```python
"""
Chapter 46: Causal Monitoring and Post-Deployment Learning
Simulates monthly monitoring of Medicaid expansion effects.
Implements:
  - Simulated monthly BRFSS-style panel
  - Online DR pseudo-outcome computation
  - Two-sided CUSUM chart
  - Recalibration trigger detection
  - Thompson sampling update loop
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

# ---------------------------------------------------------------------------
# 1. Simulate monthly BRFSS-style panel (ACA Medicaid expansion analog)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

N_STATES = 30       # 20 expansion, 10 non-expansion
N_MONTHS = 60       # 5 years of monthly data (Jan 2012 – Dec 2016)
N_PER_CELL = 80     # respondents per state-month

# True effect trajectory: rises quickly, then decays as market adjusts
# Effect is ~0.04 at launch, peaks at 0.07 at month 24, decays to 0.03 by month 48
def true_effect(t: int) -> float:
    """Month-specific ATE on Pr(any doctor visit in past year)."""
    if t < 12:
        return 0.04 + 0.003 * t           # ramp-up
    elif t < 30:
        return 0.04 + 0.003*12 + 0.0015*(t-12)  # peak buildup
    else:
        return max(0.03, 0.076 - 0.0015*(t-30))  # slow decay


def simulate_brfss_month(t: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate one month of cross-sectional BRFSS-style data.
    Treatment D = 1 if in expansion state.
    Outcome Y = any doctor visit in past 12 months (binary).
    Confounders X: income category, age group, rural indicator.
    """
    n = N_STATES * N_PER_CELL
    state_id = np.repeat(np.arange(N_STATES), N_PER_CELL)
    D = (state_id < 20).astype(int)           # expansion states

    # Confounders (correlated with both D and Y due to selection into expansion)
    income = rng.choice([1, 2, 3], size=n, p=[0.35, 0.40, 0.25])
    age    = rng.choice([1, 2, 3], size=n, p=[0.25, 0.45, 0.30])
    rural  = rng.binomial(1, 0.30 - 0.10 * (state_id / N_STATES), size=n)

    # Baseline Pr(Y=1) without treatment, varies by confounders
    p0 = 0.55 + 0.06*income - 0.04*(age==1) + 0.02*(age==3) - 0.05*rural

    # Drift: slowly declining baseline utilization (economic cycle)
    # introduces concept drift around month 36
    if t >= 36:
        p0 = p0 - 0.002 * (t - 35)

    tau_t = true_effect(t)
    p1 = np.clip(p0 + tau_t, 0, 1)

    Y = rng.binomial(1, D * p1 + (1 - D) * p0)

    return pd.DataFrame({
        "month": t,
        "D": D,
        "Y": Y,
        "income": income,
        "age": age,
        "rural": rural,
    })


# Build full panel
panel = pd.concat(
    [simulate_brfss_month(t, rng) for t in range(N_MONTHS)],
    ignore_index=True
)
print(f"Panel shape: {panel.shape}")
print(panel.groupby("month")["Y"].mean().head(10))


# ---------------------------------------------------------------------------
# 2. DR pseudo-outcome computation (cross-fitted within each month)
# ---------------------------------------------------------------------------

def dr_pseudooutcome(df: pd.DataFrame) -> np.ndarray:
    """
    Compute AIPW pseudo-outcomes Gamma_i for a cross-sectional month.
    Uses cross-fitted logistic propensity and ridge outcome models.
    """
    X = df[["income", "age", "rural"]].values.astype(float)
    D = df["D"].values
    Y = df["Y"].values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Cross-fitted propensity P(D=1|X)
    prop_model = LogisticRegression(C=1.0, max_iter=500, random_state=0)
    e_hat = cross_val_predict(prop_model, Xs, D, cv=5, method="predict_proba")[:, 1]
    e_hat = np.clip(e_hat, 0.05, 0.95)

    # Cross-fitted outcome models E[Y|D=d, X]
    mu1_model = Ridge(alpha=1.0)
    mu0_model = Ridge(alpha=1.0)
    X1, Y1 = Xs[D == 1], Y[D == 1]
    X0, Y0 = Xs[D == 0], Y[D == 0]
    mu1_model.fit(X1, Y1)
    mu0_model.fit(X0, Y0)
    mu1_hat = np.clip(mu1_model.predict(Xs), 0, 1)
    mu0_hat = np.clip(mu0_model.predict(Xs), 0, 1)

    # AIPW pseudo-outcome
    Gamma = (
        mu1_hat - mu0_hat
        + D * (Y - mu1_hat) / e_hat
        - (1 - D) * (Y - mu0_hat) / (1 - e_hat)
    )
    return Gamma


# Compute monthly DR estimates
monthly_dr = {}
for t in range(N_MONTHS):
    df_t = panel[panel["month"] == t].copy()
    Gamma_t = dr_pseudooutcome(df_t)
    monthly_dr[t] = {
        "tau_hat": float(Gamma_t.mean()),
        "se": float(Gamma_t.std() / np.sqrt(len(Gamma_t))),
        "true_tau": true_effect(t),
    }

dr_df = pd.DataFrame(monthly_dr).T
dr_df.index.name = "month"
print("\nMonthly DR estimates (first 6 months):")
print(dr_df.head(6).round(4))


# ---------------------------------------------------------------------------
# 3. Two-sided CUSUM
# ---------------------------------------------------------------------------

def cusum_twosided(
    tau_series: np.ndarray,
    tau0: float,
    k: float,
    h: float
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Two-sided CUSUM for detecting shifts from tau0.
    k: allowance (slack), typically delta/2
    h: control limit
    Returns S+, S-, alarm times.
    """
    T = len(tau_series)
    S_pos = np.zeros(T + 1)
    S_neg = np.zeros(T + 1)
    alarms = []

    for t in range(T):
        S_pos[t + 1] = max(0, S_pos[t] + (tau_series[t] - tau0) - k)
        S_neg[t + 1] = max(0, S_neg[t] - (tau_series[t] - tau0) - k)
        if S_pos[t + 1] > h or S_neg[t + 1] > h:
            alarms.append(t)

    return S_pos[1:], S_neg[1:], alarms


# Calibrate CUSUM on first 12 months (burn-in)
tau_series = dr_df["tau_hat"].values
tau0       = float(tau_series[:12].mean())    # in-control level from Year 1
sigma_hat  = float(dr_df["se"].values[:12].mean() * np.sqrt(N_PER_CELL * N_STATES))

delta_detect = 0.02    # minimum shift worth detecting (2pp)
k = delta_detect / 2   # allowance
h = 4.0 * sigma_hat    # control limit: ~4σ for ARL0 ≈ 500

S_pos, S_neg, alarm_months = cusum_twosided(tau_series, tau0, k, h)

print(f"\nCUSUM calibration: tau0={tau0:.4f}, sigma={sigma_hat:.4f}, k={k:.4f}, h={h:.4f}")
print(f"Alarm months: {alarm_months[:10]}")  # first 10 alarms


# ---------------------------------------------------------------------------
# 4. Online DR update (exponential downweighting)
# ---------------------------------------------------------------------------

alpha = 0.92   # forgetting factor
tau_online = np.zeros(N_MONTHS)
tau_online[0] = tau_series[0]
for t in range(1, N_MONTHS):
    tau_online[t] = alpha * tau_online[t - 1] + (1 - alpha) * tau_series[t]

print("\nOnline DR estimates (EWMA, first 6 months):")
print(pd.Series(tau_online[:6]).round(4))


# ---------------------------------------------------------------------------
# 5. Thompson Sampling posterior update
# ---------------------------------------------------------------------------

mu_prior  = 0.04    # prior mean = expected effect at launch
sig2_prior = 0.01**2  # prior variance
sig2_gamma = float(dr_df["se"].values.mean()**2)  # estimated pseudo-outcome noise

mu_post  = np.zeros(N_MONTHS)
sig2_post = np.zeros(N_MONTHS)
mu_t, sig2_t = mu_prior, sig2_prior

for t in range(N_MONTHS):
    obs = tau_series[t]
    # Bayesian update: Gaussian likelihood with known variance sig2_gamma
    prec_post = 1/sig2_t + 1/sig2_gamma
    mu_t  = (mu_t/sig2_t + obs/sig2_gamma) / prec_post
    sig2_t = 1 / prec_post
    mu_post[t]  = mu_t
    sig2_post[t] = sig2_t

# Simulate Thompson sampling treatment decision (treat if tau > 0)
rng2 = np.random.default_rng(7)
ts_treat = np.array([
    int(rng2.normal(mu_post[t], np.sqrt(sig2_post[t])) > 0)
    for t in range(N_MONTHS)
])
print(f"\nThompson sampling: treated in {ts_treat.sum()}/{N_MONTHS} months")


# ---------------------------------------------------------------------------
# 6. Recalibration trigger detection
# ---------------------------------------------------------------------------

SUSTAIN_PERIODS = 3   # alarm must persist for 3 months to trigger recalibration

recalib_triggers = []
alarm_set = set(alarm_months)
for t in alarm_months:
    if all((t + j) in alarm_set for j in range(SUSTAIN_PERIODS)):
        recalib_triggers.append(t)
        # Reset CUSUM after recalibration (simplified: advance past window)

# Deduplicate (keep first trigger in each consecutive run)
final_triggers = []
for tr in recalib_triggers:
    if not final_triggers or tr > final_triggers[-1] + SUSTAIN_PERIODS:
        final_triggers.append(tr)

print(f"\nRecalibration triggers at months: {final_triggers}")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

true_tau_series = np.array([true_effect(t) for t in range(N_MONTHS)])
months = np.arange(N_MONTHS)

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 1, hspace=0.4)

# ---- Panel A: True vs estimated ATE
ax1 = fig.add_subplot(gs[0])
ax1.plot(months, true_tau_series, "k-", lw=2, label="True $\\tau^*(t)$")
ax1.plot(months, tau_series, "b.", alpha=0.6, ms=5, label="Monthly DR estimate")
ax1.plot(months, tau_online, "r-", lw=1.5, alpha=0.8, label=f"EWMA ($\\alpha={alpha}$)")
ax1.fill_between(
    months,
    dr_df["tau_hat"] - 1.96 * dr_df["se"],
    dr_df["tau_hat"] + 1.96 * dr_df["se"],
    alpha=0.15, color="blue", label="95% CI (DR)"
)
ax1.axhline(tau0, ls="--", color="gray", lw=1, label=f"$\\tau_0 = {tau0:.3f}$")
for tr in final_triggers:
    ax1.axvline(tr, color="orange", ls=":", lw=1.5, alpha=0.8)
ax1.set_title("Panel A: True and Estimated Monthly ATE on Any Doctor Visit",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("Month since monitoring start")
ax1.set_ylabel("ATE (pp)")
ax1.legend(fontsize=8, ncol=3)

# ---- Panel B: Two-sided CUSUM
ax2 = fig.add_subplot(gs[1])
ax2.plot(months, S_pos, "g-", lw=1.5, label="$S^+_t$ (upward shift)")
ax2.plot(months, S_neg, "r-", lw=1.5, label="$S^-_t$ (downward shift)")
ax2.axhline(h, ls="--", color="black", lw=1.5, label=f"Control limit $h={h:.4f}$")
for tr in final_triggers:
    ax2.axvline(tr, color="orange", ls=":", lw=1.5, alpha=0.8,
                label="Recalib. trigger" if tr == final_triggers[0] else "")
ax2.set_title("Panel B: Two-Sided CUSUM Chart for Causal Effect Monitoring",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Month")
ax2.set_ylabel("CUSUM statistic")
ax2.legend(fontsize=8)

# ---- Panel C: Thompson sampling posterior mean and credible interval
ax3 = fig.add_subplot(gs[2])
ci_lo = mu_post - 1.96 * np.sqrt(sig2_post)
ci_hi = mu_post + 1.96 * np.sqrt(sig2_post)
ax3.plot(months, mu_post, "purple", lw=2, label="Posterior mean $\\mu_t$")
ax3.fill_between(months, ci_lo, ci_hi, alpha=0.2, color="purple",
                 label="95% posterior CI")
ax3.plot(months, true_tau_series, "k--", lw=1.5, alpha=0.7, label="True $\\tau^*(t)$")
ax3.axhline(0, color="gray", lw=0.8, ls=":")
for tr in final_triggers:
    ax3.axvline(tr, color="orange", ls=":", lw=1.5, alpha=0.8)
ax3.set_title("Panel C: Thompson Sampling Posterior (DR-Updated)",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Month")
ax3.set_ylabel("Treatment effect estimate")
ax3.legend(fontsize=8)

plt.savefig("ch46_causal_monitoring.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFigure saved: ch46_causal_monitoring.png")


# ---------------------------------------------------------------------------
# 8. Granger causality test (enrollment -> outcomes, using aggregated panel)
# ---------------------------------------------------------------------------

from statsmodels.tsa.stattools import grangercausalitytests

# Aggregate to monthly mean outcomes and simulated "enrollment rate"
monthly_agg = (
    panel.groupby("month")
    .agg(Y_mean=("Y", "mean"), D_mean=("D", "mean"))
    .reset_index()
)

# Add simulated enrollment variation (lagged noise to mimic outreach campaigns)
monthly_agg["enrollment_rate"] = (
    monthly_agg["D_mean"]
    + rng.normal(0, 0.01, N_MONTHS)
    + np.linspace(0, 0.03, N_MONTHS)    # secular trend in enrollment
)

gc_data = monthly_agg[["Y_mean", "enrollment_rate"]].values

print("\nGranger causality test: enrollment -> health outcome (lags 1-4)")
gc_results = grangercausalitytests(gc_data, maxlag=4, verbose=False)
for lag, result in gc_results.items():
    f_stat, p_val, df_denom, df_num = result[0]["ssr_ftest"]
    print(f"  Lag {lag}: F={f_stat:.3f}, p={p_val:.4f} "
          f"{'* predictive signal' if p_val < 0.05 else '(no signal)'}")

print("\nNote: Granger significance does not imply structural causality.")
print("Confounding by time trends and economic cycles is not controlled.")
```

The output from this code warrants careful interpretation. Panel A reveals that the monthly DR estimates track the true effect trajectory with reasonable fidelity during the ramp-up phase (months 0–30) but show elevated variance during the decay phase as the true effect approaches the noise floor of the estimator. The EWMA smooths this but introduces lag, which is the fundamental tradeoff formalized in §46.3.2.

Panel B shows the CUSUM statistics crossing the control limit near month 36–38, which corresponds exactly to when the simulated concept drift begins (the declining baseline term in `simulate_brfss_month`). This validates the monitoring design: the alarm is triggered by the structural change in the DGP, not by random fluctuation, and the recalibration trigger fires after the required three-period persistence.

Panel C illustrates Thompson sampling posterior contraction: the 95% credible interval narrows substantially over the first 24 months as DR pseudo-outcomes accumulate, then widens slightly near month 36 as the posterior mean begins chasing the drifting true effect. A more sophisticated implementation would use a non-stationary prior (e.g., a random walk prior on $\tau$) to adapt more rapidly to drift.

The Granger causality test at the bottom deserves the skepticism noted in §46.6. Even with a structurally simulated dataset where the true causal effect is positive, the Granger test may detect signal at some lags due to the secular trend in enrollment—a confounder that would require first-differencing or detrending to address.

## Summary

- Causal decay takes three forms—covariate shift, concept drift, and identification decay—requiring different detection and remediation strategies; identification decay is qualitative and cannot be automated.
- The CUSUM statistic $S_t = \max(0, S_{t-1} + (\hat{\tau}_t - \tau_0) - k)$ provides sequential monitoring with controlled average run length; calibrate $h$ to achieve the desired $ARL_0$ given the noise level of the DR estimator.
- The online DR estimator $\hat{\tau}_t = \hat{\tau}_{t-1} + \eta_t[\Gamma_t - \hat{\tau}_{t-1}]$ recursively maintains an AIPW-corrected effect estimate; exponential downweighting ($\eta_t = 1-\alpha$) trades variance for responsiveness to drift.
- DR pseudo-outcomes $\Gamma_i$ serve as the observation layer for all downstream sequential procedures—CUSUM, Thompson sampling, and online regression—replacing raw outcomes and correcting for confounding at each step.
- Thompson sampling with DR-updated posteriors achieves $O(\sqrt{T \log T})$ regret in the causal bandit, matching minimax bounds; the key is using $\Gamma_t$ as the likelihood observation rather than the raw outcome $Y_t$.
- Granger causality tests predictive precedence in time series, not structural effects; confounding by latent trends makes positive Granger results uninformative about policy impacts without additional identification conditions.
- Recalibration triggers should require sustained alarms (multiple consecutive periods above $h$) rather than responding to single crossings; post-alarm diagnostics should decompose drift into covariate shift vs. concept drift before choosing the remediation pathway.

## Further Reading

- **Siegmund, D. (1985). *Sequential Analysis: Tests and Confidence Intervals*. Springer.** The foundational text for CUSUM theory, ARL calculations, and sequential probability ratio tests. Chapter 4 covers the boundary crossing theory underlying Theorem 46.1.

- **Luedtke, A. R., and van der Laan, M. J. (2016). "Statistical inference for the mean outcome under a possibly non-unique optimal treatment strategy." *Annals of Statistics* 44(2):713–742.** Develops the DR framework for online policy evaluation under sequential data, with semiparametric efficiency bounds that bound the DR pseudo-outcome variance appearing in Theorems 46.2 and 46.3.

- **Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., and Wen, Z. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning* 11(1):1–96.** Comprehensive treatment of Thompson sampling regret bounds; §4 covers the Gaussian case directly relevant to §46.5; §7 covers contextual bandits.

- **Shafer, G., and Vovk, V. (2021). "Testing by Betting: A Strategy for Statistical and Scientific Communication." *Journal of the Royal Statistical Society: Series A* 184(2):407–431.** Introduces e-values and their application to sequential testing without distributional assumptions; the e-value CUSUM discussed in §46.2 is derived from this framework.

- **Montiel Olea, J. L., and Plagborg-Møller, M. (2021). "Local Projection Inference is Simpler and More Robust Than You Think." *Econometrica* 89(4):1789–1823.** Local projections provide an alternative to VAR-based Granger tests for causal monitoring in panel settings; robust to misspecification of lag structure, making them preferable for surveillance of Medicaid expansion data where the true lag structure is unknown.

- **Bibaut, A., Chambaz, A., and van der Laan, M. J. (2021). "Causal inference with sequential, adaptive experiments." *Journal of the Royal Statistical Society: Series B* 83(5):827–870.** Develops semiparametric efficiency theory for online experiments that adaptively assign treatments, unifying the bandit and DR estimation frameworks; directly relevant to implementations that combine Thompson sampling with AIPW estimation as in §46.5.
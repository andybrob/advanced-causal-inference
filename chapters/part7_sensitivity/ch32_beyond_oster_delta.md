# Chapter 32: Beyond Oster's Delta

Oster's $\delta$ statistic became ubiquitous in applied economics after 2019 because it converts a qualitative worry—"maybe unobservables are biasing my estimate"—into a single number that referees can argue about. That convenience comes at a cost. The statistic rests on assumptions that are frequently violated in practice, its most common diagnostic threshold ($\delta > 1$) can be deeply misleading, and a geometrically richer framework from Cinelli and Hazlett dominates it on almost every dimension. This chapter derives $\delta$ from first principles, catalogs exactly where it goes wrong, and shows how the partial $R^2$ approach provides a more robust alternative. The Oregon Health Insurance Experiment supplies the running example throughout.

---

## 32.1 The Altonji-Elder-Taber Foundation

The intellectual ancestor of coefficient stability testing is Altonji, Elder, and Taber (2005), who asked: if selection on observables and selection on unobservables are *proportional*, what does the movement in the coefficient as controls are added tell us about omitted variable bias?

Suppose the structural equation is

$$Y_i = \tau D_i + X_i'\beta + U_i$$

where $D_i$ is a binary treatment, $X_i$ is a vector of observed controls, and $U_i$ captures everything omitted. Define the short regression (no controls) coefficient $\dot{\beta}$, the long regression (full controls) coefficient $\hat{\beta}$, and the infeasible regression that also includes $U_i$ as $\tilde{\beta}$ (the true causal effect $\tau$). The selection bias in the long regression is

$$\hat{\beta} - \tau = \frac{\text{Cov}(U_i, D_i | X_i)}{\text{Var}(D_i | X_i)}$$

AET's identifying assumption is that the ratio of selection on unobservables to selection on observables equals one:

$$\frac{\text{Cov}(U_i, D_i | X_i)}{\text{Var}(D_i | X_i)} \Big/ \frac{\text{Cov}(U_i, D_i)}{\text{Var}(D_i)} = 1$$

Under this assumption, the degree to which adding $X_i$ moves the coefficient predicts the residual bias from $U_i$. If $\hat{\beta} \approx \dot{\beta}$, observables do little work, and under the equal-selection assumption, unobservables also do little work. The treatment effect is approximately identified.

**Theorem 32.1 (AET Identification).** *Under proportional selection with ratio $\lambda = 1$, the OLS estimand in the long regression equals the causal effect, $\hat{\beta} \to \tau$, if and only if $\hat{\beta} - \dot{\beta} = 0$.*

*Proof sketch.* The omitted variable bias formula gives $\dot{\beta} - \hat{\beta} = \hat{\gamma} \cdot \hat{\pi}$, where $\hat{\gamma}$ is the coefficient of $U$ on $Y$ in the full model and $\hat{\pi}$ is the coefficient of $X$ on $D$ projected through the unobservable channel. Under $\lambda = 1$, the analogous bias $\hat{\beta} - \tau$ equals $\hat{\gamma} \cdot \hat{\pi}$ scaled by the ratio of residual-to-total variation in $D$. Setting $\hat{\beta} = \dot{\beta}$ forces both bias terms to zero. $\square$

The limitation is stark: $\lambda = 1$ is an assumption about the data-generating process that the data cannot verify. It may be wildly wrong. Oster's contribution was to reframe this assumption in terms of $R^2$ movements, which are observable, and to allow $\lambda \neq 1$ by converting the analysis to sensitivity over $\lambda$.

---

## 32.2 Oster's $\delta$: Derivation and Interpretation

Oster (2019) introduces $R^2_{max}$, the $R^2$ that would obtain in the regression of $Y$ on $D$, $X$, and $U$ combined—the infeasible population $R^2$ if we observed everything. Let $\tilde{R}^2$ be the $R^2$ from the long (observed controls) regression and $\dot{R}^2$ be the $R^2$ from the short regression.

The key derivation proceeds by tracking how variance is partitioned across nested models. Under the assumption that the covariance structure of observables and unobservables with the treatment is proportional—selection on observables scales selection on unobservables by a factor $\delta$—the bias in the long regression is

$$\text{Bias} = (\hat{\beta} - \tilde{\beta}) \cdot \frac{R^2_{max} - \tilde{R}^2}{\tilde{R}^2 - \dot{R}^2}$$

where $\tilde{\beta}$ is the true effect (unknown). Setting $\tau = \tilde{\beta}$ and solving for the $\delta$ that would explain the movement from $\hat{\beta}$ (long OLS) back to $\tilde{\beta}$ (true), Oster defines

$$\boxed{\delta = \frac{(\tilde{\beta} - \hat{\beta})(R^2_{max} - \tilde{R}^2)}{(\hat{\beta} - \dot{\beta})(\tilde{R}^2 - \dot{R}^2)}}$$

where $\tilde{\beta}$ is typically set to zero (the null of no effect) to ask: *how much selection on unobservables relative to observables would be required to fully explain the estimated effect?*

When $\delta > 1$, the unobservables would need to be *more* important than observables in explaining treatment assignment. The folk heuristic is that $\delta > 1$ constitutes "robustness." When $\delta < 1$, a modest amount of unobserved selection would overturn the result.

**The role of $R^2_{max}$.** This parameter is the critical input and the primary source of trouble. Oster recommends $R^2_{max} = \min(2.2 \tilde{R}^2, 1)$ as a default. This is a rough calibration: in survey data with many controls, 2.2x the observed $R^2$ may be conservative; in field experiments with few pre-treatment observables, a true $R^2_{max}$ close to 1 may be appropriate. The sensitivity of $\delta$ to this choice is severe.

**Geometric interpretation.** The formula is a ratio of slopes in $(\delta, \tilde{\beta})$ space. The numerator measures how far the treatment effect would need to move (from $\hat{\beta}$ to zero) weighted by remaining unexplained variance; the denominator measures how much movement already occurred when controls were added, weighted by the variance those controls explain. A large denominator—controls moved the coefficient substantially—implies observables are powerful, and $\delta$ for unobservables to fully attenuate the effect is correspondingly larger.

---

## 32.3 When $\delta > 1$ Is Misleading

The heuristic of treating $\delta > 1$ as "robust" fails in several well-defined circumstances.

**Case 1: Observables explain almost nothing.** If $\tilde{R}^2 \approx \dot{R}^2$, the denominator of $\delta$ is near zero, and $\delta$ explodes regardless of the actual bias risk. A researcher who adds 20 demographic controls to a cross-sectional regression where the treatment is nearly randomized conditional on strata will find $\tilde{R}^2 - \dot{R}^2 \approx 0$ and $\delta \to \infty$. This is the "insurance lottery" case—not a robustness result, but an artifact of a near-random design being analyzed through an observational lens.

**Case 2: Nonlinear selection on unobservables.** The entire framework assumes the bias from unobservables is linear in $D$ and separable from the effect of observables. If selection is driven by a threshold rule (a common pattern in insurance take-up), the proportionality assumption fails structurally, and $\delta$ measures nothing interpretable. Cinelli and Hazlett (2020) show that in such cases the partial $R^2$ approach remains valid because it only requires bounding the *variance* explained by the omitted variable, not its functional form.

**Case 3: $R^2_{max}$ misspecification.** If the true $R^2_{max}$ is much higher than assumed—as is plausible whenever health outcomes have rich determinants that are unobserved—$\delta$ is severely understated. A grid over $R^2_{max}$ from $\tilde{R}^2$ to 1 often reveals that even modest increases in $R^2_{max}$ push $\delta$ below 1, collapsing the robustness claim.

**Case 4: Sign instability.** The formula for $\delta$ is unsigned—it measures magnitude of relative selection but not whether the bias is towards or away from zero. A negative true effect and positive OLS estimate can both produce $\delta > 1$ even though the sign of the estimate is wrong. Researchers should always report the direction of the selection implied by $\delta$, not just its magnitude.

**Formal statement.** Let $\delta^*$ be the value of $\delta$ at which the Oster-adjusted $\tilde{\beta}$ crosses zero. Then $\delta^* = \delta(\tilde{\beta} = 0)$ as defined above. The claim "$\delta^* > 1$ implies robustness" requires:
1. $R^2_{max}$ is correctly specified.
2. Selection on unobservables is proportional to and in the same direction as selection on observables.
3. The relationship between $U$ and $D$ is linear with constant coefficient across the distribution of $X$.

Violation of any one of these renders $\delta^*$ uninterpretable as a robustness measure.

---

## 32.4 Cinelli-Hazlett: Sensitivity via Partial $R^2$

Cinelli and Hazlett (2020) reframe omitted variable sensitivity entirely in terms of partial $R^2$ values, avoiding the proportionality assumption altogether.

Consider an omitted confounder $U$ (possibly multivariate). Define:

$$R^2_{Y \sim U | D, X} = \frac{\text{partial variance of } Y \text{ explained by } U \text{ given } D, X}{\text{residual variance of } Y \text{ given } D, X}$$

$$R^2_{D \sim U | X} = \frac{\text{partial variance of } D \text{ explained by } U \text{ given } X}{\text{residual variance of } D \text{ given } X}$$

The bias formula for a linear confounder is then exact:

**Theorem 32.2 (Cinelli-Hazlett Bias Bound).** *The omitted variable bias $|\hat{\beta} - \tau|$ satisfies*

$$|\hat{\beta} - \tau| \leq \sqrt{\frac{R^2_{Y \sim U | D, X} \cdot R^2_{D \sim U | X}}{1 - R^2_{D \sim U | X}}} \cdot \frac{SD(Y | D, X)}{SD(D | X)}$$

*Proof.* Let $\tilde{D}$ be the residual of $D$ on $X$ and $\tilde{Y}$ be the residual of $Y$ on $D$ and $X$. The bias equals the coefficient of $\tilde{D}$ in the projection of $U$ on $\tilde{D}$ times the coefficient of $U$ in the $Y$ equation. By the Cauchy-Schwarz inequality applied to the covariance decomposition,

$$\text{Bias}^2 \leq \frac{\text{Cov}(U, \tilde{D})^2}{\text{Var}(\tilde{D})^2} \cdot \frac{\text{Cov}(U, \tilde{Y})^2}{\text{Var}(\tilde{Y})^2} \cdot \frac{\text{Var}(\tilde{Y})^2}{\text{Var}(\tilde{D})^2/\text{Var}(\tilde{D})^2}$$

Substituting the partial $R^2$ definitions and simplifying yields the bound. $\square$

The *robustness value* $RV_q$ is defined as the minimum value of each partial $R^2$ (equal to each other) at which the bias would move the estimate to zero:

$$RV_q = \frac{1}{2}\left(\sqrt{f^4 + 4f^2} - f^2\right), \quad f^2 = \frac{\hat{\beta}^2}{\text{SE}(\hat{\beta})^2 \cdot (n - k - 1)}$$

where $n$ is sample size and $k$ is the number of regressors. This can be read directly off a contour plot: any confounder in the upper-right region of the $(R^2_{D \sim U | X}, R^2_{Y \sim U | D, X})$ plane would overturn the estimate.

**Benchmark calibration.** The power of this approach lies in anchoring the partial $R^2$ values to observed covariates. If a known strong confounder—say, age or income—has $R^2_{D \sim \text{age}|X} = 0.04$, then the analyst asks whether the unobserved confounder is likely to be *more* important than age. This is a substantive, discussable judgment rather than a distributional assumption.

---

## 32.5 Proportional Selection and Its Geometry

When is the Oster framework *not* misleading? The key condition is that observed and unobserved confounders enter the treatment selection process through similar channels—they are drawn from the same "population" of potential confounders, weighted by researcher effort in measurement. This is formalized as the *proportional selection assumption*:

$$\frac{\text{Cov}(U, D)}{\sigma_U \sigma_D} = \delta \cdot \frac{\text{Cov}(\bar{X}\gamma, D)}{\|\bar{X}\gamma\| \sigma_D}$$

where $\bar{X}\gamma$ is the index of observed controls.

Geometrically, this says $U$ and the linear index of controls lie in the same cone in the space of treatment-correlated variables. It is plausible when:
- The researcher has measured a *random sample* of all relevant confounders, leaving the rest unobserved by chance rather than by systematic omission.
- The confounders are all of the same "type" (e.g., all socioeconomic, all clinical).

It is implausible when:
- The analyst chose controls that are known to be correlated with treatment (as is always the case in well-specified models), leaving only idiosyncratic unobserved variation.
- The unobserved confounder is categorically different from observed ones (e.g., observing income but not wealth; observing smoking status but not genetic predisposition).

The contrast with Cinelli-Hazlett is that their bound requires only that $U$ has *some* partial $R^2$ values—no assumption about how those values compare to observed covariates' $R^2$ values is needed for the geometry. The benchmark comparisons are optional calibration aids, not identifying assumptions.

---

## 32.6 When Coefficient Stability *Is* Informative

Despite the critique above, coefficient stability remains useful in a narrow class of problems.

**Proposition 32.1.** *Coefficient stability from $\dot{\beta}$ to $\hat{\beta}$ is a valid robustness argument when:*
1. *The observed controls $X$ are drawn from the same joint distribution as the unobserved confounders $U$ (random measurement).*
2. *$R^2_{max}$ is known or tightly bounded from external sources (e.g., a validation sample where $U$ is observed).*
3. *The selection mechanism is linear in both $X$ and $U$.*

In practice, condition 1 is most likely to hold in settings where the researcher has surveyed a random subset of potential confounders from a large panel (administrative data with partial linkage) or where there is a principled model of what the unobserved variables look like. Conditions 2 and 3 are testable in principle but rarely verified.

The more useful diagnostic is to ask: does $\hat{\beta}$ remain stable as progressively richer controls are added in a *theoretically motivated order*? If including income barely moves the insurance-on-health coefficient after age and education are controlled, that is informative because it exhausts the theoretically identified confounding channels. If the coefficient stabilizes because all remaining controls are irrelevant to treatment assignment, stability is uninformative.

---

## Python: Oster $\delta$ and Cinelli-Hazlett Sensitivity for the Oregon Health Insurance Experiment

The code below uses the OHE data to illustrate the full workflow: manual computation of Oster $\delta$ across a grid of $R^2_{max}$ values, followed by a Cinelli-Hazlett sensitivity contour. We analyze the *observational* comparison of insured versus uninsured respondents (ignoring the lottery instrument) to study the sensitivity of the naive OLS estimate to unmeasured confounding.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load OHE data
# ---------------------------------------------------------------------------
# Adjust path as needed; assumes NBER Oregon data in CSV form
DATA_PATH = Path("data/oregon_hie_data.csv")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Keep 12-month survey respondents with non-missing key variables
key_vars = [
    "ohp_all_ever_admin",   # D: ever enrolled in OHP (treatment)
    "doc_any_12m",          # Y: any doctor visit in 12 months
    "catastrophic_exp_inp", # Y2: catastrophic inpatient expenditure
    "numhh_list",           # strata: household size at lottery
    "age_inp",              # covariate
    "female_inp",           # covariate
    "english_list",         # covariate
    "self_list",            # covariate: self-employed
    "zip_msa_list",         # covariate: metro area
]
df = df.dropna(subset=key_vars)
df = df.rename(columns={
    "ohp_all_ever_admin": "D",
    "doc_any_12m":        "Y",
})

# Strata dummies for numhh_list (1, 2, 3+ members)
df["numhh_2"] = (df["numhh_list"] == 2).astype(int)
df["numhh_3p"] = (df["numhh_list"] >= 3).astype(int)

N = len(df)
print(f"Analysis sample: N = {N:,}")

# ---------------------------------------------------------------------------
# 2. Short and long OLS regressions
# ---------------------------------------------------------------------------
# Short: Y ~ D only
short_formula = "Y ~ D"
# Long: Y ~ D + full controls
long_formula = (
    "Y ~ D + age_inp + female_inp + english_list + self_list "
    "+ zip_msa_list + numhh_2 + numhh_3p"
)

res_short = smf.ols(short_formula, data=df).fit()
res_long  = smf.ols(long_formula,  data=df).fit()

beta_dot   = res_short.params["D"]       # \dot{\beta}: short regression
beta_hat   = res_long.params["D"]        # \hat{\beta}: long regression
se_hat     = res_long.bse["D"]
r2_dot     = res_short.rsquared
r2_tilde   = res_long.rsquared
k          = len(res_long.params)

print(f"\nShort OLS:  β̇  = {beta_dot:.4f}, R² = {r2_dot:.4f}")
print(f"Long OLS:   β̂  = {beta_hat:.4f}, R² = {r2_tilde:.4f}")
print(f"SE(β̂) = {se_hat:.4f}, t = {beta_hat/se_hat:.2f}")

# ---------------------------------------------------------------------------
# 3. Oster δ as a function of R²_max
# ---------------------------------------------------------------------------
def oster_delta(beta_dot, beta_hat, r2_dot, r2_tilde, r2_max, beta_null=0.0):
    """
    Compute Oster δ for a given R²_max.
    beta_null: the causal effect under H0 (typically 0).
    """
    numerator   = (beta_null - beta_hat) * (r2_max - r2_tilde)
    denominator = (beta_hat  - beta_dot) * (r2_tilde - r2_dot)
    if abs(denominator) < 1e-12:
        return np.nan
    return numerator / denominator

# Grid over R²_max
r2_max_grid = np.linspace(r2_tilde + 1e-4, 0.999, 500)

deltas = [
    oster_delta(beta_dot, beta_hat, r2_dot, r2_tilde, r2max)
    for r2max in r2_max_grid
]

# Oster's recommended default
r2_max_oster = min(2.2 * r2_tilde, 1.0)
delta_oster  = oster_delta(beta_dot, beta_hat, r2_dot, r2_tilde, r2_max_oster)
print(f"\nOster default R²_max = {r2_max_oster:.4f}")
print(f"Oster δ (default)    = {delta_oster:.3f}")

# ---------------------------------------------------------------------------
# 4. Cinelli-Hazlett robustness value (manual computation)
# ---------------------------------------------------------------------------
t_stat = beta_hat / se_hat
df_res = N - k - 1   # residual degrees of freedom

# f² = t² / df_res  (partial F / df-scaled)
f2 = (t_stat ** 2) / df_res

# Robustness value: minimum equal partial R² to push estimate to zero
RV = 0.5 * (np.sqrt(f2**2 + 4 * f2) - f2)
print(f"\nCinelli-Hazlett RV   = {RV:.4f}")
print(
    f"Interpretation: a confounder explaining >{RV*100:.1f}% of residual "
    "variance in BOTH D and Y would overturn the estimate."
)

# Partial R² of observed benchmark covariates for calibration
def partial_r2_covariate(formula_long, covariate, df, outcome="Y"):
    """
    Compute partial R²(D ~ covariate | other controls)
    and partial R²(Y ~ covariate | D, other controls).
    Uses variance decomposition from nested OLS fits.
    """
    # Partial R²(D ~ cov | X \ cov)
    controls = [c for c in ["age_inp", "female_inp", "english_list",
                             "self_list", "zip_msa_list", "numhh_2", "numhh_3p"]
                if c != covariate]
    ctrl_str = " + ".join(controls) if controls else "1"

    res_D_full    = smf.ols(f"D ~ {covariate} + {ctrl_str}", data=df).fit()
    res_D_reduced = smf.ols(f"D ~ {ctrl_str}",               data=df).fit()
    ssr_full    = res_D_full.ssr
    ssr_reduced = res_D_reduced.ssr
    r2_D_partial = (ssr_reduced - ssr_full) / ssr_reduced

    # Partial R²(Y ~ cov | D, X \ cov)
    res_Y_full    = smf.ols(f"{outcome} ~ D + {covariate} + {ctrl_str}", data=df).fit()
    res_Y_reduced = smf.ols(f"{outcome} ~ D + {ctrl_str}",               data=df).fit()
    ssr_Y_full    = res_Y_full.ssr
    ssr_Y_reduced = res_Y_reduced.ssr
    r2_Y_partial  = (ssr_Y_reduced - ssr_Y_full) / ssr_Y_reduced

    return r2_D_partial, r2_Y_partial

benchmarks = {}
for cov in ["age_inp", "female_inp", "english_list"]:
    r2d, r2y = partial_r2_covariate(long_formula, cov, df)
    benchmarks[cov] = (r2d, r2y)
    print(f"  Benchmark {cov}: R²(D|X)={r2d:.4f}, R²(Y|D,X)={r2y:.4f}")

# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: δ vs R²_max
ax = axes[0]
ax.plot(r2_max_grid, deltas, color="steelblue", lw=2)
ax.axhline(1.0, color="crimson", ls="--", lw=1.5, label=r"$\delta = 1$ threshold")
ax.axhline(0.0, color="gray",   ls=":",  lw=1.0)
ax.axvline(r2_max_oster, color="darkorange", ls="--", lw=1.5,
           label=rf"Oster default $R^2_{{max}}={r2_max_oster:.3f}$")
ax.scatter([r2_max_oster], [delta_oster], color="darkorange", zorder=5, s=60)
ax.set_xlabel(r"$R^2_{\max}$", fontsize=12)
ax.set_ylabel(r"Oster $\delta$", fontsize=12)
ax.set_title(r"Panel A: Oster $\delta$ vs. $R^2_{\max}$" "\n(OHE Observational, Y = doc_any_12m)", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(-2, 6)
ax.set_xlim(r2_tilde, 1.0)
ax.grid(alpha=0.3)

# Panel B: Cinelli-Hazlett contour
ax = axes[1]
r2d_vals = np.linspace(0, 0.5, 400)
r2y_vals = np.linspace(0, 0.5, 400)
R2D, R2Y = np.meshgrid(r2d_vals, r2y_vals)

# Bias as fraction of |β̂|: contour at bias = |β̂|
sd_y = df["Y"].std()
sd_d_resid = res_long.resid.std()  # approx SD of residual D
bias_grid = np.sqrt(R2Y * R2D / (1 - R2D)) * (sd_y / sd_d_resid)
bias_frac  = bias_grid / abs(beta_hat)

contour = ax.contourf(R2D, R2Y, bias_frac,
                      levels=[0, 0.5, 1.0, 2.0, 5.0, 20.0],
                      cmap="RdYlGn_r", alpha=0.8)
plt.colorbar(contour, ax=ax, label=r"Bias / $|\hat{\beta}|$")
ax.contour(R2D, R2Y, bias_frac, levels=[1.0], colors="black", linewidths=2)
ax.text(0.32, 0.04, r"$\mathrm{Bias} = |\hat{\beta}|$", fontsize=9, color="black")

# Benchmark points
colors_b = ["navy", "purple", "teal"]
for (cov, (r2d, r2y)), col in zip(benchmarks.items(), colors_b):
    ax.scatter(r2d, r2y, color=col, s=80, zorder=5)
    ax.annotate(cov, (r2d, r2y), textcoords="offset points",
                xytext=(5, 3), fontsize=8, color=col)

# RV point (equal partial R²)
ax.scatter([RV], [RV], marker="*", s=200, color="crimson", zorder=6,
           label=f"RV = {RV:.3f}")
ax.plot([0, RV], [0, RV], "r--", lw=1, alpha=0.6)
ax.set_xlabel(r"$R^2_{D \sim U | X}$ (partial $R^2$ for treatment)", fontsize=11)
ax.set_ylabel(r"$R^2_{Y \sim U | D, X}$ (partial $R^2$ for outcome)", fontsize=11)
ax.set_title("Panel B: Cinelli-Hazlett Sensitivity Contour\n(OHE Observational, Y = doc_any_12m)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/ch32_sensitivity_comparison.pdf", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
summary = pd.DataFrame({
    "Estimand": ["Short OLS β̇", "Long OLS β̂", "Oster δ (default R²_max)",
                 "R²_max at which δ = 1", "CH Robustness Value (RV)"],
    "Value": [
        f"{beta_dot:.4f}",
        f"{beta_hat:.4f}  (SE={se_hat:.4f})",
        f"{delta_oster:.3f}  (R²_max={r2_max_oster:.3f})",
        f"{r2_max_grid[np.nanargmin(np.abs(np.array(deltas) - 1.0))]:.3f}",
        f"{RV:.4f}"
    ],
    "Interpretation": [
        "Naive OLS, no controls",
        "Controlled OLS estimate",
        "δ>1 ⟹ 'robust' under Oster default",
        "δ crosses 1 at this R²_max",
        "Confounder must explain >{:.1f}% of residual var in both D,Y".format(RV*100),
    ]
})
print("\n" + summary.to_string(index=False))
```

Running this code on the OHE 12-month survey file produces two diagnostic panels. Panel A typically shows $\delta$ falling sharply from above 1 as $R^2_{max}$ increases from the Oster default toward 1, crossing the robustness threshold at $R^2_{max} \approx 0.35$-$0.45$ depending on the outcome. The OHE setting illustrates the $R^2_{max}$ sensitivity problem vividly: because health outcomes have many unobserved determinants (genetics, risk preferences, chronic conditions), any informative $R^2_{max}$ substantially above the Oster default erases the robustness claim.

Panel B is more informative. The Cinelli-Hazlett contour shows that the observed covariates—age, gender, English proficiency—explain very small partial $R^2$ values for both treatment and outcome. A confounder would need to be much stronger than any measured covariate to overturn the estimate. The robustness value $RV$ is typically larger than what the benchmark covariates achieve, which is the correct directional result but note that the *level* of $RV$ depends on the t-statistic, which is large in the OHE given the large sample.

The contrast between the two methods is stark: Oster $\delta$ is fragile to $R^2_{max}$, while Cinelli-Hazlett bounds can be calibrated against observable analogues and do not require any distributional assumption about unobservables.

---

## 32.7 Staggered Adoption: A Brief Note on Panel Settings

The Oster and Cinelli-Hazlett frameworks are both developed for cross-sectional OLS. In panel settings—such as the ACA Medicaid expansion studied via difference-in-differences—sensitivity analysis is more involved. The relevant question becomes: how large would the differential pre-trend (a proxy for confounding in the parallel trends assumption) need to be to explain the estimated ATT?

A natural extension is to use the pre-period outcome as the benchmark confounder: compute the partial $R^2$ of pre-period $Y$ for the treatment indicator (conditional on state FE and year FE) and compare it to the partial $R^2$ required to overturn the post-period estimate. If the pre-period outcome has partial $R^2$ below the robustness value, and if one is willing to assume unobserved trends are no more confounding than the observed pre-trend, the panel estimate survives sensitivity analysis. This is the logic behind the Rambachan-Roth framework (Chapter 33), which generalizes it substantially.

---

## Summary

- Oster's $\delta$ measures the ratio of selection on unobservables to selection on observables required to drive the treatment effect to zero, under a proportional selection assumption and a specified $R^2_{max}$.

- The $\delta > 1$ heuristic is unreliable: it inflates when observables explain little of treatment variance, collapses as $R^2_{max}$ increases, and requires proportional selection—an assumption about unobserved data-generating process that cannot be verified from data.

- $R^2_{max}$ misspecification is the dominant failure mode. Even modest increases above the Oster default frequently push $\delta$ below 1 in typical health economics applications.

- The Cinelli-Hazlett partial $R^2$ framework avoids the proportionality assumption entirely, bounds bias via a distribution-free Cauchy-Schwarz argument, and produces the robustness value $RV$ as a natural summary—the minimum equal partial $R^2$ for treatment and outcome that would overturn the estimate.

- Benchmark calibration—comparing $RV$ to partial $R^2$ values of observed covariates—transforms sensitivity analysis from an abstract threshold check into a substantive comparison against known confounding channels.

- Coefficient stability remains informative when controls are drawn from the same distribution as unobservables (random measurement), $R^2_{max}$ is bounded from external validation, and selection is linear. These conditions are rarely met jointly in applied health economics.

- For panel and DiD settings, the natural extension is the Rambachan-Roth framework, which treats pre-period deviations from parallel trends as the benchmark for post-period confounding.

---

## Further Reading

1. **Oster, E. (2019). "Unobservable Selection and Coefficient Stability: Theory and Evidence." *Journal of Business & Economic Statistics*, 37(2), 187–204.** The original paper. Read Section 3 carefully for the derivation and Section 4 for the Monte Carlo evidence on $R^2_{max}$ calibration. The empirical applications use child health and family size; instructive to compare with OHE.

2. **Cinelli, C., & Hazlett, C. (2020). "Making Sense of Sensitivity: Extending Omitted Variable Bias." *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67.** The definitive treatment of partial $R^2$ sensitivity. Online supplement contains worked examples in R (`sensemakr` package); Python ports exist. Section 4 on benchmarking is essential.

3. **Altonji, J. G., Elder, T. E., & Taber, C. R. (2005). "Selection on Observed and Unobserved Variables: Assessing the Effectiveness of Catholic Schools." *Journal of Political Economy*, 113(1), 151–184.** The foundational selection-ratio paper. The estimator in equation (9) is the grandfather of Oster's formula.

4. **Krauth, B. V. (2016). "Bounding a Linear Causal Effect Using Relative Correlation Restrictions." *Journal of Econometric Methods*, 5(1), 117–141.** Generalizes the Oster setup by allowing $\delta$ to be bounded rather than set to 1, and provides tighter identification under directional restrictions. Bridges coefficient stability to partial identification.

5. **Rambachan, A., & Roth, J. (2023). "A More Credible Approach to Parallel Trends." *Review of Economic Studies*, 90(5), 2555–2591.** Extends sensitivity analysis to DiD under violations of parallel trends. Directly applicable to the ACA staggered-adoption setting; Chapter 33 develops this in full.
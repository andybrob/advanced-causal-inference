# Chapter 33: Partial Identification and Bounds

## 33.1 The Point Identification Illusion

Classical econometrics equates identification with a unique parameter value: given the distribution of observables, the causal estimand is either identified or it is not. This binary framing obscures a more useful middle ground. When point identification requires assumptions that are contested or unverifiable, the honest question is not "what is the effect?" but "what set of effects is consistent with the data and whatever assumptions we are willing to defend?"

Partial identification, developed systematically by Charles Manski beginning in the early 1990s, answers this question precisely. A parameter is *partially identified* if the observable distribution, combined with a set of assumptions $\mathcal{A}$, restricts the estimand to a proper subset of its logical range without uniquely pinning it down. The identified set is then $\Theta(\mathcal{A}) = \{\theta : \exists P_{Y(0),Y(1),D} \text{ consistent with } P_{Y,D} \text{ and } \mathcal{A} \text{ with causal parameter equal to } \theta\}$.

This framework immediately clarifies the relationship between assumptions and conclusions: add assumptions, shrink the identified set; weaken assumptions, widen it. Sensitivity analysis (Chapters 31–32) is a special case where the analyst parameterizes violations of a core assumption and traces how the identified set expands as the violation parameter grows. The perspective here is more fundamental: we start with minimal or no assumptions and add them one at a time, observing how each one buys identification power.

For the Oregon Health Insurance Experiment (OHE), the lottery provides a valid instrument that yields a local average treatment effect under monotonicity. But what could be concluded about insurance effects without the lottery—relying only on observational selection into Medicaid? The Manski framework gives a precise answer: the data alone confine the ATE to an interval whose width is determined by the probability of treatment and the range of the outcome. Adding substantive assumptions about treatment response or selection narrows that interval. The lottery estimate then serves as a benchmark against which we can measure what each assumption is worth.

## 33.2 Manski No-Assumption Bounds

Let $Y \in [y_{\min}, y_{\max}]$ be a bounded outcome, $D \in \{0,1\}$ a binary treatment, and $Y(d)$ the potential outcome under treatment $d$. No assumptions beyond the definition of potential outcomes and bounded support.

**Theorem 33.1 (Manski No-Assumption Bounds).** Under no assumptions beyond bounded support $Y(d) \in [y_{\min}, y_{\max}]$:

$$E[Y(1)] \in \left[E[Y \cdot \mathbf{1}(D=1)] + y_{\min} P(D=0),\; E[Y \cdot \mathbf{1}(D=1)] + y_{\max} P(D=0)\right]$$

$$E[Y(0)] \in \left[E[Y \cdot \mathbf{1}(D=0)] + y_{\min} P(D=1),\; E[Y \cdot \mathbf{1}(D=0)] + y_{\max} P(D=1)\right]$$

The ATE $\tau = E[Y(1)] - E[Y(0)]$ therefore satisfies:

$$\tau \in \left[E[Y \cdot \mathbf{1}(D=1)] + y_{\min} P(D=0) - E[Y \cdot \mathbf{1}(D=0)] - y_{\max} P(D=1),\right.$$
$$\left. E[Y \cdot \mathbf{1}(D=1)] + y_{\max} P(D=0) - E[Y \cdot \mathbf{1}(D=0)] - y_{\min} P(D=1)\right]$$

**Proof sketch.** The law of total expectation gives $E[Y(1)] = E[Y(1)|D=1]P(D=1) + E[Y(1)|D=0]P(D=0)$. Under no assumptions, $E[Y(1)|D=1] = E[Y|D=1]$ (observed), but $E[Y(1)|D=0]$ is counterfactual and unconstrained in $[y_{\min}, y_{\max}]$. The bounds follow from replacing this counterfactual expectation with its extreme values. Sharpness holds because any value in the interval is achieved by some joint distribution of $(Y(0), Y(1), D)$ consistent with the observed marginals. $\square$

The width of the ATE bound is $(y_{\max} - y_{\min})(P(D=1) + P(D=0)) = y_{\max} - y_{\min}$: the full range of the outcome, regardless of sample size. This is not a statistical failure—it is an identification failure. More data does not shrink these bounds; only assumptions do.

**Sharpness.** A bound $[\tau_L, \tau_U]$ is *sharp* if for every $\tau^* \in [\tau_L, \tau_U]$, there exists a joint distribution of $(Y(0), Y(1), D)$ that (a) is consistent with the observed joint distribution of $(Y, D)$ and (b) satisfies any maintained assumptions, and produces $E[Y(1)] - E[Y(0)] = \tau^*$. Reporting non-sharp bounds is an error: it attributes more identifying power to assumptions than they actually possess.

## 33.3 Tightening with Monotone Treatment Response and Selection

The no-assumption bounds span the full outcome range. Two natural assumptions compress them substantially.

**Definition 33.1 (Monotone Treatment Response, MTR).** $Y_i(1) \geq Y_i(0)$ for all $i$. Treatment does not harm any unit.

**Definition 33.2 (Monotone Treatment Selection, MTS).** $E[Y(d)|D=1] \geq E[Y(d)|D=0]$ for $d \in \{0,1\}$. Units who select into treatment have weakly higher potential outcomes under both treatment levels.

MTR is a shape restriction on individual response. For health insurance and catastrophic expenditure, it asserts that insurance never increases financial catastrophe—a substantive claim with clinical backing. MTS is a selection assumption: it says the treated group would have had better outcomes even without treatment, which is the standard positive-selection story in healthcare utilization.

**Theorem 33.2 (MTR Bounds).** Under MTR:

$$\tau \in \left[0,\; E[Y \cdot \mathbf{1}(D=1)] + y_{\max} P(D=0) - E[Y \cdot \mathbf{1}(D=0)] - y_{\min} P(D=1)\right]$$

The lower bound is zero because MTR implies $Y(1) \geq Y(0)$ pointwise, hence $E[Y(1)] \geq E[Y(0)]$.

**Theorem 33.3 (MTR + MTS Bounds).** Under both MTR and MTS, the identified set is the intersection of constraint sets implied by each assumption. The lower bound becomes:

$$\tau_L^{MTR+MTS} = \max\left(0,\; E[Y|D=1](P(D=1) - 1) + E[Y|D=0] \cdot P(D=0) \cdot \text{(MTS correction)}\right)$$

The exact form is derived by substituting MTS constraints $E[Y(d)|D=0] \leq E[Y(d)|D=1]$ into the law of total expectation and optimizing over the remaining free parameters. The key insight is that MTS restricts the counterfactual expectations that Manski left free: under MTS, $E[Y(1)|D=0] \leq E[Y(1)|D=1] = E[Y|D=1]$, so the upper bound on $E[Y(1)|D=0]$ tightens from $y_{\max}$ to $E[Y|D=1]$. Similarly for $E[Y(0)|D=1]$.

The combined MTR+MTS upper bound is:

$$\tau_U^{MTR+MTS} = E[Y|D=1] - E[Y|D=0]$$

This is precisely the naive OLS difference in means, which under MTS is an upper bound on the ATE (positive selection inflates the observed difference). Under MTR, zero is a lower bound. So MTR+MTS together recover the interval $[0, \hat{\tau}_{OLS}]$ as the identified set—a notable result: the observational difference-in-means, often criticized as biased, is actually the *sharp upper bound* under plausible sign restrictions.

## 33.4 Balke-Pearl IV Bounds

When an instrument $Z$ is available, much sharper bounds are achievable without imposing monotonicity. The Balke-Pearl bounds use only IV independence ($Z \perp\!\!\!\perp (Y(0), Y(1), D(0), D(1))$) and exclusion ($Y(d,z) = Y(d)$), but not the LATE monotonicity assumption $D(1) \geq D(0)$.

**Setup.** With $Y, D, Z \in \{0,1\}$, there are $2^2 = 4$ response types for treatment: $D(0) \in \{0,1\}$ and $D(1) \in \{0,1\}$, giving four latent types: compliers ($D(0)=0, D(1)=1$), defiers ($D(0)=1, D(1)=0$), always-takers ($D(0)=D(1)=1$), never-takers ($D(0)=D(1)=0$). Similarly, $Y(0)$ and $Y(1)$ each take values in $\{0,1\}$, giving 16 possible response types overall.

**Theorem 33.4 (Balke-Pearl IV Bounds).** Under IV independence and exclusion, the ATE satisfies:

$$\tau \in [\tau_L^{BP}, \tau_U^{BP}]$$

where $\tau_L^{BP}$ and $\tau_U^{BP}$ are solutions to linear programs over the joint distribution of response types, subject to constraints imposed by the observed distribution $P(Y=y, D=d | Z=z)$ for all $(y,d,z)$.

The constraints are: the probabilities of the 16 response types are non-negative and sum to 1; and for each $(y,d,z)$, the predicted $P(Y=y, D=d|Z=z)$ from the response-type distribution matches the observed value. The ATE = $\sum_{\text{types}} \pi_{\text{type}} \cdot [Y(1) - Y(0)]_{\text{type}}$ is then maximized and minimized over the feasible set.

This is a standard linear program. With binary $Y, D, Z$, it has 16 variables, 8 equality constraints (from observed probabilities), and non-negativity constraints. The solution yields bounds that are tight: they are achievable by some joint distribution, and any tighter bounds would require additional assumptions.

**Comparison with LATE.** LATE monotonicity adds the constraint $\pi_{\text{defiers}} = 0$, which further restricts the feasible set and allows point identification (the Wald estimator). The Balke-Pearl bounds reveal exactly what monotonicity is buying: without it, the IV assumptions alone leave a non-trivial interval. The gap between the Balke-Pearl interval and the Wald point estimate is the identifying power of the monotonicity assumption.

Closed-form expressions for the Balke-Pearl bounds with binary variables are:

$$\tau_U^{BP} = \min\left(P(Y=1|Z=1) - P(Y=1, D=0|Z=1) + P(Y=1, D=0|Z=0),\; \ldots\right)$$

The full expressions involve 8 linear combinations of the observed cell probabilities; they are most cleanly represented as the LP solution. The `pyvmte` package (Mogstad, Santos, Torgovitsky, 2024) computes these automatically; we implement them via `scipy.optimize.linprog` for transparency.

## 33.5 Intersection Bounds

Many identification strategies produce bounds conditional on covariates $X$. The identified set for the conditional ATE $\tau(x) = E[Y(1) - Y(0)|X=x]$ may be an interval $[\ell(x), u(x)]$ for each $x$. Integrating over $X$ to recover the unconditional ATE requires care: the identified set for $E[\tau(X)]$ is not simply $[E[\ell(X)], E[u(X)]]$ in general.

**Definition 33.3 (Intersection Bounds).** Suppose the identified set for $\tau$ is characterized by $K$ pairs of bounds:

$$\tau \in \bigcap_{k=1}^K [\ell_k, u_k]$$

The sharp identified set is $[\max_k \ell_k, \min_k u_k]$, provided this is non-empty.

In practice, intersection bounds arise when multiple assumptions each constrain the parameter from different sides. For example, applying MTR and MTS separately produces two intervals; the intersection is sharper than either alone. When conditioning on strata (as in the OHE, where $\texttt{numhh\_list}$ defines household-size strata), stratum-specific bounds can be intersected with aggregate-level bounds derived from different assumptions.

**Inference on intersection bounds.** Because $\max_k \ell_k$ involves a maximum of estimated quantities, standard delta-method inference fails. Chernozhukov, Lee, and Rosen (2013) develop confidence regions for intersection bounds using moment inequality methods. The key insight is that the confidence region for the identified set $[\tau_L, \tau_U]$ should cover the *set*, not the *point*—a distinction that matters for reporting. A 95% confidence interval $[\hat{\tau}_L - c_\alpha \hat{\sigma}_L, \hat{\tau}_U + c_\alpha \hat{\sigma}_U]$ is constructed to contain the identified set with probability $1 - \alpha$, which is more conservative than a standard CI because it accounts for set-valued uncertainty.

## 33.6 Sensitivity Analysis as Partial Identification

The connection to Chapters 31–32 is exact. A sensitivity parameter $\Gamma$ (Rosenbaum) or $\Lambda$ (marginal sensitivity model) defines a set of allowable confounders. For each value of the sensitivity parameter, the identified set for $\tau$ is an interval. As the sensitivity parameter varies from its "no unmeasured confounding" value to its extreme, the identified set expands from a point to the no-assumption Manski bounds.

Formally, let $\mathcal{A}(\Gamma)$ denote the assumption set that restricts unmeasured confounding to level $\Gamma$. Then:

$$\Theta(\mathcal{A}(1)) = \{\hat{\tau}_{obs}\} \quad \text{(point, at no confounding)}$$
$$\Theta(\mathcal{A}(\infty)) = [\tau_L^{Manski}, \tau_U^{Manski}] \quad \text{(Manski bounds, at infinite confounding)}$$

The sensitivity curve $\Gamma \mapsto \Theta(\mathcal{A}(\Gamma))$ is the partial identification trajectory. Reporting this curve is more informative than reporting either a single point estimate or a single robustness check. The "breakdown point" $\Gamma^*$ at which zero enters the identified set is directly interpretable: it is the minimum confounding that could explain away the estimated effect.

This framing also clarifies that every point identification strategy implicitly claims $\Gamma = 1$ (or the equivalent in the chosen parameterization). Partial identification makes this implicit claim explicit and asks what price in assumption strength one is paying for point identification.

## Python: Manski, MTR+MTS, and Balke-Pearl Bounds on OHE Catastrophic Expenditure

```python
"""
Chapter 33: Partial Identification and Bounds
Oregon Health Insurance Experiment
Outcome: catastrophic_exp_inp (1 = catastrophic inpatient expenditure)
Treatment: ohp_all_ever_admin (1 = enrolled in Medicaid)
Instrument: selected (1 = won lottery)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog
from scipy import stats

# ── 1. Load OHE data ──────────────────────────────────────────────────────────
# Data available at https://data.nber.org/oregon/
# Using the 12-month survey + admin file merged on person_id
# For reproducibility we read the publicly archived CSV; adjust path as needed.

try:
    df = pd.read_csv("~/data/ohe/oregonhie_descriptive_vars.csv")
    df_surv = pd.read_csv("~/data/ohe/oregonhie_survey12m_vars.csv")
    df = df.merge(df_surv, on="person_id", how="inner")

    Z = df["selected"].values.astype(float)          # lottery win
    D = df["ohp_all_ever_admin"].values.astype(float) # ever enrolled
    Y = df["catastrophic_exp_inp"].values.astype(float)
    strata = df["numhh_list"].values                  # household size strata

    # Drop observations with missing outcome
    mask = ~np.isnan(Y) & ~np.isnan(D) & ~np.isnan(Z)
    Z, D, Y, strata = Z[mask], D[mask], Y[mask], strata[mask]
    print(f"N = {len(Y)}, E[D] = {D.mean():.3f}, E[Y] = {Y.mean():.3f}")

except FileNotFoundError:
    # ── Simulation fallback matching OHE moments ──────────────────────────────
    rng = np.random.default_rng(42)
    N = 12_229
    numhh = rng.choice([1, 2, 3], size=N, p=[0.6, 0.3, 0.1])
    Z = rng.binomial(1, 0.5, N)
    # First stage: P(D=1|Z=1) ~ 0.26, P(D=1|Z=0) ~ 0.14
    p_D_Z1 = 0.26; p_D_Z0 = 0.14
    D = np.where(Z == 1,
                 rng.binomial(1, p_D_Z1, N),
                 rng.binomial(1, p_D_Z0, N))
    # Outcome: catastrophic expenditure, ~0.05 overall, insurance reduces by ~0.01
    # Latent index with positive selection
    U = rng.normal(0, 1, N)
    logit_Y = -3.0 + 0.3 * D - 0.6 * (U > 0.5)  # positive selection: high-U selects D=1
    Y = rng.binomial(1, 1 / (1 + np.exp(-logit_Y)), N).astype(float)
    strata = numhh
    print(f"Simulated N={N}, E[D]={D.mean():.3f}, E[Y]={Y.mean():.3f}")


# ── 2. Manski No-Assumption Bounds ───────────────────────────────────────────

y_min, y_max = 0.0, 1.0   # binary outcome

def manski_bounds(Y, D, y_min=0.0, y_max=1.0):
    """Sharp no-assumption bounds on ATE."""
    p1 = D.mean()
    p0 = 1 - p1
    EY1_obs = (Y * D).mean()    # E[Y * 1(D=1)]
    EY0_obs = (Y * (1-D)).mean() # E[Y * 1(D=0)]

    # E[Y(1)] bounds
    EY1_lo = EY1_obs + y_min * p0
    EY1_hi = EY1_obs + y_max * p0

    # E[Y(0)] bounds
    EY0_lo = EY0_obs + y_min * p1
    EY0_hi = EY0_obs + y_max * p1

    # ATE = E[Y(1)] - E[Y(0)]
    ate_lo = EY1_lo - EY0_hi
    ate_hi = EY1_hi - EY0_lo

    return {
        "EY1": (EY1_lo, EY1_hi),
        "EY0": (EY0_lo, EY0_hi),
        "ATE": (ate_lo, ate_hi),
        "width": ate_hi - ate_lo,
    }

manski = manski_bounds(Y, D)
print(f"\nManski no-assumption ATE bounds: [{manski['ATE'][0]:.4f}, {manski['ATE'][1]:.4f}]")
print(f"  Width: {manski['width']:.4f}")


# ── 3. MTR Bounds (insurance weakly reduces catastrophic expenditure) ─────────

def mtr_bounds(Y, D, y_min=0.0, y_max=1.0):
    """MTR: Y(1) <= Y(0) (insurance weakly reduces catastrophic expenditure).
    
    Note: for a bad outcome (catastrophic expense), MTR means treatment weakly
    REDUCES outcome: Y(1) <= Y(0), so ATE <= 0.
    """
    manski_b = manski_bounds(Y, D, y_min, y_max)
    # MTR (treatment reduces bad outcome) forces ATE <= 0
    ate_lo = manski_b["ATE"][0]
    ate_hi = min(0.0, manski_b["ATE"][1])
    return {"ATE": (ate_lo, ate_hi), "width": ate_hi - ate_lo}

mtr = mtr_bounds(Y, D)
print(f"\nMTR bounds (insurance weakly reduces catastrophe):")
print(f"  ATE bounds: [{mtr['ATE'][0]:.4f}, {mtr['ATE'][1]:.4f}]")
print(f"  Width: {mtr['width']:.4f}")


# ── 4. MTR + MTS Bounds ───────────────────────────────────────────────────────

def mtr_mts_bounds(Y, D, y_min=0.0, y_max=1.0):
    """
    MTR: Y(1) <= Y(0)  (treatment weakly reduces bad outcome)
    MTS: E[Y(d)|D=1] <= E[Y(d)|D=0]  (positive selection into treatment
         means treated have weakly lower potential catastrophic expenditure)
    
    Under MTS: E[Y(1)|D=0] >= E[Y(1)|D=1] = E[Y|D=1]  (lower potential outcome
    for treated, so counterfactual for untreated is bounded below by E[Y|D=1])
    
    Combined upper bound: E[Y(1)] - E[Y(0)] <= 0  (from MTR)
    Combined lower bound: difference-in-means (under MTS, naive DIM understates
    the beneficial effect for bad outcomes)
    """
    EY1 = Y[D == 1].mean()   # E[Y|D=1]
    EY0 = Y[D == 0].mean()   # E[Y|D=0]
    p1 = D.mean()
    p0 = 1 - p1

    # Under MTS: E[Y(1)|D=0] >= E[Y|D=1] and E[Y(0)|D=1] <= E[Y|D=0]
    # E[Y(1)] = E[Y|D=1]*p1 + E[Y(1)|D=0]*p0 >= EY1*p1 + EY1*p0 = EY1
    # E[Y(0)] = E[Y|D=0]*p0 + E[Y(0)|D=1]*p1 <= EY0*p0 + EY0*p1 = EY0
    # So ATE = E[Y(1)] - E[Y(0)] >= EY1 - EY0 (the naive DIM is a lower bound)
    ate_lo_mts = EY1 - EY0   # lower bound from MTS alone (for bad outcome)

    # MTR forces ATE <= 0
    ate_lo = ate_lo_mts      # DIM (negative for beneficial treatment)
    ate_hi = 0.0             # MTR

    return {
        "ATE": (ate_lo, ate_hi),
        "width": ate_hi - ate_lo,
        "dim": EY1 - EY0,
    }

mtr_mts = mtr_mts_bounds(Y, D)
print(f"\nMTR + MTS bounds:")
print(f"  ATE bounds: [{mtr_mts['ATE'][0]:.4f}, {mtr_mts['ATE'][1]:.4f}]")
print(f"  Width: {mtr_mts['width']:.4f}")
print(f"  (Naive DIM = {mtr_mts['dim']:.4f})")


# ── 5. IV Wald Estimate (benchmark) ──────────────────────────────────────────

def wald_iv(Y, D, Z):
    """Wald estimator with delta-method SE."""
    # First stage
    fs_num = Y[Z==1].mean() - Y[Z==0].mean()
    fs_den = D[Z==1].mean() - D[Z==0].mean()
    late = fs_num / fs_den

    # Delta-method SE
    n1 = (Z == 1).sum(); n0 = (Z == 0).sum()
    var_rf = Y[Z==1].var() / n1 + Y[Z==0].var() / n0
    var_fs = D[Z==1].var() / n1 + D[Z==0].var() / n0
    cov_rf_fs = (np.cov(Y[Z==1], D[Z==1])[0,1] / n1 +
                 np.cov(Y[Z==0], D[Z==0])[0,1] / n0)
    beta = fs_num / fs_den
    var_late = (1/fs_den**2) * (var_rf - 2*beta*cov_rf_fs + beta**2*var_fs)
    se = np.sqrt(var_late)
    return late, se

late, late_se = wald_iv(Y, D, Z)
late_ci = (late - 1.96*late_se, late + 1.96*late_se)
print(f"\nWald LATE (IV): {late:.4f} (SE={late_se:.4f})")
print(f"  95% CI: [{late_ci[0]:.4f}, {late_ci[1]:.4f}]")


# ── 6. Balke-Pearl IV Bounds via Linear Programming ──────────────────────────

def balke_pearl_bounds(Y, D, Z):
    """
    Balke-Pearl bounds via LP.
    Variables: pi[y1,y0,d1,d0] for y1,y0,d1,d0 in {0,1} -- 16 response types
    y1 = Y(1), y0 = Y(0), d1 = D(1), d0 = D(0)
    
    Index: i = 8*y1 + 4*y0 + 2*d1 + d0  (0..15)
    
    Observed cell probabilities: P(Y=y, D=d | Z=z), 8 values
    Constraints: sum_types consistent with each observed cell probability
    """
    # Estimate observed cell probabilities
    cells = {}
    for z in [0, 1]:
        mask_z = Z == z
        n_z = mask_z.sum()
        for y in [0, 1]:
            for d in [0, 1]:
                cells[(y, d, z)] = ((Y[mask_z] == y) & (D[mask_z] == d)).sum() / n_z

    # 16 response types: index = 8*y1 + 4*y0 + 2*d1 + d0
    def idx(y1, y0, d1, d0):
        return 8*y1 + 4*y0 + 2*d1 + d0

    # ATE = E[Y(1)] - E[Y(0)] = sum_types pi[type] * (y1 - y0)
    ate_coef = np.zeros(16)
    for y1 in [0,1]:
        for y0 in [0,1]:
            for d1 in [0,1]:
                for d0 in [0,1]:
                    ate_coef[idx(y1,y0,d1,d0)] = y1 - y0

    # Equality constraints from observables
    # P(Y=y, D=d|Z=z) = sum_{types: D(z)=d, Y(D(z))=y} pi[type]
    A_eq = []
    b_eq = []
    for z in [0, 1]:
        for y in [0, 1]:
            for d in [0, 1]:
                row = np.zeros(16)
                for y1 in [0,1]:
                    for y0 in [0,1]:
                        for d1 in [0,1]:
                            for d0 in [0,1]:
                                # D(z)=d and Y(D(z))=y
                                d_z = d1 if z == 1 else d0
                                y_dz = y1 if d_z == 1 else y0
                                if d_z == d and y_dz == y:
                                    row[idx(y1,y0,d1,d0)] += 1
                A_eq.append(row)
                b_eq.append(cells[(y, d, z)])

    # Sum-to-one constraint
    A_eq.append(np.ones(16))
    b_eq.append(1.0)

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    bounds_vars = [(0, 1)] * 16

    # Minimize ATE (lower bound)
    res_lo = linprog(ate_coef, A_eq=A_eq, b_eq=b_eq, bounds=bounds_vars,
                     method='highs')
    # Maximize ATE (upper bound)
    res_hi = linprog(-ate_coef, A_eq=A_eq, b_eq=b_eq, bounds=bounds_vars,
                     method='highs')

    if res_lo.status == 0 and res_hi.status == 0:
        return res_lo.fun, -res_hi.fun
    else:
        return None, None

bp_lo, bp_hi = balke_pearl_bounds(Y, D, Z)
print(f"\nBalke-Pearl IV bounds (no monotonicity): [{bp_lo:.4f}, {bp_hi:.4f}]")
print(f"  Width: {bp_hi - bp_lo:.4f}")
print(f"  LATE point estimate contained in bounds: "
      f"{bp_lo <= late <= bp_hi}")


# ── 7. Bootstrap CIs for Manski and MTR+MTS Bounds ───────────────────────────

def bootstrap_bounds(Y, D, bound_fn, B=500, alpha=0.05, **kwargs):
    """Percentile bootstrap for bound endpoints."""
    rng = np.random.default_rng(0)
    lows, highs = [], []
    n = len(Y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        b = bound_fn(Y[idx], D[idx], **kwargs)
        lows.append(b["ATE"][0])
        highs.append(b["ATE"][1])
    ci_lo = (np.percentile(lows, 100*alpha/2), np.percentile(lows, 100*(1-alpha/2)))
    ci_hi = (np.percentile(highs, 100*alpha/2), np.percentile(highs, 100*(1-alpha/2)))
    return ci_lo, ci_hi

print("\nBootstrapping bound CIs (B=500)...")
manski_ci_lo, manski_ci_hi   = bootstrap_bounds(Y, D, manski_bounds, B=500)
mtrmts_ci_lo, mtrmts_ci_hi   = bootstrap_bounds(Y, D, mtr_mts_bounds, B=500)
print(f"  Manski lower bound 95% CI: [{manski_ci_lo[0]:.4f}, {manski_ci_lo[1]:.4f}]")
print(f"  Manski upper bound 95% CI: [{manski_ci_hi[0]:.4f}, {manski_ci_hi[1]:.4f}]")


# ── 8. Visualization ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Bound intervals under increasing assumptions
labels = [
    "No assumptions\n(Manski)",
    "MTR only",
    "MTR + MTS",
    "Balke-Pearl\n(IV, no mono.)",
    "Wald LATE\n(IV + mono.)",
]
lows  = [manski["ATE"][0], mtr["ATE"][0], mtr_mts["ATE"][0],
         bp_lo, late - 1.96*late_se]
highs = [manski["ATE"][1], mtr["ATE"][1], mtr_mts["ATE"][1],
         bp_hi, late + 1.96*late_se]
colors = ["#d73027", "#fc8d59", "#fee090", "#4575b4", "#313695"]

ax = axes[0]
for i, (lab, lo, hi, col) in enumerate(zip(labels, lows, highs, colors)):
    ax.barh(i, hi - lo, left=lo, height=0.5, color=col, alpha=0.85)
    ax.text(hi + 0.003, i, f"[{lo:.3f}, {hi:.3f}]", va='center', fontsize=8)

ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("ATE on Catastrophic Inpatient Expenditure")
ax.set_title("Panel A: Identified Sets Under\nIncreasing Assumptions")
ax.set_xlim(-1.05, 0.7)

# Panel B: Bound width vs. assumption strength
ax2 = axes[1]
assumption_strength = [0, 1, 2, 3, 4]
widths = [manski["width"], mtr["width"], mtr_mts["width"],
          bp_hi - bp_lo, late_se * 2 * 1.96]
ax2.plot(assumption_strength, widths, 'o-', color='#2166ac', linewidth=2,
         markersize=8)
for i, (lab, w) in enumerate(zip(["None", "MTR", "MTR+MTS", "IV (BP)", "IV+Mono"], widths)):
    ax2.annotate(f"{lab}\n({w:.3f})", (i, w),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
ax2.set_xticks(assumption_strength)
ax2.set_xticklabels(["None", "MTR", "MTR\n+MTS", "Balke-\nPearl", "LATE"],
                    fontsize=9)
ax2.set_ylabel("Bound Width")
ax2.set_title("Panel B: Bound Width vs.\nAssumption Strength")
ax2.set_ylim(0, 1.2)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("ch33_partial_identification_bounds.png", dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: ch33_partial_identification_bounds.png")


# ── 9. Summary Table ──────────────────────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 33.1: Identified Sets for ATE of Medicaid on Catastrophic")
print("           Inpatient Expenditure (Oregon Health Insurance Experiment)")
print("="*70)
print(f"{'Method':<30} {'Lower':>8} {'Upper':>8} {'Width':>8} {'Assumptions'}")
print("-"*70)
rows = [
    ("Manski (no assumptions)",     manski["ATE"][0], manski["ATE"][1],  "None"),
    ("MTR",                          mtr["ATE"][0],    mtr["ATE"][1],     "Y(1)≤Y(0)"),
    ("MTR + MTS",                    mtr_mts["ATE"][0],mtr_mts["ATE"][1],"MTR + pos. sel."),
    ("Balke-Pearl (IV, no mono.)",   bp_lo,            bp_hi,             "IV indep + excl."),
    ("LATE 95% CI",                  late_ci[0],       late_ci[1],        "IV + monotonicity"),
]
for name, lo, hi, assump in rows:
    print(f"{name:<30} {lo:>8.4f} {hi:>8.4f} {hi-lo:>8.4f}  {assump}")
print("="*70)
```

## 33.7 Inference on Partially Identified Parameters

Reporting a bound estimate without uncertainty quantification is incomplete. Two distinct inferential targets exist:

1. **Confidence set for the identified set** $[\tau_L, \tau_U]$: an interval $[L_n, U_n]$ that covers the true identified set with probability $1-\alpha$. This requires $P(L_n \leq \tau_L \text{ and } U_n \geq \tau_U) \geq 1-\alpha$.

2. **Confidence set for the true parameter** $\tau^*$: an interval that covers the true value (which may be anywhere in the identified set) with probability $1-\alpha$.

Target 2 is more conservative. If the identified set $[\tau_L, \tau_U]$ has width $w$ and uncertainty in each endpoint is $\pm c_\alpha \sigma$, the CS for the true parameter has width $w + 2c_\alpha \sigma$, while the CS for the identified set has width $w + 2c_\alpha \sigma$ but centered differently. In practice, when the identified set is wide, the distinction matters less than the choice of assumptions.

For Manski bounds with binary outcomes, each bound endpoint is a sample average, so the CLT applies directly. The bootstrap (as implemented above) is consistent for the endpoints. For intersection bounds involving maxima of estimated quantities, Chernozhukov et al. (2013) provide a moment-inequality based approach that avoids the conservative Bonferroni correction.

**Stoye's minimax regret bounds.** An alternative to set-valued inference is the minimax regret approach (Stoye, 2012): among all decision rules mapping data to a decision $\in \{0,1\}$ (e.g., adopt or reject policy), choose the rule that minimizes maximum regret over the identified set. This collapses the set-valued output to a point decision while respecting the fundamental uncertainty.

## Summary

- **Partial identification** replaces the binary identified/not-identified framing with a continuous one: the identified set $\Theta(\mathcal{A})$ shrinks as assumptions $\mathcal{A}$ strengthen, reaching a point only when the data and assumptions together over-determine the parameter.

- **Manski no-assumption bounds** on the ATE have width equal to $y_{\max} - y_{\min}$ regardless of sample size; they are sharp and serve as a baseline against which all other assumptions are measured.

- **MTR and MTS** are sign restrictions on individual response and population selection; together they recover $[0, \hat{\tau}_{OLS}]$ (for a beneficial treatment on a good outcome), demonstrating that the naive OLS estimate is the sharp upper bound under plausible assumptions rather than an uninterpretable biased estimate.

- **Balke-Pearl IV bounds** achieve point identification without monotonicity using only IV independence and exclusion; they are computed via linear programming over the joint distribution of response types and are always weakly wider than the LATE CI—the gap measures exactly what the monotonicity assumption is worth.

- **Intersection bounds** sharpen identified sets by combining multiple constraint sets; inference requires moment-inequality methods rather than the delta method because maxima of estimated quantities have non-standard distributions.

- **Sensitivity analysis is partial identification**: the sensitivity parameter traces a path through the space of assumption sets, and the boundary value at which zero enters the identified set is the breakdown point—a more informative summary than a single sensitivity p-value.

- Reporting width reduction across the assumption ladder (Table 33.1, Panel B) makes transparent the price each assumption charges in terms of credibility versus informativeness; the analyst and reader can then judge whether the price is worth paying.

## Further Reading

- **Manski, C. F. (1990).** "Nonparametric Bounds on Treatment Effects." *American Economic Review Papers and Proceedings*, 80(2), 319–323. The founding paper. Establishes that partial identification is a rigorous research program, not a confession of failure.

- **Manski, C. F., and Pepper, J. V. (2000).** "Monotone Instrumental Variables: With an Application to the Returns to Schooling." *Econometrica*, 68(4), 997–1010. Develops MTR and MTS formally; shows how sign restrictions on selection and response tighten bounds without point identification.

- **Balke, A., and Pearl, J. (1997).** "Bounds on Treatment Effects from Studies with Imperfect Compliance." *JASA*, 92(439), 1171–1176. The LP formulation of IV bounds without monotonicity; demonstrates that the Wald estimator requires assumptions well beyond IV independence.

- **Chernozhukov, V., Lee, S., and Rosen, A. M. (2013).** "Intersection Bounds: Estimation and Inference." *Econometrica*, 81(2), 667–737. Provides the asymptotic theory and inference procedures for bounds characterized by moment inequalities; essential for empirical implementation beyond the Manski binary case.

- **Mogstad, M., Santos, A., and Torgovitsky, A. (2018).** "Using Instrumental Variables for Inference about Policy Relevant Treatment Parameters." *Econometrica*, 86(5), 1589–1619. Extends IV bounds to policy-relevant estimands beyond LATE using the marginal treatment effect framework; the `pyvmte` package implements their approach.

- **Stoye, J. (2012).** "Minimax Regret Treatment Choice with Covariates or with Limited Validity of Experiments." *Journal of Econometrics*, 166(1), 138–156. Bridges partial identification and decision theory; shows how to make optimal binary decisions when only a set is identified, without imposing prior distributions over the identified set.
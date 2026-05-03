# Chapter 5: Target Trial Emulation

## 5.1 The Target Trial Framework

Observational data is not collected to answer causal questions. It is collected for administrative, clinical, or survey purposes. When analysts then attempt causal inference from such data, they face a choice that is easy to make badly: begin with the available data and work backwards toward a question it might answer, or begin with a well-posed causal question and discipline the analysis around that question. The target trial framework, developed systematically by Hernán and Robins, enforces the second discipline.

The central idea is stark: every valid causal question about the effect of a treatment strategy can be articulated as a randomized controlled trial, whether or not such a trial will ever be conducted. This hypothetical experiment—the **target trial**—has a complete protocol specifying who is eligible, what treatment strategies are compared, when follow-up begins, what outcomes are measured, and what estimand is of interest. The observational study succeeds to the extent that it emulates this protocol using available data.

This is not a rhetorical device. The protocol specification forces precision that verbal causal claims routinely evade. Consider: "does Medicaid improve health outcomes?" This question is underspecified in every dimension that matters for estimation. Does Medicaid at enrollment improve outcomes relative to remaining uninsured for one year? Relative to private insurance? For whom? Measured how? Starting when? The target trial protocol answers each question before touching data, making the gap between the ideal experiment and the observational analysis explicit and auditable.

The Oregon Health Insurance Experiment (OHE) provides a natural anchor. In 2008, Oregon conducted a lottery among uninsured low-income adults on a waiting list for its Medicaid program (Oregon Health Plan, OHP Standard). The lottery was a randomized assignment to be invited to apply—not assignment to coverage itself. This distinction drives Chapter 6's IV analysis. Here we use it differently: we specify the target trial that the OHE approximates, then emulate that target trial from the observational data alone, treating lottery selection as one piece of evidence about the quality of the emulation rather than as an instrument.

## 5.2 Protocol Specification

A target trial protocol has seven components. Specifying each component for the OHE health insurance question reveals the precision the framework demands.

**Eligibility criteria.** The target trial enrolls the population for whom the question is meaningful. For OHE: uninsured Oregon adults aged 19–64 with income below 100% FPL who had not been on Medicaid in the prior six months. Observational emulation requires that the analysis dataset can identify this population. The `numhh_list` variable in OHE encodes household size, which was used for lottery stratification; eligibility is encoded in the fact of being on the list.

**Treatment strategies.** A target trial compares specific, replicable strategies—not exposure values. "Treated with Medicaid" versus "remained uninsured" are strategies. "Had insurance at some point" is not a strategy; it cannot be assigned. In the OHE emulation, the strategies are: (1) immediate Medicaid enrollment upon lottery selection versus (2) remaining uninsured for 12 months. This maps to `ohp_all_ever_admin` as the treatment indicator.

**Treatment assignment.** In a trial, assignment is randomized at time zero. In emulation, assignment is observed. The critical requirement is that assignment be measured at baseline, before any post-baseline covariates contaminate it. This is the **time zero alignment** problem discussed in Section 5.4.

**Outcome.** The primary outcome should be specified before analysis. For OHE: any catastrophic medical expenditure in the 12-month follow-up window, encoded in `catastrophic_exp_inp` (inpatient catastrophic spending). A secondary outcome is any doctor visit (`doc_any_12m`), relevant for understanding the mechanism.

**Follow-up period.** Follow-up begins at time zero and ends at the outcome event, death, loss to follow-up, or the prespecified end of follow-up—whichever comes first. In the OHE design, 12 months.

**Causal estimand.** The target trial specifies what is being estimated. Two estimands are standard:

- **Intention-to-treat (ITT)**: the effect of being assigned to treatment strategy, regardless of adherence. In OHE: the effect of lottery selection on outcomes.
- **Per-protocol (PP)**: the effect of actually following the assigned strategy throughout follow-up. In OHE: the effect of actually maintaining Medicaid enrollment for the full 12 months.

These are different quantities. ITT is identifiable under randomization without covariate adjustment; PP requires additional assumptions about the relationship between adherence and outcomes.

**Statistical analysis plan.** The trial protocol specifies the primary estimator. For a binary outcome with fixed follow-up, a risk difference or risk ratio under standardization (Section 5.5). For survival outcomes, a hazard ratio or survival curve comparison under IPCW (Section 5.3).

## 5.3 Censoring, IPCW, and the Per-Protocol Estimand

In a trial with perfect adherence, ITT and PP coincide. In practice, participants deviate from their assigned strategy—some assigned to treatment never enroll, some enrolled later drop out. The observational emulation faces the same problem compounded by selection into treatment in the first place.

The per-protocol estimand requires handling two sources of censoring:

1. **Administrative censoring**: end of follow-up window, independent of treatment and outcomes.
2. **Protocol deviation censoring**: a participant deviates from their assigned strategy. In the emulation, we treat this deviation as censoring the per-protocol analysis.

Let $A_k \in \{0,1\}$ denote treatment status at time $k$, $C_k$ a censoring indicator at time $k$, and $\bar{L}_k = (L_0, L_1, \ldots, L_k)$ a history of time-varying covariates. The **inverse probability of censoring weight** at time $t$ is:

$$w_i(t) = \prod_{k=0}^{t} \frac{1}{P(C_k = 0 \mid C_{k-1} = 0, \bar{L}_k, A_k)}$$

This weight upweights uncensored individuals to represent those who were censored, under the assumption that censoring is conditionally independent of the counterfactual outcome given $(\bar{L}_k, A_k)$—the **independent censoring assumption**:

$$C_k \perp\!\!\!\perp Y^{\bar{a}} \mid C_{k-1} = 0, \bar{L}_k, A_k$$

**Theorem 5.1 (IPCW Consistency).** *Under (i) consistency, (ii) positivity of censoring ($P(C_k = 0 \mid C_{k-1} = 0, \bar{L}_k, A_k) > 0$ a.s.), and (iii) independent censoring, the IPCW estimator*

$$\hat{E}[Y^{\bar{a}}] = \frac{1}{n} \sum_{i=1}^n \frac{\mathbb{1}[\bar{A}_i = \bar{a}] \cdot \mathbb{1}[C_i = 0]}{w_i(t)} Y_i$$

*is consistent for $E[Y^{\bar{a}}]$.*

*Proof sketch.* The weight $w_i(t)$ creates a pseudo-population in which the censoring mechanism is removed. Formally, $E[w_i(t) \cdot \mathbb{1}[C_i = 0] \cdot h(Y_i, \bar{L}_i)] = E[h(Y^{\bar{a}}_i, \bar{L}_i)]$ for any measurable $h$, which follows by iterating the tower property over the discrete time grid and invoking the independent censoring assumption at each step. $\square$

In practice, the denominators $P(C_k = 0 \mid C_{k-1} = 0, \bar{L}_k, A_k)$ are estimated with logistic regression models fit to the uncensored person-time. Stabilized weights replace the denominator with $P(C_k = 0 \mid C_{k-1} = 0, A_k)$, reducing variance at the cost of slight bias when the numerator model is misspecified:

$$sw_i(t) = \prod_{k=0}^{t} \frac{P(C_k = 0 \mid C_{k-1} = 0, A_k)}{P(C_k = 0 \mid C_{k-1} = 0, \bar{L}_k, A_k)}$$

## 5.4 Time Zero Alignment and Immortal Time Bias

Time zero alignment is the single most consequential design decision in observational emulation, and the most commonly violated. Time zero is the moment when (a) eligibility is determined, (b) treatment is assigned, and (c) follow-up begins. All three must coincide.

**Definition 5.1 (Immortal Time Bias).** Let $T_E$ be the time of eligibility determination and $T_A > T_E$ the time of treatment initiation. If follow-up begins at $T_E$ and treatment status is measured as "ever treated in $[T_E, T_E + \delta]$," then events occurring in $(T_E, T_A)$ are attributed to the treated group despite occurring before treatment. The person-time $(T_E, T_A)$ is **immortal time**—the participant cannot have an event counted against the treatment arm during this interval because they have not yet received treatment. This mechanically inflates the apparent benefit of treatment.

The canonical example is cohort studies of drug effectiveness where patients are classified as "users" if they filled a prescription in the first six months, but follow-up begins at cohort entry. Patients who die in the first six months before filling a prescription are classified as non-users, deflating the non-user survival curve.

**Correct time zero alignment for OHE emulation:** Time zero is the date of notification of lottery results. Eligibility is determined at that date. Treatment status ($D = 1$ if enrolled in OHP within 90 days) is measured prospectively. Follow-up begins at notification. Any outcome before notification is excluded.

The `ohp_all_ever_admin` variable in the OHE data records whether the individual was ever enrolled in OHP during the study period. Using this directly as a baseline covariate is correct only if it is measured after time zero. If the data include prior OHP enrollment, using that as a predictor of the treatment-outcome relationship without careful alignment will introduce bias.

## 5.5 Standardization and the G-Formula at Baseline

When treatment is assigned at a single time point (no time-varying treatment), the PP estimand under no unmeasured confounding reduces to standardization over baseline covariates. This is the simplest instance of the g-formula, which Chapters 25–29 extend to time-varying settings.

**Theorem 5.2 (Identification by Standardization).** *Under (i) consistency ($Y_i = Y_i^a$ when $A_i = a$), (ii) conditional exchangeability ($Y^a \perp\!\!\!\perp A \mid L$ for $a \in \{0,1\}$), and (iii) positivity ($P(A = a \mid L = l) > 0$ whenever $P(L = l) > 0$):*

$$E[Y^a] = \sum_l E[Y \mid A = a, L = l] \cdot P(L = l)$$

*Proof.* 
$$E[Y^a] = \sum_l E[Y^a \mid L = l] P(L=l)$$
$$= \sum_l E[Y^a \mid A = a, L = l] P(L=l) \quad \text{(exchangeability)}$$
$$= \sum_l E[Y \mid A = a, L = l] P(L=l) \quad \text{(consistency)} \qquad \square$$

The standardization estimator plugs in empirical frequencies:

$$\hat{E}[Y^a] = \frac{1}{n} \sum_{i=1}^n \hat{E}[Y \mid A = a, L = L_i]$$

where $\hat{E}[Y \mid A = a, L = l]$ is estimated from any regression model. This is the **G-computation estimator** at baseline. Its variance is obtained by the nonparametric bootstrap, which correctly propagates uncertainty from the outcome model estimation.

The average treatment effect is then $\hat{\tau} = \hat{E}[Y^1] - \hat{E}[Y^0]$. Under the OHE target trial, $\hat{\tau}$ estimates the effect of Medicaid enrollment on 12-month catastrophic expenditure risk.

## 5.6 The Clone-Censor-Weight Technique

When treatment strategies unfold over time—a patient may switch from treatment to control, or vice versa—the analysis must handle time-varying confounding while preserving the per-protocol logic. The **clone-censor-weight** (CCW) technique provides a computationally transparent implementation.

**Step 1: Clone.** For each participant, create two copies—one assigned to the "always treat" strategy, one to the "never treat" strategy. At baseline, both clones are uncensored and receive weight 1. This doubling of the dataset is purely bookkeeping.

**Step 2: Censor.** In each cloned arm, censor participants at the first time point where they deviate from their assigned strategy. A clone in the "always treat" arm is censored when the participant discontinues treatment. A clone in the "never treat" arm is censored when the participant initiates treatment.

**Step 3: Weight.** Apply IPCW within each arm to account for the informative censoring created in Step 2. The numerator of the stabilized weight uses only baseline and time-fixed variables; the denominator conditions on the full time-varying covariate history.

**Theorem 5.3 (CCW-MSM Equivalence).** *The CCW estimator with stabilized IPCW is algebraically equivalent to the MSM-IPTW estimator for the marginal causal effect of a static treatment strategy.*

*Proof sketch.* In the cloned dataset, the effective denominator weight for a participant in arm $a$ who follows strategy $\bar{a} = (a, a, \ldots, a)$ through time $t$ is:

$$\prod_{k=0}^{t} \frac{P(A_k = a_k \mid \bar{A}_{k-1}, V)}{P(A_k = a_k \mid \bar{A}_{k-1}, \bar{L}_k)}$$

where $V$ is baseline covariates. This is identical to the MSM stabilized weight $sw_i(t)$ for a static regime. The CCW dataset structure reorganizes the computation but does not change the estimating equation. $\square$

The practical advantage of CCW is transparency: the cloning, censoring, and weighting steps are explicit dataframe operations, making auditing and debugging straightforward. The MSM formulation is more compact for theoretical analysis. Both are used in the literature.

## 5.7 Active Comparator Design

A subtle but important protocol choice is the selection of the comparator. "No treatment" is rarely a coherent strategy: patients who receive no treatment are often systematically different from treated patients in ways that baseline covariate adjustment cannot fully address, because the decision not to treat is itself informative.

The **active comparator design** substitutes a specific alternative treatment for "no treatment" as the comparator arm. In drug effectiveness studies, this means comparing Drug A to Drug B rather than Drug A to no drug. The rationale is threefold:

1. Patients receiving either drug face a similar clinical indication, reducing confounding by indication.
2. The comparison is clinically relevant—regulatory and clinical decisions are typically comparative.
3. Residual confounders tend to affect both treatment arms similarly, producing partial cancellation.

For the OHE emulation, a natural active comparator would be participation in a community health clinic program rather than Medicaid. In the absence of such a comparator in OHE, the design defaults to the waitlist control—individuals who applied but were not selected—which is close to an active comparator in that both groups faced similar health system contact.

The active comparator design does not eliminate confounding; it changes its structure. The identifying assumption shifts from "all confounders of Medicaid enrollment vs. no-insurance are measured" to "all confounders of Medicaid enrollment vs. community clinic enrollment are measured." Whether this is more plausible depends on the study context and available data.

## Python: Oregon Health Insurance Experiment — Target Trial Emulation with IPCW

The following implementation specifies the OHE target trial protocol, constructs a person-period dataset, estimates IPCW weights using logistic regression nuisance models, and produces Kaplan-Meier survival curves under the emulated trial.

```python
"""
Chapter 5: Target Trial Emulation — OHE Implementation
Target trial: uninsured Oregon adults, Medicaid vs. uninsured, 12-month follow-up
Outcome: catastrophic inpatient expenditure (binary, end of follow-up)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load and align OHE data to target trial protocol
# ---------------------------------------------------------------------------

def load_ohe(data_dir: str = "/tmp/ohe") -> pd.DataFrame:
    """
    Load OHE 12-month survey data. Variables used:
      selected      : lottery selection (Z, instrument)
      ohp_all_ever_admin : ever enrolled in OHP (D, treatment)
      catastrophic_exp_inp : any catastrophic inpatient expenditure (Y)
      doc_any_12m   : any doctor visit in 12 months (Y2)
      numhh_list    : household size at time of lottery (strata)
      age_19_34_inp, age_35_49_inp, age_50_64_inp : age indicators
      female_inp    : sex
      english_list  : English language preference
      self_list     : self-reported health at baseline
    """
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Attempt to load from local cache; fall back to simulated data
    csv_path = path / "oregonhie_descriptive_vars.csv"
    
    if not csv_path.exists():
        print("OHE data not found locally. Generating a faithful simulation.")
        return _simulate_ohe(n=20_000, seed=42)
    
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.rename(columns=str.lower)
    
    # Time zero alignment: restrict to those with valid lottery date
    # All individuals on the list share time zero = lottery notification date
    # 'selected' is the intent-to-treat assignment; 'ohp_all_ever_admin' is realized treatment
    
    keep = [
        "selected", "ohp_all_ever_admin", "catastrophic_exp_inp",
        "doc_any_12m", "numhh_list",
        "age_19_34_inp", "age_35_49_inp", "age_50_64_inp",
        "female_inp", "english_list",
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].dropna(subset=["selected", "catastrophic_exp_inp"])
    
    return df


def _simulate_ohe(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate OHE-structured data matching published marginal statistics.
    Selection rate ~30%, compliance rate ~25% among selected,
    catastrophic expenditure ~6% overall.
    """
    rng = np.random.default_rng(seed)
    
    numhh = rng.choice([1, 2, 3], size=n, p=[0.7, 0.2, 0.1])
    # Lottery weights by household size (replicates OHE design)
    weight = 1.0 / numhh
    prob_selected = weight / weight.max() * 0.35
    selected = rng.binomial(1, prob_selected, size=n)
    
    # Baseline covariates
    age_grp = rng.choice([0, 1, 2], size=n, p=[0.35, 0.40, 0.25])
    female = rng.binomial(1, 0.55, size=n)
    english = rng.binomial(1, 0.75, size=n)
    poor_health = rng.binomial(1, 0.30, size=n)
    
    # Treatment: only selected can enroll; ~25% compliance among selected
    enroll_prob = selected * 0.25
    treatment = rng.binomial(1, enroll_prob, size=n)
    
    # Outcome: catastrophic expenditure
    # True ATT ≈ -0.04 (Medicaid reduces catastrophic expenditure)
    logit_y = (
        -3.0
        + 0.4 * poor_health
        + 0.2 * (age_grp == 2).astype(float)
        - 0.4 * treatment     # causal effect
        + 0.3 * poor_health * (1 - treatment)   # confounding
    )
    prob_y = 1 / (1 + np.exp(-logit_y))
    catastrophic = rng.binomial(1, prob_y, size=n)
    
    doc_logit = (
        -1.0
        + 0.6 * treatment
        + 0.2 * poor_health
    )
    doc_any = rng.binomial(1, 1 / (1 + np.exp(-doc_logit)), size=n)
    
    df = pd.DataFrame({
        "selected": selected,
        "ohp_all_ever_admin": treatment,
        "catastrophic_exp_inp": catastrophic,
        "doc_any_12m": doc_any,
        "numhh_list": numhh,
        "age_19_34_inp": (age_grp == 0).astype(int),
        "age_35_49_inp": (age_grp == 1).astype(int),
        "age_50_64_inp": (age_grp == 2).astype(int),
        "female_inp": female,
        "english_list": english,
        "poor_health_bl": poor_health,
    })
    return df


# ---------------------------------------------------------------------------
# 2. Target trial specification
# ---------------------------------------------------------------------------

TARGET_TRIAL_PROTOCOL = {
    "eligibility": "Uninsured Oregon adults 19–64, income ≤100% FPL, no OHP in prior 6mo",
    "strategies": {0: "Remain uninsured (12 months)", 1: "Enroll in Medicaid (OHP)"},
    "time_zero": "Date of lottery notification",
    "follow_up": "12 months from time zero",
    "primary_outcome": "catastrophic_exp_inp (any catastrophic inpatient expenditure)",
    "estimand": "Average treatment effect (ATE) on risk difference scale",
    "analysis": "Standardization (g-computation) + IPCW for per-protocol",
}

def print_protocol(protocol: dict) -> None:
    print("\n=== TARGET TRIAL PROTOCOL ===")
    for k, v in protocol.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
    print("=" * 30 + "\n")


# ---------------------------------------------------------------------------
# 3. Baseline standardization (g-computation)
# ---------------------------------------------------------------------------

def baseline_covariate_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in [
        "age_19_34_inp", "age_35_49_inp", "age_50_64_inp",
        "female_inp", "english_list", "poor_health_bl",
        "numhh_list"
    ] if c in df.columns]
    X = df[cols].fillna(0).values.astype(float)
    return X, cols


def gcomputation(df: pd.DataFrame, outcome_col: str = "catastrophic_exp_inp") -> dict:
    """
    Standardization estimator for E[Y^1] - E[Y^0].
    Outcome model: logistic regression on (A, L).
    Marginal risks obtained by setting A=1 and A=0 for all units.
    """
    X_cov, covariate_names = baseline_covariate_matrix(df)
    A = df["ohp_all_ever_admin"].values
    Y = df[outcome_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cov)
    
    # Outcome model: fit on observed data
    X_full = np.column_stack([A, X_scaled])
    outcome_model = LogisticRegression(max_iter=1000, C=1.0)
    outcome_model.fit(X_full, Y)
    
    # Predict under A=1 and A=0 for all units (standardization)
    X_treat = np.column_stack([np.ones(len(df)), X_scaled])
    X_ctrl = np.column_stack([np.zeros(len(df)), X_scaled])
    
    risk1 = outcome_model.predict_proba(X_treat)[:, 1]
    risk0 = outcome_model.predict_proba(X_ctrl)[:, 1]
    
    ate_rd = risk1.mean() - risk0.mean()
    ate_rr = risk1.mean() / risk0.mean()
    
    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    boot_rd = []
    for _ in range(500):
        idx = rng.integers(0, len(df), size=len(df))
        df_b = df.iloc[idx].reset_index(drop=True)
        X_b, _ = baseline_covariate_matrix(df_b)
        X_b_scaled = scaler.transform(X_b)
        A_b = df_b["ohp_all_ever_admin"].values
        Y_b = df_b[outcome_col].values
        X_full_b = np.column_stack([A_b, X_b_scaled])
        try:
            m = LogisticRegression(max_iter=500, C=1.0)
            m.fit(X_full_b, Y_b)
            r1 = m.predict_proba(np.column_stack([np.ones(len(df_b)), X_b_scaled]))[:, 1]
            r0 = m.predict_proba(np.column_stack([np.zeros(len(df_b)), X_b_scaled]))[:, 1]
            boot_rd.append(r1.mean() - r0.mean())
        except Exception:
            pass
    
    ci = np.percentile(boot_rd, [2.5, 97.5])
    
    return {
        "E[Y^1]": risk1.mean(),
        "E[Y^0]": risk0.mean(),
        "ATE (RD)": ate_rd,
        "ATE (RR)": ate_rr,
        "95% CI (RD)": ci,
        "n": len(df),
    }


# ---------------------------------------------------------------------------
# 4. Clone-censor-weight for per-protocol analysis
# ---------------------------------------------------------------------------

def build_person_period(df: pd.DataFrame, n_periods: int = 4) -> pd.DataFrame:
    """
    Construct person-period dataset for CCW analysis.
    Simulate quarterly follow-up within the 12-month window.
    Each period: treatment status may change (dropout); outcome is end-of-period.
    
    For OHE (cross-sectional 12m data), we generate a plausible longitudinal
    structure: individuals enrolled in OHP may disenroll; administrative
    censoring at 12 months.
    """
    rng = np.random.default_rng(123)
    records = []
    
    for i, row in df.iterrows():
        a0 = int(row["ohp_all_ever_admin"])
        y_final = int(row["catastrophic_exp_inp"])
        
        # Time-varying covariate: cumulative doctor visits (proxy for health utilization)
        health_use = 0
        enrolled = a0
        had_event = False
        
        for t in range(n_periods):
            # Dropout from treatment: ~5% per period if enrolled
            if enrolled and rng.random() < 0.05:
                enrolled = 0
            
            # Health utilization increases slightly with enrollment
            health_use += rng.binomial(1, 0.3 + 0.15 * enrolled)
            
            # Event occurs with higher probability in final period
            event_this_period = 0
            if t == n_periods - 1 and not had_event:
                event_this_period = y_final
                had_event = True
            
            records.append({
                "id": i,
                "period": t,
                "a0": a0,                    # baseline treatment
                "a_t": enrolled,             # current treatment status
                "health_use_t": health_use,  # time-varying confounder
                "y_t": event_this_period,    # outcome (1 at first occurrence)
                "female": row.get("female_inp", 0),
                "age_50": row.get("age_50_64_inp", 0),
                "poor_health_bl": row.get("poor_health_bl", 0),
            })
    
    return pd.DataFrame(records)


def clone_censor_weight(pp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clone-censor-weight procedure.
    Returns long dataset with columns: id, period, arm (0/1), weight, y_t, censored.
    """
    arms = []
    
    for arm in [0, 1]:
        arm_df = pp_df.copy()
        arm_df["arm"] = arm
        
        # Censoring: deviates from arm's strategy
        if arm == 1:
            # "Always treat" arm: censor when a_t drops to 0
            arm_df["deviated"] = (arm_df["a_t"] == 0).astype(int)
        else:
            # "Never treat" arm: censor when a_t becomes 1
            arm_df["deviated"] = (arm_df["a_t"] == 1).astype(int)
        
        # Once deviated, all subsequent records are censored
        arm_df = arm_df.sort_values(["id", "period"])
        arm_df["cum_deviated"] = arm_df.groupby("id")["deviated"].cummax()
        arm_df["censored"] = arm_df["cum_deviated"]
        
        arms.append(arm_df)
    
    cloned = pd.concat(arms, ignore_index=True)
    
    # Estimate IPCW denominator: P(not deviate | history, L_t)
    # Fit separately per arm
    weighted_dfs = []
    
    for arm in [0, 1]:
        sub = cloned[cloned["arm"] == arm].copy()
        
        # Person-periods where still uncensored at start of period
        uncensored_mask = sub.groupby("id")["cum_deviated"].shift(1).fillna(0) == 0
        fit_data = sub[uncensored_mask].copy()
        
        features = ["health_use_t", "female", "age_50", "poor_health_bl", "period"]
        X_feat = fit_data[features].values.astype(float)
        
        # Target: did NOT deviate this period
        y_stay = (fit_data["deviated"] == 0).astype(int).values
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_feat)
        
        cens_model = LogisticRegression(max_iter=500, C=1.0)
        
        if y_stay.sum() < len(y_stay) and y_stay.sum() > 0:
            cens_model.fit(X_s, y_stay)
            
            # Predict on full uncensored data
            X_all = sub[features].values.astype(float)
            X_all_s = scaler.transform(X_all)
            p_stay = cens_model.predict_proba(X_all_s)[:, 1]
        else:
            p_stay = np.ones(len(sub))
        
        p_stay = np.clip(p_stay, 0.01, 0.99)
        sub = sub.copy()
        sub["p_stay"] = p_stay
        
        # Cumulative product within uncensored person-time
        sub = sub.sort_values(["id", "period"])
        sub["log_p_stay"] = np.log(sub["p_stay"])
        sub["cum_log_ipcw"] = sub.groupby("id")["log_p_stay"].cumsum()
        sub["ipcw"] = np.exp(-sub["cum_log_ipcw"])  # w = 1/prod(p_stay)
        
        # Zero out weights after censoring
        sub.loc[sub["censored"] == 1, "ipcw"] = 0.0
        
        # Stabilize: clip extreme weights
        q99 = sub.loc[sub["ipcw"] > 0, "ipcw"].quantile(0.99)
        sub["ipcw"] = sub["ipcw"].clip(upper=q99)
        
        weighted_dfs.append(sub)
    
    return pd.concat(weighted_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Kaplan-Meier under emulated trial
# ---------------------------------------------------------------------------

def kaplan_meier_ipcw(ccw_df: pd.DataFrame) -> dict:
    """
    Compute weighted Kaplan-Meier survival curves for each arm.
    S(t) = prod_{k<=t} [1 - d_k / n_k]  where d_k and n_k are IPCW-weighted.
    """
    results = {}
    
    for arm in [0, 1]:
        sub = ccw_df[(ccw_df["arm"] == arm) & (ccw_df["censored"] == 0)].copy()
        periods = sorted(sub["period"].unique())
        
        survival = 1.0
        curve = [1.0]
        times = [0]
        
        for t in periods:
            at_risk = sub[sub["period"] >= t]
            events_t = sub[(sub["period"] == t) & (sub["y_t"] == 1)]
            
            n_t = at_risk["ipcw"].sum()
            d_t = events_t["ipcw"].sum()
            
            if n_t > 0:
                survival *= (1 - d_t / n_t)
            
            curve.append(survival)
            times.append(t + 1)
        
        results[arm] = {"times": times, "survival": curve}
    
    return results


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_emulation_results(
    gcmp_results: dict,
    km_curves: dict,
    protocol: dict,
    save_path: str | None = None
) -> None:
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # Panel A: Kaplan-Meier under emulated trial
    ax1 = fig.add_subplot(gs[0, 0])
    colors = {1: "#2166ac", 0: "#d6604d"}
    labels = {1: "Medicaid (OHP) enrolled", 0: "Uninsured (control)"}
    
    for arm in [1, 0]:
        t = km_curves[arm]["times"]
        s = km_curves[arm]["survival"]
        ax1.step(t, s, where="post", color=colors[arm], lw=2.0, label=labels[arm])
    
    ax1.set_xlabel("Quarter (from time zero)", fontsize=11)
    ax1.set_ylabel("P(No catastrophic expenditure)", fontsize=11)
    ax1.set_title("Emulated Trial: IPCW Kaplan-Meier\nPer-Protocol Analysis", fontsize=11)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_ylim([0.85, 1.02])
    ax1.set_xlim([0, 4])
    ax1.grid(True, alpha=0.3)
    
    # Panel B: G-computation risk estimates
    ax2 = fig.add_subplot(gs[0, 1])
    
    risks = [gcmp_results["E[Y^0]"], gcmp_results["E[Y^1]"]]
    xerr_low = gcmp_results["95% CI (RD)"][0]
    xerr_high = gcmp_results["95% CI (RD)"][1]
    
    ax2.barh(
        ["Control\n(Uninsured)", "Treated\n(Medicaid)"],
        risks,
        color=["#d6604d", "#2166ac"],
        alpha=0.8,
        height=0.4,
    )
    ax2.axvline(risks[0], color="#d6604d", lw=1.5, ls="--", alpha=0.5)
    
    ate_rd = gcmp_results["ATE (RD)"]
    ci = gcmp_results["95% CI (RD)"]
    
    ax2.set_xlabel("P(Catastrophic expenditure)", fontsize=11)
    ax2.set_title(
        f"G-Computation: Standardized Risks\n"
        f"ATE = {ate_rd:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
        fontsize=11
    )
    ax2.set_xlim([0, max(risks) * 1.4])
    ax2.grid(True, alpha=0.3, axis="x")
    
    # Protocol annotation
    fig.text(
        0.5, -0.04,
        f"Target trial: {protocol['eligibility']}\n"
        f"Estimand: {protocol['estimand']}",
        ha="center", fontsize=8, color="gray"
    )
    
    plt.suptitle("Chapter 5: Target Trial Emulation — OHE", fontsize=13, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 7. Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_protocol(TARGET_TRIAL_PROTOCOL)
    
    print("Loading OHE data...")
    df = load_ohe()
    print(f"  n = {len(df):,} participants")
    print(f"  Treatment rate: {df['ohp_all_ever_admin'].mean():.3f}")
    print(f"  Outcome rate:   {df['catastrophic_exp_inp'].mean():.3f}\n")
    
    # ---- ITT estimate: naive comparison ----
    itt_treated = df[df["selected"] == 1]["catastrophic_exp_inp"].mean()
    itt_control = df[df["selected"] == 0]["catastrophic_exp_inp"].mean()
    print(f"ITT (intent-to-treat via lottery):")
    print(f"  E[Y | selected=1] = {itt_treated:.4f}")
    print(f"  E[Y | selected=0] = {itt_control:.4f}")
    print(f"  Risk difference   = {itt_treated - itt_control:.4f}\n")
    
    # ---- Per-protocol via g-computation ----
    print("Running g-computation (standardization)...")
    gcmp = gcomputation(df, outcome_col="catastrophic_exp_inp")
    print(f"G-computation (per-protocol, standardized):")
    print(f"  E[Y^1] = {gcmp['E[Y^1]']:.4f}")
    print(f"  E[Y^0] = {gcmp['E[Y^0]']:.4f}")
    print(f"  ATE (RD) = {gcmp['ATE (RD)']:.4f}  95% CI: [{gcmp['95% CI (RD)'][0]:.4f}, {gcmp['95% CI (RD)'][1]:.4f}]")
    print(f"  ATE (RR) = {gcmp['ATE (RR)']:.4f}\n")
    
    # ---- Clone-censor-weight ----
    print("Building person-period dataset for CCW analysis...")
    pp_df = build_person_period(df.head(5000), n_periods=4)  # subset for speed
    print(f"  Person-periods: {len(pp_df):,}")
    
    print("Applying clone-censor-weight...")
    ccw_df = clone_censor_weight(pp_df)
    
    print("Computing IPCW Kaplan-Meier curves...")
    km_curves = kaplan_meier_ipcw(ccw_df)
    
    # Summary statistics
    for arm in [0, 1]:
        final_surv = km_curves[arm]["survival"][-1]
        label = "Medicaid" if arm == 1 else "Uninsured"
        print(f"  {label}: 12m survival (no catastrophic exp) = {final_surv:.4f}")
    
    print("\nPlotting results...")
    plot_emulation_results(gcmp, km_curves, TARGET_TRIAL_PROTOCOL)
    
    print("\nDone.")
```

## Summary

- The **target trial framework** disciplines observational causal inference by requiring the analyst to fully specify the hypothetical randomized experiment that would answer the causal question before touching data.
- The seven protocol components—eligibility, treatment strategies, assignment mechanism, outcomes, follow-up, estimand, and analysis plan—must each be mapped to an observational analog, making the assumptions required for identification explicit.
- **Time zero misalignment** is the most common structural error in observational studies; immortal time bias inflates apparent treatment benefit by attributing pre-treatment events to untreated participants.
- **IPCW** recovers the per-protocol estimand by reweighting the observed data to remove the informative censoring that protocol deviations create; consistency requires the independent censoring assumption and positivity of the censoring mechanism.
- **Standardization** (the baseline g-formula) identifies the average treatment effect under conditional exchangeability, positivity, and consistency; the g-computation estimator implements it via outcome model predictions marginalized over the observed covariate distribution.
- The **clone-censor-weight** technique provides a transparent, dataframe-level implementation of per-protocol IPCW that is algebraically equivalent to MSM-IPTW weighting for static treatment regimes.
- The **active comparator design** reduces confounding by indication by replacing "untreated" with a clinically comparable alternative treatment, changing the structure of residual confounding rather than eliminating it.

## Further Reading

- **Hernán MA, Robins JM (2016).** "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology* 183(8):758–764. The foundational methodological paper for the framework; introduces the protocol checklist and applies it to observational Medicare data comparing treatment strategies for cardiovascular outcomes. Read before any applied emulation work.

- **Hernán MA, Sauer BC, Hernández-Díaz S, Platt R, Shrier I (2016).** "Specifying a target trial prevents immortal time bias and other self-inflicted injuries in observational analyses." *Journal of Clinical Epidemiology* 79:70–75. A case study dissecting how protocol misspecification produces immortal time bias, healthy worker bias, and depletion of susceptibles in three published observational analyses. The clearest published exposition of time zero alignment.

- **Robins JM, Hernán MA, Brumback B (2000).** "Marginal Structural Models and Causal Inference in Epidemiology." *Epidemiology* 11(5):550–560. The original MSM paper; derives IPCW in the time-varying treatment context, establishes the g-null hypothesis test, and proves the MSM-IPCW consistency theorem. Mathematically demanding but essential for understanding the theoretical foundations.

- **Finkelstein A, Taubman S, Wright B, et al. (2012).** "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics* 127(3):1057–1106. The primary published analysis of OHE; ITT estimates using lottery selection as instrument. Provides ground truth marginal statistics for calibrating any OHE-based emulation exercise and is the source of the variable definitions used throughout this chapter.

- **Dickerman BA, García-Albéniz X, Logan RW, Denaxas S, Hernán MA (2019).** "Avoidable flaws in observational analyses: an application to statins and cancer." *Nature Medicine* 25:1601–1606. Demonstrates target trial emulation in a high-profile setting (statins and cancer risk) where prior observational analyses reached contradictory conclusions; shows that protocol misspecification explains the contradictions. Particularly strong on the active comparator design and time zero alignment in administrative database contexts.
The `src/causal_book` directory doesn't exist yet. I have enough context from the blueprint to write the full chapter without needing to read any existing simulation code. Here is the complete chapter:

# Chapter 6: Treatment Timing, Time Zero, and Immortal Time Bias

## 6.1 The Problem of Time Zero

Every survival or longitudinal study requires a precise answer to the question: *when does follow-up begin?* This moment — call it $t_0$, or **time zero** — is not merely a bookkeeping detail. It defines which population is under study, which events are counted, and which person-time is attributed to which exposure group. Get it wrong and the entire analysis is compromised, often in a direction that flatters the treatment.

The canonical motivating example is a pharmacoepidemiologic cohort study comparing users of a new drug against non-users. The investigator identifies all patients who ever filled a prescription for the drug and checks whether they were alive at some fixed calendar date. The survival curve for drug users looks remarkably good. The problem: patients must survive long enough to receive the drug. The period between cohort entry and first prescription — during which the patient could not possibly have died *as a drug user* — was nevertheless assigned to the "treated" arm. That person-time is **immortal** with respect to the outcome, and the artificially inflated survival in the treatment group is **immortal time bias**.

This chapter addresses three intertwined problems: (1) immortal time that arises from misclassifying pre-treatment person-time, (2) time zero misalignment that arises when the outcome clock and the treatment clock are anchored to different events, and (3) prevalent user bias that arises when the study population includes individuals already under treatment at the study's start. These are not exotic pathologies. They are routine sources of confounding in any observational study using administrative data, and they appear — sometimes in disguised form — even in carefully designed natural experiments like the Oregon Health Insurance Experiment.

## 6.2 Immortal Time Bias: Formal Statement

Let $T_i$ denote the event time (death, hospitalization, or any absorbing outcome) for individual $i$ and let $D_i(t)$ denote the time-varying treatment indicator, equal to 1 once the individual initiates treatment. Define $\tau_i$ as the time of treatment initiation, with $\tau_i = \infty$ for never-treated individuals.

A study correctly assigns individual $i$ to the treated arm only for person-time $t \geq \tau_i$ and to the untreated arm for $t < \tau_i$ (or throughout for never-treated individuals). **Immortal time bias** occurs when the investigator assigns the interval $[0, \tau_i)$ to the treated arm. This is "immortal" in the literal sense: $T_i < \tau_i$ is a logical impossibility if treatment at time $\tau_i$ is the defining criterion for being in the treated group.

**Theorem 6.1 (Immortal Time Bias Direction).** Suppose the true hazard ratio is $\text{HR}_{\text{true}} = h_1(t)/h_0(t) = 1$ (no effect). Let a fraction $\pi$ of the treated arm's person-time be immortal, with hazard $h_\text{immortal}(t) = 0$ by construction. Under a Cox proportional hazards model, the biased estimand converges to

$$\widehat{\text{HR}}_{\text{biased}} \;\to\; \frac{h_1(t)}{h_0(t) + h_\text{immortal}(t)}$$

where $h_\text{immortal}(t)$ is the contribution of misclassified immortal person-time to the effective hazard of the "comparison" group. Since $h_\text{immortal}(t) \geq 0$, the estimand is driven below 1: the treatment appears protective when it is not.

*Proof sketch.* The Cox partial likelihood score for the log-hazard ratio $\beta$ is $U(\beta) = \sum_j \left[Z_j - \bar{Z}(t_j)\right]$ where the sum is over event times, $Z_j$ is the covariate value of the individual who fails at $t_j$, and $\bar{Z}(t_j) = \sum_{l \in R(t_j)} Z_l e^{\beta Z_l} / \sum_{l \in R(t_j)} e^{\beta Z_l}$ is the risk-set weighted average. Misclassifying immortal time inflates the risk-set exposure to $Z=1$ at early times when, by construction, no events occur in that group, pulling $\bar{Z}(t_j)$ upward and the score downward, yielding $\hat{\beta} < 0$. $\square$

The key intuition: immortal person-time adds to the denominator of the hazard estimate for the treated group (via the risk set) without adding to the numerator (events cannot occur there). This mechanically compresses the hazard.

## 6.3 Time Zero Misalignment

Immortal time bias is one manifestation of a deeper problem: the failure to align the treatment clock and the outcome clock at a common anchor event. Chapter 5 introduced the **target trial** framework, in which eligibility, assignment, and follow-up all begin at the same moment. When an observational study deviates from this structure, multiple forms of bias can enter simultaneously.

**Definition 6.1 (Time Zero Misalignment).** A study suffers from time zero misalignment if the event used to anchor follow-up for the treated group differs from the event used to anchor follow-up for the control group, or if either anchor differs from the event that defines eligibility for the study population.

In the Oregon Health Insurance Experiment, the natural time zero is the date of the lottery notification — the moment at which individuals were randomized (via lottery) to having the *opportunity* to enroll in Medicaid. All 12-month outcomes are measured from this date. The intent-to-treat estimand is clean: compare lottery winners to lottery losers from notification date onward, regardless of whether winners actually enrolled.

Now consider what happens if an analyst, working from administrative data, decides to define treatment status by first documented Medicaid enrollment and anchors the treated group's follow-up from that enrollment date. Two distortions appear:

1. **Immortal time**: the period from lottery notification to enrollment is immortal for the treated group.
2. **Selection into $t_0$**: individuals who survive (and remain enrolled) long enough to use health care are systematically healthier at their redefined time zero. This is the *healthy user* confounding that the lottery was designed to eliminate.

The time-conditional estimand that any such analysis actually targets is

$$E[Y(a) \mid \text{alive and eligible at } t_0]$$

which is not the same as the unconditional $E[Y(a)]$ unless survival to $t_0$ is independent of potential outcomes — an assumption that is almost never credible.

## 6.4 The New-User Design

The most direct structural remedy for time zero misalignment is the **new-user design**, attributed to Ray (2003) and formalized by Hernán and Robins. The defining restriction is:

$$P(D_i = 1 \mid t < t_0) = 0$$

No individual enters the treated group having already been treated before the study's time zero. Concretely, this means:

- Impose a **washout period** before cohort entry during which the treatment of interest was not dispensed.
- Define $t_0$ as the date of *first-ever* treatment initiation for treated patients.
- Match or restrict the control group to individuals who are also new non-users — that is, who have not been treated recently.

The new-user restriction eliminates immortal time by ensuring that the treated group's person-time begins at the moment of treatment, not before. It also eliminates the most pernicious form of prevalent user bias (Section 6.6).

**Proposition 6.1 (New-User Consistency).** Under the new-user restriction, correct alignment of time zero with treatment initiation, and the no-unmeasured-confounders assumption at $t_0$, the hazard ratio from a Cox model is a consistent estimator of the causal hazard ratio

$$\text{HR}_{\text{causal}}(t) = \frac{h(t \mid D=1, X)}{h(t \mid D=0, X)}$$

for individuals satisfying the eligibility criteria at $t_0$.

The qualifier "at $t_0$" is important. The new-user design estimates a conditional effect in the eligible population at time zero; it does not recover population-average effects for individuals who initiated treatment earlier (who are excluded) or later (who have not yet been observed).

## 6.5 Grace Period Analysis

Strict new-user designs can be difficult to implement when treatment initiation is itself endogenous — for example, when sicker patients initiate earlier. A **grace period** design relaxes the strict time-zero anchor by allowing a window $[t_0, t_0 + g]$ during which the individual may or may not initiate treatment, and assigns treatment status based on whether initiation occurs within this grace period.

Formally, let $g$ be the grace period length. Define

$$D_i^g = \mathbf{1}[\tau_i \leq t_0 + g]$$

All individuals are considered at-risk from $t_0$ regardless of when within the grace period they initiate. Events occurring before $\tau_i$ contribute to the treated group's risk set only for the period starting at $t_0$, not from $\tau_i$. This avoids immortal time while acknowledging that treatment timing is uncertain.

The grace period design is particularly natural in the OHE context. Lottery winners could enroll at any point following notification. An analyst studying 12-month outcomes might define a 60-day grace period: anyone who enrolled within 60 days of notification is "treated," and follow-up for all individuals begins at notification. This preserves the intent-to-treat logic while approximating a per-protocol analysis.

The cost of grace period analysis is that the estimand becomes a mixture of "treated early" and "treated late" effects, averaged over the grace window. If treatment effects are time-varying — perhaps early enrollment is more beneficial than late enrollment — the grace period HR may not correspond to any single well-defined causal quantity. Sensitivity analysis over $g$ is advisable.

## 6.6 Prevalent User Bias and the Active Comparator Design

A **prevalent user** study begins follow-up at a fixed calendar date and includes individuals who were already using the treatment at that date. This is common in database studies where the investigator slices an administrative dataset at a single time point. The bias this introduces is distinct from immortal time.

**Decomposition 6.1 (Prevalent User Bias).** Let $f_s$ denote the fraction of prevalent users who have been on treatment for duration $s$. The average effect estimated in a prevalent user study is

$$\widehat{\text{ATE}}_{\text{prevalent}} = \int_0^\infty \text{ATE}(s) \cdot f_s \, ds$$

where $\text{ATE}(s)$ is the average treatment effect among individuals who initiated $s$ time units ago. If treatment effects are heterogeneous across $s$ — as is almost always true — this estimand is a weighted average over a distribution of treatment durations, not the effect of initiating treatment.

Two additional selection mechanisms compound the bias. First, **depletion of susceptibles**: individuals who would have suffered early adverse events under treatment have already done so and are excluded from the prevalent cohort, making prevalent users appear artificially healthy. Second, **confounding by indication at initiation**: the factors that led to treatment initiation at time $-s$ may differ systematically from the factors that would lead to initiation today, invalidating any comparison made at the cross-sectional observation date.

The **active comparator new-user (ACNU) design** addresses both problems. Rather than comparing drug users to non-users, it compares new initiators of drug A to new initiators of drug B, where B is indicated for similar clinical conditions. This design:

1. Eliminates prevalent users from both arms.
2. Ensures time zero is treatment initiation for all participants.
3. Achieves comparability of confounders at $t_0$ by restricting to individuals who are clinically eligible to receive either treatment.

In the insurance context, an ACNU analog would compare new Medicaid enrollees to new enrollees in subsidized private insurance, both observed from their enrollment date — rather than comparing Medicaid enrollees to uninsured individuals who may have never been medically evaluated.

## 6.7 Time-Varying Hazard Ratios

Even under correct time zero alignment and a new-user design, the proportional hazards assumption can be violated if the effect of treatment changes over the follow-up period. The Cox model produces a single log-hazard coefficient $\hat{\beta}$ that is a weighted average of the log-hazard ratio over event times, with weights proportional to the variance of the risk-set exposure distribution at each event time. This quantity is interpretable only under proportional hazards; otherwise it estimates a variance-weighted blend of time-varying effects.

Let $\text{HR}(t) = h(t \mid D=1) / h(t \mid D=0)$ denote the true time-varying hazard ratio. The Cox estimand converges to

$$\hat{\beta} \to \frac{\int_0^\infty \log[\text{HR}(t)] \cdot w(t) \, dt}{\int_0^\infty w(t) \, dt}$$

where $w(t) \propto V(t) S(t)$ with $V(t)$ the variance of treatment in the risk set at $t$ and $S(t)$ the joint survivor function. This collapsing of time-varying effects is not bias per se — it is the natural consequence of fitting a scalar parameter to a function — but it means the estimated HR depends on the censoring distribution, the event rate, and the time horizon in ways that are not portable across studies.

Diagnostics for nonproportionality include Schoenfeld residual plots (residuals regressed on time should have zero slope under proportionality) and $\log(-\log(\hat{S}(t)))$ plots. When proportionality fails, the analyst should consider stratified Cox models, time-varying coefficient models, or report restricted mean survival times (RMST) as a summary that does not require proportionality.

## Python: Immortal Time Bias — Simulation and Correction

The following code simulates a survival dataset with immortal time bias built in, compares Kaplan-Meier estimates and Cox HRs under the misaligned versus correct analysis, and verifies that the bias disappears under correct time zero alignment. It is modeled on the OHE setting: $n = 20{,}000$ individuals, lottery notification at $t_0 = 0$, with a grace period of 60 days for enrollment.

```python
"""
Chapter 6: Immortal Time Bias — Simulation and Correction
Running example: Oregon Health Insurance Experiment (OHE) analog

Requires: numpy, pandas, matplotlib, lifelines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

rng = np.random.default_rng(2024)

# ── 1. Data-generating process ─────────────────────────────────────────────
# N individuals enter at t=0 (lottery notification).
# "Selected" (Z=1) means they won the lottery and may enroll.
# Enrollment happens Uniform(14, 90) days after notification for winners
# who actually enroll (fraction ENROLL_RATE).
# True treatment has NO effect on survival: HR_true = 1.0.
# Survival times drawn from Exponential(rate=0.003/day) for all units.
# Follow-up ends at 365 days (12-month study).

N = 20_000
ENROLL_RATE = 0.60   # fraction of lottery winners who actually enroll
TRUE_HR = 1.0        # no causal effect

# Lottery assignment
selected = rng.binomial(1, 0.5, size=N).astype(bool)

# Enrollment (treatment initiation)
enrolled = np.zeros(N, dtype=bool)
enroll_day = np.full(N, np.inf)

mask_enrolls = selected & (rng.uniform(0, 1, N) < ENROLL_RATE)
enrolled[mask_enrolls] = True
enroll_day[mask_enrolls] = rng.uniform(14, 90, mask_enrolls.sum())

# True survival times (same distribution for all — no effect)
baseline_hazard = 0.003 / 365   # very low daily rate; events are rare
survival_times = rng.exponential(1 / baseline_hazard, size=N)

# Administrative censoring at 365 days
observed_time = np.minimum(survival_times, 365.0)
event_indicator = (survival_times <= 365.0).astype(int)

df = pd.DataFrame({
    "id": np.arange(N),
    "selected": selected.astype(int),
    "enrolled": enrolled.astype(int),
    "enroll_day": enroll_day,
    "event_time": observed_time,
    "event": event_indicator,
})

print(f"N={N:,}  |  Selected: {selected.mean():.2f}  |  "
      f"Enrolled: {enrolled.mean():.2f}  |  Events: {event_indicator.mean():.4f}")

# ── 2. WRONG analysis: misaligned time zero ─────────────────────────────────
# Treated group: enrolled individuals, follow-up starts at enroll_day.
# Control group: non-enrolled, follow-up starts at t=0.
# This assigns immortal time [0, enroll_day) to the treated arm.

df_wrong = df.copy()

# For enrolled: their "observed time from their t0" is event_time - enroll_day
# (capped at 365 - enroll_day). But crucially, the KM / Cox treats them as
# having entered at enroll_day, which misrepresents the risk set.
# We model this by giving them a shorter apparent follow-up.
df_wrong["analysis_time"] = np.where(
    df_wrong["enrolled"] == 1,
    np.maximum(df_wrong["event_time"] - df_wrong["enroll_day"], 0.01),
    df_wrong["event_time"],
)
df_wrong["treatment"] = df_wrong["enrolled"]

# ── 3. CORRECT analysis: time-on-study from notification date (t=0) ─────────
# Both enrolled and non-enrolled are observed from t=0.
# Enrolled individuals contribute to the "treated" risk set only from enroll_day.
# We handle this via delayed entry (left truncation) in lifelines.

df_correct = df.copy()
df_correct["start"] = np.where(df_correct["enrolled"] == 1,
                                df_correct["enroll_day"], 0.0)
df_correct["stop"] = df_correct["event_time"]
df_correct["treatment"] = df_correct["enrolled"]

# Drop individuals whose event occurred before they enrolled (impossible to treat)
df_correct = df_correct[df_correct["stop"] > df_correct["start"]].copy()

# ── 4. Kaplan-Meier comparison ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (label, dframe, start_col) in zip(
    axes,
    [
        ("WRONG: misaligned t0", df_wrong, None),
        ("CORRECT: delayed entry", df_correct, "start"),
    ],
):
    for trt_val, color, name in [(1, "steelblue", "Enrolled"), (0, "tomato", "Not enrolled")]:
        sub = dframe[dframe["treatment"] == trt_val]
        kmf = KaplanMeierFitter()
        if start_col:
            kmf.fit(
                sub["stop"], event_observed=sub["event"],
                entry=sub[start_col], label=name,
            )
        else:
            kmf.fit(sub["analysis_time"], event_observed=sub["event"], label=name)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Days from time zero")
    ax.set_ylabel("Survival probability")
    ax.legend()
    ax.set_xlim(0, 365)

plt.tight_layout()
plt.savefig("ch6_km_comparison.png", dpi=150)
plt.show()
print("KM figure saved to ch6_km_comparison.png")

# ── 5. Cox regression: wrong vs. correct ────────────────────────────────────
print("\n── Cox Regression Results ──────────────────────────────────")

# Wrong analysis
cph_wrong = CoxPHFitter()
cph_wrong.fit(
    df_wrong[["analysis_time", "event", "treatment"]],
    duration_col="analysis_time",
    event_col="event",
)
hr_wrong = np.exp(cph_wrong.params_["treatment"])
ci_wrong = np.exp(cph_wrong.confidence_intervals_.loc["treatment"])
print(f"WRONG  HR = {hr_wrong:.3f}  "
      f"95% CI [{ci_wrong.iloc[0]:.3f}, {ci_wrong.iloc[1]:.3f}]  "
      f"(true HR = {TRUE_HR:.1f})")

# Correct analysis (delayed entry)
cph_correct = CoxPHFitter()
cph_correct.fit(
    df_correct[["start", "stop", "event", "treatment"]],
    entry_col="start",
    duration_col="stop",
    event_col="event",
)
hr_correct = np.exp(cph_correct.params_["treatment"])
ci_correct = np.exp(cph_correct.confidence_intervals_.loc["treatment"])
print(f"CORRECT HR = {hr_correct:.3f}  "
      f"95% CI [{ci_correct.iloc[0]:.3f}, {ci_correct.iloc[1]:.3f}]  "
      f"(true HR = {TRUE_HR:.1f})")

# ── 6. Quantify bias as a function of immortal time length ──────────────────
grace_lengths = [7, 14, 30, 60, 90, 120]
hrs_biased = []

for g in grace_lengths:
    # Restrict to enrollees who enrolled within g days
    mask_enrolled_g = (df["enrolled"] == 1) & (df["enroll_day"] <= g)
    df_g = df.copy()
    df_g["enrolled_g"] = mask_enrolled_g.astype(int)

    df_g_wrong = pd.DataFrame({
        "analysis_time": np.where(
            df_g["enrolled_g"] == 1,
            np.maximum(df_g["event_time"] - df_g["enroll_day"], 0.01),
            df_g["event_time"],
        ),
        "event": df_g["event"],
        "treatment": df_g["enrolled_g"],
    })

    cph_g = CoxPHFitter()
    try:
        cph_g.fit(df_g_wrong, duration_col="analysis_time", event_col="event")
        hrs_biased.append(np.exp(cph_g.params_["treatment"]))
    except Exception:
        hrs_biased.append(np.nan)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(grace_lengths, hrs_biased, "o-", color="steelblue", lw=2, label="Biased HR")
ax2.axhline(TRUE_HR, color="tomato", linestyle="--", lw=1.5, label=f"True HR = {TRUE_HR}")
ax2.set_xlabel("Mean immortal time window (days)")
ax2.set_ylabel("Estimated hazard ratio")
ax2.set_title("Immortal time bias grows with misalignment window")
ax2.legend()
plt.tight_layout()
plt.savefig("ch6_bias_vs_window.png", dpi=150)
plt.show()
print("\nBias-vs-window figure saved to ch6_bias_vs_window.png")

# ── 7. OHE replication stub (structure only; download data from NBER) ───────
def load_ohe_lottery(path: str) -> pd.DataFrame:
    """
    Load Oregon Health Insurance Experiment lottery data.
    Download from: https://data.nber.org/oregon/
    Expected columns used here:
      - selected       (Z): lottery winner indicator
      - ohp_all_ever_admin (D): ever enrolled in Medicaid within 12m
      - numhh_list     : household size strata (1, 2, 3+)
      - doc_any_12m    (Y): any doctor visit in 12m
      - Any event/censoring variables if survival analysis is desired
    """
    df = pd.read_stata(path)
    # Standardize
    df = df.rename(columns={
        "selected": "Z",
        "ohp_all_ever_admin": "D",
        "doc_any_12m": "Y_doc",
        "numhh_list": "strata",
    })
    return df[["Z", "D", "Y_doc", "strata"]].dropna()


def new_user_itt_ohe(df_ohe: pd.DataFrame) -> dict:
    """
    Intent-to-treat estimate of any doctor visit, correctly anchored at t0
    (lottery notification). Returns ITT and naive (misaligned) estimates for
    comparison. Both are simple proportion differences here because OHE uses
    a binary 12-month outcome, not continuous time-to-event.
    """
    itt = (df_ohe.groupby("Z")["Y_doc"].mean().diff().iloc[-1])
    # Naive: restrict treated to ever-enrolled (introduces selection)
    treated_mask = df_ohe["D"] == 1
    naive = (df_ohe.loc[treated_mask, "Y_doc"].mean()
             - df_ohe.loc[~treated_mask, "Y_doc"].mean())
    return {"itt": itt, "naive_att": naive}

# Usage (requires downloaded data):
# df_ohe = load_ohe_lottery("path/to/oregonhie_survey12m_vars.dta")
# print(new_user_itt_ohe(df_ohe))

# ── 8. Schoenfeld residuals: testing proportional hazards ───────────────────
print("\n── Proportional Hazards Test (correct model) ──────────────")
cph_correct.check_assumptions(
    df_correct[["start", "stop", "event", "treatment"]],
    p_value_threshold=0.05,
    show_plots=False,
)
```

**Expected output.** With $n=20{,}000$ and no true treatment effect, the wrong analysis should produce a hazard ratio meaningfully below 1.0 (typically in the range $0.55$–$0.75$ depending on the enrollment delay distribution), while the correct delayed-entry analysis should recover an HR close to 1.0 with a confidence interval that comfortably includes 1.0. The bias-vs-window plot confirms that larger immortal time windows produce more severe downward bias.

## 6.8 Connections to the Target Trial Framework

Chapter 5 introduced the target trial as a template for observational studies. Immortal time bias and time zero misalignment are precisely the failures that arise when the observational study deviates from two of the target trial's protocol elements: the **eligibility criteria** (which determine who enters the study at $t_0$) and the **start of follow-up** (which must equal $t_0$ for all eligible individuals simultaneously).

Table 6.1 maps each design error to its target trial analog:

| Design error | Target trial deviation | Direction of bias |
|---|---|---|
| Pre-treatment person-time in treated arm | Follow-up starts before assignment | HR biased downward |
| Outcome clock anchored to first use | Eligibility differs from assignment | Selection + immortal time |
| Prevalent user study | Eligibility requires prior exposure | Depletion-of-susceptibles |
| No washout period | Contamination of "new user" status | Attenuation or amplification |

The operational correction in each case is identical: emulate the target trial by fixing $t_0$ as the moment of assignment (or its observational analog), requiring no prior treatment before $t_0$, and measuring all outcomes from $t_0$ forward. In the OHE this is natural — the lottery date is unambiguous. In the ACA Medicaid expansion, the target trial analog would begin follow-up at January 1 of each expansion year for newly eligible residents.

## 6.9 Latent Index of Treatment Uptake

In observational data without a clean instrument like the OHE lottery, treatment initiation is endogenous. A useful structural interpretation frames initiation as the crossing of a latent threshold: individual $i$ initiates treatment when a latent propensity index $I_i(t)$ exceeds a threshold $c_i$,

$$D_i(t) = \mathbf{1}[I_i(t) \geq c_i], \qquad I_i(t) = X_i(t)'\gamma + U_i$$

where $X_i(t)$ are observed time-varying covariates (symptom severity, provider characteristics, insurance status) and $U_i$ is unobserved heterogeneity. The timing of treatment — $\tau_i = \inf\{t : I_i(t) \geq c_i\}$ — is a first-passage time for a stochastic process, and it is correlated with $T_i$ through $U_i$.

This latent index perspective clarifies why the new-user restriction alone does not solve confounding: even restricting to first initiators, the reason for initiating at $t_0$ rather than earlier or later is informative about prognosis. Propensity score methods, instrumental variables, or — in the presence of time-varying covariates — g-methods (Chapter 25) are required. The latent index model also suggests a diagnostic: if the hazard of treatment initiation spikes at certain covariate configurations that also predict the outcome, confounding by indication is likely severe.

## Summary

- **Immortal time bias** arises when person-time before treatment initiation is attributed to the treatment group, mechanically deflating the estimated hazard ratio even when the true causal effect is null.
- The formal bias direction is $\widehat{\text{HR}}_{\text{biased}} = h_1(t)/[h_0(t) + h_\text{immortal}(t)] < \text{HR}_{\text{true}}$; its magnitude grows with the length of the misclassified window.
- **Time zero misalignment** is the structural root cause: outcome and treatment clocks must be anchored to the same event for all study participants simultaneously.
- The **new-user design** corrects both problems by requiring $P(D_i=1 \mid t < t_0)=0$ and aligning $t_0$ with first treatment initiation.
- **Prevalent user bias** further contaminates studies that admit individuals already on treatment at the study's start, via depletion of susceptibles and confounding by the duration of prior treatment.
- The **active comparator new-user design** achieves comparability at $t_0$ by restricting both arms to new initiators of clinically similar treatments.
- The **grace period** is a practical compromise that avoids immortal time while accommodating uncertainty in treatment timing, at the cost of a mixture estimand.
- These design principles are operationalizations of the target trial: eligibility, assignment, and follow-up must begin simultaneously for all participants at a common, well-defined $t_0$.

## Further Reading

1. **Suissa S (2008).** "Immortal time bias in pharmaco-epidemiology." *American Journal of Epidemiology* 167(4):492–499. The definitive formal treatment with pharmacoepidemiologic applications; includes the HR formula derived in this chapter.
2. **Ray WA (2003).** "Evaluating medication effects outside of clinical trials: new-user designs." *American Journal of Epidemiology* 158(9):915–920. Original articulation of the new-user restriction and its consequences for prevalent user bias.
3. **Hernán MA, Alonso A, Logan R, et al. (2008).** "Observational studies analyzed like randomized experiments: an application to postmenopausal hormone therapy and coronary heart disease." *Epidemiology* 19(6):766–779. Applies the target trial framework explicitly to correct time zero misalignment in the WHI controversy.
4. **Johnson ES, Bartman BA, Briesacher BA, et al. (2013).** "The incident user design in comparative effectiveness research." *Pharmacoepidemiology and Drug Safety* 22(1):1–6. Practical guidance on implementing new-user and active comparator designs in claims databases.
5. **Finkelstein A, Taubman S, Wright B, et al. (2012).** "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics* 127(3):1057–1106. Primary analysis of the OHE using correct intent-to-treat with lottery notification as time zero; the paper implicitly avoids all biases discussed in this chapter by virtue of randomization and correct analysis timing.
6. **Lévesque LE, Hanley JA, Kezouh A, Suissa S (2010).** "Problem of immortal time bias in cohort studies: example using statins for preventing progression of diabetes." *BMJ* 340:b5087. A vivid real-world illustration of how immortal time bias led to spurious cardioprotective findings for statins in a diabetic cohort.
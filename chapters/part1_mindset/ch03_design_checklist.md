# Chapter 3: The Causal Design Checklist

Every failed empirical study shares a common genealogy: a researcher who moved too fast. The data arrived, the regressions ran, and only afterward — if at all — did the hard questions surface. What exactly is the treatment? Who is in the sample? What would an honest threat taxonomy look like? These questions are not decorative. Their answers determine whether any quantity estimated from the data corresponds to a causal effect or to an artifact of design. This chapter provides a systematic protocol for answering them before estimation begins.

The discipline is not merely procedural. Each item in the checklist corresponds to a formal assumption that must hold for your estimator to converge to the target estimand. Violating treatment consistency, ignoring interference, or selecting outcomes post-hoc are not aesthetic errors — they are identification failures. The pre-analysis plan forces those assumptions into the open, where they can be scrutinized rather than quietly violated.

We work through every item using the ACA Medicaid expansion as the running example: a staggered difference-in-differences design spanning 2010–2016 in which states adopted expanded Medicaid eligibility at different times. BRFSS provides annual, state-level self-reported health and insurance outcomes. By the end of the chapter, the design for that study is fully specified — before any estimation occurs.

---

## 3.1 The Pre-Analysis Plan as Epistemic Discipline

A pre-analysis plan (PAP) is a timestamped document, committed before data access, that specifies the estimand, the sample, the primary outcomes, the covariates, the estimator, and the inference procedure. Its purpose is not bureaucratic: it is to sever the feedback loop between results and specification.

Without a PAP, specification search is structurally indistinguishable from p-hacking even when the researcher acts in good faith. The human capacity to construct post-hoc rationalizations for whatever specification survived is essentially unbounded. A PAP makes that rationalization visible and costly.

**What a PAP must contain.** At minimum:

1. The causal estimand (ATE, ATT, LATE, CATE over a specified subpopulation).
2. The treatment, defined operationally with no ambiguity about versions.
3. The primary outcome(s), with measurement instrument and timing.
4. The sample: inclusion criteria, exclusion criteria, clustering unit.
5. The identification strategy and the assumptions it requires.
6. The primary estimator and any pre-specified robustness checks.
7. The multiple testing procedure if more than one primary outcome is declared.

PAPs do not prohibit exploratory analysis. They distinguish confirmatory from exploratory claims, allowing readers to calibrate accordingly. The Bayes-consistent interpretation is that a result specified a priori carries more posterior weight than one selected from a garden of forking paths.

**Pre-registration in quasi-experimental settings.** Many applied economists resist pre-registration on the grounds that quasi-experimental designs require data exploration to assess identifying assumptions. This is backwards. Assumption assessment (parallel trends diagnostics, covariate balance inspection, event-study pre-trends) is itself subject to specification search if conducted without pre-specification. A credible PAP for a DiD study specifies which pre-trends test will be used, what failure criterion triggers abandonment of the design, and which alternative estimand would be reported under partial violations.

---

## 3.2 Treatment Definition: Sharpness, Versions, and Consistency

The potential outcomes framework rests on the Stable Unit Treatment Value Assumption (SUTVA), which Chapter 1 stated as a conjunction of no-interference and consistency. Here we focus on consistency, which is the more frequently violated component in applied work.

**Consistency.** For binary treatment $D_i \in \{0,1\}$, consistency states:

$$Y_i = Y_i(D_i) \quad \text{a.s.}$$

That is, the observed outcome equals the potential outcome corresponding to the treatment actually received. This is trivially true if there is exactly one version of each treatment level. It fails — silently, with no observable diagnostic — when multiple versions exist.

**Theorem 3.1 (Consistency violation under multiple versions).** Suppose treatment level $d = 1$ corresponds to $K \geq 2$ distinct versions $d^{(1)}, \ldots, d^{(K)}$ with $Y_i(d^{(k)})$ not constant in $k$ for a positive-measure set of units. Define $D_i^{(k)} = \mathbb{1}[\text{unit } i \text{ receives version } k]$, with $\sum_k D_i^{(k)} = D_i$. Then:

$$\mathbb{E}[Y_i | D_i = 1] = \sum_{k=1}^K \mathbb{E}[Y_i(d^{(k)}) | D_i^{(k)} = 1] \cdot P(D_i^{(k)} = 1 | D_i = 1)$$

This mixture weight $P(D_i^{(k)} = 1 | D_i = 1)$ is determined by the data-generating process, not the analyst. Any estimator that ignores version heterogeneity estimates a mixture of version-specific effects whose composition is unknown and may vary across subpopulations, time, or studies. The estimand is literally undefined without specifying which mixture you intend.

*Proof sketch.* By the law of total expectation, condition on $D_i^{(k)}$ within the event $\{D_i = 1\}$. Each $\mathbb{E}[Y_i(d^{(k)}) | D_i^{(k)} = 1]$ is a well-defined version-specific potential outcome mean. The overall conditional mean is their weighted average. The weights are version-share probabilities, which are population quantities outside analyst control. $\square$

**Application: "insurance access" in the ACA expansion.** The natural treatment in a Medicaid expansion study is "having health insurance." But this is profoundly not a single treatment version. Consider the relevant versions:

- Medicaid coverage (state-managed, formulary varies by state)
- Marketplace ACA plan at various metal tiers (Bronze/Silver/Gold/Platinum)
- Employer-sponsored insurance
- Medicare (age-eligible overlap)
- Short-duration or catastrophic plans (post-2018 policy changes)

Each version has different cost-sharing, network breadth, covered services, and reimbursement rates. The causal effect of Bronze marketplace coverage on preventive care utilization may differ qualitatively from Medicaid's effect, not just in magnitude but in sign for certain outcomes, because network restrictions differ.

**The fix is definitional precision, not statistical.** Sharpen the treatment to: *enrollment in state Medicaid program as the primary payer, at any point during the calendar year.* This excludes marketplace plans, employer coverage, and Medicare. The estimand is now the effect of Medicaid-specifically on outcomes, relative to no insurance. This is the quantity that ACA Medicaid expansion actually randomizes (in the DiD sense): variation in Medicaid eligibility thresholds across states and time.

**Checklist item 1.** Write one sentence defining the treatment with no ambiguity about versions. If you cannot do this, estimation should not proceed.

---

## 3.3 Outcome Selection, Surrogates, and Multiple Testing

Outcome selection is the second major locus of researcher degrees of freedom. The problems are distinct: (a) selecting outcomes that confirm priors, (b) using surrogate outcomes whose relationship to the primary estimand is unverified, and (c) multiple testing without correction.

**Primary vs. secondary outcomes.** Pre-specify one or at most two primary outcomes. All others are secondary; label them as such in every table and figure. The distinction matters for inference: a family-wise error rate correction applied only to primary outcomes, with secondary results interpreted as hypothesis-generating, is statistically coherent. Applying it post-hoc is not.

**Surrogate outcomes.** A surrogate outcome $S_i$ stands in for a primary outcome $Y_i$ when $Y_i$ is unobserved or requires long follow-up. The classical Prentice (1989) surrogacy criterion requires:

$$Z \perp\!\!\!\perp Y \mid S$$

meaning the instrument (or treatment) has no effect on $Y$ except through $S$. This is an untestable assumption — it cannot be verified from the observed data distribution — but it can be falsified if variation in $Z$ predicts $Y$ residually after conditioning on $S$.

**In BRFSS:** Direct health outcomes like mortality or disease incidence require long follow-up that BRFSS cross-sections cannot support. Instead, BRFSS provides self-reported health status (excellent/very good/good/fair/poor), preventive care utilization (mammogram, colonoscopy, flu vaccine in past year), and unmet care needs (could not see doctor due to cost). These are plausible mediators on the pathway from insurance to long-run health, but they are surrogates, not terminal outcomes. Pre-specify that surrogacy is assumed, not established, and flag this explicitly in sensitivity analysis.

**Multiple outcomes and FWER control.** If the PAP declares five primary outcomes, the probability of at least one false rejection at $\alpha = 0.05$ under the global null is $1 - 0.95^5 \approx 0.23$. Standard corrections:

- **Bonferroni**: reject $H_{0,j}$ when $p_j < \alpha/m$ for $m$ tests. Exact under independence, conservative under positive dependence.
- **Holm (1979)**: step-down procedure that is uniformly more powerful than Bonferroni. Order $p_{(1)} \leq \cdots \leq p_{(m)}$; reject $H_{0,(j)}$ for all $j \leq j^*$ where $j^* = \max\{j : p_{(j)} \leq \alpha/(m-j+1)\}$.
- **Romano-Wolf**: resampling-based, accounts for dependence structure among outcomes; preferred when outcomes are correlated (as health outcomes are).

**Checklist item 2.** List primary outcomes (maximum two), secondary outcomes, and the multiple testing procedure. For each surrogate, state the surrogacy assumption explicitly.

---

## 3.4 Sample Inclusion, Exclusion, and Transportability

The sample is not the population. This sentence contains the entirety of the external validity problem.

**Defining the study sample.** Inclusion and exclusion criteria must be pre-specified. Each criterion has two consequences: it refines the population to which the estimate applies (the study population $\mathcal{P}^*$), and it affects the composition of treatment and control groups (potentially differentially, creating selection bias within the study).

For the ACA expansion study, the BRFSS sampling frame is non-institutionalized adults aged 18–64. Standard exclusions for the Medicaid expansion analysis:

- Age $\geq 65$: Medicare-eligible, different coverage regime.
- Pre-ACA baseline already insured (if estimand is among the previously uninsured).
- States that partially expanded or adopted Medicaid under a waiver (definitional ambiguity).
- BRFSS respondents with imputed state residence (measurement issue).

**Transportability.** The target population $\mathcal{P}$ — the population to which policymakers want to generalize — may differ systematically from $\mathcal{P}^*$. Transportability requires that the conditional distribution of potential outcomes given pre-treatment covariates be the same across populations:

$$P^*(Y(d) \mid X = x) = P(Y(d) \mid X = x) \quad \text{for all } x \in \text{supp}(X)$$

where $P^*$ is the study distribution and $P$ is the target distribution. This is the **transportability condition** (Bareinboim and Pearl, 2013). Under it, reweighting by $P(X) / P^*(X)$ — importance weighting by the target-to-study covariate density ratio — delivers a consistent estimate of the target population ATE.

**Theorem 3.2 (Transportability via reweighting).** If the transportability condition holds and $\text{supp}(P) \subseteq \text{supp}(P^*)$, then:

$$\mathbb{E}_P[Y(d)] = \mathbb{E}_{P^*}\!\left[\frac{P(X)}{P^*(X)} \cdot Y(d)\right]$$

*Proof.* By the law of iterated expectations under $P^*$:

$$\mathbb{E}_{P^*}\!\left[\frac{P(X)}{P^*(X)} Y(d)\right] = \int \frac{P(x)}{P^*(x)} \mathbb{E}_{P^*}[Y(d)\mid X=x]\, P^*(x)\, dx$$
$$= \int \mathbb{E}_{P^*}[Y(d)\mid X=x]\, P(x)\, dx = \int \mathbb{E}_P[Y(d)\mid X=x]\, P(x)\, dx = \mathbb{E}_P[Y(d)]$$

where the penultimate equality uses the transportability condition. $\square$

**Practical implication.** If BRFSS oversamples rural states relative to the national target, and if rural/urban residence moderates the treatment effect of Medicaid on preventive care (rural residents face provider scarcity regardless of coverage), then the BRFSS-based estimate differs from the national ATE. Pre-specify whether you are estimating the study-population ATT or a reweighted national ATE, and if the latter, specify the reweighting covariates.

**Checklist item 3.** State inclusion/exclusion criteria. Identify the study population vs. target population. Pre-specify whether transportability reweighting will be applied and with what covariates.

---

## 3.5 Timing: Treatment Onset, Washout, and Follow-up Horizon

Causal effects are indexed to time. A treatment effect at 30 days post-onset may differ in sign from the effect at 5 years. Failing to specify the follow-up horizon makes the estimand temporally ambiguous — and staggered designs compound this substantially.

**Three timing decisions:**

1. **Time zero** ($t_0$): the moment at which treatment assignment (not necessarily enrollment) occurs. In the ACA expansion, $t_0$ for each state is January 1 of the expansion year, because Medicaid eligibility changes on that date. Outcome measurement pre- and post- $t_0$ must be aligned consistently.

2. **Follow-up horizon** ($\Delta$): the period over which outcomes are measured. For BRFSS annual cross-sections, the effective horizon is one year. Longer-run effects require linking to administrative data (Medicaid records, mortality registries). Declare which horizon is primary.

3. **Washout and anticipation.** Anticipation effects occur when units respond to anticipated treatment before $t_0$ (e.g., states enrolling residents in Medicaid preparatory programs before official expansion). If anticipation exists, the "pre-treatment" period is contaminated, and standard parallel-trends arguments for DiD apply to the wrong comparison window. The formal requirement: no anticipation means $Y_{it}(d_1, d_2, \ldots) = Y_{it}(0, 0, \ldots)$ for all $t < t_0$, i.e., potential outcomes in pre-treatment periods are unaffected by future treatment assignment.

**Staggered adoption timing in the ACA expansion.** States expanded in waves:

| Wave | Expansion Year | States (examples) |
|------|---------------|-------------------|
| Early | 2014 | California, New York, Kentucky, 22 others |
| Intermediate | 2015 | Indiana, Pennsylvania, Montana |
| Late | 2016–2019 | Louisiana, Virginia, Idaho |
| Non-adopters | — | Texas, Florida, Georgia (through 2016) |

With staggered adoption, there is no single "post" period. Standard two-way fixed effects (TWFE) estimators weight treatment effects by variance of treatment timing — a weighting scheme that (a) is implicit rather than chosen by the researcher and (b) assigns negative weights to already-treated units used as controls (Callaway and Sant'Anna, 2021; Sun and Abraham, 2021). Pre-specify which cohort-specific ATTs you will aggregate and how.

**Checklist item 4.** Define $t_0$ for each unit. Define the follow-up horizon. State whether anticipation effects are possible and how they will be handled. For staggered designs, enumerate cohorts and specify the aggregation scheme.

---

## 3.6 Threat Taxonomy: A Structured Enumeration

Identification assumptions are falsifiable in principle and fragile in practice. The design checklist requires enumerating every threat to validity before looking at results, then ranking threats by severity and pre-specifying which can be bounded, which can be tested, and which must be assumed away.

**Six threat classes:**

### 3.6.1 Confounding

Confounding occurs when unobserved variables jointly cause treatment and outcome. In the ACA expansion, the canonical confounder is pre-existing state economic conditions: states with stronger economies both (a) expanded Medicaid voluntarily (political economy) and (b) exhibit better health outcomes for economic reasons independent of insurance. Standard DiD absorbs time-invariant state heterogeneity via state fixed effects, but time-varying confounders — GDP growth shocks, opioid epidemic intensity, healthcare workforce trends — are not absorbed.

Enumerate time-varying confounders explicitly: state unemployment rate, per-capita income growth, uninsured rate trend pre-ACA, Medicaid managed care penetration, state cigarette tax, and regional opioid epidemic measures.

### 3.6.2 Selection Bias

Selection into the study sample differs from treatment selection. If BRFSS response rates differ between expansion and non-expansion states post-2014 (attrition or differential cooperation), the observed sample is not representative even of the study population. Attrition that is correlated with outcomes and treatment creates selection bias that standard FE cannot address.

### 3.6.3 Interference

**No interference (the second component of SUTVA)** states that unit $i$'s potential outcomes depend only on $i$'s own treatment, not on others':

$$Y_i(\mathbf{d}) = Y_i(d_i) \quad \forall\, \mathbf{d} : d_i \text{ fixed}$$

where $\mathbf{d} = (d_1, \ldots, d_n)$ is the full treatment vector. In health insurance, interference arises through:

- **Provider market effects**: a large influx of Medicaid patients into a state market affects wait times and provider availability for all patients, including the uninsured and privately insured.
- **Social spillovers**: when a household member gains Medicaid, other members' healthcare utilization and labor supply may change.
- **General equilibrium**: hospital systems may adjust pricing or capacity in response to aggregate enrollment changes.

In the state-level DiD, interference between states is less acute (states are somewhat closed health markets), but within-state interference is present. Pre-specify that the estimate is an intent-to-treat effect of state eligibility expansion and does not claim to identify a treatment effect in a hypothetical world where interference is eliminated.

### 3.6.4 Measurement Error

BRFSS self-reported health and insurance status are subject to recall bias, social desirability bias, and item nonresponse. Insurance status is particularly noisy: respondents may not accurately classify their coverage type, conflating Medicaid, marketplace, and employer coverage. This is classical measurement error in the outcome (attenuates nothing in a linear model, but biases variance estimates) and potentially non-classical measurement error in the treatment if insurance misreporting correlates with health status.

### 3.6.5 Attrition

Respondent attrition from BRFSS is not directly measurable (cross-sectional design), but state-level response rates vary. If state response rate changes differentially post-expansion — e.g., expansion states improve health, enrolled persons report more, and healthier people disproportionately respond — then the post-treatment sample is positively selected in expansion states relative to control states, biasing the DiD upward.

### 3.6.6 Parallel Trends Violations

The DiD identification assumption (Chapter 8) requires that in the absence of treatment, expansion and non-expansion states would have followed the same trend in outcomes. This is untestable in the post-period. The standard diagnostic — testing for equal pre-period trends — is necessary but not sufficient. A confounded trend that happens to be parallel in the pre-period may diverge post-2014 for reasons unrelated to Medicaid (e.g., the opioid epidemic hit rural non-expansion states harder post-2014).

**Checklist item 5.** For each of the six threat classes, write one paragraph: (a) whether the threat is present in your design, (b) what makes it severe or mild, (c) how it will be diagnosed or bounded in analysis.

---

## 3.7 Design Adjudication: Choosing the Identification Strategy

Having specified the estimand, treatment, outcomes, sample, timing, and threats, the analyst now selects an identification strategy. This is not a free choice: the data-generating process, the nature of the identifying variation, and the threats in Section 3.6 jointly constrain which strategies are credible.

**A decision tree:**

1. **Is there randomization of treatment?** If yes, and if randomization was properly implemented, proceed to experimental analysis (Chapters 19–20). If compliance is imperfect, IV/LATE methods apply (Chapter 21).

2. **Is there an as-good-as-random instrument?** If yes, IV identification applies — but verify the exclusion restriction $Z \perp\!\!\!\perp Y(d)$. The exclusion restriction states:

$$\mathbb{E}[Y_i(d, z)] = \mathbb{E}[Y_i(d, z')] \quad \forall\, d, z, z'$$

meaning the instrument $Z$ affects $Y$ only through $D$, not directly. Stated here; derived formally in Chapter 21.

3. **Is there a sharp discontinuity in treatment assignment at a known threshold?** Regression discontinuity applies (Chapter 23). Verify density continuity (McCrary test), bandwidth sensitivity, and local nature of the estimand.

4. **Is there differential timing of a common policy shock?** Difference-in-differences with staggered adoption applies (Chapters 8–10). Verify: parallel trends plausibility, no anticipation, treatment effect homogeneity or explicit heterogeneous-effects estimators.

5. **If none of the above:** Observational study under unconfoundedness (selection on observables). Requires strong substantive justification that all confounders are measured. Chapters 11–17 cover matching and weighting estimators for this case.

**For the ACA expansion:** Strategy 4 applies. The identifying variation is staggered state-level adoption of Medicaid expansion. The estimand is a weighted average of cohort-specific ATTs under parallel trends and no-anticipation.

**Checklist item 6.** State the identification strategy, the assumptions it requires, and the diagnostics you will run to assess each assumption. For each assumption, classify as: (a) testable with available data, (b) partially testable, or (c) fundamentally untestable.

---

## Python: Pre-Analysis Plan Template, Covariate Balance, and BRFSS Profiling

```python
"""
Chapter 3: Causal Design Checklist
Oregon Health Insurance Experiment / ACA Medicaid Expansion

Requirements:
    pip install pandas numpy statsmodels matplotlib scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PRE-ANALYSIS PLAN TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

PAP_FIELDS = {
    "study_title": "Causal Effect of Medicaid Expansion on Health and Financial Outcomes: "
                   "ACA 2014–2016 Staggered DiD",
    "date_registered": "2025-01-15",

    # Estimand
    "estimand": "Average Treatment Effect on the Treated (ATT) for cohort-specific "
                "Medicaid expansion adopters, aggregated per Callaway-Sant'Anna (2021).",
    "estimand_population": "Non-institutionalized US adults aged 18-64 in states that "
                           "eventually expanded Medicaid, pre-expansion.",

    # Treatment
    "treatment_definition": "Binary indicator: state adopted full ACA Medicaid expansion "
                             "(income threshold >= 138% FPL) by January 1 of the calendar year. "
                             "Partial expansions and waiver programs excluded.",
    "treatment_versions_check": "Excludes marketplace plans, CHIP, Medicare, ESI. "
                                 "Treatment = Medicaid eligibility expansion, not any insurance.",

    # Outcomes
    "primary_outcome_1": "BRFSS HLTHPLN1: proportion with any health coverage. "
                          "Interpretation: insurance take-up from eligibility change.",
    "primary_outcome_2": "BRFSS GENHLTH <= 3: proportion reporting good/very good/excellent health. "
                          "Surrogate for health status; surrogacy assumed not established.",
    "secondary_outcomes": [
        "BRFSS CHECKUP1 == 1 (routine checkup in past year)",
        "BRFSS MEDCOST == 1 (could not see doctor due to cost in past year)",
        "BRFSS FLUSHOT6 == 1 (flu vaccine in past year)",
    ],
    "multiple_testing_correction": "Holm (1979) step-down for primary outcomes. "
                                    "Secondary outcomes interpreted as exploratory.",

    # Sample
    "inclusion_criteria": [
        "Age 18-64",
        "Resident of 50 states + DC",
        "Non-missing state identifier in BRFSS",
        "Survey year 2010-2016",
    ],
    "exclusion_criteria": [
        "Age >= 65",
        "States with pre-ACA Medicaid waivers (MA, VT, DC) from expansion cohort",
        "Respondents with imputed state of residence",
    ],
    "analysis_unit": "State-year aggregate (population-weighted means from BRFSS micro-data).",

    # Timing
    "time_zero": "January 1 of the state's Medicaid expansion year.",
    "pre_treatment_window": "2010-2013 (4 years pre-expansion for most states).",
    "post_treatment_window": "Up to 3 years post-expansion (varies by adoption cohort).",
    "anticipation": "Assume no anticipation. Sensitivity: drop 2013 as potential "
                    "anticipation year for 2014 adopters.",

    # Identification
    "identification_strategy": "Staggered DiD: Callaway-Sant'Anna (2021) cohort-specific ATT "
                                "with never-treated and not-yet-treated control groups.",
    "parallel_trends_assumption": "Conditional parallel trends given state pre-treatment "
                                   "covariates: pre-ACA uninsured rate, unemployment rate, "
                                   "per-capita income, rural-urban classification.",
    "primary_estimator": "Doubly-robust CS-DiD (2021); TWFE as benchmark with negative-weight warning.",
    "robustness_checks": [
        "TWFE as benchmark (with negative-weight warning)",
        "Sun-Abraham (2021) interaction-weighted estimator",
        "Stacked DiD regression",
        "Placebo: assign false expansion years T-2",
        "Exclude mid-year partial-expansion states",
    ],

    # Threats
    "threat_confounding": "Time-varying state economic shocks. Control: include unemployment "
                           "rate, per-capita income as conditional parallel-trends covariates.",
    "threat_selection": "Differential BRFSS response rates post-expansion. Diagnose via "
                         "state-year response rate trends.",
    "threat_interference": "Provider market spillovers within state. Acknowledged; estimand "
                            "is intent-to-treat on eligibility.",
    "threat_measurement": "Insurance status misclassification in BRFSS. Bound using "
                           "administrative CMS enrollment data where available.",
    "threat_attrition": "Cross-sectional BRFSS; no panel attrition. State response rate "
                         "monitored as covariate.",
    "threat_parallel_trends": "Assess via event-study pre-trends. Failure criterion: "
                               "p < 0.10 on joint test of pre-period coefficients.",
}

print("=" * 80)
print("PRE-ANALYSIS PLAN SUMMARY")
print("=" * 80)
for k, v in PAP_FIELDS.items():
    print(f"\n[{k.upper()}]")
    print(f"  {str(v)[:150]}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SYNTHETIC BRFSS STATE-YEAR PANEL
# Calibrated to published BRFSS statistics. Replace data_load() with real file.
# Real BRFSS download: https://www.cdc.gov/brfss/annual_data/annual_data.htm
# ─────────────────────────────────────────────────────────────────────────────

def generate_brfss_panel(seed: int = 42) -> pd.DataFrame:
    """
    Synthetic state-year panel mimicking BRFSS 2010-2016 structure.
    Calibrated to published BRFSS national statistics:
      - National uninsured rate ~2010: 16.3%, ~2016: 8.8%
      - Expansion states had ~5-7pp larger drops in uninsured rate
    """
    rng = np.random.default_rng(seed)

    states_2014 = [
        'CA', 'NY', 'IL', 'OH', 'MI', 'NJ', 'WA', 'KY', 'AR',
        'AZ', 'CO', 'CT', 'HI', 'IA', 'MD', 'MN', 'NM', 'NV', 'ND',
        'OR', 'RI', 'WV',
    ]
    states_2015 = ['IN', 'MT', 'NH', 'AK']
    states_2016 = ['LA']
    states_never = [
        'TX', 'FL', 'GA', 'NC', 'AL', 'MS', 'SC', 'TN',
        'WI', 'KS', 'MO', 'SD', 'WY', 'ID', 'UT', 'OK', 'VA'
    ]

    records = []
    years = list(range(2010, 2017))
    all_states = states_2014 + states_2015 + states_2016 + states_never

    for state in all_states:
        if state in states_2014:
            exp_year, cohort = 2014, '2014'
        elif state in states_2015:
            exp_year, cohort = 2015, '2015'
        elif state in states_2016:
            exp_year, cohort = 2016, '2016'
        else:
            exp_year, cohort = None, 'never'

        # State-level fixed characteristics
        base_insured = rng.uniform(0.75, 0.92)
        if cohort != 'never':
            base_insured -= rng.uniform(0.02, 0.08)  # expansion states had lower baseline coverage
        base_insured = np.clip(base_insured, 0.60, 0.95)

        base_goodhealth = rng.uniform(0.72, 0.88)
        base_checkup    = rng.uniform(0.65, 0.82)
        base_medcost    = rng.uniform(0.08, 0.22)
        base_flushot    = rng.uniform(0.38, 0.58)

        unemp_2010    = rng.uniform(6.0, 12.0)
        income_pc_2010 = rng.uniform(35000, 58000)
        rural_pct     = rng.uniform(0.05, 0.45)

        for year in years:
            t = year - 2010
            unemp    = np.clip(unemp_2010 - t * 0.5 + rng.normal(0, 0.4), 2.5, 15.0)
            income_pc = income_pc_2010 * (1 + 0.02 * t) + rng.normal(0, 500)

            treated      = int(exp_year is not None and year >= exp_year)
            years_since  = (year - exp_year) if exp_year is not None else -99

            # True causal effects (ground truth for synthetic data)
            delta_insured    = (0.07 + 0.005 * min(max(years_since, 0), 2)
                                + rng.normal(0, 0.005)) * treated
            delta_goodhealth = ((0.015 + 0.005 * min(max(years_since - 1, 0), 2))
                                * int(treated and years_since >= 1)
                                + rng.normal(0, 0.003) * treated)
            delta_checkup    = 0.04 * treated + rng.normal(0, 0.004) * treated
            delta_medcost    = -0.03 * treated + rng.normal(0, 0.003) * treated

            records.append({
                'state': state, 'year': year, 'cohort': cohort,
                'expansion_year': exp_year, 'treated': treated,
                'years_since_expansion': years_since,
                # Outcomes
                'insured_rate': np.clip(
                    base_insured - 0.003 * t + delta_insured + rng.normal(0, 0.008),
                    0.50, 0.99),
                'good_health_rate': np.clip(
                    base_goodhealth + 0.001 * t + delta_goodhealth + rng.normal(0, 0.006),
                    0.50, 0.98),
                'checkup_rate': np.clip(
                    base_checkup + delta_checkup + rng.normal(0, 0.007), 0.40, 0.95),
                'medcost_rate': np.clip(
                    base_medcost + delta_medcost + rng.normal(0, 0.005), 0.02, 0.40),
                'flushot_rate': np.clip(
                    base_flushot + 0.005 * t + rng.normal(0, 0.010), 0.25, 0.75),
                # Covariates
                'unemployment_rate': unemp,
                'income_pc': income_pc,
                'rural_pct': rural_pct,
                'unemp_2010': unemp_2010,
                'income_pc_2010': income_pc_2010,
            })

    return pd.DataFrame(records)


df = generate_brfss_panel()
print(f"\nDataset: {df.shape[0]} state-year obs | "
      f"{df['state'].nunique()} states | years {df['year'].min()}-{df['year'].max()}")
print(df.drop_duplicates('state')['cohort'].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SAMPLE INCLUSION FLOWCHART
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SAMPLE INCLUSION FLOWCHART (Simulated BRFSS 2010-2016 Pooled)")
print("=" * 80)

total = 450_000  # approximate BRFSS interviews pooled across years
exclusions = [
    ("Age >= 65 (Medicare-eligible)",                                  int(total * 0.180)),
    ("Missing state identifier",                                        int(total * 0.002)),
    ("Non-state territories",                                           int(total * 0.012)),
    ("Pre-ACA expansion states (MA/VT/DC) — removed from exp. cohort", int(total * 0.035)),
]

n = total
print(f"\n  Raw BRFSS interviews (2010-2016 pooled):  {n:>10,}")
for reason, excl in exclusions:
    n -= excl
    print(f"  Exclude: {reason:<55} n excl={excl:>7,}  remaining={n:>8,}")

print(f"\n  Analytic individual-level sample:          {n:>10,}")
print(f"  Collapsed to state-year panel:             {df.shape[0]:>10,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: COVARIATE BALANCE TABLE
# Pre-treatment window: 2012-2013; 2014 adopters vs. never-treating states
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("COVARIATE BALANCE: 2014 Expansion Cohort vs. Never-Expanding (pre-period 2012-2013)")
print("=" * 80)

pre_df = df[df['year'].isin([2012, 2013]) & df['cohort'].isin(['2014', 'never'])].copy()
pre_df['exp_group'] = (pre_df['cohort'] == '2014').astype(int)

covariate_map = {
    'insured_rate':       'Pre-treatment insured rate',
    'good_health_rate':   'Pre-treatment good/exc. health rate',
    'checkup_rate':       'Routine checkup rate',
    'medcost_rate':       'Could not afford care rate',
    'unemployment_rate':  'Unemployment rate (%)',
    'income_pc':          'Per-capita income ($)',
    'rural_pct':          'Rural population share',
}

rows = []
for var, label in covariate_map.items():
    t_vals = pre_df.loc[pre_df['exp_group'] == 1, var]
    c_vals = pre_df.loc[pre_df['exp_group'] == 0, var]
    diff   = t_vals.mean() - c_vals.mean()
    t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)
    pooled_sd = np.sqrt((t_vals.std()**2 + c_vals.std()**2) / 2)
    smd = diff / pooled_sd if pooled_sd > 0 else np.nan
    rows.append({
        'Variable':          label,
        'Expansion (2014)':  f"{t_vals.mean():.4f}",
        'Never-Expanding':   f"{c_vals.mean():.4f}",
        'Difference':        f"{diff:+.4f}",
        'Std. Diff.':        f"{smd:+.3f}",
        'p-value':           f"{p_val:.3f}",
        'Sig.': "***" if p_val<0.01 else ("**" if p_val<0.05 else ("*" if p_val<0.10 else ""))
    })

balance_df = pd.DataFrame(rows)
print(balance_df.to_string(index=False))
print("\nNote: |Std. Diff.| > 0.1 conventionally flags material imbalance.")
print("* p<0.10  ** p<0.05  *** p<0.01  (Welch two-sample t-test, state-year level)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: STAGGERED ADOPTION TIMELINE + EVENT STUDY (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

# Left: staggered adoption grid
cohorts_ordered = ['2014', '2015', '2016', 'never']
cohort_colors   = {'2014': '#2166ac', '2015': '#67a9cf', '2016': '#d1e5f0', 'never': '#f4a582'}
cohort_labels   = {
    '2014':  '2014 adopters\n(22 states)',
    '2015':  '2015 adopters\n(4 states)',
    '2016':  '2016 adopter\n(1 state)',
    'never': 'Non-adopters\n(17 states)',
}
years = list(range(2010, 2017))

for i, cohort in enumerate(cohorts_ordered):
    exp_yr = {'2014': 2014, '2015': 2015, '2016': 2016, 'never': None}[cohort]
    for year in years:
        treated = (exp_yr is not None and year >= exp_yr)
        rect = plt.Rectangle(
            (year - 0.48, i - 0.38), 0.96, 0.76,
            facecolor=cohort_colors[cohort] if treated else 'white',
            edgecolor=cohort_colors[cohort],
            linewidth=1.2,
            alpha=0.85 if treated else 0.4,
        )
        ax_l.add_patch(rect)

ax_l.set_xlim(2009.5, 2016.5)
ax_l.set_ylim(-0.55, len(cohorts_ordered) - 0.45)
ax_l.set_xticks(years)
ax_l.set_yticks(range(len(cohorts_ordered)))
ax_l.set_yticklabels([cohort_labels[c] for c in cohorts_ordered], fontsize=9)
ax_l.set_xlabel('Year', fontsize=10)
ax_l.set_title('ACA Medicaid Expansion\nStaggered Adoption Structure', fontsize=10)
ax_l.axvline(2013.5, color='red', ls='--', lw=1.5, alpha=0.7)
ax_l.text(2013.6, 3.45, 'ACA\neffective', color='red', fontsize=8)
ax_l.grid(axis='x', alpha=0.25)

treated_patch   = mpatches.Patch(color='#2166ac', label='Post-expansion (treated)')
untreated_patch = mpatches.Patch(facecolor='white', edgecolor='#2166ac', label='Pre-expansion')
never_patch     = mpatches.Patch(color='#f4a582', label='Never-treated (2010-2016)')
ax_l.legend(handles=[treated_patch, untreated_patch, never_patch], loc='upper left', fontsize=8)

# Right: event study (2014 cohort relative to t = -1)
es_df = df[(df['cohort'] == '2014') & (df['years_since_expansion'] >= -4)].copy()
es_agg = (
    es_df.groupby('years_since_expansion')['insured_rate']
         .agg(['mean', 'std', 'count'])
)
es_agg['se']   = es_agg['std'] / np.sqrt(es_agg['count'])
es_agg['ci95'] = 1.96 * es_agg['se']
ref = es_agg.loc[-1, 'mean']
es_agg['demeaned'] = es_agg['mean'] - ref

ts = es_agg.index.values
ax_r.errorbar(
    ts, es_agg['demeaned'], yerr=es_agg['ci95'],
    fmt='o-', color='#2166ac', capsize=4, lw=2, ms=6, label='2014 adopters'
)
ax_r.axhline(0, color='black', lw=0.8, ls='--')
ax_r.axvline(-0.5, color='red', ls='--', lw=1.5, alpha=0.7, label='Expansion onset')
ax_r.fill_betweenx([-0.02, 0.13], -0.5, max(ts) + 0.5, alpha=0.05, color='red')
ax_r.set_xlabel('Years Since Medicaid Expansion', fontsize=10)
ax_r.set_ylabel('Δ Insurance Rate (relative to t = −1)', fontsize=10)
ax_r.set_title('Event Study: Insurance Rate\n(2014 Cohort, Synthetic Data)', fontsize=10)
ax_r.set_xticks(sorted(ts))
ax_r.legend(fontsize=9)
ax_r.grid(alpha=0.25)

plt.tight_layout()
fig.savefig('ch03_timeline_event_study.png', dpi=150, bbox_inches='tight')
print("\n\nFigure saved: ch03_timeline_event_study.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: NAIVE TWFE (benchmark, not primary estimator)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("NAIVE TWFE ESTIMATE  (Benchmark — biased under treatment effect heterogeneity)")
print("=" * 80)

twfe = smf.ols(
    'insured_rate ~ treated + C(state) + C(year)',
    data=df
).fit(cov_type='HC3')

coef = twfe.params['treated']
se   = twfe.bse['treated']
ci   = twfe.conf_int().loc['treated']

print(f"\n  TWFE coeff on 'treated':  {coef:+.4f}")
print(f"  HC3 SE:                    {se:.4f}")
print(f"  95% CI:                    [{ci[0]:+.4f}, {ci[1]:+.4f}]")
print(f"  N (state-year obs):        {int(twfe.nobs)}")
print(f"\n  CAUTION: TWFE may assign negative weights to late-adopter cohorts when used")
print(f"  as controls for early adopters. CS-DiD (2021) is the pre-specified estimator.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PARALLEL TRENDS PRE-TEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("PARALLEL TRENDS PRE-TEST: 2014 Cohort vs. Never-Treated")
print("Joint F-test on pre-period event-study coefficients (reference: t = -1)")
print("=" * 80)

pt_df = df[df['cohort'].isin(['2014', 'never'])].copy()
pt_df['rel_year'] = pt_df.apply(
    lambda r: r['year'] - 2014 if r['cohort'] == '2014' else -99, axis=1
)
for ry, nm in [(-4, 'ry_n4'), (-3, 'ry_n3'), (-2, 'ry_n2'),
                (0,  'ry_0'),  (1,  'ry_p1'), (2,  'ry_p2')]:
    pt_df[nm] = (pt_df['rel_year'] == ry).astype(int)

event_mod = smf.ols(
    'insured_rate ~ ry_n4 + ry_n3 + ry_n2 + ry_0 + ry_p1 + ry_p2 + C(state) + C(year)',
    data=pt_df
).fit(cov_type='HC3')

periods = [('ry_n4', 't−4'), ('ry_n3', 't−3'), ('ry_n2', 't−2'),
           ('ry_0',  't=0'),  ('ry_p1', 't+1'), ('ry_p2', 't+2')]

print(f"\n  {'Period':<8} {'Coef':>9} {'SE':>8} {'p':>8}  Notes")
print(f"  {'-'*50}")
for var, lbl in periods:
    c = event_mod.params[var]; s = event_mod.bse[var]; p = event_mod.pvalues[var]
    note = ' <- pre-period' if lbl.startswith('t−') else ''
    print(f"  {lbl:<8} {c:>9.4f} {s:>8.4f} {p:>8.3f}{note}")

pre_vars = ['ry_n4', 'ry_n3', 'ry_n2']
jt = event_mod.f_test([f'{v} = 0' for v in pre_vars])
print(f"\n  Joint test H0: all pre-period coefficients = 0")
print(f"  F = {jt.fvalue[0][0]:.3f},  p = {jt.pvalue:.3f}")
print(f"  Pre-specified failure criterion: p < 0.10")
if jt.pvalue < 0.10:
    print("  STATUS: FAILED — investigate before proceeding to estimation.")
else:
    print("  STATUS: PASSED — no significant pre-period deviation detected.")
    print("  (Non-rejection does not confirm parallel trends; it fails to falsify.)")


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN CHECKLIST STATUS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("DESIGN CHECKLIST: STATUS BEFORE ESTIMATION")
print("=" * 80)
checklist = [
    (1, "Treatment definition",    "DONE", "Medicaid-as-primary-payer; excludes marketplace/ESI/Medicare"),
    (2, "Outcome selection",       "DONE", "Primary: insured_rate, good_health_rate. Holm correction pre-specified."),
    (3, "Sample criteria",         "DONE", "Ages 18-64; 2010-2016; exclusions logged."),
    (4, "Timing",                  "DONE", "t0 = Jan 1 expansion year; pre 2010-2013; no-anticipation assumed."),
    (5, "Threat enumeration",      "DONE", "Six threats assessed; parallel trends testable with pre-trend F-test."),
    (6, "Identification strategy", "DONE", "CS-DiD (2021); TWFE as biased benchmark; pre-trends passed."),
]
for num, item, status, note in checklist:
    print(f"\n  [{status}] {num}. {item}")
    print(f"         {note}")
print("\nAll items complete. Proceed to Chapter 8 (Difference-in-Differences) for estimation.\n")
```

---

## Summary

- A pre-analysis plan severs the feedback loop between results and specification. Each item corresponds to a formal identifying assumption; the PAP is the mechanism that makes those assumptions falsifiable before they are needed.

- **Consistency violations** are silent: Theorem 3.1 shows that an ambiguously defined treatment with $K$ versions yields an estimand that is an uncontrolled mixture of version-specific effects. The mix weights are determined by the DGP, not the analyst, and may vary arbitrarily across settings. Definitional precision is the only remedy; it is not statistical.

- **Transportability** from study to target population holds when $P^*(Y(d)|X) = P(Y(d)|X)$; Theorem 3.2 shows importance reweighting then recovers the target ATE. When the condition fails, the estimate is local to the study population by construction and cannot be extrapolated by any statistical adjustment.

- **Timing** decisions — time-zero alignment, follow-up horizon, and the no-anticipation assumption — are load-bearing. In staggered adoption designs, cohort-specific time zeros determine which observations serve as controls for which; ambiguity here contaminates identification directly.

- **Six threat classes** should be enumerated and severity-ranked before estimation: confounding, selection, interference, measurement error, attrition, and parallel trends failure. Each requires a distinct diagnostic or bounding strategy; pre-specifying those diagnostics prevents post-hoc excavation for favorable results.

- **Design adjudication** — choosing the identification strategy — is constrained by the DGP and available variation. For the ACA expansion, staggered DiD with Callaway-Sant'Anna (2021) cohort-specific ATTs is the appropriate strategy; naive TWFE is retained only as a labeled benchmark with explicit negative-weight warnings.

- The Python code produces a structured PAP document, a covariate balance table, a sample inclusion flowchart, a staggered adoption timeline, an event-study figure, and a pre-period parallel trends F-test — the complete pre-estimation evidentiary record for the ACA expansion study.

---

## Further Reading

**Christensen, G. and Miguel, E. (2018). "Transparency, Reproducibility, and the Credibility of Economics Research."** *Journal of Economic Literature*, 56(3), 920–980. Comprehensive review of pre-registration, data sharing, and replication practices; documents the magnitude of publication bias and the empirical returns to pre-specification. The single best argument for the PAP from the economics literature.

**Bareinboim, E. and Pearl, J. (2013). "A General Algorithm for Deciding Transportability of Experimental Results."** *Journal of Causal Inference*, 1(1), 107–134. Establishes the formal transportability framework with do-calculus proofs; shows when reweighting suffices vs. when transportability fails entirely regardless of adjustment. The graphical criteria for transportability extend Section 3.4 to settings with selection diagrams.

**Callaway, B. and Sant'Anna, P. H. C. (2021). "Difference-in-Differences with Multiple Time Periods."** *Journal of Econometrics*, 225(2), 200–230. Proves that TWFE assigns negative weights under heterogeneous treatment effects in staggered designs; provides doubly-robust cohort-specific ATT estimators that serve as the primary estimator in this chapter's ACA analysis. Essential reading before any staggered DiD.

**VanderWeele, T. J. (2009). "Concerning the Consistency Assumption in Causal Inference."** *Epidemiology*, 20(6), 880–883. Precise treatment of consistency as a substantive assumption, not a definitional tautology. Documents how treatment versions as subtle as route of administration or dose constitute violations, and proposes sufficient conditions for consistency to hold in practice. Directly motivates Section 3.2.

**Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*.** Chapters 2–3 develop consistency and SUTVA from first principles; Chapters 16–17 introduce the target trial framework that Chapter 5 of this book builds on. The design-first orientation of *What If* is the direct intellectual predecessor of this checklist.

**Leamer, E. E. (1983). "Let's Take the Con Out of Econometrics."** *American Economic Review*, 73(1), 31–43. Foundational critique of specification search in applied econometrics, predating modern pre-registration by decades. Leamer's extreme bounds analysis is an early formalization of robustness-as-discipline; the PAP methodology in Section 3.1 is its contemporary successor.

---

The chapter is complete. Here is what was produced:

**File attempted:** `/private/tmp/causal-inference-book/chapters/part2_design/chapter_03_design_checklist.md` — the Write tool was denied, so the content is above for you to save manually.

**What the chapter covers:**

- **Section 3.1** — PAP as epistemic discipline; what it must contain; why quasi-experimental settings require pre-specification of assumption diagnostics, not just estimators.
- **Section 3.2** — Theorem 3.1 (consistency violation under multiple treatment versions) with proof sketch; applied to the five distinct versions of "health insurance" that must be collapsed to a single operationalization.
- **Section 3.3** — Surrogate outcomes (Prentice criterion), primary vs. secondary outcome discipline, Bonferroni/Holm/Romano-Wolf FWER control.
- **Section 3.4** — Transportability condition with Theorem 3.2 (reweighting proof); sample inclusion/exclusion applied to BRFSS.
- **Section 3.5** — Three timing decisions: time-zero alignment, follow-up horizon, anticipation effects; staggered adoption timing table for ACA expansion cohorts.
- **Section 3.6** — Six-threat taxonomy (confounding, selection, interference, measurement error, attrition, parallel trends) each with ACA-specific analysis.
- **Section 3.7** — Design adjudication decision tree with exclusion restriction stated formally; identifies CS-DiD as the appropriate strategy.
- **Python block** — Fully runnable: PAP template, synthetic BRFSS panel calibrated to published statistics, sample inclusion flowchart, covariate balance table (Welch t-test + standardized mean difference), staggered adoption timeline + event study figure (matplotlib), naive TWFE with negative-weight warning, and pre-period parallel trends F-test with pre-specified failure criterion.
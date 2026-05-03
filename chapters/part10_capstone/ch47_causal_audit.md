# Chapter 47: The Causal Audit — Full Diagnostic Workflow

A causal claim is only as strong as its weakest link in the identification-estimation-inference chain. "Auditing" that chain means applying a structured, exhaustive diagnostic protocol that converts vague credibility judgments into quantifiable, documented evidence. This chapter builds that protocol from first principles, applies it to the Oregon Health Insurance Experiment (OHE) analysis chain, and produces a replicable audit artifact — a machine-readable report that a reviewer, replicator, or future self can interrogate without re-reading the original paper.

The chapter is deliberately forensic in tone. We assume the analysis is already done; the question is whether to believe it.

---

## 47.1 Anatomy of a Causal Audit

An audit is not a reanalysis. It accepts the analyst's primary specification and asks: *given that specification, what must be true for the estimates to be credible, and how close does the evidence come to satisfying those requirements?*

We decompose the audit into six layers, ordered roughly from logically prior to logically posterior:

**Layer 1 — Design audit.** Was the variation in treatment assignment plausibly exogenous? This is the identification question. For an RCT it asks whether randomization was properly executed; for an IV it asks whether exclusion and independence hold; for DiD it asks whether parallel trends is plausible.

**Layer 2 — Balance audit.** In finite samples, even valid randomization produces covariate imbalance. The balance audit quantifies residual imbalance and assesses whether it is large enough to threaten internal validity.

**Layer 3 — Estimation audit.** Given the identification strategy, is the chosen estimator consistent under the assumed DGP? Are the functional form assumptions defensible? Are there better-suited alternatives?

**Layer 4 — Inference audit.** Are the standard errors credible? This covers cluster structure, heteroskedasticity, small-sample corrections, and multiple testing adjustments.

**Layer 5 — Sensitivity audit.** How much unmeasured confounding, violation of instrument exclusion, or departure from parallel trends would be required to overturn the primary finding? This is the quantitative credibility question.

**Layer 6 — External validity audit.** To what population does the estimate apply, and how different is that population from the target of the policy question?

Each layer produces a set of flags: **green** (requirement satisfied), **yellow** (concern, quantified), or **red** (potential fatal flaw). The audit concludes with a flag summary table and a written credibility statement.

---

## 47.2 The Sensitivity Cascade

The sensitivity audit (Layer 5) is the most technically demanding layer. We organize it as a *sensitivity cascade*: a ranked sequence of sensitivity metrics applied to the same estimate, from the most conservative to the least, each reporting a threshold at which the conclusion reverses.

**Definition 47.1 (Sensitivity cascade).** Let $\hat\tau$ be the primary estimate with associated confidence interval $[\hat\tau_L, \hat\tau_U]$. The sensitivity cascade is an ordered tuple

$$\mathcal{S}(\hat\tau) = \left(\Gamma^*, \, E\text{-value}, \, RV_{q,\alpha}\right)$$

where:

- $\Gamma^*$ is the Rosenbaum sensitivity parameter: the minimum odds ratio of unmeasured confounding required to explain away the estimate under the sensitivity model $\Gamma \geq 1$.
- $E\text{-value}$ (VanderWeele–Ding) is the minimum strength of association (on the risk ratio scale) that an unmeasured confounder must have with *both* treatment and outcome, conditional on measured covariates, to fully explain the observed association.
- $RV_{q,\alpha}$ is the robustness value (Cinelli–Hazlett): the percentage of residual variance in treatment and outcome that an omitted variable must explain to move the estimate by a fraction $q$ of its magnitude, or to make the $\alpha$-level confidence interval include zero.

**Theorem 47.1 (E-value lower bound).** For a risk ratio $RR$ with $RR > 1$,

$$E\text{-value} = RR + \sqrt{RR(RR - 1)}.$$

For a confidence limit $RR_{\text{lower}} > 1$, the E-value for the confidence interval is obtained by substituting $RR_{\text{lower}}$.

*Proof sketch.* The bound follows from the sharp sensitivity analysis of VanderWeele and Ding (2017). Under the assumption that unmeasured confounder $U$ satisfies $\text{RR}_{UD} \leq B$ and $\text{RR}_{UY|D} \leq B$, the bias factor is at most $B^2/(2B-1)$. Setting this equal to $RR$ and solving yields the stated expression. $\square$

The cascade is informative because the three metrics capture different threat models. $\Gamma^*$ is appropriate when the sensitivity model is formulated in terms of selection odds (natural for matched designs). The E-value is natural for regression settings. $RV_{q,\alpha}$ is the most actionable for regression: it directly answers how large an omitted variable must be relative to the covariates already in the model.

**Definition 47.2 (Credibility thresholds).** We call an estimate *strongly credible* if $E\text{-value} > 2$, *moderately credible* if $1.5 < E\text{-value} \leq 2$, and *weakly credible* if $E\text{-value} \leq 1.5$. These thresholds are conventions, not theorems; the analyst must justify them in context.

---

## 47.3 The Specification Curve

A single primary specification is a point in a vast space of defensible analytical choices. The specification curve (Simonsohn, Simmons, and Nelson 2020; Steiger and Ventura 2021) maps the distribution of estimates across that space.

**Definition 47.3 (Specification curve).** Let $\mathcal{F}$ be the set of defensible specifications — varying control sets, functional forms, sample restrictions, and estimators — for a given causal question. The specification curve is the function $f: \mathcal{F} \to \mathbb{R}$ mapping each specification to its point estimate. The *summary statistic* of the specification curve is

$$\text{SC}(\hat\tau) = \left(\text{median}_{s \in \mathcal{F}} f(s),\; \frac{|\{s : f(s) \cdot \hat\tau > 0\}|}{|\mathcal{F}|}\right),$$

i.e., the median estimate and the fraction of specifications that share the sign of the primary estimate.

A *robust* finding satisfies two conditions: (i) the median specification-curve estimate is of the same sign and comparable magnitude to the primary, and (ii) at least 80% of specifications share the sign of the primary.

**Theorem 47.2 (Specification curve under null).** Under $H_0: \tau = 0$ and under the maintained assumption that specifications within $\mathcal{F}$ are exchangeable (i.e., no specification is intrinsically privileged), the fraction of specifications with $f(s) > 0$ is approximately $\text{Uniform}[0,1]$ in large samples. A fraction significantly above 0.5 (by a one-sided binomial test) constitutes evidence against the null.

*Remark.* The exchangeability assumption is strong. Specifications are rarely exchangeable; the analyst typically has a well-motivated primary specification. The specification curve is most useful as a robustness check and a diagnostic for researcher degrees of freedom, not as a primary inference tool.

---

## 47.4 Balance Audit: Theory and Thresholds

In a randomized experiment with finite sample, the balance audit quantifies the degree to which the randomization produced covariate balance. In an observational study, it quantifies residual imbalance after the identification strategy (matching, weighting, regression) is applied.

**Definition 47.4 (Standardized mean difference).** For covariate $X$ with treatment group mean $\bar X_1$ and control group mean $\bar X_0$, and pooled standard deviation $S_p = \sqrt{(S_1^2 + S_0^2)/2}$,

$$\text{SMD}(X) = \frac{\bar X_1 - \bar X_0}{S_p}.$$

The rule-of-thumb threshold $|\text{SMD}| < 0.1$ (Austin 2011) designates "negligible" imbalance. Values in $(0.1, 0.2)$ are "small"; values above $0.2$ trigger a yellow flag.

**Theorem 47.3 (SMD and bias).** Under a linear potential outcomes model $Y(d) = \alpha + \tau d + \beta^\top X + \varepsilon$, the omitted variable bias from excluding $X_j$ from the regression is

$$\text{bias}(\hat\tau) = \beta_j \cdot \text{SMD}(X_j) \cdot S_p^{(j)},$$

where $S_p^{(j)}$ is the pooled SD of $X_j$. Thus $\text{SMD}(X_j) < 0.1$ bounds the absolute bias at $0.1 \beta_j S_p^{(j)}$.

*Proof.* Under OLS, omitting $X_j$ biases $\hat\tau$ by $\hat\beta_j \cdot \hat\delta_j$ where $\hat\delta_j$ is the coefficient from regressing $X_j$ on $D$. In a two-group comparison, $\hat\delta_j = \bar X_1^{(j)} - \bar X_0^{(j)}$, which equals $\text{SMD}(X_j) \cdot S_p^{(j)}$. $\square$

The practical implication: even with a large effect of $X_j$ on $Y$ (large $\beta_j$), a small SMD bounds the worst-case bias from that covariate.

---

## 47.5 Power Audit

A power audit is retrospective: given the achieved sample size, what was the minimum detectable effect (MDE), and is the primary estimate plausibly large enough to be detected with adequate power?

**Definition 47.5 (Minimum detectable effect).** For a two-sided test at level $\alpha$ with power $1 - \kappa$, standard error $\hat\sigma$, the MDE is

$$\text{MDE} = (z_{1-\alpha/2} + z_{1-\kappa}) \cdot \hat\sigma,$$

where $z_p$ is the $p$-th quantile of the standard normal. At $\alpha = 0.05$, $\kappa = 0.20$, this gives $\text{MDE} \approx 2.8 \hat\sigma$.

A power audit flags two pathologies:

1. **Underpowered detection**: $|\hat\tau| < \text{MDE}$, meaning the study could not have reliably detected an effect of the magnitude found even if it were real. Statistically significant estimates in this regime are susceptible to the "winner's curse" — upward bias from conditioning on significance.

2. **Overpowered trivial effects**: $|\hat\tau| \gg \text{MDE}$ and $|\hat\tau|$ is substantively negligible. Here the study is large enough to detect effects too small to matter, and statistical significance conflates with practical significance.

---

## 47.6 Applying the Audit to the OHE Analysis Chain

The OHE provides a clean audit target: a lottery-based IV with known assignment mechanism, rich administrative and survey data, and an unusually large replication literature. We reconstruct the audit chain as it would be applied to a completed analysis, covering the IV estimates of insurance take-up on healthcare utilization (doc_any_12m) and financial protection (catastrophic_exp_inp).

### Design Audit

The lottery instrument (selected) satisfies the three IV conditions by construction:

- **Relevance**: The first-stage $F$-statistic is above 1000 in all reported specifications, far above the Staiger-Stock threshold of 10 (and the more stringent Lee et al. 2022 threshold for robust inference of $F > 104.7$).
- **Independence**: Assignment was by lottery conditional on household list size (numhh_list). Conditioning on stratum fixed effects is therefore required; failing to do so introduces correlation between the instrument and unobserved household characteristics.
- **Exclusion**: The lottery selected households for the opportunity to apply; it had no direct effect on health outcomes except through insurance coverage. The main exclusion threat — health behavior changes from *applying* independent of gaining coverage — is theoretically possible but empirically bounded by intent-to-treat estimates for non-lottery-related outcomes.

**Design flag**: Green, with the caveat that stratum fixed effects must be included.

### Balance Audit

Even with valid randomization, the OHE balance table (Finkelstein et al. 2012, Table 1) shows small but non-zero differences. The audit formalizes this as $\text{SMD}$ statistics across pre-randomization covariates.

### Estimation Audit

The primary OHE estimator is 2SLS with stratum fixed effects. The audit checks:
- First-stage $F$-statistic and weak-instrument robust confidence intervals (Anderson-Rubin, conditional likelihood ratio).
- LATE vs. ATE distinction: the 2SLS estimate is the LATE for compliers (those who would take up insurance if and only if selected). Generalizing to the full population requires additional assumptions.
- Functional form: linear probability model for binary outcomes. Alternative: IV probit. The audit checks whether marginal effects differ materially.

### Inference Audit

- Standard errors must be clustered at the household level because selection was by household.
- Multiple outcomes (healthcare utilization, financial outcomes, health status) require a multiple testing correction or pre-specified primary outcome. The audit checks whether the pre-specified primary outcome is clearly identified.

### Sensitivity Audit

The sensitivity cascade is applied to the IV estimate of insurance on doc_any_12m: the fraction who had any doctor visit in the 12-month survey window.

### External Validity Audit

The OHE compliers are Medicaid-eligible low-income adults in Oregon in 2008. The external validity audit documents the extent to which this population differs from: (i) the full Medicaid-eligible population, (ii) those affected by ACA Medicaid expansion, and (iii) a general population policy target.

---

## Python: Full Causal Audit of the OHE Analysis Chain
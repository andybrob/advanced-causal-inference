# Advanced Causal Inference: A Graduate Econometrician's Field Guide

**Running Example**: *What is the causal effect of health insurance access on health and financial outcomes?*

---

## Part I — The Causal Inference Mindset After Graduate Econometrics

---

### Chapter 1: The Identification-First Principle

#### Core Question
Why does identification logic precede estimation choice, and what separates credible from spurious causal claims?

#### Key Concepts
- The fundamental problem of causal inference
- Potential outcomes notation (Rubin) vs. structural equations (Pearl)
- Estimands vs. estimators vs. estimates
- The identification-estimation-inference triad
- What graduate econometrics training leaves underspecified
- Credibility revolution in empirical economics
- The asymmetry between ruling in and ruling out confounding

#### Key Mathematical Results
- **Stable Unit Treatment Value Assumption (SUTVA)**: formalized as $Y_i = Y_i(d)$ when $D_i = d$, no interference, no hidden versions
- **Consistency**: $Y_i = Y_i(D_i)$ linking potential and observed outcomes
- **ATE decomposition**: $\mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]$, nonidentified without further assumptions
- **Bias decomposition**: $\mathbb{E}[Y^{obs}|D=1] - \mathbb{E}[Y^{obs}|D=0] = ATE + \text{selection bias} + \text{heterogeneity bias}$

#### Running Example Application
Introduce the Oregon Health Insurance Experiment lottery as the motivating dataset; establish why a naive income-insured comparison fails and what identifying variation the lottery provides.

#### Python Implementation Notes
- Load OHE data via NBER; inspect treatment assignment, outcomes (`doc_any_12m`, `catastrophic_exp_inp`)
- Compute naive OLS gap and decompose bias heuristically
- `pandas`, `statsmodels.formula.api`, `linearmodels`
- Output: a table showing naive vs. lottery-based estimates side by side

#### Connections
Builds from nothing; enables all subsequent identification arguments. Returns to bias decomposition in Chapter 11 (propensity scores) and Chapter 31 (sensitivity analysis).

---

### Chapter 2: Estimands — ATEs, CATEs, LATEs, and Policy Values

#### Core Question
What exactly are we trying to estimate, and how does the choice of estimand shape every downstream decision?

#### Key Concepts
- Population ATE, ATT, ATC — when each is policy-relevant
- Conditional average treatment effect (CATE) and heterogeneity
- Local average treatment effect (LATE) and its population
- Policy value and optimal policy estimands
- Weighted estimands and their implicit populations
- Estimand-design alignment
- Pre-registration of estimands

#### Key Mathematical Results
- **ATT vs. ATE**: $ATT = \mathbb{E}[Y_i(1)-Y_i(0)|D_i=1]$; differs from ATE under heterogeneity
- **LATE (Wald)**: $LATE = \mathbb{E}[Y_i(1)-Y_i(0)|D_i(1)>D_i(0)]$, the complier subpopulation
- **Policy value**: $V(\pi) = \mathbb{E}[\pi(X_i)Y_i(1) + (1-\pi(X_i))Y_i(0)]$ for rule $\pi$
- **CATE identification**: $\tau(x) = \mathbb{E}[Y_i(1)-Y_i(0)|X_i=x]$ under unconfoundedness

#### Running Example Application
Map each estimand to a policy question in the OHE: ATE for universal coverage debate, ATT for expansion of Medicaid to current enrollees, LATE for lottery compliers, CATE for heterogeneous subgroup targeting.

#### Python Implementation Notes
- Manually compute ATT vs. ATE under simulated heterogeneity
- Show how reweighting changes estimate magnitude
- `numpy`, `scipy`, `pandas`
- Output: estimand taxonomy table with OHE policy interpretation column

#### Connections
Builds on Chapter 1's identification framing; directly governs choices in Chapters 13–14 (meta-learners, policy learning) and Chapter 45 (ATE to ROI).

---

### Chapter 3: The Causal Design Checklist

#### Core Question
What systematic questions must an analyst answer before touching data, and how do design choices constrain valid inference?

#### Key Concepts
- The pre-analysis plan as epistemic discipline
- Treatment definition sharpness
- Outcome selection and surrogate outcomes
- Sample inclusion/exclusion criteria and generalizability
- Timing: treatment onset, washout, follow-up horizon
- Threat taxonomy: confounding, selection, interference, measurement error
- Design adjudication: observational vs. quasi-experimental vs. experimental

#### Key Mathematical Results
- **Consistency violation**: multiple treatment versions $d', d''$ with $Y_i(d') \neq Y_i(d'')$ break identification
- **Transportability condition**: $P(Y(d)|X) = P^*(Y(d)|X)$ required for external validity
- **Exclusion restriction**: $Z \perp\!\!\!\perp Y(d)$ for valid IV; stated here, derived in Chapter 21
- **No-interference**: formalized in SUTVA; interference handled in Chapter 41

#### Running Example Application
Walk through the full design checklist applied to the ACA Medicaid expansion study: define "insurance access," select BRFSS outcomes, specify pre/post windows, enumerate confounders.

#### Python Implementation Notes
- Template Jupyter notebook: design document as structured markdown cells
- Codebook inspection utilities (`pandas-profiling`, `ydata-profiling`)
- Covariate balance tabulation pre-estimation
- Output: a reproducible design document embedded in notebook

#### Connections
Builds on Chapters 1–2; serves as the diagnostic frame for every empirical chapter. Revisited explicitly in Chapter 47 (Causal Audit).

---

### Chapter 4: Graphs for Econometricians — DAGs, SWIGs, and Selection Diagrams

#### Core Question
How do graphical models formalize identification, and what do they reveal that potential outcomes notation obscures?

#### Key Concepts
- Directed Acyclic Graphs (DAGs): nodes, edges, d-separation
- Backdoor criterion and adjustment sets
- Frontdoor criterion
- Single World Intervention Graphs (SWIGs): unifying DAGs and potential outcomes
- Selection diagrams and sample selection bias
- Collider bias: when conditioning induces confounding
- M-bias and when "more controls" hurts

#### Key Mathematical Results
- **Backdoor criterion**: $P(Y|do(X)) = \sum_z P(Y|X,Z)P(Z)$ when $Z$ blocks all backdoor paths
- **Frontdoor formula**: multi-step identification via mediator
- **d-separation**: $X \perp\!\!\!\perp Y | Z$ in graph iff all paths blocked; implies conditional independence in distribution
- **Do-calculus rules (Pearl)**: three rules governing $do()$ operator manipulations

#### Running Example Application
Draw the DAG for insurance on health outcomes: lottery $Z$ → enrollment $D$ → outcomes $Y$, with income, age, health status as confounders; identify backdoor paths closed by lottery design.

#### Python Implementation Notes
- `pgmpy` or `causalgraphicalmodels` for DAG specification
- `daggity`-style d-separation queries via Python wrappers
- Visualize adjustment sets; show collider bias simulation
- Output: DAG figure + d-separation test results for OHE structure

#### Connections
Builds on Chapters 1–3; provides graphical language used throughout the book. Critical for Chapter 25 (time-varying confounding) and Chapter 36 (transportability).

---

## Part II — Design Before Estimation

---

### Chapter 5: Target Trial Emulation

#### Core Question
How should an observational study be structured to mimic the randomized trial that would answer the causal question?

#### Key Concepts
- The target trial framework (Hernán–Robins)
- Protocol specification: eligibility, treatment strategies, outcomes, follow-up, estimand
- Per-protocol vs. intention-to-treat analysis
- Cloning-censoring-weighting technique
- Time zero alignment
- The observational analog of each protocol component
- Active comparator design

#### Key Mathematical Results
- **Censoring weights**: IPCW $w_i(t) = \prod_{k \leq t} \frac{1}{P(C_k=0|C_{k-1}=0, \bar{L}_k, A_k)}$
- **Standardization**: $E[Y^{a}] = \sum_l E[Y|A=a, L=l]P(L=l)$
- **Consistency under protocol**: formal equivalence between target trial estimand and emulation estimand under stated assumptions
- **Clone-censor-weight identity**: algebraic equivalence of cloning approach to MSM weighting

#### Running Example Application
Specify the target trial for the OHE: eligibility = uninsured Oregon adults; treatment = Medicaid enrollment; follow-up = 12 months; outcome = catastrophic medical expenditure.

#### Python Implementation Notes
- `lifelines` for survival analysis setup
- Manual cloning/censoring procedure in `pandas`
- IPCW computation with logistic nuisance models
- Output: Kaplan-Meier curves under emulated trial protocol

#### Connections
Builds on Chapter 3 (design checklist); required conceptual foundation for Chapter 25–29 (g-methods) where repeated treatment assignment demands this rigor.

---

### Chapter 6: Treatment Timing, Time Zero, and Immortal Time Bias

#### Core Question
How do errors in defining treatment onset invalidate causal estimates, and how are they corrected?

#### Key Concepts
- Immortal time bias: person-time before treatment misclassified
- Time zero misalignment
- Latent index of treatment uptake
- Prevalent user bias vs. new user design
- Depletion of susceptibles
- Time-varying covariates and their role in defining time zero
- Drug utilization study principles

#### Key Mathematical Results
- **Immortal time bias magnitude**: shown as a function of the misclassified interval length and event rate
- **Hazard ratio distortion**: analytic result showing direction and magnitude of bias
- **New-user estimand**: restriction to $T_{enrollment} = 0$ and its effect on estimand population
- **Lag-time sensitivity**: formal sensitivity parameter for unknown latency

#### Running Example Application
Apply to ACA Medicaid expansion: define time zero as state expansion date; show how pre-expansion coverage shifts create immortal-time-analogous bias if poorly handled.

#### Python Implementation Notes
- Simulate immortal time bias with synthetic panel data
- Construct correct person-time datasets using `pandas` wide-to-long transformations
- `lifelines.CoxPHFitter` with time-varying covariates
- Output: bias magnitude plot as function of misclassification window

#### Connections
Builds on Chapter 5; directly relevant to Chapter 27 (MSMs) and Chapter 29 (dynamic treatment regimes) where time-zero precision is critical.

---

### Chapter 7: Negative Controls, Placebos, and Falsification Tests

#### Core Question
How do non-causal tests provide affirmative evidence that a causal design is valid?

#### Key Concepts
- Negative control outcomes: outcomes that should not respond
- Negative control exposures: exposures that share confounders but not mechanism
- Placebo treatments and placebo periods
- Pre-trend tests as negative controls
- Calibration of falsification tests
- The proximal causal inference framework (Miao–Tchetgen)
- Bias amplification from wrong negative controls

#### Key Mathematical Results
- **Proximal identification**: under two negative control variables $W, Z$, $\tau$ identified via bridge function equation $\mathbb{E}[Y(0)|W] = \mathbb{E}[h(Z,X)Y|X]$
- **Rosenbaum's design sensitivity**: $\tilde{\Gamma}$ as the sensitivity parameter a study can detect
- **Pre-trend coefficient null**: $\hat{\delta}_{pre} \approx 0$ as implication of parallel trends
- **Negative control bias bound**: $|bias| \leq f(\text{negative control association strength})$

#### Running Example Application
In OHE: use mortality (low 12-month incidence) as negative control outcome; use self-reported happiness as negative control for financial outcomes; test lottery balance on pre-lottery characteristics.

#### Python Implementation Notes
- Covariate balance tests post-lottery; randomization check
- Placebo treatment assignment permutation test
- `scipy.stats` permutation infrastructure
- Output: balance table, negative control outcome estimates with confidence intervals

#### Connections
Builds on Chapters 3–4; sharpens credibility of designs in Parts III–V; revisited systematically in Chapter 34 (placebo test system) and Chapter 31 (sensitivity analysis).

---

### Chapter 8: Measurement Error, Proxy Treatments, and Proxy Outcomes

#### Core Question
How does classical and non-classical measurement error bias causal estimates, and what are the identification fixes?

#### Key Concepts
- Classical measurement error in treatment: attenuation bias
- Non-classical measurement error: direction-indeterminate bias
- Proxy treatments and partial identification
- Differential misclassification in outcomes
- Errors-in-variables instrumental variables
- Deconvolution approaches
- Proxy outcomes and surrogate endpoints

#### Key Mathematical Results
- **Attenuation bias**: $\hat{\beta}_{OLS} \xrightarrow{p} \beta \cdot \frac{\sigma^2_D}{\sigma^2_D + \sigma^2_\epsilon}$; reliability ratio formula
- **IV correction**: using repeated measurements as instruments for measurement-error-free IV estimator
- **Surrogate sufficiency condition**: $Y(d) \perp\!\!\!\perp S(d) | D$ breaks surrogate validity; Prentice criterion
- **SIMEX**: simulation extrapolation estimator for bias correction

#### Running Example Application
OHE self-reported insurance status vs. administrative records; show how self-report misclassification attenuates IV estimates; use administrative enrollment as instrument for reported coverage.

#### Python Implementation Notes
- Simulate attenuation bias under varying reliability ratios
- SIMEX implementation from scratch in `numpy`
- `statsmodels` IV2SLS with mismeasured treatment
- Output: attenuation bias plot, SIMEX-corrected estimates

#### Connections
Builds on Chapter 1 (identification assumptions); interacts with Chapter 21 (IV) for correction strategies and Chapter 31 (sensitivity analysis) for bounding residual error.

---

## Part III — Selection on Observables, Done Seriously

---

### Chapter 9: Regression Adjustment Revisited

#### Core Question
When is regression a valid estimator of causal effects, and what does misspecification cost?

#### Key Concepts
- Unconfoundedness (ignorability, selection on observables)
- Linear regression as a weighted estimator — what weights it implicitly uses
- Extrapolation and lack of overlap: when OLS fails silently
- Frisch-Waugh-Lovell and partial regression
- Heteroskedasticity-robust and cluster-robust inference
- Regression as approximation to CATE
- Overlap condition: positivity

#### Key Mathematical Results
- **Unconfoundedness**: $Y_i(d) \perp\!\!\!\perp D_i | X_i$ for all $d$
- **OLS implicit weights**: $\hat{\tau}_{OLS} = \int \hat{\tau}(x) \omega(x) dP(x)$ where $\omega(x) = \frac{Var(D|X=x)}{E[Var(D|X)]}$; Angrist (1998)
- **Frisch-Waugh-Lovell**: $\hat{\beta}_D$ from long regression equals coefficient from regressing $\tilde{Y}$ on $\tilde{D}$ (residuals on residuals)
- **Overlap (positivity)**: $0 < P(D=1|X=x) < 1$ a.s.; violation causes extrapolation

#### Running Example Application
Regress OHE health outcomes on Medicaid enrollment controlling for age, income, pre-lottery health; show covariate-conditional estimates and overlap diagnostics.

#### Python Implementation Notes
- `statsmodels.OLS` with HC3 and cluster-robust SE
- Overlap diagnostics: propensity score histogram by treatment arm
- `matplotlib` for regression residual plots
- Output: coefficient table, overlap plot, Frisch-Waugh residual scatter

#### Connections
Builds on Chapter 1 (unconfoundedness); motivates Chapter 10 (propensity scores) as an alternative weighting scheme; contrast with Chapter 12 (DML) for nonparametric regression.

---

### Chapter 10: Propensity Scores Without Ritual

#### Core Question
What does the propensity score theorem actually guarantee, and when does propensity-score weighting fail?

#### Key Concepts
- Rosenbaum–Rubin balancing score theorem
- Inverse probability weighting (IPW): Horvitz-Thompson estimator
- Hajek (normalized) vs. Horvitz-Thompson IPW
- Propensity score trimming and overlap weights
- Machine learning for propensity estimation: dangers and disciplines
- Covariate balance vs. propensity score balance
- Matching: nearest-neighbor, caliper, genetic matching

#### Key Mathematical Results
- **Balancing score theorem**: $Y(d) \perp\!\!\!\perp D | e(X)$ where $e(X) = P(D=1|X)$; dimension reduction
- **IPW (Horvitz-Thompson)**: $\hat{\tau}_{IPW} = \frac{1}{n}\sum_i \frac{D_i Y_i}{\hat{e}(X_i)} - \frac{(1-D_i)Y_i}{1-\hat{e}(X_i)}$
- **Hajek normalization**: self-normalized weights; reduces finite-sample variance
- **Overlap weights**: $w(x) = e(x)(1-e(x))$; targets population with clinical equipoise (Li et al.)

#### Running Example Application
Estimate propensity for Medicaid enrollment using pre-lottery covariates in OHE; compare HT-IPW, Hajek, and overlap-weighted estimates of medical utilization outcomes.

#### Python Implementation Notes
- `sklearn.LogisticRegression` for propensity model; GBRT alternative
- `zepid` or manual IPW computation
- LOVE plots for covariate balance via `tableone`
- Output: LOVE plot, weight distribution histogram, sensitivity to trimming threshold

#### Connections
Builds on Chapters 9; motivates the doubly robust estimator in Chapter 11 by showing that propensity misspecification alone is fatal.

---

### Chapter 11: Doubly Robust Estimation

#### Core Question
How can we combine an outcome model and a propensity model so that misspecification of either one (but not both) leaves us consistent?

#### Key Concepts
- Augmented inverse probability weighting (AIPW)
- Double robustness: the two chances at consistency
- Influence function representation of AIPW
- Semiparametric efficiency bound
- Nonparametric plug-in and its bias: $n^{-1/2}$ vs. $n^{-1/4}$ convergence
- Product bias: why slow nuisance rates still yield fast causal rate
- Targeted maximum likelihood estimation (TMLE) as a doubly robust alternative

#### Key Mathematical Results
- **AIPW estimator**: $\hat{\tau}_{AIPW} = \frac{1}{n}\sum_i \left[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{D_i(Y_i-\hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-D_i)(Y_i-\hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}\right]$
- **Double robustness**: $E[\hat{\tau}_{AIPW}] \to \tau$ if either $\hat{\mu}$ or $\hat{e}$ is consistent
- **Semiparametric efficiency bound**: $V^* = E[\text{IF}^2]$ where IF is the efficient influence function; AIPW achieves this
- **Product bias**: remainder $= O(\|\hat{\mu}-\mu\| \cdot \|\hat{e}-e\|)$; both nuisances at $n^{-1/4}$ suffices

#### Running Example Application
Apply AIPW to OHE: use logistic regression for propensity, LASSO for outcome; compare to IPW and OLS; show robustness by deliberately misspecifying one model.

#### Python Implementation Notes
- Manual AIPW loop in `numpy`/`pandas`
- Bootstrap variance estimation; analytical EIF-based SE
- `econml.dr.LinearDRLearner` for structured implementation
- Output: ATE estimate table (OLS, IPW, AIPW), misspecification robustness simulation

#### Connections
Builds on Chapters 9–10; directly extended by Chapter 12 (DML replaces parametric nuisances with ML) and Chapter 13 (AIPW scaffold for meta-learners).

---

### Chapter 12: Double/Debiased Machine Learning

#### Core Question
How can machine learning nuisance estimators be incorporated into semiparametrically efficient causal estimators without introducing regularization bias?

#### Key Concepts
- Neyman orthogonality: score functions insensitive to nuisance perturbations
- Cross-fitting: sample splitting to avoid overfitting contamination
- The Robinson (1988) partial linear model
- DML for partially linear, interactive, and IV models
- Regularization bias and why naive ML plug-in fails
- Riesz representers and nuisance functionals
- Rate conditions: $n^{-1/4}$ for nuisance, $n^{-1/2}$ for target

#### Key Mathematical Results
- **Neyman orthogonality condition**: $\partial_\eta E[\psi(W;\theta_0,\eta)]|_{\eta=\eta_0} = 0$; score insensitive to nuisance direction
- **Robinson transformation**: $\tilde{Y}_i = Y_i - \hat{m}(X_i)$, $\tilde{D}_i = D_i - \hat{\ell}(X_i)$; then $\hat{\theta} = (\sum \tilde{D}_i^2)^{-1}\sum \tilde{D}_i \tilde{Y}_i$
- **Cross-fitting variance**: $\hat{\theta}_{DML} = \frac{1}{K}\sum_{k=1}^K \hat{\theta}^{(k)}$; achieves semiparametric efficiency
- **Convergence rate**: $\sqrt{n}(\hat{\theta}_{DML}-\theta_0) \xrightarrow{d} N(0, V^*)$ under nuisance rates $o(n^{-1/4})$

#### Running Example Application
Apply DML to OHE: estimate effect of Medicaid on medical utilization using LASSO/random forest nuisances for confounders; compare to parametric AIPW and OLS.

#### Python Implementation Notes
- `econml.dml.DML`, `LinearDML`, `NonParamDML`
- Manual 5-fold cross-fitting with `sklearn` pipeline
- Hyperparameter tuning via `sklearn.model_selection`
- Output: DML vs. OLS coefficient comparison, nuisance model diagnostics (residual plots)

#### Connections
Builds on Chapter 11 (doubly robust, EIF); extends naturally to Chapter 13 (meta-learners use same cross-fitting) and Chapter 22 (DML for IV).

---

### Chapter 13: Meta-Learners for Heterogeneous Treatment Effects

#### Core Question
How can flexible ML estimators be combined to estimate CATEs, and when does each meta-learner outperform the others?

#### Key Concepts
- S-learner, T-learner, X-learner, R-learner
- Bias-variance tradeoffs across learner families
- X-learner for unbalanced treatment groups
- R-learner loss: Robinson transformation + second-stage regression
- Local centering in causal forests as a special R-learner
- Cross-fitting in meta-learners
- Evaluation of CATEs: QINI curves, RATE, calibration

#### Key Mathematical Results
- **R-learner loss**: $\hat{\tau} = \arg\min_\tau \sum_i [(\tilde{Y}_i - \tau(X_i)\tilde{D}_i)^2 + \lambda\|\tau\|]$; derived from Robinson transformation
- **X-learner imputation**: $\hat{\tau}_0(x) = \mu_1(x) - Y_i(0)$ for treated; $\hat{\tau}_1(x) = Y_i(1) - \mu_0(x)$ for control; combined via propensity weighting
- **Oracle MSE decomposition**: $MSE(\hat{\tau}) = \text{Bias}^2 + \text{Variance}$; T-learner has high bias under unbalance
- **RATE metric**: $RATE(\hat{\tau}) = \mathbb{E}[\tau(X) | \text{top-}q\%] - ATE$; measures heterogeneity capture

#### Running Example Application
Estimate heterogeneous effects of Medicaid on financial hardship across income terciles and health status in OHE; compare S/T/X/R learner estimates; visualize CATE distribution.

#### Python Implementation Notes
- `econml.metalearners`: `SLearner`, `TLearner`, `XLearner`
- `econml.dml.CausalForestDML` as R-learner implementation
- QINI curve and RATE via `econml.cate_interpreter`
- Output: CATE violin plots by subgroup, QINI curve, learner comparison table

#### Connections
Builds on Chapters 11–12 (doubly robust and DML foundations); enables Chapter 14 (policy learning) and Chapter 38 (causal forests in depth).

---

### Chapter 14: Policy Learning and Treatment Rules

#### Core Question
Given heterogeneous treatment effect estimates, how do we construct and evaluate optimal treatment allocation rules?

#### Key Concepts
- Empirical welfare maximization (EWM)
- Doubly robust policy value estimator
- Policy trees: interpretable rules
- Regret bounds and sample complexity
- Class-constrained policy optimization
- Multi-action and continuous-action extensions
- Off-policy evaluation (OPE) and doubly robust scoring

#### Key Mathematical Results
- **Policy value (DR estimate)**: $\hat{V}(\pi) = \frac{1}{n}\sum_i \Gamma_i \cdot \mathbf{1}[\pi(X_i)=1]$ where $\Gamma_i$ is the AIPW pseudo-outcome
- **Regret bound**: $V(\pi^*) - V(\hat{\pi}) = O_p(n^{-1/2})$ for EWM over VC class $\Pi$; Athey-Wager (2021)
- **Policy tree depth-$d$ regret**: $O(\sqrt{d \log(np) / n})$ over depth-$d$ trees
- **OPE consistency**: DR-based OPE is $\sqrt{n}$-consistent and semiparametrically efficient

#### Running Example Application
Construct a targeting rule for Medicaid expansion: who benefits most from enrollment? Use OHE CATEs to define a binary policy tree for targeting by age, income, pre-existing conditions.

#### Python Implementation Notes
- `econml.policy.PolicyTree`, `DRPolicyTree`
- Policy value estimates with bootstrap CI
- `graphviz` for tree visualization
- Output: policy tree figure, estimated value uplift vs. universal enrollment

#### Connections
Builds on Chapter 13 (CATEs); connects to Chapter 43 (business applications) and Chapter 45 (ATE to ROI decision framework).

---

## Part IV — Difference-in-Differences, Event Studies, and Panels

---

### Chapter 15: Difference-in-Differences — The Modern Core

#### Core Question
What does the canonical 2×2 DiD estimate, what assumptions justify it, and where does the classic treatment break down?

#### Key Concepts
- 2×2 DiD as a within-group, within-period double-difference
- Parallel trends assumption: statement and testability
- Strict exogeneity in panels
- Anticipation effects
- Heterogeneous treatment timing setup
- Functional form sensitivity: log vs. levels
- Inference: clustering standard errors

#### Key Mathematical Results
- **2×2 DiD estimator**: $\hat{\tau}^{DiD} = (\bar{Y}_{treated,post} - \bar{Y}_{treated,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$
- **TWFE equivalence**: $\hat{\tau}^{TWFE}$ from $Y_{it} = \alpha_i + \gamma_t + \tau D_{it} + \varepsilon_{it}$ equals $\hat{\tau}^{DiD}$ in 2×2
- **Parallel trends (strict)**: $E[Y_{it}(0) - Y_{is}(0)|G_i=g] = E[Y_{it}(0) - Y_{is}(0)|G_i=g']$ for all $t,s,g,g'$
- **Anticipation condition**: $Y_{it}(d) = Y_{it}(0)$ for $t < $ treatment date

#### Running Example Application
Compare pre/post health outcomes in Medicaid-expansion vs. non-expansion states using BRFSS; implement TWFE with state and year FE; document identifying variation.

#### Python Implementation Notes
- `linearmodels.PanelOLS` with entity and time fixed effects
- Cluster-robust SE by state (`cov_type='clustered'`)
- Pre-period trend visualization
- Output: TWFE coefficient, pre-period parallel trends plot, event study setup

#### Connections
Builds on Chapter 3 (design); motivates Chapters 16–17 by demonstrating TWFE failure under staggered adoption.

---

### Chapter 16: Staggered Adoption and the TWFE Problem

#### Core Question
Why does two-way fixed effects fail under staggered treatment timing, and what is the decomposition that shows it?

#### Key Concepts
- Staggered adoption: units treated at different calendar times
- Goodman-Bacon decomposition
- "Forbidden comparisons": late-treated as controls for early-treated
- Negative weighting in TWFE
- Heterogeneous treatment effects over time
- Parallel trends under staggered adoption
- The clean-comparison principle

#### Key Mathematical Results
- **Bacon decomposition**: $\hat{\tau}^{TWFE} = \sum_{k,\ell} s_{k\ell} \hat{\tau}^{2\times2}_{k\ell}$; weights $s_{k\ell}$ can be negative
- **Negative weight condition**: weight on $(k,\ell)$ 2×2 is negative when treatment effect grows over time
- **TWFE bias formula**: $E[\hat{\tau}^{TWFE}] = \tau + \text{bias}$ where bias $\neq 0$ under effect heterogeneity
- **Relative weight on forbidden comparisons**: explicit formula as function of group sizes and timing

#### Running Example Application
Apply Goodman-Bacon decomposition to ACA expansion data (2010–2016): visualize the 2×2 components and their weights; identify which state comparisons receive negative weight.

#### Python Implementation Notes
- `bacondecomp` Python package or manual implementation
- Weight visualization scatter plot (2×2 ATT vs. weight)
- `linearmodels` TWFE as baseline comparison
- Output: Bacon decomposition plot, table of negative-weight comparisons

#### Connections
Builds on Chapter 15; directly motivates Chapter 17 (modern estimators); connects to Chapter 18 (event studies) for visual diagnosis of the problem.

---

### Chapter 17: Modern DiD Estimators

#### Core Question
How do Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, and Gardner estimators restore valid identification under staggered adoption?

#### Key Concepts
- Group-time ATTs: $ATT(g,t)$ as the building block
- Aggregation: simple, event-study, group-weighted
- Callaway-Sant'Anna: doubly robust group-time ATTs
- Sun-Abraham: interaction-weighted estimator
- Borusyak-Jaravel-Spiess: imputation estimator
- Gardner (2021): two-stage DiD
- "Never-treated" vs. "not-yet-treated" comparison groups

#### Key Mathematical Results
- **Group-time ATT**: $ATT(g,t) = E[Y_t(g) - Y_t(0)|G=g]$ for group $g$ treated at $t=g$
- **Callaway-Sant'Anna DR estimand**: doubly robust group-time ATT using both propensity and outcome model
- **Sun-Abraham decomposition**: $\hat{\tau}^{TWFE} = \sum_g \sum_\ell \hat{\delta}_{g\ell} \cdot CATT(g,\ell)$ with explicit heterogeneity weights
- **BJS imputation**: $\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)$ using control-group-imputed counterfactual $\hat{Y}_{it}(0)$

#### Running Example Application
Estimate effect of ACA Medicaid expansion on uninsured rates using all four modern estimators; aggregate to event-study and simple ATT; compare to naive TWFE.

#### Python Implementation Notes
- `csdid` Python package (Callaway-Sant'Anna)
- `pyfixest` for BJS imputation
- `sunab` via `statsmodels` extensions or manual interaction coding
- Output: side-by-side ATT estimates, event-study plots for all four estimators

#### Connections
Builds on Chapters 15–16; Chapter 18 extends to event study presentation; Chapter 19 introduces synthetic control as an alternative when control groups are scarce.

---

### Chapter 18: Event Studies as Evidence, Not Decoration

#### Core Question
How should event studies be specified, interpreted, and stress-tested to provide genuine causal evidence rather than visual persuasion?

#### Key Concepts
- Event study coefficients as pre/post difference profiles
- Normalization and the reference period choice
- Pre-trend testing: joint F-test vs. visual inspection
- Anticipation in event windows
- Confidence band construction
- Aggregation of group-time ATTs into event-study format
- Sensitivity of event-study shape to estimator choice

#### Key Mathematical Results
- **Event-study estimand**: $\beta_k = E[Y_{i,t_0+k}(1) - Y_{i,t_0+k}(0)]$ for event-time $k$
- **Pre-trend test**: $H_0: \beta_{-K} = \cdots = \beta_{-1} = 0$; joint Wald test with clustered SE
- **Honest pre-trend inference**: Rambachan-Roth (2023) sensitivity: bounds on post-period effects given bounded pre-trend violations
- **Stacking estimator**: event-study via stacked dataset of clean 2×2 comparisons

#### Running Example Application
Plot ACA expansion event study for insurance coverage and self-reported health using BRFSS; apply Rambachan-Roth sensitivity to bound estimates under non-parallel pre-trend.

#### Python Implementation Notes
- `pyfixest` event study syntax
- `HonestDiD` Python wrapper (Rambachan-Roth sensitivity)
- `matplotlib` with confidence bands and reference-period annotation
- Output: event study figure with honest sensitivity bands

#### Connections
Builds on Chapters 15–17; provides the visualization layer for all DiD designs; Rambachan-Roth connects to Chapter 31 (sensitivity analysis).

---

### Chapter 19: Synthetic Control and Matrix Completion

#### Core Question
When treated units are few and control groups are many, how do synthetic control and matrix completion reconstruct the counterfactual?

#### Key Concepts
- Abadie-Diamond-Hainmueller synthetic control
- Convex combination donor pool weights
- Pre-period fit and interpolation bias
- Inference via permutation (placebo in space)
- Synthetic DiD (SDID): combining SC and DiD
- Matrix completion (MC-NNM): nuclear norm minimization
- SDID as a doubly robust extension of SC

#### Key Mathematical Results
- **SC estimand**: $\hat{\tau}_t = Y_{1t} - \sum_{j \geq 2} \hat{w}_j Y_{jt}$; weights $\hat{w}$ minimize pre-period prediction error
- **Convexity constraint**: $\hat{w} \in \Delta^{J}$ (simplex); interpolation not extrapolation
- **MC-NNM**: $\min_{L} \|P_\Omega(Y - L)\|_F^2 + \lambda \|L\|_*$; nuclear norm regularization
- **SDID**: $(\hat{\tau}, \hat{\alpha}, \hat{\beta}, \hat{\delta}) = \arg\min \sum_{it} (Y_{it} - \alpha_i - \beta_t - \tau W_{it} - \delta)^2 \omega_i^{SC} \lambda_t^{SC}$

#### Running Example Application
Apply synthetic control to a single state's Medicaid expansion (e.g., Oregon 2014) using non-expansion states as donors; replicate with MC-NNM; compare SDID to TWFE.

#### Python Implementation Notes
- `synth` Python package or manual constrained optimization via `scipy.optimize`
- `sdid` Python port
- `fancyimpute` or custom MC-NNM with `cvxpy`
- Output: synthetic control plot, placebo distribution, SDID comparison table

#### Connections
Builds on Chapters 15–17; SC inference connects to Chapter 34 (placebo tests); SDID connects to Chapter 11 (doubly robust ideas).

---

### Chapter 20: Panel Robustness and Sensitivity

#### Core Question
How do we stress-test panel and DiD estimates against violations of parallel trends, treatment anticipation, and functional form?

#### Key Concepts
- Parallel trends sensitivity (Rambachan-Roth framework)
- Relaxing parallel trends: conditional parallel trends
- Anticipation and leads
- Synthetic control placebo inference
- Wild cluster bootstrap for small-cluster inference
- Aggregation choices and their effects on estimates
- Specification curve for panel models

#### Key Mathematical Results
- **Rambachan-Roth sensitivity**: under $|\Delta PT| \leq M$, identified set for $\beta_{post}$ is an interval shrinking with $M$
- **Wild cluster bootstrap validity**: valid under few clusters with correct asymptotic refinements; MacKinnon-Webb conditions
- **Conditional parallel trends**: $E[Y_{it}(0)-Y_{is}(0)|X_i, G_i=g] = E[Y_{it}(0)-Y_{is}(0)|X_i, G_i=g']$
- **Manski bounds applied to DiD**: partial identification when PT is not point-identifying

#### Running Example Application
Apply Rambachan-Roth sensitivity to ACA expansion estimates; show how confidence sets widen as parallel trends tolerance increases; implement wild cluster bootstrap with few states.

#### Python Implementation Notes
- `HonestDiD` Python package
- Wild cluster bootstrap via `wildboottest`
- Specification curve over aggregation schemes with `pandas` + `matplotlib`
- Output: sensitivity plot, bootstrap distribution, specification curve figure

#### Connections
Builds on Chapters 15–19; connects to Chapter 31 (general sensitivity analysis) and Chapter 33 (partial identification).

---

## Part V — Instruments, Discontinuities, and Quasi-Experimental Leverage

---

### Chapter 21: Instrumental Variables Beyond 2SLS

#### Core Question
What does IV identify under heterogeneous treatment effects, when are instruments valid, and how do we diagnose weak instruments?

#### Key Concepts
- LATE identification under monotonicity
- Monotonicity assumption: statement, plausibility, testability
- Weak instruments: first-stage F, concentration parameter
- Many-instrument bias and LIML, HLIM, JIVE
- Shift-share (Bartik) instruments and the Goldsmith-Pinkham critique
- Judge-leniency and examiner designs
- Robust IV inference: Anderson-Rubin, conditional likelihood ratio

#### Key Mathematical Results
- **LATE**: $\hat{\tau}_{IV} \xrightarrow{p} \frac{E[Y_i(1)-Y_i(0)|D_i(1)>D_i(0)]}{E[D_i(1)-D_i(0)]}$ under monotonicity
- **2SLS numerator**: $\hat{\tau}_{2SLS} = (Z'X)^{-1}Z'Y$; in IV with one instrument reduces to Wald
- **Weak IV (Staiger-Stock)**: $F < 10$ (rule of thumb); formal size distortion analysis via concentration parameter $\mu^2$
- **Shift-share identification**: Goldsmith-Pinkham condition — identification from industry shares, not national shocks

#### Running Example Application
Use OHE lottery as the instrument for Medicaid enrollment; estimate LATE; test monotonicity via complier subpopulation analysis; diagnose first-stage strength.

#### Python Implementation Notes
- `linearmodels.IV2SLS`, `IVLIML`
- First-stage F-statistics and weak-IV robust AR test
- Complier covariate analysis via `numpy` conditioning
- Output: first-stage table, LATE estimate, AR confidence interval

#### Connections
Builds on Chapter 1 (LATE definition) and Chapter 2 (estimands); extends to Chapter 22 (MTE) and Chapter 24 (encouragement designs).

---

### Chapter 22: Marginal Treatment Effects and Selection Models

#### Core Question
How do marginal treatment effects unify IV estimates and reveal treatment effect heterogeneity along the selection margin?

#### Key Concepts
- MTE as the derivative of the IV estimand with respect to the instrument
- Unobserved resistance: $U_D$ and the selection equation
- Mapping from MTE to ATE, ATT, ATU, LATE
- Local IV identification of MTE
- Estimation via local polynomial regression
- Policy-relevant treatment effect (PRTE)
- Generalized Roy model as structural foundation

#### Key Mathematical Results
- **MTE definition**: $MTE(x,u_D) = E[Y(1)-Y(0)|X=x, U_D=u_D]$
- **LATE as MTE integral**: $LATE(p,p') = \int_{p}^{p'} MTE(u_D) \, du_D / (p'-p)$
- **ATE as MTE integral**: $ATE = \int_0^1 MTE(u_D) \, du_D$
- **IV as weighted MTE**: $\hat{\tau}_{IV} = \int MTE(u_D) \omega^{IV}(u_D) \, du_D$ with instrument-specific weights

#### Running Example Application
Estimate MTE for Medicaid enrollment effect on medical utilization using the OHE lottery propensity as the instrument propensity; plot MTE curve against $u_D$; compute ATE and ATT from MTE.

#### Python Implementation Notes
- Local polynomial MTE via `scipy.stats.gaussian_kde` and `numpy`
- Propensity score as instrument propensity ($p(Z)$ = $P(D=1|Z)$)
- `statsmodels` local linear regression
- Output: MTE plot over $[0,1]$, ATE/ATT/LATE from MTE integration

#### Connections
Builds on Chapter 21 (IV); connects to Chapter 30 (dynamic treatment regimes, policy-relevant treatment effects) and Chapter 36 (transportability via MTE reweighting).

---

### Chapter 23: Regression Discontinuity and Geographic Boundaries

#### Core Question
How does RD identify causal effects at a threshold, what are the assumptions, and when do geographic boundaries provide valid quasi-experiments?

#### Key Concepts
- Sharp vs. fuzzy RD
- Local randomization interpretation
- Continuity-based identification
- Bandwidth selection: CCT (Calonico-Cattaneo-Titiunik) optimal bandwidth
- Local linear regression and its boundary properties
- Manipulation testing (McCrary density test)
- Geographic (border) RD and spatial discontinuities
- Donut RD for heaping and manipulation

#### Key Mathematical Results
- **Sharp RD estimand**: $\tau_{RD} = \lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]$; local average at threshold
- **CCT bandwidth**: $h^* = C_K \sigma^2(c) / (f(c) B^2)$ balancing squared bias and variance
- **Fuzzy RD as IV**: $\hat{\tau}_{FRD} = \frac{\text{jump in } E[Y|X]}{\text{jump in } E[D|X]}$ at threshold; LATE for compliers at cutoff
- **McCrary test**: density test statistic for $H_0: f(c^+) = f(c^-)$; manipulation implies density jump

#### Running Example Application
Illustrate geographic RD using Medicaid eligibility income thresholds in ACA: counties near FPL 138% cutoff; demonstrate bandwidth sensitivity; apply McCrary density test to income running variable.

#### Python Implementation Notes
- `rdrobust` Python package
- `rddensity` for McCrary test
- Bandwidth sensitivity plots via `matplotlib`
- Output: RD plot, CCT estimate, density test p-value, bandwidth robustness table

#### Connections
Builds on Chapter 5 (design) and Chapter 21 (IV for fuzzy RD); connects to Chapter 34 (falsification) via donut and placebo threshold tests.

---

### Chapter 24: Encouragement Designs and Imperfect Compliance

#### Core Question
How do randomized encouragements to take treatment enable causal estimation when full compliance is unachievable?

#### Key Concepts
- Encouragement design as embedded IV
- Intention-to-treat (ITT) vs. LATE
- One-sided and two-sided noncompliance
- Complier characteristics: who responds to encouragement?
- Per-protocol analysis pitfalls
- Factorial encouragement designs
- Heterogeneous compliance and instrument strength

#### Key Mathematical Results
- **ITT**: $ITT = E[Y_i | Z_i=1] - E[Y_i | Z_i=0]$; identified directly, no assumptions beyond randomization
- **LATE via Wald**: $LATE = ITT / E[D_i | Z_i=1] - E[D_i | Z_i=0]$; ratio of ITT to first stage
- **Complier mean outcome**: $E[Y(0)|complier] = \frac{E[Y \cdot \mathbf{1}(D=0)|Z=1] - E[Y \cdot \mathbf{1}(D=0)|Z=0]}{-P(complier)}$
- **Never-taker/always-taker exclusion**: formally requires $Y(0,Z) = Y(0,0)$ and $Y(1,Z) = Y(1,1)$

#### Running Example Application
OHE lottery as encouragement: lottery win encourages Medicaid enrollment but doesn't guarantee it; compute ITT and LATE; characterize complier subpopulation via covariate analysis.

#### Python Implementation Notes
- `linearmodels.IV2SLS` for ITT and LATE
- Complier covariate profiling: $E[X|complier]$ computed via conditional moments
- Two-stage compliance breakdown table
- Output: complier profile table, ITT/LATE side-by-side, compliance rate by subgroup

#### Connections
Builds on Chapter 21 (IV) and Chapter 5 (target trial); connects to Chapter 42 (experiments with noncompliance and interference).

---

## Part VI — Time-Varying Treatments and G-Methods

---

### Chapter 25: Why Standard Regression Fails with Time-Varying Confounding

#### Core Question
What is time-varying confounding, why does standard regression fail (including over-adjustment), and what does correct identification require?

#### Key Concepts
- Time-varying treatment: $A_t$, time-varying confounders: $L_t$
- The causal diagram for sequential decisions
- Intermediate confounders: variables affected by treatment and affecting future outcome
- Why conditioning on $L_t$ blocks causal paths (over-adjustment)
- Why not conditioning on $L_t$ leaves open backdoor paths
- Structural nested models as the resolution
- G-methods taxonomy: g-formula, MSM, SNM

#### Key Mathematical Results
- **Sequential ignorability**: $Y(\bar{a}) \perp\!\!\!\perp A_t | \bar{A}_{t-1}, \bar{L}_t$ for all $t$
- **Time-varying positivity**: $P(A_t=a_t|\bar{A}_{t-1}=\bar{a}_{t-1}, \bar{L}_t=\bar{l}_t) > 0$
- **Over-adjustment bias**: conditioning on $L_t$ blocks $A_{t-1} \to L_t \to Y$ path; sign and magnitude of bias derived via simulation
- **Collider bias via DAG**: structural result that $L_t$ is a collider on path $A_{t-1} \to L_t \leftarrow U$

#### Running Example Application
Use BRFSS repeated cross-sections: insurance status $A_t$, health behavior $L_t$ (smoking, diet), outcome health status $Y_T$; show that standard panel regression is biased; motivate g-methods.

#### Python Implementation Notes
- Simulate time-varying confounding DGP in `numpy`
- Show OLS, lagged OLS, and FE estimates vs. true effect
- DAG visualization with `pgmpy`
- Output: bias simulation table, DAG figure for BRFSS structure

#### Connections
Builds on Chapters 4 (DAGs) and 6 (treatment timing); introduces Chapters 26–29 (g-methods) and Chapter 29 (dynamic treatment regimes).

---

### Chapter 26: The Parametric G-Formula

#### Core Question
How does the g-computation formula identify and estimate effects of sustained treatment strategies via sequential standardization?

#### Key Concepts
- Intervention distribution: $do(A_t = a_t)$ for $t = 0, \ldots, T$
- G-formula as iterated expectation via sequential standardization
- Static interventions and dynamic interventions
- Parametric models for time-varying components
- Monte Carlo g-computation
- Bias from model misspecification in sequential models
- Natural course vs. intervened distribution

#### Key Mathematical Results
- **G-formula**: $E[Y(\bar{a})] = \sum_{\bar{l}} E[Y|\bar{A}=\bar{a}, \bar{L}=\bar{l}] \prod_{t} P(L_t=l_t|\bar{A}_{t-1}=\bar{a}_{t-1}, \bar{L}_{t-1}=\bar{l}_{t-1})$
- **Sequential ignorability sufficiency**: g-formula identifies $E[Y(\bar{a})]$ iff sequential ignorability and positivity hold
- **Natural course**: setting $\bar{a} = $ observed treatment; matches observed mean under no unmeasured confounding
- **Monte Carlo variance**: bootstrap SE for g-computation; influence function for analytical SE

#### Running Example Application
Apply g-formula to BRFSS data: estimate potential health outcome under sustained insurance coverage ($A_t=1$ for all $t$) vs. natural course; implement Monte Carlo g-computation.

#### Python Implementation Notes
- Sequential logistic/linear models in `sklearn` per time period
- Monte Carlo simulation loop for counterfactual outcomes
- `pandas` for longitudinal data management
- Output: natural course vs. intervened distribution comparison, bootstrapped CI

#### Connections
Builds on Chapter 25; complemented by Chapter 27 (MSM/IPTW as alternative estimator) and Chapter 28 (SNM for structural efficiency).

---

### Chapter 27: Marginal Structural Models and Inverse Probability Weighting

#### Core Question
How do marginal structural models and IPTW identify causal effects of time-varying treatments by creating a pseudo-population free of time-varying confounding?

#### Key Concepts
- Marginal structural model as a model for potential outcomes
- Stabilized vs. unstabilized weights
- Inverse probability of treatment weights (IPTW)
- Inverse probability of censoring weights (IPCW)
- Combined IPTW × IPCW
- Weight truncation and its bias-variance tradeoff
- Doubly robust MSM estimators

#### Key Mathematical Results
- **Stabilized IPTW**: $SW_i = \prod_{t=0}^T \frac{P(A_t=a_t|\bar{A}_{t-1})}{P(A_t=a_t|\bar{A}_{t-1}, \bar{L}_t)}$; stabilized by marginal treatment probability
- **MSM identification**: under sequential ignorability, $E^{SW}[Y | \bar{A}=\bar{a}] = E[Y(\bar{a})]$
- **Combined weights**: $W_i = IPTW_i \times IPCW_i$ for handling both treatment and censoring
- **Truncation bias-variance tradeoff**: $\hat{\tau}_{trunc} = \hat{\tau}_{full} + O(\lambda^{-1})$ bias vs. variance reduction $O(\lambda^{-2})$

#### Running Example Application
Apply IPTW-MSM to BRFSS: estimate effect of continuous insurance coverage on 3-year health outcomes; model treatment weights with logistic regression; compare truncated vs. full weights.

#### Python Implementation Notes
- Sequential logistic regression for IPTW in `sklearn`
- Weighted MSM outcome model via `statsmodels.WLS`
- Weight diagnostics: effective sample size, weight distribution
- Output: IPTW-MSM estimate, weight distribution plot, truncation sensitivity table

#### Connections
Builds on Chapter 26 (g-formula) and Chapter 10 (IPW for cross-sectional); doubly robust extension connects to Chapter 11 (AIPW); dynamic extensions in Chapter 29.

---

### Chapter 28: Structural Nested Models and G-Estimation

#### Core Question
How do structural nested models and g-estimation consistently estimate causal effects while remaining semiparametrically efficient under weaker assumptions than MSMs?

#### Key Concepts
- Structural nested mean models (SNMMs)
- Blip-to-zero transformation
- G-estimation procedure: solving estimating equations
- Optimal blip function
- Structural nested distribution models
- Rank preservation assumption
- Connection to efficient score equations

#### Key Mathematical Results
- **SNMM**: $E[Y_{k+1}(\bar{0}_k | \bar{A}) - Y_{k+1}(\bar{0}_{k+1} | \bar{A}) | \bar{A}_k, \bar{L}_k] = \gamma_k(\bar{a}_k, \bar{l}_k; \psi)$; blip function
- **G-estimation**: solve $\sum_i H(\psi) \{A_t - E[A_t|\bar{A}_{t-1},\bar{L}_t]\} = 0$ for $\psi$
- **Consistency**: g-estimator consistent if treatment model correct; complementary to g-formula (outcome model)
- **Semiparametric efficiency**: optimal estimating function achieves semiparametric efficiency bound under SNMM

#### Running Example Application
Apply g-estimation to BRFSS for time-varying insurance on health outcomes; estimate blip-to-zero parameters $\psi$; compare to MSM estimates from Chapter 27.

#### Python Implementation Notes
- Manual g-estimation via `scipy.optimize.fsolve`
- Treatment residual model with `sklearn.LogisticRegression`
- Analytical standard errors via sandwich estimator in `numpy`
- Output: SNMM parameter estimates vs. MSM estimates, SE comparison

#### Connections
Builds on Chapters 26–27; more efficient than g-formula under correct treatment model; connects to Chapter 29 (dynamic regimes where SNMM naturally extends).

---

### Chapter 29: Dynamic Treatment Regimes and Reinforcement Learning Connections

#### Core Question
How do we estimate optimal dynamic treatment strategies that adapt to patient or unit history, and how does this connect to reinforcement learning?

#### Key Concepts
- Dynamic treatment regimes (DTRs): $d = (d_1, d_2, \ldots, d_T)$
- Value function and Q-function for a regime
- Backward induction / dynamic programming
- G-estimation for DTRs
- SMART (sequential multiple assignment randomized trial) designs
- Connections to RL: MDPs, policy gradient, Q-learning
- Doubly robust DTR estimation (DRDR)

#### Key Mathematical Results
- **Value of regime $d$**: $V^d = E[Y^d(\bar{A}^d)]$ where $\bar{A}^d$ is the treatment history under $d$
- **Q-function**: $Q_t(h_t, a_t) = E[Y | H_t=h_t, A_t=a_t, \text{follow } d \text{ thereafter}]$
- **Bellman recursion**: $Q_{T-1}(h_{T-1},a_{T-1}) = E[Q_T(H_T, d_T(H_T)) | h_{T-1},a_{T-1}]$
- **DRDR doubly robust**: $\hat{V}^d_{DR} = \frac{1}{n}\sum_i \frac{\mathbf{1}(\bar{A}_i = \bar{d}(H_i))}{\prod_t e_t(A_{it}|H_{it})} Y_i + \text{augmentation term}$

#### Running Example Application
Frame BRFSS insurance coverage decisions as a 3-period DTR: estimate optimal regime (insure if previous health deteriorated, maintain otherwise); compare to static always-insure regime.

#### Python Implementation Notes
- Q-learning backward induction in `numpy`
- `sklearn` for $Q$-function regression
- `econml` `DynamicDML` if available; manual DRDR estimator
- Output: optimal DTR description, value function estimate, comparison to static strategies

#### Connections
Builds on Chapters 27–28; connects to Chapter 14 (policy learning for static decisions) and Chapter 46 (causal monitoring for sequential decision-making).

---

### Chapter 30: Modified Treatment Policies and Dynamic Strategy Interventions

#### Core Question
What are modified treatment policies, why do they avoid positivity violations that point interventions suffer, and how are they estimated?

#### Key Concepts
- Modified treatment policy (MTP): $d(a,l)$ shifts natural value
- Additive and multiplicative shifts
- Avoiding positivity violations via bounded shifts
- Longitudinal targeted maximum likelihood (ltmle) for MTPs
- Incremental propensity score interventions
- Comparison to static and dynamic interventions
- Stochastic interventions

#### Key Mathematical Results
- **MTP identification**: under sequential ignorability, $E[Y^d] = E^{Q^d}[Y]$ where $Q^d$ is the intervened outcome regression
- **Incremental propensity score**: $d_\delta(a,l) = \frac{\delta a \cdot e(l)}{\delta a \cdot e(l) + (1-a)(1-e(l))}$ for odds-ratio shift $\delta$
- **Positivity-free property**: for additive MTP $d(a,l) = a + \delta$, positivity holds whenever $\delta$ stays within natural range
- **ltmle targeting step**: clever covariate $H_t$ used to target each time-point model; maintains double robustness

#### Running Example Application
Apply additive MTP to BRFSS: estimate effect of marginally improving insurance access (odds-ratio shift $\delta=2$) on 3-year outcomes; avoid positivity issues from static "all insured" intervention.

#### Python Implementation Notes
- `lmtp` Python package (Diaz-Williams)
- Manual incremental PS intervention estimator
- Sensitivity analysis over $\delta$ values
- Output: dose-response curve for $\delta \in [1,5]$, comparison to static intervention

#### Connections
Builds on Chapters 26–29; most general g-method; connects to Chapter 36 (external validity via stochastic interventions) and Chapter 46 (deployment policy shifts).

---

## Part VII — Sensitivity Analysis and Credibility

---

### Chapter 31: Sensitivity Analysis for Unobserved Confounding

#### Core Question
How do we quantify how much unmeasured confounding would be required to explain away a causal estimate, and which sensitivity frameworks are most useful?

#### Key Concepts
- Rosenbaum bounds: sensitivity in matched observational studies
- E-value (VanderWeele-Ding): minimum confounding strength
- Cinelli-Hazlett omitted variable bias framework
- Sensitivity for regression, IPW, and doubly robust estimators
- Partial $R^2$ parameterization of confounding
- Amplification and confounding strength
- Benchmarking against observed covariates

#### Key Mathematical Results
- **Rosenbaum bound**: $\Gamma$ = maximum odds ratio for unmeasured confounder; find $\Gamma^*$ where $p$-value exceeds $\alpha$
- **E-value**: $E = RR + \sqrt{RR(RR-1)}$; minimum RR of confounder with both treatment and outcome
- **OVB formula (Cinelli-Hazlett)**: $\hat{\tau}_{long} = \hat{\tau}_{short} - \frac{Cov(U,D)}{Var(D)} \cdot \frac{Cov(U,Y|\cdot)}{Cov(U,U|\cdot)}$; partial $R^2$ parameterization
- **Robustness value (RV)**: $RV = $ minimum partial $R^2$ of unmeasured confounder needed to reduce estimate to zero

#### Running Example Application
Apply Cinelli-Hazlett framework to OHE observational comparison (ignoring lottery); compute E-value for health outcome estimates; benchmark against partial $R^2$ of measured confounders.

#### Python Implementation Notes
- `sensemakr` Python package (Cinelli-Hazlett)
- E-value computation from point estimates and confidence intervals
- Contour plots of bias as function of partial $R^2$ parameters
- Output: sensitivity contour plot, E-value table, RV benchmarks

#### Connections
Builds on Chapter 1 (identification assumptions); extends Chapter 7 (negative controls) and Chapter 20 (DiD robustness); connects to Chapter 33 (partial identification).

---

### Chapter 32: Beyond Oster's Delta

#### Core Question
What does Oster's coefficient stability test actually measure, when is it valid or misleading, and what are stronger alternatives?

#### Key Concepts
- Altonji-Elder-Taber (AET) selection on observables = selection on unobservables
- Oster's $\delta$: the equal-selection assumption and its derivation
- $R^2_{max}$ sensitivity and how to set it
- Failures: when $\delta > 1$ is misleading, non-linear selection
- Cinelli-Hazlett as a superior alternative
- Proportional selection and its geometric interpretation
- When coefficient stability is informative

#### Key Mathematical Results
- **Oster $\delta$**: $\delta = \frac{(\tilde{\beta}-\hat{\beta})R^2_{max}}{(\hat{\beta}-\dot{\beta})(R^2_{max}-\tilde{R}^2)}$; ratio of selection on unobservables to selection on observables
- **AET identification**: $\tau = 0$ iff $\delta = 1$ under equal selection assumption
- **Bias formula**: $Bias = -\frac{Cov(U,D)}{Var(\tilde{D})} \sigma_U$ shows $\delta$ measures this ratio
- **Breakdown point**: $\delta^*$ at which $\tau$ would cross zero; point estimate analog of E-value

#### Running Example Application
Apply Oster $\delta$ to the OHE observational (uninsured vs. insured) comparison; compute $\delta$ under varying $R^2_{max}$; compare conclusion to Cinelli-Hazlett contour.

#### Python Implementation Notes
- Manual Oster $\delta$ from two nested regressions in `statsmodels`
- `sensemakr` for Cinelli-Hazlett comparison
- Sensitivity to $R^2_{max}$ via grid search
- Output: $\delta$ vs. $R^2_{max}$ plot, side-by-side comparison table

#### Connections
Builds on Chapter 31; completes the sensitivity analysis toolkit; directly precedes Chapter 33 (partial identification as the most general framework).

---

### Chapter 33: Partial Identification and Bounds

#### Core Question
What causal conclusions are possible without point identification, and how do sharp bounds discipline inference under weaker assumptions?

#### Key Concepts
- Manski's no-assumption bounds
- Monotone treatment response (MTR)
- Monotone treatment selection (MTS)
- Optimal transport bounds
- IV bounds (Balke-Pearl)
- Intersection bounds
- Sensitivity analysis as a special case of partial identification

#### Key Mathematical Results
- **Manski no-assumption bounds**: $E[Y(1)] \in [E[Y \cdot \mathbf{1}(D=1)] + y_{min} P(D=0),\ E[Y \cdot \mathbf{1}(D=1)] + y_{max} P(D=0)]$
- **MTR+MTS tighter bounds**: intersection of two constraint sets; width depends on observable selection strength
- **Balke-Pearl IV bounds**: tight bounds on ATE using only IV independence and exclusion restriction, without monotonicity
- **Sharp bound characterization**: set $[\tau_L, \tau_U]$ is sharp iff there exists a joint distribution of $(Y(0),Y(1),D,Z)$ in the identified set achieving each endpoint

#### Running Example Application
Compute Manski bounds for the ATE of Medicaid on catastrophic expenditure in OHE without using the lottery; tighten with MTR (insurance weakly improves health); compare interval width to lottery IV estimate.

#### Python Implementation Notes
- Manual Manski bound computation in `numpy`
- `pyvmte` or manual LP for Balke-Pearl bounds via `scipy.optimize.linprog`
- Bound width vs. assumption strength visualization
- Output: bound intervals table, assumption-strength trade-off plot

#### Connections
Builds on Chapter 31–32 (sensitivity as partial identification); connects to Chapter 36 (partial identification under selection for transportability).

---

### Chapter 34: Placebo Tests as a System

#### Core Question
How should placebo tests be systematically designed, executed, and interpreted to provide structured evidence of design validity?

#### Key Concepts
- Taxonomy of placebo tests: outcome, treatment, period, geography
- Pre-trend placebos in DiD
- In-space placebos (synthetic control)
- Permutation inference and its validity
- False discovery rate under multiple placebos
- Placebo calibration: what rejection rate is expected under the null?
- Combining multiple placebos into a summary test

#### Key Mathematical Results
- **Permutation p-value**: $p = \frac{1}{M}\sum_{m=1}^M \mathbf{1}[|\hat{\tau}_m^{placebo}| \geq |\hat{\tau}^{actual}|]$; exact under exchangeability
- **Fisher randomization test validity**: exact in finite sample under sharp null of no effect for any unit
- **In-space SC permutation**: $p_{SC} = \frac{\text{rank of actual RMSPE ratio}}{J}$ where $J$ = number of donor units
- **Expected rejection under null**: $\alpha$ for well-calibrated test; over-rejection signals pre-trend or design flaw

#### Running Example Application
For ACA expansion DiD: implement 50 state-level placebo treatments; compute permutation distribution; verify test size calibration; for synthetic control, compute in-space placebos.

#### Python Implementation Notes
- Permutation loop over placebo treatment assignments in `pandas`
- In-space SC placebos with `synth` package
- Permutation distribution visualization
- Output: permutation distribution plot, rank-based p-value, false positive rate estimate

#### Connections
Builds on Chapter 7 (negative controls) and Chapter 19 (SC); reinforces Chapters 15–20 (DiD); generalizes to Chapter 35 (multiple testing).

---

### Chapter 35: Multiple Testing, Researcher Degrees of Freedom, and Specification Search

#### Core Question
How do multiple comparisons and researcher flexibility inflate false discovery rates, and what disciplined corrections restore valid inference?

#### Key Concepts
- Family-wise error rate (FWER) vs. false discovery rate (FDR)
- Bonferroni and Holm corrections
- Benjamini-Hochberg FDR control
- Romano-Wolf stepdown for correlated tests
- Pre-registration and analysis plans as FWER controls
- Specification curves (Simonsohn-Simmons-Nelson)
- Multiverse analysis
- Researcher degrees of freedom in causal analysis

#### Key Mathematical Results
- **Bonferroni**: reject $H_k$ iff $p_k \leq \alpha/m$; controls FWER $\leq \alpha$
- **Benjamini-Hochberg**: reject $H_{(k)}$ iff $p_{(k)} \leq \frac{k}{m}\alpha$; controls FDR $\leq \alpha$ under independence
- **Romano-Wolf**: FWER-controlling stepdown based on bootstrap of joint test distribution; exact under stationarity
- **Specification curve null**: permutation distribution of specification curve statistic under $H_0: \tau = 0$

#### Running Example Application
Report 8 outcome variables in OHE (inpatient, outpatient, ED, prescriptions, physical health, mental health, financial outcomes); apply Holm and BH corrections; build specification curve over control set choices.

#### Python Implementation Notes
- `statsmodels.stats.multitest` for BH, Holm
- Romano-Wolf via `wildboottest` or manual bootstrap
- Specification curve via `specurve` Python package
- Output: corrected p-value table, specification curve figure

#### Connections
Builds on Chapter 34 (placebos); completes the credibility toolkit; connects to Chapter 47 (causal audit workflow) and Chapter 48 (building credible observational studies).

---

### Chapter 36: External Validity and Transportability

#### Core Question
When and how can a causal effect estimated in one population be transported to another, and what assumptions does this require?

#### Key Concepts
- Internal vs. external validity
- Selection diagrams for transport
- S-admissibility and transport formula
- Inverse probability of sampling weights (IPSW)
- Generalizability of RCT results to target population
- Transportability vs. generalizability (Pearl-Bareinboim)
- MTE-based reweighting for PRTE

#### Key Mathematical Results
- **Transport formula**: $E_\pi[Y(d)] = \sum_x E[Y|D=d,X=x]P_\pi(X=x)$ under $X$-admissibility
- **IPSW**: $\hat{\tau}_\pi = \frac{1}{n}\sum_i \frac{P_\pi(X_i)}{P_{study}(X_i)} \hat{\tau}(X_i)$ weights study sample to target
- **Bareinboim-Pearl transport theorem**: graphical criterion for identifiability via selection diagrams
- **PRTE via MTE**: $PRTE = \int MTE(u_D) [P_\pi(u_D) - P_{study}(u_D)] du_D$

#### Running Example Application
Transport OHE LATE (complier population) to Oregon general population: estimate IPSW weights using census target distribution; compute PRTE; bound non-identified portion via Manski.

#### Python Implementation Notes
- IPSW estimation via `sklearn.LogisticRegression` with target/study indicator
- Target population simulation from census marginals
- Sensitivity analysis over transport assumptions
- Output: transported ATE vs. LATE, bound width comparison

#### Connections
Builds on Chapters 4 (selection diagrams), 22 (MTE), 33 (bounds); connects to Chapter 49 (method selection under constraints) and Chapter 36 closes out the credibility section.

---

## Part VIII — Causal Inference with Machine Learning and Modern Data

---

### Chapter 37: Prediction Is Not Causation, But Prediction Still Matters

#### Core Question
What is the precise relationship between predictive ML and causal inference, and where does each belong in an analysis pipeline?

#### Key Concepts
- Supervised learning targets $E[Y|X]$, not $E[Y(d)|X]$
- Correlation, prediction, causation: the tripartite distinction
- Where ML helps: nuisance estimation, CATE estimation, feature selection
- Where ML misleads: causal interpretation of feature importance, propensity from observational data
- Causal regularization
- Label leakage and post-treatment variables
- Shapley values and causal attribution

#### Key Mathematical Results
- **Prediction risk**: $R(\hat{f}) = E[(Y - \hat{f}(X))^2]$; minimized by $E[Y|X]$, not causal effect
- **Causal effect**: $\tau(x) = E[Y(1)|X=x] - E[Y(0)|X=x] \neq E[Y|D=1,X=x] - E[Y|D=0,X=x]$ under confounding
- **Selection-induced confounding**: $E[Y|D=d,X] = E[Y(d)|X] + \frac{Cov(U,Y|D=d,X)}{1}$ confounded by $U$
- **Post-treatment variable bias**: conditioning on post-treatment variable $M$ changes estimand to direct effect; typically unwanted

#### Running Example Application
Show that BRFSS ML model predicting health outcomes from insurance status achieves good predictive accuracy but badly estimates causal effect; contrast with DML estimates from Chapter 12.

#### Python Implementation Notes
- `sklearn` gradient boosted tree for prediction
- SHAP values via `shap` library; discuss causal misinterpretation
- Side-by-side: SHAP feature importance vs. DML causal estimate
- Output: prediction accuracy table, SHAP bar plot, causal vs. predictive estimate comparison

#### Connections
Motivates ML's proper role; frames Chapters 38–42 where ML is used correctly within causal frameworks.

---

### Chapter 38: Causal Forests and Honest Trees

#### Core Question
How do causal forests estimate CATEs with valid confidence intervals via honesty, subsampling, and local centering?

#### Key Concepts
- Honesty: splitting sample used for splits vs. estimates
- Adaptive nearest-neighbor interpretation of forests
- Generalized random forests (GRF) framework
- Local centering (residualization) as preprocessing
- Subsampling for variance estimation
- Asymptotic normality of forest estimates
- Variable importance in causal forests

#### Key Mathematical Results
- **GRF estimating equation**: $\hat{\tau}(x) = \arg\min_\tau \sum_i \alpha_i(x)(Y_i - \mu(X_i) - \tau \cdot (D_i - e(X_i)))^2$ with kernel weights $\alpha_i(x)$
- **Honest asymptotic normality**: $\sqrt{s} (\hat{\tau}(x) - \tau(x)) \xrightarrow{d} N(0, \sigma^2(x))$ where $s$ is subsample size
- **Local centering**: residualize $Y$ and $D$ before splitting; equivalent to R-learner with forest second stage
- **Variance via infinitesimal jackknife**: $\hat{V}(\hat{\tau}(x)) = \frac{n-1}{n} \sum_i Cov^2(\hat{\tau}(x), N_{bi})$ over trees

#### Running Example Application
Estimate CATE surface for OHE: effect of Medicaid on financial hardship; local centering with RF propensity; plot heterogeneity across income and age; compare to linear CATE from Chapter 13.

#### Python Implementation Notes
- `econml.grf.CausalForest` (GRF-based)
- `grf` R package via `rpy2` if needed for edge cases
- Confidence interval coverage simulation
- Output: CATE heatmap (income × age), variable importance plot, CI coverage table

#### Connections
Builds on Chapter 13 (meta-learners) and Chapter 12 (DML); the honest tree variant resolves the overfitting concern raised in both; connects to Chapter 14 (policy trees).

---

### Chapter 39: Representation Learning and Causal Inference

#### Core Question
How can deep learning representation methods solve covariate overlap problems and enable causal estimation from high-dimensional observational data?

#### Key Concepts
- Balanced representations: CFRNet (Johansson-Shalit-Sontag)
- Integral probability metrics (IPM) for balance
- DragonNet: joint treatment and outcome representation
- Propensity-induced regularization
- Counterfactual prediction in representation space
- Synthetic interventions and matrix factorization
- Limitations: distribution shift in representation space

#### Key Mathematical Results
- **CFRNet bound**: $\epsilon_{PEHE} \leq \hat{\epsilon}_{F} + B_{IPM}(\hat{\Phi}_1, \hat{\Phi}_0) + C$; PEHE bounded by factual error + representation imbalance
- **Wasserstein IPM**: $W_1(\hat{\Phi}_1, \hat{\Phi}_0) = \sup_{f: \|f\|_L \leq 1} E[f(\hat{\Phi}(X))|D=1] - E[f(\hat{\Phi}(X))|D=0]$
- **DragonNet shared head**: $\hat{\tau}(x) = g_1(\Phi(x)) - g_0(\Phi(x))$ with shared $\Phi$ and targeted regularization
- **Propensity dropout**: regularization term $\lambda E[(\hat{e}(X) - e(X))^2]$ encourages balanced representation

#### Running Example Application
Apply CFRNet to BRFSS high-dimensional covariates (50+ variables) for insurance-on-health estimation; compare representation balance to logistic propensity score; evaluate on OHE held-out outcomes.

#### Python Implementation Notes
- `PyTorch` implementation of CFRNet
- IPM computation via `geomloss` (Sinkhorn)
- Hyperparameter tuning via `optuna`
- Output: representation t-SNE plot by treatment, PEHE vs. IPM tradeoff curve

#### Connections
Builds on Chapters 10–11 (propensity, AIPW); uses DL instead of kernel/tree methods; connects to Chapter 40 (unstructured data).

---

### Chapter 40: Causal Inference with Text, Images, and Unstructured Data

#### Core Question
How can unstructured data serve as treatments, outcomes, confounders, or instruments in causal analyses?

#### Key Concepts
- Text as confounder: unsupervised de-confounding
- Treatment measurement from text (sentiment, topic models)
- Outcome from text: NLP-derived health outcomes
- Image-based treatment/outcome proxies
- Causal text generation: contrastive counterfactuals
- BERT embeddings as high-dimensional covariates
- Double ML with text nuisances

#### Key Mathematical Results
- **Text-as-confounder**: $P(Y|do(D),X_{text}) = P(Y|D, X_{text})$ under text-blocking of all confounders; identification condition
- **Topic model residualization**: $\tilde{D} = D - E[D|\text{topics}]$; Robinson transform with LDA nuisance
- **Causal sentence embedding**: counterfactual $Y(1-d)$ approximated by nearest neighbor in embedding space with flipped treatment token
- **High-dimensional nuisance rate**: under sparsity of text embeddings, LASSO achieves $o(n^{-1/4})$ rate in DML

#### Running Example Application
Use BRFSS open-text general health descriptions (if available) or physician notes proxy; alternatively, illustrate method with Congressional Record text as instrument for health policy.

#### Python Implementation Notes
- `transformers` (HuggingFace) for BERT embeddings
- `gensim` for LDA topic model nuisance
- DML with LASSO text nuisance via `sklearn.linear_model.Lasso`
- Output: topic-residualized DML estimate, sensitivity to embedding dimensionality

#### Connections
Builds on Chapter 12 (DML with ML nuisances); natural application area for Chapter 37 (prediction vs. causation); connects to Chapter 43 (text in business settings).

---

### Chapter 41: Interference, Networks, and Spillovers

#### Core Question
When treatments spill over to untreated units, how is standard causal identification violated, and what frameworks handle interference?

#### Key Concepts
- SUTVA violation under interference
- Partial interference: interference within clusters only
- Exposure mappings: mapping network neighborhood to effective treatment
- Direct and spillover effects: total, direct, indirect
- Bernoulli and clustered network experimental designs
- Network-clustered IV
- Stochastic interference and average treatment policies

#### Key Mathematical Results
- **Exposure mapping**: $Y_i(\mathbf{a}) = Y_i(f_i(\mathbf{a}))$ under Aronow-Samii; reduces from $2^n$ to tractable potential outcomes
- **Direct effect**: $DE(a,a') = E[Y_i(a, a) - Y_i(a', a)]$; own-treatment effect holding neighbors fixed
- **Spillover effect**: $SE(a) = E[Y_i(a,a') - Y_i(a,a)]$; neighbor-treatment effect
- **Horvitz-Thompson under interference**: $\hat{\tau}_{HT} = \frac{1}{n}\sum_i \frac{Y_i \mathbf{1}(f_i(A)=e)}{P(f_i(A)=e)}$; weighted by exposure mapping probability

#### Running Example Application
Medicaid expansion network spillovers: estimate spillover effect of neighbor state expansion on own-state health outcomes; define exposure mapping by state adjacency and expansion timing.

#### Python Implementation Notes
- `networkx` for network construction
- Exposure mapping computation with adjacency matrix
- Clustered Horvitz-Thompson estimator in `numpy`
- Output: direct vs. spillover effect estimates, network exposure distribution

#### Connections
Builds on Chapter 1 (SUTVA), Chapter 5 (target trial), Chapter 24 (noncompliance); interference connects to Chapter 42 (experiments with interference).

---

### Chapter 42: Causal Inference in Experiments with Noncompliance and Interference

#### Core Question
How do we handle the joint complications of treatment noncompliance and network spillovers in experimental settings?

#### Key Concepts
- Encouragement design + interference: compound identification problem
- Per-protocol analysis under interference
- Clustered encouragement designs
- Two-stage randomization for spillover identification
- Composite exposure mappings
- LATE under clustered assignment
- Pseudo-outcomes under interference

#### Key Mathematical Results
- **LATE under interference**: $LATE_{cluster} = \frac{E[Y|Z_{cluster}=1] - E[Y|Z_{cluster}=0]}{E[D|Z_{cluster}=1] - E[D|Z_{cluster}=0]}$; valid under cluster-level monotonicity
- **Two-stage randomization**: identify both direct (LATE) and spillover effects via random saturation design (Baird et al.)
- **Random saturation ITT**: $ITT_{direct} = E[Y|Z_i=1,\pi_c] - E[Y|Z_i=0,\pi_c]$ at fixed cluster saturation $\pi_c$
- **Random saturation spillover**: $\frac{d}{d\pi_c} E[Y|Z_i=0, \pi_c]$; marginal effect of saturation on untreated

#### Running Example Application
Hypothetical extension of OHE: simulate cluster-level lottery where counties assigned different saturation rates; estimate direct LATE and spillover for community health outcomes.

#### Python Implementation Notes
- Simulate two-stage randomization DGP in `numpy`
- Cluster-level Wald estimator
- Nonparametric spillover function estimation via `sklearn` kernel regression
- Output: spillover function plot, LATE vs. saturation rate table

#### Connections
Builds on Chapters 24 (encouragement), 41 (interference); closes out the complications of experiments; precedes business applications where both issues arise.

---

## Part IX — Business, Policy, and Decision Systems

---

### Chapter 43: Causal Inference for Pricing, Marketing, and Product Interventions

#### Core Question
How do the causal methods developed throughout the book apply to pricing experiments, marketing attribution, and product feature rollouts?

#### Key Concepts
- Price elasticity estimation via demand experiments
- Incrementality testing in digital marketing
- A/B testing with network effects (marketplace interference)
- Switchback experiments for time-series interference
- Holdout-based attribution
- Long-run vs. short-run treatment effects
- Novelty effects and habituation

#### Key Mathematical Results
- **Price elasticity IV**: $\hat{\varepsilon} = \frac{\partial \log Q}{\partial \log P}$; estimated via cost-shifter IV
- **Switchback estimator**: $\hat{\tau} = \frac{1}{T}\sum_{t} \hat{\tau}_t$ where $\hat{\tau}_t$ is within-period DID; bias from carryover derived
- **Multi-touch attribution impossibility**: without randomization, $E[\hat{\alpha}_{last-touch}] \neq \tau$; Shapley attribution as cooperative game value
- **Long-run via proxy**: $\hat{\tau}_{LR} = \hat{\tau}_{ST} \times \frac{\partial LR}{\partial ST}$; proxy effect extrapolation formula

#### Running Example Application
Parallel application: insurance price-sensitivity to premium changes using Medicaid expansion copayment variation; apply IV for price elasticity; contrast with observational OLS.

#### Python Implementation Notes
- `econml` for CATE estimation on marketing segments
- Switchback simulation with carryover in `numpy`
- Shapley-value attribution via `shap`
- Output: elasticity estimate with IV vs. OLS, switchback bias simulation

#### Connections
Applies Chapters 21 (IV pricing), 12 (DML), 13 (CATE), 41 (interference) to business context; connects to Chapter 45 (ROI) and Chapter 46 (monitoring).

---

### Chapter 44: Causal Inference for Forecasting and Planning

#### Core Question
How do causal estimates, structural models, and policy simulations combine to produce forecasts that are robust to interventions?

#### Key Concepts
- Causal vs. predictive forecasting
- Intervention-robust prediction (do-calculus for forecasting)
- Structural causal models for scenario analysis
- Synthetic control for causal forecasting
- DiD-based counterfactual simulation
- Recalibration under distribution shift
- Uncertainty propagation in causal forecasts

#### Key Mathematical Results
- **Causal forecast**: $\hat{Y}_{T+h}^{do(D=d)} = \hat{E}[Y_{T+h}|do(D=d), X_{T+h}]$; requires identification, not just prediction
- **Intervention robustness condition**: $P(Y|do(X), S=1) = P(Y|do(X), S=0)$ under transportability across regime shifts
- **SC forecast**: $\hat{Y}_{1,T+h}^{SC} = \sum_j \hat{w}_j Y_{j,T+h}$; extrapolates synthetic control into forecast period
- **Structural propagation**: $Var(\hat{Y}_{T+h}^{causal}) = Var(\hat{\tau}) \cdot d^2 + Var(\hat{Y}_{T+h}^{base})$; uncertainty from causal effect + baseline

#### Running Example Application
Forecast the effect of universal Medicaid expansion to all states: use ACA expansion DiD estimates as structural parameters; propagate uncertainty; compare to projection without causal adjustment.

#### Python Implementation Notes
- `statsforecast` for baseline forecasting
- Causal adjustment layer via DiD estimates from Chapter 17
- Monte Carlo uncertainty propagation in `numpy`
- Output: forecast fan chart with and without causal adjustment, uncertainty decomposition

#### Connections
Applies Part IV (DiD) and Part V (IV) to forecasting problems; connects to Chapter 46 (monitoring) and Chapter 45 (ROI) as the decision layer.

---

### Chapter 45: From ATE to ROI — Decision-Theoretic Causal Inference

#### Core Question
How does a causal estimate translate into a business or policy decision under costs, constraints, and uncertainty?

#### Key Concepts
- Decision-theoretic framework: utility, costs, budget constraints
- ATE → expected benefit calculation
- Threshold analysis: MVPD (minimum viable policy delta)
- Uncertainty in ROI: propagating CI to decision
- Expected value of information (EVOI)
- Cost-effectiveness analysis for policy
- Risk-averse decision making under causal uncertainty

#### Key Mathematical Results
- **Policy value**: $ROI(\pi) = \mathbb{E}[B(\tau(X))\pi(X)] - C \cdot E[\pi(X)]$; benefit minus cost
- **Threshold ATE**: $\tau^* = C/B$; break-even treatment effect
- **EVOI**: $EVOI = E_\tau[\max_d U(d,\tau)] - \max_d E_\tau[U(d,\tau)]$; value of resolving uncertainty
- **Posterior expected utility**: $E_{\tau|\text{data}}[U(\text{expand})] = \int U(\text{expand}, \tau) p(\tau|\text{data}) d\tau$; Bayesian decision

#### Running Example Application
Translate OHE Medicaid LATE into cost-effectiveness ratio for policymakers; compute threshold effect size for budget neutrality; propagate CI to probability that expansion is cost-effective.

#### Python Implementation Notes
- Cost-effectiveness simulation via `numpy` Monte Carlo
- `scipy.stats` for posterior over causal effect (normal approximation)
- EVOI computation via integration
- Output: cost-effectiveness plane, EVOI as function of sample size, break-even threshold diagram

#### Connections
Integrates all estimation chapters into decision layer; builds on Chapter 14 (policy learning); connects to Chapter 46 (post-deployment monitoring).

---

### Chapter 46: Causal Monitoring and Post-Deployment Learning

#### Core Question
How do we maintain causal validity of deployed interventions over time, detect distributional shift, and update causal models from streaming data?

#### Key Concepts
- Distribution shift detection: covariate and concept drift
- Sequential causal testing
- Observational causal monitoring (causal surveillance)
- Online doubly robust estimation
- Bandit algorithms and causal exploration-exploitation
- Causal discovery from time series
- Model decay and recalibration triggers

#### Key Mathematical Results
- **CUSUM for causal shift**: $S_t = \max(0, S_{t-1} + (\hat{\tau}_t - \tau_0) - k)$; control limit $h$ for detection
- **Online DR estimate**: $\hat{\tau}_t = \hat{\tau}_{t-1} + \eta_t [\Gamma_t - \hat{\tau}_{t-1}]$ where $\Gamma_t$ is the $t$-th DR pseudo-outcome
- **Thompson sampling with causal structure**: prior $\tau \sim N(\mu_0, \sigma^2_0)$; posterior updated via DR observations
- **Granger causality**: $Y_t \not\perp X_{t-1} | Y_{t-1}, \ldots$; predictive causality (not structural); limitations discussed

#### Running Example Application
Simulate ongoing monitoring of Medicaid expansion effects: construct CUSUM chart for monthly health outcome trends; detect when post-expansion effects stabilize; trigger recalibration.

#### Python Implementation Notes
- CUSUM implementation in `numpy`
- Online DR update loop
- `river` (online ML library) for streaming estimates
- Output: CUSUM chart, online DR convergence plot, recalibration trigger dates

#### Connections
Builds on Chapters 14 (policy), 29 (dynamic regimes); applies sequential methods from Chapter 35 (multiple testing); closes Part IX before capstone.

---

## Part X — Capstone Designs

---

### Chapter 47: The Causal Audit — Full Diagnostic Workflow

#### Core Question
Given a completed causal analysis, how do we systematically audit its identification, estimation, and inference for credibility?

#### Key Concepts
- Audit checklist structure (design, identification, estimation, inference, sensitivity, external validity)
- Red flags vs. yellow flags
- Replication and robustness documentation
- Preregistration audit
- Code and data audit
- Third-party review protocol
- Communicating uncertainty honestly

#### Key Mathematical Results
- **Sensitivity cascade**: ordered set of $(\Gamma^*, E\text{-value}, RV)$ for each estimate; thresholds for "credible"
- **Specification curve summary statistic**: median estimate and fraction of specifications with same sign as primary
- **Balance audit**: standardized mean difference $SMD < 0.1$ as rule-of-thumb for covariate balance
- **Power audit**: minimum detectable effect at achieved sample size and $\alpha$

#### Running Example Application
Conduct full causal audit of the OHE analysis chain built across all chapters: verify identification, check balance, apply sensitivity analysis battery, document specification curve over control choices.

#### Python Implementation Notes
- Structured audit checklist as `pandas` DataFrame
- Automated balance report via `tableone`
- Sensitivity battery via `sensemakr` + Rambachan-Roth
- Output: audit report notebook, red/yellow/green flag summary table

#### Connections
Synthesizes Part VII (sensitivity) with all preceding chapters; pairs with Chapter 48 (building credible studies) and Chapter 49 (method selection).

---

### Chapter 48: Building a High-Credibility Observational Study

#### Core Question
What design and analysis choices, made prospectively, maximize the credibility of an observational causal study?

#### Key Concepts
- New-user, active-comparator design
- Pre-specification of analysis plan
- Blinded outcome analysis
- Negative control outcome battery (pre-committed)
- Pre-analysis diagnostics vs. specification search
- Replication cohort design
- Reporting standards: STROBE, CONSORT-analog for observational

#### Key Mathematical Results
- **Pre-specified analysis power**: $n \geq (z_{\alpha/2} + z_\beta)^2 (\sigma_1^2 + \sigma_0^2) / \tau^2_{MDE}$; sample size for MDE
- **Expected false discovery**: $E[FDR] = m_0 \alpha / (m_0 \alpha + m_1 (1-\beta))$ as function of true nulls $m_0$
- **Replication probability**: $P(\text{replicate}) = P(|Z_{rep}| > z_{\alpha/2})$ under original estimate distribution
- **Evidence factor formula**: $\prod_k p_k^{1/K}$ for $K$ independent design replications; combined evidence

#### Running Example Application
Retrospectively score the ACA expansion study against high-credibility checklist; identify items that could have been pre-specified; design a hypothetical pre-registration document.

#### Python Implementation Notes
- Sample size calculator for DiD using `statsmodels.stats.power`
- Pre-registration template as Jupyter notebook structure
- Evidence factor computation for multi-design combination
- Output: credibility scorecard, pre-registration template notebook

#### Connections
Synthesizes Chapter 3 (design checklist) and Chapter 7 (negative controls) with full methodological toolkit; companion to Chapter 47 (audit).

---

### Chapter 49: Choosing the Right Method Under Constraints

#### Core Question
Given a research setting with specific data, design, and uncertainty constraints, which causal method should be used and why?

#### Key Concepts
- Decision tree for method selection
- Data requirements matrix: panel, cross-section, IV, experiment
- Assumption severity ranking
- Empirical content vs. credibility tradeoff
- Combining multiple identification strategies
- Sensitivity-adjusted method comparison
- The weakest link principle in identification chains

#### Key Mathematical Results
- **Efficiency ordering**: under correct specification, semiparametric bound $V^*$ is common floor; methods differ in assumptions needed to achieve it
- **Assumption violation bias**: tabulated bias formulas for each method under each threat (unconfoundedness violation, PT violation, exclusion violation)
- **Combined estimator**: $\hat{\tau}_{combined} = \lambda \hat{\tau}_{IV} + (1-\lambda) \hat{\tau}_{DiD}$; optimal $\lambda$ from minimum MSE combination
- **Honesty-credibility frontier**: formal tradeoff between sample efficiency and robustness to specification error

#### Running Example Application
Apply decision framework to the insurance question under three scenarios: only cross-sectional data, panel with staggered adoption, lottery instrument; show which method is selected and why.

#### Python Implementation Notes
- Decision tree flowchart via `graphviz`
- Assumption violation simulation: bias under each method as function of violation severity
- Multi-study combination estimator
- Output: method selection flowchart, bias-robustness tradeoff table

#### Connections
Synthesizes the entire book; connects most directly to Chapter 3 (design checklist) and Part VII (sensitivity).

---

### Chapter 50: The Frontier — Open Problems and Emerging Methods

#### Core Question
What are the most important unsolved problems in causal inference, and what methodological directions show the most promise?

#### Key Concepts
- Causal discovery from observational data (PC, FCI, NOTEARS)
- Double robustness under model misspecification (higher-order influence functions)
- Causal representation learning and identifiability
- Foundation models and causal structure
- Federated causal inference (distributed data)
- Causal inference for large language models
- Semi-parametric theory gaps: efficiency bounds for complex functionals

#### Key Mathematical Results
- **NOTEARS constraint**: $h(W) = \text{tr}(e^{W \circ W}) - d = 0$ encodes DAG acyclicity as smooth constraint
- **Higher-order IF**: $\tau = \sum_{k=0}^K (-1)^k \Delta^k \tau_{plugin} + o(n^{-1/2})$; bias correction to higher order
- **ICA identifiability**: $P(X) = \prod_i P_i(s_i)$ implies unique $A$ in $X = As$ up to permutation and scale (Comon 1994)
- **Federated efficiency**: $\hat{\tau}_{fed} = \sum_k w_k \hat{\tau}_k$ with $w_k \propto n_k / V_k$; federated semiparametric efficiency

#### Running Example Application
Frame each frontier topic relative to the OHE/Medicaid setting: what would causal discovery reveal about the insurance DAG? What would federated estimation add across state data silos?

#### Python Implementation Notes
- `causal-learn` for PC/FCI algorithms
- `notears-torch` for NOTEARS
- `PySyft` for federated learning prototype
- Output: discovered DAG vs. assumed DAG, federated vs. pooled comparison

#### Connections
Builds on all preceding chapters; pointers to research literature for each open problem.

---

## Appendix A: Mathematical Toolkit

### A.1 Potential Outcomes and Structural Equations
- Formal axioms, SUTVA, consistency, composition
- Equivalence of Rubin and Pearl frameworks
- Key identification theorems (backdoor, frontdoor, IV)

### A.2 Influence Functions and Semiparametric Theory
- Von Mises expansion and functional derivatives
- Efficient influence function derivation procedure
- Semiparametric efficiency bound: information bound calculation
- Pathwise differentiability

### A.3 Concentration Inequalities and Empirical Process Theory
- Uniform laws of large numbers
- Donsker classes
- Rate conditions for nonparametric nuisance estimation
- Covering numbers and metric entropy

### A.4 Linear Algebra and Matrix Methods
- Spectral methods for synthetic control and matrix completion
- Nuclear norm and low-rank approximation
- Singular value decomposition in causal models

---

## Appendix B: Software Patterns

### B.1 The Causal Inference Stack
- Core: `numpy`, `pandas`, `scipy`, `statsmodels`
- ML nuisances: `sklearn`, `lightgbm`, `xgboost`
- Causal: `econml`, `dowhy`, `causalml`, `pyfixest`
- Specialized: `rdrobust`, `csdid`, `synth`, `sensemakr`

### B.2 Cross-Fitting Template
- Generic $K$-fold cross-fitting class in Python
- Integration with `sklearn.pipeline`

### B.3 Doubly Robust Estimation Template
- AIPW base class with pluggable nuisance models
- Bootstrap and analytical SE implementations

### B.4 Visualization Standards
- Event study plot template
- Balance plot (LOVE plot) template
- Sensitivity contour plot template
- CATE distribution visualization

---

## Appendix C: Simulation Laboratory

### C.1 DGP Library
- Unconfounded observational DGP with controllable confounding strength
- IV DGP with weak instrument parameter
- Staggered adoption DGP with heterogeneous effects
- Time-varying confounding DGP

### C.2 Benchmarking Suite
- Monte Carlo comparison template: $M = 1000$ replications
- Metrics: bias, RMSE, coverage, interval length
- Reporting: `pandas` summary table + `matplotlib` violin plots

### C.3 Sensitivity Simulation
- Rosenbaum bound power curves
- E-value empirical validation
- Rambachan-Roth coverage simulation

---

## Appendix D: Reading Map

### D.1 Dependency Graph
- Chapter prerequisites mapped as DAG
- Fast tracks: IV-only, DiD-only, ML-only paths

### D.2 Foundational Papers by Chapter
- One primary reference per chapter (seminal paper)
- One recent extension per chapter (post-2018)

### D.3 Cross-Reference Index
- Method → chapters where applied
- Dataset → chapters where used
- Estimand → chapters where identified

---

## Dataset and Code Structure

### Dataset Download Instructions

```python
# ── Oregon Health Insurance Experiment (OHE) ────────────────────────────────
import urllib.request, zipfile, pathlib

OHE_URL = "https://nber.org/oregon/4.data.html"
# Direct data link (NBER):
urllib.request.urlretrieve(
    "https://data.nber.org/oregon/oregonhie_descriptive_vars.dta",
    "data/raw/ohe/oregonhie_descriptive_vars.dta"
)
# Also retrieve:
# oregonhie_survey12m_vars.dta  — 12-month survey outcomes
# oregonhie_ed_vars.dta         — Emergency department utilization
# Load with:
import pandas as pd
ohe = pd.read_stata("data/raw/ohe/oregonhie_descriptive_vars.dta")

# ── BRFSS (Behavioral Risk Factor Surveillance System) ───────────────────────
# Annual files via CDC FWDB format; 2010–2016 for ACA analysis
BRFSS_BASE = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"

for year in range(2010, 2017):
    url = BRFSS_BASE.format(year=year)
    urllib.request.urlretrieve(url, f"data/raw/brfss/LLCP{year}.zip")
    with zipfile.ZipFile(f"data/raw/brfss/LLCP{year}.zip") as z:
        z.extractall(f"data/raw/brfss/{year}/")
# Load with pyreadstat or pandas read_sas

# ── ACA Medicaid Expansion State Panel ───────────────────────────────────────
# Expansion dates from Kaiser Family Foundation
import pandas as pd

KFF_URL = "https://www.kff.org/medicaid/state-indicator/medicaid-expansion-enrollment/"
# Download manually from KFF; or use pre-compiled:
expansion_dates = pd.read_csv("data/raw/aca/kff_expansion_dates.csv")

# BRFSS state-year aggregates merged to expansion status:
# Use BRFSS individual files + expansion_dates for DiD panel

# ── CPS (Current Population Survey) — Alternative insurance measure ──────────
CPS_URL = "https://cps.ipums.org/cps/"
# Download via IPUMS API (requires free account):
import ipumspy
# See ipumspy documentation for extract creation

# ── Synthetic data for time-varying confounding chapters ─────────────────────
import numpy as np

def simulate_brfss_longitudinal(n=5000, T=4, seed=42):
    """Simulate longitudinal insurance-health DGP for Chapters 25–30."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, n)          # unmeasured health propensity
    L = np.zeros((n, T))
    A = np.zeros((n, T))
    Y = np.zeros((n, T))
    for t in range(T):
        L[:, t] = 0.5 * U + (0.3 * A[:, t-1] if t > 0 else 0) + rng.normal(0, 0.5, n)
        A[:, t] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8*L[:,t] + 0.5*U)))).astype(float)
        Y[:, t] = 0.3 * A[:, t] + 0.4 * U + 0.2 * L[:, t] + rng.normal(0, 0.5, n)
    return A, L, Y, U
```

---

### Recommended GitHub Repository Structure

```
causal-inference-book/
│
├── data/
│   ├── raw/
│   │   ├── ohe/                    # Oregon HIF raw .dta files
│   │   ├── brfss/                  # BRFSS annual files by year
│   │   └── aca/                    # ACA expansion dates, CPS extracts
│   ├── processed/
│   │   ├── ohe_clean.parquet
│   │   ├── brfss_panel.parquet
│   │   └── aca_state_panel.parquet
│   └── README.md                   # Data download instructions
│
├── src/
│   ├── causal_book/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── loaders.py          # Dataset loading utilities
│   │   │   └── simulate.py         # DGP simulation functions (App. C)
│   │   ├── estimators/
│   │   │   ├── aipw.py             # AIPW base class
│   │   │   ├── dml.py              # DML cross-fitting template
│   │   │   ├── iv.py               # IV utilities
│   │   │   ├── did.py              # DiD and event study utilities
│   │   │   ├── gformula.py         # G-computation
│   │   │   └── sensitivity.py      # Sensitivity analysis wrappers
│   │   └── viz/
│   │       ├── balance.py          # LOVE plot template
│   │       ├── event_study.py      # Event study plot template
│   │       └── cate.py             # CATE visualization
│   └── setup.py
│
├── notebooks/
│   ├── part1_mindset/
│   │   ├── ch01_identification_first.ipynb
│   │   ├── ch02_estimands.ipynb
│   │   ├── ch03_design_checklist.ipynb
│   │   └── ch04_dags.ipynb
│   ├── part2_design/
│   │   ├── ch05_target_trial.ipynb
│   │   ├── ch06_treatment_timing.ipynb
│   │   ├── ch07_negative_controls.ipynb
│   │   └── ch08_measurement_error.ipynb
│   ├── part3_selection_observables/
│   │   ├── ch09_regression_adjustment.ipynb
│   │   ├── ch10_propensity_scores.ipynb
│   │   ├── ch11_doubly_robust.ipynb
│   │   ├── ch12_dml.ipynb
│   │   ├── ch13_meta_learners.ipynb
│   │   └── ch14_policy_learning.ipynb
│   ├── part4_did_panels/
│   │   ├── ch15_did_core.ipynb
│   │   ├── ch16_twfe_problem.ipynb
│   │   ├── ch17_modern_did.ipynb
│   │   ├── ch18_event_studies.ipynb
│   │   ├── ch19_synthetic_control.ipynb
│   │   └── ch20_panel_robustness.ipynb
│   ├── part5_quasi_experimental/
│   │   ├── ch21_iv_beyond_2sls.ipynb
│   │   ├── ch22_mte.ipynb
│   │   ├── ch23_rdd.ipynb
│   │   └── ch24_encouragement.ipynb
│   ├── part6_gmethods/
│   │   ├── ch25_time_varying_confounding.ipynb
│   │   ├── ch26_gformula.ipynb
│   │   ├── ch27_msm_iptw.ipynb
│   │   ├── ch28_snm_gestimation.ipynb
│   │   ├── ch29_dynamic_regimes.ipynb
│   │   └── ch30_mtp.ipynb
│   ├── part7_sensitivity/
│   │   ├── ch31_sensitivity_unobserved.ipynb
│   │   ├── ch32_oster_delta.ipynb
│   │   ├── ch33_partial_identification.ipynb
│   │   ├── ch34_placebo_system.ipynb
│   │   ├── ch35_multiple_testing.ipynb
│   │   └── ch36_transportability.ipynb
│   ├── part8_ml/
│   │   ├── ch37_prediction_causation.ipynb
│   │   ├── ch38_causal_forests.ipynb
│   │   ├── ch39_representation_learning.ipynb
│   │   ├── ch40_text_images.ipynb
│   │   ├── ch41_interference.ipynb
│   │   └── ch42_experiments_noncompliance.ipynb
│   ├── part9_applications/
│   │   ├── ch43_business_interventions.ipynb
│   │   ├── ch44_forecasting.ipynb
│   │   ├── ch45_ate_to_roi.ipynb
│   │   └── ch46_causal_monitoring.ipynb
│   ├── part10_capstone/
│   │   ├── ch47_causal_audit.ipynb
│   │   ├── ch48_credibility.ipynb
│   │   ├── ch49_method_selection.ipynb
│   │   └── ch50_frontier.ipynb
│   └── appendices/
│       ├── appA_math_toolkit.ipynb
│       ├── appB_software_patterns.ipynb
│       ├── appC_simulation_lab.ipynb
│       └── appD_reading_map.ipynb
│
├── tests/
│   ├── test_estimators.py          # Unit tests for src/ estimators
│   └── test_simulations.py         # Verify DGP properties
│
├── environment.yml                 # Conda environment
├── requirements.txt                # pip requirements
├── pyproject.toml
└── README.md
```

---

### Jupyter Notebook Naming Convention

```
ch{NN}_{snake_case_chapter_title}.ipynb
```

**Convention rules:**
- `NN` is zero-padded two-digit chapter number
- `snake_case_chapter_title` is the chapter title in lowercase with underscores (drop articles and prepositions)
- Appendices: `appA_`, `appB_`, `appC_`, `appD_`
- All notebooks begin with a standard header cell: chapter number, title, dataset used, and estimated runtime
- Each notebook ends with a **Summary cell** and a **Further Reading cell** linking to `appD_reading_map.ipynb`

**Examples:**
```
ch01_identification_first_principle.ipynb
ch12_double_debiased_ml.ipynb
ch17_modern_did_estimators.ipynb
ch21_iv_beyond_2sls.ipynb
ch38_causal_forests_honest_trees.ipynb
appC_simulation_laboratory.ipynb
```
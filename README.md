# Advanced Causal Inference: A Graduate Econometrician's Field Guide

**Running example**: *What is the causal effect of health insurance access on health and financial outcomes?*

A rigorous, practical guide to modern causal inference methods for readers who already hold a graduate econometrics background. Every method is implemented in Python against the same running example — the Oregon Health Insurance Experiment and ACA Medicaid expansion — so each chapter builds directly on the last.

---

## Structure

| Part | Chapters | Focus |
|------|----------|-------|
| [I — Mindset](chapters/part1_mindset/) | 1–4 | Identification-first thinking, estimands, design, DAGs |
| [II — Design Before Estimation](chapters/part2_design/) | 5–8 | Target trial, treatment timing, negative controls, measurement error |
| [III — Selection on Observables](chapters/part3_selection_observables/) | 9–14 | OLS → IPW → doubly robust → DML → meta-learners → policy learning |
| [IV — DiD, Event Studies, Panels](chapters/part4_did_panels/) | 15–20 | Modern DiD, TWFE problem, CS/SA/BJS estimators, synthetic control |
| [V — Quasi-Experimental Leverage](chapters/part5_quasi_experimental/) | 21–24 | IV beyond 2SLS, MTE, RDD, encouragement designs |
| [VI — G-Methods](chapters/part6_gmethods/) | 25–30 | G-formula, MSM/IPTW, SNM, dynamic regimes, modified treatment policies |
| [VII — Sensitivity & Credibility](chapters/part7_sensitivity/) | 31–36 | E-values, Oster, Manski bounds, placebo systems, multiple testing |
| [VIII — ML & Modern Data](chapters/part8_ml/) | 37–42 | Causal forests, representation learning, text/images, interference |
| [IX — Applications](chapters/part9_applications/) | 43–46 | Pricing, forecasting, ATE→ROI, causal monitoring |
| [X — Capstone](chapters/part10_capstone/) | 47–50 | Causal audit, full study design, method selection, frontier |
| [Appendices](appendices/) | A–D | Math toolkit, software patterns, simulation lab, reading map |

---

## Running Example Datasets

All datasets are publicly available. See [`data/README.md`](data/README.md) for download instructions.

| Dataset | Used in | Source |
|---------|---------|--------|
| Oregon Health Insurance Experiment (OHE) | Ch. 1–14, 21, 31–36 | [NBER](https://data.nber.org/oregon/) |
| BRFSS + ACA Medicaid Expansion Panel | Ch. 15–20, 25–30 | [CDC BRFSS](https://www.cdc.gov/brfss/) |
| Simulated longitudinal DGP | Ch. 25–30 | `src/causal_book/data/simulate.py` |

---

## Setup

```bash
git clone https://github.com/andybrob/advanced-causal-inference.git
cd advanced-causal-inference
pip install -r requirements.txt
# Download data:
python src/causal_book/data/download.py
```

---

## Chapter Index

### Part I — The Causal Inference Mindset After Graduate Econometrics
- [Chapter 1: The Identification-First Principle](chapters/part1_mindset/ch01_identification_first.md)
- [Chapter 2: Estimands — ATEs, CATEs, LATEs, and Policy Values](chapters/part1_mindset/ch02_estimands.md)
- [Chapter 3: The Causal Design Checklist](chapters/part1_mindset/ch03_design_checklist.md)
- [Chapter 4: Graphs for Econometricians — DAGs, SWIGs, and Selection Diagrams](chapters/part1_mindset/ch04_dags.md)

### Part II — Design Before Estimation
- Chapter 5: Target Trial Emulation *(coming soon)*
- Chapter 6: Treatment Timing, Time Zero, and Immortal Time Bias *(coming soon)*
- Chapter 7: Negative Controls, Placebos, and Falsification Tests *(coming soon)*
- Chapter 8: Measurement Error, Proxy Treatments, and Proxy Outcomes *(coming soon)*

### Part III — Selection on Observables, Done Seriously
- Chapter 9: Regression Adjustment Revisited *(coming soon)*
- Chapter 10: Propensity Scores Without Ritual *(coming soon)*
- Chapter 11: Doubly Robust Estimation *(coming soon)*
- Chapter 12: Double / Debiased Machine Learning *(coming soon)*
- Chapter 13: Meta-Learners for Heterogeneous Treatment Effects *(coming soon)*
- Chapter 14: Policy Learning and Treatment Rules *(coming soon)*

### Part IV — Difference-in-Differences, Event Studies, and Panels
- Chapter 15–20 *(coming soon)*

### Part V — Instruments, Discontinuities, and Quasi-Experimental Leverage
- Chapter 21–24 *(coming soon)*

### Part VI — Time-Varying Treatments and G-Methods
- Chapter 25–30 *(coming soon)*

### Part VII — Sensitivity Analysis and Credibility
- Chapter 31–36 *(coming soon)*

### Part VIII — Causal Inference with Machine Learning and Modern Data
- Chapter 37–42 *(coming soon)*

### Part IX — Business, Policy, and Decision Systems
- Chapter 43–46 *(coming soon)*

### Part X — Capstone Designs
- Chapter 47–50 *(coming soon)*

---

## Key Packages

```
causalml          # Meta-learners, uplift models
econml            # DML, causal forests, policy learning
linearmodels      # Panel IV, between/within estimators
doubleml          # DML reference implementation
csdid             # Callaway-Sant'Anna DiD
pgmpy             # Graphical models
zepid             # G-methods, marginal structural models
statsmodels       # OLS, IV, panel, time-series
scikit-learn      # ML nuisance estimation
```

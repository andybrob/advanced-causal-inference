# Chapter 50: The Frontier — Open Problems and Emerging Methods

## 50.1 Causal Discovery from Observational Data

Every chapter of this book begins with a DAG. We drew arrows from insurance selection to utilization, from income to health, from state policy to enrollment. But those graphs were *assumed*. The analyst specified them based on domain knowledge, prior literature, and judgment calls that rarely appear in the methods section. Causal discovery is the project of recovering graph structure from data — without assuming it.

The foundational result is the Markov equivalence class. Two DAGs $G_1$ and $G_2$ are *Markov equivalent* if they encode the same conditional independences — the same set of d-separations. Observational data alone, in the absence of interventions, can at best identify the equivalence class, not the DAG itself.

**Definition 50.1 (Completed Partially Directed Acyclic Graph, CPDAG).** A CPDAG $\mathcal{C}$ represents a Markov equivalence class if: (i) $\mathcal{C}$ contains an undirected edge $i - j$ iff every DAG in the class contains either $i \to j$ or $j \to i$; (ii) $\mathcal{C}$ contains $i \to j$ iff every DAG in the class contains $i \to j$.

The PC algorithm (Spirtes, Glymour, Scheines 2000) recovers the CPDAG under two assumptions: *Causal Faithfulness* (every conditional independence in the data corresponds to a d-separation in $G$) and *Causal Sufficiency* (no unmeasured common causes). It proceeds in two phases: (1) skeleton recovery via sequential conditional independence testing, and (2) orientation via v-structure detection and Meek rules.

**Algorithm 50.1 (PC Skeleton Recovery).** Start with the complete graph. For each pair $(i,j)$ and increasing conditioning set size $k = 0, 1, \ldots$: test $X_i \perp\!\!\!\perp X_j \mid \mathbf{S}$ for all $|\mathbf{S}| = k$ subsets of neighbors of $i$. Remove edge $i-j$ if any such independence holds. Store the separating set $\text{Sep}(i,j)$.

PC's sample complexity is well-understood. Under Gaussianity, independence tests use Fisher's z-transform, and the algorithm is consistent at rate $O(\log p / n)$ for sparse graphs. The FCI algorithm (Fast Causal Inference) relaxes causal sufficiency, allowing hidden common causes and representing them with bidirected edges $i \leftrightarrow j$ in a Partial Ancestral Graph (PAG).

**Theorem 50.1 (PC Consistency, Kalisch and Bühlmann 2007).** Under faithfulness, causal sufficiency, and Gaussian errors with $\|\Sigma\|_\infty \leq c$, the PC algorithm with significance level $\alpha_n \to 0$ satisfying $\alpha_n = O(n^{-\kappa})$ for some $\kappa > 0$ recovers the true CPDAG with probability tending to 1 as $n, p \to \infty$ with $\log p = o(n^{1-2\kappa})$.

The NOTEARS approach (Zheng et al. 2018) recasts the combinatorial search over DAGs as a continuous optimization problem. The key insight is an algebraic characterization of acyclicity.

**Definition 50.2 (NOTEARS Acyclicity Constraint).** For a weighted adjacency matrix $W \in \mathbb{R}^{d \times d}$, define

$$h(W) = \text{tr}\!\left(e^{W \circ W}\right) - d = 0$$

where $W \circ W$ is the elementwise square and $e^M = \sum_{k=0}^\infty M^k / k!$ is the matrix exponential. Then $h(W) = 0$ if and only if $W$ corresponds to a DAG.

*Proof sketch.* $(W \circ W)_{ij} = w_{ij}^2 \geq 0$. The $(k,k)$ entry of $(W \circ W)^m$ counts weighted walks of length $m$ from $k$ to $k$. Such a walk exists iff the graph has a cycle. Hence $h(W) = 0$ iff no cycle exists, iff $W$ is a DAG. $\square$

NOTEARS solves:

$$\min_W \frac{1}{2n}\|X - XW\|_F^2 + \lambda\|W\|_1 \quad \text{subject to } h(W) = 0$$

using an augmented Lagrangian. The objective is the least-squares score for a structural equation model $X_i = \sum_j w_{ji} X_j + \varepsilon_i$.

What would these algorithms reveal about the Oregon Health Insurance Experiment? The OHE has a known partial structure: lottery selection $Z$ precedes insurance take-up $D$, which precedes health outcomes $Y$. But the relationships among baseline covariates — household size, income proxies, prior health — are not specified. Running PC on the pre-randomization covariates would recover (testable) conditional independence structure among them, potentially identifying which controls are genuine confounders versus colliders. Using the discovered graph naively as if it were the true DAG propagates discovery uncertainty — an open problem we return to in §50.6.

## 50.2 Higher-Order Influence Functions and Debiased Estimation

Semiparametric efficiency theory (Chapter 25) guarantees that estimators built from the efficient influence function (EIF) achieve the asymptotic variance bound $\mathcal{V} = E[\phi^2(O; \eta)]$ where $\phi$ is the EIF and $\eta$ collects nuisance parameters. Double-robustness gives consistency when one of two nuisance models is misspecified. But this machinery has a gap: *when all nuisance models are estimated at slow rates*, first-order bias correction fails to yield $\sqrt{n}$-consistency.

Let $\tau = E[m(O; \eta)]$ be the target functional. The first-order plug-in estimator $\hat\tau_{\text{plug}} = \Psi(\hat\eta)$ has bias $\Psi(\hat\eta) - \Psi(\eta_0) \approx \langle D_\eta\Psi, \hat\eta - \eta_0\rangle$ where $D_\eta\Psi$ is the Gateaux derivative. The one-step correction adds $n^{-1}\sum_i \phi(O_i; \hat\eta)$ to cancel this first-order bias. The residual bias is $O(\|\hat\eta - \eta_0\|^2)$, which is $o(n^{-1/2})$ only if nuisance estimation error is $o(n^{-1/4})$ — fast enough.

**Definition 50.3 (Higher-Order Influence Functions).** The $k$-th order influence function of $\Psi$ at $\eta_0$ in direction $h$ is defined recursively by

$$\phi^{(k)}(o; \eta_0) = \frac{d}{d\epsilon}\bigg|_{\epsilon=0} \phi^{(k-1)}(o; \eta_{0,\epsilon})$$

where $\eta_{0,\epsilon}$ perturbs $\eta_0$ along $h$ with magnitude $\epsilon$.

The higher-order von Mises expansion gives:

$$\tau = \sum_{k=0}^K \frac{(-1)^k}{k!} \int \phi^{(k)}(o_1, \ldots, o_k; \eta_0) \prod_{j=1}^k d(\hat{P} - P_0)(o_j) + R_{K+1}$$

where the remainder satisfies $R_{K+1} = O(\|\hat\eta - \eta_0\|^{K+1})$. For $K=2$, the correction involves a U-statistic of order 2:

$$\hat\tau_{\text{HO}} = \hat\tau_{\text{plug}} + \frac{1}{n}\sum_i \phi^{(1)}(O_i; \hat\eta) + \frac{1}{n(n-1)}\sum_{i\neq j} \phi^{(2)}(O_i, O_j; \hat\eta)$$

**Theorem 50.2 (Higher-Order Bias Correction).** Under regularity conditions, if $\|\hat\eta - \eta_0\|_{L_2} = O(n^{-s})$ for some $s > 0$, then $\hat\tau_{\text{HO}}$ with $K = \lceil 1/(2s) \rceil$ orders of correction satisfies

$$\sqrt{n}(\hat\tau_{\text{HO}} - \tau_0) \xrightarrow{d} N(0, \mathcal{V})$$

without requiring $s > 1/4$.

This matters for the ATE in the Medicaid setting when propensity scores or outcome regressions are estimated with high-dimensional covariates or highly flexible neural networks that converge at rates slower than $n^{-1/4}$. With $K=2$ corrections the estimator tolerates $s > 1/6$; with $K=3$, it tolerates $s > 1/8$. The frontier question is: for which complex functionals (e.g., distributional effects, infinite-dimensional treatment effects) can one derive $\phi^{(k)}$ in closed form and compute the U-statistics efficiently?

The computational cost scales as $O(n^K)$, which is prohibitive for $K \geq 3$ at realistic sample sizes. Recent work on degenerate U-statistic approximations and kernel methods brings this to $O(n \log n)$ for specific functionals, but a general scalable procedure remains open.

## 50.3 Causal Representation Learning and ICA Identifiability

Observed variables are rarely the causal primitives. In the Medicaid context, "health" is a latent construct measured imperfectly by ER visits, self-reported status, and blood pressure readings. Causal representation learning asks: when can we recover latent causal variables $\mathbf{S}$ from observed mixtures $\mathbf{X}$, and when does the recovered representation support valid causal inference?

The classical result is Comon's theorem on Independent Component Analysis.

**Theorem 50.3 (ICA Identifiability, Comon 1994).** Let $X = AS$ where $A \in \mathbb{R}^{d \times d}$ is invertible and $S = (S_1, \ldots, S_d)^\top$ has mutually independent components with at most one Gaussian marginal. Then $A$ is identifiable up to permutation and scaling of columns.

*Proof sketch.* If $\tilde A$ is another mixing matrix with $X = \tilde A \tilde S$, then $\tilde S = \tilde A^{-1} A S = B S$ for $B = \tilde A^{-1}A$. Independence of $\tilde S$ requires independence of $BS$. The Darmois-Skitovitch theorem implies that for independent random variables, a linear combination is Gaussian iff all contributing variables are Gaussian. Since at most one $S_i$ is Gaussian, each row of $B$ must have exactly one nonzero entry — i.e., $B$ is a signed permutation matrix. $\square$

ICA identifiability extends to the nonlinear case only with additional structure. Hyvarinen and Morioka (2016) showed that nonlinear ICA is identifiable with auxiliary observed variables (time, segment labels) via a *contrastive* objective. Khemakhem et al. (2020) generalize this to variational autoencoders with conditionally factorial priors.

**Definition 50.4 (iVAE Identifiability).** Assume $X = f(S, U)$ where $f$ is injective, $S$ are latent sources, and $U$ are observed auxiliary variables. If $P(S \mid U) = \prod_i p_i(s_i \mid U)$ (conditional independence given $U$), then $f$ and $P(S\mid U)$ are identifiable up to component-wise reparametrization.

The causal representation learning frontier is the *causal disentanglement* problem: can we recover a representation $\hat S$ such that interventions on $S_i$ induce the correct downstream effects? Partial answers exist for linear-Gaussian models and for specific nonlinear cases with perfect interventions observed across multiple environments (Brehmer et al. 2022, von Kügelgen et al. 2023). The general nonlinear case with unknown intervention targets remains open.

For the OHE, a structural representation model might posit latent "health capital" and "financial vulnerability" as common causes of multiple outcomes. If we observe panel measurements (12-month and 24-month), an ICA-like model over residuals could recover these latents. But identifiability requires non-Gaussianity of the latent components and careful treatment of the time dimension.

## 50.4 Foundation Models and Causal Structure

Large language models trained on internet-scale text encode statistical patterns that partially reflect causal structure — temporal ordering, mechanism descriptions, intervention language. The question is whether this encoded knowledge can be formalized, extracted, and used for causal inference tasks.

**Definition 50.5 (LLM as Prior over DAGs).** Let $\mathcal{G}_d$ be the space of DAGs on $d$ nodes with labels $\{V_1, \ldots, V_d\}$. A language model $\mathcal{M}$ induces a distribution $P_\mathcal{M}(G)$ over $\mathcal{G}_d$ via causal queries: prompt $\mathcal{M}$ with "Does $V_i$ cause $V_j$?" and convert logit differences to edge probabilities.

Several empirical findings suggest LLMs have non-trivial causal knowledge: they score above chance on causal direction tasks (Kıcıman et al. 2023), they respect known causal asymmetries (causes precede effects temporally), and they can generate plausible causal graphs for well-documented domains. But they also confound correlation with causation in their training data, hallucinate mechanisms, and fail systematically on unfamiliar domains.

The more tractable question is *LLMs as weak supervision for causal discovery*. The idea is to use $P_\mathcal{M}(G)$ as a prior in a Bayesian causal discovery framework:

$$P(G \mid X) \propto P(X \mid G) \cdot P_\mathcal{M}(G)$$

where $P(X \mid G)$ is the marginal likelihood under the structural equation model. This combines data-driven independence evidence with LLM-encoded domain knowledge, with neither dominating. Vashishtha et al. (2023) show gains on BioGRID and other benchmarks, but theoretical guarantees are absent: $P_\mathcal{M}(G)$ has unknown miscalibration, and it is unclear when LLM errors compound or cancel data errors.

A separate line connects LLMs to *causal inference for text treatments*. In the Medicaid context, consider estimating the effect of insurance eligibility letters on enrollment — letters that differ in framing, language, and tone. The treatment is now a text object, the propensity score is a function of the text distribution, and the outcome model must relate text semantics to behavior. Egami et al. (2022) and Feder et al. (2022) develop frameworks for this; the efficiency theory for text-valued treatments is almost entirely open.

**Open Problem 50.1.** Define a semiparametric model for text-valued treatments $T \in \mathcal{T}$ (a metric space of documents) and binary outcome $Y$. Derive the efficient influence function for $\tau(t_1, t_2) = E[Y(t_1) - Y(t_2)]$ under no-unmeasured-confounding. Characterize the minimax estimation rate as a function of the intrinsic dimension of $\mathcal{T}$.

## 50.5 Federated Causal Inference

State-level Medicaid expansion data is distributed across agencies with data-sharing restrictions. The BRFSS is aggregated by CDC from state surveys. Estimating a pooled ATE requires either centralizing data (often legally impossible) or exchanging sufficient statistics. Federated learning provides the infrastructure; federated *causal* inference requires additional care about identifiability across sites.

Let there be $K$ data silos, each holding $n_k$ i.i.d. observations from distribution $P_k(Y, D, X)$. Heterogeneity across sites is inevitable: enrollment rates, population health baselines, and local healthcare supply differ by state. The key questions are: (1) what is identified from federated data under heterogeneous $P_k$; (2) what communication protocol achieves semiparametric efficiency.

**Definition 50.6 (Federated AIPW Estimator).** Suppose each site computes a local AIPW estimate

$$\hat\tau_k = \frac{1}{n_k}\sum_{i \in \mathcal{D}_k} \left[\frac{D_i Y_i}{\hat e_k(X_i)} - \frac{(D_i - \hat e_k(X_i))\hat\mu_k(X_i)}{\hat e_k(X_i)}\right] - \frac{(1-D_i)Y_i}{1 - \hat e_k(X_i)}$$

with local nuisance estimates $(\hat e_k, \hat\mu_k)$. The federated estimator is

$$\hat\tau_{\text{fed}} = \sum_{k=1}^K w_k \hat\tau_k, \quad w_k = \frac{n_k / \hat V_k}{\sum_j n_j / \hat V_j}$$

where $\hat V_k$ is the estimated variance of $\hat\tau_k$.

**Theorem 50.4 (Federated Semiparametric Efficiency).** Under site-specific nuisance rates $O(n_k^{-1/4})$, homogeneous treatment effect $\tau_0$, and bounded variance ratios, $\hat\tau_{\text{fed}}$ achieves asymptotic variance

$$\mathcal{V}_{\text{fed}} = \left(\sum_k n_k / \mathcal{V}_k\right)^{-1}$$

which equals the pooled semiparametric efficiency bound when $\mathcal{V}_k = \mathcal{V}$ for all $k$.

*Proof sketch.* Each $\hat\tau_k$ is asymptotically normal with variance $\mathcal{V}_k / n_k$ by standard AIPW theory. Under independence across sites and the optimal inverse-variance weights, the combined estimator variance is $({\sum_k n_k/\mathcal{V}_k})^{-1}$. When $\mathcal{V}_k = \mathcal{V}$ this equals $\mathcal{V} / N$ for $N = \sum_k n_k$, the pooled bound. $\square$

The efficiency result requires only that each site communicates $(\hat\tau_k, \hat V_k)$ — two scalars. No individual-level data leaves the silo. The open problem is heterogeneous treatment effects: if $\tau_k \neq \tau_{k'}$, the meta-estimator is targeting a weighted average that may not correspond to any policy-relevant quantity. Identification of the distribution of $\tau_k$ across sites — the *site-effect distribution* — requires exchangeability assumptions or random-effects models whose validity cannot be checked from the data.

A second open problem is federated *causal discovery*: running PC or NOTEARS with only local independence tests and aggregated sufficient statistics. For linear Gaussian models, the covariance matrix can be federated exactly. For nonparametric tests, aggregating p-values via Fisher's method or Stouffer's method introduces unknown correlations across tests that invalidate standard FCI orientation rules.

## 50.6 Semi-Parametric Theory Gaps

Several functionals that arise naturally in causal inference lack complete semiparametric efficiency theory. We enumerate three.

**Gap 1: Optimal Transport Causal Effects.** Define the distributional treatment effect as $\tau_{OT} = W_2(P_{Y(1)}, P_{Y(0)})$, the 2-Wasserstein distance between potential outcome distributions. The efficient influence function for $W_2$ is known in the location-scale family but not for general nonparametric distributions. The plug-in estimator using empirical distributions converges at rate $n^{-2/(d+4)}$ in dimension $d$ — dimension-dependent and slow. Whether a one-step correction can achieve $n^{-1/2}$ for $d=1$ is conjectured but unproven.

**Gap 2: Average Derivative in Partially Linear Models.** The partially linear model $Y = \tau D + g(X) + \varepsilon$ has a well-known Robinson (1988) estimator. But when $D = m(X) + \eta$ with unknown $m$, the efficient influence function for $\tau$ involves the conditional density $f_{D|X}$, and semiparametric efficiency bounds under shape restrictions on $g$ (monotone, convex) are not characterized.

**Gap 3: Staggered Adoption Efficiency.** In the ACA expansion setting, units adopt treatment at different calendar times. Callaway-Sant'Anna (2021) and Sun-Abraham (2021) identify group-time ATEs $ATT(g, t)$ under parallel trends. But the semiparametric efficiency bound for the aggregated estimand $\bar\tau = \sum_{g,t} \omega_{g,t} ATT(g,t)$ has not been derived for general weight functions $\omega_{g,t}$. The efficient estimator for specific aggregations (equal weights, event-time aggregation) is unknown.

## Python: Causal Discovery on OHE Covariates and Federated ATE Estimation

```python
"""
Chapter 50: Causal Discovery and Federated Estimation
Requires: causal-learn, notears-torch (or notears from lingam), numpy, pandas, scipy,
          sklearn, statsmodels, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

# ─── 1. Load OHE baseline covariates ──────────────────────────────────────────
# Variables available pre-randomization for causal discovery
# Using publicly available NBER Oregon data
# https://data.nber.org/oregon/

def load_ohe_covariates(path: str) -> pd.DataFrame:
    """
    Load OHE 12-month survey data.
    Returns pre-randomization covariates plus selection indicator.
    """
    df = pd.read_stata(path)
    covariate_cols = [
        'numhh_list',    # household size (strata variable)
        'age_inp',       # age
        'female_inp',    # female indicator
        'english_inp',   # English language
        'self_list',     # self-selected into lottery
        'weight_12m',    # survey weight
        'selected',      # lottery instrument Z
        'ohp_all_ever_admin',   # ever enrolled D
        'doc_any_12m',          # saw doctor Y1
        'catastrophic_exp_inp', # catastrophic expenditure Y2
    ]
    existing = [c for c in covariate_cols if c in df.columns]
    return df[existing].dropna()


def simulate_ohe_structure(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate OHE-like data with known DAG structure for validation.
    True DAG: age -> insurance, female -> insurance, insurance -> doc_visit,
              insurance -> no_catastrophic, numhh -> selected, selected -> insurance
    """
    rng = np.random.default_rng(seed)
    numhh = rng.integers(1, 4, size=n)
    age = rng.normal(40, 12, n).clip(18, 80)
    female = rng.binomial(1, 0.55, n)
    english = rng.binomial(1, 0.80, n)

    # Lottery selection depends only on household size (strata)
    p_selected = 0.30 - 0.05 * (numhh - 1)
    selected = rng.binomial(1, p_selected.clip(0.1, 0.5))

    # Insurance take-up: instrument + age + female
    log_odds_ins = -1.0 + 1.8 * selected - 0.01 * (age - 40) + 0.2 * female
    insurance = rng.binomial(1, 1 / (1 + np.exp(-log_odds_ins)))

    # Outcomes
    log_odds_doc = -0.5 + 1.2 * insurance + 0.01 * age - 0.1 * (numhh - 1)
    doc_visit = rng.binomial(1, 1 / (1 + np.exp(-log_odds_doc)))

    log_odds_cat = 0.5 - 1.5 * insurance + 0.02 * (age - 40)
    catastrophic = rng.binomial(1, 1 / (1 + np.exp(-log_odds_cat)))

    return pd.DataFrame({
        'numhh': numhh, 'age': age, 'female': female, 'english': english,
        'selected': selected, 'insurance': insurance,
        'doc_visit': doc_visit, 'catastrophic': catastrophic
    })


# ─── 2. PC Algorithm: skeleton recovery via partial correlations ───────────────

def partial_correlation(data: np.ndarray, i: int, j: int,
                        conditioning_set: list) -> tuple[float, float]:
    """
    Compute partial correlation of X_i and X_j given X_S.
    Returns (pcorr, p_value) using Fisher z-test.
    """
    n, p = data.shape
    if len(conditioning_set) == 0:
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
    else:
        S = conditioning_set
        # Regress X_i and X_j on X_S, take residual correlation
        X_S = data[:, S]
        X_S = np.column_stack([np.ones(n), X_S])
        coef_i = np.linalg.lstsq(X_S, data[:, i], rcond=None)[0]
        coef_j = np.linalg.lstsq(X_S, data[:, j], rcond=None)[0]
        res_i = data[:, i] - X_S @ coef_i
        res_j = data[:, j] - X_S @ coef_j
        r = np.corrcoef(res_i, res_j)[0, 1]

    r = np.clip(r, -0.9999, 0.9999)
    z = np.arctanh(r)
    df_eff = n - len(conditioning_set) - 2
    se = 1.0 / np.sqrt(max(df_eff - 1, 1))
    t_stat = z / se
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return r, p_val


def pc_skeleton(data: np.ndarray, alpha: float = 0.05,
                max_cond_size: int = 3) -> tuple[np.ndarray, dict]:
    """
    PC skeleton recovery via conditional independence testing.
    Returns adjacency matrix and separation sets.
    """
    n, p = data.shape
    adj = np.ones((p, p)) - np.eye(p)   # start complete
    sep_sets = {}

    for cond_size in range(max_cond_size + 1):
        pairs = [(i, j) for i in range(p) for j in range(i+1, p) if adj[i, j]]
        for i, j in pairs:
            if not adj[i, j]:
                continue
            # Conditioning candidates: current neighbors of i (excluding j)
            neighbors_i = [k for k in range(p) if adj[i, k] and k != j]
            if len(neighbors_i) < cond_size:
                continue
            # Test all subsets of size cond_size
            from itertools import combinations
            for S in combinations(neighbors_i, cond_size):
                _, p_val = partial_correlation(data, i, j, list(S))
                if p_val > alpha:
                    adj[i, j] = adj[j, i] = 0
                    sep_sets[(i, j)] = sep_sets[(j, i)] = list(S)
                    break

    return adj, sep_sets


def orient_v_structures(adj: np.ndarray, sep_sets: dict, p: int) -> np.ndarray:
    """
    Orient v-structures: i - k - j with i not adjacent to j and k not in Sep(i,j).
    Returns directed adjacency matrix (entry [i,j]=1 means i->j).
    """
    dag = adj.copy().astype(float)  # will orient edges
    for k in range(p):
        neighbors_k = [v for v in range(p) if adj[k, v]]
        for i in range(len(neighbors_k)):
            for j in range(i+1, len(neighbors_k)):
                ni, nj = neighbors_k[i], neighbors_k[j]
                if adj[ni, nj]:   # ni and nj are adjacent — not a v-structure
                    continue
                sep = sep_sets.get((ni, nj), sep_sets.get((nj, ni), []))
                if k not in sep:
                    # Orient ni -> k <- nj
                    dag[k, ni] = 0  # remove k->ni direction
                    dag[k, nj] = 0  # remove k->nj direction
    return dag


# ─── 3. NOTEARS: continuous optimization for DAG learning ──────────────────────

def notears_linear(X: np.ndarray, lambda1: float = 0.1,
                   max_iter: int = 100, h_tol: float = 1e-8,
                   rho_max: float = 1e16) -> np.ndarray:
    """
    NOTEARS algorithm for linear SEM with L1 penalty.
    Implements augmented Lagrangian on the acyclicity constraint h(W)=0.
    Returns weighted adjacency matrix W where W[i,j] = weight of i->j.
    """
    from scipy.optimize import minimize

    n, d = X.shape
    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-8)

    def h_func(W: np.ndarray) -> float:
        """h(W) = tr(exp(W◦W)) - d"""
        WW = W * W
        M = np.eye(d)
        expm_WW = np.zeros((d, d))
        fact = 1.0
        Mk = np.eye(d)
        for k in range(1, 20):
            Mk = Mk @ WW
            fact *= k
            expm_WW += Mk / fact
        expm_WW += np.eye(d)
        return np.trace(expm_WW) - d

    def grad_h(W: np.ndarray) -> np.ndarray:
        """Gradient of h(W) wrt W."""
        WW = W * W
        # Use matrix exponential approximation
        from scipy.linalg import expm
        E = expm(WW)
        return (E + E.T) * W  # chain rule through W◦W

    def objective(w_vec: np.ndarray) -> tuple[float, np.ndarray]:
        W = w_vec.reshape(d, d)
        W = W * (1 - np.eye(d))  # zero diagonal

        # Least squares loss
        residual = X_scaled - X_scaled @ W
        loss = 0.5 / n * np.sum(residual ** 2)
        grad_loss = -1.0 / n * X_scaled.T @ residual

        # Augmented Lagrangian terms (added by outer loop)
        h_val = h_func(W)
        h_grad = grad_h(W)

        # L1 regularization (approximate with smooth version)
        l1_loss = lambda1 * np.sum(np.abs(W))
        l1_grad = lambda1 * np.sign(W)

        total = loss + l1_loss
        grad_total = grad_loss + l1_grad

        return total, grad_total.flatten()

    # Augmented Lagrangian outer loop
    rho = 1.0
    alpha = 0.0   # Lagrange multiplier
    W_est = np.zeros((d, d))
    h_prev = np.inf

    for _ in range(max_iter):
        # Inner minimization with penalty term
        def aug_obj(w_vec):
            W = w_vec.reshape(d, d) * (1 - np.eye(d))
            residual = X_scaled - X_scaled @ W
            loss = 0.5 / n * np.sum(residual ** 2)
            l1_loss = lambda1 * np.sum(np.abs(W))
            h_val = h_func(W)
            h_grad = grad_h(W)
            penalty = alpha * h_val + 0.5 * rho * h_val ** 2
            total = loss + l1_loss + penalty
            grad_loss = -1.0 / n * X_scaled.T @ residual
            l1_grad = lambda1 * np.sign(W)
            grad_penalty = (alpha + rho * h_val) * h_grad
            return total, (grad_loss + l1_grad + grad_penalty).flatten()

        result = minimize(aug_obj, W_est.flatten(), method='L-BFGS-B',
                         jac=True, options={'maxiter': 1000, 'ftol': 1e-12})
        W_est = result.x.reshape(d, d) * (1 - np.eye(d))
        h_val = h_func(W_est)

        # Update multiplier and penalty
        alpha += rho * h_val
        if h_val > 0.25 * h_prev:
            rho = min(rho * 10, rho_max)
        h_prev = h_val

        if abs(h_val) < h_tol:
            break

    # Threshold small weights
    W_est[np.abs(W_est) < 0.1] = 0.0
    return W_est


# ─── 4. Federated AIPW Estimator ──────────────────────────────────────────────

def aipw_single_site(Y: np.ndarray, D: np.ndarray,
                     X: np.ndarray) -> tuple[float, float]:
    """
    Compute AIPW estimate of ATE at a single site.
    Returns (tau_hat, variance_hat).
    """
    n = len(Y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Propensity score via cross-fitted logistic regression
    e_hat = cross_val_predict(
        LogisticRegressionCV(cv=5, max_iter=1000), X_s, D,
        cv=5, method='predict_proba'
    )[:, 1]
    e_hat = np.clip(e_hat, 0.05, 0.95)

    # Outcome model via cross-fitted Lasso (separate for D=0, D=1)
    mu1_hat = cross_val_predict(LassoCV(cv=5, max_iter=2000),
                                X_s[D == 1], Y[D == 1], cv=5)
    mu0_hat = cross_val_predict(LassoCV(cv=5, max_iter=2000),
                                X_s[D == 0], Y[D == 0], cv=5)

    # Project to full sample
    from sklearn.linear_model import LassoCV as LCV
    m1 = LCV(cv=5, max_iter=2000).fit(X_s[D == 1], Y[D == 1])
    m0 = LCV(cv=5, max_iter=2000).fit(X_s[D == 0], Y[D == 0])
    mu1_full = m1.predict(X_s)
    mu0_full = m0.predict(X_s)

    # AIPW scores
    psi = (D * Y / e_hat - (D - e_hat) * mu1_full / e_hat -
           ((1 - D) * Y / (1 - e_hat) + (D - e_hat) * mu0_full / (1 - e_hat)))
    tau_hat = psi.mean()
    var_hat = np.var(psi, ddof=1) / n
    return tau_hat, var_hat


def simulate_aca_states(n_states: int = 10, obs_per_state: int = 500,
                        true_ate: float = 0.06, seed: int = 42) -> list[dict]:
    """
    Simulate ACA Medicaid expansion data across states.
    Each state has a local ATE near true_ate with state-level heterogeneity.
    Returns list of site data dicts.
    """
    rng = np.random.default_rng(seed)
    sites = []
    for k in range(n_states):
        n = obs_per_state + rng.integers(-100, 100)
        # State-level confounders: income, urban fraction, baseline health
        X = rng.normal(0, 1, size=(n, 4))

        # State-level ATE heterogeneity
        tau_k = true_ate + rng.normal(0, 0.02)

        # Propensity (expansion status depends on state politics ~ income proxy)
        e = 1 / (1 + np.exp(-(-0.5 + 0.4 * X[:, 0] + 0.2 * X[:, 1])))
        D = rng.binomial(1, e)

        # Potential outcomes
        Y0 = 0.5 + 0.3 * X[:, 0] - 0.1 * X[:, 2] + rng.normal(0, 0.3, n)
        Y1 = Y0 + tau_k + 0.05 * X[:, 1]
        Y = D * Y1 + (1 - D) * Y0

        sites.append({'Y': Y, 'D': D, 'X': X, 'tau_k': tau_k, 'n': n, 'k': k})
    return sites


def federated_aipw(sites: list[dict]) -> dict:
    """
    Federated AIPW: each site computes local estimate, combine with
    inverse-variance weights. No individual data leaves silo.
    """
    local_estimates = []
    for site in sites:
        tau_k, var_k = aipw_single_site(site['Y'], site['D'], site['X'])
        local_estimates.append({'k': site['k'], 'n': site['n'],
                                'tau': tau_k, 'var': var_k,
                                'tau_true': site['tau_k']})

    # Inverse-variance weights
    inv_vars = np.array([1.0 / e['var'] for e in local_estimates])
    weights = inv_vars / inv_vars.sum()
    tau_fed = sum(w * e['tau'] for w, e in zip(weights, local_estimates))
    var_fed = 1.0 / inv_vars.sum()

    # Pooled naive comparison (as if data were combined)
    all_Y = np.concatenate([s['Y'] for s in sites])
    all_D = np.concatenate([s['D'] for s in sites])
    all_X = np.vstack([s['X'] for s in sites])
    tau_pooled, var_pooled = aipw_single_site(all_Y, all_D, all_X)

    return {
        'local': local_estimates,
        'tau_fed': tau_fed, 'var_fed': var_fed,
        'tau_pooled': tau_pooled, 'var_pooled': var_pooled,
        'weights': weights
    }


# ─── 5. Run and visualize ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: Causal Discovery on Simulated OHE Data")
    print("=" * 60)

    df_sim = simulate_ohe_structure(n=3000)
    var_names = df_sim.columns.tolist()
    data_array = df_sim.values.astype(float)

    # PC skeleton recovery
    print("\nRunning PC skeleton recovery (alpha=0.01)...")
    adj_pc, sep_sets = pc_skeleton(data_array, alpha=0.01, max_cond_size=2)
    adj_oriented = orient_v_structures(adj_pc, sep_sets, len(var_names))

    print("\nPC Algorithm — Discovered adjacency (upper triangle = directed edges):")
    print(pd.DataFrame(adj_oriented, index=var_names, columns=var_names).astype(int))

    # True DAG for comparison
    true_edges = [
        ('numhh', 'selected'), ('selected', 'insurance'),
        ('age', 'insurance'), ('female', 'insurance'),
        ('insurance', 'doc_visit'), ('insurance', 'catastrophic'),
        ('age', 'catastrophic')
    ]
    p = len(var_names)
    true_adj = np.zeros((p, p))
    for (src, tgt) in true_edges:
        i, j = var_names.index(src), var_names.index(tgt)
        true_adj[i, j] = 1.0

    # Compare skeleton precision/recall
    discovered_skel = (adj_pc > 0).astype(int)
    true_skel = ((true_adj + true_adj.T) > 0).astype(int)
    np.fill_diagonal(discovered_skel, 0)
    np.fill_diagonal(true_skel, 0)
    tp = (discovered_skel * true_skel).sum() / 2
    fp = (discovered_skel * (1 - true_skel)).sum() / 2
    fn = ((1 - discovered_skel) * true_skel).sum() / 2
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nSkeleton Precision: {precision:.3f}  Recall: {recall:.3f}  "
          f"F1: {2*precision*recall/(precision+recall+1e-9):.3f}")

    # NOTEARS
    print("\nRunning NOTEARS (lambda=0.1)...")
    # Use subset of continuous variables for NOTEARS
    cont_vars = ['numhh', 'age', 'female', 'selected', 'insurance', 'doc_visit']
    data_cont = df_sim[cont_vars].values.astype(float)
    W_notears = notears_linear(data_cont, lambda1=0.1, max_iter=50)
    print("\nNOTEARS weight matrix (rows=source, cols=target, threshold |w|>0.1):")
    W_df = pd.DataFrame(np.round(W_notears, 3), index=cont_vars, columns=cont_vars)
    print(W_df)

    # ── Figure 1: Discovered vs. true DAG comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def draw_dag(ax, adj_matrix, names, title):
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

        n_nodes = len(names)
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = {i: (0.5 + 0.42 * np.cos(a), 0.5 + 0.42 * np.sin(a))
                     for i, a in enumerate(angles)}

        # Draw edges
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] > 0.05:
                    x0, y0 = positions[i]
                    x1, y1 = positions[j]
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle='->', color='steelblue',
                                               lw=1.5, connectionstyle='arc3,rad=0.1'))

        # Draw nodes
        for i, name in enumerate(names):
            x, y = positions[i]
            ax.add_patch(plt.Circle((x, y), 0.06, color='white',
                                     ec='steelblue', lw=2, zorder=3))
            ax.text(x, y, name, ha='center', va='center',
                    fontsize=7, fontweight='bold', zorder=4)

    draw_dag(axes[0], true_adj, var_names, "True DAG (OHE Structure)")
    draw_dag(axes[1], adj_oriented, var_names, "PC Algorithm: Discovered CPDAG")
    plt.tight_layout()
    plt.savefig("/tmp/ch50_causal_discovery.png", dpi=150, bbox_inches='tight')
    print("\nDAG comparison saved to /tmp/ch50_causal_discovery.png")

    print("\n" + "=" * 60)
    print("PART 2: Federated AIPW on Simulated ACA State Data")
    print("=" * 60)

    sites = simulate_aca_states(n_states=10, obs_per_state=600,
                                 true_ate=0.06, seed=7)
    true_ate_avg = np.mean([s['tau_k'] for s in sites])
    print(f"\nTrue average ATE across states: {true_ate_avg:.4f}")

    results = federated_aipw(sites)

    print(f"\nFederated AIPW:  tau = {results['tau_fed']:.4f}  "
          f"SE = {np.sqrt(results['var_fed']):.4f}  "
          f"95% CI = ({results['tau_fed'] - 1.96*np.sqrt(results['var_fed']):.4f}, "
          f"{results['tau_fed'] + 1.96*np.sqrt(results['var_fed']):.4f})")
    print(f"Pooled AIPW:     tau = {results['tau_pooled']:.4f}  "
          f"SE = {np.sqrt(results['var_pooled']):.4f}  "
          f"95% CI = ({results['tau_pooled'] - 1.96*np.sqrt(results['var_pooled']):.4f}, "
          f"{results['tau_pooled'] + 1.96*np.sqrt(results['var_pooled']):.4f})")

    # Local estimates table
    print("\nLocal site estimates:")
    print(f"{'Site':>4}  {'n':>5}  {'tau_true':>9}  {'tau_hat':>9}  "
          f"{'SE':>7}  {'weight':>7}")
    for est, w in zip(results['local'], results['weights']):
        print(f"{est['k']:>4}  {est['n']:>5}  {est['tau_true']:>9.4f}  "
              f"{est['tau']:>9.4f}  {np.sqrt(est['var']):>7.4f}  {w:>7.4f}")

    # ── Figure 2: Federated vs. local estimates
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Forest plot of local estimates
    ax = axes[0]
    local = results['local']
    ks = [e['k'] for e in local]
    taus = np.array([e['tau'] for e in local])
    ses = np.array([np.sqrt(e['var']) for e in local])
    true_taus = np.array([e['tau_true'] for e in local])

    ax.errorbar(taus, ks, xerr=1.96 * ses, fmt='o', color='steelblue',
                ecolor='steelblue', elinewidth=1.5, capsize=4, label='Site AIPW (95% CI)')
    ax.scatter(true_taus, ks, marker='|', color='crimson', s=200, zorder=5,
               label='True site ATE')
    ax.axvline(results['tau_fed'], color='darkgreen', linestyle='--', lw=2,
               label=f'Federated: {results["tau_fed"]:.3f}')
    ax.axvline(true_ate_avg, color='crimson', linestyle=':', lw=2,
               label=f'True avg ATE: {true_ate_avg:.3f}')
    ax.set_xlabel('Estimated ATE')
    ax.set_ylabel('State')
    ax.set_title('Panel A: Forest Plot — State-Level Estimates', fontweight='bold')
    ax.set_yticks(ks)
    ax.set_yticklabels([f'State {k+1}' for k in ks])
    ax.legend(fontsize=8)

    # Panel B: Federated vs. pooled comparison with efficiency
    ax = axes[1]
    n_states_range = range(2, 11)
    fed_ses, pooled_ses = [], []
    for ns in n_states_range:
        sub_sites = simulate_aca_states(n_states=ns, obs_per_state=600, seed=99)
        r = federated_aipw(sub_sites)
        fed_ses.append(np.sqrt(r['var_fed']))
        pooled_ses.append(np.sqrt(r['var_pooled']))

    ax.plot(list(n_states_range), fed_ses, 'o-', color='steelblue',
            label='Federated AIPW SE')
    ax.plot(list(n_states_range), pooled_ses, 's--', color='darkgreen',
            label='Pooled AIPW SE')
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Standard Error of ATE')
    ax.set_title('Panel B: Efficiency — Federated vs. Pooled', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig("/tmp/ch50_federated_estimation.png", dpi=150, bbox_inches='tight')
    print("\nFederated estimation figure saved to /tmp/ch50_federated_estimation.png")
    print("\nDone.")
```

## Summary

- Causal discovery algorithms (PC, FCI, NOTEARS) recover Markov equivalence classes from conditional independence tests; they identify DAG structure up to the CPDAG without requiring analyst-specified graphs, but require faithfulness and causal sufficiency assumptions that are untestable.
- NOTEARS recasts the combinatorial DAG search as smooth continuous optimization via the acyclicity constraint $h(W) = \text{tr}(e^{W \circ W}) - d = 0$, enabling gradient-based DAG learning but sacrificing theoretical consistency guarantees of score-based or constraint-based methods.
- Higher-order influence functions extend double-robustness to settings where nuisance rates are slower than $n^{-1/4}$: the $K$-th order von Mises expansion corrects bias to $O(\|\hat\eta - \eta_0\|^{K+1})$, tolerating nuisance convergence rates of $O(n^{-1/(2K+2)})$ while achieving semiparametric efficiency.
- ICA and causal representation learning identify latent causal variables under independence and non-Gaussianity conditions, with the nonlinear identifiable case requiring auxiliary observed variables; the general causal disentanglement problem with unknown intervention targets remains open.
- Federated AIPW achieves the pooled semiparametric efficiency bound by combining local inverse-variance-weighted estimates, requiring only two scalars per site — but heterogeneous treatment effects across sites undermine the target estimand unless exchangeability is assumed.
- Critical open problems include: efficiency bounds for optimal-transport causal effects, semiparametric theory for text-valued treatments, federated causal discovery with aggregated independence tests, and provably calibrated LLM priors over DAGs.
- Foundation models encode causal knowledge that is empirically useful but theoretically uncharacterized — the frontier is formalizing this as a prior in Bayesian causal discovery with quantified miscalibration.

## Further Reading

- **Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and Search*, 2nd ed. MIT Press.** The definitive reference for constraint-based causal discovery; proves PC/FCI correctness and develops the theory of Markov equivalence, faithfulness, and causal sufficiency in depth.

- **Zheng, Aragam, Ravikumar, Xing (2018). "DAGs with NO TEARS: Continuous Optimization for Structure Learning." *NeurIPS*.** Introduces the $h(W) = \text{tr}(e^{W \circ W}) - d$ characterization and the augmented Lagrangian algorithm; catalyzed a generation of continuous-optimization causal discovery methods.

- **Robins, Li, Tchetgen, van der Vaart (2008). "Higher-Order Influence Functions and Minimax Estimation of Nonlinear Functionals." *IMS Collections*.** Derives the higher-order von Mises expansion and proves that $K$ orders of correction suffice for nuisance rates $s > 1/(2K+2)$; the theoretical foundation for bias correction beyond double-robustness.

- **Khemakhem, Kingma, Monti, Hyvarinen (2020). "Variational Autoencoders and Nonlinear ICA: A Unifying Framework." *AISTATS*.** Proves identifiability of deep generative models under conditionally factorial priors, connecting variational inference to causal representation learning with formal guarantees.

- **Vashishtha, Reddy, Bhatt, Ettinger, Alikhani, Blodgett, Muresan (2023). "Causal Inference from Text: A Concise Survey." *ACL Findings*.** Reviews causal discovery and estimation with text data including LLM-based graph construction; identifies open problems at the intersection of NLP and causal inference.

- **Duan, Ning, Chen, Tian (2023). "Heterogeneous Federated Learning via Grouped Sequential-Interactive Gradient Pursuit." *ICML*.** Addresses federated estimation under heterogeneous site distributions; develops theory for when federated and pooled estimators agree and quantifies the efficiency loss from site heterogeneity.
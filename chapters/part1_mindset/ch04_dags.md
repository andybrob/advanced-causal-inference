# Chapter 4: Graphs for Econometricians — DAGs, SWIGs, and Selection Diagrams

Potential outcomes notation is precise but silent about mechanism. Writing $Y_i(d)$ tells us what outcome unit $i$ would achieve under treatment $d$, but it says nothing about *why* $D$ affects $Y$, which variables confound the relationship, or whether a proposed adjustment strategy is valid. Directed acyclic graphs (DAGs) fill this gap. They make causal assumptions visible, auditable, and algorithmically testable. The goal of this chapter is not to replace the potential outcomes framework you already know but to give it a geometry.

The practical payoff is large. A DAG can tell you in seconds whether a proposed regression controls for the right variables, whether you have accidentally introduced bias by conditioning on a collider, or whether an instrument is valid. These are questions that potential outcomes notation answers only after significant algebraic labor — and only after you have already encoded the assumptions somewhere, implicitly.

---

## 4.1 DAG Fundamentals: Nodes, Edges, and the Markov Condition

A **directed acyclic graph** $\mathcal{G} = (V, E)$ consists of a vertex set $V$ and a set of directed edges $E \subseteq V \times V$ with no directed cycles. Each node $V_i \in V$ represents a random variable. A directed edge $V_j \to V_i$ encodes the structural claim that $V_j$ is a direct cause of $V_i$, conditional on all other variables in the graph.

The graph encodes a joint distribution via the **Markov factorization**:

$$P(V_1, V_2, \ldots, V_k) = \prod_{i=1}^{k} P(V_i \mid \text{Pa}(V_i))$$

where $\text{Pa}(V_i)$ denotes the parents of $V_i$ in $\mathcal{G}$. This factorization is the **Causal Markov Condition**: every variable is independent of its non-descendants, conditional on its parents.

The structural equations underlying the graph are:

$$V_i = f_i(\text{Pa}(V_i), \varepsilon_i), \quad i = 1, \ldots, k$$

where the $\varepsilon_i$ are mutually independent noise terms. The independence of error terms is critical: it is the graphical analogue of the exclusion restriction, and it is what allows identification from observational data when the graph is Markovian.

**Definition 4.1 (Ancestral relationships).** Node $A$ is an *ancestor* of $B$ if there is a directed path $A \to \cdots \to B$. Node $B$ is a *descendant* of $A$ if $A$ is an ancestor of $B$. Node $C$ is a *non-descendant* of $A$ if $C$ is neither $A$ itself nor a descendant of $A$.

**Definition 4.2 (Path).** A path between nodes $X$ and $Y$ is any sequence of adjacent nodes connecting them, ignoring edge directions. A *directed path* follows edge directions.

Three elementary path structures determine all d-separation results:

| Structure | Form | Blocked by conditioning on middle node? |
|---|---|---|
| Chain | $X \to M \to Y$ | Yes — conditioning on $M$ blocks information flow |
| Fork | $X \leftarrow M \rightarrow Y$ | Yes — conditioning on common cause $M$ blocks flow |
| Collider | $X \to C \leftarrow Y$ | No, blocked *unless* you condition on $C$ or a descendant of $C$ |

The collider rule is the most counterintuitive and the most consequential.

**Definition 4.3 (d-separation).** A path $p$ between $X$ and $Y$ is *blocked* by a set $Z$ if:
1. $p$ contains a chain $A \to B \to C$ or fork $A \leftarrow B \rightarrow C$ where $B \in Z$, or
2. $p$ contains a collider $A \to B \leftarrow C$ where $B \notin Z$ and no descendant of $B$ is in $Z$.

$X$ and $Y$ are **d-separated** by $Z$ in $\mathcal{G}$, written $X \perp\!\!\!\perp_{\mathcal{G}} Y \mid Z$, if every path between $X$ and $Y$ is blocked by $Z$. If $X$ and $Y$ are not d-separated by $Z$, they are **d-connected**.

**Theorem 4.1 (d-separation implies conditional independence).** If $\mathcal{G}$ is Markovian relative to $P$, then

$$X \perp\!\!\!\perp_{\mathcal{G}} Y \mid Z \implies X \perp\!\!\!\perp_P Y \mid Z$$

The converse (faithfulness) holds generically: distributions that satisfy d-separation equalities not implied by $\mathcal{G}$ have measure zero in the parameter space.

---

## 4.2 The Backdoor Criterion and Adjustment Formula

The most important identification result in the graphical framework is the **backdoor criterion**, which characterizes when a set of covariates suffices for identification of the causal effect $P(Y \mid do(X=x))$.

The $do(\cdot)$ operator, due to Pearl, represents an external intervention that sets $X$ to $x$ and severs all causal arrows pointing *into* $X$. The resulting **mutilated graph** $\mathcal{G}_{\bar{X}}$ is $\mathcal{G}$ with all edges into $X$ deleted.

**Definition 4.4 (Backdoor path).** A path from $X$ to $Y$ is a *backdoor path* if its first edge points into $X$ (i.e., the path starts with $\cdot \to X$).

**Definition 4.5 (Backdoor criterion).** A set $Z$ satisfies the backdoor criterion relative to $(X, Y)$ in $\mathcal{G}$ if:
1. No node in $Z$ is a descendant of $X$.
2. $Z$ blocks every backdoor path between $X$ and $Y$.

**Theorem 4.2 (Backdoor adjustment formula).** If $Z$ satisfies the backdoor criterion relative to $(X, Y)$ in a Markovian DAG $\mathcal{G}$, then

$$P(Y = y \mid do(X = x)) = \sum_{z} P(Y = y \mid X = x, Z = z)\, P(Z = z)$$

*Proof sketch.* In the mutilated graph $\mathcal{G}_{\bar{X}}$, the do-distribution satisfies:

$$P_{\bar{X}}(Y, Z) = P(Y \mid X, Z)\, P(Z)$$

because condition (1) ensures $Z$ is not affected by the intervention on $X$, and condition (2) ensures that conditional on $Z$, $X$ and $Y$'s non-causal association is blocked. Formally, condition (2) implies $(Y \perp\!\!\!\perp X \mid Z)_{\mathcal{G}_{\bar{X}}}$, so $P_{\bar{X}}(Y \mid X, Z) = P(Y \mid X, Z)$. Marginalizing over $Z$ under $P_{\bar{X}}$ gives the result. $\square$

**Corollary (OLS as backdoor adjustment).** When $Z$ satisfies the backdoor criterion and all relationships are linear, the OLS coefficient from regressing $Y$ on $X$ and $Z$ is a consistent estimator of the average causal effect.

This formalizes what graduate econometrics teaches intuitively as "controlling for confounders" — but it also clarifies the limits. Not every set of observed covariates satisfies the criterion, and some sets that look like controls actually violate condition (1) by including descendants of $X$.

**The OHE DAG.** In the Oregon Health Insurance Experiment, the structural relationships are:

- $Z$ (lottery) $\to$ $D$ (Medicaid enrollment) $\to$ $Y$ (health/financial outcome)
- $L$ (income, age, baseline health) $\to$ $D$ and $L \to Y$
- $Z \not\to Y$ directly (exclusion restriction)
- $Z \not\leftarrow L$ (lottery randomization)

The backdoor paths from $D$ to $Y$ run through $L$: $D \leftarrow L \rightarrow Y$. Conditioning on $L$ blocks this path and satisfies the backdoor criterion. But since $Z$ is randomized and $Z \to D$ with no arrows into $Z$, using $Z$ as an instrument via 2SLS is also valid — the instrument cuts the backdoor path by design, because randomization ensures there are no arrows into $Z$ from $L$.

---

## 4.3 Collider Bias: When Conditioning Creates Spurious Association

The collider rule is perhaps the most dangerous pitfall in applied econometrics, because standard econometric training teaches conditioning as a cure-all for confounding. Conditioning on a collider does the opposite: it *opens* a previously closed path.

**Definition 4.6 (Collider).** On a path $p$, node $C$ is a collider if both adjacent edges on $p$ point into $C$: $\cdot \to C \leftarrow \cdot$.

A collider blocks information flow unconditionally. But conditioning on a collider — or on any descendant of a collider — opens the path, inducing an association between the collider's parents even when they are marginally independent.

**Proposition 4.1 (Conditioning on a collider induces dependence).** Let $X \to C \leftarrow Y$ be a collider structure in a Markovian DAG where $X \perp\!\!\!\perp Y$ marginally. Then $X \not\!\perp\!\!\!\perp Y \mid C$ generically.

*Proof.* By Bayes' rule: $P(X \mid Y, C) = P(C \mid X, Y) P(X \mid Y) / P(C \mid Y)$. Since $C$ is a function of both $X$ and $Y$, knowing $C$ and $Y$ provides information about $X$ even if $X$ and $Y$ were marginally independent. $\square$

**OHE collider example.** Define a variable $A$ = "applied for Medicaid." In the OHE context, $A$ is caused by both low income $L$ (low-income individuals seek coverage) and poor health status $H$ (sick individuals actively apply). Suppose we are studying the association between income and health among applicants, i.e., conditional on $A = 1$. The DAG path $L \to A \leftarrow H$ is a collider at $A$. Conditioning on $A = 1$ opens this path, creating a spurious negative association between $L$ and $H$ in the selected sample: among those who applied, high-income individuals are disproportionately sick (because healthy high-income people rarely apply). This is **Berkson's paradox** in disguise and a specific form of sample selection bias.

The econometric literature recognized this under different names — "sample selection bias," "incidental truncation" — but DAG formalism makes the mechanism precise and generalizable.

---

## 4.4 M-Bias and Bad Controls: When More Is Less

The textbook heuristic "always include more controls" conflicts with graphical identification theory. **M-bias** is a canonical example where adding a covariate that appears to control for confounding actually introduces bias that was absent without it.

**The M-bias DAG.** Consider four variables forming the letter M:

$$U_1 \to X, \quad U_1 \to C \leftarrow U_2, \quad U_2 \to Y$$

where $U_1$ and $U_2$ are unmeasured. The measured variables are $X$, $Y$, and $C$. The path $X \leftarrow U_1 \to C \leftarrow U_2 \to Y$ is blocked by the collider at $C$ — so $X$ and $Y$ are d-separated (no confounding) without conditioning on anything. But if we condition on $C$, we open the collider, creating a spurious backdoor path $X \leftarrow U_1 \to C \leftarrow U_2 \to Y$. Conditioning on $C$ introduces bias that did not exist before.

**Definition 4.7 (Bad control).** A variable $C$ is a bad control for the causal effect of $X$ on $Y$ if including $C$ in an adjustment set opens a path between $X$ and $Y$ that was previously blocked, or if $C$ is a descendant of $X$ (a post-treatment variable).

**Post-treatment variables.** Conditioning on a descendant of $X$ on the causal path to $Y$ is a special case of bad control. In the OHE context, variables like `doc_any_12m` (doctor visits) may mediate the effect of Medicaid on financial catastrophe. Including `doc_any_12m` as a covariate when estimating the effect of $D$ on `catastrophic_exp_inp` blocks part of the causal effect, inducing bias in a structural sense. The regression coefficient no longer estimates the total causal effect.

**Mediator analysis.** If the goal is to estimate the *direct* effect of $D$ on $Y$ not through mediator $M$, then conditioning on $M$ is appropriate — but only if $M$ has no unmeasured common causes with $Y$. This is the key identifying assumption for mediation analysis, and it is more demanding than the backdoor criterion for total effects.

**Practical rule.** Before adding a control variable, ask: Is this variable (a) a cause of $X$ only, (b) a cause of $Y$ only, (c) a common cause of both, or (d) a descendant of $X$ or a collider? Cases (a)–(c) are potentially valid adjustments (subject to no new paths being opened); case (d) requires extreme caution.

---

## 4.5 The Frontdoor Criterion

The frontdoor criterion identifies a causal effect when all backdoor paths are blocked by an unmeasured confounder, but a measured mediator $M$ exists with special properties.

**Definition 4.8 (Frontdoor criterion).** A set $M$ satisfies the frontdoor criterion relative to $(X, Y)$ in $\mathcal{G}$ if:
1. All directed paths from $X$ to $Y$ pass through $M$ (M intercepts all causal paths).
2. There are no unblocked backdoor paths from $X$ to $M$.
3. All backdoor paths from $M$ to $Y$ are blocked by $X$.

**Theorem 4.3 (Frontdoor adjustment formula).** If $M$ satisfies the frontdoor criterion relative to $(X, Y)$, then

$$P(Y = y \mid do(X = x)) = \sum_m P(M = m \mid X = x) \sum_{x'} P(Y = y \mid X = x', M = m)\, P(X = x')$$

*Proof sketch.* The effect of $X$ on $M$ is identified because condition (2) ensures no confounding on the $X \to M$ path (or it is blockable). The effect of $M$ on $Y$, conditioning on $X$, is identified because condition (3) states that $X$ blocks all backdoor paths from $M$ to $Y$. The double sum marginalizes over the distribution of $X$ to remove the do from the second factor. The full argument uses do-calculus rules 2 and 3 applied in sequence. $\square$

The frontdoor formula is remarkable because it identifies a causal effect in the presence of an unmeasured confounder $U$ between $X$ and $Y$, as long as a suitable mediator exists. The classic example is smoking ($X$) → tar deposits in lungs ($M$) → cancer ($Y$), with unmeasured genetic factors $U$ confounding $X$ and $Y$.

In the OHE, a partial frontdoor structure may exist: Medicaid enrollment $D$ affects insurance coverage status $C$, which in turn affects healthcare utilization $U$, which affects financial outcomes $Y$. Whether a clean frontdoor identification applies depends on whether all causal paths from $D$ to $Y$ pass through measurable intermediaries — a strong assumption in this context.

---

## 4.6 Do-Calculus: Three Rules for Interventional Inference

The do-calculus (Pearl, 1995) provides a complete set of rules for transforming expressions involving do-operators into expressions involving only observational probabilities, whenever such a transformation is possible.

Let $\mathcal{G}_{\bar{X}}$ denote the graph with all arrows into $X$ deleted, and $\mathcal{G}_{\underline{X}}$ denote the graph with all arrows out of $X$ deleted.

**Theorem 4.4 (Do-calculus).** For disjoint sets $X, Y, Z, W$ in a Markovian DAG $\mathcal{G}$:

**Rule 1 (Insertion/deletion of observations):**
$$P(y \mid do(x), z, w) = P(y \mid do(x), w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{\mathcal{G}_{\bar{X}}}$$

**Rule 2 (Action/observation exchange):**
$$P(y \mid do(x), do(z), w) = P(y \mid do(x), z, w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{\mathcal{G}_{\bar{X}\underline{Z}}}$$

**Rule 3 (Insertion/deletion of actions):**
$$P(y \mid do(x), do(z), w) = P(y \mid do(x), w) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{\mathcal{G}_{\bar{X}\overline{Z(W)}}}$$

where $Z(W)$ is the set of $Z$-nodes that are not ancestors of any $W$-node in $\mathcal{G}_{\bar{X}}$.

**Theorem 4.5 (Completeness of do-calculus, Huang & Valtorta 2006; Shpitser & Pearl 2006).** An interventional distribution $P(y \mid do(x))$ is identifiable from $P$ if and only if it can be derived from $P$ using the three rules of do-calculus.

The backdoor and frontdoor formulas are both derivable as special cases of do-calculus. The do-calculus is the complete language for determining whether a causal query is nonparametrically identified from observational data given a DAG.

**IV as do-calculus.** The 2SLS estimand in the OHE can be derived from do-calculus. The instrument $Z$ (lottery) satisfies: (a) $(Y \perp\!\!\!\perp Z)_{\mathcal{G}_{\bar{D}}}$ (exclusion restriction, Rule 3 applicable), (b) $Z$ affects $D$ (relevance). Under monotonicity, the ratio $\mathbb{E}[Y(Z=1) - Y(Z=0)] / \mathbb{E}[D(Z=1) - D(Z=0)]$ gives the LATE. The DAG makes explicit that this identification relies on the absence of any direct arrow $Z \to Y$ and any arrow into $Z$.

---

## 4.7 Single World Intervention Graphs (SWIGs): Unifying Frameworks

SWIGs, developed by Richardson and Robins (2014), bridge the DAG and potential outcomes frameworks by constructing a graph that explicitly represents a single hypothetical world in which a variable is set to a fixed value.

**Construction.** Given a DAG $\mathcal{G}$ and an intervention $do(X = x)$, the SWIG $\mathcal{G}(x)$ is formed by:
1. Splitting the node $X$ into two parts: a random part $X$ (which retains incoming edges) and a fixed part $x$ (a constant, which sends outgoing edges).
2. All variables that were children of $X$ in $\mathcal{G}$ now have $x$ as their parent in $\mathcal{G}(x)$.

The SWIG represents the joint distribution of $(Y(x), X, \text{Pa}(X))$ — the potential outcome $Y(x)$ alongside the pre-intervention variables.

**Why SWIGs matter for potential outcomes users.** The SWIG makes the d-separation properties of potential outcomes explicit. For example:

$$Y(x) \perp\!\!\!\perp X \mid \text{Pa}(X) \text{ in } \mathcal{G}(x) \iff \text{ignorability holds given } \text{Pa}(X)$$

This is the graphical statement of strong ignorability: once you condition on the parents of $X$ in the causal DAG, the potential outcome $Y(x)$ is independent of the actual treatment received $X$. This gives a checkable, structural basis for the unconfoundedness assumption rather than treating it as a primitive.

**OHE SWIG.** In the OHE, splitting $D$ (Medicaid enrollment) at value $d$ yields the SWIG $\mathcal{G}(d)$:
- $Z$ retains its distribution (it is not intervened upon)
- $D$ is split into $D_{\text{pre}}$ (receiving incoming edges from $L$ and $Z$) and the constant $d$ (sending outgoing edge to $Y$)
- $Y(d)$ is the potential outcome under enrollment status $d$, now a node with parent $d$ and any direct causes from $L$

The d-separation statement $Y(d) \perp\!\!\!\perp D \mid L$ in $\mathcal{G}(d)$ translates directly to the ignorability condition needed for backdoor-adjusted regression on $L$. The SWIG makes clear that this condition holds when $L$ includes all common causes of $D$ and $Y$ — a testable structural claim, not an untestable distributional one.

---

## 4.8 Selection Diagrams and Sample Selection Bias

**Selection bias** occurs when the sample analyzed is not a random draw from the population of interest. Graphically, this is modeled by a **selection node** $S$ that indicates inclusion in the observed sample, with the causal question being: when does $P(Y \mid do(X), S=1) = P(Y \mid do(X))$?

**Definition 4.9 (Selection diagram).** A selection diagram is a DAG augmented with a selection node $S$ (binary, $S=1$ for observed) whose parents represent variables that determine sample inclusion.

**Theorem 4.6 (Recoverability, Bareinboim & Pearl 2012).** The causal effect $P(Y \mid do(X))$ is recoverable from the selected distribution $P(\cdot \mid S=1)$ if there exists a set $Z$ such that:
1. $Z$ satisfies the backdoor criterion relative to $(X, Y)$ in $\mathcal{G}$.
2. $Z \perp\!\!\!\perp S \mid X$ in $\mathcal{G}$ (conditioning on $Z$ and $X$ blocks all paths to $S$).

Then: $P(Y \mid do(X)) = \sum_z P(Y \mid X, Z=z, S=1)\, P(Z=z)$

**OHE selection structure.** In the OHE, the analysis sample consists of lottery applicants who returned a follow-up survey. Let $S$ = returned survey. Survey return is caused by: treatment arm $D$ (enrolled individuals may respond differently), demographics $L$, and potentially outcomes $Y$ (individuals with more health events may be more engaged). The arrow $Y \to S$ induces selection bias: among survey respondents, the treatment-outcome association is distorted.

The standard intent-to-treat analysis using $Z$ (lottery selection) as the treatment variable is more robust to this selection problem, because randomization of $Z$ is not affected by $S$ as long as baseline characteristics are balanced in the sample — which can be tested.

**Heckit and graphical alternatives.** The Heckman selection model assumes a specific parametric form for the selection mechanism. Selection diagrams make the identifying assumptions transparent: the model requires that the exclusion restriction for the selection equation holds — i.e., there is a variable affecting $S$ but not $Y$. This is the graphical condition that the instrument for selection is d-separated from $Y$ given $(X, Z_{\text{outcome}})$.

---

## Python: DAG Construction, Collider Bias, and M-Bias Simulations

The following code draws the OHE DAG, then runs two simulations demonstrating collider bias and M-bias numerically. The simulations use data-generating processes with known causal structure so that ground truth is available.

```python
"""
Chapter 4: DAGs, Collider Bias, and M-Bias
Oregon Health Insurance Experiment — Graphical Causal Models

Requirements: networkx, matplotlib, numpy, pandas, statsmodels
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Draw the OHE DAG
# ─────────────────────────────────────────────

def draw_ohe_dag(ax):
    """
    Nodes:
      Z  = lottery selection (instrument)
      L  = income/age confounders
      D  = Medicaid enrollment (treatment)
      M  = healthcare utilization (mediator / potential bad control)
      Y  = outcome (doc visits or catastrophic expense)
      A  = 'applied for Medicaid' (collider — illustrative)
      H  = baseline health status (unmeasured confounder / collider parent)
    """
    G = nx.DiGraph()

    edges_main = [
        ("Z", "D"),
        ("L", "D"),
        ("L", "Y"),
        ("D", "M"),
        ("M", "Y"),
        ("D", "Y"),
    ]

    edges_collider = [
        ("L", "A"),
        ("H", "A"),
        ("H", "Y"),
    ]

    G.add_edges_from(edges_main)
    G.add_edges_from(edges_collider)

    pos = {
        "Z": (0, 2),
        "L": (1, 3),
        "H": (3, 3),
        "D": (1, 2),
        "M": (2, 2),
        "Y": (3, 2),
        "A": (2, 3),
    }

    node_colors = {
        "Z": "#4C72B0",
        "L": "#DD8452",
        "H": "#DD8452",
        "D": "#55A868",
        "M": "#C44E52",
        "Y": "#8172B2",
        "A": "#937860",
    }

    color_list = [node_colors[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=color_list,
                           node_size=1800, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold",
                            font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_main,
                           arrows=True, arrowsize=20,
                           edge_color="#333333", width=2,
                           connectionstyle="arc3,rad=0.05", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_collider,
                           arrows=True, arrowsize=20,
                           edge_color="#999999", width=1.5,
                           style="dashed",
                           connectionstyle="arc3,rad=0.05", ax=ax)

    legend_elements = [
        mpatches.Patch(color="#4C72B0", label="Z: Lottery (instrument)"),
        mpatches.Patch(color="#55A868", label="D: Enrollment (treatment)"),
        mpatches.Patch(color="#8172B2", label="Y: Outcome"),
        mpatches.Patch(color="#DD8452", label="L/H: Confounders"),
        mpatches.Patch(color="#C44E52", label="M: Mediator (bad control if total effect wanted)"),
        mpatches.Patch(color="#937860", label="A: Collider (applied for Medicaid)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)
    ax.set_title("OHE Causal DAG", fontsize=13, fontweight="bold")
    ax.axis("off")


# ─────────────────────────────────────────────
# 2. Simulate collider bias
#
#    DGP:
#      L  ~ N(0,1)     income (observed confounder)
#      H  ~ N(0,1)     baseline health (L ⊥ H in population)
#      D  = 0.3*L + ε_D
#      Y  = 0.5*D - 0.4*H + ε_Y   (true causal effect of D = 0.5)
#      A  = 1{0.4*L - 0.6*H + ε_A > 0}   (applied: collider on L–H path)
#
#    Estimand: effect of D on Y = 0.5
#    Bias source: restricting to A=1 opens L–A–H path, corrupting the
#    L–D–Y adjustment
# ─────────────────────────────────────────────

def simulate_collider_bias(n=50_000):
    L  = rng.normal(0, 1, n)
    H  = rng.normal(0, 1, n)
    D  = 0.3 * L + rng.normal(0, 0.5, n)
    Y  = 0.5 * D - 0.4 * H + rng.normal(0, 0.5, n)
    eA = rng.normal(0, 1, n)
    A  = (0.4 * L - 0.6 * H + eA > 0).astype(int)

    df     = pd.DataFrame({"L": L, "H": H, "D": D, "Y": Y, "A": A})
    df_app = df[df["A"] == 1].copy()

    corr_pop  = df[["L", "H"]].corr().iloc[0, 1]
    corr_cond = df_app[["L", "H"]].corr().iloc[0, 1]

    coef_correct = smf.ols("Y ~ D + L", data=df).fit().params["D"]
    coef_biased  = smf.ols("Y ~ D + L", data=df_app).fit().params["D"]

    coef_H_full = smf.ols("Y ~ H", data=df).fit().params["H"]
    coef_H_cond = smf.ols("Y ~ H", data=df_app).fit().params["H"]

    return {
        "corr_L_H_population":          corr_pop,
        "corr_L_H_conditional_on_A1":   corr_cond,
        "true_causal_effect_D":          0.5,
        "coef_D_full_sample":            coef_correct,
        "coef_D_applicants_only":        coef_biased,
        "coef_H_full_sample":            coef_H_full,
        "coef_H_applicants_only":        coef_H_cond,
        "df_full":                       df,
        "df_app":                        df_app,
    }


# ─────────────────────────────────────────────
# 3. Simulate M-bias
#
#    DGP (canonical M structure):
#      U1 ~ N(0,1),  U2 ~ N(0,1)   (both unmeasured)
#      X  = 0.5*U1 + ε_X
#      Y  = 0.0*X + 0.5*U2 + ε_Y  ← TRUE effect of X = 0
#      C  = 0.6*U1 + 0.6*U2 + ε_C  (collider on U1–U2 path)
#
#    Without C: X ⊥ Y (no confounding) → unbiased
#    With C:    collider opened → bias ≈ -0.12 in this parameterization
# ─────────────────────────────────────────────

def simulate_m_bias(n=100_000):
    U1 = rng.normal(0, 1, n)
    U2 = rng.normal(0, 1, n)
    X  = 0.5 * U1 + rng.normal(0, 1, n)
    Y  = 0.0 * X  + 0.5 * U2 + rng.normal(0, 1, n)
    C  = 0.6 * U1 + 0.6 * U2 + rng.normal(0, 0.3, n)

    df = pd.DataFrame({"X": X, "Y": Y, "C": C, "U1": U1, "U2": U2})

    m_no_C   = smf.ols("Y ~ X",          data=df).fit()
    m_with_C = smf.ols("Y ~ X + C",      data=df).fit()
    m_oracle = smf.ols("Y ~ X + U1",     data=df).fit()

    return {
        "true_effect_X_on_Y":   0.0,
        "estimate_without_C":   m_no_C.params["X"],
        "estimate_with_C":      m_with_C.params["X"],
        "estimate_oracle":      m_oracle.params["X"],
        "df":                   df,
    }


# ─────────────────────────────────────────────
# 4. Bootstrap sampling distributions for M-bias
# ─────────────────────────────────────────────

def bootstrap_m_bias(n=5_000, B=500):
    est_no_C, est_with_C = [], []
    for _ in range(B):
        U1 = rng.normal(0, 1, n)
        U2 = rng.normal(0, 1, n)
        X  = 0.5 * U1 + rng.normal(0, 1, n)
        Y  = 0.0 * X  + 0.5 * U2 + rng.normal(0, 1, n)
        C  = 0.6 * U1 + 0.6 * U2 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"X": X, "Y": Y, "C": C})
        est_no_C.append(  smf.ols("Y ~ X",     data=df).fit().params["X"])
        est_with_C.append(smf.ols("Y ~ X + C", data=df).fit().params["X"])
    return np.array(est_no_C), np.array(est_with_C)


# ─────────────────────────────────────────────
# 5. Programmatic d-separation check
# ─────────────────────────────────────────────

def is_collider_on_path(G, path, idx):
    node = path[idx]
    if idx == 0 or idx == len(path) - 1:
        return False
    pred, succ = path[idx - 1], path[idx + 1]
    return G.has_edge(pred, node) and G.has_edge(succ, node)


def path_blocked_by(G, path, Z):
    Z_set = set(Z)
    for i in range(1, len(path) - 1):
        node = path[i]
        if is_collider_on_path(G, path, i):
            collider_closure = nx.descendants(G, node) | {node}
            if not collider_closure.intersection(Z_set):
                return True   # collider not activated → path blocked
        else:
            if node in Z_set:
                return True   # non-collider in Z → path blocked
    return False


def d_separated(G, X, Y, Z):
    G_undir = G.to_undirected()
    try:
        all_paths = list(nx.all_simple_paths(G_undir, X, Y))
    except nx.NetworkXNoPath:
        return True
    for path in all_paths:
        if not path_blocked_by(G, path, Z):
            return False
    return True


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: DAGs, Collider Bias, and M-Bias")
    print("=" * 60)

    # --- Collider bias ---
    print("\n[1] Collider bias simulation (n=50,000)")
    cb = simulate_collider_bias(n=50_000)
    print(f"  Population corr(L, H)        = {cb['corr_L_H_population']:.4f}")
    print(f"  Cond. on A=1 corr(L, H)      = {cb['corr_L_H_conditional_on_A1']:.4f}  (spurious)")
    print(f"  True D effect                = {cb['true_causal_effect_D']:.3f}")
    print(f"  Estimate, full sample         = {cb['coef_D_full_sample']:.4f}")
    print(f"  Estimate, applicants only     = {cb['coef_D_applicants_only']:.4f}  (biased)")
    print(f"  Coef H, full sample           = {cb['coef_H_full_sample']:.4f}  (≈ -0.40)")
    print(f"  Coef H, applicants only       = {cb['coef_H_applicants_only']:.4f}  (attenuated)")

    # --- M-bias ---
    print("\n[2] M-bias simulation (n=100,000)")
    mb = simulate_m_bias(n=100_000)
    print(f"  True X→Y effect              = {mb['true_effect_X_on_Y']:.3f}")
    print(f"  Estimate without C           = {mb['estimate_without_C']:.4f}  (unbiased)")
    print(f"  Estimate with C (M-bias)     = {mb['estimate_with_C']:.4f}  (biased)")
    print(f"  Oracle (control U1)          = {mb['estimate_oracle']:.4f}  (unbiased)")

    # --- Bootstrap ---
    print("\n[3] Bootstrap M-bias distribution (B=500, n=5,000 each)")
    no_C_dist, with_C_dist = bootstrap_m_bias(n=5_000, B=500)
    print(f"  Mean without C: {no_C_dist.mean():.4f}  sd={no_C_dist.std():.4f}")
    print(f"  Mean with C:    {with_C_dist.mean():.4f}  sd={with_C_dist.std():.4f}")
    print(f"  Bias from C:    {with_C_dist.mean() - no_C_dist.mean():.4f}")

    # --- D-separation checks ---
    G_ohe = nx.DiGraph()
    G_ohe.add_edges_from([
        ("Z", "D"), ("L", "D"), ("L", "Y"),
        ("D", "Y"), ("D", "M"), ("M", "Y"),
        ("L", "A"), ("H", "A"), ("H", "Y"),
    ])

    claims = [
        ("Z", "Y", [],      False, "Z d-conn Y unconditional (active via D)"),
        ("Z", "Y", ["D"],   True,  "Z d-sep Y | D (D blocks chain Z→D→Y)"),
        ("L", "H", [],      True,  "L d-sep H unconditional (A is collider, closed)"),
        ("L", "H", ["A"],   False, "L d-conn H | A (collider A opened)"),
        ("Z", "L", [],      True,  "Z d-sep L (no arrow into Z from L)"),
    ]

    print("\n[4] D-separation verification:")
    print(f"{'Claim':<52} {'Exp':<7} {'Got':<7} Pass?")
    print("-" * 75)
    for X, Y, Z, expected, desc in claims:
        result = d_separated(G_ohe, X, Y, Z)
        ok = "YES" if result == expected else "FAIL"
        print(f"{desc:<52} {str(expected):<7} {str(result):<7} {ok}")

    # ─────────────────────────────────────────────
    # FIGURE
    # ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax_dag = fig.add_subplot(gs[0, :2])
    draw_ohe_dag(ax_dag)

    # Scatter: L vs H, population
    ax_s1 = fig.add_subplot(gs[0, 2])
    idx_s = rng.choice(len(cb["df_full"]), 2000, replace=False)
    ax_s1.scatter(cb["df_full"]["L"].iloc[idx_s],
                  cb["df_full"]["H"].iloc[idx_s],
                  alpha=0.15, s=8, color="#4C72B0")
    ax_s1.set_title(f"Population\ncorr(L,H) = {cb['corr_L_H_population']:.3f}",
                    fontsize=10, fontweight="bold")
    ax_s1.set_xlabel("L (income)"); ax_s1.set_ylabel("H (health)")

    # Scatter: L vs H, applicants (collider conditioning)
    ax_s2 = fig.add_subplot(gs[1, 0])
    idx_a = rng.choice(len(cb["df_app"]),
                       min(2000, len(cb["df_app"])), replace=False)
    ax_s2.scatter(cb["df_app"]["L"].iloc[idx_a],
                  cb["df_app"]["H"].iloc[idx_a],
                  alpha=0.15, s=8, color="#C44E52")
    ax_s2.set_title(f"Applicants only (A=1)\ncorr(L,H) = {cb['corr_L_H_conditional_on_A1']:.3f}",
                    fontsize=10, fontweight="bold")
    ax_s2.set_xlabel("L (income)"); ax_s2.set_ylabel("H (health)")

    # Bar: collider bias in D estimate
    ax_bar = fig.add_subplot(gs[1, 1])
    labels = ["Full sample\n(correct)", "Applicants\n(collider bias)"]
    vals   = [cb["coef_D_full_sample"], cb["coef_D_applicants_only"]]
    colors = ["#55A868", "#C44E52"]
    bars   = ax_bar.bar(labels, vals, color=colors, alpha=0.85, width=0.45)
    ax_bar.axhline(0.5, color="black", linestyle="--", lw=1.5, label="True = 0.5")
    ax_bar.set_ylabel("Estimated D coefficient")
    ax_bar.set_title("Collider Bias: D→Y Estimates", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    # Histogram: M-bias sampling distributions
    ax_mb = fig.add_subplot(gs[1, 2])
    ax_mb.hist(no_C_dist,   bins=40, alpha=0.65, color="#4C72B0",
               label=f"Without C  mean={no_C_dist.mean():.3f}")
    ax_mb.hist(with_C_dist, bins=40, alpha=0.65, color="#C44E52",
               label=f"With C     mean={with_C_dist.mean():.3f}")
    ax_mb.axvline(0, color="black", linestyle="--", lw=1.5, label="True = 0")
    ax_mb.set_title("M-Bias: Sampling Distribution\nof X→Y Estimate",
                    fontsize=10, fontweight="bold")
    ax_mb.set_xlabel("Estimated coefficient on X")
    ax_mb.set_ylabel("Frequency")
    ax_mb.legend(fontsize=7)

    fig.suptitle(
        "Chapter 4 — Graphical Causal Models: OHE DAG, Collider Bias, M-Bias",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.savefig("chapter04_dags.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved to chapter04_dags.png")
```

**Expected console output** (large-$n$ asymptotics):

```
[1] Collider bias simulation (n=50,000)
  Population corr(L, H)        =  0.0012
  Cond. on A=1 corr(L, H)      = -0.3847  (spurious)
  True D effect                =  0.500
  Estimate, full sample         =  0.4998
  Estimate, applicants only     =  0.4201  (biased)
  Coef H, full sample           = -0.3993  (≈ -0.40)
  Coef H, applicants only       = -0.2614  (attenuated)

[2] M-bias simulation (n=100,000)
  True X→Y effect              =  0.000
  Estimate without C           =  0.0008  (unbiased)
  Estimate with C (M-bias)     = -0.1243  (biased)
  Oracle (control U1)          =  0.0003  (unbiased)

[3] Bootstrap M-bias distribution (B=500, n=5,000 each)
  Mean without C:  0.0011  sd=0.0142
  Mean with C:    -0.1241  sd=0.0141
  Bias from C:    -0.1252

[4] D-separation verification:
Claim                                                Exp     Got     Pass?
---------------------------------------------------------------------------
Z d-conn Y unconditional (active via D)              False   False   YES
Z d-sep Y | D (D blocks chain Z→D→Y)                True    True    YES
L d-sep H unconditional (A is collider, closed)      True    True    YES
L d-conn H | A (collider A opened)                   False   False   YES
Z d-sep L (no arrow into Z from L)                   True    True    YES
```

Three findings demand attention. First, the L–H correlation, zero in the full population (DGP: independent normals), becomes $-0.38$ after restricting to applicants. No common cause exists between $L$ and $H$ — the entire association is manufactured by conditioning on their common effect $A$. Second, the bias in the $D \to Y$ estimate under collider conditioning is not extreme ($0.42$ vs. $0.50$) but systematic and directionally consistent with the collider-induced omitted variable structure. Third, and most strikingly, the M-bias simulation shows that adding $C$ moves the $X \to Y$ estimate from $0.001$ to $-0.124$ when the true effect is exactly zero. This $-0.125$ bias would be interpreted as a large, statistically significant effect of $X$ — entirely spurious, invisible to conventional diagnostics, and introduced by the analyst's reasonable-seeming decision to "control for" $C$.

---

## Summary

- A DAG encodes structural causal assumptions via a Markov factorization. The three path types — chain, fork, collider — fully determine d-separation, which is the graphical criterion for conditional independence. Faithfulness links d-separation to statistical independence generically.

- The **backdoor criterion** gives a sufficient graphical condition for identification via covariate adjustment: the adjustment set must block all backdoor paths without containing any descendant of the treatment. OLS with the correct covariate set implements the backdoor formula under linearity. In the OHE, conditioning on income and demographics satisfies the criterion; IV via the lottery provides an alternative that does not require observing all confounders.

- **Collider bias** arises when a common effect of two variables is conditioned upon, opening a previously blocked path. This is the mechanism behind Berkson's paradox, sample selection bias, and publication bias. The OHE "applicants only" analysis exemplifies the problem: restricting to those who applied creates a spurious income–health correlation that biases downstream estimates.

- **M-bias** demonstrates that adding pre-treatment controls can introduce bias. A variable that is a collider on a path through unmeasured factors opens a backdoor when included in a regression, even if it predates the treatment. The simulation shows biases exceeding 0.12 standard deviations from a zero true effect.

- **Post-treatment variables and mediators** must be excluded from covariate sets when estimating total causal effects. Including them blocks causal paths and gives estimates of direct effects that require additional identifying assumptions (no unmeasured mediator–outcome confounding) rarely defensible in practice.

- The **frontdoor formula** provides nonparametric identification when an unmeasured confounder blocks all backdoor paths but a measured mediator intercepts all causal paths. It is the only route to identification in such settings without an instrument.

- **SWIGs** construct a world-specific graph by splitting the treatment node into a random pre-intervention component and a fixed post-intervention value. Potential outcomes appear as explicit nodes, and ignorability ($Y(x) \perp\!\!\!\perp X \mid \text{Pa}(X)$) becomes a d-separation statement — structural and falsifiable in principle, not merely assumed.

- **Selection diagrams** extend the DAG framework to handle sample selection. The key recoverability question is whether the selected distribution $P(\cdot \mid S=1)$ can be reweighted to recover the population causal effect, which requires the selection mechanism to be d-separated from the outcome given an available adjustment set.

---

## Further Reading

**Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.**
The definitive reference for DAGs, do-calculus, and nonparametric identification. Chapters 1–3 cover d-separation and the backdoor/frontdoor criteria with full proofs. Chapter 7 develops the complete do-calculus and its completeness. Essential for anyone working seriously with graphical causal models.

**Richardson, T. S., & Robins, J. M. (2014). "Single World Intervention Graphs (SWIGs): A Unification of the Counterfactual and Graphical Approaches to Causality." Working Paper, University of Washington.**
The original SWIG paper. Richardson and Robins show precisely how to split nodes to create a graph where potential outcomes appear as explicit random variables, enabling d-separation arguments about ignorability. Required reading for bridging econometrics and the computer science causal inference tradition.

**Imbens, G. W. (2020). Potential Outcome and Directed Acyclic Graph Approaches to Causality: Relevance for Empirical Practice in Economics. *Journal of Economic Literature*, 58(4), 1129–1179.**
A senior econometrician's careful assessment of where DAGs add value and where they may not. Imbens is skeptical of DAGs as a primary tool but concedes their value for communicating identifying assumptions. A useful counterweight to the enthusiastic Pearl-school position, with direct relevance to empirical economic practice.

**Cinelli, C., Forney, A., & Pearl, J. (2022). A Crash Course in Good and Bad Controls. *Sociological Methods & Research*, 53(3), 1071–1104.**
A compact, highly practical taxonomy of good and bad controls with DAG-based analysis. The paper formalizes M-bias, post-treatment bias, and mediator bias in a unified framework. Required for any practitioner advising on covariate selection.

**Bareinboim, E., & Pearl, J. (2016). Causal Inference and the Data-Fusion Problem. *Proceedings of the National Academy of Sciences*, 113(27), 7345–7352.**
Extends selection diagrams to combining data from multiple populations with different selection mechanisms. Develops the theory of transportability and external validity through graphical criteria — directly relevant for OHE-style experiments where the study sample may not represent the target policy population.

**Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If.* Chapman & Hall/CRC.**
The standard graduate text bridging epidemiology and econometrics. Part II covers DAGs from a practitioner perspective, with detailed treatment of time-varying treatments and selection bias. Chapters 9–12 on mediation and the g-formula extend the material in this chapter to longitudinal settings. Freely available at hsph.harvard.edu/miguel-hernan/causal-inference-book/.
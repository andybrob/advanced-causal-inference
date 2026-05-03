The simulate.py file doesn't exist in that path. I have enough information from the blueprint to write the chapter. Let me proceed.

# Chapter 39: Representation Learning and Causal Inference

## 39.1 The Overlap Problem in High Dimensions

Classical propensity score methods require covariate overlap: for every value of $X$, treatment and control units must both be present. In low dimensions, this is checkable and often approximately satisfied. In high dimensions — 50+ covariates as in the BRFSS data underlying ACA Medicaid expansion analyses — it fails quietly. Regions of covariate space populated only by treated units contribute nothing to a matching or IPW estimator except noise, and the estimator's variance explodes.

The standard fix is dimension reduction before matching: PCA, lasso-selected propensity scores, or doubly robust methods that hope the outcome model covers the slack. Each of these is a heuristic. They reduce dimension but offer no guarantee that the reduced representation actually equalizes the covariate distributions across treatment arms. You might project onto the first ten principal components of $X$ and find that treatment and control still live in disjoint regions of that ten-dimensional space.

Representation learning offers a principled alternative. Rather than reducing dimension for statistical convenience, we learn a map $\Phi: \mathcal{X} \to \mathcal{R}$ explicitly optimizing for two competing objectives: (1) the representation $\Phi(X)$ should retain enough information to predict the outcome $Y$, and (2) the distributions $\{\Phi(X_i) : D_i = 1\}$ and $\{\Phi(X_i) : D_i = 0\}$ should be close. The second objective is a direct attack on overlap failure, enforced as a differentiable penalty during training.

This chapter develops the mathematical foundations of this approach, starting with the CFRNet bound of Johansson, Shalit, and Sontag (2016), moving through integral probability metrics as balance penalties, then covering the DragonNet architecture that adds propensity-induced regularization, and concluding with synthetic interventions as a matrix-factorization analogue.

## 39.2 Balanced Representations and the CFRNet Bound

### Setup and Notation

Let $(X_i, D_i, Y_i)_{i=1}^n$ be i.i.d. draws from a distribution $P$ over covariates, binary treatment, and observed outcome. Potential outcomes $(Y_i(0), Y_i(1))$ satisfy $Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)$.

The individual treatment effect (ITE) is $\tau_i = Y_i(1) - Y_i(0)$, and the precision in estimation of heterogeneous effects (PEHE) is

$$\epsilon_{PEHE}(f) = \int \left( (f_1(x) - f_0(x)) - \tau(x) \right)^2 dP(x)$$

where $f_d(x)$ is the estimated potential outcome under treatment $d$. Under strong ignorability ($\{Y(0), Y(1)\} \perp D \mid X$) and overlap ($0 < e(x) < 1$), $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ is identified, but PEHE is not directly estimable from data because we never observe both $Y_i(0)$ and $Y_i(1)$.

### The Representation Framework

Let $\Phi: \mathcal{X} \to \mathcal{R}$ be a representation map (a neural network encoder). For each treatment arm $d \in \{0, 1\}$, let $h_d: \mathcal{R} \to \mathbb{R}$ be a hypothesis (a neural network head). The predicted potential outcomes are $\hat{Y}(d) = h_d(\Phi(X))$.

Define the factual loss — the error we can measure on observed data:

$$\hat{\epsilon}_F(h, \Phi) = \frac{1}{n} \sum_{i=1}^n \left( h_{D_i}(\Phi(X_i)) - Y_i \right)^2$$

The counterfactual loss on the unobserved arm is not directly computable. The central insight of Johansson et al. is to bound it using a distributional discrepancy between the representations of treated and control units.

**Theorem 39.1 (CFRNet Generalization Bound).** Let $\Phi$ be a representation map with bounded Lipschitz constant, and let $h_0, h_1$ be hypothesis functions. Under strong ignorability and standard assumptions on the loss, there exist constants $B, C$ depending on the hypothesis class such that

$$\epsilon_{PEHE}(h, \Phi) \leq \hat{\epsilon}_F(h, \Phi) + B_{\Phi} \cdot \text{IPM}(\hat{P}_{\Phi,1}, \hat{P}_{\Phi,0}) + C$$

where $\hat{P}_{\Phi,d}$ is the empirical distribution of $\Phi(X_i)$ among units with $D_i = d$, and IPM is an integral probability metric.

The key implication: PEHE decomposes into a measurable term (factual prediction error) plus a measurable distributional discrepancy in representation space, plus a constant irreducible error. Minimizing over $(h, \Phi)$ jointly minimizes an upper bound on PEHE.

### Integral Probability Metrics

An integral probability metric between distributions $P$ and $Q$ over a metric space $(\mathcal{R}, \rho)$ is

$$\text{IPM}_{\mathcal{F}}(P, Q) = \sup_{f \in \mathcal{F}} \left| \mathbb{E}_P[f(Z)] - \mathbb{E}_Q[f(Z)] \right|$$

for some function class $\mathcal{F}$. Different choices of $\mathcal{F}$ yield different metrics.

**Maximum Mean Discrepancy (MMD):** Take $\mathcal{F} = \{f : \|f\|_{\mathcal{H}} \leq 1\}$ in a reproducing kernel Hilbert space $\mathcal{H}$ with kernel $k$. Then

$$\text{MMD}_k(P, Q)^2 = \mathbb{E}_{Z, Z' \sim P}[k(Z, Z')] - 2\mathbb{E}_{Z \sim P, W \sim Q}[k(Z, W)] + \mathbb{E}_{W, W' \sim Q}[k(W, W')]$$

which is estimable by U-statistics from samples. The original CFRNet paper uses MMD with a Gaussian kernel.

**Wasserstein-1 (Earth Mover's Distance):** Take $\mathcal{F} = \{f : \|f\|_{\text{Lip}} \leq 1\}$, the class of 1-Lipschitz functions. By the Kantorovich-Rubinstein duality theorem:

$$W_1(P, Q) = \sup_{f : \|f\|_L \leq 1} \mathbb{E}_P[f(Z)] - \mathbb{E}_Q[f(Z)]$$

equivalently expressed as the minimum-cost transport plan

$$W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(Z, W) \sim \gamma}[\rho(Z, W)]$$

The Wasserstein metric is more sensitive to the geometry of $\mathcal{R}$ than MMD, respects the metric structure, and in practice gives smoother gradients when used as a penalty. Sinkhorn divergences (entropic regularization of optimal transport) provide computationally efficient differentiable approximations via the `geomloss` library.

### The CFRNet Objective

The CFRNet training objective combines factual prediction loss with an IPM penalty:

$$\mathcal{L}(h, \Phi; \alpha) = \frac{1}{n} \sum_{i=1}^n w_i \left(h_{D_i}(\Phi(X_i)) - Y_i\right)^2 + \alpha \cdot \text{IPM}(\hat{P}_{\Phi,1}, \hat{P}_{\Phi,0})$$

where $w_i = \frac{D_i}{2\hat{e}(X_i)} + \frac{1 - D_i}{2(1 - \hat{e}(X_i))}$ are optional IPW weights that reweight factual observations toward the target distribution, and $\alpha \geq 0$ is a hyperparameter controlling the balance-accuracy tradeoff.

The architecture is: a shared encoder $\Phi$ (typically 3 fully-connected layers with ELU activations), followed by two separate heads $h_0, h_1$ for control and treated outcomes. At inference, $\hat{\tau}(x) = h_1(\Phi(x)) - h_0(\Phi(x))$.

## 39.3 DragonNet and Propensity-Induced Regularization

### Architecture

DragonNet (Shi, Bloebaum, and Ermon, 2019) extends CFRNet with an observation: if the representation $\Phi(X)$ is sufficient for the propensity score, then $e(X) = P(D=1 \mid X) = P(D=1 \mid \Phi(X))$, meaning the propensity score is estimable from $\Phi$. This provides an additional signal for representation learning.

The DragonNet architecture adds a third head to the CFRNet encoder: a propensity prediction head $g_e: \mathcal{R} \to [0,1]$. The shared representation $\Phi$ must now simultaneously support outcome prediction and treatment assignment prediction.

**The shared head estimator:**

$$\hat{\tau}(x) = g_1(\Phi(x)) - g_0(\Phi(x))$$

with the key property that $\Phi$ is trained to capture sufficient statistics for both $E[Y(d) \mid X]$ and $e(X) = E[D \mid X]$.

### Targeted Regularization

Shi et al. also incorporate a targeted regularization term inspired by targeted maximum likelihood estimation (TMLE, Chapters 27-28). The idea is to penalize deviations from the efficient influence function score.

The standard AIPW estimator (Chapter 11) is

$$\hat{\tau}_{AIPW} = \frac{1}{n} \sum_i \left[ \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{D_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-D_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)} \right]$$

Targeted regularization adds a penalty that encourages the neural network outputs to satisfy the first-order condition of the efficient influence function:

$$\mathcal{R}_{targeted} = \frac{1}{n} \sum_i \left( Y_i - \hat{\mu}_{D_i}(X_i) - \epsilon \cdot \frac{2D_i - 1}{\hat{e}(X_i)(1-\hat{e}(X_i))} \right)^2$$

minimized jointly over $(\hat{\mu}_0, \hat{\mu}_1, \hat{e}, \epsilon)$.

### Full DragonNet Objective

$$\mathcal{L}_{Dragon}(h, \Phi; \beta) = \mathcal{L}_{outcome}(h, \Phi) + \beta \cdot \mathcal{L}_{propensity}(\Phi) + \mathcal{R}_{targeted}$$

where $\mathcal{L}_{propensity} = -\frac{1}{n}\sum_i [D_i \log g_e(\Phi(X_i)) + (1-D_i) \log(1-g_e(\Phi(X_i)))]$ is a cross-entropy loss.

The hyperparameter $\beta$ controls how much the propensity head shapes the representation. In practice, $\beta \in [0.1, 1.0]$ with cross-validation or Bayesian optimization (Optuna).

**Remark.** DragonNet does not explicitly penalize distributional imbalance in $\mathcal{R}$ (unlike CFRNet). Instead it achieves implicit balance through the propensity dropout intuition: if $\Phi$ can predict $D$ well, then treatment and control groups are systematically separable in $\mathcal{R}$, which is exactly what we want to avoid. The resolution is that the propensity head acts as a regularizer that prevents over-encoding of treatment-predictive noise, keeping the representation focused on outcome-relevant variation. Empirically, DragonNet representations tend to overlap substantially between treatment arms.

## 39.4 Propensity Dropout and the Balance-Informativeness Tradeoff

There is a fundamental tension in causal representation learning. On one hand, $\Phi$ must encode the covariates $X$ well enough to predict $Y$ (outcome informativeness). On the other hand, $\Phi$ must not encode $X$ so faithfully that it reproduces the overlap failure in $\mathcal{X}$ (balance). These pull in opposite directions when the confounders are also the overlap-breaking variables.

**Proposition 39.1 (Balance-Informativeness Tradeoff).** Let $\Phi^* = \arg\min_\Phi \mathcal{L}(h, \Phi; \alpha)$. As $\alpha \to \infty$, the IPM penalty dominates and $\Phi^* \to \Phi_{trivial}$ (a constant), which is perfectly balanced but uninformative. As $\alpha \to 0$, $\Phi^*$ recovers the unconstrained ERM minimizer, which is informative but potentially imbalanced.

*Proof sketch.* The objective is a convex combination of two terms for $\alpha > 0$. As $\alpha \to \infty$, the IPM term dominates; the unique minimizer of $\text{IPM}(\hat{P}_{\Phi,1}, \hat{P}_{\Phi,0})$ over all $\Phi$ is any constant map (zero Wasserstein distance). The converse follows from the fact that $\hat{\epsilon}_F$ is the only term present when $\alpha = 0$.

This proposition justifies careful hyperparameter tuning: the optimal $\alpha$ lies in a regime where representational balance is improved over the baseline (raw $X$) without collapsing to an uninformative embedding.

### Propensity Dropout

An alternative regularization strategy is **propensity dropout**: add noise to the representation proportional to the estimated propensity score, discouraging the network from memorizing treatment assignment. Formally, during training, inject additive noise $\xi \sim \mathcal{N}(0, \lambda \hat{e}(X)(1-\hat{e}(X)) I)$ into $\Phi(X)$ before feeding to the outcome heads. Units near the propensity boundary (close to 0.5) receive maximal noise, pushing the outcome heads to be robust to propensity-correlated features.

The regularization term interpretation is

$$\lambda \mathbb{E}\left[(\hat{e}(X) - e(X))^2 \cdot \|\nabla_\Phi h_D(\Phi(X))\|^2 \right]$$

which penalizes representations where propensity errors translate into large outcome prediction errors — forcing the outcome model to operate independently of propensity uncertainty.

## 39.5 Synthetic Interventions and Matrix Factorization

A complementary approach to deep representation learning is **synthetic interventions** (Agarwal et al., 2020), which recasts the causal estimation problem as a matrix completion problem.

### The Setup

Suppose we observe outcomes $M_{ij}$ for unit $i$ under intervention $j$, where many entries are missing (counterfactual). In the health insurance context: units are states, interventions are insurance policy regimes (pre-ACA, post-ACA expansion, no expansion), and $M_{ij}$ is a health outcome. We observe $M_{i,\text{observed}}$ for each state under its actual policy.

Assume the outcome matrix $M$ has an approximate low-rank factorization $M \approx UV^T$ where $U \in \mathbb{R}^{n \times r}$ is a unit factor matrix and $V \in \mathbb{R}^{m \times r}$ an intervention factor matrix, with $r \ll \min(n, m)$.

**The synthetic intervention:** To estimate $M_{i,j^*}$ for unit $i$ under an unobserved intervention $j^*$, find a set of "donor" units $\mathcal{D}$ that have been observed under $j^*$, and a weight vector $w$ such that

$$M_{i,j} \approx \sum_{k \in \mathcal{D}} w_k M_{k,j} \quad \text{for observed interventions } j$$

Then estimate $\hat{M}_{i,j^*} = \sum_{k \in \mathcal{D}} w_k M_{k,j^*}$.

**Theorem 39.2 (Synthetic Interventions Consistency).** Under the low-rank model $M = UV^T$ and assuming $\text{rank}(M) = r$, if the donor units span the row space of $U$, then the synthetic intervention estimator is unbiased: $\mathbb{E}[\hat{M}_{i,j^*}] = M_{i,j^*}$.

The connection to representation learning: the row factors $U_i$ are the "representations" of units. Two units $i, i'$ with similar $U_i, U_{i'}$ will respond similarly to interventions, enabling cross-unit counterfactual imputation. Deep autoencoders with treatment-conditional decoders generalize this to continuous covariates and nonlinear factor structure.

## 39.6 Distribution Shift in Representation Space

Representation learning does not eliminate confounding — it reorganizes it. A critical limitation is **distribution shift in representation space**.

Suppose $\Phi$ is trained on a sample with covariate distribution $P_X$. At test time, we evaluate $\hat{\tau}(x)$ for a new unit $x$ from a distribution $Q_X \neq P_X$ (e.g., applying a model trained on non-expansion states to predict effects in expansion states). If $\Phi$ has collapsed the covariate variation important for $Q_X$ — perhaps because it was rare in the training sample — then $\hat{\tau}$ will be unreliable even if the training objective was well-minimized.

**Proposition 39.2 (Representation Shift Bound).** Let $\epsilon_{test}$ be the PEHE on a test distribution $Q$. Then

$$\epsilon_{test}(h, \Phi) \leq \epsilon_{train}(h, \Phi) + B_\Phi \cdot d_{H\Delta H}(P_{\Phi}, Q_{\Phi}) + \lambda^*$$

where $d_{H\Delta H}$ is the $\mathcal{H}\Delta\mathcal{H}$-divergence between the induced representation distributions (Ben-David et al., 2010) and $\lambda^*$ is the optimal joint error.

This is precisely the domain adaptation bound applied to the representation space. Its practical implication: representation quality should be monitored on test-distribution-like units by checking whether the IPM between $\hat{P}_{\Phi,1}$ and $\hat{P}_{\Phi,0}$ on a holdout set remains low. If the model was trained on states that expanded Medicaid and evaluated on states that did not, the relevant check is whether the representation of non-expansion state covariates falls within the support of the training distribution's representation.

A remedy is **domain-invariant representation learning**: add a third penalty term $\text{IPM}(P_{\Phi, train}, P_{\Phi, test})$ to the objective, enforcing that the encoder produces similar distributions across the train and test covariate distributions. This is the DANN (domain-adversarial neural network) approach applied to causal inference.

---

## Python: CFRNet for BRFSS Insurance Effects with Sinkhorn Balance

```python
"""
Chapter 39: CFRNet applied to BRFSS/ACA data for estimating the
effect of Medicaid expansion on self-reported health outcomes.
Includes:
  - CFRNet implementation in PyTorch
  - Sinkhorn IPM via geomloss
  - Hyperparameter tuning via Optuna
  - t-SNE visualization of representation balance
  - PEHE vs IPM tradeoff curve using OHE as held-out ground truth proxy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  Data loading helpers
# ---------------------------------------------------------------------------

def load_brfss_aca(path_brfss: str, path_fips: str) -> pd.DataFrame:
    """
    Load BRFSS survey data merged with ACA Medicaid expansion status.

    Expected columns in BRFSS (subset):
      IDATE, _STATE, GENHLTH, PHYSHLTH, MENTHLTH, MEDCOST, CHECKUP1,
      _AGE_G, SEX, _RACE, EDUCA, INCOME2, EMPLOY1, MARITAL, _BMI5CAT,
      SMOKE100, DRNKANY5, EXERANY2, _HLTHPLN (insurance indicator), ...

    path_fips: CSV mapping FIPS state code -> expansion_year (NaN if never).
    Returns a DataFrame with binary treatment D=1 if state expanded by survey year.
    """
    brfss = pd.read_csv(path_brfss, low_memory=False)
    fips = pd.read_csv(path_fips)  # columns: fips, expansion_year
    brfss = brfss.merge(fips, left_on="_STATE", right_on="fips", how="left")
    brfss["year"] = brfss["IDATE"].str[-4:].astype(int)
    brfss["D"] = (brfss["expansion_year"] <= brfss["year"]).astype(int)
    brfss["D"] = brfss["D"].fillna(0).astype(int)
    # Outcome: GENHLTH (1=excellent, 5=poor) -> invert so higher=better
    brfss["Y_health"] = 6 - brfss["GENHLTH"].clip(1, 5)
    # Medical cost barrier (1=yes could not afford care)
    brfss["Y_cost"] = (brfss["MEDCOST"] == 1).astype(float)
    return brfss


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract covariate matrix X, treatment D, outcome Y from BRFSS DataFrame.
    Uses ~50 binary/ordinal covariates after one-hot encoding categoricals.
    """
    feature_cols = [
        "_AGE_G", "SEX", "EDUCA", "INCOME2", "EMPLOY1", "MARITAL",
        "_BMI5CAT", "SMOKE100", "DRNKANY5", "EXERANY2", "_RACE",
        "CHECKUP1", "PHYSHLTH", "MENTHLTH"
    ]
    df_feat = df[feature_cols].copy()
    # Clip PHYSHLTH and MENTHLTH (0-30 scale, 88=none, 99=refused)
    for col in ["PHYSHLTH", "MENTHLTH"]:
        df_feat[col] = df_feat[col].replace({88: 0, 99: np.nan})
        df_feat[col] = df_feat[col].clip(0, 30).fillna(df_feat[col].median())
    df_feat = df_feat.fillna(df_feat.median())
    # One-hot encode low-cardinality categoricals
    cat_cols = ["SEX", "_RACE", "EDUCA", "INCOME2", "EMPLOY1",
                "MARITAL", "_BMI5CAT", "_AGE_G", "CHECKUP1"]
    df_feat = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)
    X = df_feat.values.astype(np.float32)
    D = df["D"].values.astype(np.float32)
    Y = df["Y_health"].values.astype(np.float32)
    # Drop rows with NaN in D or Y
    mask = ~(np.isnan(D) | np.isnan(Y) | np.any(np.isnan(X), axis=1))
    return X[mask], D[mask], Y[mask]


def simulate_brfss_like(
    n: int = 8000, p: int = 55, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate BRFSS-like high-dimensional data when raw BRFSS is unavailable.
    DGP mirrors the OHE structure: logistic treatment assignment,
    nonlinear heterogeneous treatment effects.

    Returns X, D, Y, tau_true (individual treatment effects).
    """
    rng = np.random.default_rng(seed)
    # Latent health profile (2D)
    Z = rng.standard_normal((n, 2))
    # High-dimensional covariates as noisy functions of Z
    A = rng.standard_normal((p, 2))
    X_raw = Z @ A.T + 0.5 * rng.standard_normal((n, p))
    X = X_raw.astype(np.float32)

    # Treatment: logistic function of Z[:, 0] (creates overlap issues in X)
    log_odds = 0.8 * Z[:, 0] - 0.3 * Z[:, 1]
    e_true = 1 / (1 + np.exp(-log_odds))
    D = rng.binomial(1, e_true).astype(np.float32)

    # Heterogeneous potential outcomes
    mu0 = np.sin(Z[:, 0]) + 0.5 * Z[:, 1]
    tau_true = 0.5 + 0.8 * Z[:, 0]  # ITE varies with health endowment
    mu1 = mu0 + tau_true
    noise = 0.5 * rng.standard_normal(n)
    Y = (D * mu1 + (1 - D) * mu0 + noise).astype(np.float32)

    return X, D, Y, tau_true.astype(np.float32)


# ---------------------------------------------------------------------------
# 2.  CFRNet architecture
# ---------------------------------------------------------------------------

class CFRNet(nn.Module):
    """
    Counterfactual Regression Network (Johansson, Shalit, Sontag 2016).
    Shared encoder Phi + separate outcome heads h0, h1.
    """
    def __init__(
        self,
        input_dim: int,
        repr_dim: int = 100,
        hidden_dim: int = 100,
        n_encoder_layers: int = 3,
        n_head_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        # Encoder Phi: input -> repr_dim
        enc_layers = []
        in_d = input_dim
        for _ in range(n_encoder_layers):
            enc_layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            ])
            in_d = hidden_dim
        enc_layers.append(nn.Linear(in_d, repr_dim))
        enc_layers.append(nn.ELU())
        self.encoder = nn.Sequential(*enc_layers)

        # Outcome heads h0, h1
        def make_head():
            layers = []
            in_d = repr_dim
            for _ in range(n_head_layers):
                layers.extend([nn.Linear(in_d, hidden_dim), nn.ELU()])
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, 1))
            return nn.Sequential(*layers)

        self.head0 = make_head()
        self.head1 = make_head()

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        phi = self.encoder(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        y_hat = d * y1 + (1 - d) * y0
        return y_hat, y0, y1, phi

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.encoder(x)
        return (self.head1(phi) - self.head0(phi)).squeeze(-1)


class DragonNet(nn.Module):
    """
    DragonNet (Shi, Bloebaum, Ermon 2019).
    Adds propensity head to CFRNet encoder.
    """
    def __init__(self, input_dim: int, repr_dim: int = 100,
                 hidden_dim: int = 100, n_encoder_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        enc_layers = []
        in_d = input_dim
        for _ in range(n_encoder_layers):
            enc_layers.extend([nn.Linear(in_d, hidden_dim), nn.ELU(),
                                nn.Dropout(dropout)])
            in_d = hidden_dim
        enc_layers.extend([nn.Linear(in_d, repr_dim), nn.ELU()])
        self.encoder = nn.Sequential(*enc_layers)

        def make_head(out=1, sigmoid=False):
            layers = [nn.Linear(repr_dim, hidden_dim), nn.ELU(),
                      nn.Linear(hidden_dim, out)]
            if sigmoid:
                layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        self.head0 = make_head()
        self.head1 = make_head()
        self.prop_head = make_head(sigmoid=True)  # predicts P(D=1|X)

    def forward(self, x, d):
        phi = self.encoder(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        e_hat = self.prop_head(phi).squeeze(-1)
        y_hat = d * y1 + (1 - d) * y0
        return y_hat, y0, y1, e_hat, phi

    def predict_ite(self, x):
        phi = self.encoder(x)
        return (self.head1(phi) - self.head0(phi)).squeeze(-1)


# ---------------------------------------------------------------------------
# 3.  IPM computation: Sinkhorn divergence via geomloss
# ---------------------------------------------------------------------------

def sinkhorn_ipm(
    phi1: torch.Tensor,
    phi0: torch.Tensor,
    blur: float = 0.05
) -> torch.Tensor:
    """
    Compute Sinkhorn divergence between treated (phi1) and control (phi0)
    representations. Uses geomloss.SamplesLoss.

    blur: entropic regularization parameter (lower -> closer to W1).
    """
    try:
        from geomloss import SamplesLoss
    except ImportError:
        # Fallback: MMD with RBF kernel if geomloss not installed
        return mmd_rbf(phi1, phi0)

    loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=blur, backend="auto")
    return loss_fn(phi1, phi0)


def mmd_rbf(phi1: torch.Tensor, phi0: torch.Tensor,
            sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF kernel as fallback IPM."""
    def rbf(a, b):
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-diff.pow(2).sum(-1) / (2 * sigma ** 2))
    return rbf(phi1, phi1).mean() - 2 * rbf(phi1, phi0).mean() + rbf(phi0, phi0).mean()


# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------

def train_cfrnet(
    X_train: np.ndarray,
    D_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float = 1.0,
    repr_dim: int = 100,
    hidden_dim: int = 100,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 256,
    device: str = "cpu",
    model_type: str = "cfrnet",  # "cfrnet" or "dragonnet"
    beta: float = 0.5,           # propensity weight for DragonNet
    seed: int = 0
) -> nn.Module:
    torch.manual_seed(seed)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    D_t = torch.tensor(D_train, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, D_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    if model_type == "dragonnet":
        model = DragonNet(input_dim, repr_dim=repr_dim, hidden_dim=hidden_dim)
    else:
        model = CFRNet(input_dim, repr_dim=repr_dim, hidden_dim=hidden_dim)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, db, yb in loader:
            optimizer.zero_grad()
            if model_type == "dragonnet":
                y_hat, y0, y1, e_hat, phi = model(xb, db)
                outcome_loss = mse(y_hat, yb)
                prop_loss = bce(e_hat, db)
                # Targeted regularization (simplified)
                eps = 1e-6
                tr_loss = ((yb - y_hat - (2*db - 1) /
                            (e_hat.clamp(eps, 1-eps))) ** 2).mean()
                loss = outcome_loss + beta * prop_loss + 0.1 * tr_loss
            else:
                y_hat, y0, y1, phi = model(xb, db)
                outcome_loss = mse(y_hat, yb)
                # IPM between treated and control representations
                phi1 = phi[db == 1]
                phi0 = phi[db == 0]
                if phi1.shape[0] > 1 and phi0.shape[0] > 1:
                    ipm_loss = sinkhorn_ipm(phi1, phi0)
                else:
                    ipm_loss = torch.tensor(0.0, device=device)
                loss = outcome_loss + alpha * ipm_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    return model


# ---------------------------------------------------------------------------
# 5.  Hyperparameter tuning via Optuna
# ---------------------------------------------------------------------------

def tune_cfrnet(
    X_tr, D_tr, Y_tr, X_val, D_val, Y_val,
    tau_val_proxy,  # proxy ITE for PEHE (from OHE experiment or simulation)
    n_trials: int = 30,
    device: str = "cpu"
) -> dict:
    """
    Tune CFRNet hyperparameters using Optuna.
    Objective: validation PEHE (when tau_val_proxy available)
    or factual MSE + alpha * IPM on validation set.
    """
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        repr_dim = trial.suggest_categorical("repr_dim", [50, 100, 200])
        hidden_dim = trial.suggest_categorical("hidden_dim", [100, 200])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        n_epochs = trial.suggest_int("n_epochs", 100, 300, step=50)

        model = train_cfrnet(
            X_tr, D_tr, Y_tr,
            alpha=alpha, repr_dim=repr_dim, hidden_dim=hidden_dim,
            lr=lr, n_epochs=n_epochs, device=device, seed=trial.number
        )
        model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
            tau_hat = model.predict_ite(X_v).cpu().numpy()

        if tau_val_proxy is not None:
            pehe = np.sqrt(np.mean((tau_hat - tau_val_proxy) ** 2))
            return pehe
        else:
            # Fall back to factual MSE on validation
            D_v = torch.tensor(D_val, dtype=torch.float32).to(device)
            Y_v = torch.tensor(Y_val, dtype=torch.float32).to(device)
            with torch.no_grad():
                y_hat, _, _, phi = model(X_v, D_v)
                factual_mse = ((y_hat.cpu().numpy() - Y_val) ** 2).mean()
                phi1 = phi[D_v == 1]; phi0 = phi[D_v == 0]
                ipm = sinkhorn_ipm(phi1, phi0).item() if phi1.shape[0] > 1 else 0.0
            return factual_mse + alpha * ipm

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# 6.  Evaluation helpers
# ---------------------------------------------------------------------------

def compute_ipm_at_alpha(
    X: np.ndarray, D: np.ndarray, Y: np.ndarray,
    alphas: list[float], device: str = "cpu"
) -> pd.DataFrame:
    """
    Train CFRNet at each alpha value, record validation factual MSE and IPM.
    Returns DataFrame for tradeoff curve plotting.
    """
    records = []
    for alpha in alphas:
        model = train_cfrnet(X, D, Y, alpha=alpha, n_epochs=150, device=device)
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(device)
            Dt = torch.tensor(D, dtype=torch.float32).to(device)
            _, _, _, phi = model(Xt, Dt)
            phi1 = phi[Dt == 1]; phi0 = phi[Dt == 0]
            ipm_val = sinkhorn_ipm(phi1, phi0).item()
            Yt = torch.tensor(Y, dtype=torch.float32).to(device)
            y_hat, _, _, _ = model(Xt, Dt)
            mse_val = ((y_hat.cpu().numpy() - Y) ** 2).mean()
        records.append({"alpha": alpha, "ipm": ipm_val, "factual_mse": mse_val})
    return pd.DataFrame(records)


def propensity_ipm_baseline(
    X: np.ndarray, D: np.ndarray, device: str = "cpu"
) -> float:
    """
    Compute Sinkhorn IPM in raw X space (after standardization).
    This is the 'no representation' baseline.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    X1 = torch.tensor(X_scaled[D == 1]).to(device)
    X0 = torch.tensor(X_scaled[D == 0]).to(device)
    with torch.no_grad():
        return sinkhorn_ipm(X1, X0).item()


def logistic_ps_ipm(X: np.ndarray, D: np.ndarray, device: str = "cpu") -> float:
    """
    Fit logistic propensity score, project X onto scalar PS,
    compute IPM in 1D PS space (Rosenbaum-Rubin balancing score theorem).
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lr_ps = LogisticRegression(max_iter=1000, C=1.0)
    lr_ps.fit(X_sc, D)
    ps = lr_ps.predict_proba(X_sc)[:, 1].astype(np.float32)
    ps1 = torch.tensor(ps[D == 1].reshape(-1, 1)).to(device)
    ps0 = torch.tensor(ps[D == 0].reshape(-1, 1)).to(device)
    with torch.no_grad():
        return sinkhorn_ipm(ps1, ps0).item()


# ---------------------------------------------------------------------------
# 7.  Visualization: t-SNE of representations
# ---------------------------------------------------------------------------

def plot_representation_tsne(
    model: nn.Module,
    X: np.ndarray,
    D: np.ndarray,
    device: str = "cpu",
    title: str = "CFRNet Representation (t-SNE)"
) -> plt.Figure:
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        phi = model.encoder(Xt).cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=500)
    emb = tsne.fit_transform(phi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (X_in, label, t) in zip(axes, [
        (X, D, "Raw covariates (standardized, PC1-PC2)"),
        (phi, D, title)
    ]):
        if label is D and X_in is X:
            from sklearn.decomposition import PCA
            scaler = StandardScaler()
            X_plot = PCA(2).fit_transform(scaler.fit_transform(X_in))
        else:
            X_plot = emb
        ax.scatter(X_plot[D == 0, 0], X_plot[D == 0, 1],
                   alpha=0.3, s=5, c="steelblue", label="Control")
        ax.scatter(X_plot[D == 1, 0], X_plot[D == 1, 1],
                   alpha=0.3, s=5, c="tomato", label="Treated")
        ax.set_title(t)
        ax.legend(markerscale=3)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_pehe_ipm_tradeoff(tradeoff_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogx(tradeoff_df["alpha"], tradeoff_df["ipm"], "o-")
    axes[0].set_xlabel("alpha (log scale)")
    axes[0].set_ylabel("Sinkhorn IPM")
    axes[0].set_title("Balance vs Alpha")

    axes[1].scatter(tradeoff_df["ipm"], tradeoff_df["factual_mse"],
                    c=np.log(tradeoff_df["alpha"]), cmap="viridis", s=60)
    axes[1].set_xlabel("Sinkhorn IPM (representation imbalance)")
    axes[1].set_ylabel("Factual MSE")
    axes[1].set_title("Balance-Accuracy Tradeoff")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(
                                   np.log(tradeoff_df["alpha"].min()),
                                   np.log(tradeoff_df["alpha"].max())))
    plt.colorbar(sm, ax=axes[1], label="log(alpha)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8.  Main pipeline: simulate -> train -> evaluate -> plot
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Data: use simulation (replace load_brfss_aca with real BRFSS if available)
    # -----------------------------------------------------------------------
    X, D, Y, tau_true = simulate_brfss_like(n=10000, p=55, seed=0)
    D_int = D.astype(int)
    print(f"Dataset: n={len(X)}, p={X.shape[1]}, "
          f"treatment rate={D.mean():.3f}, ATE_true={tau_true.mean():.3f}")

    # Standardize
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)

    # Train/val/test split
    idx = np.arange(len(X_sc))
    idx_tr, idx_test = train_test_split(idx, test_size=0.2, random_state=1)
    idx_tr, idx_val = train_test_split(idx_tr, test_size=0.15, random_state=2)

    X_tr, D_tr, Y_tr = X_sc[idx_tr], D[idx_tr], Y[idx_tr]
    X_val, D_val, Y_val = X_sc[idx_val], D[idx_val], Y[idx_val]
    X_test, D_test, Y_test = X_sc[idx_test], D[idx_test], Y[idx_test]
    tau_test = tau_true[idx_test]

    # -----------------------------------------------------------------------
    # Baselines: IPM in raw X and logistic PS spaces
    # -----------------------------------------------------------------------
    baseline_ipm_raw = propensity_ipm_baseline(X_sc, D_int, device)
    baseline_ipm_ps = logistic_ps_ipm(X_sc, D_int, device)
    print(f"\nBaseline IPM (raw X, std):      {baseline_ipm_raw:.4f}")
    print(f"Baseline IPM (logistic PS):     {baseline_ipm_ps:.4f}")

    # -----------------------------------------------------------------------
    # Hyperparameter tuning via Optuna
    # -----------------------------------------------------------------------
    print("\nRunning Optuna hyperparameter search (30 trials)...")
    best_params = tune_cfrnet(
        X_tr, D_tr, Y_tr, X_val, D_val, Y_val,
        tau_val_proxy=tau_true[idx_val],
        n_trials=30, device=device
    )
    print(f"Best params: {best_params}")

    # -----------------------------------------------------------------------
    # Train final CFRNet and DragonNet with best alpha
    # -----------------------------------------------------------------------
    best_alpha = best_params.get("alpha", 1.0)
    best_repr = best_params.get("repr_dim", 100)
    best_hidden = best_params.get("hidden_dim", 100)
    best_lr = best_params.get("lr", 1e-3)
    best_epochs = best_params.get("n_epochs", 200)

    print("\nTraining final CFRNet...")
    cfrnet = train_cfrnet(
        X_tr, D_tr, Y_tr, alpha=best_alpha,
        repr_dim=best_repr, hidden_dim=best_hidden,
        lr=best_lr, n_epochs=best_epochs,
        device=device, model_type="cfrnet"
    )

    print("Training DragonNet...")
    dragonnet = train_cfrnet(
        X_tr, D_tr, Y_tr,
        repr_dim=best_repr, hidden_dim=best_hidden,
        lr=best_lr, n_epochs=best_epochs,
        device=device, model_type="dragonnet", beta=0.5
    )

    # -----------------------------------------------------------------------
    # Evaluate: PEHE and ATE on test set
    # -----------------------------------------------------------------------
    for name, model in [("CFRNet", cfrnet), ("DragonNet", dragonnet)]:
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
            tau_hat = model.predict_ite(Xt).cpu().numpy()
        pehe = np.sqrt(np.mean((tau_hat - tau_test) ** 2))
        ate_hat = tau_hat.mean()
        ate_true = tau_test.mean()
        print(f"\n{name}:")
        print(f"  PEHE = {pehe:.4f}")
        print(f"  ATE estimate = {ate_hat:.4f}, true = {ate_true:.4f}")

        # IPM of representations on test set
        with torch.no_grad():
            Dt = torch.tensor(D_test, dtype=torch.float32).to(device)
            if name == "CFRNet":
                _, _, _, phi = model(Xt, Dt)
            else:
                _, _, _, _, phi = model(Xt, Dt)
            phi1 = phi[Dt == 1]; phi0 = phi[Dt == 0]
            repr_ipm = sinkhorn_ipm(phi1, phi0).item()
        print(f"  Representation IPM = {repr_ipm:.4f} "
              f"(raw X baseline: {baseline_ipm_raw:.4f})")

    # -----------------------------------------------------------------------
    # Tradeoff curve: PEHE vs IPM across alpha values
    # -----------------------------------------------------------------------
    print("\nComputing PEHE-IPM tradeoff curve...")
    alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    tradeoff_df = compute_ipm_at_alpha(X_tr, D_tr, Y_tr, alphas, device=device)
    print(tradeoff_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig_tsne = plot_representation_tsne(cfrnet, X_test, D_int[idx_test],
                                         device=device, title="CFRNet Representation")
    fig_tsne.savefig("chapter39_tsne.png", dpi=150, bbox_inches="tight")
    print("\nSaved: chapter39_tsne.png")

    fig_tradeoff = plot_pehe_ipm_tradeoff(tradeoff_df)
    fig_tradeoff.savefig("chapter39_tradeoff.png", dpi=150, bbox_inches="tight")
    print("Saved: chapter39_tradeoff.png")


if __name__ == "__main__":
    main()
```

---

## Summary

- **CFRNet frames causal effect estimation as representation learning with a distributional balance penalty.** The key bound decomposes PEHE into measurable factual prediction error, measurable IPM between treated and control representations, and an irreducible constant — creating a directly optimizable surrogate for the unidentifiable PEHE.

- **Integral probability metrics provide the balancing signal.** MMD (RKHS-based) is analytically tractable; Wasserstein-1 respects the geometry of the representation space and yields smoother gradients. Sinkhorn divergences are computationally efficient differentiable approximations suitable for mini-batch training.

- **DragonNet adds a propensity head to the shared encoder**, creating implicit balance through the requirement that the representation supports treatment assignment prediction. Targeted regularization further enforces alignment between neural network outputs and the AIPW efficient influence function.

- **The balance-informativeness tradeoff is fundamental.** As the IPM penalty weight $\alpha \to \infty$, the encoder collapses to a constant (perfectly balanced, useless). Optimal $\alpha$ is dataset-specific and must be tuned; Optuna with a validation PEHE criterion is effective when proxy ITEs are available from experimental data.

- **Synthetic interventions generalize the approach to matrix factorization settings**, recovering counterfactuals via weighted combinations of donor units under low-rank outcome structure — an analogue of synthetic control that scales to many interventions.

- **Distribution shift in representation space is the key failure mode.** Models trained on expansion states may produce unreliable estimates for never-expansion states if their covariates fall outside the training representation's support. Domain-invariant penalties (DANN-style) or explicit test-distribution coverage checks are necessary safeguards.

- **Representation learning methods do not relax the identification assumptions** — strong ignorability and overlap are still required. They improve finite-sample performance by actively engineering overlap in a learned space, not by identifying causal effects under unmeasured confounding.

---

## Further Reading

1. **Johansson, Shalit, and Sontag (2016). "Learning Representations for Counterfactual Inference."** ICML 2016. The original CFRNet paper; derives the IPM-based PEHE bound and demonstrates improvements over propensity-weighted baselines on semi-synthetic datasets. Essential starting point.

2. **Shi, Bloebaum, and Ermon (2019). "Adapting Neural Networks for the Estimation of Treatment Effects."** NeurIPS 2019. Introduces DragonNet; demonstrates that incorporating propensity information into the representation yields better ATE estimates than IPM-penalized methods on the IHDP and Jobs benchmarks.

3. **Agarwal, Shah, Shen, and Song (2020). "Synthetic Interventions."** Extends synthetic control to multiple interventions via matrix factorization; provides finite-sample guarantees under the approximate low-rank model. Directly applicable to staggered Medicaid expansion.

4. **Curth and van der Schaar (2021). "Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms."** AISTATS 2021. Systematic comparison of meta-learners (T/S/X/R-learners) with representation methods; clarifies when representation learning adds value beyond simpler plug-in approaches.

5. **Ben-David, Blitzer, Crammer, Kulesza, Pereira, and Vaughan (2010). "A Theory of Learning from Different Domains."** Machine Learning. Provides the $\mathcal{H}\Delta\mathcal{H}$-divergence framework underlying the distribution shift bound in Section 39.6; the domain adaptation literature is the theoretical parent of representation-based causal methods.

6. **Hassanpour and Greiner (2020). "Learning Disentangled Representations for CounterFactual Regression."** ICLR 2020. Decomposes the representation into instrument-like, confounding, and adjustment components — a structural refinement of CFRNet that reduces the balance-informativeness tension by learning factored representations.
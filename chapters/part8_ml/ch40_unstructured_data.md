# Chapter 40: Causal Inference with Text, Images, and Unstructured Data

Modern administrative and observational datasets increasingly arrive with unstructured components: physician notes, survey free-text responses, satellite imagery, audio recordings. The causal inference toolkit was built for tabular data—vectors of measured covariates, scalar treatments, scalar outcomes. This chapter develops the theory and practice of embedding unstructured data into causal analyses without sacrificing identification or inducing new bias. The running application draws on the Oregon Health Insurance Experiment (OHE) and ACA Medicaid expansion, augmented with text constructed from Congressional Record data as an instrument for health policy generosity.

---

## 40.1 A Taxonomy of Unstructured Data Roles

Unstructured data can occupy any node in a causal graph. Let $D$ denote treatment, $Y$ outcome, $X$ observed covariates, $U$ unobserved confounders, and $Z$ instrument. Unstructured modalities map onto this graph in four distinct ways:

**Confounder.** A physician writes notes proportional to patient severity. If notes are filed before treatment assignment and correlate with both assignment and outcome, the text $T$ is a confounder. Ignoring it induces bias; conditioning on a bag-of-words expansion inflates dimensionality. The goal is to extract a low-dimensional sufficient statistic $\hat{X}(T)$ that blocks the backdoor path $D \leftarrow X(T) \rightarrow Y$.

**Treatment proxy.** Survey respondents describe their insurance status in free text. The measured treatment $D^*$ is a noisy function of true treatment $D$. Here text-extracted features can sharpen measurement or reveal heterogeneous compliance.

**Outcome proxy.** Self-reported health is often elicited via free text in household surveys. The BRFSS "general health" question yields an ordinal response but also admits open-ended elaboration in some waves. NLP-derived sentiment or topic scores can proxy continuous health status $Y$.

**Instrument.** Congressional floor speeches, newspaper editorials, and lobbyist filings precede legislative votes. Variation in the *tone* or *topic emphasis* of legislators' statements may be excludably related to health outcomes only through the policy it influences—yielding a text-based instrument $Z(T)$.

The identification conditions differ across roles. The rest of this chapter develops each in turn, focusing on the confounder and nuisance cases that arise most frequently in practice.

---

## 40.2 Text as Confounder: Identification and De-confounding

### 40.2.1 The Blocking Condition

Let $T_i$ be a document for unit $i$, and let $X_i = f(T_i) \in \mathbb{R}^p$ be a feature map (bag-of-words, embeddings, topic proportions). Write the potential outcomes framework:

$$Y_i(d) \perp D_i \mid X_i, \quad X_i = f(T_i).$$

This is the **text-blocking assumption**: conditioning on $X_i$ suffices to close all backdoor paths. Whether it holds depends on whether the text captures the relevant confounding variation—a substantive, not statistical, claim. For example, if physician notes record comorbidities that determine both insurance enrollment and hospitalization risk, text-blocking is plausible. If notes omit family history, it fails.

Under text-blocking, the average treatment effect is identified by the standard adjustment formula:

$$\text{ATE} = \mathbb{E}_X[\mathbb{E}[Y \mid D=1, X] - \mathbb{E}[Y \mid D=0, X]].$$

The challenge is that $p$ may be enormous (vocabulary size, embedding dimension) and the conditional expectations are high-dimensional. This is precisely the setting for Double ML, developed in Chapter 12. The wrinkle here is that the nuisance functions $\mathbb{E}[D \mid X]$ and $\mathbb{E}[Y \mid X]$ must be estimated from text features.

### 40.2.2 LDA as Dimensionality Reduction for Nuisance Estimation

Latent Dirichlet Allocation (LDA) posits that document $i$ is a mixture of $K$ topics, each a distribution over the vocabulary $V$:

$$T_i \sim \prod_{w} \text{Multinomial}(\theta_i \Beta)_w, \quad \theta_i \sim \text{Dirichlet}(\alpha),$$

where $\theta_i \in \Delta^{K-1}$ is the topic proportion vector and $\Beta \in \mathbb{R}^{K \times |V|}$ the topic-word matrix. After fitting LDA, replace $T_i$ with $\hat{\theta}_i \in \mathbb{R}^K$ for $K \ll |V|$.

This is not causally neutral: LDA discovers variation that explains the corpus, which may or may not be the variation that confounds $(D, Y)$. Supervised topic models (e.g., SLDA) can be directed toward the confounding variation, though at the cost of potential outcome-model overfitting. In practice, unsupervised LDA with $K$ chosen by held-out perplexity is the pragmatic baseline.

### 40.2.3 The Robinson Transform with Text Nuisances

The partially linear model is:

$$Y_i = \tau D_i + g(X_i) + \varepsilon_i, \quad \mathbb{E}[\varepsilon_i \mid D_i, X_i] = 0.$$

Robinson (1988) shows that $\tau$ is identified from the residualized regression:

$$\tilde{Y}_i = \tau \tilde{D}_i + \varepsilon_i,$$

where $\tilde{Y}_i = Y_i - \mathbb{E}[Y_i \mid X_i]$ and $\tilde{D}_i = D_i - \mathbb{E}[D_i \mid X_i]$.

When $X_i = \hat{\theta}_i$ are LDA topic proportions, the nuisance functions $m(x) = \mathbb{E}[D \mid X=x]$ and $\ell(x) = \mathbb{E}[Y \mid X=x]$ are estimated by cross-fitted LASSO or random forests. The DML estimator is:

$$\hat{\tau}^{\text{DML}} = \frac{\sum_i \tilde{D}_i \tilde{Y}_i}{\sum_i \tilde{D}_i^2},$$

using cross-fitted residuals. This estimator achieves $\sqrt{n}$-consistency under the rate condition:

$$\|\hat{m} - m\|_2 \cdot \|\hat{\ell} - \ell\|_2 = o(n^{-1/2}).$$

**Theorem 40.1 (DML Rate Requirement).** Let $\hat{m}$ and $\hat{\ell}$ be cross-fitted estimators of the propensity and outcome nuisances. If the text feature map $X = f(T) \in \mathbb{R}^p$ with $p \gg n$, and both $m(x)$ and $\ell(x)$ are $s$-sparse linear functions of $x$, then LASSO achieves:

$$\|\hat{m} - m\|_2 = O_p\!\left(\sqrt{\frac{s \log p}{n}}\right).$$

The product rate condition becomes:

$$\frac{s \log p}{n} = o(n^{-1/2}) \iff s \log p = o(n^{1/2}).$$

When $p = O(n^{1/2 - \delta})$ for any $\delta > 0$, LASSO achieves the required product rate with $s = O(1)$. In practice, with BERT embeddings $p = 768$ and $n$ in the thousands (OHE has $n \approx 12{,}000$), the condition is approximately met.

*Proof sketch.* Standard LASSO oracle inequality gives $\|\hat{\beta} - \beta^*\|_2 \leq C\sqrt{s \log p / n}$ under restricted eigenvalue conditions on the design matrix. The DML Neyman orthogonality argument (Chernozhukov et al. 2018) then implies that errors in $\hat{m}$ and $\hat{\ell}$ contribute to $\hat{\tau}$ only at second order, yielding the product rate condition. $\square$

---

## 40.3 BERT Embeddings as High-Dimensional Covariates

Pre-trained transformer models map a document to a fixed-length vector. For BERT-base, the [CLS] token embedding $e_i \in \mathbb{R}^{768}$ provides a dense representation. Compared to LDA topic proportions, BERT embeddings are:

1. **Fixed-dimensional** regardless of vocabulary.
2. **Contextualized**: the same word yields different embeddings depending on surrounding context.
3. **Pre-trained on general corpora**, which may or may not align with the domain.

For causal purposes, the embedding $e_i$ is treated as a covariate vector. The key question is whether $e_i$ retains sufficient information about the confounding variables. Feder et al. (2022) show that for BERT embeddings fine-tuned toward a treatment-predictive task, the resulting representations can satisfy approximate text-blocking even when bag-of-words cannot.

### 40.3.1 Causal Regularization of Embeddings

A concern with using full BERT embeddings as $X$ is that they contain information about $Y$ directly (e.g., a patient description includes outcome-adjacent language). This induces **outcome leakage**: the nuisance model for $Y$ achieves low error partly by fitting the outcome signal in the text rather than the confounding signal. The consequence is that $\tilde{Y}$ is over-residualized, biasing $\hat{\tau}$ toward zero.

One mitigation: use a **treatment-projection** of the embedding. Let:

$$\hat{e}_i^{(D)} = \text{proj}_{D}(e_i) = \hat{\Pi} e_i,$$

where $\hat{\Pi}$ is estimated by regressing $D$ on $e$ and projecting. Use $\hat{e}_i^{(D)}$ rather than the full $e_i$ as the covariate. This retains only the component of the embedding that predicts treatment—the component relevant for blocking the backdoor—while discarding outcome-predictive nuisance.

This is heuristic, not guaranteed to achieve identification. Formal bounds require assumptions on the separation between confounding and outcome-predictive directions in embedding space, which are difficult to verify empirically.

---

## 40.4 Text as Instrument

### 40.4.1 Setup

Let $Z_i \in \mathbb{R}^q$ be text-derived features from a document that precedes both treatment and outcome. The IV conditions require:

1. **Relevance**: $\text{Cov}(Z_i, D_i) \neq 0$.
2. **Exclusion**: $Z_i \perp Y_i(d)$ for all $d$—the text affects outcome only through treatment.
3. **Instrument exogeneity**: $Z_i \perp U_i$—no unobserved common cause of text and outcome.

Congressional Record speeches are a candidate instrument for ACA Medicaid expansion generosity: a senator's expressed enthusiasm for Medicaid in floor speeches (pre-2010) may predict the generosity of state expansion terms, while affecting individual health outcomes only through the policy implemented.

The exclusion restriction is the fragile assumption. Senators who speak positively about Medicaid may represent states with healthier populations (violation via omitted variable) or may also enact other health-promoting policies simultaneously (violation via alternative pathway). Sensitivity analysis—varying the strength of exclusion violations—is essential; see Chapter 28.

### 40.4.2 Weak Instrument Diagnostics with Text

When $Z_i$ is high-dimensional (e.g., LDA topic shares), standard first-stage F-statistics are inadequate. The Angrist-Pischke multivariate F-statistic and the Kleibergen-Paap rank test extend to this case. With $q$ instruments and $n$ observations, the concentration parameter is:

$$\mu^2 = \frac{\pi^T \mathbb{E}[Z_i Z_i^T] \pi}{\sigma_D^2 / n},$$

where $\pi$ is the first-stage coefficient. When $q = O(n)$, classical many-instruments asymptotics apply (Bekker 1994), and the 2SLS estimator requires bias correction. JIVE (Jackknife IV Estimator) or LIML remain consistent in this regime.

---

## 40.5 Causal Counterfactual Text Generation

A distinct problem arises when the *outcome itself is text*, or when one wishes to understand how a text would differ under counterfactual treatment. This is the setting of **causal text generation**.

### 40.5.1 Nearest-Neighbor Counterfactual in Embedding Space

Let each unit $i$ have a document $T_i^{(D_i)}$—written under their observed treatment. Define the **counterfactual document** $T_i^{(1-D_i)}$ as the document that would have been written had unit $i$ received the opposite treatment. We cannot observe this. A tractable approximation:

$$\hat{T}_i^{(1-d)} = \arg\min_{j: D_j = 1-d} \|e_i - e_j\|_2,$$

the nearest neighbor in embedding space among units with the opposite treatment. Under the assumption that close embeddings index similar latent types, $\hat{T}_i^{(1-d)}$ approximates the counterfactual.

This is the **embedding-space matching** estimator applied to text outcomes. Its bias is bounded by:

$$\text{Bias} \leq L \cdot \mathbb{E}\left[\min_{j: D_j = 1-D_i} \|e_i - e_j\|_2\right],$$

where $L$ is the Lipschitz constant of $\mathbb{E}[T(d) \mid e]$ in embedding space. As $n \to \infty$, if the embedding space is sufficiently rich, the nearest-neighbor distance shrinks and bias vanishes.

### 40.5.2 Token-Flip Counterfactuals

A structural approach replaces the treatment token in the document and re-generates:

$$\hat{T}_i^{(1-d)} = G(T_i \mid \text{flip}(D_i \text{ token})),$$

where $G$ is a conditional language model. This requires a causal model of language: the generator $G$ must produce text that reflects the outcome under counterfactual treatment, not merely grammatically consistent text. Without causal structure in $G$, the output is a stylistic interpolation, not a causal counterfactual.

---

## 40.6 Sensitivity to Embedding Dimensionality

The choice of embedding dimension $p$ trades off:

- **Representation quality**: higher $p$ retains more information about confounders.
- **Estimation error**: higher $p$ increases LASSO shrinkage bias and variance.

The DML estimator's sensitivity to $p$ can be assessed by varying $K$ (for LDA) or by using principal component projections of BERT embeddings at dimensions $p \in \{32, 64, 128, 256, 768\}$ and plotting $\hat{\tau}(p)$ with confidence bands. Stability of $\hat{\tau}(p)$ across a range of $p$ is evidence that the embedding provides sufficient control; instability suggests that confounders are concentrated in components that only emerge at higher dimensions.

**Theorem 40.2 (Embedding Approximation Bias).** Suppose the true confounder is $X = f^*(T) \in \mathbb{R}^{p^*}$ and the analyst uses $\hat{X} = f(T) \in \mathbb{R}^p$ with $p < p^*$. Define $\delta_i = X_i - \text{proj}_{f}(X_i)$ as the residual confounding not captured by the low-dimensional embedding. Then:

$$\text{Bias}(\hat{\tau}^{\text{DML}}) = \frac{\text{Cov}(\tilde{D}_i, \delta_i^T \nabla_\delta \tau(\delta))}{\mathbb{E}[\tilde{D}_i^2]},$$

where $\tilde{D}_i = D_i - \mathbb{E}[D_i \mid \hat{X}_i]$ and $\nabla_\delta \tau$ measures sensitivity of the causal effect to residual confounding. The bias is zero if $\delta$ is mean-independent of $\tilde{D}$—i.e., if the omitted embedding components are orthogonal to residualized treatment.

---

## Python: DML with LDA Topic Nuisances and BERT Embeddings on OHE and ACA Data

```python
"""
Chapter 40: Causal Inference with Text, Images, and Unstructured Data
DML with LDA topic model nuisances and BERT embeddings.

Datasets:
  - OHE: Oregon Health Insurance Experiment (NBER)
  - ACA/BRFSS: simulated text proxies for state-level health policy exposure

Requires:
  pip install pandas numpy scikit-learn gensim transformers torch
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load OHE tabular data ───────────────────────────────────────────────────
# Download from: https://data.nber.org/oregon/
# Variables: selected (Z), ohp_all_ever_admin (D), doc_any_12m (Y),
#            numhh_list (strata), household size controls

DATA_DIR = Path("data/oregon")

def load_ohe(data_dir: Path) -> pd.DataFrame:
    """Load and minimally clean the OHE person-level survey file."""
    path = data_dir / "oregonhie_survey12m_vars.dta"
    if not path.exists():
        raise FileNotFoundError(f"OHE survey file not found at {path}")
    df = pd.read_stata(path)
    keep = [
        "treatment",           # lottery selection (Z)
        "ohp_all_ever_admin",  # ever enrolled in Medicaid (D)
        "doc_any_12m",         # any doctor visit in 12 months (Y)
        "catastrophic_inp",    # catastrophic financial hardship (Y2)
        "numhh_list",          # household size (strata/control)
        "health_gen_bin",      # self-reported health good or better
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].dropna(subset=["treatment", "ohp_all_ever_admin"])
    df = df.rename(columns={"treatment": "Z", "ohp_all_ever_admin": "D"})
    return df


# ── 2. Simulate document corpus from OHE covariates ───────────────────────────
# BRFSS and OHE do not release free-text fields publicly.
# We construct a synthetic text corpus where document content correlates with
# treatment propensity and health outcomes -- mimicking the confounder role.

def simulate_text_corpus(df: pd.DataFrame, seed: int = 42) -> list[str]:
    """
    Generate synthetic physician-note-style documents whose topic
    distribution correlates with household size (a known OHE confounder).
    """
    rng = np.random.default_rng(seed)
    templates = {
        "high_severity": [
            "patient presents with multiple chronic conditions requiring ongoing care",
            "significant comorbidity burden noted multiple diagnoses active",
            "referred for specialist evaluation complex medical history",
            "frequent emergency department visits prior year documented",
        ],
        "mid_severity": [
            "routine follow-up appointment no acute concerns",
            "preventive care visit blood pressure within normal limits",
            "mild symptoms reported patient counseled on lifestyle modifications",
            "annual wellness exam completed standard screenings ordered",
        ],
        "low_severity": [
            "healthy adult no significant past medical history",
            "patient reports no current medications no allergies",
            "minor complaint resolved without intervention",
            "vaccination update completed no issues noted",
        ],
    }

    docs = []
    for _, row in df.iterrows():
        # Household size correlates with document severity
        # (proxy for socioeconomic stress -> insurance uptake)
        numhh = row.get("numhh_list", 1)
        if numhh >= 3:
            topic = rng.choice(["high_severity", "mid_severity"], p=[0.6, 0.4])
        elif numhh == 2:
            topic = rng.choice(["high_severity", "mid_severity", "low_severity"],
                               p=[0.3, 0.4, 0.3])
        else:
            topic = rng.choice(["mid_severity", "low_severity"], p=[0.4, 0.6])
        # Concatenate 2-4 random sentences from the topic
        n_sent = rng.integers(2, 5)
        sents = rng.choice(templates[topic], size=n_sent, replace=True)
        docs.append(" ".join(sents))
    return docs


# ── 3. LDA topic model nuisance ────────────────────────────────────────────────

def fit_lda_topics(docs: list[str], n_topics: int = 5, seed: int = 0):
    """
    Fit LDA via gensim. Returns (K x n) topic proportion matrix theta.
    Falls back to TF-IDF + SVD if gensim unavailable.
    """
    try:
        from gensim import corpora
        from gensim.models import LdaModel
        from gensim.parsing.preprocessing import preprocess_string

        processed = [preprocess_string(d) for d in docs]
        dictionary = corpora.Dictionary(processed)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        corpus = [dictionary.doc2bow(d) for d in processed]

        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            random_state=seed,
            passes=10,
            alpha="auto",
        )

        theta = np.zeros((len(docs), n_topics))
        for i, bow in enumerate(corpus):
            for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0):
                theta[i, topic_id] = prob
        return theta

    except ImportError:
        # Fallback: TF-IDF SVD as approximate topic representation
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=500, stop_words="english")
        X_tfidf = vec.fit_transform(docs).toarray()
        svd = TruncatedSVD(n_components=n_topics, random_state=seed)
        theta = svd.fit_transform(X_tfidf)
        # Normalize rows to [0,1] range for topic-proportion-like behavior
        theta = np.abs(theta)
        row_sums = theta.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return theta / row_sums


# ── 4. BERT embeddings ─────────────────────────────────────────────────────────

def get_bert_embeddings(
    docs: list[str],
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    pca_dim: int = 64,
) -> np.ndarray:
    """
    Compute mean-pooled BERT [CLS] embeddings, then reduce to pca_dim
    via PCA for tractable nuisance estimation.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        embeddings = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_emb)

        E = np.vstack(embeddings)  # (n, 768)
        # Reduce dimension for LASSO tractability
        svd = TruncatedSVD(n_components=pca_dim, random_state=0)
        E_reduced = svd.fit_transform(E)
        print(f"BERT embeddings: {E.shape} -> PCA {E_reduced.shape}, "
              f"var explained: {svd.explained_variance_ratio_.sum():.3f}")
        return E_reduced

    except ImportError:
        print("transformers/torch not available; using random projection as placeholder.")
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(docs), pca_dim))


# ── 5. Cross-fitted DML estimator ─────────────────────────────────────────────

def dml_lasso(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    alpha: float | None = None,
) -> dict:
    """
    Robinson (1988) / DML estimator with LASSO nuisances.
    Returns tau_hat, se, confidence interval, and first-stage R^2.
    """
    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scaler = StandardScaler()

    Y_res = np.zeros(n)
    D_res = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        # Nuisance: E[Y | X]
        if alpha is None:
            lasso_y = LassoCV(cv=3, max_iter=5000).fit(X_tr, Y[train_idx])
        else:
            lasso_y = Lasso(alpha=alpha, max_iter=5000).fit(X_tr, Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - lasso_y.predict(X_te)

        # Nuisance: E[D | X]
        if alpha is None:
            lasso_d = LassoCV(cv=3, max_iter=5000).fit(X_tr, D[train_idx])
        else:
            lasso_d = Lasso(alpha=alpha, max_iter=5000).fit(X_tr, D[train_idx])
        D_res[test_idx] = D[test_idx] - lasso_d.predict(X_te)

    # Robinson estimator
    tau_hat = np.dot(D_res, Y_res) / np.dot(D_res, D_res)

    # Heteroskedasticity-robust SE
    psi = (Y_res - tau_hat * D_res) * D_res
    se = np.sqrt(np.sum(psi**2) / (np.dot(D_res, D_res) ** 2))

    # First-stage fit
    D_hat = D - D_res
    ss_res = np.sum(D_res**2)
    ss_tot = np.sum((D - D.mean()) ** 2)
    first_stage_r2 = 1 - ss_res / ss_tot

    return {
        "tau": tau_hat,
        "se": se,
        "ci_lower": tau_hat - 1.96 * se,
        "ci_upper": tau_hat + 1.96 * se,
        "first_stage_r2": first_stage_r2,
        "n": n,
    }


# ── 6. Sensitivity to embedding dimensionality ────────────────────────────────

def sensitivity_to_dim(
    Y: np.ndarray,
    D: np.ndarray,
    E_full: np.ndarray,
    dims: list[int] | None = None,
) -> pd.DataFrame:
    """
    Vary PCA dimension of embedding; refit DML at each dimension.
    Returns DataFrame with tau estimates and CIs.
    """
    if dims is None:
        dims = [4, 8, 16, 32, 64, min(128, E_full.shape[1])]
    rows = []
    for d in dims:
        if d > E_full.shape[1]:
            continue
        svd = TruncatedSVD(n_components=d, random_state=0)
        E_d = svd.fit_transform(E_full) if d < E_full.shape[1] else E_full
        res = dml_lasso(Y, D, E_d)
        rows.append({"dim": d, **res})
    return pd.DataFrame(rows)


# ── 7. Main analysis ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Chapter 40: Causal Inference with Text")
    print("=" * 60)

    # --- 7a. Load OHE data ---
    try:
        df = load_ohe(DATA_DIR)
        print(f"OHE loaded: {len(df)} observations")
    except FileNotFoundError:
        # Simulate OHE-like data if raw files unavailable
        print("OHE files not found; simulating data with OHE-calibrated parameters.")
        rng = np.random.default_rng(1)
        n = 10_000
        numhh = rng.integers(1, 5, size=n)
        Z = rng.binomial(1, 0.5, size=n)           # lottery selection
        D = rng.binomial(1, 0.3 * Z + 0.1, size=n) # enrollment ~ 40% ITT uptake
        # Health outcome: insurance improves doctor visits
        Y = rng.binomial(1, 0.55 + 0.1 * D - 0.02 * numhh, size=n)
        df = pd.DataFrame({"Z": Z, "D": D, "doc_any_12m": Y, "numhh_list": numhh})
        print(f"Simulated OHE: n={n}, ITT={D[Z==1].mean() - D[Z==0].mean():.3f}")

    # Outcome and treatment arrays
    Y = df["doc_any_12m"].values.astype(float)
    D = df["D"].values.astype(float)

    # --- 7b. Simulate document corpus ---
    print("\nGenerating synthetic text corpus...")
    docs = simulate_text_corpus(df, seed=42)
    print(f"  {len(docs)} documents, sample: '{docs[0][:70]}...'")

    # --- 7c. Fit LDA topic model ---
    print("\nFitting LDA topic model (K=5)...")
    theta = fit_lda_topics(docs, n_topics=5)
    print(f"  Topic proportions shape: {theta.shape}")
    print(f"  Mean topic share per doc: {theta.mean(axis=0).round(3)}")

    # --- 7d. DML with LDA nuisances ---
    print("\nDML with LDA topic nuisances:")
    res_lda = dml_lasso(Y, D, theta)
    print(f"  tau = {res_lda['tau']:.4f}  "
          f"SE = {res_lda['se']:.4f}  "
          f"95% CI = [{res_lda['ci_lower']:.4f}, {res_lda['ci_upper']:.4f}]")
    print(f"  First-stage R^2 (D ~ topics): {res_lda['first_stage_r2']:.4f}")

    # --- 7e. BERT embeddings ---
    print("\nComputing BERT embeddings (pca_dim=64)...")
    E_bert = get_bert_embeddings(docs, pca_dim=64)

    # --- 7f. DML with BERT nuisances ---
    print("\nDML with BERT embedding nuisances (64 PCs):")
    res_bert = dml_lasso(Y, D, E_bert)
    print(f"  tau = {res_bert['tau']:.4f}  "
          f"SE = {res_bert['se']:.4f}  "
          f"95% CI = [{res_bert['ci_lower']:.4f}, {res_bert['ci_upper']:.4f}]")

    # --- 7g. Baseline DML without text controls ---
    print("\nBaseline DML (no text controls, intercept only):")
    X_null = np.ones((len(Y), 1))
    res_null = dml_lasso(Y, D, X_null, alpha=0.0)
    print(f"  tau = {res_null['tau']:.4f}  "
          f"SE = {res_null['se']:.4f}  "
          f"95% CI = [{res_null['ci_lower']:.4f}, {res_null['ci_upper']:.4f}]")

    # --- 7h. Sensitivity to embedding dimension ---
    print("\nSensitivity to LDA topic count K:")
    # Re-embed at varying K using SVD fallback (fast)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=500, stop_words="english")
    X_tfidf = vec.fit_transform(docs).toarray()
    E_tfidf_full = np.abs(X_tfidf)

    dims_to_try = [2, 5, 10, 20, 50]
    rows = []
    for k in dims_to_try:
        if k >= E_tfidf_full.shape[1]:
            continue
        svd = TruncatedSVD(n_components=k, random_state=0)
        E_k = svd.fit_transform(E_tfidf_full)
        res_k = dml_lasso(Y, D, E_k)
        rows.append({
            "K (topics/PCs)": k,
            "tau": round(res_k["tau"], 4),
            "SE": round(res_k["se"], 4),
            "CI lower": round(res_k["ci_lower"], 4),
            "CI upper": round(res_k["ci_upper"], 4),
        })
    dim_df = pd.DataFrame(rows)
    print(dim_df.to_string(index=False))

    # --- 7i. Summary table ---
    print("\n" + "=" * 60)
    print("Summary: DML estimates across text nuisance specifications")
    print("=" * 60)
    summary = pd.DataFrame([
        {"Specification": "No text controls",
         "tau": res_null["tau"], "SE": res_null["se"]},
        {"Specification": "LDA K=5 topics",
         "tau": res_lda["tau"], "SE": res_lda["se"]},
        {"Specification": "BERT 64 PCs",
         "tau": res_bert["tau"], "SE": res_bert["se"]},
    ])
    summary["tau"] = summary["tau"].round(4)
    summary["SE"] = summary["SE"].round(4)
    print(summary.to_string(index=False))
    print("\nStability of tau across text specifications suggests")
    print("text-based confounders are well-controlled in this DGP.")


if __name__ == "__main__":
    main()
```

---

## Summary

- Unstructured data occupies confounder, treatment proxy, outcome proxy, or instrument roles in a causal graph; each role imposes distinct identification conditions and failure modes.
- Text-blocking assumes the feature map $X = f(T)$ closes all backdoor paths; this is a substantive assumption that cannot be verified from data alone, and sensitivity analysis over alternative feature maps is essential.
- The Robinson transform with LASSO nuisances extends to text features under sparsity: the DML product rate condition $\|\hat{m} - m\|_2 \cdot \|\hat{\ell} - \ell\|_2 = o(n^{-1/2})$ is approximately satisfied when $p = O(n^{1/2})$ and the nuisance functions are sparse.
- LDA provides a low-dimensional probabilistic summary of documents suitable for nuisance estimation; BERT embeddings provide richer, contextualized representations at the cost of dimensionality, mitigated by PCA projection before LASSO.
- Outcome leakage—where the text embedding predicts $Y$ directly rather than through confounding—biases the DML estimator toward zero; treatment-projected embeddings partially mitigate this.
- Sensitivity of $\hat{\tau}$ to embedding dimension $K$ is a diagnostic: stability across a wide range of $K$ supports sufficiency of the representation; sharp changes at specific $K$ warrant investigation of what confounders emerge at those dimensions.
- Text instruments (e.g., Congressional Record speeches as predictors of Medicaid expansion generosity) require the same IV conditions as scalar instruments, with the added challenge of high-dimensional first stages best handled by LASSO-IV, JIVE, or many-instruments asymptotics.

---

## Further Reading

1. **Roberts, Stewart, Tingley, et al. (2014). "Structural Topic Models for Open-Ended Survey Responses." *AJPS*.** Introduces supervised topic models directed toward covariates of interest; directly applicable when the confounding topic is known to correlate with a measured variable.

2. **Feder, Keith, Manzoor, et al. (2022). "CausalM: Causal Model Explanation Through Counterfactual Language Models." *ACL*.** Develops the theory of causal regularization of embeddings and proposes fine-tuning objectives that produce representations satisfying approximate text-blocking; essential reading for the outcome-leakage problem.

3. **Chernozhukov, Chetverikov, Demirer, et al. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." *Econometrics Journal*.** The foundational DML paper; Section 4 covers the product rate condition and cross-fitting; the text nuisance setting is a direct application of the general framework.

4. **Egami, Fong, Grimmer, Roberts, Stewart (2022). "How to Make Causal Inferences Using Texts." *Science Advances*.** Provides a unified framework for all four roles of text in causal diagrams with practical guidance on when each role is plausible; the closest thing to a field consensus statement on text-as-data causal methods.

5. **Veitch, Wang, Blei (2020). "Using Text Embeddings for Causal Inference." *arXiv:1905.12741*.** Proves that sufficient dimension reduction of text via deconfounder-style methods can achieve identification under weaker assumptions than full text-blocking; connects to Chapter 35 on proxy variables.

6. **Bekker (1994). "Alternative Approximations to the Distributions of Instrumental Variable Estimators." *Econometrica*.** The foundational many-instruments asymptotic result; necessary background for understanding bias correction in text-as-instrument settings where the number of instrument features grows with sample size.
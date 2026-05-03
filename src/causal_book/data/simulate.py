"""
Simulation utilities for the G-methods chapters (25–30).

Generates a longitudinal DGP mimicking the insurance-health setting:
- Binary treatment A_t (insurance coverage at time t)
- Time-varying confounder L_t (health status, affected by prior treatment)
- Outcome Y_t (health outcomes)
- Unmeasured U (baseline health propensity)

Treatment and confounders feed back into each other, violating the
no-unmeasured-confounding assumption for standard regression and DiD.
"""

import numpy as np
import pandas as pd


def simulate_brfss_longitudinal(
    n: int = 5000,
    T: int = 4,
    true_effect: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a longitudinal insurance-health DGP with time-varying confounding.

    Parameters
    ----------
    n : int
        Number of individuals.
    T : int
        Number of time periods.
    true_effect : float
        True average causal effect of insurance on health (per period).
    seed : int
        Random seed.

    Returns
    -------
    A : ndarray, shape (n, T)
        Binary treatment (insurance coverage).
    L : ndarray, shape (n, T)
        Time-varying confounder (health status proxy).
    Y : ndarray, shape (n, T)
        Outcome (health score, higher = better).
    U : ndarray, shape (n,)
        Unmeasured baseline health propensity.

    Notes
    -----
    DGP:
        U_i ~ N(0, 1)                         unmeasured propensity
        L_{i,t} = 0.5*U_i + 0.3*A_{i,t-1} + eps_L
        A_{i,t} ~ Bernoulli(sigmoid(0.8*L_{i,t} + 0.5*U_i))
        Y_{i,t} = true_effect*A_{i,t} + 0.4*U_i + 0.2*L_{i,t} + eps_Y

    Because L_t is both a confounder (L -> Y) and affected by prior
    treatment (A_{t-1} -> L_t), conditioning on L in regression creates
    a bad control / treatment-induced confounder problem. G-methods handle
    this correctly.
    """
    rng = np.random.default_rng(seed)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    U = rng.normal(0, 1, n)
    L = np.zeros((n, T))
    A = np.zeros((n, T))
    Y = np.zeros((n, T))

    for t in range(T):
        prior_A = A[:, t - 1] if t > 0 else np.zeros(n)
        L[:, t] = 0.5 * U + 0.3 * prior_A + rng.normal(0, 0.5, n)
        prop = sigmoid(0.8 * L[:, t] + 0.5 * U)
        A[:, t] = rng.binomial(1, prop, n).astype(float)
        Y[:, t] = true_effect * A[:, t] + 0.4 * U + 0.2 * L[:, t] + rng.normal(0, 0.5, n)

    return A, L, Y, U


def to_long_dataframe(A, L, Y, U) -> pd.DataFrame:
    """Convert simulation arrays to a long-format DataFrame."""
    n, T = A.shape
    records = []
    for i in range(n):
        for t in range(T):
            records.append({
                "id": i,
                "t": t,
                "A": A[i, t],
                "L": L[i, t],
                "Y": Y[i, t],
                "U": U[i],  # unmeasured — for diagnostics only
                "A_lag": A[i, t - 1] if t > 0 else np.nan,
                "L_lag": L[i, t - 1] if t > 0 else np.nan,
            })
    return pd.DataFrame(records)

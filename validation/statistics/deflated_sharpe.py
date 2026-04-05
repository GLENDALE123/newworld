"""Deflated Sharpe Ratio: PSR + DSR with multiple-testing correction (AFML Ch. 11)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from validation.report import DeflatedSharpeResult
from validation.statistics.sharpe_utils import (
    sharpe_standard_error,
    probabilistic_sharpe_ratio,
)

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(
    n_trials: int,
    variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> float:
    """Compute the expected maximum Sharpe ratio from n_trials independent trials.

    E[max SR] = sqrt(V) * ((1-gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e)))

    where gamma = Euler-Mascheroni constant, V = variance, N = n_trials.

    Args:
        n_trials: Number of strategy trials attempted.
        variance: Variance of returns.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.
        n_obs: Number of observations.

    Returns:
        Expected maximum Sharpe ratio (non-annualized).
    """
    if n_trials <= 1:
        return 0.0

    gamma = EULER_MASCHERONI
    N = max(n_trials, 2)  # Prevent division by zero

    # Phi_inv(1 - 1/N) -- can fail for very large N, but norm.ppf handles it
    z1 = stats.norm.ppf(1.0 - 1.0 / N)
    z2 = stats.norm.ppf(1.0 - 1.0 / (N * np.e))

    e_max_sr = np.sqrt(variance) * ((1 - gamma) * z1 + gamma * z2)
    return float(e_max_sr)


def compute_deflated_sharpe(
    returns: np.ndarray,
    sharpe_observed: float,
    n_trials: int = 1,
    sharpe_benchmark: float = 0.0,
) -> DeflatedSharpeResult:
    """Compute PSR and DSR for observed Sharpe ratio.

    PSR: probability that observed Sharpe > benchmark.
    DSR: PSR where benchmark = expected max Sharpe from n_trials trials.

    Args:
        returns: Array of periodic returns.
        sharpe_observed: Observed (non-annualized) Sharpe ratio.
        n_trials: Number of strategy trials attempted.
        sharpe_benchmark: PSR benchmark Sharpe (default 0).

    Returns:
        DeflatedSharpeResult with PSR, DSR, and supporting statistics.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    skew = float(stats.skew(returns)) if n >= 3 else 0.0
    kurt = float(stats.kurtosis(returns, fisher=False)) if n >= 4 else 3.0
    variance = float(np.var(returns, ddof=1)) if n > 1 else 0.0

    se = sharpe_standard_error(sharpe_observed, n, skew, kurt)

    # PSR: P(SR > benchmark)
    psr = probabilistic_sharpe_ratio(sharpe_observed, sharpe_benchmark, n, skew, kurt)
    # PSR p-value: probability of observing a SR this high under H0: SR=benchmark
    # We want the *complement*: p-value = 1 - PSR (significance of being above benchmark)
    # But convention: higher PSR = more significant. We use 1-PSR as p-value.
    psr_pvalue = 1.0 - psr

    # DSR: PSR with benchmark = expected max SR
    e_max_sr = expected_max_sharpe(n_trials, variance, skew, kurt, n)
    dsr = probabilistic_sharpe_ratio(sharpe_observed, e_max_sr, n, skew, kurt)
    dsr_pvalue = 1.0 - dsr

    return DeflatedSharpeResult(
        psr_pvalue=psr_pvalue,
        dsr_pvalue=dsr_pvalue,
        expected_max_sharpe=e_max_sr,
        sharpe_std_error=se,
        sharpe_observed=sharpe_observed,
        n_trials=n_trials,
    )

"""Sharpe ratio utilities: annualization, standard error, confidence interval, PSR."""

from __future__ import annotations

import numpy as np
from scipy import stats


def annualized_sharpe(
    returns: np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio from a return series.

    Args:
        returns: Array of periodic returns.
        periods_per_year: Number of periods per year (252 for daily, 52 for weekly, etc.)

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret < 1e-15:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(periods_per_year))


def sharpe_standard_error(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute standard error of the Sharpe ratio (per AFML formula).

    SE(SR) = sqrt((1 - skew*SR + (kurtosis-1)/4 * SR^2) / (T-1))

    Args:
        sharpe: Observed (non-annualized) Sharpe ratio.
        n: Number of return observations.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns (excess kurtosis + 3; use raw kurtosis here).

    Returns:
        Standard error of the Sharpe ratio.
    """
    if n <= 1:
        return float("inf")
    numerator = 1.0 - skew * sharpe + (kurtosis - 1) / 4.0 * sharpe**2
    # Guard against negative numerator from extreme skew/kurtosis
    numerator = max(numerator, 0.0)
    return float(np.sqrt(numerator / (n - 1)))


def sharpe_confidence_interval(
    sharpe: float,
    n: int,
    confidence: float = 0.95,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> tuple[float, float]:
    """Compute confidence interval for the Sharpe ratio.

    Args:
        sharpe: Observed (non-annualized) Sharpe ratio.
        n: Number of return observations.
        confidence: Confidence level (default 0.95).
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    se = sharpe_standard_error(sharpe, n, skew, kurtosis)
    z = stats.norm.ppf((1 + confidence) / 2)
    return (sharpe - z * se, sharpe + z * se)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Probabilistic Sharpe Ratio (PSR).

    PSR = Phi((SR_observed - SR_benchmark) / SE(SR))

    Returns a probability in [0, 1]. Values > 0.95 indicate the observed
    Sharpe is significantly greater than the benchmark at 5% level.

    Args:
        observed_sharpe: Observed (non-annualized) Sharpe ratio.
        benchmark_sharpe: Benchmark Sharpe ratio (typically 0).
        n: Number of return observations.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.

    Returns:
        PSR value (probability).
    """
    se = sharpe_standard_error(observed_sharpe, n, skew, kurtosis)
    if se == 0 or se == float("inf"):
        return 0.5
    z = (observed_sharpe - benchmark_sharpe) / se
    return float(stats.norm.cdf(z))

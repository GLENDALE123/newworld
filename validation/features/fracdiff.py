"""Fractional Differentiation with fixed-width window (AFML Ch. 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from validation.report import FracDiffResult


def _compute_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute fractional differentiation weights using recursive formula.

    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

    Weights are truncated when abs(w_k) < threshold.

    Args:
        d: Fractional differentiation order.
        threshold: Cutoff for weight magnitude.

    Returns:
        Array of weights [w_0, w_1, ..., w_K].
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        if k > 10000:  # safety limit
            break
    return np.array(weights)


def fracdiff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fractional differentiation to a time series.

    x_t^d = sum(w_k * x_{t-k}) for k=0..window

    Args:
        series: Input time series.
        d: Differentiation order (0 < d < 1 for fractional).
        threshold: Weight truncation threshold.

    Returns:
        Fractionally differenced series. First (window-1) values are NaN.
    """
    weights = _compute_weights(d, threshold)
    window = len(weights)
    values = series.values.astype(np.float64)

    # Vectorized convolution instead of python loop
    conv = np.convolve(values, weights, mode="full")[:len(values)]
    conv[:window - 1] = np.nan

    return pd.Series(conv, index=series.index, name=series.name)


def find_optimal_d(
    series: pd.Series,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.05,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
) -> tuple[float, float, pd.Series]:
    """Find the minimum d that makes the series stationary (ADF test).

    Searches d from d_min to d_max in steps of d_step.
    Returns the smallest d where ADF p-value < adf_pvalue.

    Args:
        series: Input time series.
        d_min: Minimum d to search.
        d_max: Maximum d to search.
        d_step: Step size for d search.
        threshold: Weight truncation threshold.
        adf_pvalue: Target ADF p-value for stationarity.

    Returns:
        Tuple of (optimal_d, adf_pvalue_at_d, differenced_series).
    """
    d_values = np.arange(d_min, d_max + d_step / 2, d_step)

    for d in d_values:
        if d == 0.0:
            # d=0 means no differencing; test original series
            diffed = series.copy()
        else:
            diffed = fracdiff(series, d, threshold)

        # Drop NaN values for ADF test
        clean = diffed.dropna()
        if len(clean) < 20:
            continue

        try:
            adf_result = adfuller(clean, maxlag=1, autolag=None)
            pval = adf_result[1]
        except Exception:
            continue

        if pval < adf_pvalue:
            return (float(d), float(pval), diffed)

    # If no d found, return d=1.0 (full differencing)
    diffed = fracdiff(series, 1.0, threshold)
    clean = diffed.dropna()
    try:
        adf_result = adfuller(clean, maxlag=1, autolag=None)
        pval = float(adf_result[1])
    except Exception:
        pval = 1.0
    return (1.0, pval, diffed)


def analyze_features(
    features_df: pd.DataFrame,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.05,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
) -> FracDiffResult:
    """Find optimal fractional differentiation d for each feature column.

    Args:
        features_df: DataFrame with feature columns.
        d_min, d_max, d_step: Search range for d.
        threshold: Weight truncation threshold.
        adf_pvalue: Target ADF p-value.

    Returns:
        FracDiffResult with per-feature d values and ADF p-values.
    """
    feature_d_values = {}
    feature_adf_pvalues = {}

    for col in features_df.columns:
        series = features_df[col].dropna()
        if len(series) < 30:
            feature_d_values[col] = 1.0
            feature_adf_pvalues[col] = 1.0
            continue

        d, pval, _ = find_optimal_d(series, d_min, d_max, d_step, threshold, adf_pvalue)
        feature_d_values[col] = d
        feature_adf_pvalues[col] = pval

    return FracDiffResult(
        feature_d_values=feature_d_values,
        feature_adf_pvalues=feature_adf_pvalues,
    )

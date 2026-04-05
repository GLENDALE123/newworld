"""Sample weights based on label concurrency and uniqueness (AFML Ch. 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_indicator_matrix(
    tbm_timestamps: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a binary indicator matrix: rows=timestamps, cols=label indices.

    indicator[t, i] = 1 if label i is active at time t (t_start_i <= t <= t_end_i).
    """
    indicator = pd.DataFrame(0, index=price_index, columns=tbm_timestamps.index)
    for i, row in tbm_timestamps.iterrows():
        mask = (indicator.index >= row["t_start"]) & (indicator.index <= row["t_end"])
        indicator.loc[mask, i] = 1
    return indicator


def _compute_concurrency_fast(
    tbm_timestamps: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.Series:
    """Compute concurrency without building full indicator matrix.

    Uses a sweep-line approach: O(n log n) instead of O(n^2).
    """
    ts_values = price_index.values
    starts = tbm_timestamps["t_start"].values
    ends = tbm_timestamps["t_end"].values

    # For each timestamp, count how many labels are active
    # A label i is active at time t if starts[i] <= t <= ends[i]
    # Use searchsorted for vectorized counting
    starts_sorted = np.sort(starts)
    ends_sorted = np.sort(ends)

    # At time t: active labels = (labels that started <= t) - (labels that ended < t)
    started = np.searchsorted(starts_sorted, ts_values, side="right")
    ended = np.searchsorted(ends_sorted, ts_values, side="left")
    concurrency = started - ended

    return pd.Series(concurrency, index=price_index, dtype=np.int64)


def _compute_uniqueness_fast(
    tbm_timestamps: pd.DataFrame,
    concurrency: pd.Series,
) -> pd.Series:
    """Compute average uniqueness per label using vectorized operations."""
    price_index = concurrency.index
    inv_conc = (1.0 / concurrency.replace(0, np.inf)).values
    ts_values = price_index.values

    starts = tbm_timestamps["t_start"].values
    ends = tbm_timestamps["t_end"].values

    uniqueness = np.empty(len(tbm_timestamps), dtype=np.float64)
    for i in range(len(tbm_timestamps)):
        mask = (ts_values >= starts[i]) & (ts_values <= ends[i])
        active = inv_conc[mask]
        uniqueness[i] = active.mean() if len(active) > 0 else 0.0

    return pd.Series(uniqueness, index=tbm_timestamps.index)


def compute_concurrency(indicator_matrix: pd.DataFrame) -> pd.Series:
    """Compute concurrency at each time step from indicator matrix."""
    return indicator_matrix.sum(axis=1)


def compute_uniqueness(indicator_matrix: pd.DataFrame) -> pd.Series:
    """Compute average uniqueness per label from indicator matrix."""
    concurrency = indicator_matrix.sum(axis=1)
    inv_concurrency = 1.0 / concurrency.replace(0, np.inf)

    uniqueness = pd.Series(index=indicator_matrix.columns, dtype=np.float64)
    for col in indicator_matrix.columns:
        active_mask = indicator_matrix[col] > 0
        if active_mask.any():
            uniqueness[col] = inv_concurrency[active_mask].mean()
        else:
            uniqueness[col] = 0.0
    return uniqueness


def compute_sample_weights(
    close_prices: pd.Series,
    tbm_timestamps: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute sample weights from concurrency, uniqueness, and returns.

    weight_i = uniqueness_i * |return_i|

    Uses fast vectorized computation (no full indicator matrix).
    """
    price_index = close_prices.index

    # Fast concurrency (sweep-line, O(n log n))
    concurrency = _compute_concurrency_fast(tbm_timestamps, price_index)

    # Fast uniqueness (vectorized loop)
    uniqueness = _compute_uniqueness_fast(tbm_timestamps, concurrency)

    # Vectorized absolute returns
    starts = tbm_timestamps["t_start"].values
    ends = tbm_timestamps["t_end"].values
    close_vals = close_prices.values
    ts_vals = price_index.values

    start_idx = np.searchsorted(ts_vals, starts)
    end_idx = np.searchsorted(ts_vals, ends)

    # Clip indices to valid range
    start_idx = np.clip(start_idx, 0, len(close_vals) - 1)
    end_idx = np.clip(end_idx, 0, len(close_vals) - 1)

    start_prices = close_vals[start_idx]
    end_prices = close_vals[end_idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        abs_returns = np.abs((end_prices - start_prices) / start_prices)
    abs_returns = np.nan_to_num(abs_returns, nan=0.0, posinf=0.0, neginf=0.0)
    abs_returns = pd.Series(abs_returns, index=tbm_timestamps.index)

    sample_weights = uniqueness * abs_returns

    # Normalize weights so they sum to len(labels)
    if sample_weights.sum() > 0:
        sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

    return concurrency, uniqueness, sample_weights

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

    Args:
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns, indexed by label index.
        price_index: DatetimeIndex of all price timestamps.

    Returns:
        DataFrame of shape (len(price_index), len(tbm_timestamps)) with 0/1 values.
    """
    indicator = pd.DataFrame(0, index=price_index, columns=tbm_timestamps.index)
    for i, row in tbm_timestamps.iterrows():
        t_start = row["t_start"]
        t_end = row["t_end"]
        mask = (indicator.index >= t_start) & (indicator.index <= t_end)
        indicator.loc[mask, i] = 1
    return indicator


def compute_concurrency(indicator_matrix: pd.DataFrame) -> pd.Series:
    """Compute concurrency at each time step: number of active labels.

    Args:
        indicator_matrix: Binary indicator matrix (rows=timestamps, cols=labels).

    Returns:
        Series indexed by timestamps with integer concurrency counts.
    """
    return indicator_matrix.sum(axis=1)


def compute_uniqueness(indicator_matrix: pd.DataFrame) -> pd.Series:
    """Compute average uniqueness per label.

    uniqueness_i = mean(1/concurrency_t) for all t where label i is active.

    Args:
        indicator_matrix: Binary indicator matrix (rows=timestamps, cols=labels).

    Returns:
        Series indexed by label index with uniqueness values in (0, 1].
    """
    concurrency = indicator_matrix.sum(axis=1)
    # Avoid division by zero: timestamps with no active labels get inf, but
    # they won't appear in any label's average since indicator is 0 there.
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

    where return_i = (close[t_end_i] - close[t_start_i]) / close[t_start_i]

    Args:
        close_prices: Series of close prices with DatetimeIndex.
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns.

    Returns:
        Tuple of (concurrency, uniqueness, sample_weights) Series.
    """
    indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
    concurrency = compute_concurrency(indicator)
    uniqueness = compute_uniqueness(indicator)

    # Compute absolute returns per label
    abs_returns = pd.Series(index=tbm_timestamps.index, dtype=np.float64)
    for i, row in tbm_timestamps.iterrows():
        t_start = row["t_start"]
        t_end = row["t_end"]
        if t_start in close_prices.index and t_end in close_prices.index:
            ret = (close_prices[t_end] - close_prices[t_start]) / close_prices[t_start]
            abs_returns[i] = abs(ret)
        else:
            abs_returns[i] = 0.0

    sample_weights = uniqueness * abs_returns

    # Normalize weights so they sum to len(labels)
    if sample_weights.sum() > 0:
        sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

    return concurrency, uniqueness, sample_weights

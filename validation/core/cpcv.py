"""Combinatorial Purged Cross-Validation (AFML Ch. 12)."""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import numpy as np
import pandas as pd

from validation.core.purged_kfold import purge_train_indices, apply_embargo
from validation.report import CPCVResult
from validation.statistics.sharpe_utils import annualized_sharpe


class ModelProtocol(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _split_into_groups(
    n_samples: int,
    n_groups: int,
) -> list[np.ndarray]:
    """Split sample indices into n_groups contiguous blocks.

    Returns:
        List of arrays, each containing indices for one group.
    """
    group_size = n_samples // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))
    return groups


def cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_groups: int = 6,
    k_test_groups: int = 2,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
) -> CPCVResult:
    """Run Combinatorial Purged Cross-Validation.

    Algorithm:
    1. Split data into n_groups contiguous blocks.
    2. Generate C(n_groups, k_test_groups) combinations.
    3. For each combination, k_test_groups = test, rest = train (with purge+embargo).
    4. Predict on test groups, concatenate in time order to form a backtest path.
    5. Compute Sharpe for each path.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with train()/predict() interface.
        tbm_timestamps: DataFrame with 't_start'/'t_end' columns.
        sample_weight: Optional sample weights.
        n_groups: Number of groups to split data into.
        k_test_groups: Number of groups used as test per combination.
        embargo_pct: Embargo fraction.
        periods_per_year: For Sharpe annualization.

    Returns:
        CPCVResult with path-level Sharpe distribution.
    """
    n_samples = len(X)
    groups = _split_into_groups(n_samples, n_groups)
    combos = list(combinations(range(n_groups), k_test_groups))

    path_sharpes = []

    for combo in combos:
        test_groups = sorted(combo)
        train_groups = [g for g in range(n_groups) if g not in test_groups]

        test_idx = np.concatenate([groups[g] for g in test_groups])
        train_idx = np.concatenate([groups[g] for g in train_groups])

        # Purge + embargo for each test group separately
        purged_train = train_idx.copy()
        for tg in test_groups:
            tg_indices = groups[tg]
            purged_train = purge_train_indices(
                purged_train, tg_indices, tbm_timestamps, X.index
            )
            purged_train = apply_embargo(purged_train, tg_indices, n_samples, embargo_pct)

        if len(purged_train) == 0:
            continue

        X_train = X.iloc[purged_train]
        y_train = y.iloc[purged_train]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        sw_train = sample_weight.iloc[purged_train] if sample_weight is not None else None

        model.train(X_train, y_train, sample_weight=sw_train)
        predictions = model.predict(X_test)

        # Compute path "returns": +1 for correct, -1 for incorrect
        signed_returns = (predictions == y_test.values).astype(float) * 2 - 1
        sr = annualized_sharpe(signed_returns, periods_per_year)
        path_sharpes.append(sr)

    if len(path_sharpes) == 0:
        return CPCVResult(
            path_sharpes=[], mean_sharpe=0.0, std_sharpe=0.0,
            median_sharpe=0.0, pct_negative=0.0,
        )

    sharpes_arr = np.array(path_sharpes)
    return CPCVResult(
        path_sharpes=path_sharpes,
        mean_sharpe=float(np.mean(sharpes_arr)),
        std_sharpe=float(np.std(sharpes_arr, ddof=1)) if len(sharpes_arr) > 1 else 0.0,
        median_sharpe=float(np.median(sharpes_arr)),
        pct_negative=float(np.mean(sharpes_arr < 0)),
    )

"""Purged K-Fold Cross-Validation with embargo (AFML Ch. 7)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from validation.report import PurgedCVResult
from validation.statistics.sharpe_utils import annualized_sharpe


class ModelProtocol(Protocol):
    """Minimal model interface required by validation."""
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _get_train_test_indices(
    n_samples: int,
    n_splits: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train/test for a given fold.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    fold_size = n_samples // n_splits
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size if fold < n_splits - 1 else n_samples
    test_indices = np.arange(test_start, test_end)
    train_indices = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])
    return train_indices, test_indices


def purge_train_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    tbm_timestamps: pd.DataFrame,
    all_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """Remove training samples whose label time-span overlaps with any test sample.

    Args:
        train_indices: Array of training sample positions.
        test_indices: Array of test sample positions.
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns.
        all_timestamps: Full DatetimeIndex of all samples.

    Returns:
        Purged training indices.
    """
    test_positions = test_indices
    test_t_starts = tbm_timestamps.iloc[test_positions]["t_start"]
    test_t_ends = tbm_timestamps.iloc[test_positions]["t_end"]
    test_min = test_t_starts.min()
    test_max = test_t_ends.max()

    purged = []
    for idx in train_indices:
        train_start = tbm_timestamps.iloc[idx]["t_start"]
        train_end = tbm_timestamps.iloc[idx]["t_end"]
        # Overlap: train label span intersects test span
        if train_end < test_min or train_start > test_max:
            purged.append(idx)
        # else: overlaps with test -> purge (don't include)
    return np.array(purged, dtype=int)


def apply_embargo(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
    embargo_pct: float,
) -> np.ndarray:
    """Remove training samples in the embargo zone right after the test set.

    Args:
        train_indices: Array of training sample positions (already purged).
        test_indices: Array of test sample positions.
        n_samples: Total number of samples.
        embargo_pct: Fraction of total samples to embargo.

    Returns:
        Training indices with embargo applied.
    """
    embargo_size = int(n_samples * embargo_pct)
    if embargo_size == 0:
        return train_indices
    test_end = test_indices[-1]
    embargo_end = test_end + embargo_size
    return train_indices[~((train_indices > test_end) & (train_indices <= embargo_end))]


def purged_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
) -> PurgedCVResult:
    """Run Purged K-Fold cross-validation.

    For each fold:
    1. Split into train/test by contiguous time blocks.
    2. Purge: remove train samples overlapping with test label spans.
    3. Embargo: remove train samples right after test end.
    4. Train model, predict on test, compute Sharpe and accuracy.

    Args:
        X: Feature DataFrame.
        y: Label Series (+1/-1).
        model: Model with train() and predict() methods.
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns.
        sample_weight: Optional sample weights for training.
        n_splits: Number of folds.
        embargo_pct: Embargo fraction.
        periods_per_year: For Sharpe annualization.

    Returns:
        PurgedCVResult with fold-level and mean statistics.
    """
    n_samples = len(X)
    fold_sharpes = []
    fold_accuracies = []

    for fold in range(n_splits):
        train_idx, test_idx = _get_train_test_indices(n_samples, n_splits, fold)

        # Purge overlapping samples
        train_idx = purge_train_indices(
            train_idx, test_idx, tbm_timestamps, X.index
        )

        # Apply embargo
        train_idx = apply_embargo(train_idx, test_idx, n_samples, embargo_pct)

        if len(train_idx) == 0:
            continue

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        sw_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        model.train(X_train, y_train, sample_weight=sw_train)
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = float(np.mean(predictions == y_test.values))
        fold_accuracies.append(accuracy)

        # Sharpe from predictions: use predicted direction * actual direction as "returns"
        # This is a simple proxy: correct predictions yield +1, incorrect -1
        signed_returns = (predictions == y_test.values).astype(float) * 2 - 1
        sr = annualized_sharpe(signed_returns, periods_per_year)
        fold_sharpes.append(sr)

    if len(fold_sharpes) == 0:
        return PurgedCVResult(
            fold_sharpes=[], fold_accuracies=[],
            mean_sharpe=0.0, std_sharpe=0.0,
        )

    return PurgedCVResult(
        fold_sharpes=fold_sharpes,
        fold_accuracies=fold_accuracies,
        mean_sharpe=float(np.mean(fold_sharpes)),
        std_sharpe=float(np.std(fold_sharpes, ddof=1)) if len(fold_sharpes) > 1 else 0.0,
    )

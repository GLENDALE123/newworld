"""Feature importance: MDI, MDA, SFI with overfit detection (AFML Ch. 8)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from validation.core.purged_kfold import (
    _get_train_test_indices,
    purge_train_indices,
    apply_embargo,
)
from validation.report import FeatureImportanceResult
from validation.statistics.sharpe_utils import annualized_sharpe


class ModelProtocol(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class ImportanceModelProtocol(ModelProtocol, Protocol):
    """Model that also supports feature importance (e.g., CatBoost)."""
    def get_feature_importance(self) -> np.ndarray: ...


def compute_mdi(
    X: pd.DataFrame,
    y: pd.Series,
    model: ImportanceModelProtocol,
    sample_weight: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute Mean Decrease Impurity (MDI) feature importance.

    Trains the model on the full dataset and extracts built-in feature importance.
    This is an in-sample metric.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with get_feature_importance() method.
        sample_weight: Optional sample weights.

    Returns:
        DataFrame with columns ['feature', 'importance'], sorted descending.
    """
    model.train(X, y, sample_weight=sample_weight)
    importances = model.get_feature_importance()

    # Normalize to sum to 1
    total = np.sum(importances)
    if total > 0:
        importances = importances / total

    result = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": importances,
    })
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_mda(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute Mean Decrease Accuracy (MDA) feature importance via Purged K-Fold.

    For each fold and each feature:
    1. Compute baseline accuracy on test set.
    2. Shuffle the feature column in the test set.
    3. Compute accuracy with shuffled feature.
    4. Decrease = baseline - shuffled accuracy.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with train()/predict().
        tbm_timestamps: DataFrame with 't_start'/'t_end'.
        sample_weight: Optional sample weights.
        n_splits: Number of CV folds.
        embargo_pct: Embargo fraction.
        random_state: Random seed for shuffling.

    Returns:
        DataFrame with columns ['feature', 'mean_decrease', 'std_decrease'], sorted descending.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    features = X.columns.tolist()

    # Collect per-fold per-feature decrease
    decreases = {feat: [] for feat in features}

    for fold in range(n_splits):
        train_idx, test_idx = _get_train_test_indices(n_samples, n_splits, fold)
        train_idx = purge_train_indices(train_idx, test_idx, tbm_timestamps, X.index)
        train_idx = apply_embargo(train_idx, test_idx, n_samples, embargo_pct)

        if len(train_idx) == 0:
            continue

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx]

        sw = sample_weight.iloc[train_idx] if sample_weight is not None else None
        model.train(X_train, y_train, sample_weight=sw)

        # Baseline accuracy
        baseline_preds = model.predict(X_test)
        baseline_acc = float(np.mean(baseline_preds == y_test.values))

        # Shuffle each feature and measure accuracy decrease
        for feat in features:
            X_test_shuffled = X_test.copy()
            X_test_shuffled[feat] = rng.permutation(X_test_shuffled[feat].values)

            shuffled_preds = model.predict(X_test_shuffled)
            shuffled_acc = float(np.mean(shuffled_preds == y_test.values))

            decreases[feat].append(baseline_acc - shuffled_acc)

    result = pd.DataFrame({
        "feature": features,
        "mean_decrease": [np.mean(decreases[f]) if decreases[f] else 0.0 for f in features],
        "std_decrease": [np.std(decreases[f], ddof=1) if len(decreases[f]) > 1 else 0.0 for f in features],
    })
    return result.sort_values("mean_decrease", ascending=False).reset_index(drop=True)


def compute_sfi(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compute Single Feature Importance (SFI).

    For each feature, train a model using only that feature with Purged K-Fold CV
    and compute the mean Sharpe ratio.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with train()/predict().
        tbm_timestamps: DataFrame with 't_start'/'t_end'.
        sample_weight: Optional sample weights.
        n_splits: Number of CV folds.
        embargo_pct: Embargo fraction.
        periods_per_year: For Sharpe annualization.

    Returns:
        DataFrame with columns ['feature', 'sharpe'], sorted descending.
    """
    n_samples = len(X)
    features = X.columns.tolist()
    sfi_results = []

    for feat in features:
        X_single = X[[feat]]
        fold_sharpes = []

        for fold in range(n_splits):
            train_idx, test_idx = _get_train_test_indices(n_samples, n_splits, fold)
            train_idx = purge_train_indices(train_idx, test_idx, tbm_timestamps, X.index)
            train_idx = apply_embargo(train_idx, test_idx, n_samples, embargo_pct)

            if len(train_idx) == 0:
                continue

            X_train = X_single.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_single.iloc[test_idx]
            y_test = y.iloc[test_idx]

            sw = sample_weight.iloc[train_idx] if sample_weight is not None else None
            model.train(X_train, y_train, sample_weight=sw)
            predictions = model.predict(X_test)

            signed_returns = (predictions == y_test.values).astype(float) * 2 - 1
            sr = annualized_sharpe(signed_returns, periods_per_year)
            fold_sharpes.append(sr)

        mean_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        sfi_results.append({"feature": feat, "sharpe": mean_sharpe})

    result = pd.DataFrame(sfi_results)
    return result.sort_values("sharpe", ascending=False).reset_index(drop=True)


def detect_overfit_features(
    mdi_ranking: pd.DataFrame,
    mda_ranking: pd.DataFrame,
    sfi_ranking: pd.DataFrame,
) -> list[str]:
    """Detect overfit features using cross-referencing of MDI, MDA, SFI.

    Overfit criteria (from spec):
    - MDI top 30% but MDA bottom 50% -> overfit suspect
    - MDA std > MDA mean -> unstable
    - SFI Sharpe < 0 -> no standalone predictive power

    A feature flagged by 2+ criteria is classified as overfit.

    Args:
        mdi_ranking: DataFrame with 'feature', 'importance'.
        mda_ranking: DataFrame with 'feature', 'mean_decrease', 'std_decrease'.
        sfi_ranking: DataFrame with 'feature', 'sharpe'.

    Returns:
        List of overfit feature names.
    """
    features = mdi_ranking["feature"].tolist()
    n_features = len(features)

    # MDI top 30%
    mdi_top_n = max(1, int(n_features * 0.3))
    mdi_top_set = set(mdi_ranking.head(mdi_top_n)["feature"].tolist())

    # MDA bottom 50%
    mda_sorted = mda_ranking.sort_values("mean_decrease", ascending=True)
    mda_bottom_n = max(1, int(n_features * 0.5))
    mda_bottom_set = set(mda_sorted.head(mda_bottom_n)["feature"].tolist())

    # Build per-feature lookup
    mda_lookup = dict(zip(mda_ranking["feature"], zip(mda_ranking["mean_decrease"], mda_ranking["std_decrease"])))
    sfi_lookup = dict(zip(sfi_ranking["feature"], sfi_ranking["sharpe"]))

    overfit = []
    for feat in features:
        flags = 0

        # Criterion 1: MDI top 30% AND MDA bottom 50%
        if feat in mdi_top_set and feat in mda_bottom_set:
            flags += 1

        # Criterion 2: MDA std > MDA mean (unstable)
        if feat in mda_lookup:
            mean_dec, std_dec = mda_lookup[feat]
            if std_dec > abs(mean_dec) and abs(mean_dec) > 0:
                flags += 1

        # Criterion 3: SFI Sharpe < 0
        if feat in sfi_lookup and sfi_lookup[feat] < 0:
            flags += 1

        if flags >= 2:
            overfit.append(feat)

    return overfit


def compute_all_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model: ImportanceModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
    random_state: int = 42,
) -> FeatureImportanceResult:
    """Run all feature importance methods and detect overfit features.

    Args:
        X, y, model, tbm_timestamps: Standard inputs.
        sample_weight: Optional sample weights.
        n_splits, embargo_pct, periods_per_year: CV parameters.
        random_state: For MDA shuffling.

    Returns:
        FeatureImportanceResult with MDI, MDA, SFI rankings and overfit detection.
    """
    mdi = compute_mdi(X, y, model, sample_weight)
    mda = compute_mda(X, y, model, tbm_timestamps, sample_weight, n_splits, embargo_pct, random_state)
    sfi = compute_sfi(X, y, model, tbm_timestamps, sample_weight, n_splits, embargo_pct, periods_per_year)

    overfit_features = detect_overfit_features(mdi, mda, sfi)
    n_features = len(X.columns)
    overfit_ratio = len(overfit_features) / n_features if n_features > 0 else 0.0

    return FeatureImportanceResult(
        mdi_ranking=mdi,
        mda_ranking=mda,
        sfi_ranking=sfi,
        overfit_features=overfit_features,
        overfit_ratio=overfit_ratio,
    )

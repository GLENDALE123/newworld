import numpy as np
import pandas as pd
import pytest

from validation.features.importance import (
    compute_mdi,
    compute_mda,
    compute_sfi,
    detect_overfit_features,
    compute_all_importance,
)


class MockImportanceModel:
    """Model with feature importance support for testing."""

    def __init__(self):
        self._majority = 1
        self._n_features = 0
        self._importances = None

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        self._n_features = X.shape[1]
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])
        # Fake importances: first feature is most "important"
        self._importances = np.arange(self._n_features, 0, -1, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._majority)

    def get_feature_importance(self) -> np.ndarray:
        if self._importances is None:
            return np.ones(self._n_features)
        return self._importances


class RandomModel:
    """Model that predicts randomly, to simulate low-quality features."""

    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)

    def train(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        return self._rng.choice([1, -1], len(X))

    def get_feature_importance(self):
        return np.ones(3)  # dummy


def _make_importance_data(n: int = 100, n_features: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    X = pd.DataFrame(
        {f"feat_{i}": rng.standard_normal(n) for i in range(n_features)},
        index=idx,
    )
    y = pd.Series(rng.choice([1, -1], n), index=idx)
    holding = 3
    t_ends = pd.DatetimeIndex([idx[min(i + holding, n - 1)] for i in range(n)])
    tbm = pd.DataFrame({"t_start": idx, "t_end": t_ends}, index=range(n))
    return X, y, tbm


class TestComputeMDI:
    def test_returns_all_features(self):
        X, y, _ = _make_importance_data(n_features=4)
        model = MockImportanceModel()
        result = compute_mdi(X, y, model)
        assert len(result) == 4
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_sums_to_one(self):
        X, y, _ = _make_importance_data(n_features=4)
        model = MockImportanceModel()
        result = compute_mdi(X, y, model)
        assert abs(result["importance"].sum() - 1.0) < 1e-10

    def test_sorted_descending(self):
        X, y, _ = _make_importance_data(n_features=4)
        model = MockImportanceModel()
        result = compute_mdi(X, y, model)
        importances = result["importance"].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))


class TestComputeMDA:
    def test_returns_all_features(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_mda(X, y, model, tbm, n_splits=3)
        assert len(result) == 3
        assert "mean_decrease" in result.columns
        assert "std_decrease" in result.columns

    def test_mda_values_are_finite(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_mda(X, y, model, tbm, n_splits=3)
        assert all(np.isfinite(result["mean_decrease"]))


class TestComputeSFI:
    def test_returns_all_features(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_sfi(X, y, model, tbm, n_splits=3)
        assert len(result) == 3
        assert "sharpe" in result.columns

    def test_sharpe_values_are_finite(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_sfi(X, y, model, tbm, n_splits=3)
        assert all(np.isfinite(result["sharpe"]))


class TestDetectOverfitFeatures:
    def test_detects_overfit(self):
        """Feature with high MDI but low MDA and negative SFI should be flagged."""
        mdi = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4", "f5"],
            "importance": [0.5, 0.2, 0.15, 0.1, 0.05],
        })
        mda = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4", "f5"],
            "mean_decrease": [0.001, 0.05, 0.03, 0.02, 0.01],  # f1: low MDA despite high MDI
            "std_decrease": [0.01, 0.01, 0.01, 0.01, 0.01],
        })
        sfi = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4", "f5"],
            "sharpe": [-0.5, 0.3, 0.2, 0.1, -0.1],  # f1: negative SFI
        })
        overfit = detect_overfit_features(mdi, mda, sfi)
        # f1: MDI top 30% (top 1 of 5), MDA bottom 50% (rank 5/5), SFI < 0 -> 3 flags
        assert "f1" in overfit

    def test_good_features_not_flagged(self):
        """Consistent features should not be flagged."""
        mdi = pd.DataFrame({
            "feature": ["f1", "f2"],
            "importance": [0.6, 0.4],
        })
        mda = pd.DataFrame({
            "feature": ["f1", "f2"],
            "mean_decrease": [0.05, 0.03],
            "std_decrease": [0.01, 0.01],
        })
        sfi = pd.DataFrame({
            "feature": ["f1", "f2"],
            "sharpe": [0.5, 0.3],
        })
        overfit = detect_overfit_features(mdi, mda, sfi)
        assert len(overfit) == 0

    def test_unstable_mda_flagged(self):
        """Feature with MDA std > MDA mean AND another flag should be overfit."""
        mdi = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "importance": [0.5, 0.3, 0.2],
        })
        mda = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "mean_decrease": [0.01, 0.05, 0.03],  # f1: low MDA
            "std_decrease": [0.05, 0.01, 0.01],    # f1: high std (unstable)
        })
        sfi = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "sharpe": [-0.2, 0.4, 0.2],  # f1: negative SFI
        })
        overfit = detect_overfit_features(mdi, mda, sfi)
        assert "f1" in overfit  # unstable + negative SFI = 2 flags (+ maybe MDI/MDA mismatch = 3)


class TestComputeAllImportance:
    def test_full_pipeline(self):
        X, y, tbm = _make_importance_data(n_features=4, n=100)
        model = MockImportanceModel()
        result = compute_all_importance(X, y, model, tbm, n_splits=3)
        assert len(result.mdi_ranking) == 4
        assert len(result.mda_ranking) == 4
        assert len(result.sfi_ranking) == 4
        assert isinstance(result.overfit_features, list)
        assert 0.0 <= result.overfit_ratio <= 1.0

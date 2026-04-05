import numpy as np
import pandas as pd
import pytest

from validation.core.purged_kfold import (
    _get_train_test_indices,
    purge_train_indices,
    apply_embargo,
    purged_kfold_cv,
)


class DummyModel:
    """Simple model that memorizes training data for testing."""

    def __init__(self):
        self._majority = 1

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        # Majority vote
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._majority)


class PerfectModel:
    """Model that always predicts correctly (cheats for testing)."""

    def __init__(self):
        self._y = None

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Return alternating +1/-1 (this won't be "perfect" but is deterministic)
        return np.array([1] * len(X))


def _make_cv_data(n: int = 100, seed: int = 42):
    """Create synthetic data for CV testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    X = pd.DataFrame(
        {"feat1": rng.standard_normal(n), "feat2": rng.standard_normal(n)},
        index=idx,
    )
    y = pd.Series(rng.choice([1, -1], n), index=idx)

    # TBM timestamps: each label spans 3 bars forward
    holding = 3
    t_starts = idx
    t_ends = pd.DatetimeIndex([
        idx[min(i + holding, n - 1)] for i in range(n)
    ])
    tbm_timestamps = pd.DataFrame(
        {"t_start": t_starts, "t_end": t_ends},
        index=range(n),
    )

    return X, y, tbm_timestamps


class TestGetTrainTestIndices:
    def test_fold_0(self):
        train, test = _get_train_test_indices(100, 5, 0)
        assert len(test) == 20
        assert len(train) == 80
        assert test[0] == 0
        assert test[-1] == 19

    def test_last_fold(self):
        train, test = _get_train_test_indices(100, 5, 4)
        assert test[-1] == 99
        assert 0 in train

    def test_no_overlap(self):
        for fold in range(5):
            train, test = _get_train_test_indices(100, 5, fold)
            assert len(set(train) & set(test)) == 0


class TestPurgeTrainIndices:
    def test_overlapping_samples_removed(self):
        """Train samples overlapping with test labels should be purged."""
        X, y, tbm = _make_cv_data(n=50)
        train_idx = np.arange(0, 40)
        test_idx = np.arange(37, 50)  # test starts at 37

        purged = purge_train_indices(train_idx, test_idx, tbm, X.index)

        # Train samples near 37 whose label extends into test should be purged
        # Label at index 35 spans 35..38, which overlaps with test starting at 37
        # So indices 35, 36 (at least) should be purged
        # But index 34 spans 34..37 which also overlaps with test_min=idx[37]
        assert len(purged) < len(train_idx)
        for idx_val in purged:
            t_end = tbm.iloc[idx_val]["t_end"]
            t_start = tbm.iloc[idx_val]["t_start"]
            test_min = tbm.iloc[test_idx[0]]["t_start"]
            test_max = tbm.iloc[test_idx[-1]]["t_end"]
            assert t_end < test_min or t_start > test_max


class TestApplyEmbargo:
    def test_embargo_removes_samples(self):
        train_idx = np.array([0, 1, 2, 20, 21, 22, 23, 24])
        test_idx = np.array([18, 19])
        # embargo 5% of 100 = 5 samples after test_end=19 -> remove 20..24
        result = apply_embargo(train_idx, test_idx, 100, 0.05)
        assert 20 not in result
        assert 24 not in result
        assert 0 in result
        assert 1 in result

    def test_zero_embargo(self):
        train_idx = np.array([0, 1, 2, 20, 21])
        test_idx = np.array([10, 11])
        result = apply_embargo(train_idx, test_idx, 100, 0.0)
        np.testing.assert_array_equal(result, train_idx)


class TestPurgedKFoldCV:
    def test_returns_correct_number_of_folds(self):
        X, y, tbm = _make_cv_data(n=100)
        model = DummyModel()
        result = purged_kfold_cv(X, y, model, tbm, n_splits=5, embargo_pct=0.01)
        assert len(result.fold_sharpes) == 5
        assert len(result.fold_accuracies) == 5

    def test_mean_sharpe_is_average(self):
        X, y, tbm = _make_cv_data(n=100)
        model = DummyModel()
        result = purged_kfold_cv(X, y, model, tbm, n_splits=5)
        expected_mean = np.mean(result.fold_sharpes)
        assert abs(result.mean_sharpe - expected_mean) < 1e-10

    def test_with_sample_weights(self):
        X, y, tbm = _make_cv_data(n=100)
        model = DummyModel()
        weights = pd.Series(np.ones(100), index=X.index)
        result = purged_kfold_cv(X, y, model, tbm, sample_weight=weights, n_splits=5)
        assert len(result.fold_sharpes) == 5

    def test_no_data_leakage(self):
        """Verify test indices are never in purged training set."""
        X, y, tbm = _make_cv_data(n=50)

        n_splits = 5
        for fold in range(n_splits):
            train_idx, test_idx = _get_train_test_indices(50, n_splits, fold)
            purged = purge_train_indices(train_idx, test_idx, tbm, X.index)
            embargoed = apply_embargo(purged, test_idx, 50, 0.01)

            # No overlap between final train and test
            assert len(set(embargoed) & set(test_idx)) == 0

            # Additionally, no train sample's label span overlaps with test
            for ti in embargoed:
                train_start = tbm.iloc[ti]["t_start"]
                train_end = tbm.iloc[ti]["t_end"]
                test_min = tbm.iloc[test_idx[0]]["t_start"]
                test_max = tbm.iloc[test_idx[-1]]["t_end"]
                assert train_end < test_min or train_start > test_max

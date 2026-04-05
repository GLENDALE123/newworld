import numpy as np
import pandas as pd
import pytest
from itertools import combinations
from math import comb

from validation.core.cpcv import _split_into_groups, cpcv


class DummyModel:
    def __init__(self):
        self._majority = 1

    def train(self, X, y, sample_weight=None):
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])

    def predict(self, X):
        return np.full(len(X), self._majority)


def _make_cpcv_data(n: int = 120, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    X = pd.DataFrame(
        {"feat1": rng.standard_normal(n), "feat2": rng.standard_normal(n)},
        index=idx,
    )
    y = pd.Series(rng.choice([1, -1], n), index=idx)
    holding = 3
    t_ends = pd.DatetimeIndex([idx[min(i + holding, n - 1)] for i in range(n)])
    tbm = pd.DataFrame({"t_start": idx, "t_end": t_ends}, index=range(n))
    return X, y, tbm


class TestSplitIntoGroups:
    def test_correct_number_of_groups(self):
        groups = _split_into_groups(120, 6)
        assert len(groups) == 6

    def test_covers_all_indices(self):
        groups = _split_into_groups(120, 6)
        all_idx = np.concatenate(groups)
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(120))

    def test_contiguous(self):
        groups = _split_into_groups(120, 6)
        for g in groups:
            diffs = np.diff(g)
            assert np.all(diffs == 1)

    def test_uneven_split(self):
        groups = _split_into_groups(13, 3)
        all_idx = np.concatenate(groups)
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(13))


class TestCPCV:
    def test_correct_number_of_paths(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        expected_paths = comb(6, 2)  # C(6,2) = 15
        assert len(result.path_sharpes) == expected_paths

    def test_pct_negative_in_range(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        assert 0.0 <= result.pct_negative <= 1.0

    def test_mean_equals_average(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        assert abs(result.mean_sharpe - np.mean(result.path_sharpes)) < 1e-10

    def test_median_sharpe(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        assert abs(result.median_sharpe - np.median(result.path_sharpes)) < 1e-10

    def test_with_sample_weights(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        weights = pd.Series(np.ones(120), index=X.index)
        result = cpcv(X, y, model, tbm, sample_weight=weights, n_groups=6, k_test_groups=2)
        assert len(result.path_sharpes) == comb(6, 2)

    def test_small_groups(self):
        """Test with small data: 4 groups, 1 test group."""
        X, y, tbm = _make_cpcv_data(n=40)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=4, k_test_groups=1)
        assert len(result.path_sharpes) == comb(4, 1)  # 4 paths

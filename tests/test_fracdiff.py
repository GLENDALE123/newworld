import numpy as np
import pandas as pd
import pytest

from validation.features.fracdiff import (
    _compute_weights,
    fracdiff,
    find_optimal_d,
    analyze_features,
)


class TestComputeWeights:
    def test_d_zero_single_weight(self):
        """d=0 should give just [1.0] since w_1 = -1*(0-0)/1 = 0."""
        weights = _compute_weights(0.0)
        assert len(weights) == 1
        assert weights[0] == 1.0

    def test_d_one_two_weights(self):
        """d=1 should give [1, -1] (full first difference)."""
        weights = _compute_weights(1.0, threshold=1e-5)
        assert len(weights) == 2
        assert abs(weights[0] - 1.0) < 1e-10
        assert abs(weights[1] - (-1.0)) < 1e-10

    def test_d_half_decreasing(self):
        """d=0.5 weights should decrease in magnitude."""
        weights = _compute_weights(0.5)
        magnitudes = np.abs(weights)
        assert magnitudes[0] > magnitudes[1] > magnitudes[2]

    def test_weights_sum_for_d_1(self):
        """For d=1, the weights represent first difference: x_t - x_{t-1}."""
        weights = _compute_weights(1.0)
        assert abs(sum(weights)) < 1e-10  # 1 + (-1) = 0


class TestFracdiff:
    def test_d_1_is_first_difference(self):
        """Fracdiff with d=1 should approximate first difference."""
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        series = pd.Series(np.cumsum(np.ones(100)), index=idx)  # 1, 2, 3, ...
        diffed = fracdiff(series, 1.0)
        # First difference of [1,2,3,...] should be [nan, 1, 1, 1, ...]
        clean = diffed.dropna()
        assert all(abs(v - 1.0) < 1e-10 for v in clean.values)

    def test_d_0_is_identity(self):
        """Fracdiff with d=0 should return original series."""
        idx = pd.date_range("2023-01-01", periods=50, freq="D")
        series = pd.Series(np.arange(50, dtype=float), index=idx)
        diffed = fracdiff(series, 0.0)
        # d=0 -> only weight is [1], so result should equal original
        clean = diffed.dropna()
        np.testing.assert_array_almost_equal(clean.values, series.values)

    def test_output_length_matches_input(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        series = pd.Series(np.random.default_rng(42).standard_normal(100), index=idx)
        diffed = fracdiff(series, 0.5)
        assert len(diffed) == len(series)

    def test_leading_nans(self):
        idx = pd.date_range("2023-01-01", periods=1000, freq="D")
        series = pd.Series(np.random.default_rng(42).standard_normal(1000), index=idx)
        diffed = fracdiff(series, 0.5)
        # Should have some leading NaN values (window - 1)
        n_nans = diffed.isna().sum()
        assert n_nans > 0
        assert n_nans < len(series)


class TestFindOptimalD:
    def test_stationary_series_d_zero(self):
        """Already stationary series (white noise) should need d=0."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        series = pd.Series(rng.standard_normal(500), index=idx)
        d, pval, _ = find_optimal_d(series)
        assert d == 0.0
        assert pval < 0.05

    def test_random_walk_needs_differencing(self):
        """Random walk should need d > 0."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        series = pd.Series(np.cumsum(rng.standard_normal(500)), index=idx)
        d, pval, diffed = find_optimal_d(series)
        assert d > 0
        assert pval < 0.05

    def test_optimal_d_between_0_and_1(self):
        """For a near-unit-root process, optimal d should be fractional."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        # AR(1) with phi close to 1 -> needs some differencing but not full
        ar = np.zeros(500)
        for i in range(1, 500):
            ar[i] = 0.99 * ar[i-1] + rng.standard_normal()
        series = pd.Series(ar, index=idx)
        d, pval, _ = find_optimal_d(series, d_step=0.1)
        assert 0.0 <= d <= 1.0


class TestAnalyzeFeatures:
    def test_multiple_features(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=300, freq="D")
        df = pd.DataFrame({
            "stationary": rng.standard_normal(300),
            "random_walk": np.cumsum(rng.standard_normal(300)),
        }, index=idx)
        result = analyze_features(df, d_step=0.2)
        assert "stationary" in result.feature_d_values
        assert "random_walk" in result.feature_d_values
        assert result.feature_d_values["stationary"] <= result.feature_d_values["random_walk"]

    def test_returns_fracdiff_result(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=200, freq="D")
        df = pd.DataFrame({"feat1": rng.standard_normal(200)}, index=idx)
        result = analyze_features(df, d_step=0.25)
        assert isinstance(result.feature_d_values, dict)
        assert isinstance(result.feature_adf_pvalues, dict)

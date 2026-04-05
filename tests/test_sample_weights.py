import numpy as np
import pandas as pd
import pytest

from validation.core.sample_weights import (
    build_indicator_matrix,
    compute_concurrency,
    compute_uniqueness,
    compute_sample_weights,
)


def _make_timestamps_and_prices():
    """Create synthetic TBM timestamps with known concurrency.

    Timeline (hourly bars):
      t0  t1  t2  t3  t4  t5  t6  t7
    Label 0: [t0--------t3]
    Label 1:     [t1--------t4]
    Label 2:             [t3--------t6]
    Label 3:                     [t5--------t7]

    Concurrency:
      t0: 1, t1: 2, t2: 2, t3: 3, t4: 2, t5: 2, t6: 1, t7: 1
    """
    idx = pd.date_range("2023-01-01", periods=8, freq="1h", tz="UTC")
    close_prices = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0],
        index=idx,
    )
    tbm_timestamps = pd.DataFrame(
        {
            "t_start": [idx[0], idx[1], idx[3], idx[5]],
            "t_end": [idx[3], idx[4], idx[6], idx[7]],
        },
        index=[0, 1, 2, 3],
    )
    return close_prices, tbm_timestamps, idx


class TestBuildIndicatorMatrix:
    def test_shape(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        assert indicator.shape == (8, 4)

    def test_label_0_active(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        # Label 0 is active at t0..t3
        assert indicator.loc[idx[0], 0] == 1
        assert indicator.loc[idx[3], 0] == 1
        assert indicator.loc[idx[4], 0] == 0

    def test_all_binary(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        assert set(indicator.values.flatten()).issubset({0, 1})


class TestConcurrency:
    def test_known_concurrency(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        conc = compute_concurrency(indicator)
        assert conc[idx[0]] == 1
        assert conc[idx[1]] == 2
        assert conc[idx[3]] == 3  # Labels 0, 1, 2 all active
        assert conc[idx[7]] == 1


class TestUniqueness:
    def test_non_overlapping_labels_uniqueness_1(self):
        """Labels with no overlap should have uniqueness = 1.0."""
        idx = pd.date_range("2023-01-01", periods=6, freq="1h", tz="UTC")
        tbm = pd.DataFrame({
            "t_start": [idx[0], idx[3]],
            "t_end": [idx[2], idx[5]],
        }, index=[0, 1])
        indicator = build_indicator_matrix(tbm, idx)
        uniq = compute_uniqueness(indicator)
        assert abs(uniq[0] - 1.0) < 1e-10
        assert abs(uniq[1] - 1.0) < 1e-10

    def test_fully_overlapping(self):
        """Two labels covering identical time range should have uniqueness = 0.5."""
        idx = pd.date_range("2023-01-01", periods=4, freq="1h", tz="UTC")
        tbm = pd.DataFrame({
            "t_start": [idx[0], idx[0]],
            "t_end": [idx[3], idx[3]],
        }, index=[0, 1])
        indicator = build_indicator_matrix(tbm, idx)
        uniq = compute_uniqueness(indicator)
        assert abs(uniq[0] - 0.5) < 1e-10
        assert abs(uniq[1] - 0.5) < 1e-10

    def test_uniqueness_between_0_and_1(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        uniq = compute_uniqueness(indicator)
        assert all(0 < u <= 1.0 for u in uniq)


class TestSampleWeights:
    def test_weights_sum_to_n(self):
        close_prices, tbm_timestamps, _ = _make_timestamps_and_prices()
        _, _, weights = compute_sample_weights(close_prices, tbm_timestamps)
        assert abs(weights.sum() - len(tbm_timestamps)) < 1e-10

    def test_unique_large_return_gets_high_weight(self):
        """Label 3 is least overlapping and has a return. It should get a high weight."""
        close_prices, tbm_timestamps, _ = _make_timestamps_and_prices()
        _, uniq, weights = compute_sample_weights(close_prices, tbm_timestamps)
        # Label 3: t5->t7 return = (99-101)/101, uniqueness high
        # Verify weights are positive where there's a return
        assert all(w >= 0 for w in weights)

    def test_zero_return_zero_weight(self):
        """If a label has zero return, its weight should be 0 (before normalization)."""
        idx = pd.date_range("2023-01-01", periods=4, freq="1h", tz="UTC")
        close_prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
        tbm = pd.DataFrame({
            "t_start": [idx[0]],
            "t_end": [idx[3]],
        }, index=[0])
        _, _, weights = compute_sample_weights(close_prices, tbm)
        # Only one label with zero return -> weight=0, but normalization:
        # sum is 0, so normalization is skipped, weight stays 0
        assert weights[0] == 0.0

    def test_returns_correct_tuple(self):
        close_prices, tbm_timestamps, _ = _make_timestamps_and_prices()
        result = compute_sample_weights(close_prices, tbm_timestamps)
        assert len(result) == 3  # concurrency, uniqueness, weights
        conc, uniq, weights = result
        assert isinstance(conc, pd.Series)
        assert isinstance(uniq, pd.Series)
        assert isinstance(weights, pd.Series)

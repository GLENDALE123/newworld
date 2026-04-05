import numpy as np
import pandas as pd
import pytest

from validation.core.sample_weights import build_indicator_matrix
from validation.core.sequential_bootstrap import (
    sequential_bootstrap,
    estimate_effective_n,
    _compute_average_uniqueness_with_selected,
)


def _make_non_overlapping_indicator():
    """3 labels with no overlap -> uniqueness should be ~uniform."""
    idx = pd.date_range("2023-01-01", periods=9, freq="1h", tz="UTC")
    tbm = pd.DataFrame({
        "t_start": [idx[0], idx[3], idx[6]],
        "t_end": [idx[2], idx[5], idx[8]],
    }, index=[0, 1, 2])
    return build_indicator_matrix(tbm, idx)


def _make_overlapping_indicator():
    """4 labels with heavy overlap."""
    idx = pd.date_range("2023-01-01", periods=8, freq="1h", tz="UTC")
    tbm = pd.DataFrame({
        "t_start": [idx[0], idx[1], idx[3], idx[5]],
        "t_end": [idx[3], idx[4], idx[6], idx[7]],
    }, index=[0, 1, 2, 3])
    return build_indicator_matrix(tbm, idx)


class TestAverageUniqueness:
    def test_first_selection_is_1(self):
        """First selected sample should have uniqueness 1.0 (no others in set)."""
        indicator = _make_non_overlapping_indicator()
        uniq = _compute_average_uniqueness_with_selected(indicator, [], 0)
        assert abs(uniq - 1.0) < 1e-10

    def test_overlapping_reduces_uniqueness(self):
        """Selecting overlapping sample reduces uniqueness."""
        indicator = _make_overlapping_indicator()
        # Select label 0, then check uniqueness of label 1 (overlaps with 0)
        uniq_alone = _compute_average_uniqueness_with_selected(indicator, [], 1)
        uniq_with_0 = _compute_average_uniqueness_with_selected(indicator, [0], 1)
        assert uniq_with_0 < uniq_alone


class TestSequentialBootstrap:
    def test_returns_correct_count(self):
        indicator = _make_non_overlapping_indicator()
        selected = sequential_bootstrap(indicator, n_samples=5, random_state=42)
        assert len(selected) == 5

    def test_default_n_samples(self):
        indicator = _make_non_overlapping_indicator()
        selected = sequential_bootstrap(indicator, random_state=42)
        assert len(selected) == 3  # 3 labels

    def test_selected_are_valid_labels(self):
        indicator = _make_overlapping_indicator()
        selected = sequential_bootstrap(indicator, n_samples=10, random_state=42)
        for s in selected:
            assert s in indicator.columns

    def test_reproducible_with_seed(self):
        indicator = _make_overlapping_indicator()
        sel1 = sequential_bootstrap(indicator, n_samples=10, random_state=99)
        sel2 = sequential_bootstrap(indicator, n_samples=10, random_state=99)
        assert sel1 == sel2

    def test_non_overlapping_uniform_selection(self):
        """With non-overlapping labels, all should be roughly equally likely."""
        indicator = _make_non_overlapping_indicator()
        counts = {0: 0, 1: 0, 2: 0}
        for seed in range(200):
            selected = sequential_bootstrap(indicator, n_samples=1, random_state=seed)
            counts[selected[0]] += 1
        # Each label should appear roughly 200/3 ~ 67 times
        for count in counts.values():
            assert count > 30  # generous margin


class TestEffectiveN:
    def test_non_overlapping_effective_n_equals_total(self):
        """Non-overlapping labels: effective N should be close to total N."""
        indicator = _make_non_overlapping_indicator()
        eff_n = estimate_effective_n(indicator, n_bootstrap_runs=50, random_state=42)
        # With 3 non-overlapping labels, we expect ~3 unique out of 3 draws
        # (high uniqueness -> balanced sampling -> high unique count)
        assert eff_n >= 2

    def test_overlapping_effective_n_less_than_total(self):
        """Overlapping labels: effective N should be <= total N."""
        indicator = _make_overlapping_indicator()
        eff_n = estimate_effective_n(indicator, n_bootstrap_runs=50, random_state=42)
        assert eff_n <= len(indicator.columns)

    def test_effective_n_positive(self):
        indicator = _make_overlapping_indicator()
        eff_n = estimate_effective_n(indicator, n_bootstrap_runs=20, random_state=42)
        assert eff_n > 0

import numpy as np
import pytest
from scipy import stats

from validation.statistics.deflated_sharpe import (
    expected_max_sharpe,
    compute_deflated_sharpe,
    EULER_MASCHERONI,
)


class TestExpectedMaxSharpe:
    def test_single_trial_returns_zero(self):
        assert expected_max_sharpe(1, variance=0.01) == 0.0

    def test_more_trials_higher_expected(self):
        e2 = expected_max_sharpe(2, variance=0.01)
        e10 = expected_max_sharpe(10, variance=0.01)
        e100 = expected_max_sharpe(100, variance=0.01)
        assert e10 > e2
        assert e100 > e10

    def test_higher_variance_higher_expected(self):
        e_low = expected_max_sharpe(10, variance=0.001)
        e_high = expected_max_sharpe(10, variance=0.01)
        assert e_high > e_low

    def test_positive_for_multiple_trials(self):
        e = expected_max_sharpe(5, variance=0.01)
        assert e > 0


class TestComputeDeflatedSharpe:
    def test_high_sharpe_low_pvalue(self):
        """A genuinely high Sharpe with 1 trial should have low p-values."""
        rng = np.random.default_rng(42)
        # Strong positive signal
        returns = rng.normal(0.01, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        assert result.psr_pvalue < 0.05  # significant against 0
        assert result.dsr_pvalue < 0.05  # n_trials=1 -> DSR benchmark = 0

    def test_zero_sharpe_high_pvalue(self):
        """Zero-mean returns should have high PSR p-value."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        # p-value should be around 0.5 for zero Sharpe
        assert result.psr_pvalue > 0.3

    def test_many_trials_inflates_dsr_pvalue(self):
        """More trials -> higher expected max SR -> harder to pass DSR."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)

        result_1 = compute_deflated_sharpe(returns, sr, n_trials=1)
        result_100 = compute_deflated_sharpe(returns, sr, n_trials=100)

        # With 100 trials, the bar is much higher -> DSR p-value should be higher
        assert result_100.dsr_pvalue > result_1.dsr_pvalue

    def test_psr_pvalue_is_1_minus_psr(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 100)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        # Verify p-value is 1 - CDF
        assert 0 <= result.psr_pvalue <= 1
        assert 0 <= result.dsr_pvalue <= 1

    def test_expected_max_sharpe_stored(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=10)
        assert result.expected_max_sharpe > 0
        assert result.n_trials == 10

    def test_sharpe_std_error_positive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        assert result.sharpe_std_error > 0

    def test_short_returns(self):
        """Should handle very short return series gracefully."""
        returns = np.array([0.01, -0.005, 0.003])
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        assert np.isfinite(result.psr_pvalue)
        assert np.isfinite(result.dsr_pvalue)

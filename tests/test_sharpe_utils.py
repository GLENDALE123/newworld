import numpy as np
import pytest

from validation.statistics.sharpe_utils import (
    annualized_sharpe,
    sharpe_standard_error,
    sharpe_confidence_interval,
    probabilistic_sharpe_ratio,
)


class TestAnnualizedSharpe:
    def test_positive_returns(self):
        # Constant positive daily return of 0.1%
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        sr = annualized_sharpe(returns, periods_per_year=252)
        # Should be roughly mean/std * sqrt(252) ~ 0.1 * sqrt(252) ~ 1.58
        assert sr > 0

    def test_zero_std_returns_zero(self):
        returns = [0.01] * 100  # constant returns -> std=0
        assert annualized_sharpe(returns) == 0.0

    def test_single_observation(self):
        assert annualized_sharpe([0.05]) == 0.0

    def test_empty_returns(self):
        assert annualized_sharpe([]) == 0.0

    def test_known_value(self):
        # 252 daily returns with mean=0.001, std=0.01
        # SR = 0.001/0.01 * sqrt(252) = 0.1 * 15.87 = 1.587
        returns = np.full(252, 0.001)
        returns[0] = 0.001  # force non-zero std by using ddof=1
        # Actually, constant returns -> std=0 with ddof=1 too. Use 2 values.
        returns = [0.011, -0.009] * 126  # mean=0.001, std~=0.01414
        sr = annualized_sharpe(returns, periods_per_year=252)
        expected = 0.001 / np.std([0.011, -0.009] * 126, ddof=1) * np.sqrt(252)
        assert abs(sr - expected) < 1e-10


class TestSharpeStandardError:
    def test_basic_computation(self):
        # SE = sqrt((1 - 0*SR + (3-1)/4 * SR^2) / (n-1))
        # For SR=1, n=100, skew=0, kurtosis=3:
        # SE = sqrt((1 + 0.5 * 1) / 99) = sqrt(1.5 / 99) = 0.1231
        se = sharpe_standard_error(1.0, 100, skew=0.0, kurtosis=3.0)
        expected = np.sqrt(1.5 / 99)
        assert abs(se - expected) < 1e-4

    def test_n_equals_1_returns_inf(self):
        assert sharpe_standard_error(1.0, 1) == float("inf")

    def test_skew_effect(self):
        se_no_skew = sharpe_standard_error(1.0, 100, skew=0.0, kurtosis=3.0)
        se_neg_skew = sharpe_standard_error(1.0, 100, skew=-1.0, kurtosis=3.0)
        # Negative skew with positive SR -> larger numerator -> larger SE
        assert se_neg_skew > se_no_skew

    def test_zero_sharpe(self):
        se = sharpe_standard_error(0.0, 100, skew=0.0, kurtosis=3.0)
        expected = np.sqrt(1.0 / 99)
        assert abs(se - expected) < 1e-4


class TestSharpeConfidenceInterval:
    def test_symmetric_at_zero_skew(self):
        lo, hi = sharpe_confidence_interval(1.0, 100, confidence=0.95, skew=0.0, kurtosis=3.0)
        mid = (lo + hi) / 2
        assert abs(mid - 1.0) < 1e-10

    def test_wider_at_lower_confidence(self):
        lo_95, hi_95 = sharpe_confidence_interval(1.0, 100, confidence=0.95)
        lo_99, hi_99 = sharpe_confidence_interval(1.0, 100, confidence=0.99)
        assert (hi_99 - lo_99) > (hi_95 - lo_95)

    def test_narrows_with_more_data(self):
        lo_100, hi_100 = sharpe_confidence_interval(1.0, 100, confidence=0.95)
        lo_1000, hi_1000 = sharpe_confidence_interval(1.0, 1000, confidence=0.95)
        assert (hi_1000 - lo_1000) < (hi_100 - lo_100)


class TestProbabilisticSharpeRatio:
    def test_observed_equals_benchmark(self):
        psr = probabilistic_sharpe_ratio(0.5, 0.5, 100)
        assert abs(psr - 0.5) < 1e-10

    def test_high_sharpe_high_psr(self):
        psr = probabilistic_sharpe_ratio(2.0, 0.0, 252)
        assert psr > 0.95

    def test_negative_sharpe_low_psr(self):
        psr = probabilistic_sharpe_ratio(-1.0, 0.0, 252)
        assert psr < 0.05

    def test_more_data_increases_psr(self):
        psr_100 = probabilistic_sharpe_ratio(0.5, 0.0, 100)
        psr_1000 = probabilistic_sharpe_ratio(0.5, 0.0, 1000)
        assert psr_1000 > psr_100

    def test_psr_bounds(self):
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 252)
        assert 0.0 <= psr <= 1.0

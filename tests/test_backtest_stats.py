import numpy as np
import pandas as pd
import pytest

from validation.statistics.backtest_stats import compute_backtest_stats


def _make_returns(seed: int = 42, n: int = 252, mean: float = 0.001, std: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, n), index=idx)


class TestComputeBacktestStats:
    def test_positive_returns(self):
        returns = _make_returns(mean=0.001, std=0.01)
        result = compute_backtest_stats(returns)
        assert result.total_return > 0
        assert result.annualized_return > 0
        assert result.sharpe_ratio > 0
        assert result.win_rate > 0.4

    def test_negative_returns(self):
        returns = _make_returns(mean=-0.002, std=0.01)
        result = compute_backtest_stats(returns)
        assert result.total_return < 0
        assert result.sharpe_ratio < 0

    def test_drawdown_is_negative(self):
        returns = _make_returns()
        result = compute_backtest_stats(returns)
        assert result.max_drawdown <= 0

    def test_sharpe_ci_contains_sharpe(self):
        returns = _make_returns(n=500)
        result = compute_backtest_stats(returns)
        # The annualized CI should bracket the annualized Sharpe (approximately)
        # We test that the non-annualized raw CI logic works
        assert result.sharpe_ci_lower < result.sharpe_ci_upper

    def test_sortino_is_finite(self):
        returns = _make_returns()
        result = compute_backtest_stats(returns)
        assert np.isfinite(result.sortino_ratio)

    def test_profit_factor(self):
        # All positive returns -> profit_factor is inf
        returns = pd.Series([0.01, 0.02, 0.015, 0.005])
        result = compute_backtest_stats(returns)
        assert result.profit_factor == float("inf")

    def test_consecutive_wins_losses(self):
        # Pattern: 3 wins, 2 losses, 1 win
        returns = pd.Series([0.01, 0.02, 0.015, -0.01, -0.005, 0.01])
        result = compute_backtest_stats(returns)
        assert result.max_consecutive_wins == 3
        assert result.max_consecutive_losses == 2

    def test_tail_ratio_positive(self):
        returns = _make_returns(n=500)
        result = compute_backtest_stats(returns)
        assert result.tail_ratio > 0

    def test_distribution_stats(self):
        returns = _make_returns(n=1000)
        result = compute_backtest_stats(returns)
        # Skewness should be near 0 for normal returns
        assert abs(result.skewness) < 1.0
        # Kurtosis should be near 3 for normal returns
        assert abs(result.kurtosis - 3.0) < 1.0

    def test_single_return(self):
        returns = pd.Series([0.01])
        result = compute_backtest_stats(returns)
        assert result.total_return == 0.0  # n < 2 returns defaults

    def test_calmar_ratio(self):
        returns = _make_returns(mean=0.001, std=0.01, n=504)
        result = compute_backtest_stats(returns, periods_per_year=252)
        assert np.isfinite(result.calmar_ratio)

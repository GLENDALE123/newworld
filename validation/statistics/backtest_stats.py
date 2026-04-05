"""Institutional-grade backtest performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from validation.statistics.sharpe_utils import (
    annualized_sharpe,
    sharpe_confidence_interval,
    sharpe_standard_error,
)
from validation.report import BacktestStatsResult


def compute_backtest_stats(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> BacktestStatsResult:
    """Compute comprehensive backtest statistics from a return series.

    Args:
        returns: Series of periodic returns (e.g., daily returns).
        periods_per_year: Periods per year for annualization.

    Returns:
        BacktestStatsResult with all computed statistics.
    """
    r = returns.values.astype(np.float64)
    n = len(r)

    if n < 2:
        return BacktestStatsResult(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sharpe_std_error=float("inf"), sharpe_ci_lower=0.0, sharpe_ci_upper=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0, max_drawdown=0.0,
            max_drawdown_duration=0, win_rate=0.0, profit_factor=0.0,
            tail_ratio=0.0, skewness=0.0, kurtosis=0.0,
            max_consecutive_wins=0, max_consecutive_losses=0,
        )

    # ---------- Returns ----------
    cumulative = np.cumprod(1 + r)
    total_return = float(cumulative[-1] - 1)
    n_years = n / periods_per_year
    annualized_return = float((cumulative[-1]) ** (1 / max(n_years, 1e-10)) - 1) if cumulative[-1] > 0 else -1.0

    # ---------- Risk ----------
    sr = annualized_sharpe(r, periods_per_year)

    # Non-annualized Sharpe for SE/CI computation
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1) if n > 1 else 1e-10
    sr_raw = mean_r / std_r if std_r > 0 else 0.0
    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=False))  # raw kurtosis (not excess)

    se = sharpe_standard_error(sr_raw, n, skew, kurt)
    ci_lo, ci_hi = sharpe_confidence_interval(sr_raw, n, 0.95, skew, kurt)
    # Scale CI to annualized
    scale = np.sqrt(periods_per_year)
    ci_lo_ann = ci_lo * scale
    ci_hi_ann = ci_hi * scale
    se_ann = se * scale

    # Sortino: use downside deviation
    downside = r[r < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = float(mean_r / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    # ---------- Drawdown ----------
    cum_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / cum_max - 1
    max_dd = float(np.min(drawdowns))

    # Max drawdown duration (periods between peaks)
    peak_indices = np.where(cumulative >= cum_max)[0]
    if len(peak_indices) > 1:
        durations = np.diff(peak_indices)
        max_dd_duration = int(np.max(durations))
    else:
        max_dd_duration = n

    # Calmar
    calmar = float(annualized_return / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    # ---------- Trade statistics ----------
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / n) if n > 0 else 0.0

    sum_wins = np.sum(wins)
    sum_losses = abs(np.sum(losses))
    profit_factor = float(sum_wins / sum_losses) if sum_losses > 0 else float("inf") if sum_wins > 0 else 0.0

    # Tail ratio: 95th percentile of wins / abs(5th percentile of losses)
    p95 = np.percentile(r, 95) if n > 0 else 0.0
    p05 = abs(np.percentile(r, 5)) if n > 0 else 1e-10
    tail_ratio = float(p95 / p05) if p05 > 1e-10 else 0.0

    # ---------- Distribution ----------
    # skew and kurt already computed above

    # ---------- Consecutive wins/losses ----------
    max_consec_wins = _max_consecutive(r > 0)
    max_consec_losses = _max_consecutive(r < 0)

    return BacktestStatsResult(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sr,
        sharpe_std_error=se_ann,
        sharpe_ci_lower=ci_lo_ann,
        sharpe_ci_upper=ci_hi_ann,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        tail_ratio=tail_ratio,
        skewness=skew,
        kurtosis=kurt,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
    )


def _max_consecutive(mask: np.ndarray) -> int:
    """Count the longest run of True values in a boolean array."""
    if len(mask) == 0:
        return 0
    # Pad with False to catch runs at the edges
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    runs = ends - starts
    return int(np.max(runs))

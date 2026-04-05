import numpy as np
from backtest.analysis import compute_metrics


def test_compute_metrics():
    equity_curve = np.array([100000, 101000, 100500, 102000, 101500, 103000])
    metrics = compute_metrics(equity_curve)

    assert "total_return_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics
    assert metrics["total_return_pct"] == 3.0


def test_compute_metrics_flat():
    equity_curve = np.array([100000, 100000, 100000])
    metrics = compute_metrics(equity_curve)
    assert metrics["total_return_pct"] == 0.0
    assert metrics["total_trades"] == 0


def test_compute_metrics_drawdown():
    equity_curve = np.array([100000, 110000, 90000, 95000])
    metrics = compute_metrics(equity_curve)
    # Peak was 110000, dropped to 90000 = 18.18% drawdown
    assert metrics["max_drawdown_pct"] > 18
    assert metrics["max_drawdown_pct"] < 19

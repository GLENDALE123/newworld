import numpy as np


def compute_metrics(equity_curve: np.ndarray) -> dict:
    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return_pct = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100

    # Sharpe (annualized, assuming hourly data -> 8760 hours/year)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(8760)
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_drawdown_pct = float(np.max(drawdown))

    # Win Rate
    winning = returns[returns > 0]
    total = returns[returns != 0]
    win_rate = (len(winning) / len(total) * 100) if len(total) > 0 else 0.0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return_pct": round(total_return_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": len(total),
    }

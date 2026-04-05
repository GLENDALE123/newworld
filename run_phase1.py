"""
ultraTM Phase 1: TBM + CatBoost Walk-Forward Pipeline

1. Load BTCUSDT 1h OHLCV from existing data
2. Build features + TBM labels
3. Walk-forward train CatBoost
4. NautilusTrader backtest
5. Full performance report
"""
import numpy as np
import pandas as pd

from config.settings import Settings
from data.loader import load_kline
from features.pipeline import FeaturePipeline
from models.catboost_model import TradingModel
from backtest.runner import run_backtest
from backtest.analysis import compute_metrics


def main():
    settings = Settings()
    print(f"=== ultraTM Phase 1: {settings.symbol} {settings.timeframe} ===\n")

    # 1. Load data
    print("[1/5] Loading data...")
    ohlcv = load_kline(settings.data_dir, settings.symbol, settings.timeframe)
    print(f"      {len(ohlcv)} bars loaded ({ohlcv.index.min()} ~ {ohlcv.index.max()})")

    # 2. Build features + TBM labels
    print("[2/5] Building features + TBM labels...")
    pipeline = FeaturePipeline(settings)
    dataset = pipeline.build(ohlcv)
    label_counts = dataset["label"].value_counts()
    print(f"      Dataset: {len(dataset)} rows, {len(dataset.columns)} columns")
    print(f"      Labels: +1={label_counts.get(1.0, 0)}, -1={label_counts.get(-1.0, 0)}")

    # 3. Walk-forward training
    print("[3/5] Walk-forward training...")
    feature_cols = [c for c in dataset.columns if c != "label"]

    train_bars = settings.train_window_months * 30 * 24
    val_bars = settings.val_window_months * 30 * 24

    model = TradingModel(
        iterations=settings.catboost_iterations,
        depth=settings.catboost_depth,
        learning_rate=settings.catboost_learning_rate,
        task_type="GPU",
    )

    try:
        wf_results = model.walk_forward(dataset, train_window=train_bars, val_window=val_bars)
    except Exception:
        print("      GPU not available, falling back to CPU...")
        model = TradingModel(
            iterations=settings.catboost_iterations,
            depth=settings.catboost_depth,
            learning_rate=settings.catboost_learning_rate,
            task_type="CPU",
        )
        wf_results = model.walk_forward(dataset, train_window=train_bars, val_window=val_bars)

    preds = wf_results["predictions"]
    actuals = wf_results["actuals"]
    indices = wf_results["indices"]
    print(f"      Walk-forward predictions: {len(preds)}")

    # 4. NautilusTrader backtest
    print("[4/5] Running NautilusTrader backtest...")

    # Build signals dict: {timestamp_ns -> signal}
    # NautilusTrader uses nanosecond timestamps for ts_event
    signals = {}
    for idx, pred in zip(indices, preds):
        ts = idx
        if isinstance(ts, pd.Timestamp):
            ts_ns = int(ts.value)  # pandas Timestamp.value is already nanoseconds
        else:
            ts_ns = int(pd.Timestamp(ts).value)
        signals[ts_ns] = pred

    print(f"      {len(signals)} signals prepared")

    # Slice OHLCV to walk-forward prediction period for backtest
    wf_start = indices[0]
    wf_end = indices[-1]
    # Include some warmup bars before first signal for volatility computation
    warmup = settings.volatility_span + 10
    ohlcv_start_idx = max(0, ohlcv.index.get_loc(wf_start) - warmup)
    bt_ohlcv = ohlcv.iloc[ohlcv_start_idx:]

    engine = run_backtest(settings, bt_ohlcv, signals)

    # 5. Performance report
    print("\n[5/5] Performance Report")
    print("=" * 60)

    # --- ML Metrics ---
    print("\n  [ML Metrics]")
    accuracy = np.mean(preds == actuals) * 100
    print(f"  Accuracy:          {accuracy:.1f}%")

    long_mask = preds == 1.0
    short_mask = preds == -1.0

    if long_mask.sum() > 0:
        long_prec = np.mean(actuals[long_mask] == 1.0) * 100
        print(f"  Long Precision:    {long_prec:.1f}% ({long_mask.sum()} signals)")

    if short_mask.sum() > 0:
        short_prec = np.mean(actuals[short_mask] == -1.0) * 100
        print(f"  Short Precision:   {short_prec:.1f}% ({short_mask.sum()} signals)")

    print(f"  Signal Dist:       Long {long_mask.sum()} ({long_mask.sum()/len(preds)*100:.0f}%) / Short {short_mask.sum()} ({short_mask.sum()/len(preds)*100:.0f}%)")

    # --- Trading Metrics from NautilusTrader ---
    print("\n  [Trading Metrics]")

    # Get final equity
    accounts = engine.cache.accounts()
    account = accounts[0] if accounts else None
    if account:
        final_balance = float(account.balance_total(account.currencies()[0]))
        initial = settings.initial_capital
        total_return = ((final_balance - initial) / initial) * 100

        print(f"  Initial Capital:   ${initial:,.0f}")
        print(f"  Final Balance:     ${final_balance:,.2f}")
        print(f"  Total Return:      {total_return:+.2f}%")

    # Position-level stats from generate_positions_report
    try:
        position_reports = engine.trader.generate_positions_report()
    except Exception:
        position_reports = pd.DataFrame()

    if not position_reports.empty:
        n_trades = len(position_reports)
        print(f"  Total Trades:      {n_trades}")

        if "realized_pnl" in position_reports.columns:
            # NautilusTrader appends currency suffix like "-0.96 USDT"
            pnl = position_reports["realized_pnl"].apply(
                lambda x: float(str(x).split()[0]) if isinstance(x, str) else float(x)
            )
            winners = (pnl > 0).sum()
            losers = (pnl < 0).sum()
            win_rate = winners / n_trades * 100

            avg_win = pnl[pnl > 0].mean() if winners > 0 else 0
            avg_loss = pnl[pnl < 0].mean() if losers > 0 else 0
            gross_profit = pnl[pnl > 0].sum()
            gross_loss = abs(pnl[pnl < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            print(f"  Winners/Losers:    {winners}/{losers}")
            print(f"  Win Rate:          {win_rate:.1f}%")
            print(f"  Avg Win PnL:       ${avg_win:+,.2f}")
            print(f"  Avg Loss PnL:      ${avg_loss:+,.2f}")
            print(f"  Profit Factor:     {profit_factor:.2f}")

        if "duration_ns" in position_reports.columns:
            avg_dur_hours = position_reports["duration_ns"].astype(float).mean() / 3.6e12
            print(f"  Avg Hold Time:     {avg_dur_hours:.1f}h")
    else:
        print("  Total Trades:      0 (no trades executed)")

    # --- Buy & Hold Benchmark ---
    print("\n  [Buy & Hold Benchmark]")
    bh_start = float(bt_ohlcv["close"].iloc[0])
    bh_end = float(bt_ohlcv["close"].iloc[-1])
    bh_return = ((bh_end - bh_start) / bh_start) * 100
    print(f"  BTC B&H Return:    {bh_return:+.2f}% ({bh_start:.0f} -> {bh_end:.0f})")
    if account:
        excess = total_return - bh_return
        print(f"  Excess Return:     {excess:+.2f}%")

    print("\n" + "=" * 60)
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()

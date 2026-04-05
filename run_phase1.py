"""
ultraTM Phase 1: TBM + CatBoost Walk-Forward Pipeline

1. Load BTCUSDT 1h OHLCV from existing data
2. Build features + TBM labels
3. Walk-forward train CatBoost
4. Print performance report
"""
import numpy as np

from config.settings import Settings
from data.loader import load_kline
from features.pipeline import FeaturePipeline
from models.catboost_model import TradingModel


def main():
    settings = Settings()
    print(f"=== ultraTM Phase 1: {settings.symbol} {settings.timeframe} ===\n")

    # 1. Load data
    print("[1/4] Loading data...")
    ohlcv = load_kline(settings.data_dir, settings.symbol, settings.timeframe)
    print(f"      {len(ohlcv)} bars loaded ({ohlcv.index.min()} ~ {ohlcv.index.max()})")

    # 2. Build features + TBM labels
    print("[2/4] Building features + TBM labels...")
    pipeline = FeaturePipeline(settings)
    dataset = pipeline.build(ohlcv)
    label_counts = dataset["label"].value_counts()
    print(f"      Dataset: {len(dataset)} rows, {len(dataset.columns)} columns")
    print(f"      Labels: +1={label_counts.get(1.0, 0)}, -1={label_counts.get(-1.0, 0)}")

    # 3. Walk-forward training
    print("[3/4] Walk-forward training...")
    feature_cols = [c for c in dataset.columns if c != "label"]

    # Convert months to bars (1h = 24 bars/day * 30 days/month)
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
    print(f"      Walk-forward predictions: {len(preds)}")

    # 4. Performance report
    print("\n[4/4] Performance Report")
    print("=" * 50)

    accuracy = np.mean(preds == actuals) * 100
    print(f"  Accuracy:        {accuracy:.1f}%")

    # Per-direction metrics
    long_mask = preds == 1.0
    short_mask = preds == -1.0

    if long_mask.sum() > 0:
        long_precision = np.mean(actuals[long_mask] == 1.0) * 100
        print(f"  Long Precision:  {long_precision:.1f}% ({long_mask.sum()} signals)")
    else:
        print("  Long Precision:  N/A (no long signals)")

    if short_mask.sum() > 0:
        short_precision = np.mean(actuals[short_mask] == -1.0) * 100
        print(f"  Short Precision: {short_precision:.1f}% ({short_mask.sum()} signals)")
    else:
        print("  Short Precision: N/A (no short signals)")

    # Label distribution in predictions
    print(f"\n  Signal Distribution:")
    print(f"    Long:  {long_mask.sum()} ({long_mask.sum()/len(preds)*100:.1f}%)")
    print(f"    Short: {short_mask.sum()} ({short_mask.sum()/len(preds)*100:.1f}%)")

    print("\n" + "=" * 50)
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()

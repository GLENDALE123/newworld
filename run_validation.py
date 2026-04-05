"""
Run AFML validation framework against Phase 1 strategy.
"""
import numpy as np
import pandas as pd

from config.settings import Settings
from data.loader import load_kline
from features.pipeline import FeaturePipeline
from models.catboost_model import TradingModel
from validation import StrategyValidator


def main():
    settings = Settings()
    print(f"=== AFML Validation: {settings.symbol} {settings.timeframe} ===\n")

    # 1. Load data
    print("[1/3] Loading data + building features...")
    ohlcv = load_kline(settings.data_dir, settings.symbol, settings.timeframe)
    pipeline = FeaturePipeline(settings)
    dataset = pipeline.build(ohlcv)

    features = dataset.drop(columns=["label"])
    labels = dataset["label"]
    print(f"      {len(dataset)} samples, {len(features.columns)} features")

    # 2. Create model (reduced iterations + GPU for validation speed)
    print("[2/3] Preparing model (100 iterations, GPU)...")
    model = TradingModel(
        iterations=100,
        depth=settings.catboost_depth,
        learning_rate=0.1,
        task_type="GPU",
    )

    # 3. Run validation
    print("[3/3] Running AFML validation...\n")
    validator = StrategyValidator(
        ohlcv=ohlcv.loc[dataset.index],
        features=features,
        labels=labels,
        model=model,
        tbm_config={
            "pt_multiplier": settings.tbm_pt_multiplier,
            "sl_multiplier": settings.tbm_sl_multiplier,
            "max_holding_bars": settings.tbm_max_holding_bars,
        },
        n_trials=1,
        n_splits=3,
        n_groups=4,
        k_test_groups=1,
        skip_sfi=True,
        bootstrap_runs=20,
    )

    report = validator.run_full_validation()
    report.print_full()


if __name__ == "__main__":
    main()

# ultraTM v6 — Multi-Strategy Architecture Iteration

## What Was Built

### 1. Regime Detector (`execution/regime_detector.py`)
- 4-regime classification: SURGE, DUMP, RANGE, VOLATILE
- ADX-based directional movement + ATR volatility + volume confirmation
- **Regime transition detection** (RANGE→SURGE = breakout opportunity)
- Per-coin state tracking with confidence scores

### 2. Strategy Router (`execution/strategy_router.py`)
- Multi-dimensional trade selection engine
- 32 label mapping: 4 strategies × 2 directions × 4 regimes
- Category-specific rules per regime
- TradeSignal with score ranking (probability × EV × regime_confidence)
- Portfolio-level selection with diversity constraints

### 3. Updated Coin Classifier (`execution/coin_classifier.py`)
- Multi-strategy per category (NOT just trend following)
- Regime-specific allowed strategies and probability thresholds
- Alpha coin prioritization (proven per-trade alpha coins)
- 244 coins supported, 9 proven alpha coins tagged

### 4. PLE v6 Architecture (`ple/model_v6.py`)
- **Regime embedding** — explicit market state input
- **Sparse MoE routing** — 8 experts, top-2 selected per sample
- **Strategy-grouped heads** — separate output groups per strategy type
- **Coin embedding** — universal multi-coin (from v5)
- **Temporal encoder** — recent bar pattern awareness
- ~2M parameters

### 5. Polars Feature Factory (`features/factory_polars.py`)
- **2.6x faster** than pandas on single coin
- 53 features (price + temporal context) from 15m kline
- Lazy evaluation for memory efficiency
- Streaming: one coin at a time (24GB RAM safe)

### 6. ONNX Export (`ple/onnx_export.py`)
- CPU-only inference for Oracle server deployment
- v4 and v6 model export support
- ONNX Runtime with graph optimization
- Benchmark capability

### 7. Temporal Context Features (`features/temporal_context.py`)
- 32 temporal features: return trajectory, momentum consistency, range trends
- Squeeze detection, directional movement balance
- Designed to capture recent sequential patterns

### 8. Loss Function Experiments (`ple/loss_v6.py`)
- **Focal BCE**: gamma=2.0 makes model too conservative (100% no-trade)
- **Focal + Selectivity**: still too conservative
- **Plain BCE (v4 loss)**: best for model training — let router handle selectivity

## Key Findings

### 1. Architecture Works
- v6 MoE training converges (loss: 0.94 → -0.81)
- MoE load balance improved from 2.6 → 0.69 (experts distribute evenly)
- Gate weights learn meaningful patterns (price expert dominates at 0.65)

### 2. Model ≠ Selectivity
**Critical insight**: Selectivity belongs in the ROUTER, not the MODEL.
- Model's job = accurate probability estimation (BCE loss)
- Router's job = selectivity (only take high-conviction signals)
- Putting selectivity in the loss function collapses to zero trades

### 3. Feature Alignment Issue
- Saved v5 models expect 392 features (5m/15m/1h/4h multi-TF)
- Current data only has 15m klines → 36 features generated
- **Need 5m kline data** for proper model utilization

### 4. 214-Coin Pipeline Validated
- Full pipeline processed 214 coins in seconds
- Regime: 80% RANGE, 13% SURGE, 7% DUMP, 1% VOLATILE
- 34 signals from 214 coins (16% signal rate) — selective by design
- 3 strategy types active simultaneously
- Alpha coins ranked higher (as expected)

### 5. Polars Migration Beneficial
- 2.6x speedup even for single coin
- Will be even faster at batch level (244 coins)
- Memory-efficient for Oracle server (24GB RAM)

## Production Architecture (for Oracle server)

```
학습 (로컬 GPU):
  15m data → polars features → train v6 → export ONNX

추론 (Oracle 서버, 24GB RAM, no GPU):
  WebSocket klines → polars feature pipeline
  → ONNX Runtime inference (CPU)
  → RegimeDetector (4-state)
  → StrategyRouter (category-specific multi-strategy)
  → CapitalStrategy (leverage/sizing)
  → Execute via Binance API
```

## Next Steps (Priority Order)

1. **5m kline data** — needed for proper 392-feature generation
2. **Retrain with multi-TF features** — v6 + temporal context + cross-TF
3. **Walk-forward validation** — 3 windows × 3 seeds on alpha coins
4. **Polars full pipeline** — add cross-TF and cross-asset features
5. **Paper trading** — Oracle server deployment with ONNX

## Files Created/Modified

| File | Status |
|------|--------|
| `execution/regime_detector.py` | NEW |
| `execution/strategy_router.py` | NEW |
| `execution/coin_classifier.py` | UPDATED (multi-strategy) |
| `ple/model_v6.py` | NEW |
| `ple/trainer_v6.py` | NEW |
| `ple/loss_v6.py` | NEW |
| `ple/onnx_export.py` | NEW |
| `features/factory_polars.py` | NEW |
| `features/temporal_context.py` | NEW |
| `run_strategy_test.py` | NEW |
| `experiments/iter_v6_regime_moe.py` | NEW |
| `experiments/iter_multistrat_backtest.py` | NEW |

# ultraTM Milestone Report — 32 Iterations

## System Performance (Validated)

```
Walk-Forward (3 windows, fee-inclusive 0.08%):
  Mean Return:  +10-11%
  Best Single:  +73.57% (iter 11)
  Bear Market:  +25-35% (consistent)
  Bull Market:  -10% (unsolved)
  Max DD:       3.57-10.47% (configurable)
```

## Architecture

```
PLE v4 (Progressive Layered Extraction)
├── Feature-Partitioned Experts (price/volume/metrics)
├── Attention-Gated Fusion
├── Multi-Label Sigmoid (16 strategies)
├── BCE + Calibration Loss
└── 228K parameters, GPU training ~60s
```

## 32 Experiments — What Works, What Doesn't

### WORKS
| Technique | First Tried | Impact | Iter |
|-----------|-------------|--------|------|
| L_equity removal | 11 | +60% improvement | **Critical** |
| Sequence features | 16 | +2% improvement | Important |
| Adaptive SMA thresholds | 19 | +7% improvement | Important |
| Pos-weight BCE | 11 | Prevents NO_TRADE | Critical |
| ATR dynamic barriers | TBM v2 | Fee-viable labels | Critical |
| Fee-deducted RAR | TBM v2 | Realistic training | Critical |
| Sample weights (speed × magnitude) | TBM v2 | Better signal focus | Moderate |
| Mixup training | 32 | Stable regularization | Moderate |

### DOESN'T WORK (for this problem)
| Technique | Why Not | Iter |
|-----------|---------|------|
| CatBoost per label | No shared representation | 24 |
| PLE + CatBoost ensemble | Signals cancel out | 25 |
| Regime model switching | CatBoost still shorts in bull | 26 |
| Focal Loss | Lower recall than BCE | 29 |
| Dynamic TP/SL from model | ATR barriers more reliable | 31 |
| Soft SMA filter | Gray zone generates noise | 14 |
| L_equity (any variant) | Gradient explosion/saturation | 5,10,11 |
| Larger model (256dim) | Slower convergence | 15 |
| 5m data (ffill from 15m) | Redundant information | 21 |
| Swing-only (no intraday) | Loses short-term edge | 6 |

## Unsolved Problems

1. **Bull market** — model has structural short bias from training data
   - Solved with full 2020-2026 data (iter 27: 1490L>965S)
   - But requires historical 1h kline for TBM labels
   
2. **Multi-asset** — ETH/SOL timezone mismatch
   - Data available, connection logic needs fix

3. **Live execution** — NautilusTrader strategy written but untested
   - `strategy/ple_strategy.py` ready
   - `run_production.py` pipeline ready

## File Inventory (76 Python files)

```
ple/          — PLE v1-v4 models, losses, trainers, inference
labeling/     — TBM v1-v2, multi-label generator
features/     — Feature factory v1-v2, auto feature generation
validation/   — AFML framework (Purged CV, DSR, feature importance)
strategy/     — NautilusTrader strategies (ml_strategy, ple_strategy)
backtest/     — Runner, analysis
models/       — CatBoost model, production weights
config/       — Settings
reports/      — 32 iteration reports + summaries
```

## Next Steps for Production

1. **Collect 2020-2024 1h kline** → train on full bull/bear cycles
2. **Fix ETH/SOL timezone** → multi-asset diversification
3. **Paper trading** → NautilusTrader sandbox with Binance testnet
4. **Monitoring** → performance tracking, drift detection
5. **Live deployment** → gradual scale-up with 1% capital

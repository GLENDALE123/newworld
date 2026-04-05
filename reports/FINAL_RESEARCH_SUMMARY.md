# ultraTM Final Research Summary — 22 Iterations

## Best Walk-Forward Configuration (Iter 19)

```
Architecture: PLE v4 (multi-label sigmoid, feature-partitioned experts)
Features:    411 (price/volume/metrics/cross + seq lookback=8 + fracdiff)
Labels:      16 strategies (intraday + swing, long + short, 4 regimes)
Loss:        BCE + calibration (NO equity loss)
Execution:   Adaptive thresholds (SMA-based) + 3% sizing + DD recovery
Fees:        0.08% round-trip deducted from labels

Walk-Forward (3 windows):
  W1 (flat):  +11.89%  B&H +1.47%
  W2 (bull):  -10.06%  B&H +11.69%
  W3 (bear):  +30.55%  B&H -24.45%
  Mean:       +10.79%
```

## Iteration History

| Iter | Mean | Key Innovation | Impact |
|------|------|---------------|--------|
| 1 | -12.56% | PLE v4 baseline | Starting point |
| 2 | +12.65% | 15m base + 4x data | First positive |
| 5 | +27.86% | L_equity gradient fix | Best single split |
| 8 | +4.75% | 3-model ensemble | Lowest DD (3.57%) |
| 11 | +73.57% | **L_equity removed** | Prediction/execution separation |
| 12 | +1.45% | Walk-forward validation | Reality check |
| 16 | +3.53% | **Sequence features** | Temporal patterns |
| 17 | +8.54% | SMA direction boost | Trend alignment |
| **19** | **+10.79%** | **Adaptive thresholds** | **Best validated** |
| 22 | -0.40% | Order flow (CVD) | First balanced L/S |

## Key Insights

1. **Separate prediction from execution** (iter 11): removing equity loss from training was the biggest single improvement
2. **ATR-based dynamic barriers** are essential for fee-viable labels
3. **Sequence features** (lookback=8) capture market momentum patterns
4. **Adaptive thresholds** (SMA-based direction bias) outperform hard filters
5. **Order flow** (CVD, buy ratio) enables long signals but quality needs work
6. **Bull market remains unsolved** — model has structural short bias

## Technical Specifications

### Model
- PLE v4: 228K parameters, 128dim experts, 64dim output
- Feature-partitioned: price/volume/metrics/cross experts
- Attention-gated fusion
- Multi-label sigmoid (NOT softmax)
- BCE + MAE/MFE calibration loss

### Labels (TBM v2)
- ATR-based dynamic barriers
- 4 strategies: scalp(5m)/intraday(15m)/daytrade(1h)/swing(4h)
- 2 directions: long/short
- 4 regimes: surge/dump/range/volatile
- Fee-deducted RAR (0.08%)
- Sample weights: magnitude × speed

### Features (Factory v2)
- 197 base + 240 sequence = 437 total
- Multi-timeframe price (ATR, returns, vol at 5m/15m/1h/4h)
- Order flow (CVD momentum, buy ratio, trade intensity)
- Derivatives (OI changes, funding z-scores, LS ratios)
- Cross-asset (ETH/SOL correlation, lead-lag)
- Fractional differentiation (d=0.3)

### Execution Layer
- Adaptive thresholds: above SMA→long 0.40/short 0.55, below→reverse
- 3% base sizing with DD recovery (scale down at 3%+ DD)
- EV-based strategy selection: P*reward - (1-P)*risk - fee
- NautilusTrader integration: pending

## Next Steps

1. **NautilusTrader live connection** — execute iter 19 strategy in paper trading
2. **Multi-asset** — ETH/SOL timezone fix, portfolio diversification
3. **Bull market model** — separate long-only model or trend-following overlay
4. **Real-time inference pipeline** — WebSocket data → features → model → orders
5. **Monitoring dashboard** — performance tracking, drift detection

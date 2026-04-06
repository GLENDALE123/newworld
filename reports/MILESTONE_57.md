# ultraTM Milestone Report — 57 Iterations

## Confirmed Performance (Seed-Reproducible)

```
Model:     PLE v4 (VSN OFF, e128_o64_f192)
Config:    dropout=0.2, R-Drop α=1.0, BS=2048
Data:      2020-06-01 ~ 2026-02-28 (40,895 samples, 382 features)
Workers:   12 (deterministic seeding via worker_init_fn)

Walk-Forward (3 windows × 3 seeds):
  Seed 42:  +41.9%
  Seed 123: +52.6%
  Seed 777: +44.3%
  Mean:     +46.3%, Std: 4.6%
```

## Evolution Timeline

| Phase | Iters | Key Change | Mean | Impact |
|-------|-------|-----------|------|--------|
| Baseline | 1-32 | PLE v4, limited data | +11% | — |
| Data expansion | 33-34 | 15m→1h resample, full history | +26% | **+15%** |
| Regularization | 36-37 | Dropout 0.2, R-Drop α=1.0 | +40% | **+14%** |
| Architecture | 41-43 | Fusion 192, expert tuning | +47% | **+7%** |
| Execution layer | 45-48 | VWAP entry, trailing TP, multi-TF exit | +47% | **0%** |
| Dynamic hold | 49-52 | Confidence/prob-based hold | +47% | **0% (fake)** |
| Feature engineering | 53-55 | FracDiff, FFT, cross-asset | +22% | **-25%** |
| VSN (TFT) | 56-57 | Variable Selection Network | +18% | **-28%** |
| **Final (VSN OFF)** | **57** | **Confirmed baseline** | **+46%** | **+46%** |

## What Works (Proven)

| Technique | Impact | Notes |
|-----------|--------|-------|
| Full history data (6 years) | +15% | 15m→1h resample |
| Dropout 0.2 | +14% | Was 0.1, doubled returns |
| R-Drop α=1.0 | +3-5% | Consistency regularization |
| Fusion dim 192 | +7% | More capacity for 382 features |
| Mixup Beta(0.2) | Stable | Data augmentation |
| BCE + pos_weight | Critical | Prevents NO_TRADE |
| L_equity removal | +60% | Prediction ≠ execution |
| numba + vectorization | 70× faster | User optimized TBM/fracdiff |

## What Doesn't Work (Proven)

| Technique | Effect | Iter |
|-----------|--------|------|
| SWA | -4% | 33 |
| Multi-seed ensemble avg | Dilutes signal | 35 |
| Manual regime features | -10% | 39-40 |
| Regime expert partition | -22% | 40 |
| Label smoothing | No effect | 41 |
| VSN (Variable Selection) | -28% on full data | 56-57 |
| FracDiff features | -16% | 55 |
| FFT frequency features | -16% | 55 |
| Cross-asset (ETH/SOL) | -12% | 55 |
| VWAP precision entry | +1.7% (marginal) | 45 |
| Grid entry | +0.1% | 46 |
| Trailing TP (1m) | -20% | 46 |
| Multi-TF consensus exit | -20% | 48 |
| Confidence-scaled hold | = fixed×2.0 (fake) | 49-50 |
| Longer training (>50ep) | Overfitting | 44 |
| Larger model (>400K) | Diminishing returns | 42 |
| Batch size 16384 | LR schedule collapse | 51 |

## Key Insights

1. **Data quantity > everything** — 6 years vs 2 years was the single biggest improvement
2. **Regularization > architecture** — Dropout/R-Drop gave more than model size changes
3. **Feature additions hurt** — factory_v2's 382 features are already sufficient/oversaturated
4. **Execution layer is not the bottleneck** — Model prediction quality determines returns
5. **Confidence head is broken** — Always outputs ~1.0, not useful for dynamic decisions
6. **Hold period × return is linear** — ×2 hold = ×2.6 return (but ×2 capital time)
7. **Seed variance indicates overfitting** — Even with all regularization, std 4.6%

## Production Config

```python
PLEv4(
    expert_hidden=128, expert_output=64, fusion_dim=192,
    dropout=0.2, use_vsn=False, n_strategies=32,
)
# Training: epochs=50, BS=2048, lr=5e-4, patience=7
# R-Drop α=1.0, Mixup Beta(0.2), AdamW wd=1e-4
# Workers: 12 train, 4 val (deterministic seeding)
# Data: 2020-06-01 to 2026-02-28, 15m→1h resample
```

## Next Steps

Model level has converged. Remaining high-impact directions:

1. **Multi-task learning** — Train on 224 coins simultaneously (shared representation)
2. **W2 bear market fix** — Window 2 consistently negative (-7% to +1%)
3. **Position manager** — Multi-slot, leverage, strategy allocation (RL prep)
4. **Paper trading** — NautilusTrader deployment
5. **Data pipeline** — Polars optimization, real-time feature computation

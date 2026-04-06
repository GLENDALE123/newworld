# ultraTM Final Milestone — 75 Iterations

## Portfolio Performance (Walk-Forward, Seed 42)

```
Top 5 Coins (3 windows each):
  BCH:  [+259%, +91%, +76%]  → mean +142%  ✓ All positive
  ONT:  [+141%, +141%, +23%] → mean +102%  ✓ All positive
  BTC:  [+185%, +69%, -9%]   → mean  +82%
  LINK: [+157%, +62%, +6%]   → mean  +75%  ✓ All positive
  ATOM: [+11%, +148%, +19%]  → mean  +59%  ✓ All positive

  Portfolio Mean: +92%
  All-Window Positive: 4/5 coins (80%)
```

## Alpha Sources (Verified, No Lookahead)

| Source | Feature Count | Impact | Mechanism |
|--------|-------------|--------|-----------|
| OI Divergence | 6 | +60% | OI↓Price↑ = deleverage bull |
| Funding Rate | 3 | ~5% | Extreme funding = contrarian |
| LS Ratio | 6 | ~2% | Crowd positioning = contrarian |
| Base (factory_v2) | 392 | Baseline | Multi-TF price/volume/flow |
| **Total** | **~409** | **+74% BTC** | |

## Architecture

```
PLE v4
├── Feature Partitions: price(~290), volume(~100), metrics(~11), derivatives(~9)
├── Experts: 256 hidden → 128 output per partition
├── Attention-Gated Fusion: 256 dim
├── Multi-Label Sigmoid: 32 strategies
├── Dropout 0.2, R-Drop α=1.0, Mixup Beta(0.2)
├── Total: ~728K parameters
└── Training: 50ep, BS=2048, AdamW, OneCycleLR, patience=7
```

## Key Learnings (75 Iterations)

### What Produces Alpha
1. **Derivatives data** (OI, funding, LS) — the dominant signal
2. **Full historical data** (2020~) — more data = better generalization
3. **Multi-coin pretrain** — reduces std from 4% to 1.8%
4. **Per-coin fine-tune** — enables multi-asset portfolio

### What Doesn't Work
- Feature additions (FracDiff, FFT, cross-asset correlation, regime features)
- Execution layer optimization (VWAP, trailing, grid, multi-TF exit)
- VSN (Variable Selection Network) on full data
- Longer training, larger batch on small data
- Confidence-based dynamic holding (= just "hold longer")
- SWA, label smoothing, focal loss

### Architecture Insights
- Dropout 0.2 > 0.1 (doubled returns)
- Expert output 64 is optimal for BTC, 128 for multi-coin
- Fusion 192-256 sweet spot (128 too small, 512 overfits)
- R-Drop α=1.0 improves seed stability
- num_workers=12 with deterministic seeding for reproducibility

## Production Files

```
run_multicoin.py  — Multi-coin pipeline (Top 5 portfolio)
run_production.py — Single-coin pipeline (BTC)
ultrathink/       — Cached feature/label pipeline
execution/        — Position Manager, precision entry
ple/              — PLE v4 model, trainer, loss
labeling/         — TBM v2 (numba vectorized)
features/         — Feature factory v2
```

## Next Phase: Paper Trading

1. NautilusTrader strategy update (latest config)
2. Binance testnet connection
3. Real-time feature computation pipeline
4. Performance monitoring dashboard
5. Gradual capital scale-up (1% → 5% → full)

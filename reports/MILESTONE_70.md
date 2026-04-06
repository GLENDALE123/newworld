# ultraTM Milestone Report — 70 Iterations

## Final Performance

```
Walk-Forward (3 windows × 3 seeds, 2020-06 ~ 2026-02):

  Scratch + OI/Funding/LS:
    Mean:  +73.73%
    Std:   4.1%
    Seeds: [+68%, +75%, +78%]

  12-coin Pretrain + OI/Funding:
    Mean:  +62.15%
    Std:   1.8%
    Seeds: [+64%, +60%, +63%]

  Baseline (no derivatives features):
    Mean:  +11.86%
    Std:   0.4%
```

## Alpha Sources (Verified)

| Source | Impact | Mechanism |
|--------|--------|-----------|
| **OI divergence** | **+60%** | OI↓Price↑ = deleveraging bull (+0.18%/day) |
| **Funding rate** | ~+5% | Extreme funding = contrarian signal |
| **LS ratio** | ~+2% | Crowd positioning = contrarian alpha |
| Taker ratio | 0% | No predictive power |
| BTC-ETH corr | 0% | No incremental alpha |
| FracDiff/FFT | Negative | Redundant with existing features |

## Architecture

```
PLE v4 (e256_o128_f256, 728K params)
├── Feature-Partitioned Experts (price/volume/metrics)
├── Attention-Gated Fusion
├── Multi-Label Sigmoid (32 strategies)
├── Derivatives features: OI div + funding + LS ratio (17 features)
├── Dropout 0.2, R-Drop α=1.0, Mixup Beta(0.2)
├── 409 total features, BS=2048, 50 epochs
└── Optional: 12-coin pre-train for stability (std 4.1% → 1.8%)
```

## Infrastructure

- **UltraThink**: Cached feature/label pipeline (1.4s reload)
- **Position Manager**: Multi-slot, RL-ready (state/action/reward)
- **Seed Reproducibility**: worker_init_fn + generator
- **Vectorized**: numba TBM, convolve fracdiff
- **Data**: 220 coins, 15m from 2020-01 (Binance Vision)
- **Lookahead**: Verified PASS (5 tests)

## Next Steps

1. **Paper trading** — NautilusTrader + Binance testnet
2. **Per-coin fine-tune** — Multi-asset with individual models
3. **Position Manager backtest** — Multi-slot with fine-tuned models
4. **Real-time pipeline** — WebSocket → features → model → orders
5. **RL integration** — Position allocation optimization

# ultraTM — Crypto Futures Trading System

Institutional-grade quantitative trading system for Binance USDT-M futures.
88 iterations of systematic R&D. 18 alpha coins verified.

## Quick Start

```bash
# 1. Backtest single coin
python run_production.py --symbol DUSKUSDT

# 2. Backtest portfolio (18 coins)
python run_multicoin.py

# 3. Paper trading
python run_live.py --paper

# 4. Data download
python -m data.download_binance_vision --symbols DUSKUSDT CHRUSDT BCHUSDT
```

## Performance (OOS Verified, 3-seed × 3-window)

| Coin | Alpha/trade | Std | Confidence |
|------|-----------|-----|------------|
| DUSK | +0.52% | 0.19% | ★★★ |
| CHR | +0.48% | 0.01% | ★★★★★ |
| BCH | +0.24% | 0.01% | ★★★★★ |
| + 15 more coins | +0.10-0.41% | varies | ★★-★★★ |

**Key finding: Alpha exists in altcoins, NOT in BTC.**

## Architecture

```
PLE v4 (Progressive Layered Extraction)
├── Feature-Partitioned Experts (price/volume/metrics)
├── Attention-Gated Fusion (256 dim)
├── Multi-Label Sigmoid (32 strategies)
├── Dropout 0.2 + R-Drop α=1.0 + Mixup
└── ~860K params per coin model
```

## Alpha Source

Base price/volume features from factory_v2 (392 features + 240 sequence).
**NO derivatives features needed** (OI/Funding proven harmful in iter 80-81).

Less liquid altcoins have persistent price patterns that the PLE model captures.

## Project Structure

```
run_live.py           — Paper/live trading loop
run_multicoin.py      — Multi-coin backtest
run_production.py     — Single-coin pipeline
ultrathink/           — Cached feature/label pipeline
ple/                  — PLE v4 model (model_v4, trainer_v4, loss_v4)
strategy/             — PortfolioSignalGenerator
execution/            — PositionManager (RL-ready)
labeling/             — TBM v2 (numba vectorized)
features/             — Feature factory v2
models/production_v2/ — Saved models + config
experiments/          — 88 iteration scripts
reports/              — 88 iteration reports
```

## 88 Iterations Summary

- **Iter 1-32**: PLE architecture, baseline +11%
- **Iter 33-43**: Data expansion + regularization → +47%
- **Iter 45-52**: Execution layer (no improvement)
- **Iter 55-57**: Feature additions (all negative)
- **Iter 58-65**: Multi-coin pretrain (std reduction)
- **Iter 66-70**: OI features (+60% return, later proven fake)
- **Iter 80-82**: Honest assessment — only altcoin base features work
- **Iter 83-88**: 18 alpha coins, top 3 fully verified

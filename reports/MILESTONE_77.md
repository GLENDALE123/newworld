# ultraTM Milestone — 77 Iterations

## Final Portfolio (3-Seed × 3-Window Validated)

```
Pretrain Portfolio (10-coin pretrain → per-coin fine-tune):
  BTC:  +65%  std 0.9%   ← most stable
  BCH:  +65%  std 1.5%   ← 15x stability improvement
  ONT:  +82%  std 8.7%   ← highest return
  ──────────────────────
  Mean: +71%  std 3.7%

Scratch Portfolio (no pretrain):
  BTC:  +79%  std 4.4%
  BCH: +113%  std 22.6%
  ONT: +114%  std 29.2%
  ──────────────────────
  Mean: +102% std 18.7%
```

## Evolution (77 Iterations)

| Phase | Key Discovery | Impact |
|-------|--------------|--------|
| 1-32 | PLE v4 multi-label | Baseline +11% |
| 33 | Full history (15m→1h) | +26% |
| 36-37 | Dropout 0.2 + R-Drop | +40% |
| 43 | Architecture tuning | +47% |
| **66** | **OI/Funding features** | **+72% (+60% from derivatives!)** |
| 70 | + LS ratio | +74% |
| 72-75 | Multi-coin scan | Top 5: +92% |
| **77** | **Pretrain stabilization** | **+71% std 3.7% (risk-adjusted optimal)** |

## Alpha Sources

```
OI Divergence:     +60%  (dominant — verified +0.18%/day)
Funding Rate:       ~5%  (extreme funding = contrarian)
LS Ratio:           ~2%  (crowd positioning = contrarian)
Multi-coin Pretrain: 0%  (no alpha, but 5-15x std reduction)
```

## System Architecture

```
Data Layer:
  220 coins × Binance Vision (2020~)
  UltraThink cached pipeline (1.4s reload)

Feature Layer:
  factory_v2: 392 base features (multi-TF price/volume/flow)
  + 9 OI/Funding features (verified alpha)
  + 6 LS ratio features
  + ~240 sequence features (lag/chg)
  Total: ~409 features

Model Layer:
  PLE v4 (e256_o128_f256, ~728K params)
  Dropout 0.2, R-Drop α=1.0, Mixup Beta(0.2)
  32 strategies (4 styles × 2 dirs × 4 regimes)
  Optional: 10-coin pretrain for stability

Execution Layer:
  Position Manager (multi-slot, RL-ready)
  SMA50 adaptive thresholds (0.40/0.55)
  EV-based strategy selection
  DD-based position sizing

Validation:
  3-window walk-forward × 3 seeds
  Lookahead bias: PASS (5 tests)
  Seed reproducibility: worker_init_fn + generator
```

## Files

```
run_multicoin.py    — Multi-coin production runner
run_production.py   — Single-coin runner
ultrathink/         — Cached pipeline (cache.py, pipeline.py)
ple/                — PLE v4 (model_v4, trainer_v4, loss_v4)
labeling/           — TBM v2 (numba vectorized)
features/           — Factory v2
execution/          — Position Manager, precision entry
experiments/        — 77 iteration scripts
reports/            — 77 iteration reports + milestones
```

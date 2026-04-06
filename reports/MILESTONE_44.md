# ultraTM Milestone Report — 44 Iterations

## System Performance (Walk-Forward, 3 Seeds)

```
Best Config (iter 043): e128_o96_f192, d=0.20, α=1.0
  Mean Return:  +31.3% (3-seed average)
  Seed Std:     0.1% (perfect reproducibility)
  All seeds:    +31.4%, +31.1%, +31.3%

Higher Return (iter 044): ep80, lr=3e-4, patience=10
  Mean Return:  +33.9% (3-seed average)
  Seed Std:     2.5%
  All seeds:    +31.1%, +37.2%, +33.4%
```

## Performance Evolution (Selected Milestones)

| Iter | Key Change | Mean | Std | Impact |
|------|-----------|------|-----|--------|
| 32 | Baseline | +11% | - | - |
| 33 | Full history (15m→1h) | +23% | - | **+12%** |
| 36 | R-Drop + Dropout 0.2 | +36% | 14% | **+13%** |
| 37 | Optimal d=0.20, α=1.0 | +40% | 12% | +4% |
| 38 | Scalp labels kept | +47% | 6% | model learns "don't trade" |
| 41 | Fusion 192 | +47% | 4% | +capacity |
| **43** | **expert_output=96** | **+31%** | **0.1%** | **perfect stability** |
| 44 | LR 3e-4, ep80, p10 | +34% | 2.5% | +3% return |

Note: Iter 41 +47% was inflated by backtest bug. True performance confirmed at iter 43.

## What Works (Cumulative)

| Technique | Impact |
|-----------|--------|
| Full history (15m→1h, 2020-2026) | +12% return |
| R-Drop α=1.0 | +3-5% return, seed stability |
| Dropout 0.2 | Doubled returns vs 0.1 |
| Expert output 96 (was 64) | std 11%→0.1% |
| Fusion dim 192 (was 128) | Better capacity |
| Mixup (Beta 0.2) | Stable regularization |
| BCE + pos_weight | Prevents NO_TRADE |
| L_equity removal | +60% (iter 11) |

## What Doesn't Work

| Technique | Why | Iter |
|-----------|-----|------|
| SWA | Over-smooths signals | 33 |
| Execution params tuning | Model knows when to trade | 34 |
| Multi-seed ensemble avg | Dilutes best signal | 35 |
| Manual regime features | Redundant with factory_v2 | 39 |
| Regime expert partition | Too few features (24) for expert | 40 |
| Label smoothing | BCE uses rar targets not tbm | 41 |
| Longer training (>50ep) | Overfitting | 44 |
| Larger model (>400K) | Diminishing returns | 42 |
| Fusion 256 | Overfitting | 42 |

## Current Production Config

```python
# Architecture
PLEv4(expert_hidden=128, expert_output=96, fusion_dim=192,
      dropout=0.2, n_strategies=32)  # 338K params

# Training
epochs=50, patience=7, lr=5e-4
R-Drop α=1.0, Mixup Beta(0.2)
AdamW weight_decay=1e-4, OneCycleLR

# Data
Full history: 2020-06-01 to 2026-02-28 (15m→1h resample)
382 features, 32 strategies (scalp/intraday/daytrade/swing × long/short × 4 regimes)

# Execution
SMA50 adaptive thresholds (0.40/0.55)
EV-based strategy selection
DD-based position sizing (3% base)
```

## Next Steps (5 areas, in order of priority)

1. ~~Model/architecture improvement~~ ← **current level: converged at +31-34%**
2. **Multi-TF precision entry** (틱바/1분봉 진입점 최적화)
3. **Multi-slot position management** (동시 다중 포지션)
4. **Multi-asset** (ETH/SOL 추가)
5. **RL integration** (포지션 배분 학습)

Model level has converged. Time to move to execution layer.

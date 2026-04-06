# Next Steps Design Document

## Verified Alpha Sources (from data analysis)

### 1. OI-Price Divergence Signal (★★★)
- OI↓ + Price↑ → next 1d +0.71% (253 events, strong)
- OI↑ + Price↓ → next 1d -0.15% (293 events, weak but correct)
- Implementation: Add as TBM regime (not feature — features addition proven harmful)
  - `deleverage_bull`: OI dropping but price rising → wider TP
  - `leverage_bear`: OI rising but price dropping → tighter SL

### 2. Funding Rate Extreme Reversal (★★)  
- Extreme high funding → 4h -0.16% (short-term reversal)
- But 1d +0.05% (recovery) → not reliable for day+ holding
- Only useful for scalp-level timing

### 3. Multi-Coin Pre-training (★★★)
- Proven: std 6.3% → 0.6% (iter 061)
- Trade-off: higher stability but lower absolute return
- Next: Larger model (1M+) to capture both

## Planned Experiments (priority order)

### Iter 064: OI Divergence Regime in TBM
- Add 5th regime "deleverage" to TBM v2
- OI↓+Price↑: set TP=3×ATR (let it run)
- OI↑+Price↓: set SL=0.8×ATR (tight stop)
- Requires: metrics data aligned with klines

### Iter 065: Full Data Large Pre-train
- Needs: 220 coins data update (waiting)
- 728K model, 20+ coins, 200K+ combined samples
- Target: +46% return + std <1%

### Iter 066: Position Manager Prototype
- Multi-slot simulator (N concurrent positions)
- EV-based slot allocation
- Leverage per slot (1x-5x)
- Account state tracking (DD, equity, exposure)
- This is RL prep infrastructure

### Iter 067: Funding Rate as TBM Condition
- During extreme funding (>0.03%): only allow contrarian scalp
- Normal funding: standard strategies
- Integrated into TBM v2 regime detection

## Architecture Evolution Path

```
Current:
  15m data → features → PLE → 32 strategies → best EV → trade

Phase 2 (next):
  Multi-TF data → features → Pre-trained PLE → strategies
  + OI/Funding regime → TBM adaptation
  + Position Manager → multi-slot allocation

Phase 3 (future):
  Real-time data → features → PLE ensemble
  + RL position manager → optimal allocation
  + Multi-asset across 20+ coins
  + Execution via NautilusTrader
```

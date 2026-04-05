# ultraTM Research Summary — 14 Iterations

## System Architecture (Final)

```
Raw Data (78 symbols: OHLCV, tick_bar, metrics, funding)
    │
    ├── Feature Factory v2 (197+ features)
    │   ├── Multi-TF price (ATR, returns, vol at 5m/15m/1h/4h)
    │   ├── Order flow (CVD, buy ratio, trade intensity)
    │   ├── Derivatives (OI, funding, long/short ratios)
    │   ├── Cross-asset (ETH/SOL correlation, lead-lag)
    │   └── Fractional differentiation
    │
    ├── Multi-Label TBM v2 (ATR-based dynamic barriers)
    │   ├── Intraday (15m): 2×ATR TP, 1×ATR SL, 1h hold
    │   ├── Swing (4h): 3×ATR TP, 1.5×ATR SL, 7d hold
    │   ├── Long + Short × 4 regimes = 16 strategies
    │   ├── Fee-deducted RAR labels (0.08% round-trip)
    │   └── Sample weights (magnitude × speed)
    │
    └── PLE v4 (Multi-Label, Independent Sigmoid)
        ├── Feature-partitioned experts (price/volume/metrics/cross)
        ├── Attention-gated fusion
        ├── 16 independent sigmoid outputs (per strategy)
        ├── BCE + Calibration loss (NO equity loss)
        └── Execution: SMA filter + EV-based selection + DD recovery

```

## Walk-Forward Results (Best Config: Iter 12)

| Window | Period | Return | B&H | Excess | WR |
|--------|--------|--------|------|--------|-----|
| 1 | 2025-01~05 (bull) | -4.23% | +9.52% | -13.75% | 44.4% |
| 2 | 2025-05~10 (flat) | +9.72% | +4.05% | +5.67% | 52.1% |
| 3 | 2025-10~26-02 (bear) | -1.15% | -39.83% | +38.69% | 46.0% |
| **Mean** | | **+1.45%** | **-8.75%** | **+10.20%** | |

## Key Findings

### What Works
1. **ATR-based dynamic barriers** — fee-viable TP/SL that adapts to volatility
2. **Multi-label sigmoid** — independent strategy activation, natural NO_TRADE
3. **Separating prediction from execution** — L_equity removal = biggest improvement
4. **Hard SMA(50) directional filter** — prevents counter-trend trades
5. **Fee-aware training** — 0.08% fee in labels prevents overtrading
6. **Sample weights** — faster+bigger wins get higher priority

### What Doesn't Work
1. **Equity loss in training** (L_equity) — gradient explosion, saturates
2. **Softmax strategy selection** — forces one pick, no NO_TRADE
3. **Soft SMA filter** — gray zone generates noise
4. **Swing-only (removing intraday)** — loses short-term edge
5. **Single model without ensemble** — high variance across seeds
6. **Fixed percentage barriers** — can't adapt to volatility regime

### Unresolved Issues
1. **Bull market underperformance** — model has bearish bias from data
2. **Walk-forward mean +1.45%** — modest absolute return
3. **Multi-asset** — ETH/SOL timezone issues unresolved
4. **Execution layer** — NautilusTrader not connected
5. **No live testing** — all results are backtested

## Iteration History

| Iter | Return | Type | Key Change |
|------|--------|------|-----------|
| 1 | -12.56% | single | PLE v4 baseline |
| 2 | +12.65% | single | 15m base + 4x data |
| 5 | +27.86% | single | L_equity fix (DD 10.5%) |
| 8 | +4.75% | single | 3-model ensemble (DD 3.57%) |
| 9 | +11.56% | single | 5-model + adaptive Kelly |
| 11 | +73.57% | single | **L_equity REMOVED** (Calmar 4.85) |
| 12 | +1.45% | WF-3 | Walk-forward validation |
| 13 | -1.64% | WF-3 | No SMA (bull improved, bear worse) |
| 14 | -2.03% | WF-3 | Soft SMA (worse than hard) |

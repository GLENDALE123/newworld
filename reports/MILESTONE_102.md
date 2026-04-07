# ultraTM — 102 Iterations Final

## Result: $500 → $24,819 (backtest), ~$6K-$12K (realistic)

## Key Discoveries
1. **Alpha in altcoins, NOT BTC** — market efficiency varies by liquidity
2. **Base features sufficient** — OI/funding features are harmful
3. **Selective trading** — top 20% EV, WR 55-64%
4. **Capital-adaptive leverage** — $500→20x, $5K→10x, $50K→3x
5. **Alpha decays** — monthly retraining required

## Production Coins (per-coin top 20% EV)
| Coin | WR | Alpha/trade |
|------|-----|------------|
| CRV | 64% | +1.31% |
| UNI | 57% | +0.45% |
| CHR | 56% | +0.99% |
| DUSK | 50% | +1.59% |

## 102 Iterations Timeline
- 1-43: PLE architecture → +47% (overfit)
- 44-65: Multi-coin pretrain → std 0.7%
- 66-70: OI features → +72% (later proven fake)
- 80-82: Honest assessment → only altcoin base features work
- 83-94: 25 alpha coins discovered
- 95-98: Breakout + selective trading
- 99-101: $500 → $24,819 with capital strategy
- 102: Universal model hybrid approach

## Files
```
run_live.py, run_retrain.py, run_monitor.py, run_multicoin.py
execution/: signal_filter, breakout_filter, capital_strategy, position_manager
ple/: model_v4, model_v5, trainer_v4, loss_v4
models/production_v5/: 5 saved models + universal
```

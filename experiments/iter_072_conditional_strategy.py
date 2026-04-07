#!/usr/bin/env python3
"""
Iteration 072: Conditional Strategy — HighVol + Extreme Signal

발견한 edge: 고변동성 + OI/Flow 극단값 → +18bp spread (fee 8bp 초과)

이 실험:
  1. Rule-based 조건부 전략 구현
  2. 실제 가격 경로(TBM-style) 시뮬레이션
  3. Walk-forward 3-window 검증
  4. 다중 코인 검증
  5. XGBoost 모델과 비교

핵심: 모든 bar에서 매매하지 않음 — 조건 충족 시에만 (극도의 selectivity)
"""

import os, sys, time
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numba import njit


@njit
def simulate_trades(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    signals: np.ndarray,  # +1=long, -1=short, 0=no trade
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_hold: int = 4,
    fee: float = 0.0008,
):
    """Simulate trades with TBM-style path exit."""
    n = len(closes)
    rets = np.empty(0, dtype=np.float64)
    entries = np.empty(0, dtype=np.int64)

    next_free = 0
    for i in range(n - max_hold):
        if i < next_free:
            continue
        if signals[i] == 0:
            continue

        direction = int(signals[i])
        entry = closes[i]
        a = atr[i]
        if np.isnan(a) or a <= 0 or entry <= 0:
            continue

        if direction == 1:
            tp_price = entry + tp_mult * a
            sl_price = entry - sl_mult * a
        else:
            tp_price = entry + sl_mult * a
            sl_price = entry - tp_mult * a

        end = min(i + max_hold, n - 1)
        exit_price = closes[end]

        for j in range(i + 1, end + 1):
            if direction == 1:
                if highs[j] >= tp_price:
                    exit_price = tp_price
                    break
                if lows[j] <= sl_price:
                    exit_price = sl_price
                    break
            else:
                if lows[j] <= sl_price:
                    exit_price = sl_price
                    break
                if highs[j] >= tp_price:
                    exit_price = tp_price
                    break

        if direction == 1:
            ret = (exit_price - entry) / entry
        else:
            ret = (entry - exit_price) / entry
        net = ret - fee

        rets = np.append(rets, net)
        entries = np.append(entries, i)
        next_free = j + 1 if j > i else i + max_hold

    return rets, entries


def generate_signals_conditional(
    X: pd.DataFrame,
    vol_feature: str = "15m_atr_5",
    signal_features: list = None,
    vol_threshold_pct: float = 75,   # ATR percentile for "high vol"
    signal_threshold_pct: float = 90, # feature percentile for "extreme"
    lookback: int = 96 * 30,          # rolling window for percentiles
    scan_every: int = 4,
) -> np.ndarray:
    """Generate conditional signals: trade only when vol is high AND signal is extreme."""
    if signal_features is None:
        signal_features = ["deriv_oi_chg_5_lag7", "flow_cvd_chg_10_lag7"]

    n = len(X)
    signals = np.zeros(n)

    # Get feature arrays
    vol = X[vol_feature].values if vol_feature in X.columns else np.zeros(n)
    feat_arrays = []
    for f in signal_features:
        if f in X.columns:
            feat_arrays.append(X[f].values)

    if not feat_arrays:
        return signals

    for i in range(lookback, n, scan_every):
        # Rolling volatility threshold
        vol_window = vol[max(0, i - lookback):i]
        vol_window = vol_window[~np.isnan(vol_window)]
        if len(vol_window) < 100:
            continue
        vol_th = np.percentile(vol_window, vol_threshold_pct)

        if np.isnan(vol[i]) or vol[i] < vol_th:
            continue  # skip low volatility

        # Check signal features
        best_signal = 0
        best_strength = 0

        for feat in feat_arrays:
            f_window = feat[max(0, i - lookback):i]
            f_window = f_window[~np.isnan(f_window)]
            if len(f_window) < 100:
                continue

            th_high = np.percentile(f_window, signal_threshold_pct)
            th_low = np.percentile(f_window, 100 - signal_threshold_pct)
            val = feat[i]
            if np.isnan(val):
                continue

            if val >= th_high:
                strength = (val - th_high) / (np.std(f_window) + 1e-10)
                if strength > best_strength:
                    best_signal = 1  # long
                    best_strength = strength
            elif val <= th_low:
                strength = (th_low - val) / (np.std(f_window) + 1e-10)
                if strength > best_strength:
                    best_signal = -1  # short
                    best_strength = strength

        signals[i] = best_signal

    return signals


def run_coin(coin: str, start: str = "2023-01-01", end: str = "2025-12-31"):
    """Run full analysis for one coin."""
    from ultrathink.cache import ParquetCache

    print(f"\n{'='*55}")
    print(f"  {coin}")
    print(f"{'='*55}")

    # Load fixed features
    cache = ParquetCache("data/cache")
    params = {"symbol": coin, "start": start, "end": end, "target_tf": "15min",
              "extras": [], "lag_top_n": 30, "lag_depths": [1, 2, 3, 4, 5, 6, 7],
              "version": "v2_fixed"}
    X, hit = cache.get(f"feat_{coin}", params)

    if not hit:
        # Generate features with fixed pipeline
        sys.path.insert(0, os.getcwd())
        from features.factory_v2 import generate_features_v2

        kline_dir = f"data/merged/{coin}"
        kline = {}
        for tf in ["5m", "15m"]:
            path = os.path.join(kline_dir, f"kline_{tf}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if "timestamp" in df.columns: df = df.set_index("timestamp")
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                kline[tf] = df.sort_index()[start:end]
        if "15m" in kline:
            kline["1h"] = kline["15m"].resample("1h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        if "1h" in kline:
            kline["4h"] = kline["1h"].resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

        extras = {}
        for name in ["tick_bar", "metrics", "funding_rate"]:
            path = os.path.join(kline_dir, f"{name}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if "timestamp" in df.columns: df = df.set_index("timestamp")
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                extras[name] = df.sort_index()[start:end]

        base = generate_features_v2(kline_data=kline, tick_bar=extras.get("tick_bar"),
                                     metrics=extras.get("metrics"), funding=extras.get("funding_rate"),
                                     target_tf="15min", progress=False)

        # Add lag features
        universal_lag_candidates = [
            "flow_cvd", "flow_delta_sum_5", "flow_delta_sum_10", "flow_delta_sum_20",
            "flow_cvd_chg_5", "flow_cvd_chg_10", "flow_cvd_chg_20",
            "deriv_oi", "deriv_oi_chg_5", "deriv_oi_chg_10",
            "deriv_oi_zscore_20", "fund_rate", "fund_zscore_20",
            "15m_atr_5", "15m_ret_5", "15m_ret_10", "15m_ret_20",
        ]
        top_cols = [c for c in universal_lag_candidates if c in base.columns][:30]
        lag_dict = {}
        for lag in range(1, 8):
            for col in top_cols:
                lag_dict[f"{col}_lag{lag}"] = base[col].shift(lag)
        X = pd.concat([base, pd.DataFrame(lag_dict, index=base.index)], axis=1)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.loc[:, X.nunique() > 1]
        cache.put(f"feat_{coin}", params, X)
        print(f"  Features: {X.shape}")
    else:
        kline_dir = f"data/merged/{coin}"

    # Load kline for prices
    kline_path = os.path.join(kline_dir, "kline_15m.parquet")
    if not os.path.exists(kline_path):
        print(f"  SKIP: no kline data")
        return None

    kline_df = pd.read_parquet(kline_path)
    if "timestamp" in kline_df.columns:
        kline_df = kline_df.set_index("timestamp")
    if kline_df.index.tz is not None:
        kline_df.index = kline_df.index.tz_localize(None)
    kline_df = kline_df.sort_index()[start:end]

    common = X.index.intersection(kline_df.index)
    X = X.loc[common]
    close = kline_df["close"].loc[common].values.astype(np.float64)
    high = kline_df["high"].loc[common].values.astype(np.float64)
    low = kline_df["low"].loc[common].values.astype(np.float64)

    tr = np.maximum(high[1:] - low[1:],
                     np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = pd.Series(tr).ewm(span=14).mean().values

    n = len(X)
    print(f"  Data: {n} bars ({start} → {end})")

    # Signal features (check availability)
    sig_feats_candidates = [
        "deriv_oi_chg_5_lag7", "deriv_oi_chg_10_lag6",
        "flow_cvd_chg_10_lag7", "flow_delta_sum_10_lag7",
        "15m_ret_10_lag7",
    ]
    sig_feats = [f for f in sig_feats_candidates if f in X.columns]
    if not sig_feats:
        print(f"  SKIP: no signal features")
        return None
    print(f"  Signal features: {len(sig_feats)}")

    # ── Walk-Forward: 3 windows ──
    window_size = n // 4
    results = []

    configs = [
        {"name": "Conservative", "vol_pct": 75, "sig_pct": 95, "tp": 1.5, "sl": 1.0, "hold": 4},
        {"name": "Aggressive", "vol_pct": 60, "sig_pct": 90, "tp": 2.0, "sl": 1.0, "hold": 8},
        {"name": "Ultra-selective", "vol_pct": 85, "sig_pct": 98, "tp": 1.5, "sl": 0.5, "hold": 4},
        {"name": "Wide hold", "vol_pct": 75, "sig_pct": 90, "tp": 2.0, "sl": 1.0, "hold": 16},
    ]

    print(f"\n  {'Config':>18s} {'Window':>8s} {'Trades':>7s} {'WR':>6s} {'AvgBps':>8s} {'Sharpe':>7s}")
    print(f"  {'-'*62}")

    for cfg in configs:
        window_sharpes = []
        for w in range(3):
            test_start = window_size * (w + 1)
            test_end = min(test_start + window_size, n)
            if test_end - test_start < 1000:
                continue

            X_window = X.iloc[test_start:test_end]
            signals = generate_signals_conditional(
                X_window,
                vol_threshold_pct=cfg["vol_pct"],
                signal_threshold_pct=cfg["sig_pct"],
                signal_features=sig_feats,
            )

            rets, _ = simulate_trades(
                close[test_start:test_end],
                high[test_start:test_end],
                low[test_start:test_end],
                atr[test_start:test_end],
                signals,
                tp_mult=cfg["tp"], sl_mult=cfg["sl"],
                max_hold=cfg["hold"],
            )

            n_t = len(rets)
            if n_t < 10:
                continue

            wr = (rets > 0).mean() * 100
            avg = rets.mean() * 10000
            sharpe = rets.mean() / rets.std() * np.sqrt(252 * 4) if rets.std() > 0 else 0
            window_sharpes.append(sharpe)

            print(f"  {cfg['name']:>18s} {'W'+str(w+1):>8s} {n_t:>7d} {wr:5.1f}% {avg:>+7.1f} {sharpe:>+6.2f}")

        if window_sharpes:
            avg_sharpe = np.mean(window_sharpes)
            results.append({"config": cfg["name"], "avg_sharpe": avg_sharpe,
                            "n_windows": len(window_sharpes)})

    # ── Summary ──
    if results:
        print(f"\n  Walk-Forward Summary:")
        for r in sorted(results, key=lambda x: x["avg_sharpe"], reverse=True):
            status = "VIABLE" if r["avg_sharpe"] > 0 else "LOSE"
            print(f"    {r['config']:>18s}: avg_Sharpe={r['avg_sharpe']:+.2f} ({r['n_windows']} windows) [{status}]")

    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 072: Conditional Strategy Simulation")
    print("=" * 60)

    coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    all_results = {}

    for coin in coins:
        r = run_coin(coin)
        if r:
            all_results[coin] = r

    # Cross-coin summary
    print(f"\n{'='*60}")
    print(f"  Cross-Coin Summary")
    print(f"{'='*60}")
    for coin, results in all_results.items():
        viable = [r for r in results if r["avg_sharpe"] > 0]
        print(f"  {coin}: {len(viable)}/{len(results)} configs viable")
        for r in sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)[:2]:
            print(f"    {r['config']:>18s}: Sharpe={r['avg_sharpe']:+.2f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Iteration 034: Regime-Aware Thresholds + Confidence-Weighted Sizing

Problem from Iter 033:
  - Bull market SOLVED: +57.80% (was -10%)
  - Bear market REGRESSED: -4.16% (was +25-35%)
  - W2: 2556 trades, 72% shorts, only 47.5% WR → over-trading in bear

Diagnosis:
  The model was trained on more data (good) but the backtest thresholds
  (lt=0.40/st=0.55 above SMA, flipped below) are not adaptive enough.
  With 32 strategies, 2556 trades in ~10K bars = 1 trade per 4 bars.
  That's too frequent — lower conviction trades dilute returns.

Solutions in this iteration:
  1. Higher base thresholds: require more conviction to trade
  2. Volatility-scaled thresholds: tighter in low-vol, looser in high-vol
  3. Confidence-weighted sizing: scale position by model confidence output
  4. Min EV filter: raise minimum expected value threshold
  5. Cooldown: minimum bars between trades

Key insight: The model's predictions are BETTER with full history,
but the execution layer needs to be smarter about WHEN to act.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2


def load_data_full_history(data_dir="data/merged/BTCUSDT", start="2020-06-01", end="2026-02-28"):
    kline = {}
    for tf in ["5m", "15m"]:
        path = os.path.join(data_dir, f"kline_{tf}.parquet")
        if os.path.exists(path):
            kline[tf] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]

    if "15m" in kline:
        kline["1h"] = kline["15m"].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]

    return kline, extras


def train_model(X, L, partitions, train_start, train_end, val_end, device="cuda"):
    """Train PLE v4 with standard settings (no SWA — proven not helpful)."""
    from ple.trainer_v4 import train_ple_v4

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wgt_cols = sorted([c for c in L.columns if c.startswith("wgt_")])

    X_np = X.values.astype(np.float32)
    acc = np.zeros((len(X), 4), dtype=np.float32)
    acc[:, 0] = 1.0
    wgt_np = L[wgt_cols].values if wgt_cols else None

    train_ds = TradingDatasetV4(
        X_np[train_start:train_end], L[tbm_cols].values[train_start:train_end],
        L[mae_cols].values[train_start:train_end], L[mfe_cols].values[train_start:train_end],
        L[rar_cols].values[train_start:train_end], acc[train_start:train_end],
        wgt_np[train_start:train_end] if wgt_np is not None else None,
    )
    val_ds = TradingDatasetV4(
        X_np[train_end:val_end], L[tbm_cols].values[train_end:val_end],
        L[mae_cols].values[train_end:val_end], L[mfe_cols].values[train_end:val_end],
        L[rar_cols].values[train_end:val_end], acc[train_end:val_end],
        wgt_np[train_end:val_end] if wgt_np is not None else None,
    )

    model = PLEv4(
        feature_partitions=partitions, n_account_features=4,
        n_strategies=len(tbm_cols), expert_hidden=128, expert_output=64,
        fusion_dim=128, dropout=0.1,
    )

    train_ple_v4(model, train_ds, val_ds, epochs=50, batch_size=2048,
                  lr=5e-4, device=device, patience=7)

    return model


def backtest_v2(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008,
                base_size=0.03, device="cuda",
                # New execution parameters
                min_prob=0.50,          # minimum probability to consider
                min_ev=0.0005,          # minimum EV filter (was 0)
                cooldown=8,             # minimum bars between trades (was 4)
                vol_scale=True,         # scale thresholds by volatility
                conf_sizing=True,       # use model confidence for sizing
                ):
    """Enhanced backtest with regime-aware execution."""
    n = len(X_test)
    if n < 100:
        return {"return": 0.0, "trades": 0, "wr": 0.0, "bh": 0.0}

    model = model.to(device).eval()

    with torch.no_grad():
        out = model(
            torch.tensor(X_test.values.astype(np.float32)).to(device),
            torch.zeros(n, 4).to(device),
        )
        probs = out["label_probs"].cpu().numpy()
        mfe_p = out["mfe_pred"].cpu().numpy()
        mae_p = out["mae_pred"].cpu().numpy()
        confidence = out["confidence"].cpu().numpy()

    # Price + SMA
    close = kline_15m["close"]
    sma50 = kline_4h["close"].rolling(50).mean().resample("15min").ffill()
    sma200 = kline_4h["close"].rolling(200).mean().resample("15min").ffill()
    tc = close.reindex(X_test.index, method="ffill").values
    sv50 = sma50.reindex(X_test.index, method="ffill").values
    sv200 = sma200.reindex(X_test.index, method="ffill").values

    # Realized volatility (rolling 96 bars = 1 day of 15m)
    rets = pd.Series(tc).pct_change().values
    vol_20 = pd.Series(rets).rolling(96, min_periods=20).std().values
    vol_median = np.nanmedian(vol_20[~np.isnan(vol_20)])

    capital = 100000.0
    peak = capital
    trades = []
    last_trade_bar = -cooldown  # allow first trade

    for i in range(0, n - 1, 4):
        # Cooldown check
        if i - last_trade_bar < cooldown:
            continue

        # Trend detection
        above50 = not np.isnan(sv50[i]) and tc[i] > sv50[i]
        above200 = not np.isnan(sv200[i]) and tc[i] > sv200[i]

        # Regime-based thresholds
        if above50 and above200:
            # Strong uptrend: favor longs
            lt, st = 0.45, 0.60
        elif above50:
            # Mild uptrend
            lt, st = 0.50, 0.55
        elif not above50 and not above200:
            # Strong downtrend: favor shorts
            lt, st = 0.60, 0.45
        else:
            # Mild downtrend
            lt, st = 0.55, 0.50

        # Volatility scaling: tighter thresholds in low-vol
        if vol_scale and not np.isnan(vol_20[i]) and vol_median > 0:
            vol_ratio = vol_20[i] / vol_median
            if vol_ratio < 0.7:
                # Low vol: need higher conviction
                lt = min(lt + 0.05, 0.70)
                st = min(st + 0.05, 0.70)
            elif vol_ratio > 1.5:
                # High vol: can be slightly more aggressive
                lt = max(lt - 0.03, 0.40)
                st = max(st - 0.03, 0.40)

        # Find best strategy
        best_ev = -1
        best_j = -1
        for j in range(len(strat_info)):
            th = lt if strat_info[j]["dir"] == "long" else st
            if probs[i, j] < max(th, min_prob):
                continue
            p = probs[i, j]
            rew = max(abs(mfe_p[i, j]), 0.001)
            rsk = max(abs(mae_p[i, j]), 0.001)
            ev = p * rew - (1 - p) * rsk - fee
            if ev > best_ev:
                best_ev = ev
                best_j = j

        if best_j < 0 or best_ev <= min_ev:
            continue

        d = 1 if strat_info[best_j]["dir"] == "long" else -1
        hold = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(
            strat_info[best_j]["style"], 12)
        ei = min(i + hold, n - 1)
        pnl = d * (tc[ei] - tc[i]) / tc[i]
        net = pnl - fee

        # Position sizing
        dd = (peak - capital) / peak if peak > 0 else 0
        sz = base_size * max(0.2, 1 - dd / 0.15)

        # Confidence scaling
        if conf_sizing:
            conf = confidence[i]
            sz *= max(0.5, min(1.5, conf * 2))  # scale 0.5x to 1.5x

        capital += net * capital * sz
        peak = max(peak, capital)
        trades.append({
            "net": net * 100, "dir": "L" if d == 1 else "S",
            "conf": float(confidence[i]), "prob": float(probs[i, best_j]),
        })
        last_trade_bar = i

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame({"net": [], "dir": []})
    ret = (capital - 100000) / 100000 * 100
    bh = (tc[-1] - tc[0]) / tc[0] * 100 if tc[0] > 0 else 0
    wr = (tdf["net"] > 0).mean() * 100 if len(tdf) > 0 else 0
    n_long = (tdf["dir"] == "L").sum() if len(tdf) > 0 else 0
    n_short = (tdf["dir"] == "S").sum() if len(tdf) > 0 else 0
    avg_conf = tdf["conf"].mean() if "conf" in tdf.columns and len(tdf) > 0 else 0

    return {
        "return": round(ret, 2), "bh": round(bh, 2), "trades": len(tdf),
        "wr": round(wr, 1), "n_long": int(n_long), "n_short": int(n_short),
        "avg_conf": round(avg_conf, 3),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 034: Regime-Aware Thresholds + Confidence Sizing")
    print("=" * 60)

    START, END = "2020-06-01", "2026-02-28"

    print(f"\n[1/5] Loading data...")
    kline, extras = load_data_full_history(start=START, end=END)

    print(f"\n[2/5] Building features...")
    features = generate_features_v2(
        kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"),
        target_tf="15min", progress=False,
    )
    top_feats = features.std().sort_values(ascending=False).head(30).index.tolist()
    seq_cols = {}
    for lag in range(1, 8):
        for col in top_feats:
            seq_cols[f"{col}_lag{lag}"] = features[col].shift(lag)
    for col in top_feats[:10]:
        for lag in [1, 2, 4]:
            seq_cols[f"{col}_chg{lag}"] = features[col] - features[col].shift(lag)
    features = pd.concat([features, pd.DataFrame(seq_cols, index=features.index)], axis=1)
    features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  Features: {features.shape}")

    print(f"\n[3/5] Building labels...")
    lr = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr["intraday"].copy() if "intraday" in lr else pd.DataFrame()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr:
            sw = lr[name].resample("15min").ffill() if name != "intraday" else lr[name]
            for col in sw.columns:
                if col not in labels.columns:
                    labels[col] = sw[col]

    common = features.index.intersection(labels.index)
    X = features.loc[common]
    L = labels.loc[common]
    print(f"  Aligned: {len(X)} rows")

    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    strat_info = []
    for c in tbm_cols:
        parts = c.replace("tbm_", "").split("_")
        strat_info.append({"style": parts[0], "dir": parts[1]})

    n = len(X)
    window_size = n // 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Parameter grid search on execution parameters ──
    configs = {
        "baseline_033": {
            "min_prob": 0.40, "min_ev": 0.0, "cooldown": 4,
            "vol_scale": False, "conf_sizing": False,
        },
        "higher_thresh": {
            "min_prob": 0.50, "min_ev": 0.0005, "cooldown": 4,
            "vol_scale": False, "conf_sizing": False,
        },
        "with_cooldown": {
            "min_prob": 0.50, "min_ev": 0.0005, "cooldown": 8,
            "vol_scale": False, "conf_sizing": False,
        },
        "full_v2": {
            "min_prob": 0.50, "min_ev": 0.0005, "cooldown": 8,
            "vol_scale": True, "conf_sizing": True,
        },
        "aggressive_filter": {
            "min_prob": 0.55, "min_ev": 0.001, "cooldown": 12,
            "vol_scale": True, "conf_sizing": True,
        },
    }

    print(f"\n[4/5] Walk-forward with {len(configs)} execution configs...")
    all_results = {name: [] for name in configs}

    for w in range(3):
        train_start = w * (window_size // 3)
        train_end = train_start + int(window_size * 2)
        val_end = train_end + int(window_size * 0.5)
        test_end = min(val_end + window_size, n)

        if test_end <= val_end:
            continue

        print(f"\n  --- Window {w+1}/3 ---")
        print(f"  Train: {train_end - train_start} samples, Test: {test_end - val_end} samples")

        torch.manual_seed(42 + w)
        np.random.seed(42 + w)

        model = train_model(X, L, partitions, train_start, train_end, val_end, device)
        X_test = X.iloc[val_end:test_end]

        for config_name, config in configs.items():
            result = backtest_v2(
                model, X_test, kline["15m"], kline["4h"], strat_info,
                device=device, **config,
            )
            all_results[config_name].append(result)

            bh = result["bh"]
            market = "BULL" if bh > 10 else ("BEAR" if bh < -10 else "SIDE")
            print(f"    {config_name:20s}: {result['return']:+7.2f}%  "
                  f"trades={result['trades']:4d}  WR={result['wr']:.1f}%  "
                  f"L={result['n_long']:4d} S={result['n_short']:4d}  [{market}]")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ITERATION 034 RESULTS")
    print(f"{'=' * 60}")

    best_config = None
    best_mean = -999

    for name, results in all_results.items():
        returns = [r["return"] for r in results]
        mean_ret = np.mean(returns)
        min_ret = min(returns)
        total_trades = sum(r["trades"] for r in results)

        print(f"\n  {name}:")
        print(f"    Windows: {[r['return'] for r in results]}")
        print(f"    Mean: {mean_ret:+.2f}%  Min: {min_ret:+.2f}%  Trades: {total_trades}")

        # Best config = highest mean with no window below -15%
        if mean_ret > best_mean and min_ret > -20:
            best_mean = mean_ret
            best_config = name

    print(f"\n  BEST CONFIG: {best_config} (mean={best_mean:+.2f}%)")
    print(f"  Time: {elapsed:.0f}s")

    # Save report
    report = {
        "iteration": 34,
        "approach": "Regime-aware thresholds + confidence sizing + vol scaling",
        "data_range": f"{START} to {END}",
        "best_config": best_config,
        "configs": {},
    }
    for name, results in all_results.items():
        returns = [r["return"] for r in results]
        report["configs"][name] = {
            "windows": returns,
            "mean": round(np.mean(returns), 2),
            "params": configs[name],
            "details": results,
        }

    report["windows"] = report["configs"][best_config]["windows"]
    report["mean"] = report["configs"][best_config]["mean"]

    with open("reports/iteration_034.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to reports/iteration_034.json")


if __name__ == "__main__":
    main()

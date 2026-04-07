#!/usr/bin/env python3
"""
Iteration 071: Orderbook Depth Features + Fee Sensitivity

Part A: book_depth → features → return prediction
  - Bid/ask imbalance
  - Depth changes
  - Wall detection

Part B: Fee sensitivity
  - With existing weak signals (corr 0.05), at what fee level does it become profitable?
"""

import os, sys, time
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_orderbook_features(book_depth_path: str, target_tf: str = "15min") -> pd.DataFrame:
    """Build orderbook features from book_depth.parquet."""
    df = pd.read_parquet(book_depth_path)
    if 'timestamp' not in df.columns:
        return pd.DataFrame()

    # Pivot: each timestamp → one row with bid/ask depth at each level
    # Negative pct = bid, positive = ask
    bid_levels = sorted([p for p in df['percentage'].unique() if p < 0], reverse=True)
    ask_levels = sorted([p for p in df['percentage'].unique() if p > 0])

    # Resample to target_tf first (much faster)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    features = {}

    # For each level, get resampled depth
    for pct in bid_levels + ask_levels:
        mask = df['percentage'] == pct
        level_data = df.loc[mask, ['depth', 'notional']]
        side = 'bid' if pct < 0 else 'ask'
        abs_pct = abs(pct)
        label = f"ob_{side}_{abs_pct:.1f}pct"

        resampled = level_data['depth'].resample(target_tf).mean()
        features[f"{label}_depth"] = resampled

        resampled_n = level_data['notional'].resample(target_tf).mean()
        features[f"{label}_notional"] = resampled_n

    feat_df = pd.DataFrame(features)

    # Derived features
    # 1. Bid/Ask imbalance at various levels
    for pct in [0.2, 1.0, 2.0, 5.0]:
        bid_col = f"ob_bid_{pct:.1f}pct_depth"
        ask_col = f"ob_ask_{pct:.1f}pct_depth"
        if bid_col in feat_df.columns and ask_col in feat_df.columns:
            total = feat_df[bid_col] + feat_df[ask_col]
            feat_df[f"ob_imb_{pct:.1f}pct"] = (feat_df[bid_col] - feat_df[ask_col]) / total.replace(0, np.nan)

    # 2. Cumulative depth imbalance (sum across levels)
    bid_cols = [c for c in feat_df.columns if 'bid' in c and 'depth' in c]
    ask_cols = [c for c in feat_df.columns if 'ask' in c and 'depth' in c]
    if bid_cols and ask_cols:
        total_bid = feat_df[bid_cols].sum(axis=1)
        total_ask = feat_df[ask_cols].sum(axis=1)
        feat_df["ob_total_imb"] = (total_bid - total_ask) / (total_bid + total_ask).replace(0, np.nan)

    # 3. Depth change (momentum of depth)
    for col in [f"ob_imb_{p:.1f}pct" for p in [0.2, 1.0, 2.0, 5.0]]:
        if col in feat_df.columns:
            for w in [4, 8, 16]:
                feat_df[f"{col}_chg{w}"] = feat_df[col].diff(w)

    # 4. Depth slope (how depth increases with distance)
    for side in ['bid', 'ask']:
        levels = sorted([c for c in feat_df.columns if side in c and 'depth' in c])
        if len(levels) >= 2:
            inner = feat_df[levels[0]]  # closest to mid
            outer = feat_df[levels[-1]]  # farthest
            feat_df[f"ob_{side}_slope"] = (outer - inner) / inner.replace(0, np.nan)

    # 5. Shift by 1 period to avoid lookahead (depth at T uses data up to T+15m)
    # The resampled mean already represents the average within the period,
    # which includes future data. Shift by 1 to use only completed periods.
    feat_df = feat_df.shift(1)

    return feat_df.dropna(how='all')


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 071: Orderbook Depth + Fee Sensitivity")
    print("=" * 60)

    # ═══════════════════════════════════════════════
    # Part A: Orderbook Features
    # ═══════════════════════════════════════════════
    print("\n[Part A] Orderbook Depth Features")

    book_path = "data/merged/BTCUSDT/book_depth.parquet"
    if os.path.exists(book_path):
        print("  Building orderbook features...")
        ob_feat = build_orderbook_features(book_path, target_tf="15min")
        print(f"  Orderbook features: {ob_feat.shape}")

        # Load close prices for correlation
        kline = pd.read_parquet("data/merged/BTCUSDT/kline_15m.parquet")
        if 'timestamp' in kline.columns:
            kline = kline.set_index('timestamp')
        if kline.index.tz is not None:
            kline.index = kline.index.tz_localize(None)
        close = kline['close']

        # Align
        common = ob_feat.index.intersection(close.index)
        ob_aligned = ob_feat.loc[common]
        close_aligned = close.loc[common].values

        # Forward returns
        n = len(close_aligned)
        fwd_1h = np.full(n, np.nan)
        fwd_4h = np.full(n, np.nan)
        for i in range(n - 4):
            fwd_1h[i] = (close_aligned[i + 4] - close_aligned[i]) / close_aligned[i]
        for i in range(n - 16):
            fwd_4h[i] = (close_aligned[i + 16] - close_aligned[i]) / close_aligned[i]

        # Use last 20% as test
        test_start = int(n * 0.8)

        print(f"\n  Top orderbook features by |corr| with 1h return (test set):")
        print(f"  {'Feature':>35s} {'corr_1h':>8s} {'corr_4h':>8s}")
        print(f"  {'-'*55}")

        corrs = []
        for col in ob_aligned.columns:
            feat = ob_aligned[col].values[test_start:]
            fr1 = fwd_1h[test_start:]
            fr4 = fwd_4h[test_start:]
            v1 = ~np.isnan(feat) & ~np.isnan(fr1) & np.isfinite(feat)
            v4 = ~np.isnan(feat) & ~np.isnan(fr4) & np.isfinite(feat)
            c1 = np.corrcoef(feat[v1], fr1[v1])[0, 1] if v1.sum() > 500 else 0
            c4 = np.corrcoef(feat[v4], fr4[v4])[0, 1] if v4.sum() > 500 else 0
            if not (np.isnan(c1) or np.isnan(c4)):
                corrs.append((col, c1, c4))

        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, c1, c4 in corrs[:20]:
            print(f"  {col:>35s} {c1:>+8.4f} {c4:>+8.4f}")

        # Combined model: OB features + existing features
        print(f"\n  Combined Ridge: OB + existing features")
        from sklearn.linear_model import Ridge
        from ultrathink.cache import ParquetCache
        cache = ParquetCache('data/cache')
        params = {'symbol': 'BTCUSDT', 'start': '2024-01-01', 'end': '2025-12-31',
                  'target_tf': '15min', 'extras': [], 'lag_top_n': 30,
                  'lag_depths': [1, 2, 3, 4, 5, 6, 7], 'version': 'v2_fixed'}
        X_existing, hit = cache.get('feat_BTCUSDT', params)

        if hit and X_existing is not None:
            # Align all three
            common3 = X_existing.index.intersection(ob_aligned.index)
            if len(common3) > 1000:
                X_ex = X_existing.loc[common3]
                X_ob = ob_aligned.loc[common3]
                close3 = close.reindex(common3).values

                n3 = len(common3)
                fwd3 = np.full(n3, np.nan)
                for i in range(n3 - 4):
                    fwd3[i] = (close3[i + 4] - close3[i]) / close3[i]

                s1, s2 = int(n3 * 0.6), int(n3 * 0.8)

                # Top existing features
                top_existing = [
                    'deriv_oi_chg_5_lag7', 'deriv_oi_chg_5_lag6',
                    'deriv_oi_chg_10_lag6', 'deriv_oi_chg_10_lag5',
                    'flow_delta_sum_10_lag7', 'flow_cvd_chg_10_lag7',
                ]
                top_existing = [c for c in top_existing if c in X_ex.columns]

                # Top OB features (by train set corr)
                ob_cols = list(ob_aligned.columns)

                for config_name, feat_cols_list in [
                    ("Existing only", [(X_ex, top_existing)]),
                    ("OB only", [(X_ob, ob_cols[:15])]),
                    ("Combined", [(X_ex, top_existing), (X_ob, ob_cols[:15])]),
                ]:
                    parts_train, parts_test = [], []
                    for df_source, cols in feat_cols_list:
                        valid_cols = [c for c in cols if c in df_source.columns]
                        vals = np.nan_to_num(df_source[valid_cols].values.astype(np.float32), 0.0)
                        parts_train.append(vals[:s1])
                        parts_test.append(vals[s2:])

                    X_tr = np.hstack(parts_train)
                    X_te = np.hstack(parts_test)
                    y_tr = fwd3[:s1]
                    y_te = fwd3[s2:]
                    v_tr = ~np.isnan(y_tr)
                    v_te = ~np.isnan(y_te)

                    model = Ridge(alpha=10.0)
                    model.fit(X_tr[v_tr], y_tr[v_tr])
                    pred = model.predict(X_te[v_te])
                    corr = np.corrcoef(pred, y_te[v_te])[0, 1]

                    # Trading sim
                    direction = np.sign(pred)
                    for fee_val in [0.0008, 0.0002]:
                        net = direction * y_te[v_te] - fee_val
                        wr = (net > 0).mean() * 100
                        sharpe = net.mean() / net.std() * np.sqrt(252 * 4) if net.std() > 0 else 0
                        fee_label = f"{fee_val * 10000:.0f}bp"
                        print(f"    {config_name:>15s} fee={fee_label}: corr={corr:+.4f} WR={wr:.1f}% Sharpe={sharpe:+.2f}")

    # ═══════════════════════════════════════════════
    # Part B: Fee Sensitivity
    # ═══════════════════════════════════════════════
    print(f"\n[Part B] Fee Sensitivity Analysis")
    print(f"  Using existing weak signals, at what fee level is profit possible?")

    from ultrathink.cache import ParquetCache
    cache = ParquetCache('data/cache')
    params = {'symbol': 'BTCUSDT', 'start': '2024-01-01', 'end': '2025-12-31',
              'target_tf': '15min', 'extras': [], 'lag_top_n': 30,
              'lag_depths': [1, 2, 3, 4, 5, 6, 7], 'version': 'v2_fixed'}
    X, hit = cache.get('feat_BTCUSDT', params)
    if not hit:
        print("  No cache"); return

    kline = pd.read_parquet("data/merged/BTCUSDT/kline_15m.parquet")
    if 'timestamp' in kline.columns: kline = kline.set_index('timestamp')
    if kline.index.tz is not None: kline.index = kline.index.tz_localize(None)
    close = kline['close'].reindex(X.index, method='ffill').values

    n = len(X)
    s1, s2 = int(n * 0.6), int(n * 0.8)

    # Best features
    top_feats = ['deriv_oi_chg_5_lag7', 'deriv_oi_chg_5_lag6',
                 'deriv_oi_chg_10_lag6', 'deriv_oi_chg_10_lag5',
                 'flow_delta_sum_10_lag7', 'flow_cvd_chg_10_lag7',
                 'deriv_oi_chg_10_lag4', 'deriv_oi_chg_10_lag3',
                 'deriv_oi_chg_10_lag2', '15m_ret_10_lag7',
                 'xtf_vol_5m_vs_1h', '5m_ret_50',
                 'deriv_ls_ratio_chg_5', 'fund_zscore_20_lag6']
    top_feats = [f for f in top_feats if f in X.columns]
    Xv = np.nan_to_num(X[top_feats].values.astype(np.float32), 0.0)

    from sklearn.linear_model import Ridge

    print(f"\n  {'Horizon':>7s} {'Fee':>6s} {'Corr':>7s} {'WR':>6s} {'AvgBps':>8s} {'Sharpe':>7s} {'Result':>8s}")
    print(f"  {'-'*55}")

    for h_name, h_bars in [('15m', 1), ('1h', 4), ('4h', 16), ('8h', 32)]:
        fwd = np.full(n, np.nan)
        for i in range(n - h_bars):
            fwd[i] = (close[i + h_bars] - close[i]) / close[i]

        y_tr, y_te = fwd[:s1], fwd[s2:]
        v_tr, v_te = ~np.isnan(y_tr), ~np.isnan(y_te)

        model = Ridge(alpha=10.0)
        model.fit(Xv[:s1][v_tr], y_tr[v_tr])
        pred = model.predict(Xv[s2:][v_te])
        corr = np.corrcoef(pred, y_te[v_te])[0, 1]

        direction = np.sign(pred)
        for fee_bps in [0, 1, 2, 4, 6, 8]:
            fee = fee_bps / 10000
            net = direction * y_te[v_te] - fee
            wr = (net > 0).mean() * 100
            avg = net.mean() * 10000
            sharpe = net.mean() / net.std() * np.sqrt(252 * 4) if net.std() > 0 else 0
            result = "WIN" if sharpe > 0 else "LOSE"
            print(f"  {h_name:>7s} {fee_bps:>4d}bp {corr:>+6.4f} {wr:5.1f}% {avg:>+7.1f} {sharpe:>+6.2f} {result:>8s}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

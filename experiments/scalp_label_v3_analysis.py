"""v3 optimal action label 분석 — ETH + BTC."""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalping.labeler_v3 import generate_scalp_labels_v3

FEE = 0.0008

for SYMBOL in ["ETHUSDT", "BTCUSDT"]:
    print(f"\n{'#'*80}")
    print(f"### {SYMBOL}")
    print(f"{'#'*80}")

    kline = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
    print(f"5m bars: {len(kline):,}")

    # ── v3 라벨 생성 ──
    print("\n[Generating v3 labels...]")
    labels = generate_scalp_labels_v3(kline, max_holds=[3, 4, 6, 8, 12], fee=FEE, min_rr=1.5)
    print(f"  Shape: {labels.shape}")

    # ── 1. Action 분포 ──
    print(f"\n{'='*70}")
    print("── Action Distribution ──\n")
    for hold in [3, 4, 6, 8, 12]:
        col = f"action_{hold}"
        act = labels[col].dropna()
        n = len(act)
        n_long = (act > 0).sum()
        n_short = (act < 0).sum()
        n_hold = (act == 0).sum()
        print(f"  hold={hold:2d} ({hold*5:3d}min) | LONG={n_long/n:.1%} SHORT={n_short/n:.1%} HOLD={n_hold/n:.1%} | "
              f"trade_rate={1-n_hold/n:.1%}")

    # ── 2. Edge 분석 ──
    print(f"\n{'='*70}")
    print("── Edge Distribution (when action != HOLD) ──\n")
    for hold in [3, 4, 6, 8, 12]:
        act = labels[f"action_{hold}"]
        el = labels[f"edge_long_{hold}"]
        es = labels[f"edge_short_{hold}"]

        # Long trades
        long_mask = act > 0
        if long_mask.sum() > 100:
            long_edge = el[long_mask].dropna()
            print(f"  hold={hold} LONG:  avg_edge={long_edge.mean()*100:+.4f}% "
                  f"median={long_edge.median()*100:+.4f}% n={len(long_edge):,}")

        # Short trades
        short_mask = act < 0
        if short_mask.sum() > 100:
            short_edge = es[short_mask].dropna()
            print(f"  hold={hold} SHORT: avg_edge={short_edge.mean()*100:+.4f}% "
                  f"median={short_edge.median()*100:+.4f}% n={len(short_edge):,}")

        # HOLD
        hold_mask = act == 0
        if hold_mask.sum() > 100:
            # What's the best possible edge in HOLD zones?
            best_edge = np.maximum(el[hold_mask].fillna(-1), es[hold_mask].fillna(-1))
            print(f"  hold={hold} HOLD:  avg_best_edge={best_edge.mean()*100:+.4f}% n={hold_mask.sum():,}")

    # ── 3. MFE/MAE 프로필 ──
    print(f"\n{'='*70}")
    print("── MFE/MAE Profile (hold=6, 30min) ──\n")
    hold = 6
    act = labels[f"action_{hold}"]
    for action_label, mask in [("LONG", act > 0), ("SHORT", act < 0), ("HOLD", act == 0)]:
        mfe_l = labels[f"mfe_long_{hold}"][mask].dropna()
        mae_l = labels[f"mae_long_{hold}"][mask].dropna()
        mfe_s = labels[f"mfe_short_{hold}"][mask].dropna()
        mae_s = labels[f"mae_short_{hold}"][mask].dropna()
        if len(mfe_l) < 100:
            continue
        print(f"  {action_label:5s} (n={mask.sum():,}):")
        print(f"    Long:  MFE={mfe_l.mean()*100:.4f}% MAE={mae_l.mean()*100:.4f}% ratio={mfe_l.mean()/(mae_l.mean()+1e-10):.2f}")
        print(f"    Short: MFE={mfe_s.mean()*100:.4f}% MAE={mae_s.mean()*100:.4f}% ratio={mfe_s.mean()/(mae_s.mean()+1e-10):.2f}")

    # ── 4. 연도별 안정성 ──
    print(f"\n{'='*70}")
    print("── Yearly Stability (hold=6) ──\n")
    hold = 6
    act = labels[f"action_{hold}"]
    el = labels[f"edge_long_{hold}"]
    es = labels[f"edge_short_{hold}"]

    for yr in range(2020, 2027):
        yr_mask = act.index.year == yr
        if yr_mask.sum() < 1000:
            continue
        sub_act = act[yr_mask].dropna()
        n = len(sub_act)
        trade_rate = (sub_act != 0).mean()
        long_rate = (sub_act > 0).mean()
        short_rate = (sub_act < 0).mean()

        # Avg edge for trades
        long_m = act[yr_mask] > 0
        short_m = act[yr_mask] < 0
        avg_el = el[yr_mask][long_m].mean() if long_m.sum() > 0 else 0
        avg_es = es[yr_mask][short_m].mean() if short_m.sum() > 0 else 0

        print(f"  {yr}: trade={trade_rate:.1%} (L={long_rate:.1%} S={short_rate:.1%}) "
              f"avg_edge_L={avg_el*100:+.4f}% avg_edge_S={avg_es*100:+.4f}%")

    # ── 5. Session별 ──
    print(f"\n{'='*70}")
    print("── Session Analysis (hold=6) ──\n")
    hour = kline.index.hour
    hold = 6
    act = labels[f"action_{hold}"]

    for sess, hrs in [("Asia 0-8", range(0,8)), ("Europe 8-14", range(8,14)),
                       ("US 14-22", range(14,22)), ("Late 22-24", range(22,24))]:
        mask = hour.isin(hrs)
        sub = act[mask].dropna()
        if len(sub) < 100:
            continue
        trade_rate = (sub != 0).mean()
        long_rate = (sub > 0).mean()
        print(f"  {sess:15s} trade={trade_rate:.1%} (L={long_rate:.1%} S={(sub<0).mean():.1%})")

    # ── 6. Vol regime별 ──
    print(f"\n{'='*70}")
    print("── Vol Regime Analysis (hold=6) ──\n")
    ret = kline["close"].pct_change()
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()
    vol_sq = vol_20 / vol_60

    for regime, lo_v, hi_v in [("Squeeze <0.7", 0, 0.7), ("Normal 0.7-1.3", 0.7, 1.3), ("Expansion >1.3", 1.3, 100)]:
        mask = vol_sq.between(lo_v, hi_v)
        sub = act[mask].dropna()
        el_sub = el[mask]
        es_sub = es[mask]
        if len(sub) < 100:
            continue
        trade_rate = (sub != 0).mean()
        long_edge_avg = el_sub[sub > 0].mean() if (sub > 0).sum() > 0 else 0
        short_edge_avg = es_sub[sub < 0].mean() if (sub < 0).sum() > 0 else 0
        print(f"  {regime:18s} trade={trade_rate:.1%} "
              f"avg_edge_L={long_edge_avg*100:+.4f}% avg_edge_S={short_edge_avg*100:+.4f}%")

    # ── 7. RR threshold 민감도 ──
    print(f"\n{'='*70}")
    print("── min_rr Sensitivity (hold=6) ──\n")
    c_arr = kline["close"].values.astype(np.float64)
    h_arr = kline["high"].values.astype(np.float64)
    l_arr = kline["low"].values.astype(np.float64)

    from scalping.labeler_v3 import _optimal_action
    for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        act_arr, _, _, _, _, el_arr, es_arr, _, _ = _optimal_action(c_arr, h_arr, l_arr, 6, FEE, rr)
        n = len(act_arr)
        trade_rate = (act_arr != 0).sum() / n
        long_rate = (act_arr > 0).sum() / n
        short_rate = (act_arr < 0).sum() / n
        # Avg edge
        long_edges = el_arr[act_arr > 0]
        short_edges = es_arr[act_arr < 0]
        avg_el = np.nanmean(long_edges) if len(long_edges) > 0 else 0
        avg_es = np.nanmean(short_edges) if len(short_edges) > 0 else 0
        print(f"  rr={rr:.1f}: trade={trade_rate:.1%} (L={long_rate:.1%} S={short_rate:.1%}) "
              f"avg_edge_L={avg_el*100:+.4f}% avg_edge_S={avg_es*100:+.4f}%")

print(f"\n{'='*70}")
print("=== DONE ===")

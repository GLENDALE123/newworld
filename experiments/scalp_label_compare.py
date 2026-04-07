"""v1 vs v2 라벨 비교 분석 — ETH 5m."""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalping.labeler import generate_scalp_labels, SCALP_PARAMS
from scalping.labeler_v2 import generate_scalp_labels_v2, SCALP_PARAMS_V2

SYMBOL = "ETHUSDT"
print(f"=== Label v1 vs v2 Comparison ({SYMBOL}) ===\n")

kline = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
print(f"5m bars: {len(kline):,} ({kline.index.min().date()} ~ {kline.index.max().date()})")

# ── v1 라벨 생성 ──
print("\n[v1 labels...]")
labels_v1 = generate_scalp_labels(kline)
print(f"  Shape: {labels_v1.shape}")

# ── v2 라벨 생성 ──
print("[v2 labels...]")
labels_v2 = generate_scalp_labels_v2(kline)
print(f"  Shape: {labels_v2.shape}")

FEE = 0.0008

# ── v1 EV 분석 ──
print(f"\n{'='*80}")
print("── v1 Label EV (raw, no fee adjustment in label) ──\n")
for tp, sl, hold in SCALP_PARAMS:
    for d in ["long", "short"]:
        s = f"{tp}_{sl}_{hold}_{d}"
        lab = labels_v1[f"label_{s}"].dropna()
        if len(lab) < 100:
            continue
        wr = (lab > 0).mean()
        mfe_m = labels_v1[f"mfe_{s}"].dropna().mean()
        mae_m = labels_v1[f"mae_{s}"].dropna().mean()
        ev = wr * mfe_m - (1 - wr) * mae_m
        ev_net = ev - FEE
        print(f"  {s:30s} WR={wr:.3f} EV={ev:+.6f} EV_net={ev_net:+.6f}")

# ── v2 EV 분석 ──
print(f"\n{'='*80}")
print("── v2 Label EV (fee-aware, dynamic barriers, 3-class) ──\n")

v2_results = []
for tp, sl, hold in SCALP_PARAMS_V2:
    for d in ["long", "short"]:
        s = f"{tp}_{sl}_{hold}_{d}"
        lab = labels_v2[f"label_{s}"].dropna()
        net = labels_v2[f"net_{s}"].dropna()
        if len(lab) < 100:
            continue

        n_total = len(lab)
        n_win = (lab > 0).sum()
        n_loss = (lab < 0).sum()
        n_chop = (lab == 0).sum()

        wr = n_win / n_total
        lr = n_loss / n_total
        chop_rate = n_chop / n_total

        avg_net = net.mean()
        avg_win_net = net[lab > 0].mean() if n_win > 0 else 0
        avg_loss_net = net[lab < 0].mean() if n_loss > 0 else 0

        mfe_m = labels_v2[f"mfe_{s}"].dropna().mean()
        mae_m = labels_v2[f"mae_{s}"].dropna().mean()

        v2_results.append({
            "suffix": s, "wr": wr, "lr": lr, "chop": chop_rate,
            "avg_net": avg_net, "avg_win": avg_win_net, "avg_loss": avg_loss_net,
            "mfe": mfe_m, "mae": mae_m, "n": n_total
        })

        print(f"  {s:30s} WR={wr:.3f} Loss={lr:.3f} Chop={chop_rate:.3f} | "
              f"avg_net={avg_net*100:+.4f}% | win={avg_win_net*100:+.4f}% loss={avg_loss_net*100:+.4f}%")

# ── v2 Best params ──
print(f"\n{'='*80}")
print("── v2 Best by avg_net ──\n")
v2_results.sort(key=lambda x: x["avg_net"], reverse=True)
for i, r in enumerate(v2_results[:10]):
    marker = " <<<" if r["avg_net"] > 0 else ""
    print(f"  {i+1:2d}. {r['suffix']:30s} avg_net={r['avg_net']*100:+.4f}% WR={r['wr']:.3f} chop={r['chop']:.3f}{marker}")

# ── 3-class 분포 분석 (2023+ 기간) ──
print(f"\n{'='*80}")
print("── 3-Class Distribution by Period ──\n")

kline_23 = kline.loc["2023-01-01":]
labels_v2_23 = labels_v2.loc[kline_23.index]

best_suffix = v2_results[0]["suffix"]
print(f"Best param: {best_suffix}\n")

lab_col = f"label_{best_suffix}"
net_col = f"net_{best_suffix}"

lab = labels_v2_23[lab_col].dropna()
net = labels_v2_23[net_col].dropna()

# 연도별
for yr in [2023, 2024, 2025, 2026]:
    mask = lab.index.year == yr
    if mask.sum() < 100:
        continue
    sub_lab = lab[mask]
    sub_net = net[mask]
    wr = (sub_lab > 0).mean()
    lr = (sub_lab < 0).mean()
    chop = (sub_lab == 0).mean()
    avg = sub_net.mean()
    print(f"  {yr}: WR={wr:.3f} Loss={lr:.3f} Chop={chop:.3f} avg_net={avg*100:+.4f}% n={len(sub_lab):,}")

# ── Vol regime별 분석 ──
print(f"\n{'='*80}")
print("── v2 Labels by Vol Regime ──\n")

ret = kline["close"].pct_change()
vol_20 = ret.rolling(20).std()
vol_60 = ret.rolling(60).std()
vol_sq = vol_20 / vol_60

lab_full = labels_v2[lab_col].copy()
net_full = labels_v2[net_col].copy()

for regime, lo, hi in [("Squeeze (<0.7)", 0, 0.7), ("Normal (0.7-1.3)", 0.7, 1.3), ("Expansion (>1.3)", 1.3, 100)]:
    mask = vol_sq.between(lo, hi) & lab_full.notna()
    sub_lab = lab_full[mask]
    sub_net = net_full[mask]
    if len(sub_lab) < 100:
        continue
    wr = (sub_lab > 0).mean()
    lr = (sub_lab < 0).mean()
    chop = (sub_lab == 0).mean()
    avg = sub_net.mean()
    print(f"  {regime:25s} WR={wr:.3f} Loss={lr:.3f} Chop={chop:.3f} avg_net={avg*100:+.4f}% n={len(sub_lab):,}")

# ── Session별 ──
print(f"\n{'='*80}")
print("── v2 Labels by Session ──\n")

hour = kline.index.hour
for sess, hours in [("Asia (0-8)", range(0,8)), ("Europe (8-14)", range(8,14)),
                     ("US (14-22)", range(14,22)), ("Late (22-24)", range(22,24))]:
    mask = hour.isin(hours) & lab_full.notna()
    sub_lab = lab_full[mask]
    sub_net = net_full[mask]
    if len(sub_lab) < 100:
        continue
    wr = (sub_lab > 0).mean()
    chop = (sub_lab == 0).mean()
    avg = sub_net.mean()
    print(f"  {sess:20s} WR={wr:.3f} Chop={chop:.3f} avg_net={avg*100:+.4f}% n={len(sub_lab):,}")

# ── v1 vs v2 직접 비교 (같은 방향, 비슷한 파라미터) ──
print(f"\n{'='*80}")
print("── v1 vs v2 Head-to-Head ──\n")

# v1: 2.0_1.0_3_long vs v2: 2.0_1.0_4_long (가장 비슷한 설정)
comparisons = [
    ("v1: 2.0_1.0_3", "v2: 2.0_1.0_4"),
    ("v1: 3.0_1.0_6", "v2: 3.0_1.0_6"),
]
for v1_tag, v2_tag in comparisons:
    for d in ["long", "short"]:
        v1_s = v1_tag.replace("v1: ", "") + f"_{d}"
        v2_s = v2_tag.replace("v2: ", "") + f"_{d}"

        v1_lab = labels_v1.get(f"label_{v1_s}")
        v2_lab = labels_v2.get(f"label_{v2_s}")
        v2_net = labels_v2.get(f"net_{v2_s}")

        if v1_lab is None or v2_lab is None:
            continue

        v1_l = v1_lab.dropna()
        v2_l = v2_lab.dropna()
        v2_n = v2_net.dropna()

        v1_wr = (v1_l > 0).mean()
        v1_mfe = labels_v1[f"mfe_{v1_s}"].dropna().mean()
        v1_mae = labels_v1[f"mae_{v1_s}"].dropna().mean()
        v1_ev = v1_wr * v1_mfe - (1 - v1_wr) * v1_mae - FEE

        v2_wr = (v2_l > 0).mean()
        v2_chop = (v2_l == 0).mean()
        v2_avg = v2_n.mean()

        print(f"  {d:5s} | v1 {v1_s:20s} WR={v1_wr:.3f} EV_net={v1_ev*100:+.4f}%")
        print(f"        | v2 {v2_s:20s} WR={v2_wr:.3f} Chop={v2_chop:.3f} avg_net={v2_avg*100:+.4f}%")
        print()

print("=== DONE ===")

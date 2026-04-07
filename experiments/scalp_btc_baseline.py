"""스캘핑 BTC 단일코인 baseline 실험.

목표: 5m BTC 데이터에서 스캘핑 알파가 존재하는지 검증
핵심 원칙: 확신 높은 자리에서만 진입, 안 하는 것도 실력

Walk-forward: 6개월 학습 → 1개월 검증 → 1개월 테스트, rolling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from datetime import timedelta

from scalping.labeler import generate_scalp_labels, SCALP_PARAMS
from scalping.features import build_scalp_features
from scalping.model import ScalpingMLP
from scalping.trainer import ScalpDataset, train_scalp_model, scalp_loss

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "merged")
SYMBOL = "BTCUSDT"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Walk-forward 설정
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 1


def load_data():
    """BTC 5m + tick_bar 로딩."""
    kline_5m = pd.read_parquet(os.path.join(DATA_DIR, SYMBOL, "kline_5m.parquet"))
    kline_5m = kline_5m.set_index("timestamp").sort_index()

    kline_15m = pd.read_parquet(os.path.join(DATA_DIR, SYMBOL, "kline_15m.parquet"))
    kline_15m = kline_15m.set_index("timestamp").sort_index()

    tick_bar = pd.read_parquet(os.path.join(DATA_DIR, SYMBOL, "tick_bar.parquet"))
    tick_bar = tick_bar.set_index("timestamp").sort_index()

    # book_ticker, funding은 optional
    bt_path = os.path.join(DATA_DIR, SYMBOL, "book_ticker.parquet")
    book_ticker = None
    if os.path.exists(bt_path):
        book_ticker = pd.read_parquet(bt_path).set_index("timestamp").sort_index()
        if "spread_bps" not in book_ticker.columns and "ask_price" in book_ticker.columns:
            mid = (book_ticker["ask_price"] + book_ticker["bid_price"]) / 2
            book_ticker["spread_bps"] = (book_ticker["ask_price"] - book_ticker["bid_price"]) / mid * 10000
        if "obi" not in book_ticker.columns and "ask_qty" in book_ticker.columns:
            book_ticker["obi"] = (book_ticker["bid_qty"] - book_ticker["ask_qty"]) / (book_ticker["bid_qty"] + book_ticker["ask_qty"])

    fr_path = os.path.join(DATA_DIR, SYMBOL, "funding_rate.parquet")
    funding = None
    if os.path.exists(fr_path):
        funding = pd.read_parquet(fr_path).set_index("timestamp").sort_index()

    return kline_5m, kline_15m, tick_bar, book_ticker, funding


def select_best_param(labels_df):
    """라벨 파라미터 중 가장 EV 높은 것 선택.

    EV = win_rate * avg_mfe - (1 - win_rate) * avg_mae
    """
    best_ev = -np.inf
    best_suffix = None

    for tp, sl, hold in SCALP_PARAMS:
        for dir_name in ["long", "short"]:
            suffix = f"{tp}_{sl}_{hold}_{dir_name}"
            lab = labels_df[f"label_{suffix}"].dropna()
            if len(lab) < 100:
                continue
            mfe = labels_df[f"mfe_{suffix}"].dropna()
            mae = labels_df[f"mae_{suffix}"].dropna()

            wr = (lab > 0).mean()
            ev = wr * mfe.mean() - (1 - wr) * mae.mean()

            if ev > best_ev:
                best_ev = ev
                best_suffix = suffix

    return best_suffix, best_ev


def walk_forward_splits(index, start_year=2022):
    """Walk-forward 윈도우 생성. 2022년부터 시작."""
    start = pd.Timestamp(f"{start_year}-01-01", tz="UTC")
    end = index.max()
    splits = []

    cursor = start
    while cursor + pd.DateOffset(months=TRAIN_MONTHS + VAL_MONTHS + TEST_MONTHS) <= end:
        train_start = cursor
        train_end = cursor + pd.DateOffset(months=TRAIN_MONTHS)
        val_end = train_end + pd.DateOffset(months=VAL_MONTHS)
        test_end = val_end + pd.DateOffset(months=TEST_MONTHS)

        splits.append({
            "train": (train_start, train_end),
            "val": (train_end, val_end),
            "test": (val_end, test_end),
        })
        cursor += pd.DateOffset(months=TEST_MONTHS)  # roll forward

    return splits


def evaluate_oos(model, features, labels_df, param_suffix, start, end, device,
                 thresholds=(0.55, 0.60, 0.65, 0.70, 0.75)):
    """OOS 평가: 다양한 확신도 임계값별 성과 측정.

    핵심: 높은 임계값 = 적은 거래 = 높은 정밀도 (이게 스캘핑의 핵심)
    """
    mask = (features.index >= start) & (features.index < end)
    X = features.loc[mask].values
    lab = labels_df.loc[mask]

    if len(X) == 0:
        return []

    model.eval()
    with torch.no_grad():
        inp = torch.tensor(np.nan_to_num(X, 0.0), dtype=torch.float32).to(device)
        out = model(inp)

    # 방향 결정: param_suffix에서 long/short 추출
    direction = "long" if param_suffix.endswith("long") else "short"
    probs = out[f"prob_{direction}"].cpu().numpy()
    mfe_pred = out[f"mfe_{direction}"].cpu().numpy()
    mae_pred = out[f"mae_{direction}"].cpu().numpy()

    actual_label = lab[f"label_{param_suffix}"].values
    actual_mfe = lab[f"mfe_{param_suffix}"].values
    actual_mae = lab[f"mae_{param_suffix}"].values

    results = []
    for thr in thresholds:
        trade_mask = (probs >= thr) & ~np.isnan(actual_label)
        n_trades = trade_mask.sum()
        if n_trades < 5:
            results.append({"threshold": thr, "n_trades": n_trades, "skip": True})
            continue

        wins = actual_label[trade_mask] > 0
        wr = wins.mean()
        avg_mfe = actual_mfe[trade_mask].mean()
        avg_mae = actual_mae[trade_mask].mean()
        ev = wr * avg_mfe - (1 - wr) * avg_mae
        # 수수료 차감 (왕복 0.08%)
        ev_net = ev - 0.0008
        selectivity = n_trades / trade_mask.shape[0]  # 전체 바 대비 거래 비율

        results.append({
            "threshold": thr,
            "n_trades": int(n_trades),
            "win_rate": round(wr, 4),
            "avg_mfe": round(avg_mfe, 6),
            "avg_mae": round(avg_mae, 6),
            "ev_gross": round(ev, 6),
            "ev_net": round(ev_net, 6),
            "selectivity": round(selectivity, 4),
            "skip": False,
        })

    return results


def run():
    print(f"=== 스캘핑 BTC Baseline ===")
    print(f"Device: {DEVICE}")

    # 1. 데이터 로딩
    print("\n[1] 데이터 로딩...")
    kline_5m, kline_15m, tick_bar, book_ticker, funding = load_data()
    print(f"  5m: {len(kline_5m):,} bars ({kline_5m.index.min()} ~ {kline_5m.index.max()})")

    # 2. 라벨 생성
    print("\n[2] 라벨 생성...")
    labels = generate_scalp_labels(kline_5m)
    print(f"  라벨 shape: {labels.shape}")

    # 3. 피처 생성
    print("\n[3] 피처 생성...")
    features = build_scalp_features(kline_5m, kline_15m, tick_bar, book_ticker, funding)
    # 인덱스 정렬
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    print(f"  피처: {features.shape[1]}개, 행: {len(features):,}")

    # NaN 비율 체크
    nan_pct = features.isna().mean().mean()
    print(f"  피처 NaN 비율: {nan_pct:.1%}")

    # 4. 베스트 라벨 파라미터 선택 (전체 데이터에서 — 이건 사전 분석용)
    print("\n[4] 라벨 파라미터 EV 분석...")
    best_suffix, best_ev = select_best_param(labels)
    print(f"  Best: {best_suffix} (EV={best_ev:.6f})")

    # 전체 파라미터 EV 출력
    for tp, sl, hold in SCALP_PARAMS:
        for d in ["long", "short"]:
            s = f"{tp}_{sl}_{hold}_{d}"
            lab = labels[f"label_{s}"].dropna()
            if len(lab) < 100:
                continue
            wr = (lab > 0).mean()
            mfe_m = labels[f"mfe_{s}"].dropna().mean()
            mae_m = labels[f"mae_{s}"].dropna().mean()
            ev = wr * mfe_m - (1 - wr) * mae_m
            tag = " <<<" if s == best_suffix else ""
            print(f"    {s:30s} WR={wr:.3f} EV={ev:+.6f}{tag}")

    # 5. Walk-forward 실험
    print(f"\n[5] Walk-forward 실험 (param={best_suffix})...")
    splits = walk_forward_splits(features.index)
    print(f"  {len(splits)} 윈도우")

    all_oos = []
    for i, sp in enumerate(splits):
        tr_s, tr_e = sp["train"]
        va_s, va_e = sp["val"]
        te_s, te_e = sp["test"]

        print(f"\n  --- Window {i+1}/{len(splits)} ---")
        print(f"  Train: {tr_s.date()} ~ {tr_e.date()}")
        print(f"  Val:   {va_s.date()} ~ {va_e.date()}")
        print(f"  Test:  {te_s.date()} ~ {te_e.date()}")

        # 데이터 슬라이싱
        tr_mask = (features.index >= tr_s) & (features.index < tr_e)
        va_mask = (features.index >= va_s) & (features.index < va_e)

        direction = "long" if best_suffix.endswith("long") else "short"
        other = "short" if direction == "long" else "long"

        def make_ds(mask):
            X = features.loc[mask].values
            L = labels.loc[mask]
            return ScalpDataset(
                features=X,
                label_long=L[f"label_{best_suffix}"].values if direction == "long" else L.get(f"label_{best_suffix.replace('long','short')}", pd.Series(np.nan, index=L.index)).values,
                label_short=L[f"label_{best_suffix}"].values if direction == "short" else L.get(f"label_{best_suffix.replace('short','long')}", pd.Series(np.nan, index=L.index)).values,
                mfe_long=L[f"mfe_{best_suffix}"].values if direction == "long" else np.full(len(L), np.nan),
                mfe_short=L[f"mfe_{best_suffix}"].values if direction == "short" else np.full(len(L), np.nan),
                mae_long=L[f"mae_{best_suffix}"].values if direction == "long" else np.full(len(L), np.nan),
                mae_short=L[f"mae_{best_suffix}"].values if direction == "short" else np.full(len(L), np.nan),
                bars_long=L[f"bars_{best_suffix}"].values if direction == "long" else np.full(len(L), np.nan),
                bars_short=L[f"bars_{best_suffix}"].values if direction == "short" else np.full(len(L), np.nan),
            )

        train_ds = make_ds(tr_mask)
        val_ds = make_ds(va_mask)

        print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")

        # 모델 학습
        n_feat = features.shape[1]
        model = ScalpingMLP(n_features=n_feat, hidden=64, dropout=0.2)
        print(f"  Parameters: {model.count_parameters():,}")

        result = train_scalp_model(
            model, train_ds, val_ds,
            epochs=50, batch_size=2048, lr=1e-3,
            device=DEVICE, patience=7, seed=42 + i,
        )
        print(f"  Best val loss: {result['best_val']:.4f} (epoch {result['epochs']})")

        # OOS 평가
        oos = evaluate_oos(model, features, labels, best_suffix, te_s, te_e, DEVICE)
        for r in oos:
            if r.get("skip"):
                continue
            r["window"] = i + 1
            r["test_period"] = f"{te_s.date()}~{te_e.date()}"
            all_oos.append(r)
            marker = "✓" if r["ev_net"] > 0 else "✗"
            print(f"    thr={r['threshold']:.2f} | trades={r['n_trades']:4d} | "
                  f"WR={r['win_rate']:.3f} | EV_net={r['ev_net']:+.6f} | "
                  f"select={r['selectivity']:.3f} {marker}")

    # 6. 종합 결과
    print("\n" + "=" * 70)
    print("=== 종합 결과 ===")
    if not all_oos:
        print("  OOS 결과 없음")
        return

    df_oos = pd.DataFrame(all_oos)
    for thr in df_oos["threshold"].unique():
        sub = df_oos[df_oos["threshold"] == thr]
        if sub["skip"].any() if "skip" in sub.columns else False:
            continue
        n = len(sub)
        avg_wr = sub["win_rate"].mean()
        avg_ev = sub["ev_net"].mean()
        total_trades = sub["n_trades"].sum()
        positive_windows = (sub["ev_net"] > 0).sum()
        avg_select = sub["selectivity"].mean()

        print(f"  thr={thr:.2f} | windows={n} | total_trades={total_trades:,} | "
              f"WR={avg_wr:.3f} | EV_net={avg_ev:+.6f} | "
              f"select={avg_select:.3f} | positive={positive_windows}/{n}")

    print("\n핵심 질문: 높은 임계값(0.70+)에서 EV_net > 0 이면 스캘핑 알파 존재")


if __name__ == "__main__":
    run()

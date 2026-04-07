"""단타 이벤트 탐색 — Phase 1: 1-8시간 수익적 순간 + 구조적 패턴.

스캘핑과 다른 점:
- 스캘핑 이벤트: 미시구조 노이즈 (buy_ratio, tc_ratio, depth)
- 단타 이벤트: 구조적 변화 (OI 급변, funding 극단, LS ratio 전환, 청산 캐스케이드)

Playbook Phase 1: "수익적이었을 순간들을 먼저 찾고, 그 순간들의 공통점을 분석"
"""

import numpy as np
import pandas as pd
import os

# 스캘핑은 중소형 알트에서만 됐지만, 단타는 메이저에서도 될 수 있음
# ETH를 기본으로 시작 (데이터 풍부, 메이저 대표)
SYMBOLS = ['ETHUSDT', 'BTCUSDT']
FEE = 0.0008  # taker round-trip

for SYMBOL in SYMBOLS:
    print(f"\n{'#'*70}")
    print(f"### {SYMBOL} — 단타 이벤트 탐색")
    print(f"{'#'*70}\n")

    k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
    metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
    funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()
    tick_5m = pd.read_parquet(f"data/cache/{SYMBOL}/tick_5m.parquet")

    cc = k5["close"].values; hh = k5["high"].values; ll = k5["low"].values
    vv = k5["volume"].values; n = len(cc)
    si = k5.index

    print(f"5m bars: {n:,} ({si.min().date()} ~ {si.max().date()})")

    # ── 1. Forward Return 분포 (1h, 2h, 4h, 8h) ──
    print(f"\n{'='*60}")
    print("── 1. Forward Return Distribution ──\n")

    for hold_5m, label in [(12, "1h"), (24, "2h"), (48, "4h"), (96, "8h")]:
        fwd = np.full(n, np.nan)
        for i in range(n - hold_5m):
            fwd[i] = (cc[i + hold_5m] - cc[i]) / cc[i]

        s = pd.Series(fwd).dropna()
        abs_s = s.abs()
        prof_long = (s > FEE).mean()
        prof_short = (s < -FEE).mean()

        print(f"  {label} ({hold_5m} bars):")
        print(f"    mean={s.mean()*100:+.4f}% std={s.std()*100:.3f}%")
        print(f"    P5={s.quantile(0.05)*100:+.3f}% P25={s.quantile(0.25)*100:+.3f}% "
              f"P75={s.quantile(0.75)*100:+.3f}% P95={s.quantile(0.95)*100:+.3f}%")
        print(f"    Fee비중: {FEE/(s.std()+1e-10)*100:.1f}% of std")
        print(f"    Long prof: {prof_long:.1%}, Short prof: {prof_short:.1%}")

    # ── 2. 구조적 피처 계산 ──
    print(f"\n{'='*60}")
    print("── 2. 구조적 피처 ──\n")

    # OI
    oi = metrics["sum_open_interest_value"].reindex(si, method="ffill") if "sum_open_interest_value" in metrics.columns else None
    if oi is not None:
        oi_chg_1h = oi.pct_change(12)
        oi_chg_4h = oi.pct_change(48)
        print(f"  OI 1h change: mean={oi_chg_1h.mean()*100:.4f}% std={oi_chg_1h.std()*100:.3f}%")
        print(f"  OI 4h change: mean={oi_chg_4h.mean()*100:.4f}% std={oi_chg_4h.std()*100:.3f}%")

    # Funding
    fr = funding["funding_rate"].reindex(si, method="ffill") if "funding_rate" in funding.columns else None
    if fr is not None:
        print(f"  Funding: mean={fr.mean()*100:.4f}% std={fr.std()*100:.4f}%")
        print(f"  Funding > 0.01%: {(fr > 0.0001).mean():.1%}")
        print(f"  Funding < -0.01%: {(fr < -0.0001).mean():.1%}")

    # Taker L/S ratio
    taker_ls = metrics["sum_taker_long_short_vol_ratio"].reindex(si, method="ffill") if "sum_taker_long_short_vol_ratio" in metrics.columns else None
    if taker_ls is not None:
        print(f"  Taker L/S: mean={taker_ls.mean():.3f} std={taker_ls.std():.3f}")

    # Top trader L/S
    top_ls = metrics["count_toptrader_long_short_ratio"].reindex(si, method="ffill") if "count_toptrader_long_short_ratio" in metrics.columns else None

    # Retail L/S
    retail_ls = metrics["count_long_short_ratio"].reindex(si, method="ffill") if "count_long_short_ratio" in metrics.columns else None

    # Price momentum
    ret = pd.Series(cc, index=si).pct_change()
    ret_1h = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(12) - 1
    ret_4h = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(48) - 1
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()

    # ── 3. Top/Bottom 5% 프로파일링 (4h forward) ──
    print(f"\n{'='*60}")
    print("── 3. 4h Forward Return Top/Bottom 5% 프로파일 ──\n")

    fwd_4h = np.full(n, np.nan)
    for i in range(n - 48):
        fwd_4h[i] = (cc[i + 48] - cc[i]) / cc[i]
    fwd_4h_s = pd.Series(fwd_4h, index=si)

    # MFE for 4h (max favorable excursion)
    mfe_4h = np.full(n, np.nan)
    for i in range(n - 48):
        max_h = max(hh[i+1:i+49])
        min_l = min(ll[i+1:i+49])
        mfe_4h[i] = max((max_h - cc[i]) / cc[i], (cc[i] - min_l) / cc[i])

    q95 = fwd_4h_s.quantile(0.95)
    q05 = fwd_4h_s.quantile(0.05)
    top5 = fwd_4h_s >= q95
    bot5 = fwd_4h_s <= q05
    mid = fwd_4h_s.between(fwd_4h_s.quantile(0.40), fwd_4h_s.quantile(0.60))

    print(f"  4h return P95: {q95*100:+.3f}%, P05: {q05*100:+.3f}%")
    print(f"  4h MFE mean: {pd.Series(mfe_4h).dropna().mean()*100:.3f}%")
    print(f"  4h MFE P95: {pd.Series(mfe_4h).dropna().quantile(0.95)*100:.3f}%")

    def profile(mask, label):
        print(f"\n  [{label}] n={mask.sum():,}")
        # OI
        if oi is not None:
            oi_1h_sub = oi_chg_1h[mask].dropna()
            oi_4h_sub = oi_chg_4h[mask].dropna()
            if len(oi_1h_sub) > 10:
                print(f"    OI chg 1h: mean={oi_1h_sub.mean()*100:+.4f}% median={oi_1h_sub.median()*100:+.4f}%")
                print(f"    OI chg 4h: mean={oi_4h_sub.mean()*100:+.4f}% median={oi_4h_sub.median()*100:+.4f}%")
        # Funding
        if fr is not None:
            fr_sub = fr[mask].dropna()
            if len(fr_sub) > 10:
                print(f"    Funding: mean={fr_sub.mean()*100:+.4f}% median={fr_sub.median()*100:+.4f}%")
        # Taker L/S
        if taker_ls is not None:
            tls = taker_ls[mask].dropna()
            if len(tls) > 10:
                print(f"    Taker L/S: mean={tls.mean():.3f} median={tls.median():.3f}")
        # Top trader L/S
        if top_ls is not None:
            ttls = top_ls[mask].dropna()
            if len(ttls) > 10:
                print(f"    TopTrader L/S: mean={ttls.mean():.3f}")
        # Retail L/S
        if retail_ls is not None:
            rls = retail_ls[mask].dropna()
            if len(rls) > 10:
                print(f"    Retail L/S: mean={rls.mean():.3f}")
        # Price momentum
        r1h = ret_1h[mask].dropna()
        r4h = ret_4h[mask].dropna()
        if len(r1h) > 10:
            print(f"    Ret 1h: mean={r1h.mean()*100:+.4f}%")
            print(f"    Ret 4h: mean={r4h.mean()*100:+.4f}%")
        # Volatility
        v20 = vol_20[mask].dropna()
        v60 = vol_60[mask].dropna()
        if len(v20) > 10:
            print(f"    Vol 20bar: mean={v20.mean()*100:.4f}%")
            squeeze = (v20 / (v60 + 1e-10))
            print(f"    Vol squeeze: mean={squeeze.mean():.3f}")
        # Session
        hours = si[mask].hour
        for sess, hrs in [("Asia 0-8", range(0,8)), ("EU 8-14", range(8,14)),
                           ("US 14-22", range(14,22))]:
            pct = hours.isin(hrs).mean()
            if pct > 0.01:
                print(f"    {sess}: {pct:.1%}", end="")
        print()

    profile(top5, "TOP 5% (큰 상승)")
    profile(bot5, "BOTTOM 5% (큰 하락)")
    profile(mid, "MIDDLE 40-60% (횡보)")

    # ── 4. 구조적 이벤트별 조건부 수익률 ──
    print(f"\n{'='*60}")
    print("── 4. 구조적 이벤트 → 4h Forward Return ──\n")

    valid_mask = fwd_4h_s.notna()

    # 4-1. OI 급변
    if oi is not None:
        print("  [OI Change → 4h Return]")
        for label, lo, hi in [("OI surge >3%", 0.03, 1.0), ("OI surge >5%", 0.05, 1.0),
                                ("OI stable ±1%", -0.01, 0.01),
                                ("OI drop <-3%", -1.0, -0.03), ("OI drop <-5%", -1.0, -0.05)]:
            mask = valid_mask & oi_chg_1h.between(lo, hi)
            sub = fwd_4h_s[mask].dropna()
            if len(sub) < 50: continue
            wr_long = (sub > FEE).mean()
            wr_short = (sub < -FEE).mean()
            print(f"    {label:20s}: n={len(sub):6,} mean={sub.mean()*100:+.4f}% "
                  f"L_prof={wr_long:.1%} S_prof={wr_short:.1%}")

    # 4-2. Funding extreme
    if fr is not None:
        print("\n  [Funding Rate → 4h Return]")
        for label, lo, hi, direction in [
            ("FR > +0.01%", 0.0001, 1.0, "short expected"),
            ("FR > +0.02%", 0.0002, 1.0, "strong short"),
            ("FR > +0.05%", 0.0005, 1.0, "extreme short"),
            ("FR < -0.01%", -1.0, -0.0001, "long expected"),
            ("FR < -0.02%", -1.0, -0.0002, "strong long"),
        ]:
            mask = valid_mask & fr.between(lo, hi)
            sub = fwd_4h_s[mask].dropna()
            if len(sub) < 50: continue
            # Direction-aligned return
            if "short" in direction:
                aligned = -sub  # short의 수익
            else:
                aligned = sub  # long의 수익
            wr = (aligned > FEE).mean()
            avg = aligned.mean()
            print(f"    {label:20s}: n={len(sub):6,} {direction:15s} WR={wr:.1%} avg={avg*100:+.4f}%")

    # 4-3. Taker L/S extreme
    if taker_ls is not None:
        print("\n  [Taker L/S Ratio → 4h Return]")
        for label, cond_fn, direction in [
            ("Taker heavy long >1.5", lambda: taker_ls > 1.5, "short (contrarian)"),
            ("Taker heavy long >2.0", lambda: taker_ls > 2.0, "short (extreme)"),
            ("Taker heavy short <0.7", lambda: taker_ls < 0.7, "long (contrarian)"),
            ("Taker heavy short <0.5", lambda: taker_ls < 0.5, "long (extreme)"),
            ("Balanced 0.8-1.2", lambda: taker_ls.between(0.8, 1.2), "neutral"),
        ]:
            mask = valid_mask & cond_fn()
            sub = fwd_4h_s[mask].dropna()
            if len(sub) < 50: continue
            if "short" in direction:
                aligned = -sub
            elif "long" in direction:
                aligned = sub
            else:
                aligned = sub.abs()  # neutral은 절대값
            wr = (aligned > FEE).mean()
            avg = aligned.mean()
            print(f"    {label:25s}: n={len(sub):6,} {direction:20s} WR={wr:.1%} avg={avg*100:+.4f}%")

    # 4-4. OI + Funding 복합
    if oi is not None and fr is not None:
        print("\n  [OI + Funding Combo → 4h Return]")
        combos = [
            ("OI drop + high FR → short", oi_chg_1h < -0.02, fr > 0.0001, -1),
            ("OI drop + neg FR → long", oi_chg_1h < -0.02, fr < -0.0001, 1),
            ("OI surge + high FR → ???", oi_chg_1h > 0.02, fr > 0.0001, 0),
            ("OI surge + neg FR → ???", oi_chg_1h > 0.02, fr < -0.0001, 0),
            ("OI drop + price drop → long bounce", (oi_chg_1h < -0.03) & (ret_1h < -0.005), True, 1),
            ("OI drop + price up → short reversal", (oi_chg_1h < -0.03) & (ret_1h > 0.005), True, -1),
        ]
        for label, cond1, cond2, d in combos:
            if isinstance(cond2, bool):
                mask = valid_mask & cond1
            else:
                mask = valid_mask & cond1 & cond2
            sub = fwd_4h_s[mask].dropna()
            if len(sub) < 30: continue
            if d == 1: aligned = sub
            elif d == -1: aligned = -sub
            else: aligned = sub.abs()
            wr = (aligned > FEE).mean()
            avg = aligned.mean()
            tag = "LONG" if d == 1 else "SHORT" if d == -1 else "ABS"
            print(f"    {label:40s}: n={len(sub):5,} {tag} WR={wr:.1%} avg={avg*100:+.4f}%")

    # 4-5. 시간대별
    print(f"\n  [Session → 4h Return]")
    hours = si.hour
    for sess, hrs in [("Asia 0-8", range(0,8)), ("EU 8-14", range(8,14)),
                       ("US 14-22", range(14,22)), ("Late 22-24", range(22,24))]:
        mask = valid_mask & hours.isin(hrs)
        sub = fwd_4h_s[mask].dropna()
        if len(sub) < 100: continue
        vol = sub.std()
        print(f"    {sess:15s}: n={len(sub):6,} mean={sub.mean()*100:+.4f}% std={vol*100:.3f}%")

print(f"\n{'='*60}")
print("=== DONE ===")

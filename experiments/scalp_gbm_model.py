"""LightGBM vs MLP 비교 — tabular data에서 tree model 우위 검증.

목표: 고확신 영역(prob≥0.65)의 거래 수를 늘리기.
현재 MLP: 483건 at prob≥0.65. GBM이 더 잘 분리하면 더 많은 고확신 거래 가능.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SYMBOL = "ETHUSDT"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TP_PCT = 0.10; SL_PCT = 0.05; MAX_HOLD = 6; FEE_M = 0.02; FEE_T = 0.08

print(f"=== LightGBM vs MLP ({SYMBOL}) ===\n")

# ── Data pipeline (cached) ──
k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
depth_cache = pd.read_parquet(f"data/cache/{SYMBOL}/depth_5m.parquet")
tick_5m = pd.read_parquet(f"data/cache/{SYMBOL}/tick_5m.parquet")
micro = pd.read_parquet(f"data/cache/{SYMBOL}/micro_5m.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()

cc = k5["close"].values; hh = k5["high"].values; ll = k5["low"].values; vv = k5["volume"].values
n = len(cc)

# Direction from cached depth
depth_imb = depth_cache["depth_imb_10"].reindex(k5.index) if "depth_imb_10" in depth_cache.columns else pd.Series(np.nan, index=k5.index)

ret3 = (pd.Series(cc, index=k5.index) / pd.Series(cc, index=k5.index).shift(3) - 1).values
depth_dir = np.where(depth_imb > 0, 1, np.where(depth_imb < 0, -1, 0))
mr_dir = -np.sign(ret3)
agree = (depth_dir == mr_dir) & (depth_dir != 0)
direction = np.where(agree, depth_dir, 0).astype(float)

# Features
ret = pd.Series(cc, index=k5.index).pct_change()
feat = pd.DataFrame(index=k5.index)
feat["ret_1"] = ret
feat["ret_3"] = ret3
feat["ret_6"] = pd.Series(cc, index=k5.index) / pd.Series(cc, index=k5.index).shift(6) - 1
feat["body"] = (k5["close"]-k5["open"])/(k5["high"]-k5["low"]+1e-10)
feat["upper_wick"] = (k5["high"]-np.maximum(k5["open"],k5["close"]))/(k5["high"]-k5["low"]+1e-10)
feat["lower_wick"] = (np.minimum(k5["open"],k5["close"])-k5["low"])/(k5["high"]-k5["low"]+1e-10)
feat["range_pct"] = (k5["high"]-k5["low"])/k5["close"]
feat["vol_5"] = ret.rolling(5).std()
feat["vol_12"] = ret.rolling(12).std()
feat["vol_20"] = ret.rolling(20).std()
feat["vol_60"] = ret.rolling(60).std()
feat["vol_squeeze"] = feat["vol_20"]/(feat["vol_60"]+1e-10)
feat["vol_accel"] = feat["vol_5"]/(feat["vol_20"]+1e-10)
feat["atr_12"] = (k5["high"]-k5["low"]).rolling(12).mean()
feat["atr_ratio"] = (k5["high"]-k5["low"])/(feat["atr_12"]+1e-10)
feat["vol_ratio"] = pd.Series(vv,index=k5.index)/(pd.Series(vv,index=k5.index).rolling(12).mean()+1e-10)

feat["tc_ratio"] = (tick_5m["trade_count"]/(tick_5m["trade_count"].rolling(12).mean()+1e-10)).reindex(k5.index)
feat["buy_ratio"] = (tick_5m["buy_volume"]/(tick_5m["buy_volume"]+tick_5m["sell_volume"]+1e-10)).reindex(k5.index)
feat["cvd_norm"] = (tick_5m["cvd_raw"]/(tick_5m["buy_volume"]+tick_5m["sell_volume"]+1e-10)).reindex(k5.index)
feat["cvd_slope_5"] = tick_5m["cvd_raw"].cumsum().diff(5).reindex(k5.index)

feat["depth_imb"] = depth_imb
feat["depth_imb_abs"] = depth_imb.abs()
feat["range_pos_20"] = (k5["close"]-k5["low"].rolling(20).min())/(k5["high"].rolling(20).max()-k5["low"].rolling(20).min()+1e-10)
vwap = (k5["close"]*k5["volume"]).rolling(20).sum()/(k5["volume"].rolling(20).sum()+1e-10)
feat["vwap_dist"] = k5["close"]/vwap - 1

# 1m micro (cached)
feat["intra_range"] = micro["intra_range"].reindex(k5.index)
feat["bull_1m"] = micro["bull_1m"].reindex(k5.index)

# Metrics
for col in ["sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        feat[col] = metrics[col].reindex(k5.index, method="ffill")
if "sum_open_interest_value" in metrics.columns:
    oi = metrics["sum_open_interest_value"].reindex(k5.index, method="ffill")
    feat["oi_change_5"] = oi.pct_change(5)

feat["hour"] = k5.index.hour
feat["is_us"] = ((feat["hour"]>=14)&(feat["hour"]<22)).astype(float)
feat["mr_strength"] = np.abs(ret3)
feat["depth_strength"] = depth_imb.abs()

feat = feat.replace([np.inf, -np.inf], np.nan)
feature_names = feat.columns.tolist()
print(f"Features: {len(feature_names)}")

# Vol filter + entry mask
va_q80 = feat["vol_accel"].quantile(0.80)
tc_q80 = feat["tc_ratio"].quantile(0.80)
entry_mask = (direction != 0) & (feat["vol_accel"] > va_q80).values & (feat["tc_ratio"] > tc_q80).values

# TP/SL labels
tp_hit = np.full(n, np.nan)
for i in range(n-MAX_HOLD):
    d = direction[i]
    if d==0 or np.isnan(d): continue
    entry=cc[i]; tp=entry*(1+d*TP_PCT/100); sl=entry*(1-d*SL_PCT/100)
    hit=0
    for j in range(i+1,i+MAX_HOLD+1):
        if d==1:
            if hh[j]>=tp: hit=1; break
            if ll[j]<=sl: hit=-1; break
        else:
            if ll[j]<=tp: hit=1; break
            if hh[j]>=sl: hit=-1; break
    tp_hit[i] = 1.0 if hit==1 else 0.0

feat_vals = feat.values
start_idx = k5.index.get_loc(k5.loc["2023-01-01":].index[0])

# ── Walk-forward ──
splits = []
cursor = pd.Timestamp("2023-01-01", tz=k5.index.tz)
end = k5.index.max()
while cursor + pd.DateOffset(months=8) <= end:
    tr_e = cursor + pd.DateOffset(months=6)
    va_e = tr_e + pd.DateOffset(months=1)
    te_e = va_e + pd.DateOffset(months=1)
    splits.append({"train":(cursor,tr_e),"val":(tr_e,va_e),"test":(va_e,te_e)})
    cursor += pd.DateOffset(months=1)

def get_pos(s, e):
    m = (k5.index>=s)&(k5.index<e)&entry_mask
    return np.array([p for p in np.where(m)[0] if not np.isnan(tp_hit[p])])

# Collect OOS predictions
oos_gbm = np.full(n, np.nan)
oos_mlp = np.full(n, np.nan)

print(f"\n[Walk-forward: {len(splits)} windows]\n")

for i, sp in enumerate(splits):
    tr_s,tr_e = sp["train"]; va_s,va_e = sp["val"]; te_s,te_e = sp["test"]
    tr_p = get_pos(tr_s,tr_e); va_p = get_pos(va_s,va_e); te_p = get_pos(te_s,te_e)
    if len(tr_p)<200 or len(va_p)<50 or len(te_p)==0: continue

    X_tr = np.nan_to_num(feat_vals[tr_p], 0); y_tr = tp_hit[tr_p]
    X_va = np.nan_to_num(feat_vals[va_p], 0); y_va = tp_hit[va_p]
    X_te = np.nan_to_num(feat_vals[te_p], 0)

    # --- LightGBM ---
    dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, y_va, feature_name=feature_names, reference=dtrain)

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
        "is_unbalance": True,
    }
    gbm = lgb.train(params, dtrain, num_boost_round=300,
                     valid_sets=[dval], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    gbm_probs = gbm.predict(X_te)
    for j, p in enumerate(te_p):
        oos_gbm[p] = gbm_probs[j]

    # --- MLP ---
    class DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    class MLP(nn.Module):
        def __init__(self, nin, h=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(nin,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.2),
                nn.Linear(h,h//2),nn.GELU(),nn.LayerNorm(h//2),nn.Dropout(0.2),
                nn.Linear(h//2,1))
        def forward(self,x): return self.net(x).squeeze(-1)

    pr = y_tr.mean()
    pw = torch.tensor([(1-pr)/(pr+1e-10)],dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    model = MLP(len(feature_names),128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,40)
    tl = DataLoader(DS(X_tr,y_tr),batch_size=512,shuffle=True,num_workers=0)
    vl = DataLoader(DS(X_va,y_va),batch_size=1024,shuffle=False,num_workers=0)

    bv=float("inf");w=0;bs=None
    for ep in range(40):
        model.train()
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE);l=crit(model(x),y);opt.zero_grad();l.backward();opt.step()
        sch.step()
        model.eval();v2=0;nb=0
        with torch.no_grad():
            for x,y in vl: x,y=x.to(DEVICE),y.to(DEVICE);v2+=crit(model(x),y).item();nb+=1
        v2/=max(nb,1)
        if v2<bv: bv=v2;w=0;bs={k:vv.cpu().clone() for k,vv in model.state_dict().items()}
        else:
            w+=1
            if w>=7: break
    model.load_state_dict(bs); model.eval()
    with torch.no_grad():
        inp = torch.tensor(X_te,dtype=torch.float32).to(DEVICE)
        mlp_probs = torch.sigmoid(model(inp)).cpu().numpy()
    for j, p in enumerate(te_p):
        oos_mlp[p] = mlp_probs[j]

    print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} n={len(te_p)} gbm_best={gbm.best_iteration}")

# ── Compare ──
print(f"\n{'='*80}")
print("=== GBM vs MLP: Trading Simulation ===\n")

def sim(probs_arr, name):
    print(f"  --- {name} ---")
    for prob_thr in [0.0, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for tp_pct, sl_pct, fee_pct, tag in [(0.10,0.05,0.02,"M"), (0.20,0.05,0.02,"M20"),
                                               (0.15,0.05,0.08,"T15"), (0.20,0.05,0.08,"T20"),
                                               (0.25,0.05,0.08,"T25")]:
            equity = 10000; trades = []
            in_pos = False; pos_bar=0; pos_dir=0; pos_entry=0; pos_tp=0; pos_sl=0

            for idx in range(start_idx, n):
                if in_pos:
                    bars = idx - pos_bar; closed = False
                    if pos_dir==1:
                        if hh[idx]>=pos_tp: pnl=tp_pct/100-fee_pct/100; closed=True
                        elif ll[idx]<=pos_sl: pnl=-sl_pct/100-fee_pct/100; closed=True
                        elif bars>=MAX_HOLD: pnl=(cc[idx]-pos_entry)/pos_entry-fee_pct/100; closed=True
                    else:
                        if ll[idx]<=pos_tp: pnl=tp_pct/100-fee_pct/100; closed=True
                        elif hh[idx]>=pos_sl: pnl=-sl_pct/100-fee_pct/100; closed=True
                        elif bars>=MAX_HOLD: pnl=(pos_entry-cc[idx])/pos_entry-fee_pct/100; closed=True
                    if closed: equity+=1000*pnl; trades.append(pnl); in_pos=False

                if not in_pos and entry_mask[idx] and direction[idx]!=0:
                    prob = probs_arr[idx]
                    if np.isnan(prob) or prob < prob_thr: continue
                    pos_bar=idx; pos_dir=int(direction[idx]); pos_entry=cc[idx]
                    pos_tp=pos_entry*(1+pos_dir*tp_pct/100)
                    pos_sl=pos_entry*(1-pos_dir*sl_pct/100)
                    in_pos=True

            if not trades: continue
            trades = np.array(trades)
            wr = (trades>0).mean(); avg = trades.mean()
            ret_pct = (equity-10000)/10000*100

            if avg > 0 or (prob_thr == 0.0 and tag == "M"):
                marker = " <<<" if avg > 0.0005 else " **" if avg > 0 else ""
                print(f"    p≥{prob_thr:.2f} {tag:4s}: n={len(trades):5,} WR={wr:.1%} "
                      f"avg={avg*100:+.4f}% ret={ret_pct:+.1f}%{marker}")

sim(oos_gbm, "LightGBM")
sim(oos_mlp, "MLP")

# Feature importance from last GBM
print(f"\n{'='*80}")
print("── GBM Feature Importance (last window) ──\n")
imp = pd.Series(gbm.feature_importance(importance_type="gain"), index=feature_names)
imp = imp.sort_values(ascending=False)
for j, (feat_name, val) in enumerate(imp.head(15).items()):
    print(f"  {j+1:2d}. {feat_name:<30s} {val:.0f}")

print(f"\n{'='*80}")
print("=== DONE ===")

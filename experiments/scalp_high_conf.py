"""고확신 영역 집중 분석: prob≥0.60에서 WR 75% 달성.
더 넓은 TP로 taker 가능성 탐색 + Kelly sizing.

질문: 모델이 확신하는 순간에 TP를 넓히면?
- WR 75%라면 TP 0.20% / SL 0.10%로도 양수 가능?
- 또는 TP 0.30% / SL 0.10%?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SYMBOL = "ETHUSDT"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_HOLD_OPTIONS = [6, 12, 18]  # test longer holds too

print(f"=== High Confidence Deep Dive ({SYMBOL}) ===\n")

# ── Data (same as before) ──
k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()

if tick.index.tz is None and k5.index.tz is not None:
    tick.index = tick.index.tz_localize(k5.index.tz)

cc = k5["close"].values; hh = k5["high"].values; ll = k5["low"].values; vv = k5["volume"].values
n = len(cc)

# Direction
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and k5.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(k5.index.tz)
bd_key = bd[bd["percentage"].isin([-1.0, 1.0])].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_piv = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
depth_imb = pd.Series(np.nan, index=k5.index)
if -1.0 in bd_piv.columns and 1.0 in bd_piv.columns:
    depth_imb = ((bd_piv[-1.0] - bd_piv[1.0]) / (bd_piv[-1.0] + bd_piv[1.0] + 1e-10)).reindex(k5.index)

ret3 = (pd.Series(cc, index=k5.index) / pd.Series(cc, index=k5.index).shift(3) - 1).values
depth_dir = np.where(depth_imb > 0, 1, np.where(depth_imb < 0, -1, 0))
mr_dir = -np.sign(ret3)
agree = (depth_dir == mr_dir) & (depth_dir != 0)
direction = np.where(agree, depth_dir, 0).astype(float)

# Features
ret = pd.Series(cc, index=k5.index).pct_change()
sf = pd.DataFrame(index=k5.index)
sf["ret"] = ret
sf["body"] = (k5["close"]-k5["open"])/(k5["high"]-k5["low"]+1e-10)
sf["range_pct"] = (k5["high"]-k5["low"])/k5["close"]
sf["vol_ratio"] = pd.Series(vv,index=k5.index)/pd.Series(vv,index=k5.index).rolling(20).mean()
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({"buy_volume":"sum","sell_volume":"sum","cvd_raw":"sum","trade_count":"sum"})
sf["buy_ratio"] = (tick_5m["buy_volume"]/(tick_5m["buy_volume"]+tick_5m["sell_volume"])).reindex(k5.index)
sf["tc_ratio"] = (tick_5m["trade_count"]/tick_5m["trade_count"].rolling(20).mean()).reindex(k5.index)
sf["cvd_norm"] = (tick_5m["cvd_raw"]/(tick_5m["buy_volume"]+tick_5m["sell_volume"]+1e-10)).reindex(k5.index)
sf["depth_imb"] = depth_imb
sf["vol_accel"] = ret.rolling(5).std()/(ret.rolling(20).std()+1e-10)
sf["upper_wick"] = (k5["high"]-np.maximum(k5["open"],k5["close"]))/(k5["high"]-k5["low"]+1e-10)
sf["lower_wick"] = (np.minimum(k5["open"],k5["close"])-k5["low"])/(k5["high"]-k5["low"]+1e-10)
sf["intra_range"] = 0  # placeholder
sf = sf.replace([np.inf,-np.inf], np.nan).fillna(0)

stat = pd.DataFrame(index=k5.index)
stat["vol_squeeze"] = ret.rolling(20).std()/(ret.rolling(60).std()+1e-10)
stat["range_pos"] = (k5["close"]-k5["low"].rolling(20).min())/(k5["high"].rolling(20).max()-k5["low"].rolling(20).min()+1e-10)
stat["vwap_dist"] = k5["close"]/((k5["close"]*k5["volume"]).rolling(20).sum()/(k5["volume"].rolling(20).sum()+1e-10))-1
stat["hour_sin"] = np.sin(2*np.pi*k5.index.hour/24)
stat["hour_cos"] = np.cos(2*np.pi*k5.index.hour/24)
if "sum_taker_long_short_vol_ratio" in metrics.columns:
    stat["taker_ls"] = metrics["sum_taker_long_short_vol_ratio"].reindex(k5.index, method="ffill")
stat = stat.replace([np.inf,-np.inf], np.nan).fillna(0)

# Vol filter
va_q80 = sf["vol_accel"].quantile(0.80); tc_q80 = sf["tc_ratio"].quantile(0.80)
entry_mask = (direction != 0) & (sf["vol_accel"] > va_q80).values & (sf["tc_ratio"] > tc_q80).values

# TP labels for training (use TP=0.10 SL=0.05 hold=6 as before)
tp_hit_train = np.full(n, np.nan)
for i in range(n-6):
    d = direction[i]
    if d == 0 or np.isnan(d): continue
    entry = cc[i]; tp = entry*(1+d*0.10/100); sl = entry*(1-d*0.05/100)
    hit = 0
    for j in range(i+1, i+7):
        if d==1:
            if hh[j]>=tp: hit=1; break
            if ll[j]<=sl: hit=-1; break
        else:
            if ll[j]<=tp: hit=1; break
            if hh[j]>=sl: hit=-1; break
    tp_hit_train[i] = 1.0 if hit==1 else 0.0

sf_vals = sf.values.astype(np.float32); stat_vals = stat.values.astype(np.float32)
n_ch = sf_vals.shape[1]; n_st = stat_vals.shape[1]; SEQ = 20

# Model + Dataset (reuse from seq_model)
class DS(Dataset):
    def __init__(self, indices):
        self.idx = indices
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        p = self.idx[i]
        x = np.concatenate([sf_vals[p], stat_vals[p]])
        return torch.tensor(np.nan_to_num(x,0), dtype=torch.float32), \
               torch.tensor(tp_hit_train[p] if not np.isnan(tp_hit_train[p]) else 0.0, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, nin, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.2),
            nn.Linear(h,h//2),nn.GELU(),nn.LayerNorm(h//2),nn.Dropout(0.2),
            nn.Linear(h//2,1))
    def forward(self,x): return self.net(x).squeeze(-1)

# WF training → get OOS probs
print("[Training MLP walk-forward...]")
splits = []
cursor = pd.Timestamp("2023-01-01", tz=k5.index.tz)
end = k5.index.max()
while cursor + pd.DateOffset(months=8) <= end:
    tr_e = cursor + pd.DateOffset(months=6)
    va_e = tr_e + pd.DateOffset(months=1)
    te_e = va_e + pd.DateOffset(months=1)
    splits.append({"train": (cursor,tr_e),"val":(tr_e,va_e),"test":(va_e,te_e)})
    cursor += pd.DateOffset(months=1)

oos_prob = np.full(n, np.nan)
for i, sp in enumerate(splits):
    tr_s,tr_e = sp["train"]; va_s,va_e = sp["val"]; te_s,te_e = sp["test"]
    def gp(s,e):
        m = (k5.index>=s)&(k5.index<e)&entry_mask
        return np.array([p for p in np.where(m)[0] if not np.isnan(tp_hit_train[p])])
    tr_p=gp(tr_s,tr_e); va_p=gp(va_s,va_e); te_p=gp(te_s,te_e)
    if len(tr_p)<200 or len(va_p)<50 or len(te_p)==0: continue
    pr = np.mean([tp_hit_train[p] for p in tr_p])
    pw = torch.tensor([(1-pr)/(pr+1e-10)],dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    model = MLP(n_ch+n_st,128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,40)
    tl=DataLoader(DS(tr_p),batch_size=512,shuffle=True,num_workers=0)
    vl=DataLoader(DS(va_p),batch_size=1024,shuffle=False,num_workers=0)
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
        for p in te_p:
            x=torch.tensor(np.nan_to_num(np.concatenate([sf_vals[p],stat_vals[p]]),0),dtype=torch.float32).unsqueeze(0).to(DEVICE)
            oos_prob[p]=torch.sigmoid(model(x)).cpu().item()
    print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} n={len(te_p)}")

# ── Test various TP/SL on high-confidence trades ──
print(f"\n{'='*80}")
print("=== HIGH CONFIDENCE (prob≥0.60) × VARIOUS TP/SL ===\n")

start_idx = k5.index.get_loc(k5.loc["2023-01-01":].index[0])

for prob_thr in [0.55, 0.60, 0.65]:
    print(f"\n  --- prob≥{prob_thr} ---")
    # Get high-conf entry positions
    hc_positions = [i for i in range(start_idx, n) if entry_mask[i] and direction[i]!=0
                    and not np.isnan(oos_prob[i]) and oos_prob[i] >= prob_thr]
    print(f"  Entries: {len(hc_positions):,}")

    for tp_pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for sl_pct in [0.05, 0.08, 0.10]:
            for max_hold in [6, 12]:
                for fee_pct in [0.02, 0.08]:
                    pnls = []
                    for pos in hc_positions:
                        d = direction[pos]; entry = cc[pos]
                        tp = entry*(1+d*tp_pct/100); sl = entry*(1-d*sl_pct/100)
                        hit = 0
                        for j in range(pos+1, min(pos+max_hold+1, n)):
                            if d==1:
                                if hh[j]>=tp: hit=1; break
                                if ll[j]<=sl: hit=-1; break
                            else:
                                if ll[j]<=tp: hit=1; break
                                if hh[j]>=sl: hit=-1; break
                        if hit==1: pnl=tp_pct/100-fee_pct/100
                        elif hit==-1: pnl=-sl_pct/100-fee_pct/100
                        else:
                            if d==1: pnl=(cc[min(pos+max_hold,n-1)]-entry)/entry-fee_pct/100
                            else: pnl=(entry-cc[min(pos+max_hold,n-1)])/entry-fee_pct/100
                        pnls.append(pnl)

                    pnls = np.array(pnls)
                    if len(pnls) < 30: continue
                    wr = (pnls>0).mean(); avg = pnls.mean()
                    tag = "M" if fee_pct==0.02 else "T"
                    if avg > 0 or (fee_pct==0.08 and avg > -0.0005):
                        marker = " <<<" if avg > 0.0005 else " **" if avg > 0 else ""
                        print(f"    TP={tp_pct:.2f} SL={sl_pct:.2f} H={max_hold:2d} {tag} "
                              f"WR={wr:.1%} avg={avg*100:+.4f}% n={len(pnls):,}{marker}")

print(f"\n{'='*80}")
print("=== DONE ===")

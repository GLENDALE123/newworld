# Scalping Research Playbook

> 이 문서는 ultraTM 프로젝트에서 스캘핑 알파를 발견하고 검증한 전체 과정을 기록한 것이다.
> 새로운 타임프레임(단타, 데이트레이딩, 스윙)의 모델을 만들 때도 이 순서를 따를 것을 권장한다.

---

## Phase 0: 전제 조건

연구를 시작하기 전에 명확히 해야 할 것:

- **타겟 타임프레임**: 홀딩 시간 범위 (스캘핑: 5-30분)
- **수수료 구조**: Taker 0.08%, Maker 0.02% (이것이 모든 것을 지배함)
- **사용 가능한 데이터**: kline, tick_bar, book_depth, metrics, funding 등
- **"충분한 수익"의 정의**: fee 차감 후 양수

---

## Phase 1: 이벤트 탐색 (Data Exploration)

> "수익적이었을 순간들을 먼저 찾고, 그 순간들의 공통점을 분석한다"

### 1-1. Forward Return 분포 분석

모델을 만들기 **전에**, 원시 데이터에서 기회가 존재하는지 확인:

- Forward N-bar 수익률의 분포 (P1, P5, P25, P50, P75, P95, P99)
- Fee보다 큰 움직임이 발생하는 비율 (전체 바의 몇 %?)
- 최대 수익 기회 (MFE)의 분포

**우리의 발견**: ETH 5m에서 15분 후 수익 > 0.08%인 바가 전체의 ~32%. 기회는 존재했다.

### 1-2. Top/Bottom 5% 프로파일링

"큰 움직임이 있었던 순간"과 "아무 일 없었던 순간"의 특성 비교:

| 분석 항목 | Top 5% (기회) | Middle (노이즈) | No-Trade Zone |
|---|---|---|---|
| 변동성 상태 | squeeze vs expansion | | |
| 거래량 패턴 | surge? decline? | | |
| CVD / buy_ratio | | | |
| 시간대 (Session) | | | |
| 캔들 패턴 | | | |
| Depth imbalance | | | |
| OI 변화 | | | |

**우리의 발견**:
- 알파 집중 조건: **US 세션 + 변동성 확대 + 거래량 폭증**
- 금지 구간: **Asia 새벽 + 저변동 + 낮은 거래량** (전체의 ~9.5%)
- trade_count가 압도적 z-score 1위 (z=+2.24)

### 1-3. 사용 가능한 데이터 전수 확인

```
✅ kline_5m (OHLCV + taker_buy_volume)
✅ tick_bar (buy_volume, sell_volume → CVD)
✅ book_depth (호가 깊이, 2023-01~)
✅ metrics (OI, taker L/S ratio, toptrader L/S)
✅ funding_rate
✅ kline_1m (미시구조)
✅ kline_15m, kline_1h (상위 TF)
✅ volume_bar (거래량 기반 바)
❌ mark_price_5m (파싱 깨짐)
❌ premium_index_5m (파싱 깨짐)
```

**교훈**: 사용 가능한 데이터를 처음에 전부 파악하라. 나중에 "이 데이터 안 썼네"하고 돌아가면 시간 낭비.

---

## Phase 2: 방향 예측력 분석 (Feature Predictability)

> "각 피처가 미래 방향을 얼마나 예측하는지 정량적으로 측정한다"

### 2-1. 피처별 IC (Information Coefficient) 측정

모든 피처에 대해:
- Spearman rank correlation with forward return
- Median split: 피처 high 그룹 vs low 그룹의 WR 차이 (Δ_WR)
- Opportunity zone 내에서만 측정 (전체가 아님)

**우리의 발견**:
```
1위: VWAP distance     IC=-0.088 (반전)
2위: Momentum 3bar     IC=-0.086 (반전)
3위: Range position    IC=-0.081 (반전)
4위: CVD slope         IC=-0.079 (반전)
5위: Depth imbalance   IC=+0.068 (유일한 순방향)
```

**핵심 인사이트**: 모든 피처가 mean-reversion 방향. Depth만 순방향.

### 2-2. 피처 조합 테스트

단일 피처로 55% WR 달성 불가 → 조합 테스트:
- 2-factor, 3-factor 조합의 WR
- 조건부 분석 (예: "bid wall + 가격 하락 → long")

**우리의 발견**: 어떤 조합도 55% WR 미달. **선형 피처 조합의 한계 확인.**

### 2-3. Look-ahead 점검 (매우 중요!)

상위 타임프레임 피처 사용 시 **반드시 shift(1)** 적용:
```python
# ❌ WRONG: 현재 1h 봉의 body → 아직 완성 안 된 봉 참조
feat["tf1h_body"] = k1h_body.reindex(k5.index, method="ffill")

# ✅ CORRECT: 이전 완성된 1h 봉
feat["tf1h_body"] = k1h_body.shift(1).reindex(k5.index, method="ffill")
```

**우리의 실수**: 1h body IC=+0.30으로 나왔는데, shift(1) 적용 후 IC=-0.05로 추락. 100% look-ahead였음.

---

## Phase 3: 라벨링 설계 (Label Engineering)

> "모델이 무엇을 예측하게 할 것인가"

### 3-1. 라벨링 방식 진화

| 버전 | 방식 | 결과 |
|---|---|---|
| v1 | 고정 TP/SL 배리어 | 전체 EV 음수 |
| v2 | Fee-aware + dynamic barrier | 개선되었으나 여전히 음수 |
| v3 | Optimal action (hindsight 최적 행동) | look-ahead 내포 → 실제 평가 시 무의미 |
| **최종** | **3-bar direction (binary)** | **가장 단순하고 정직** |

**교훈**: 
- 복잡한 라벨이 항상 좋은 것이 아니다
- v3의 optimal action은 "모델이 맞추면 수익"이라는 환상을 줬지만, 실제 close-to-close 평가에서 무효
- **가장 단순한 binary direction 라벨이 가장 정직했다**

### 3-2. 라벨과 평가의 일치

```
라벨 = "3바 후 가격이 올랐는가?" (binary)
평가 = "3바 후 close에서 청산, fee 차감" (close-to-close)
```

라벨과 평가 방식이 일치해야 한다. TP/SL로 라벨을 만들고 close-to-close로 평가하면 불일치.

---

## Phase 4: 모델 선택 (Model Selection)

> "tabular 데이터에서 무엇이 가장 잘 작동하는가"

### 4-1. 모델 비교

| 모델 | 결과 | 비고 |
|---|---|---|
| MLP (128→64) | val loss ~0.691 | 거의 랜덤 |
| MLP-256 (3층) | 미세 개선 | |
| SeqCNN (1D-CNN) | MLP보다 약간 나음 | 복잡도 대비 이점 적음 |
| **LightGBM** | **val loss ~0.674** | **압도적 승자** |
| GBM Ensemble (x3) | 추가 개선 | |

**교훈**: 피처 수 15-40개의 tabular 데이터에서 **GBM이 NN보다 확실히 우월**. NN은 시퀀스 데이터에서만 의미.

### 4-2. Dual Model 아키텍처

```
Long 모델: P(price goes up in 3 bars)
Short 모델: P(price goes down in 3 bars)

진입: max(P_long, P_short) > threshold
방향: argmax
```

방향 룰(depth+MR)을 제거하고 **모델이 직접 방향 결정**하게 하니 **+65% 수익 개선**.

---

## Phase 5: 시뮬레이션 정직성 (Honest Evaluation)

> "자기 자신을 속이지 않는 것이 가장 어렵다"

### 5-1. 발견한 Bias 목록

| Bias | 설명 | 해결 |
|---|---|---|
| **Intra-bar TP/SL** | 5m 바에서 TP와 SL 중 뭐가 먼저인지 모름 | **close-to-close로 전환** |
| **Trailing stop on 5m** | 같은 바의 high/low 동시 참조 | **1m 검증 또는 사용 금지** |
| **Global Q80 threshold** | 미래 데이터로 필터 임계값 계산 | **expanding quantile 사용** |
| **v3 label edge** | 최적 청산 시점을 아는 상태에서 P&L 계산 | **close-to-close로 전환** |
| **Coin selection bias** | 양수 코인만 골라서 성과 보고 | **유니버스 고정 후 OOS 검증** |

**5m 해상도에서 보이던 +0.094%/trade가 1m close-to-close에서 사라질 수 있다.**
(단, 최종 결과는 5m close-to-close로 TP/SL 없이 평가하여 이 bias 해당 없음)

### 5-2. 1m 해상도 검증의 교훈

5m에서 양수였던 결과를 1m close-to-close로 재검증했을 때:
- ETH: 5m +0.067% → 1m -0.077% (완전히 사라짐)
- 이유: 5m TP/SL이 intra-bar에서 TP 우선 가정을 내포

**최종 전략은 5m close-to-close (3바 후 close에서 청산)** → intra-bar bias 없음.

---

## Phase 6: 코인 유니버스 (Asset Selection)

> "어디서 거래하느냐가 어떻게 거래하느냐보다 중요하다"

### 6-1. 97개 코인 스캔 결과

```
val_loss < 0.685 (비효율 큰 시장): 대부분 양수
val_loss > 0.690 (효율적 시장): 대부분 음수

ETH: 0.691 → 15분 스캘핑 불가
BTC: 0.691 → 15분 스캘핑 불가
PEPE: 0.671 → Taker +0.054%
DOGE: 0.668 → Taker 손익분기
WLD: 0.668 → Taker +0.074%
ARB: 0.664 → Taker +0.033%
```

**핵심 교훈**: 
- **ETH/BTC에서 실패했다고 스캘핑이 불가능한 게 아니다**
- **중소형 알트에서는 같은 모델이 작동한다**
- 시장의 비효율 정도(val_loss)가 수익성을 결정

### 6-2. Universal Model

코인별로 모델을 만드는 대신 **1개의 범용 모델**:
- `coin_id`를 categorical feature로 사용
- 15개 코인 × 5.4M bars로 학습
- 코인 간 공통 패턴 + 코인별 특성 동시 학습

---

## Phase 7: 최종 전략 사양

```
모델:      Universal GBM (Dual Long/Short, Ensemble x2-3)
피처:      15개 (buy_ratio, coin_id, tc_ratio, body, vwap_dist, ...)
코인:      12개 중소형 알트 (Taker-positive만)
진입:      max(P_long, P_short) ≥ 0.58
청산:      3바(15분) 후 close
Fee:       Taker 0.08%
결과:      WR 55.5%, avg +0.094%/trade, 566 trades/day, DD 2.7%
검증:      Walk-forward OOS, close-to-close, no look-ahead
```

---

## 연구 순서 요약 (다음 모델에도 적용)

```
1. 이벤트 탐색     → "수익적 순간"을 먼저 찾는다
2. 피처 예측력     → IC 측정, 조합 테스트
3. Look-ahead 점검 → 상위TF shift, expanding quantile
4. 라벨링 설계     → 단순한 것부터, 평가와 일치시키기
5. 모델 선택       → GBM > NN (tabular), Dual model
6. 정직한 평가     → close-to-close, 1m 검증
7. 코인 유니버스   → 비효율 큰 시장에서 거래
8. Walk-forward    → 롤링 OOS 검증
9. Codex 리뷰      → 외부 시각으로 bias 점검
```

---

## 실패한 접근들 (기록용)

| 접근 | 왜 실패 | 교훈 |
|---|---|---|
| 방향 예측 (ETH/BTC) | IC 0.05~0.09, fee 이기기 불충분 | 효율적 시장에서 단기 방향 예측은 거의 불가 |
| v3 optimal action label | look-ahead 내포 | 라벨이 복잡하면 평가와 불일치 위험 |
| 5m TP/SL 배리어 | intra-bar bias | close-to-close만 신뢰 |
| Trailing stop (5m) | 같은 바 high/low 동시 참조 | 5m에서는 사용 불가 |
| Funding rate extreme | 특정 시기만 양수 (survivorship) | 전체 기간 walk-forward 필수 |
| BTC→ETH lead-lag | 상관관계 -0.02 (무의미) | cross-asset 리드래그 없음 |
| ETH/BTC pair trading | 모든 조합 음수 | ratio mean-reversion 없음 |
| Straddle (양방향) | 2x fee로 상쇄 | 방향 중립 전략은 fee 2배 |
| BB, RSI, Volume surge | 전부 fee 근처 음수 | 클래식 지표는 crypto에서 무효 |
| MLP/SeqCNN | GBM 대비 열등 | tabular에서 NN 쓰지 말 것 |

---

## 핵심 수치 참조

| 항목 | ETH (효율적) | PEPE (비효율) |
|---|---|---|
| GBM val_loss | 0.691 | 0.671 |
| 15min WR (Maker) | 67% (p≥0.58) | 61% |
| 15min avg (Maker) | -0.003% | +0.114% |
| 15min avg (Taker) | -0.063% | +0.054% |
| GBM iterations | 40-100 | 150-250 |

**val_loss가 0.685 이하면 스캘핑 가능. 0.690 이상이면 불가.**

---

## Appendix: 단타 (Intraday) 연구 결과

같은 Playbook 순서를 따라 단타 모델을 개발함.

### 스캘핑 vs 단타 차이

| | 스캘핑 | 단타 |
|---|---|---|
| 홀딩 | 15분 고정 | **1-4h 동적 (Checkpoint)** |
| 타겟 | Binary direction | **Regression (수익률 예측)** |
| 핵심 피처 | buy_ratio (미시구조) | **btc_vol, OI (포지셔닝)** |
| Fee 비중 | 23% of std | **4.7%** |
| 양수 코인 | 12/15 (알트만) | **67/85 (79%, 메이저 포함)** |
| 거래당 수익 | +0.094% | **+0.329%** (3.5배) |

### 단타 핵심 발견

1. **BTC 변동성이 알트 단타의 #1 드라이버** — BTC가 흔들릴 때 알트에서 기회
2. **Regression이 Classification보다 우월** — 스캘핑에서는 실패했지만 단타에서 성공 (4h return std 충분)
3. **Checkpoint exit** — 1h+2h 모델 동의 시 더 오래 보유, 불일치 시 조기 청산 (+39% 개선)
4. **79% 코인 양수** — 스캘핑보다 훨씬 광범위
5. **Cross-asset 피처** — btc_ret_1h, btc_oi_chg, alt_btc_z가 중요

### 단타 전략 사양

```
모델:      Regression GBM × 3 horizons (1h/2h/4h), ensemble x2
피처:      17개 (btc_vol, oi_level_z, ret_24h, funding, ...)
코인:      20개 알트 (67/85 양수 중 Top 20)
진입:      |pred_2h| > 0.002
청산:      Checkpoint — 1h+2h agree+strong→4h, agree→2h, disagree→1h
결과:      WR 52.4%, Taker +0.329%/trade, $100K total OOS
검증:      Walk-forward, close-to-close, 1m 교차 검증 (corr 0.98+)
```

### 단타에서 실패한 것

- BTC/ETH 단독 regression → iter 1~8 (학습 불가)
- 1h hold on BTC → fee 비중 12%, 불충분
- Dynamic hold (단순 best horizon 선택) → 93% 4h로 편향
- 피처 17→30+ → 과적합으로 성능 하락
- 학습 9mo→18mo → 오래된 데이터가 noise
- 조기 손절 (SL -0.5~2%) → 모두 성능 하락 (2h c2c 최적)
- Vol-normalized target → raw return보다 못함
- Separated (vol predict + direction) → combined regression보다 못함

### 단타 최종 진화 과정 (모든 개선 기록)

| # | 개선 | 효과 | WF 검증 |
|---|---|---|---|
| 1 | LGB regression 2h | baseline +0.084% | ✅ |
| 2 | Checkpoint exit (1h+2h agree) | **+176%** | ✅ |
| 3 | CatBoost 교체 | **+48%** vs LGB | ✅ |
| 4 | Composite target (1h+2h+4h equal) | **+36%** total | ✅ |
| 5 | 20코인 universal | 20/20 양수 | ✅ |
| 6 | Reg+Cls agree filter | Quality tier | ✅ |
| 7 | Tiered sizing ($300/500/800) | +12% equity | ✅ |
| **최종** | **CB + Composite + Cls + Checkpoint** | **+0.424%/trade** | **✅** |

### 최종 단타 전략 사양 (2026-04-08)

```
모델:      CatBoost regression (composite: equal 1h+2h+4h)
           + LightGBM dual classification (Quality filter p>0.55)
피처:      17개 (oi_chg, funding, taker_ls_z, ret, vol, buy_ratio,
           btc_ret_1h, btc_oi_chg, btc_vol, alt_btc_z, hour)
코인:      20개 알트 (Tier A/B/C)
진입:      |predicted composite return| > 0.002
청산:      Checkpoint — |pred|>0.005 → 4h, else → 2h
사이징:    Tiered ($300/$500/$800 by |pred|)
검증:      30-window rolling WF, 20/20 코인 양수
성과:      Quality: +0.424%/trade, WR 54.7%, 81K trades
           Volume: +0.213%/trade, 247K trades, $263K total
```

### 단타 핵심 교훈: 스케일 의존적 과적합

**8코인에서 양수인 개선이 20코인에서 실패하는 패턴이 4번 반복:**

| 개선 | 8코인 | 20코인 | 교훈 |
|---|---|---|---|
| RMSE loss | +28% | -30% | loss function 변경은 특정 코인에 과적합 |
| Target ensemble | +16% | -25% | 다중 타겟 평균화가 희석 |
| Huber loss | +82% | -4% | outlier handling이 스케일에 일반화 안 됨 |
| Time-weighted | +71% | -17% | 최근 가중이 일부 코인에만 유효 |

**프로덕션 규칙**: 모든 최종 결정은 20코인+ rolling WF에서 검증. 8코인 결과는 탐색용으로만.

### 단타 전체 실험 로그 (24번)

**성공 (8)**: Checkpoint, CatBoost, Composite target, 20-coin, Reg+Cls, Tiered sizing, depth=6, l2=3

**실패 (16)**: 30+ features, 18mo train, stop-loss, vol-normalized, separated vol+dir, CB cls, profit checkpoint, RMSE, Huber, target ensemble, cross-section ranking, cross-coin features, uncertainty filter, time-weighted, dynamic coin selection, more regularization

---

*마지막 업데이트: 2026-04-08*
*작성: Model Researcher + PM*

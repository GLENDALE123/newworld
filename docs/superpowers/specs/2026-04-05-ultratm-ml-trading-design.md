# ultraTM: TBM + 4계층 계층적 앙상블 암호화폐 선물 매매 시스템

## 개요

NautilusTrader 기반의 4계층 계층적 앙상블 매매 시스템. 바이낸스 USDT-M 선물 시장에서
멀티 종목/멀티 타임프레임 전략을 백테스트한다.

핵심 철학 두 가지:
1. **아키텍처가 지능이다.** 복잡한 딥러닝 대신 CatBoost + 룰 기반으로 통일하되,
   4계층 구조로 엣지를 만든다.
2. **TBM(Triple Barrier Method)이 기반이다.** 단순 방향 라벨이 아니라 "익절/손절/시간초과
   중 어디에 먼저 닿는가"로 문제를 정의한다. TBM은 라벨링, 메타 레이블링, 리스크 관리
   세 곳에서 동시에 작동한다.

1차 목표는 백테스트를 통한 전략 검증과 학습이며, 실거래 배포는 범위 밖이다.

## Triple Barrier Method (TBM) 개요

TBM은 모든 Phase에서 사용되는 핵심 라벨링 프레임워크다.

### 세 개의 장벽

```
가격
 ^
 |     ---- 상단 장벽 (Take Profit): pt x sigma_t ----
 |    /
 |   /  가격 경로
 |  /
 | *  <-- 진입 시점
 |  \
 |   \
 |     ---- 하단 장벽 (Stop Loss): sl x sigma_t ----
 |
 +---------------------------------------------------> 시간
 |<--- 수직 장벽 (Max Holding Period) --->|
```

- **상단 장벽**: 진입가 + pt x sigma_t (익절)
- **하단 장벽**: 진입가 - sl x sigma_t (손절)
- **수직 장벽**: 최대 보유 시간 초과 시 현재 손익으로 청산
- **sigma_t**: 최근 24시간 지수이동평균 변동성 (동적)

### 라벨 규칙

| 먼저 닿은 장벽 | 라벨 | 의미 |
|----------------|------|------|
| 상단 | +1 | 익절 성공 (롱 정답) |
| 하단 | -1 | 손절 (숏 정답) |
| 수직 (손익 > 0) | +1 | 시간 초과했지만 이익 |
| 수직 (손익 <= 0) | -1 | 시간 초과, 손실 |

### TBM이 작동하는 세 곳

| 위치 | 역할 | Phase |
|------|------|-------|
| 학습 라벨 | Primary Model의 타겟 생성 | Phase 1~ |
| 메타 라벨링 | Meta Model의 타겟 생성 (Primary가 맞았는가?) | Phase 5 |
| 리스크 관리 | 실제 매매의 TP/SL/시간 청산 기준 | Phase 1~ |

## 최종 아키텍처 (4계층 + TBM)

```
    Raw Data --> TBM Labeler --> 학습 라벨 생성 (모든 모델의 타겟)
                                       |
                                       v
[1] 데이터 계층 - Source Ensemble (Primary Models)
    "익절에 먼저 닿을까, 손절에 먼저 닿을까?"

         Price Data        Orderbook Data      On-chain Data
         (OHLCV+지표)      (호가잔량,CVD)       (펀딩비,OI,청산)
              |                  |                    |
              v                  v                    v
        +-----------+     +-----------+        +-----------+
        |  Price    |     | Liquidity |        | External  |
        |  Model    |     | Model     |        | Model     |
        | (CatBoost)|     | (CatBoost)|        | (CatBoost)|
        +-----+-----+     +-----+-----+        +-----+-----+
              |                  |                    |
              +------------------+--------------------+
                                 |
                                 v
[2] 필터링 계층 - Multi-Timeframe Filter (룰 기반)
    상위 타임프레임이 하위 신호를 검열

        4h 대추세 판단 --> 역방향 신호 차단/비중 축소
                                 |
                                 v
[3] 전문가 계층 - Regime Experts
    시장 국면에 맞는 전문가 가중치 조절

        +-------------+  +-------------+  +-------------+
        | Trend       |  | MeanRevert  |  | Volatility  |
        | Expert      |  | Expert      |  | Expert      |
        | (CatBoost)  |  | (CatBoost)  |  | (CatBoost)  |
        +------+------+  +------+------+  +------+------+
               |                |                |
               +--------+-------+--------+-------+
                        |
                  Gating Network (국면 분류, CatBoost)
                        |
                        v
[4] 메타 레이블링 계층 - Meta Decision (TBM 기반)
    "이 신호가 실제로 상단 장벽에 닿을 확률은?"

        Primary Signal (위 계층 출력: 롱/숏)
              |
              v
        +-----------+
        | Meta      |    타겟: Primary가 맞아서 상단 장벽에 닿았으면 1, 아니면 0
        | Model     |    출력: 확률 p (0.0 ~ 1.0)
        | (CatBoost)|
        +-----+-----+
              |
              v
        +-----------+
        | Bet Sizer |    p 기반 Kelly Criterion --> 포지션 사이즈
        +-----------+
              |
              v
        +-----------+
        | MLStrategy|    TBM 장벽으로 TP/SL/시간청산 설정 + 주문 실행
        | (Nautilus)|
        +-----------+
```

## 대상 시장

- **거래소**: Binance Futures (USDT-M Perpetual)
- **종목**: BTCUSDT, ETHUSDT, SOLUSDT
- **타임프레임**: 5m, 15m, 1h, 4h
- **학습 데이터**: 최근 6~12개월

## 기술 스택

| 구분 | 기술 |
|------|------|
| 트레이딩 엔진 | NautilusTrader (Binance Futures 어댑터) |
| ML 모델 (전부) | CatBoost (GPU, `task_type='GPU'`) |
| 데이터 저장 | Parquet (pandas/polars) |
| 피처 엔지니어링 | pandas, pandas-ta |
| 시각화 | matplotlib, plotly |
| 설정 관리 | Pydantic Settings |
| 테스트 | pytest |
| Python | 3.12+ |

---

## Phase 1: 기초 -- TBM + 단일 모델 백테스트

**목표**: BTCUSDT 1개 종목, 1h 1개 타임프레임으로 TBM 라벨링 → CatBoost 학습 →
NautilusTrader 백테스트가 동작하는 최소 파이프라인을 만든다.

### 1.1 프로젝트 구조

```
ultraTM/
├── data/
│   ├── collectors/
│   │   └── ohlcv.py            # Binance OHLCV 수집
│   └── storage/
│       └── parquet.py          # Parquet 저장/로딩
├── labeling/
│   └── tbm.py                  # Triple Barrier Method 라벨러
├── features/
│   ├── technical.py            # 기술지표 계산
│   └── pipeline.py             # 피처 파이프라인
├── models/
│   └── catboost_model.py       # CatBoost 학습/예측
├── strategy/
│   └── ml_strategy.py          # NautilusTrader Strategy (TBM 장벽 기반)
├── backtest/
│   ├── runner.py               # 백테스트 실행
│   └── analysis.py             # 성과 분석
├── config/
│   └── settings.py             # 설정
└── tests/
```

### 1.2 데이터 파이프라인

1. Binance REST API (`GET /fapi/v1/klines`)로 BTCUSDT 1h OHLCV 다운로드
2. Parquet 포맷으로 `data/storage/BTCUSDT_1h.parquet` 저장
3. NautilusTrader `Bar` 객체로 변환

### 1.3 TBM 라벨러

Phase 1의 가장 중요한 구현. 모든 이후 Phase의 기반이 된다.

```python
class TripleBarrierLabeler:
    def __init__(self, pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4):
        """
        pt_multiplier: 익절 장벽 = pt_multiplier x sigma_t
        sl_multiplier: 손절 장벽 = sl_multiplier x sigma_t
        max_holding_bars: 수직 장벽 (최대 보유 봉 수)
        """

    def compute_volatility(self, close_prices, span=24):
        """지수이동평균 변동성 계산"""

    def label(self, ohlcv_df) -> pd.Series:
        """
        각 봉마다:
        1. sigma_t 계산
        2. 상단 장벽 = close + pt * sigma_t
        3. 하단 장벽 = close - sl * sigma_t
        4. 이후 max_holding_bars 내에서 어느 장벽에 먼저 닿는지 판정
        5. 라벨 반환: +1, -1
        """
```

Phase 1 TBM 파라미터:
- `pt_multiplier`: 1.0 (변동성의 1배를 익절 목표)
- `sl_multiplier`: 1.0 (변동성의 1배를 손절 기준)
- `max_holding_bars`: 4 (4시간 최대 보유)

### 1.4 피처 엔지니어링

| 카테고리 | 피처 |
|----------|------|
| 추세 | SMA(20, 50), EMA(12, 26), MACD(12, 26, 9) |
| 모멘텀 | RSI(14), Stochastic(14, 3) |
| 변동성 | Bollinger Bands(20, 2), ATR(14) |
| 거래량 | OBV, Volume SMA ratio(20) |

### 1.5 CatBoost 모델

- **종목**: BTCUSDT만
- **타임프레임**: 1h만
- **입력**: 위 피처 테이블
- **타겟**: TBM 라벨 (+1=상단 장벽 도달, -1=하단 장벽 도달)
- **학습**: Walk-forward validation
  - 학습 윈도우: 3개월 롤링
  - 검증 윈도우: 1개월
- **GPU**: `task_type='GPU'`
- **평가**: Accuracy, Precision, F1, 백테스트 수익

### 1.6 NautilusTrader Strategy (TBM 장벽 기반)

Strategy가 직접 TBM 장벽을 TP/SL로 사용한다.

```python
class MLStrategy(Strategy):
    def on_bar(self, bar: Bar):
        features = self.feature_pipeline.compute(bar)
        signal = self.model.predict(features)
        sigma_t = self.compute_volatility(bar)

        if signal == 1 and not self.has_position():
            entry_price = bar.close
            tp = entry_price + self.pt * sigma_t   # 상단 장벽
            sl = entry_price - self.sl * sigma_t   # 하단 장벽
            self.enter_long(bar.instrument_id, tp=tp, sl=sl)
            self.holding_start = bar.ts_event      # 수직 장벽 타이머 시작

        elif signal == -1 and not self.has_position():
            entry_price = bar.close
            tp = entry_price - self.pt * sigma_t
            sl = entry_price + self.sl * sigma_t
            self.enter_short(bar.instrument_id, tp=tp, sl=sl)
            self.holding_start = bar.ts_event

        # 수직 장벽: 최대 보유 시간 초과 시 청산
        if self.has_position() and self.bars_held >= self.max_holding_bars:
            self.close_position()
```

### 1.7 리스크 관리

- **TP/SL**: TBM 장벽이 곧 TP/SL (변동성 스케일링)
- **포지션 사이징**: 계좌 자본의 2% 고정
- **최대 보유**: max_holding_bars 초과 시 강제 청산
- **최대 포지션**: 1개

### 1.8 성공 기준

- TBM 라벨링이 정상 동작 (라벨 분포 확인: +1/-1 비율이 극단적이지 않음)
- 데이터 수집 → TBM 라벨 → 피처 → 학습 → 백테스트 전체 파이프라인 동작
- 백테스트에서 TP/SL/시간청산이 정확히 작동
- 결과 리포트: Sharpe, MDD, Win Rate, Profit Factor

---

## Phase 2: 멀티 종목 + 멀티 타임프레임 필터

**목표**: 3개 종목으로 확장하고, 상위 타임프레임 필터링(2계층)을 추가한다.

### 2.1 멀티 종목 확장

- BTCUSDT, ETHUSDT, SOLUSDT 3개 종목
- 종목별 TBM 파라미터 자동 조정 (종목마다 변동성이 다르므로 sigma_t가 자동 반영)
- 종목별 독립 포지션 관리

### 2.2 멀티 타임프레임 필터 (2계층)

**이것이 아키텍처 2계층의 첫 구현이다.**

```
4h 대추세 판단 (룰 기반)
    |
    +--> 상승 추세: 1h 모델의 롱 신호만 허용
    +--> 하락 추세: 1h 모델의 숏 신호만 허용
    +--> 횡보: 1h 모델의 모든 신호 비중 50% 축소
```

대추세 판단 룰:
- 4h EMA(50) 위에 가격 --> 상승 추세
- 4h EMA(50) 아래에 가격 --> 하락 추세
- 4h BB(20) 폭이 ATR 대비 좁으면 --> 횡보

### 2.3 TBM 파라미터 적응

타임프레임 필터 결과에 따라 TBM 장벽도 조정:
- 추세 순방향 신호: pt_multiplier 확대 (1.5x) — 추세를 더 타게
- 횡보 신호: pt_multiplier 축소 (0.7x) — 빨리 익절

### 2.4 추가 피처

- 상위 타임프레임(4h) 지표를 하위(1h)에 병합: 4h RSI, 4h MACD, 4h BB 위치
- 5m/15m 데이터 수집 시작 (Phase 3 준비)

### 2.5 성공 기준

- 3개 종목 동시 백테스트 동작
- 타임프레임 필터 ON/OFF 비교 시 필터가 역방향 노이즈 매매를 줄이는지 확인
- TBM 장벽 적응이 고정 장벽 대비 수익률 개선

---

## Phase 3: 데이터 소스 앙상블 (1계층)

**목표**: 오더북, 온체인 데이터를 추가하고 데이터 소스별 독립 Primary Model 3개를 만든다.

### 3.1 추가 데이터 수집

| 데이터 | 소스 | 저장 |
|--------|------|------|
| 오더북 스냅샷 (5~10 레벨) | Binance WebSocket | Parquet |
| 펀딩비 | Binance REST API | Parquet |
| 미결제약정 (OI) | Binance REST API | Parquet |
| 롱숏 비율 | Binance REST API | Parquet |
| 청산 데이터 | Binance WebSocket | Parquet |

### 3.2 데이터 소스별 Primary Model (1계층 구현)

**세 모델은 완전히 독립적으로 학습된다. 타겟은 모두 동일한 TBM 라벨.**

| 모델 | 입력 피처 | 질문 |
|------|-----------|------|
| Price Model | OHLCV + 기술지표 | "가격 패턴상 익절에 먼저 닿을까?" |
| Liquidity Model | 오더북 스프레드, OBI, 깊이, 대형벽 | "수급상 익절에 먼저 닿을까?" |
| External Model | 펀딩비 변화율, OI 다이버전스, 롱숏비율, 청산 클러스터 | "파생 지표상 익절에 먼저 닿을까?" |

### 3.3 소스 앙상블 규칙

- 3개 모델 중 2개 이상 동의 --> 신호 발생
- 3개 모두 동의 --> 강한 엣지, TBM pt_multiplier 확대 가능

### 3.4 성공 기준

- 3개 소스 모델 앙상블이 Phase 2의 단일 Price Model 대비 Precision 개선
- "3개 모두 동의" 신호의 승률이 유의미하게 높은지 검증

---

## Phase 4: 국면 전문가 (3계층)

**목표**: 시장 국면(Regime)을 분류하고, 국면별 전문가 모델의 가중치를 조절한다.

### 4.1 국면 분류 (Gating Network)

CatBoost 분류기로 현재 시장 국면을 3가지로 분류:

| 국면 | 특징 | 판단 피처 |
|------|------|-----------|
| 추세장 (Trending) | 방향성 강함, 변동성 중간 | ADX > 25, BB 확장, 이평선 정배열/역배열 |
| 횡보장 (Range) | 방향성 약함, 변동성 낮음 | ADX < 20, BB 수축, 가격이 BB 중심 근처 |
| 폭발장 (Volatile) | 급등/급락, 변동성 높음 | ATR 급등, 대량 청산 발생, OI 급변 |

### 4.2 국면별 전문가

| 전문가 | 최적 국면 | TBM 파라미터 특성 |
|--------|-----------|-------------------|
| Trend Expert | 추세장 | pt 크게(1.5x), sl 보통, 보유시간 길게(8봉) |
| MeanRevert Expert | 횡보장 | pt 작게(0.7x), sl 작게, 보유시간 짧게(2봉) |
| Volatility Expert | 폭발장 | pt 크게(2.0x), sl 넓게(1.5x), 보유시간 짧게(2봉) |

각 전문가도 CatBoost 모델이며, 해당 국면 구간의 데이터만 필터링하여 학습한다.
**전문가마다 TBM 파라미터가 다르다** — 국면에 맞는 리스크/리워드 비율을 적용.

### 4.3 가중치 조절

```
Gating Network 출력: [추세: 0.7, 횡보: 0.2, 폭발: 0.1]

최종 신호 = Trend Expert * 0.7 + MR Expert * 0.2 + Vol Expert * 0.1
최종 TBM 파라미터 = Trend TBM * 0.7 + MR TBM * 0.2 + Vol TBM * 0.1
```

### 4.4 성공 기준

- Gating Network의 국면 분류 정확도 70% 이상
- 국면별 전문가 시스템이 Phase 3의 단일 앙상블 대비 드로다운 감소
- 국면별 TBM 파라미터 적응이 고정 파라미터 대비 개선

---

## Phase 5: 메타 레이블링 + Bet Sizing (4계층)

**목표**: TBM 기반 메타 레이블링으로 "방향"과 "확신"을 분리하고,
확률 기반 포지션 사이징으로 Precision을 극대화한다.

### 5.1 메타 라벨 생성 (TBM 핵심 활용)

```
1. Primary Model이 "롱" 신호 발생
2. 실제로 상단 장벽에 닿았는가?
   --> 닿았으면: meta_label = 1 (성공)
   --> 못 닿았으면: meta_label = 0 (실패)
3. 이 데이터로 Meta Model 학습
```

### 5.2 Primary Model vs Meta Model

| 구분 | Primary Model | Meta Model |
|------|---------------|------------|
| 질문 | "익절에 먼저 닿을까 손절에 먼저 닿을까?" | "Primary가 맞아서 정말 장벽에 닿을까?" |
| 타겟 | TBM 라벨 (+1/-1) | meta_label (1/0) |
| 입력 | 데이터 소스별 피처 | Primary 신호 시점의 시장 환경 |
| 출력 | 방향 (롱/숏) | 확률 p (0.0 ~ 1.0) |
| 역할 | 100번 신호 생성 | 그중 승률 높은 30번만 통과 |

### 5.3 Meta Model 입력 피처

Primary Model이 신호를 낸 시점의:
- 현재 변동성 sigma_t (TBM 장벽 폭의 직접적 근거)
- 현재 거래량 vs 24h 평균 거래량
- 소스 모델 합의 수준 (3개 중 몇 개 동의)
- Gating Network 국면 확신도
- 오더북 불균형 (OBI)
- 펀딩비 방향
- 최근 N회 매매의 연속 승/패 수 (전략 최근 성적)

### 5.4 Bet Sizing (Kelly Criterion)

Meta Model의 확률 p를 Kelly Criterion에 투입:

```
f* = (p * b - q) / b

f*: 최적 베팅 비율 (자본 대비)
p:  Meta Model 출력 확률 (승리 확률)
q:  1 - p (패배 확률)
b:  평균 승리 / 평균 손실 비율 (TBM pt/sl 비율에서 도출)
```

실전 안전장치:
- Half Kelly 적용 (f* / 2) — 과적합 리스크 감소
- 최소 비중: 자본의 0.5%
- 최대 비중: 자본의 5%
- p < 0.4일 때: 진입하지 않음 (Meta가 거부)

### 5.5 리스크 관리 최종

- **TP/SL**: 국면별 TBM 장벽 (Phase 4에서 결정)
- **시간 청산**: 국면별 수직 장벽
- **포지션 사이징**: Meta 확률 x Half Kelly
- **상관관계 필터**: 멀티 종목 동시 포지션 간 상관관계 > 0.7이면 총 익스포저 제한
- **드로다운 보호**: 최대 드로다운 10% 초과 시 REDUCING 상태

### 5.6 성공 기준

- Meta Model 도입 후 Precision 10% 이상 개선 (100번 중 30번만 치되 승률 대폭 상승)
- Kelly 사이징이 고정 사이징 대비 수익률 및 Sharpe 개선
- 전체 시스템 백테스트 Sharpe ratio > 1.0

---

## Phase별 요약

| Phase | 핵심 추가 | 아키텍처 계층 | TBM 역할 |
|-------|-----------|---------------|----------|
| 1 | BTCUSDT + 1h + CatBoost 1개 | - | 라벨링 + TP/SL |
| 2 | 멀티 종목 + 4h 필터 | 2계층 (타임프레임 필터) | 추세별 장벽 적응 |
| 3 | 오더북/온체인 + 소스별 모델 3개 | 1계층 (데이터 앙상블) | 합의 시 장벽 확대 |
| 4 | 국면 분류 + 전문가 3개 | 3계층 (국면 전문가) | 국면별 TBM 파라미터 |
| 5 | 메타 레이블링 + Kelly Sizing | 4계층 (메타 의사결정) | 메타 라벨 생성 + Bet Sizing |

## 전체 데이터 흐름 (Phase 5 완성 시)

```
1. Raw OHLCV/Orderbook/On-chain --> TBM Labeler --> 학습 라벨 (+1/-1)

2. Features --> Primary Models (Price/Liquidity/External CatBoost x3)
   --> "익절 도달 확률이 높은 방향(Side)" 출력

3. 4h 타임프레임 필터 --> 역방향 신호 차단

4. Gating Network --> 국면 판단 --> 전문가 가중 선택
   --> 국면별 TBM 파라미터 결정

5. Primary Signal + 시장 환경 --> Meta Model
   --> "이 신호가 실제 상단 장벽에 닿을 확률(p)" 출력

6. p --> Kelly Criterion --> 포지션 사이즈 결정

7. MLStrategy --> NautilusTrader 주문 실행
   (TP = 상단장벽, SL = 하단장벽, 시간청산 = 수직장벽)
```

## 범위 밖

- 실거래 (라이브 트레이딩) 배포
- 웹 UI / 대시보드
- 멀티 거래소 동시 거래
- 고빈도 매매 (HFT)
- 딥러닝 모델 (LSTM, Transformer)
- 강화학습 (RL)

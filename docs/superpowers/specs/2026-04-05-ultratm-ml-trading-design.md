# ultraTM: 4계층 계층적 앙상블 암호화폐 선물 매매 시스템

## 개요

NautilusTrader 기반의 4계층 계층적 앙상블 매매 시스템. 바이낸스 USDT-M 선물 시장에서
멀티 종목/멀티 타임프레임 전략을 백테스트한다.

핵심 철학: **아키텍처가 지능이다.** 복잡한 딥러닝 모델 대신 CatBoost + 룰 기반으로
통일하되, 4계층 구조(데이터 앙상블 → 타임프레임 필터 → 국면 전문가 → 메타 레이블링)로
엣지를 만든다.

1차 목표는 백테스트를 통한 전략 검증과 학습이며, 실거래 배포는 범위 밖이다.

## 최종 아키텍처 (4계층)

```
[1] 데이터 계층 - Source Ensemble
    각 데이터 소스별 독립 모델이 방향성 예측

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
[2] 필터링 계층 - Multi-Timeframe Filter
    상위 타임프레임이 하위를 검열 (룰 기반)

        4h 대추세 판단 --> 하락장이면 매수 신호 차단/비중 축소
                                 |
                                 v
[3] 전문가 계층 - Regime Experts
    시장 국면에 맞는 전문가의 가중치를 높임

        +-------------+  +-------------+  +-------------+
        | Trend       |  | MeanRevert  |  | Volatility  |
        | Expert      |  | Expert      |  | Expert      |
        | (CatBoost)  |  | (CatBoost)  |  | (CatBoost)  |
        +------+------+  +------+------+  +------+------+
               |                |                |
               +--------+-------+--------+-------+
                        |                |
                  Gating Network         |
                  (국면 분류, CatBoost)    |
                        |                |
                        +--------+-------+
                                 |
                                 v
[4] 메타 레이블링 계층 - Meta Decision
    "방향"과 "확신"을 분리

        Primary Signal (위 계층의 출력: 롱/숏)
              |
              v
        +-----------+
        | Meta      |    "이 신호가 맞을 확률은?"
        | Model     |    "맞다면 얼마나 베팅할까?"
        | (CatBoost)|
        +-----+-----+
              |
              v
        +-----------+
        | MLStrategy|    최종 주문 실행
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

## Phase 1: 기초 — 단일 모델 백테스트 파이프라인

**목표**: BTCUSDT 1개 종목, 1h 1개 타임프레임, CatBoost 1개 모델로 동작하는 최소
백테스트 시스템을 만든다. 이 Phase가 이후 모든 것의 뼈대가 된다.

### 1.1 프로젝트 구조 (Phase 1 범위)

```
ultraTM/
├── data/
│   ├── collectors/
│   │   └── ohlcv.py            # Binance OHLCV 수집
│   └── storage/
│       └── parquet.py          # Parquet 저장/로딩
├── features/
│   ├── technical.py            # 기술지표 계산
│   └── pipeline.py             # 피처 파이프라인
├── models/
│   └── catboost_model.py       # CatBoost 학습/예측
├── strategy/
│   └── ml_strategy.py          # NautilusTrader Strategy
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

### 1.3 피처 엔지니어링

Phase 1은 가격/거래량 기반 기본 지표만 사용한다.

| 카테고리 | 피처 |
|----------|------|
| 추세 | SMA(20, 50), EMA(12, 26), MACD(12, 26, 9) |
| 모멘텀 | RSI(14), Stochastic(14, 3) |
| 변동성 | Bollinger Bands(20, 2), ATR(14) |
| 거래량 | OBV, Volume SMA ratio(20) |

### 1.4 CatBoost 모델

- **종목**: BTCUSDT만
- **타임프레임**: 1h만
- **입력**: 위 피처 테이블
- **타겟**: 향후 3봉(3시간) 수익률 방향 (1=롱, -1=숏, 중립 구간 제외)
- **학습**: Walk-forward validation
  - 학습 윈도우: 3개월 롤링
  - 검증 윈도우: 1개월
- **GPU**: `task_type='GPU'`
- **평가**: Accuracy, Precision, F1, 백테스트 수익

### 1.5 NautilusTrader Strategy

```python
class MLStrategy(Strategy):
    def on_bar(self, bar: Bar):
        features = self.feature_pipeline.compute(bar)
        signal = self.model.predict(features)

        if signal == 1 and not self.has_long_position():
            self.enter_long(bar.instrument_id)
        elif signal == -1 and not self.has_short_position():
            self.enter_short(bar.instrument_id)
        elif signal == 0 and self.has_position():
            self.close_position()
```

### 1.6 리스크 관리

- **포지션 사이징**: 계좌 자본의 2% 고정
- **스탑로스**: ATR(14) x 1.5
- **최대 포지션**: 1개

### 1.7 성공 기준

- 데이터 수집 → 피처 → 학습 → 백테스트 파이프라인이 끊김 없이 동작
- 백테스트 결과(Sharpe, MDD, Win Rate)가 리포트로 출력됨
- Buy & Hold 대비 비교 가능한 상태

---

## Phase 2: 멀티 종목 + 멀티 타임프레임 필터

**목표**: 3개 종목으로 확장하고, 상위 타임프레임 필터링(2계층)을 추가한다.

### 2.1 멀티 종목 확장

- BTCUSDT, ETHUSDT, SOLUSDT 3개 종목
- 종목별 독립 모델 학습 또는 종목을 피처로 포함한 통합 모델
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
- 4h EMA(50) 위에 가격 → 상승 추세
- 4h EMA(50) 아래에 가격 → 하락 추세
- 4h BB(20) 폭이 ATR 대비 좁으면 → 횡보

### 2.3 추가 피처

- 상위 타임프레임(4h) 지표를 하위(1h)에 병합: 4h RSI, 4h MACD, 4h BB 위치
- 5m/15m 데이터 수집 시작 (Phase 3 준비)

### 2.4 성공 기준

- 3개 종목 동시 백테스트 동작
- 타임프레임 필터 ON/OFF 비교 시 필터가 노이즈 매매를 줄이는지 확인

---

## Phase 3: 데이터 소스 앙상블 (1계층)

**목표**: 오더북, 온체인 데이터를 추가하고 데이터 소스별 독립 모델 3개를 만든다.

### 3.1 추가 데이터 수집

| 데이터 | 소스 | 저장 |
|--------|------|------|
| 오더북 스냅샷 (5~10 레벨) | Binance WebSocket | Parquet |
| 펀딩비 | Binance REST API | Parquet |
| 미결제약정 (OI) | Binance REST API | Parquet |
| 롱숏 비율 | Binance REST API | Parquet |
| 청산 데이터 | Binance WebSocket | Parquet |

### 3.2 데이터 소스별 모델 (1계층 구현)

**세 모델은 완전히 독립적으로 학습된다.**

| 모델 | 입력 피처 | 역할 |
|------|-----------|------|
| Price Model | OHLCV + 기술지표 | 가격 패턴 기반 방향 예측 |
| Liquidity Model | 오더북 스프레드, OBI, 깊이, 대형벽 | 유동성/수급 기반 방향 예측 |
| External Model | 펀딩비 변화율, OI 다이버전스, 롱숏비율, 청산 클러스터 | 파생 지표 기반 방향 예측 |

### 3.3 소스 앙상블 규칙

- 3개 모델 중 2개 이상 동의 → 신호 발생
- 3개 모두 동의 → 강한 엣지, 비중 상향 가능

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

| 전문가 | 최적 국면 | 전략 성격 |
|--------|-----------|-----------|
| Trend Expert | 추세장 | 추세 추종 — 브레이크아웃, 이평선 교차 |
| MeanRevert Expert | 횡보장 | 역추세 — BB 상하단, RSI 과매수/과매도 |
| Volatility Expert | 폭발장 | 변동성 포착 — 급변 직전 진입, 빠른 청산 |

각 전문가도 CatBoost 모델이다. 단, 학습 데이터를 해당 국면 구간만 필터링하여 학습한다.

### 4.3 가중치 조절

```
Gating Network 출력: [추세: 0.7, 횡보: 0.2, 폭발: 0.1]

최종 신호 = Trend Expert * 0.7 + MR Expert * 0.2 + Vol Expert * 0.1
```

### 4.4 성공 기준

- Gating Network의 국면 분류 정확도 70% 이상
- 국면별 전문가 시스템이 Phase 3의 단일 앙상블 대비 드로다운 감소

---

## Phase 5: 메타 레이블링 (4계층)

**목표**: "방향 예측"과 "확신도 판단"을 분리하여 Precision을 극대화한다.

### 5.1 Primary Model vs Meta Model

| 구분 | Primary Model | Meta Model |
|------|---------------|------------|
| 질문 | "지금 롱인가 숏인가?" | "이 신호가 맞을 확률은?" |
| 입력 | Phase 1~4의 전체 파이프라인 출력 | Primary 신호 + 시장 환경 피처 |
| 출력 | 방향 (롱/숏) | 확률 (0.0 ~ 1.0) |
| 역할 | 방향 결정 | 진입 여부 + 포지션 사이즈 결정 |

### 5.2 Meta Model 입력 피처

Primary Model이 "매수" 신호를 보낸 시점의:
- 현재 변동성 (ATR, BB 폭)
- 현재 거래량 vs 평균 거래량
- 소스 모델 합의 수준 (3개 중 몇 개 동의)
- 국면 전문가 확신도
- 최근 N회 매매 승률 (전략의 최근 성적)

### 5.3 Meta Model 의사결정

```
Meta Model 출력 확률(p)에 따른 행동:

p < 0.4  --> 신호 무시 (진입하지 않음)
0.4 <= p < 0.6  --> 최소 비중 진입 (자본의 1%)
0.6 <= p < 0.8  --> 기본 비중 진입 (자본의 2%)
p >= 0.8  --> 확대 비중 진입 (자본의 3%)
```

### 5.4 리스크 관리 최종

- **포지션 사이징**: Meta Model 확률 기반 (위 표)
- **스탑로스**: ATR(14) x 1.5 (Phase 1과 동일)
- **상관관계 필터**: 멀티 종목 동시 포지션 간 상관관계 > 0.7이면 총 익스포저 제한
- **드로다운 보호**: 최대 드로다운 10% 초과 시 REDUCING 상태 (포지션 축소만 허용)

### 5.5 성공 기준

- Meta Model 도입 후 Precision 10% 이상 개선
- 전체 시스템 Sharpe ratio > 1.0 (백테스트 기준)
- "확대 비중" 진입 시 승률이 "최소 비중" 진입 대비 유의미하게 높음

---

## Phase별 요약

| Phase | 추가 요소 | 아키텍처 계층 | 핵심 학습 |
|-------|-----------|---------------|-----------|
| 1 | BTCUSDT + 1h + CatBoost 1개 | - | 파이프라인 기초 |
| 2 | 멀티 종목 + 4h 필터 | 2계층 (타임프레임 필터) | 노이즈 제거 |
| 3 | 오더북/온체인 + 소스별 모델 3개 | 1계층 (데이터 앙상블) | 다양한 관점 |
| 4 | 국면 분류 + 전문가 3개 | 3계층 (국면 전문가) | 적응적 전략 |
| 5 | 메타 레이블링 | 4계층 (메타 의사결정) | 확신도/사이징 |

## 범위 밖

- 실거래 (라이브 트레이딩) 배포
- 웹 UI / 대시보드
- 멀티 거래소 동시 거래
- 고빈도 매매 (HFT)
- 딥러닝 모델 (LSTM, Transformer)
- 강화학습 (RL)

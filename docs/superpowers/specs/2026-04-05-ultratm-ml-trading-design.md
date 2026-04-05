# ultraTM: ML 앙상블 암호화폐 선물 매매 시스템 설계

## 개요

NautilusTrader 기반의 ML 앙상블 매매 시스템으로, 바이낸스 USDT-M 선물 시장에서 멀티 종목/멀티 타임프레임 전략을 백테스트한다. CatBoost(GPU)로 시장 상태를 예측하고, RL(PPO)로 매매 의사결정을 수행하며, 앙상블 레이어에서 두 모델의 신호를 결합한다.

1차 목표는 백테스트를 통한 전략 검증과 학습이며, 실거래 배포는 범위 밖이다.

## 아키텍처

```
ultraTM/
├── data/                   # 데이터 수집 & 저장
│   ├── collectors/         # Binance API 수집기
│   │   ├── ohlcv.py        # OHLCV 캔들 수집
│   │   ├── orderbook.py    # 오더북 스냅샷 수집
│   │   └── onchain.py      # 펀딩비, OI, 롱숏비율 수집
│   ├── storage/            # Parquet 파일 관리
│   │   └── parquet.py      # 저장/로딩 유틸
│   └── providers/          # NautilusTrader 데이터 공급
│       └── nautilus.py     # Bar/OrderBook 객체 변환
├── features/               # 피처 엔지니어링
│   ├── technical.py        # 기술지표 (MA, RSI, MACD, BB, ATR, OBV)
│   ├── orderbook.py        # 오더북 피처 (스프레드, 불균형, 깊이)
│   ├── onchain.py          # 온체인 피처 (펀딩비 변화율, OI 다이버전스)
│   └── pipeline.py         # 피처 파이프라인 오케스트레이션
├── models/                 # ML 모델
│   ├── catboost_model.py   # CatBoost GPU 학습/예측
│   ├── rl_agent.py         # PPO RL 에이전트 (Stable-Baselines3)
│   └── ensemble.py         # 앙상블 레이어 (가중 투표/스태킹)
├── strategy/               # NautilusTrader 전략
│   └── ml_strategy.py      # ML 신호 기반 주문 실행
├── backtest/               # 백테스트
│   ├── runner.py           # 백테스트 실행기
│   └── analysis.py         # 성과 분석 (Sharpe, MDD, Win Rate)
├── config/                 # 설정
│   └── settings.py         # 종목, 타임프레임, 모델 파라미터
├── notebooks/              # 탐색적 분석
└── tests/                  # 테스트
```

### 데이터 흐름

```
Binance API --> Collectors --> Parquet 저장
                                   |
                                   v
                           Feature Pipeline
                            (기술지표, 오더북, 온체인)
                                   |
                    +--------------+---------------+
                    |                              |
                    v                              v
             +--------------+              +--------------+
             |  CatBoost    |              |  RL (PPO)    |
             |  (GPU)       |              |              |
             |  예측: 방향   |              |  의사결정:    |
             |              |              |  진입/청산    |
             +------+-------+              +------+-------+
                    |                              |
                    +--------------+---------------+
                                   |
                                   v
                         +----------------+
                         | 앙상블 레이어    |
                         | (가중 투표)      |
                         +-------+--------+
                                 |
                                 v
                         +----------------+
                         | MLStrategy     |
                         | 주문 실행       |
                         +-------+--------+
                                 |
                                 v
                         +----------------+
                         | NautilusTrader |
                         | 백테스트 엔진   |
                         +----------------+
```

## 대상 시장

- **거래소**: Binance Futures (USDT-M Perpetual)
- **종목**: BTCUSDT, ETHUSDT, SOLUSDT
- **타임프레임**: 5m, 15m, 1h, 4h
- **학습 데이터**: 최근 6~12개월

## Phase 1: CatBoost 기본 백테스트

Phase 1의 목표는 데이터 파이프라인과 CatBoost 단일 모델로 동작하는 백테스트 시스템을 완성하는 것이다.

### 데이터 파이프라인

1. Binance REST API로 OHLCV 캔들 데이터 다운로드
2. Parquet 포맷으로 종목/타임프레임별 저장
3. NautilusTrader `Bar` 객체로 변환하여 백테스트 엔진에 공급

### 피처 엔지니어링

| 카테고리 | 피처 |
|----------|------|
| 추세 | SMA(20, 50, 200), EMA(12, 26), MACD(12, 26, 9) |
| 모멘텀 | RSI(14), Stochastic(14, 3), ROC(10) |
| 변동성 | Bollinger Bands(20, 2), ATR(14) |
| 거래량 | OBV, VWAP, Volume SMA ratio(20) |
| 멀티 타임프레임 | 상위 타임프레임 지표를 하위에 병합 (예: 4h RSI를 15m 행에 조인) |

### CatBoost 모델

- **입력**: 위 피처 테이블
- **타겟**: 향후 N봉 수익률 방향 (1=롱, -1=숏, 중립 구간 제외)
- **학습 방식**: Walk-forward validation
  - 학습 윈도우: 3개월 롤링
  - 검증 윈도우: 1개월
  - 슬라이딩하면서 반복
- **GPU 학습**: `task_type='GPU'`
- **평가 지표**: Accuracy, Precision, F1, 실제 백테스트 수익

### NautilusTrader Strategy (Phase 1)

```python
class MLStrategy(Strategy):
    def on_bar(self, bar: Bar):
        features = self.feature_pipeline.compute(bar)
        signal = self.catboost_model.predict(features)

        if signal == 1 and not self.has_long_position():
            self.enter_long(bar.instrument_id)
        elif signal == -1 and not self.has_short_position():
            self.enter_short(bar.instrument_id)
        elif signal == 0 and self.has_position():
            self.close_position()
```

### 리스크 관리 (Phase 1)

- **포지션 사이징**: 계좌 자본의 2% 고정
- **스탑로스**: ATR(14) x 1.5 동적 스탑
- **최대 동시 포지션**: 종목당 1개

## Phase 2: 오더북 피처 확장

Phase 1의 CatBoost 모델에 오더북 기반 피처를 추가하여 예측력을 높인다.

### 추가 데이터

- Binance WebSocket으로 오더북 스냅샷 수집 (5~10 레벨)
- 시간 단위 샘플링으로 데이터 양 관리
- Parquet에 별도 테이블로 저장

### 오더북 피처

| 피처 | 설명 |
|------|------|
| Bid-Ask 스프레드 | 유동성 지표 |
| 오더북 불균형 (OBI) | `(bid_vol - ask_vol) / (bid_vol + ask_vol)` |
| 깊이 가중 가격 | 호가별 물량 가중 평균 |
| 대형 주문 벽 감지 | 임계치 초과 물량 존재 여부 |

### 변경 사항

- `features/orderbook.py`에 오더북 피처 계산 추가
- `features/pipeline.py`에서 OHLCV 피처와 오더북 피처를 시간 기준으로 조인
- CatBoost 모델은 확장된 피처 테이블로 재학습

## Phase 3: 온체인 데이터 + RL + 앙상블

### 추가 데이터 (온체인/마켓 메트릭)

| 데이터 | 소스 | 수집 주기 |
|--------|------|-----------|
| 펀딩비 (Funding Rate) | Binance REST API | 8시간 (발표 시점) |
| 미결제약정 (Open Interest) | Binance REST API | 5분 |
| 롱숏 비율 | Binance REST API | 5분 |
| 청산 데이터 | Binance WebSocket | 실시간 |

### 온체인 피처

| 피처 | 설명 |
|------|------|
| 펀딩비 변화율 | 급변 시 방향성 신호 |
| OI-가격 다이버전스 | OI 증가 + 가격 하락 = 잠재적 숏 스퀴즈 |
| 청산 클러스터 | 특정 가격대 청산 집중 → 지지/저항 |
| 롱숏 비율 극단치 | 크라우드 역행 신호 |

### RL 에이전트

- **알고리즘**: PPO (Proximal Policy Optimization)
- **프레임워크**: Stable-Baselines3 (PyTorch)
- **환경**: NautilusTrader 백테스트 엔진을 Gymnasium 환경으로 래핑
- **상태 공간**: 전체 피처 (기술지표 + 오더북 + 온체인) + 현재 포지션 상태
- **행동 공간**: Discrete(4) — 롱 진입, 숏 진입, 청산, 홀드
- **보상 함수**: 리스크 조정 수익률 (Sharpe ratio 기반)

### 앙상블 레이어

```
CatBoost 출력 (방향 예측 확률) ──┐
                                 ├──→ 앙상블 (가중 투표 → 스태킹)
RL 출력 (행동 선택) ─────────────┘
```

- **초기**: 가중 투표 — CatBoost 확률이 임계치 이상이고 RL도 동일 방향일 때 진입
- **후기**: 스태킹 — 두 모델 출력을 Logistic Regression 메타 모델이 최종 판단
- **가중치 조정**: 백테스트 Sharpe ratio 기반으로 동적 업데이트

### 리스크 관리 고도화

- **Kelly Criterion**: 기대수익/손실 비율 기반 최적 포지션 사이즈
- **상관관계 필터**: 멀티 종목 포지션 간 상관관계가 높으면 총 익스포저 제한
- **드로다운 보호**: 최대 드로다운 10% 초과 시 전략 일시 중단 (REDUCING 상태)

## 기술 스택

| 구분 | 기술 |
|------|------|
| 트레이딩 엔진 | NautilusTrader (Binance Futures 어댑터) |
| 예측 모델 | CatBoost (GPU) |
| RL | Stable-Baselines3 (PPO, PyTorch) |
| 데이터 저장 | Parquet (pandas/polars) |
| 피처 엔지니어링 | pandas, ta-lib 또는 pandas-ta |
| 시각화 | matplotlib, plotly |
| 설정 관리 | Pydantic Settings |
| 테스트 | pytest |
| Python | 3.12+ |

## 성공 기준

- Phase 1: CatBoost 단일 모델 백테스트가 Buy & Hold 대비 양의 초과수익
- Phase 2: 오더북 피처 추가 후 예측 정확도 또는 Sharpe ratio 개선
- Phase 3: 앙상블이 단일 모델 대비 드로다운 감소 또는 Sharpe ratio 개선

## 범위 밖

- 실거래 (라이브 트레이딩) 배포
- 웹 UI / 대시보드
- 멀티 거래소 동시 거래
- 고빈도 매매 (HFT)

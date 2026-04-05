# AFML 기반 전략 검증 프레임워크

## 개요

Marcos Lopez de Prado의 Advances in Financial Machine Learning(AFML) 방법론을 기반으로
한 독립 전략 검증 프레임워크. 전략과 모델을 입력하면 통계적 유의성, 과적합 여부,
피처 품질을 종합 판정한다.

기존 `models/`, `features/`, `labeling/` 코드를 수정하지 않는 독립 패키지로 구현한다.
`validation/` 패키지에 순수 검증 도구만 포함하며, 기존 코드와의 연결은 이후 단계에서 한다.

## 아키텍처

```
validation/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── purged_kfold.py          # Purged K-Fold CV + Embargo
│   ├── cpcv.py                  # Combinatorial Purged CV
│   ├── sample_weights.py        # 샘플 유니크니스 + 가중치
│   └── sequential_bootstrap.py  # Sequential Bootstrap
├── statistics/
│   ├── __init__.py
│   ├── deflated_sharpe.py       # Deflated Sharpe Ratio (PSR + DSR)
│   ├── sharpe_utils.py          # PSR, 연환산, Sharpe 표준오차
│   └── backtest_stats.py        # 정밀 성과 통계
├── features/
│   ├── __init__.py
│   ├── importance.py            # MDA, MDI, SFI 피처 중요도
│   ├── fracdiff.py              # 분수 차분 (Fractional Differentiation)
│   └── multicollinearity.py     # 다중공선성 제거 (상관/VIF)
├── validator.py                 # StrategyValidator (편의 래퍼)
└── report.py                    # ValidationReport (결과 + 판정)
```

### 데이터 흐름

```
OHLCV + Features + Labels + Model
         |
         v
  StrategyValidator.run_full_validation()
         |
         +---> [1] sample_weights (유니크니스, 동시성)
         +---> [2] sequential_bootstrap (유효 독립 샘플 수)
         +---> [3] fracdiff (피처별 최적 d값)
         +---> [4] multicollinearity (정제된 피처 목록)
         +---> [5] purged_kfold (5-fold, embargo 1%)
         +---> [6] cpcv (6그룹, 2테스트, 15경로)
         +---> [7] importance (MDI, MDA, SFI)
         +---> [8] deflated_sharpe (다중 테스트 보정)
         |
         v
  ValidationReport
    - 판정: PASS / FAIL / MARGINAL
    - 각 검증 모듈별 상세 결과
```

### 인터페이스

```python
from validation import StrategyValidator

validator = StrategyValidator(
    ohlcv=ohlcv_df,                 # DataFrame (timestamp index, OHLCV)
    features=features_df,           # DataFrame (피처 컬럼들)
    labels=labels_series,           # Series (+1/-1, TBM 라벨)
    model=model,                    # .train(X, y), .predict(X), .predict_proba(X)
    tbm_config={                    # TBM 파라미터 (샘플 가중치 계산용)
        "pt_multiplier": 1.0,
        "sl_multiplier": 1.0,
        "max_holding_bars": 4,
    },
    n_trials=1,                     # DSR용: 지금까지 시도한 전략 수
)

report = validator.run_full_validation()
report.verdict                      # "PASS" / "FAIL" / "MARGINAL"
report.print_full()                 # 전체 리포트 출력
```

모델 인터페이스 요구사항: `train(X: DataFrame, y: Series)`, `predict(X: DataFrame) -> ndarray`,
`predict_proba(X: DataFrame) -> ndarray` 메서드가 있어야 한다. 기존 `models.catboost_model.TradingModel`이 이 인터페이스를 이미 충족한다.

## 모듈 상세

### 1. 샘플 가중치 (core/sample_weights.py)

TBM 라벨은 시간 구간이 겹친다. 이 겹침을 정량화하고 샘플 가중치를 부여한다.

**동시성 (Concurrency)**:
- 각 시점 t에서 활성화된 라벨 수를 계산
- 라벨 i가 구간 [t_start_i, t_end_i]를 사용한다면, t_start_i <= t <= t_end_i인 라벨 수

**유니크니스 (Uniqueness)**:
- 라벨 i의 유니크니스 = 1 / (라벨 i 구간 내 평균 동시성)
- 겹침이 많을수록 유니크니스 낮음

**샘플 가중치**:
- weight_i = uniqueness_i * |return_i|
- 유니크하고 수익률이 큰 이벤트일수록 높은 가중치

입력:
- `close_prices`: Series
- `tbm_timestamps`: DataFrame (각 라벨의 시작/끝 시점)

출력:
- `concurrency`: Series (시점별 동시 라벨 수)
- `uniqueness`: Series (라벨별 유니크니스)
- `sample_weights`: Series (최종 가중치)

### 2. Sequential Bootstrap (core/sequential_bootstrap.py)

겹치는 라벨에서 정보량이 높은 샘플을 우선 추출하는 부트스트랩.

알고리즘:
1. 빈 선택 집합 S에서 시작
2. 각 후보 샘플의 "S와의 평균 유니크니스"를 계산
3. 유니크니스에 비례하는 확률로 샘플 하나 선택
4. S에 추가, 반복
5. 원하는 샘플 수만큼 반복

입력:
- `indicator_matrix`: DataFrame (행=시점, 열=라벨, 값=0/1 활성 여부)
- `n_samples`: int (추출할 샘플 수)

출력:
- `selected_indices`: list[int]
- `effective_n`: int (유효 독립 샘플 수 추정)

### 3. Purged K-Fold CV (core/purged_kfold.py)

시간적 정보 누수를 방지하는 교차검증.

알고리즘:
1. 데이터를 n_splits개 연속 구간으로 분할
2. 각 fold에서 하나의 구간을 검증 세트로 선택
3. **Purge**: 학습 세트에서 검증 세트의 TBM 라벨 시간 구간과 겹치는 샘플 제거
4. **Embargo**: 검증 세트 직후 embargo_pct 비율의 학습 샘플 추가 제거
5. 모델 학습 -> 검증 세트 예측 -> Sharpe/Accuracy 기록

입력:
- `X`: DataFrame (피처)
- `y`: Series (라벨)
- `model`: 학습/예측 인터페이스를 가진 모델
- `sample_weight`: Series (가중치)
- `tbm_timestamps`: DataFrame (라벨별 시작/끝 시점)
- `n_splits`: int = 5
- `embargo_pct`: float = 0.01

출력:
- `fold_sharpes`: list[float]
- `fold_accuracies`: list[float]
- `mean_sharpe`: float
- `std_sharpe`: float

### 4. Combinatorial Purged CV (core/cpcv.py)

모든 가능한 학습/검증 조합을 테스트하여 백테스트 경로의 분포를 얻는다.

알고리즘:
1. 데이터를 n_groups개 구간으로 분할
2. C(n_groups, k_test_groups) 가지 조합 생성
3. 각 조합에서 k개 구간 = 검증, 나머지 = 학습 (purge + embargo 적용)
4. 검증 구간의 예측을 시간순으로 이어 붙여 백테스트 경로 합성
5. 각 경로의 Sharpe 계산

입력:
- `X`: DataFrame, `y`: Series, `model`, `sample_weight`, `tbm_timestamps`
- `n_groups`: int = 6
- `k_test_groups`: int = 2
- `embargo_pct`: float = 0.01

출력:
- `path_sharpes`: list[float] (각 경로의 Sharpe)
- `mean_sharpe`: float
- `std_sharpe`: float
- `median_sharpe`: float
- `pct_negative`: float (Sharpe < 0인 경로 비율)

### 5. Deflated Sharpe Ratio (statistics/deflated_sharpe.py)

#### PSR (Probabilistic Sharpe Ratio)

관측된 Sharpe가 벤치마크보다 높을 확률.

```
PSR = Phi((SR_observed - SR_benchmark) / SE(SR))

SE(SR) = sqrt((1 - skew*SR + (kurtosis-1)/4 * SR^2) / (T-1))
```

- SR_observed: 관측된 Sharpe
- SR_benchmark: 기준 Sharpe (기본값 0)
- T: 관측 수
- skew, kurtosis: 수익률의 왜도/첨도
- Phi: 표준정규분포 CDF

#### DSR (Deflated Sharpe Ratio)

N개 전략을 시도한 뒤 관측된 최고 Sharpe의 유의성.

```
Expected Max SR = sqrt(V) * ((1-gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e)))

DSR = PSR(SR_observed, benchmark=Expected_Max_SR)
```

- V: 수익률 분산
- N: 시도한 전략 수 (n_trials)
- gamma: Euler-Mascheroni 상수 (0.5772...)

입력:
- `returns`: Series (수익률)
- `sharpe_observed`: float
- `n_trials`: int
- `sharpe_benchmark`: float = 0.0

출력:
- `psr_pvalue`: float
- `dsr_pvalue`: float
- `expected_max_sharpe`: float
- `sharpe_std_error`: float

### 6. Sharpe 유틸 (statistics/sharpe_utils.py)

- `annualized_sharpe(returns, periods_per_year)`: 연환산 Sharpe
- `sharpe_confidence_interval(sharpe, n, confidence=0.95)`: 신뢰구간
- `probabilistic_sharpe(observed, benchmark, n, skew, kurtosis)`: PSR 계산

### 7. 정밀 성과 통계 (statistics/backtest_stats.py)

기존 `backtest/analysis.py`를 대체하지 않고, 기관급 정밀 통계를 추가 제공.

출력 항목:
- **수익**: 총 수익률, 연환산 수익률, 월별 수익률 분해
- **리스크**: Sharpe + 표준오차 + 95% 신뢰구간, Sortino, Calmar
- **드로다운**: Max Drawdown, 회복 시간, 드로다운 분포
- **매매**: Win Rate, Profit Factor, Tail Ratio, 연속 승/패
- **분포**: 왜도, 첨도, 일별/주별 수익률 분포

### 8. 피처 중요도 (features/importance.py)

#### MDI (Mean Decrease Impurity)
- CatBoost의 `get_feature_importance(type='LossFunctionChange')` 활용
- in-sample 기준

#### MDA (Mean Decrease Accuracy)
- Purged K-Fold CV의 각 fold에서:
  1. 원본 검증 정확도 측정
  2. 피처 하나를 셔플
  3. 셔플 후 정확도 측정
  4. 정확도 하락량 = 해당 피처의 중요도
- fold별 평균 + 표준편차 반환
- out-of-sample 기준 (과적합 탐지에 핵심)

#### SFI (Single Feature Importance)
- 피처 하나만으로 모델 학습 후 Purged CV Sharpe 측정
- 모든 피처에 대해 반복
- 개별 피처의 순수 예측력

#### 과적합 피처 탐지
- MDI 상위 30%인데 MDA 하위 50% -> 과적합 의심
- MDA 표준편차 > MDA 평균 -> 불안정
- SFI Sharpe < 0 -> 단독 예측력 없음
- 위 조건 중 2개 이상 해당 -> 과적합 피처로 분류

입력:
- `X`: DataFrame, `y`: Series, `model`, `sample_weight`, `tbm_timestamps`
- `n_splits`: int = 5 (Purged CV 사용)

출력:
- `mdi_ranking`: DataFrame (피처명, 중요도)
- `mda_ranking`: DataFrame (피처명, 평균 하락, 표준편차)
- `sfi_ranking`: DataFrame (피처명, Sharpe)
- `overfit_features`: list[str]
- `overfit_ratio`: float (과적합 피처 비율)

### 9. 분수 차분 (features/fracdiff.py)

#### Fixed-Width Window Fractional Differentiation

가중치 계산:
```
w_k = -w_{k-1} * (d - k + 1) / k

x_t^d = sum(w_k * x_{t-k}) for k=0..window
```

- `window`: 가중치가 threshold(1e-5) 미만이 되면 잘라냄

#### 최적 d 탐색
1. d를 0.0에서 1.0까지 0.05 간격으로 탐색
2. 각 d로 차분한 시계열에 ADF 검정 수행
3. ADF p-value < 0.05가 되는 최소 d 선택
4. 피처별 개별 d값 반환

입력:
- `series`: Series (원본 시계열)
- `d_range`: (0.0, 1.0, 0.05) 탐색 범위
- `threshold`: float = 1e-5 (가중치 절삭)
- `adf_pvalue`: float = 0.05 (정상성 기준)

출력:
- `optimal_d`: float
- `adf_result`: dict (검정 통계량, p-value)
- `diffed_series`: Series (최적 d로 차분된 시계열)

### 10. 다중공선성 제거 (features/multicollinearity.py)

#### 상관관계 클러스터링
1. 피처 간 절대 상관관계 행렬 계산
2. threshold(0.7) 이상인 피처 쌍을 그룹화
3. 각 그룹에서 MDA 가장 높은 피처만 유지

#### VIF (Variance Inflation Factor)
1. 각 피처를 나머지 피처로 회귀
2. VIF = 1 / (1 - R^2)
3. VIF > 10인 피처를 순차 제거 (가장 높은 것부터)

입력:
- `X`: DataFrame (피처)
- `mda_ranking`: DataFrame (피처 중요도, 선택 기준)
- `corr_threshold`: float = 0.7
- `vif_threshold`: float = 10.0

출력:
- `selected_features`: list[str]
- `removed_features`: list[str] (제거 사유 포함)
- `correlation_clusters`: list[list[str]]

### 11. StrategyValidator (validator.py)

편의 래퍼. 위 모듈들을 순서대로 실행하고 ValidationReport를 생성한다.

실행 순서:
1. `sample_weights.compute()` -> 유니크니스, 가중치
2. `sequential_bootstrap.run()` -> 유효 독립 샘플 수
3. `fracdiff.find_optimal_d()` -> 피처별 최적 d값 (리포트에 포함, 피처 변환은 안 함)
4. `multicollinearity.remove()` -> 정제된 피처 목록 (리포트에 포함, 실제 제거는 안 함)
5. `purged_kfold.cross_validate()` -> fold별 Sharpe
6. `cpcv.cross_validate()` -> 경로별 Sharpe 분포
7. `importance.compute_all()` -> MDI/MDA/SFI + 과적합 피처
8. `deflated_sharpe.compute()` -> DSR p-value
9. `ValidationReport` 생성

StrategyValidator는 검증만 한다. 피처를 실제로 변환하거나 제거하지 않는다.
"이 피처는 과적합이다", "이 d값으로 차분하면 정상성이 확보된다"를 리포트할 뿐이다.
실제 적용은 사용자가 리포트를 보고 판단한다.

### 12. ValidationReport (report.py)

```python
report.verdict              # "PASS" / "FAIL" / "MARGINAL"
report.summary()            # 한 줄 요약

report.purged_cv            # PurgedCVResult
report.cpcv                 # CPCVResult
report.deflated_sharpe      # DeflatedSharpeResult
report.feature_importance   # FeatureImportanceResult
report.fracdiff             # FracDiffResult
report.sample_info          # SampleInfoResult
report.statistics           # BacktestStatsResult
report.multicollinearity    # MulticollinearityResult

report.print_full()         # 전체 리포트 출력
```

판정 로직:
```
FAIL 조건 (하나라도 해당):
  - dsr_pvalue > 0.10
  - purged_cv_mean_sharpe < 0
  - cpcv에서 Sharpe < 0인 경로가 50% 이상

PASS 조건 (모두 충족):
  - dsr_pvalue < 0.05
  - purged_cv_mean_sharpe > 0.5
  - overfit_ratio < 0.30
  - cpcv에서 Sharpe < 0인 경로가 20% 미만

MARGINAL:
  - FAIL도 PASS도 아닌 경우
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 통계 검정 | scipy.stats (norm.cdf, adfuller 등) |
| 행렬 연산 | numpy |
| 데이터 처리 | pandas |
| VIF 계산 | statsmodels (variance_inflation_factor) |
| ADF 검정 | statsmodels (adfuller) |
| 테스트 | pytest |

외부 AFML 라이브러리는 사용하지 않는다. AFML 알고리즘을 직접 구현한다.
이유: 외부 구현체의 정확성을 보장할 수 없고, 커스터마이징이 필요하며,
의존성을 최소화한다.

## 성공 기준

1. 기존 Phase 1 전략(CatBoost + 기술지표)을 넣으면 "FAIL" 판정이 나온다
2. 판정 사유가 구체적이다: "DSR p-value 0.43 > 0.05", "과적합 피처 8/16 = 50%"
3. 각 검증 모듈이 독립적으로 호출 가능하다
4. Purged CV와 CPCV에서 정보 누수가 발생하지 않는다 (테스트로 검증)
5. 합성 데이터(알려진 신호가 있는)에서 PASS 판정이 나온다

## 범위 밖

- 기존 코드(`models/`, `features/`, `labeling/`) 수정
- 피처 자동 변환/제거 (리포트만 제공)
- 실시간 모니터링 (배치 검증만)
- GPU 가속 (numpy/scipy CPU로 충분)

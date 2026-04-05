# Phase 1: TBM + 단일 모델 백테스트 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BTCUSDT 1h 데이터로 TBM 라벨링 → CatBoost 학습 → NautilusTrader 백테스트가 동작하는 최소 파이프라인을 만든다.

**Architecture:** Binance REST API에서 OHLCV를 수집해 Parquet로 저장하고, TBM으로 라벨을 생성한 뒤 기술지표 피처와 함께 CatBoost(GPU)를 학습한다. 학습된 모델은 NautilusTrader의 Strategy 클래스 안에서 on_bar 이벤트마다 예측을 수행하고, TBM 장벽 기반 TP/SL/시간청산으로 주문을 실행한다.

**Tech Stack:** Python 3.12+, NautilusTrader, CatBoost, pandas, pandas-ta, Pydantic, pytest

---

## File Structure

```
ultraTM/
├── pyproject.toml              # 프로젝트 메타데이터, 의존성
├── config/
│   └── settings.py             # Pydantic Settings (종목, TBM 파라미터, 모델 파라미터)
├── data/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   └── ohlcv.py            # Binance OHLCV REST 수집기
│   └── storage/
│       ├── __init__.py
│       └── parquet.py          # Parquet 저장/로딩 유틸
├── labeling/
│   ├── __init__.py
│   └── tbm.py                  # Triple Barrier Method 라벨러
├── features/
│   ├── __init__.py
│   ├── technical.py            # 기술지표 계산
│   └── pipeline.py             # 피처 + 라벨 조합 파이프라인
├── models/
│   ├── __init__.py
│   └── catboost_model.py       # CatBoost 학습/예측/Walk-forward
├── strategy/
│   ├── __init__.py
│   └── ml_strategy.py          # NautilusTrader Strategy
├── backtest/
│   ├── __init__.py
│   ├── runner.py               # 백테스트 실행 스크립트
│   └── analysis.py             # 성과 분석 리포트
└── tests/
    ├── __init__.py
    ├── test_settings.py
    ├── test_ohlcv_collector.py
    ├── test_parquet_storage.py
    ├── test_tbm.py
    ├── test_technical.py
    ├── test_pipeline.py
    ├── test_catboost_model.py
    ├── test_ml_strategy.py
    └── test_backtest_runner.py
```

---

### Task 1: 프로젝트 초기화 및 의존성 설정

**Files:**
- Create: `pyproject.toml`
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `tests/__init__.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: pyproject.toml 작성**

```toml
[project]
name = "ultratm"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "nautilus_trader>=1.210.0",
    "catboost>=1.2",
    "pandas>=2.2",
    "pandas-ta>=0.3.14b1",
    "pyarrow>=15.0",
    "pydantic-settings>=2.2",
    "requests>=2.31",
    "plotly>=5.18",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Settings 테스트 작성**

```python
# tests/test_settings.py
from config.settings import Settings


def test_default_settings():
    s = Settings()
    assert s.symbol == "BTCUSDT"
    assert s.timeframe == "1h"
    assert s.tbm_pt_multiplier == 1.0
    assert s.tbm_sl_multiplier == 1.0
    assert s.tbm_max_holding_bars == 4
    assert s.volatility_span == 24
    assert s.position_size_pct == 0.02
    assert s.initial_capital == 100_000.0


def test_settings_override():
    s = Settings(symbol="ETHUSDT", tbm_pt_multiplier=1.5)
    assert s.symbol == "ETHUSDT"
    assert s.tbm_pt_multiplier == 1.5
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd /home/henry/Projects/ultraTM && python -m pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: Settings 구현**

```python
# config/__init__.py
```

```python
# config/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 시장
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    binance_base_url: str = "https://fapi.binance.com"

    # TBM 파라미터
    tbm_pt_multiplier: float = 1.0
    tbm_sl_multiplier: float = 1.0
    tbm_max_holding_bars: int = 4
    volatility_span: int = 24

    # 리스크
    position_size_pct: float = 0.02
    initial_capital: float = 100_000.0

    # CatBoost
    train_window_months: int = 3
    val_window_months: int = 1
    catboost_iterations: int = 500
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05

    # 데이터 경로
    data_dir: str = "data/storage"

    model_config = {"env_prefix": "ULTRATM_"}
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /home/henry/Projects/ultraTM && python -m pytest tests/test_settings.py -v`
Expected: 2 passed

- [ ] **Step 6: 의존성 설치**

Run: `cd /home/henry/Projects/ultraTM && pip install -e ".[dev]"`

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml config/ tests/__init__.py tests/test_settings.py
git commit -m "feat: project init with settings and dependencies"
```

---

### Task 2: Binance OHLCV 수집기

**Files:**
- Create: `data/__init__.py`
- Create: `data/collectors/__init__.py`
- Create: `data/collectors/ohlcv.py`
- Create: `tests/test_ohlcv_collector.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_ohlcv_collector.py
import pandas as pd
from unittest.mock import patch, MagicMock
from data.collectors.ohlcv import OHLCVCollector


def _mock_klines_response():
    """Binance klines API 응답 형식 모킹"""
    return [
        [
            1609459200000,   # open_time
            "29000.0",       # open
            "29500.0",       # high
            "28800.0",       # low
            "29300.0",       # close
            "1000.0",        # volume
            1609462799999,   # close_time
            "29000000.0",    # quote_volume
            500,             # trades
            "600.0",         # taker_buy_base
            "17400000.0",    # taker_buy_quote
            "0",             # ignore
        ],
        [
            1609462800000,
            "29300.0",
            "29800.0",
            "29100.0",
            "29600.0",
            "1200.0",
            1609466399999,
            "35000000.0",
            600,
            "700.0",
            "20300000.0",
            "0",
        ],
    ]


@patch("data.collectors.ohlcv.requests.get")
def test_fetch_ohlcv(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _mock_klines_response()
    mock_get.return_value = mock_resp

    collector = OHLCVCollector(base_url="https://fapi.binance.com")
    df = collector.fetch("BTCUSDT", "1h", limit=2)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert df["close"].iloc[0] == 29300.0
    assert df["close"].iloc[1] == 29600.0


@patch("data.collectors.ohlcv.requests.get")
def test_fetch_ohlcv_empty(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp

    collector = OHLCVCollector(base_url="https://fapi.binance.com")
    df = collector.fetch("BTCUSDT", "1h", limit=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_ohlcv_collector.py -v`
Expected: FAIL

- [ ] **Step 3: OHLCV 수집기 구현**

```python
# data/__init__.py
```

```python
# data/collectors/__init__.py
```

```python
# data/collectors/ohlcv.py
import pandas as pd
import requests


class OHLCVCollector:
    def __init__(self, base_url: str = "https://fapi.binance.com"):
        self.base_url = base_url

    def fetch(
        self,
        symbol: str,
        interval: str,
        limit: int = 1500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        params: dict = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        resp = requests.get(f"{self.base_url}/fapi/v1/klines", params=params)
        resp.raise_for_status()
        raw = resp.json()

        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = []
        for k in raw:
            rows.append({
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    def fetch_full(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> pd.DataFrame:
        """여러 페이지를 순회하여 전체 기간 데이터 수집"""
        all_dfs = []
        current_start = start_time

        while current_start < end_time:
            df = self.fetch(symbol, interval, limit=1500, start_time=current_start, end_time=end_time)
            if df.empty:
                break
            all_dfs.append(df)
            last_ts = int(df.index[-1].timestamp() * 1000)
            if last_ts == current_start:
                break
            current_start = last_ts + 1

        if not all_dfs:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.concat(all_dfs).loc[~pd.concat(all_dfs).index.duplicated(keep="first")]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_ohlcv_collector.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add data/ tests/test_ohlcv_collector.py
git commit -m "feat: Binance OHLCV collector with pagination"
```

---

### Task 3: Parquet 저장/로딩

**Files:**
- Create: `data/storage/__init__.py`
- Create: `data/storage/parquet.py`
- Create: `tests/test_parquet_storage.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_parquet_storage.py
import pandas as pd
import tempfile
import os
from data.storage.parquet import save_parquet, load_parquet


def _sample_ohlcv():
    return pd.DataFrame(
        {
            "open": [29000.0, 29300.0],
            "high": [29500.0, 29800.0],
            "low": [28800.0, 29100.0],
            "close": [29300.0, 29600.0],
            "volume": [1000.0, 1200.0],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2021-01-01 00:00", tz="UTC"),
             pd.Timestamp("2021-01-01 01:00", tz="UTC")],
            name="timestamp",
        ),
    )


def test_save_and_load_parquet():
    df = _sample_ohlcv()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.parquet")
        save_parquet(df, path)
        assert os.path.exists(path)

        loaded = load_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)


def test_load_nonexistent_returns_none():
    result = load_parquet("/nonexistent/path.parquet")
    assert result is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_parquet_storage.py -v`
Expected: FAIL

- [ ] **Step 3: Parquet 유틸 구현**

```python
# data/storage/__init__.py
```

```python
# data/storage/parquet.py
import os
import pandas as pd


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


def load_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path, engine="pyarrow")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_parquet_storage.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add data/storage/ tests/test_parquet_storage.py
git commit -m "feat: Parquet save/load utilities"
```

---

### Task 4: TBM (Triple Barrier Method) 라벨러

**Files:**
- Create: `labeling/__init__.py`
- Create: `labeling/tbm.py`
- Create: `tests/test_tbm.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_tbm.py
import pandas as pd
import numpy as np
from labeling.tbm import TripleBarrierLabeler


def _make_ohlcv(closes: list[float], base_ts: str = "2021-01-01") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(base_ts, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 50 for c in closes],
            "low": [c - 50 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def test_compute_volatility():
    labeler = TripleBarrierLabeler()
    closes = pd.Series([100.0] * 30 + [110.0])
    vol = labeler.compute_volatility(closes, span=24)
    assert len(vol) == len(closes)
    assert vol.iloc[0] == 0.0 or np.isnan(vol.iloc[0])  # 첫 값은 0 또는 NaN


def test_label_uptrend():
    """가격이 꾸준히 상승 -> 상단 장벽 도달 -> 라벨 +1"""
    closes = [100.0] * 30 + [100, 102, 104, 106, 108, 110, 112, 114]
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler(pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4)
    labels = labeler.label(df)
    # 상승 구간의 라벨 중 +1이 존재해야 함
    assert 1 in labels.values


def test_label_downtrend():
    """가격이 꾸준히 하락 -> 하단 장벽 도달 -> 라벨 -1"""
    closes = [100.0] * 30 + [100, 98, 96, 94, 92, 90, 88, 86]
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler(pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4)
    labels = labeler.label(df)
    assert -1 in labels.values


def test_label_length_matches():
    """라벨 길이가 입력 길이와 동일"""
    closes = [100.0 + i * 0.1 for i in range(50)]
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler()
    labels = labeler.label(df)
    assert len(labels) == len(df)


def test_vertical_barrier():
    """가격이 횡보하면 수직 장벽에 닿음 - max_holding_bars 내에 상/하단 미도달"""
    closes = [100.0] * 50  # 완전 횡보
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler(pt_multiplier=5.0, sl_multiplier=5.0, max_holding_bars=4)
    labels = labeler.label(df)
    # 횡보에서는 수직 장벽 도달 -> 손익에 따라 라벨 결정
    # 완전 횡보이므로 close 변화 없음 -> -1 (손익 <= 0)
    assert -1 in labels.values
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_tbm.py -v`
Expected: FAIL

- [ ] **Step 3: TBM 구현**

```python
# labeling/__init__.py
```

```python
# labeling/tbm.py
import numpy as np
import pandas as pd


class TripleBarrierLabeler:
    def __init__(
        self,
        pt_multiplier: float = 1.0,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 4,
    ):
        self.pt_multiplier = pt_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars

    def compute_volatility(self, close_prices: pd.Series, span: int = 24) -> pd.Series:
        returns = close_prices.pct_change()
        vol = returns.ewm(span=span).std()
        return vol

    def label(self, ohlcv_df: pd.DataFrame) -> pd.Series:
        closes = ohlcv_df["close"]
        highs = ohlcv_df["high"]
        lows = ohlcv_df["low"]
        vol = self.compute_volatility(closes)

        labels = pd.Series(index=ohlcv_df.index, dtype=float)

        for i in range(len(ohlcv_df)):
            sigma_t = vol.iloc[i]
            if np.isnan(sigma_t) or sigma_t == 0:
                labels.iloc[i] = np.nan
                continue

            entry_price = closes.iloc[i]
            upper = entry_price * (1 + self.pt_multiplier * sigma_t)
            lower = entry_price * (1 - self.sl_multiplier * sigma_t)

            end_idx = min(i + self.max_holding_bars, len(ohlcv_df) - 1)
            if i >= len(ohlcv_df) - 1:
                labels.iloc[i] = np.nan
                continue

            hit_label = np.nan
            for j in range(i + 1, end_idx + 1):
                if highs.iloc[j] >= upper:
                    hit_label = 1.0
                    break
                if lows.iloc[j] <= lower:
                    hit_label = -1.0
                    break

            if np.isnan(hit_label):
                # 수직 장벽: 시간 초과 시 현재 손익으로 판정
                exit_price = closes.iloc[end_idx]
                pnl = exit_price - entry_price
                hit_label = 1.0 if pnl > 0 else -1.0

            labels.iloc[i] = hit_label

        return labels
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_tbm.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add labeling/ tests/test_tbm.py
git commit -m "feat: Triple Barrier Method labeler"
```

---

### Task 5: 기술지표 피처 엔지니어링

**Files:**
- Create: `features/__init__.py`
- Create: `features/technical.py`
- Create: `tests/test_technical.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_technical.py
import pandas as pd
import numpy as np
from features.technical import compute_technical_features


def _sample_ohlcv(n: int = 250) -> pd.DataFrame:
    np.random.seed(42)
    close = 30000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame(
        {
            "open": close - np.random.rand(n) * 50,
            "high": close + np.abs(np.random.randn(n)) * 100,
            "low": close - np.abs(np.random.randn(n)) * 100,
            "close": close,
            "volume": np.random.rand(n) * 1000 + 500,
        },
        index=pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_compute_features_columns():
    df = _sample_ohlcv()
    features = compute_technical_features(df)
    expected_cols = [
        "sma_20", "sma_50", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "stoch_k", "stoch_d",
        "bb_upper", "bb_mid", "bb_lower",
        "atr_14", "obv", "vol_sma_ratio",
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"


def test_compute_features_length():
    df = _sample_ohlcv()
    features = compute_technical_features(df)
    assert len(features) == len(df)


def test_compute_features_no_inf():
    df = _sample_ohlcv()
    features = compute_technical_features(df)
    # NaN은 초기 워밍업 기간에 있을 수 있지만 inf는 없어야 함
    numeric = features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values[~np.isnan(numeric.values)]).any()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_technical.py -v`
Expected: FAIL

- [ ] **Step 3: 기술지표 구현**

```python
# features/__init__.py
```

```python
# features/technical.py
import pandas as pd
import pandas_ta as ta


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()

    # 추세
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd.iloc[:, 0]
    df["macd_signal"] = macd.iloc[:, 1]
    df["macd_hist"] = macd.iloc[:, 2]

    # 모멘텀
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k"] = stoch.iloc[:, 0]
    df["stoch_d"] = stoch.iloc[:, 1]

    # 변동성
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["bb_lower"] = bbands.iloc[:, 0]
    df["bb_mid"] = bbands.iloc[:, 1]
    df["bb_upper"] = bbands.iloc[:, 2]

    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # 거래량
    df["obv"] = ta.obv(df["close"], df["volume"])

    vol_sma = ta.sma(df["volume"], length=20)
    df["vol_sma_ratio"] = df["volume"] / vol_sma

    # 원본 OHLCV 컬럼 제거, 피처만 반환
    feature_cols = [
        "sma_20", "sma_50", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "stoch_k", "stoch_d",
        "bb_upper", "bb_mid", "bb_lower",
        "atr_14", "obv", "vol_sma_ratio",
    ]
    return df[feature_cols]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_technical.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add features/ tests/test_technical.py
git commit -m "feat: technical indicator feature engineering"
```

---

### Task 6: 피처 파이프라인 (피처 + TBM 라벨 조합)

**Files:**
- Create: `features/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_pipeline.py
import pandas as pd
import numpy as np
from features.pipeline import FeaturePipeline
from config.settings import Settings


def _sample_ohlcv(n: int = 250) -> pd.DataFrame:
    np.random.seed(42)
    close = 30000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame(
        {
            "open": close - np.random.rand(n) * 50,
            "high": close + np.abs(np.random.randn(n)) * 100,
            "low": close - np.abs(np.random.randn(n)) * 100,
            "close": close,
            "volume": np.random.rand(n) * 1000 + 500,
        },
        index=pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_pipeline_produces_features_and_labels():
    settings = Settings()
    pipeline = FeaturePipeline(settings)
    df = _sample_ohlcv()
    result = pipeline.build(df)

    assert "label" in result.columns
    assert "rsi_14" in result.columns
    assert "macd" in result.columns


def test_pipeline_drops_nan_rows():
    settings = Settings()
    pipeline = FeaturePipeline(settings)
    df = _sample_ohlcv()
    result = pipeline.build(df)

    # NaN 행이 제거되어야 함
    assert not result.isnull().any().any()
    assert len(result) > 0
    assert len(result) < len(df)  # 워밍업 기간만큼 줄어듦


def test_pipeline_label_values():
    settings = Settings()
    pipeline = FeaturePipeline(settings)
    df = _sample_ohlcv()
    result = pipeline.build(df)

    # 라벨은 +1 또는 -1만
    assert set(result["label"].unique()).issubset({1.0, -1.0})
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: 파이프라인 구현**

```python
# features/pipeline.py
import pandas as pd
from config.settings import Settings
from features.technical import compute_technical_features
from labeling.tbm import TripleBarrierLabeler


class FeaturePipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.labeler = TripleBarrierLabeler(
            pt_multiplier=settings.tbm_pt_multiplier,
            sl_multiplier=settings.tbm_sl_multiplier,
            max_holding_bars=settings.tbm_max_holding_bars,
        )

    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        features = compute_technical_features(ohlcv)
        labels = self.labeler.label(ohlcv)
        features["label"] = labels
        features = features.dropna()
        return features
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add features/pipeline.py tests/test_pipeline.py
git commit -m "feat: feature pipeline combining indicators + TBM labels"
```

---

### Task 7: CatBoost 모델 (Walk-Forward 학습/예측)

**Files:**
- Create: `models/__init__.py`
- Create: `models/catboost_model.py`
- Create: `tests/test_catboost_model.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_catboost_model.py
import pandas as pd
import numpy as np
from models.catboost_model import TradingModel


def _sample_features(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "sma_20": np.random.randn(n),
        "sma_50": np.random.randn(n),
        "ema_12": np.random.randn(n),
        "ema_26": np.random.randn(n),
        "macd": np.random.randn(n),
        "macd_signal": np.random.randn(n),
        "macd_hist": np.random.randn(n),
        "rsi_14": np.random.rand(n) * 100,
        "stoch_k": np.random.rand(n) * 100,
        "stoch_d": np.random.rand(n) * 100,
        "bb_upper": np.random.randn(n),
        "bb_mid": np.random.randn(n),
        "bb_lower": np.random.randn(n),
        "atr_14": np.abs(np.random.randn(n)),
        "obv": np.cumsum(np.random.randn(n)),
        "vol_sma_ratio": np.random.rand(n) + 0.5,
        "label": np.random.choice([1.0, -1.0], size=n),
    }
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(data, index=idx)


def test_train_and_predict():
    df = _sample_features()
    train = df.iloc[:400]
    test = df.iloc[400:]

    model = TradingModel(iterations=50, depth=4, learning_rate=0.1)
    model.train(train.drop(columns=["label"]), train["label"])

    preds = model.predict(test.drop(columns=["label"]))
    assert len(preds) == len(test)
    assert set(preds).issubset({1.0, -1.0})


def test_walk_forward():
    df = _sample_features(n=800)

    model = TradingModel(iterations=50, depth=4, learning_rate=0.1)
    results = model.walk_forward(
        df,
        train_window=400,
        val_window=100,
    )

    assert "predictions" in results
    assert "actuals" in results
    assert len(results["predictions"]) == len(results["actuals"])
    assert len(results["predictions"]) > 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_catboost_model.py -v`
Expected: FAIL

- [ ] **Step 3: CatBoost 모델 구현**

```python
# models/__init__.py
```

```python
# models/catboost_model.py
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class TradingModel:
    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        task_type: str = "CPU",
    ):
        self.params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "task_type": task_type,
            "loss_function": "Logloss",
            "verbose": 0,
            "random_seed": 42,
        }
        self.model: CatBoostClassifier | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = CatBoostClassifier(**self.params)
        # CatBoost expects labels in {0, 1} for Logloss
        y_binary = (y == 1.0).astype(int)
        self.model.fit(X, y_binary)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        preds_binary = self.model.predict(X).flatten()
        # 0 -> -1.0, 1 -> 1.0
        return np.where(preds_binary == 1, 1.0, -1.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def walk_forward(
        self,
        data: pd.DataFrame,
        train_window: int,
        val_window: int,
    ) -> dict:
        feature_cols = [c for c in data.columns if c != "label"]
        all_preds = []
        all_actuals = []

        start = 0
        while start + train_window + val_window <= len(data):
            train_end = start + train_window
            val_end = train_end + val_window

            train_slice = data.iloc[start:train_end]
            val_slice = data.iloc[train_end:val_end]

            self.train(train_slice[feature_cols], train_slice["label"])
            preds = self.predict(val_slice[feature_cols])

            all_preds.extend(preds)
            all_actuals.extend(val_slice["label"].values)

            start += val_window

        return {
            "predictions": np.array(all_preds),
            "actuals": np.array(all_actuals),
        }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_catboost_model.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add models/ tests/test_catboost_model.py
git commit -m "feat: CatBoost model with walk-forward validation"
```

---

### Task 8: NautilusTrader MLStrategy

**Files:**
- Create: `strategy/__init__.py`
- Create: `strategy/ml_strategy.py`
- Create: `tests/test_ml_strategy.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_ml_strategy.py
from strategy.ml_strategy import MLStrategyConfig


def test_config_defaults():
    config = MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL",
    )
    assert config.pt_multiplier == 1.0
    assert config.sl_multiplier == 1.0
    assert config.max_holding_bars == 4
    assert config.volatility_span == 24
    assert config.position_size_pct == 0.02


def test_config_override():
    config = MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL",
        pt_multiplier=1.5,
        max_holding_bars=6,
    )
    assert config.pt_multiplier == 1.5
    assert config.max_holding_bars == 6
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_ml_strategy.py -v`
Expected: FAIL

- [ ] **Step 3: MLStrategy 구현**

```python
# strategy/__init__.py
```

```python
# strategy/ml_strategy.py
import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarType, InstrumentId, OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.trading.strategy import Strategy


class MLStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    pt_multiplier: float = 1.0
    sl_multiplier: float = 1.0
    max_holding_bars: int = 4
    volatility_span: int = 24
    position_size_pct: float = 0.02


class MLStrategy(Strategy):
    def __init__(self, config: MLStrategyConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.pt = config.pt_multiplier
        self.sl = config.sl_multiplier
        self.max_holding_bars = config.max_holding_bars
        self.vol_span = config.volatility_span
        self.position_size_pct = config.position_size_pct

        self.close_prices: list[float] = []
        self.bars_held: int = 0
        self.model = None  # set externally before running

    def set_model(self, model, feature_cols: list[str]) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.feature_buffer: dict[str, list[float]] = {c: [] for c in feature_cols}

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        close = float(bar.close)
        self.close_prices.append(close)

        # 포지션 보유 중이면 보유 봉수 카운트
        if self.portfolio.is_net_long(self.instrument_id) or self.portfolio.is_net_short(self.instrument_id):
            self.bars_held += 1
            # 수직 장벽: 최대 보유 시간 초과
            if self.bars_held >= self.max_holding_bars:
                self._close_all()
                return
        else:
            self.bars_held = 0

        # 변동성 계산에 충분한 데이터가 없으면 스킵
        if len(self.close_prices) < self.vol_span + 1:
            return

        if self.model is None:
            return

        # 변동성 계산
        sigma_t = self._compute_volatility()
        if sigma_t == 0 or np.isnan(sigma_t):
            return

        # 이미 포지션이 있으면 새 진입 안함
        if self.portfolio.is_net_long(self.instrument_id) or self.portfolio.is_net_short(self.instrument_id):
            return

        # 모델 예측 (외부에서 피처를 주입하는 간소화 방식)
        # 실제 백테스트에서는 runner가 피처를 미리 계산하여 모델에 전달
        signal = self._get_signal()
        if signal is None:
            return

        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        account = self.portfolio.account(self.instrument_id.venue)
        if account is None:
            return

        # 포지션 사이즈 계산
        equity = float(account.balance_total(instrument.quote_currency))
        notional = equity * self.position_size_pct
        quantity = instrument.make_qty(notional / close)

        if float(quantity) == 0:
            return

        if signal == 1.0:
            tp_price = instrument.make_price(close * (1 + self.pt * sigma_t))
            sl_price = instrument.make_price(close * (1 - self.sl * sigma_t))
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=quantity,
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self.bars_held = 0

        elif signal == -1.0:
            tp_price = instrument.make_price(close * (1 - self.pt * sigma_t))
            sl_price = instrument.make_price(close * (1 + self.sl * sigma_t))
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=quantity,
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self.bars_held = 0

    def _compute_volatility(self) -> float:
        prices = pd.Series(self.close_prices)
        returns = prices.pct_change()
        vol = returns.ewm(span=self.vol_span).std().iloc[-1]
        return float(vol) if not np.isnan(vol) else 0.0

    def _get_signal(self) -> float | None:
        """Override in backtest runner or extend with feature computation"""
        return None

    def _close_all(self) -> None:
        self.close_all_positions(self.instrument_id)
        self.bars_held = 0

    def on_stop(self) -> None:
        self._close_all()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_ml_strategy.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add strategy/ tests/test_ml_strategy.py
git commit -m "feat: NautilusTrader MLStrategy with TBM barriers"
```

---

### Task 9: 백테스트 러너 및 성과 분석

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/runner.py`
- Create: `backtest/analysis.py`
- Create: `tests/test_backtest_runner.py`

- [ ] **Step 1: 성과 분석 테스트 작성**

```python
# tests/test_backtest_runner.py
import numpy as np
from backtest.analysis import compute_metrics


def test_compute_metrics():
    # 간단한 수익 시퀀스
    equity_curve = np.array([100000, 101000, 100500, 102000, 101500, 103000])
    metrics = compute_metrics(equity_curve)

    assert "total_return_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert "win_rate" in metrics
    assert metrics["total_return_pct"] == 3.0  # (103000-100000)/100000 * 100


def test_compute_metrics_no_trades():
    equity_curve = np.array([100000, 100000, 100000])
    metrics = compute_metrics(equity_curve)
    assert metrics["total_return_pct"] == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_backtest_runner.py -v`
Expected: FAIL

- [ ] **Step 3: 성과 분석 구현**

```python
# backtest/__init__.py
```

```python
# backtest/analysis.py
import numpy as np


def compute_metrics(equity_curve: np.ndarray) -> dict:
    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return_pct = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100

    # Sharpe (annualized, assuming hourly data -> 8760 hours/year)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(8760)
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_drawdown_pct = float(np.max(drawdown))

    # Win Rate
    winning = returns[returns > 0]
    total = returns[returns != 0]
    win_rate = (len(winning) / len(total) * 100) if len(total) > 0 else 0.0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return_pct": round(total_return_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": len(total),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_backtest_runner.py -v`
Expected: 2 passed

- [ ] **Step 5: 백테스트 러너 구현**

```python
# backtest/runner.py
from decimal import Decimal

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.model import (
    BarType,
    Money,
    TraderId,
    Venue,
)
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from config.settings import Settings
from strategy.ml_strategy import MLStrategy, MLStrategyConfig


def create_engine(settings: Settings) -> BacktestEngine:
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
    )
    engine = BacktestEngine(config=config)

    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(settings.initial_capital, USDT)],
    )

    return engine


def run_backtest(settings: Settings, ohlcv_df, model, feature_cols):
    """
    End-to-end 백테스트 실행.

    Parameters
    ----------
    settings : Settings
    ohlcv_df : pd.DataFrame - OHLCV 데이터 (timestamp index)
    model : TradingModel - 학습 완료된 CatBoost 모델
    feature_cols : list[str] - 피처 컬럼 목록
    """
    engine = create_engine(settings)

    # Instrument 설정 (Binance BTCUSDT perpetual)
    instrument = TestInstrumentProvider.btcusdt_binance()
    engine.add_instrument(instrument)

    # OHLCV -> NautilusTrader Bar 객체 변환
    bar_type = BarType.from_str(f"{instrument.id}-1-HOUR-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    bars = wrangler.process(ohlcv_df)
    engine.add_data(bars)

    # Strategy 설정
    strategy_config = MLStrategyConfig(
        instrument_id=str(instrument.id),
        bar_type=str(bar_type),
        pt_multiplier=settings.tbm_pt_multiplier,
        sl_multiplier=settings.tbm_sl_multiplier,
        max_holding_bars=settings.tbm_max_holding_bars,
        volatility_span=settings.volatility_span,
        position_size_pct=settings.position_size_pct,
    )
    strategy = MLStrategy(config=strategy_config)
    strategy.set_model(model, feature_cols)
    engine.add_strategy(strategy)

    # 실행
    engine.run()

    return engine
```

- [ ] **Step 6: 커밋**

```bash
git add backtest/ tests/test_backtest_runner.py
git commit -m "feat: backtest runner and performance analysis"
```

---

### Task 10: 전체 파이프라인 통합 실행 스크립트

**Files:**
- Create: `run_phase1.py`

- [ ] **Step 1: 통합 실행 스크립트 작성**

```python
# run_phase1.py
"""
Phase 1 전체 파이프라인 실행:
1. Binance에서 BTCUSDT 1h OHLCV 수집
2. Parquet 저장
3. TBM 라벨링 + 피처 엔지니어링
4. CatBoost Walk-Forward 학습
5. NautilusTrader 백테스트
6. 성과 리포트
"""
import os
import time
import numpy as np
import pandas as pd

from config.settings import Settings
from data.collectors.ohlcv import OHLCVCollector
from data.storage.parquet import save_parquet, load_parquet
from features.pipeline import FeaturePipeline
from models.catboost_model import TradingModel
from backtest.analysis import compute_metrics


def main():
    settings = Settings()
    print(f"=== ultraTM Phase 1: {settings.symbol} {settings.timeframe} ===\n")

    # 1. 데이터 수집
    parquet_path = os.path.join(settings.data_dir, f"{settings.symbol}_{settings.timeframe}.parquet")
    ohlcv = load_parquet(parquet_path)

    if ohlcv is None:
        print("[1/6] Collecting OHLCV data from Binance...")
        collector = OHLCVCollector(base_url=settings.binance_base_url)
        # 최근 6개월
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (180 * 24 * 60 * 60 * 1000)
        ohlcv = collector.fetch_full(settings.symbol, settings.timeframe, start_ms, end_ms)
        save_parquet(ohlcv, parquet_path)
        print(f"    Saved {len(ohlcv)} bars to {parquet_path}")
    else:
        print(f"[1/6] Loaded {len(ohlcv)} bars from {parquet_path}")

    # 2. 피처 + TBM 라벨
    print("[2/6] Building features + TBM labels...")
    pipeline = FeaturePipeline(settings)
    dataset = pipeline.build(ohlcv)
    print(f"    Dataset: {len(dataset)} rows, {len(dataset.columns)} columns")
    print(f"    Label distribution: +1={sum(dataset['label']==1)}, -1={sum(dataset['label']==-1)}")

    # 3. Walk-Forward 학습
    print("[3/6] Walk-forward training...")
    feature_cols = [c for c in dataset.columns if c != "label"]
    # train_window와 val_window를 봉 수로 변환 (1h 기준, 한달 ~720봉)
    train_bars = settings.train_window_months * 30 * 24
    val_bars = settings.val_window_months * 30 * 24

    model = TradingModel(
        iterations=settings.catboost_iterations,
        depth=settings.catboost_depth,
        learning_rate=settings.catboost_learning_rate,
        task_type="GPU",
    )
    wf_results = model.walk_forward(dataset, train_window=train_bars, val_window=val_bars)
    print(f"    Walk-forward predictions: {len(wf_results['predictions'])}")

    # 4. Walk-Forward 예측 정확도
    preds = wf_results["predictions"]
    actuals = wf_results["actuals"]
    accuracy = np.mean(preds == actuals) * 100
    print(f"[4/6] Walk-forward accuracy: {accuracy:.1f}%")

    # 5. 결과 리포트
    print("[5/6] Performance report:")
    print(f"    Accuracy: {accuracy:.1f}%")
    precision_long = np.mean(actuals[preds == 1] == 1) * 100 if sum(preds == 1) > 0 else 0
    precision_short = np.mean(actuals[preds == -1] == -1) * 100 if sum(preds == -1) > 0 else 0
    print(f"    Long precision: {precision_long:.1f}%")
    print(f"    Short precision: {precision_short:.1f}%")

    print("\n[6/6] Phase 1 complete.")
    print("    Note: Full NautilusTrader backtest integration requires")
    print("    signal injection into MLStrategy (Phase 1 extension).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 커밋**

```bash
git add run_phase1.py
git commit -m "feat: Phase 1 end-to-end pipeline runner"
```

---

## Phase 1 완료 후 검증 체크리스트

- [ ] `python -m pytest tests/ -v` — 전체 테스트 통과
- [ ] `python run_phase1.py` — 전체 파이프라인 실행 성공
- [ ] 라벨 분포 확인: +1/-1 비율이 40:60 ~ 60:40 범위 내
- [ ] Walk-forward accuracy > 50% (랜덤 이상)
- [ ] Parquet 파일이 `data/storage/` 에 정상 저장됨

"""Integration test: full validation pipeline with synthetic data.

Tests two scenarios:
1. Random strategy (no signal) -> should get FAIL or MARGINAL
2. Strong-signal strategy -> should get meaningful positive results
"""

import numpy as np
import pandas as pd
import pytest

from validation import StrategyValidator, ValidationReport
from validation.core.sample_weights import compute_sample_weights, build_indicator_matrix
from validation.core.sequential_bootstrap import sequential_bootstrap, estimate_effective_n
from validation.statistics.sharpe_utils import annualized_sharpe, probabilistic_sharpe_ratio
from validation.statistics.deflated_sharpe import compute_deflated_sharpe
from validation.statistics.backtest_stats import compute_backtest_stats
from validation.features.fracdiff import fracdiff, find_optimal_d
from validation.features.multicollinearity import analyze_multicollinearity
from validation.features.importance import detect_overfit_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleModel:
    def __init__(self):
        self._majority = 1
        self._n_features = 1

    def train(self, X, y, sample_weight=None):
        self._n_features = X.shape[1]
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])

    def predict(self, X):
        return np.full(len(X), self._majority)

    def get_feature_importance(self):
        return np.ones(self._n_features)


class SignalModel:
    def __init__(self):
        self._n_features = 1
        self._threshold = 0.0

    def train(self, X, y, sample_weight=None):
        self._n_features = X.shape[1]
        col = "signal" if "signal" in X.columns else X.columns[0]
        best_acc = 0
        for t in np.linspace(X[col].min(), X[col].max(), 20):
            preds = np.where(X[col] > t, 1, -1)
            acc = np.mean(preds == y.values)
            if acc > best_acc:
                best_acc = acc
                self._threshold = t
        self._signal_col = col

    def predict(self, X):
        col = self._signal_col if hasattr(self, "_signal_col") else X.columns[0]
        return np.where(X[col] > self._threshold, 1, -1)

    def get_feature_importance(self):
        imp = np.ones(self._n_features)
        imp[0] = 5.0
        return imp


def _make_random_data(n=300, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    ohlcv = pd.DataFrame({
        "open": 100 + np.cumsum(rng.standard_normal(n) * 0.5),
        "high": 101 + np.cumsum(rng.standard_normal(n) * 0.5),
        "low": 99 + np.cumsum(rng.standard_normal(n) * 0.5),
        "close": 100 + np.cumsum(rng.standard_normal(n) * 0.5),
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)
    features = pd.DataFrame(
        {f"feat_{i}": rng.standard_normal(n) for i in range(n_features)},
        index=idx,
    )
    labels = pd.Series(rng.choice([1, -1], n), index=idx)
    return ohlcv, features, labels


def _make_signal_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    signal = rng.standard_normal(n) * 2
    noise = rng.standard_normal(n) * 0.3
    labels_raw = np.sign(signal + noise).astype(int)
    labels_raw[labels_raw == 0] = 1
    ohlcv = pd.DataFrame({
        "open": 100 + np.cumsum(rng.standard_normal(n) * 0.5),
        "high": 101 + np.cumsum(rng.standard_normal(n) * 0.5),
        "low": 99 + np.cumsum(rng.standard_normal(n) * 0.5),
        "close": 100 + np.cumsum(rng.standard_normal(n) * 0.5),
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)
    features = pd.DataFrame({
        "signal": signal,
        "noise1": rng.standard_normal(n),
        "noise2": rng.standard_normal(n),
    }, index=idx)
    labels = pd.Series(labels_raw, index=idx)
    return ohlcv, features, labels


# ---------------------------------------------------------------------------
# Module-scoped fixtures: run validation ONCE per scenario
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_report():
    ohlcv, features, labels = _make_random_data(n=200, n_features=4)
    model = SimpleModel()
    validator = StrategyValidator(
        ohlcv=ohlcv, features=features, labels=labels, model=model,
        tbm_config={"max_holding_bars": 3}, n_trials=1,
        n_splits=3, n_groups=4, k_test_groups=1, bootstrap_runs=10,
    )
    return validator.run_full_validation()


@pytest.fixture(scope="module")
def signal_report():
    ohlcv, features, labels = _make_signal_data(n=200)
    model = SignalModel()
    validator = StrategyValidator(
        ohlcv=ohlcv, features=features, labels=labels, model=model,
        tbm_config={"max_holding_bars": 3}, n_trials=1,
        n_splits=3, n_groups=4, k_test_groups=1, bootstrap_runs=10,
    )
    return validator.run_full_validation()


# ---------------------------------------------------------------------------
# Individual module tests (fast, no full pipeline)
# ---------------------------------------------------------------------------

class TestIndividualModulesIntegration:
    def test_sample_weights_end_to_end(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="1h", tz="UTC")
        close = pd.Series(100 + np.cumsum(np.random.default_rng(42).standard_normal(50) * 0.5), index=idx)
        tbm = pd.DataFrame({
            "t_start": idx,
            "t_end": pd.DatetimeIndex([idx[min(i + 3, 49)] for i in range(50)]),
        }, index=range(50))
        conc, uniq, sw = compute_sample_weights(close, tbm)
        assert len(conc) == 50
        assert len(uniq) == 50

    def test_sequential_bootstrap_end_to_end(self):
        idx = pd.date_range("2023-01-01", periods=30, freq="1h", tz="UTC")
        tbm = pd.DataFrame({
            "t_start": idx,
            "t_end": pd.DatetimeIndex([idx[min(i + 3, 29)] for i in range(30)]),
        }, index=range(30))
        indicator = build_indicator_matrix(tbm, idx)
        selected = sequential_bootstrap(indicator, n_samples=20, random_state=42)
        assert len(selected) == 20

    def test_fracdiff_end_to_end(self):
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        rw = pd.Series(np.cumsum(np.random.default_rng(42).standard_normal(500)), index=idx)
        d, pval, diffed = find_optimal_d(rw, d_step=0.2)
        assert 0 < d <= 1.0

    def test_multicollinearity_end_to_end(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal(100)
        X = pd.DataFrame({"a": base, "b": base + rng.standard_normal(100) * 0.1, "c": rng.standard_normal(100)})
        result = analyze_multicollinearity(X, corr_threshold=0.7)
        assert "c" in result.selected_features

    def test_deflated_sharpe_end_to_end(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=5)
        assert 0 <= result.psr_pvalue <= 1

    def test_backtest_stats_end_to_end(self):
        returns = pd.Series(np.random.default_rng(42).normal(0.0005, 0.01, 252))
        result = compute_backtest_stats(returns)
        assert np.isfinite(result.sharpe_ratio)


# ---------------------------------------------------------------------------
# Full pipeline tests (use fixtures, run once)
# ---------------------------------------------------------------------------

def test_random_strategy_does_not_pass(random_report):
    assert random_report.verdict in ("FAIL", "MARGINAL")


def test_signal_strategy_positive_sharpe(signal_report):
    assert signal_report.purged_cv is not None
    assert signal_report.purged_cv.mean_sharpe > 0


def test_signal_report_has_all_sections(signal_report):
    assert signal_report.cpcv is not None
    assert signal_report.feature_importance is not None


def test_random_report_print(random_report, capsys):
    random_report.print_full()
    out = capsys.readouterr().out
    assert "VALIDATION REPORT" in out
    assert "FINAL VERDICT" in out


def test_overfit_detection_with_mixed_features():
    mdi = pd.DataFrame({"feature": ["bad", "good", "unstable", "no_power"], "importance": [0.45, 0.30, 0.15, 0.10]})
    mda = pd.DataFrame({"feature": ["bad", "good", "unstable", "no_power"], "mean_decrease": [0.001, 0.05, 0.01, 0.005], "std_decrease": [0.001, 0.01, 0.05, 0.01]})
    sfi = pd.DataFrame({"feature": ["bad", "good", "unstable", "no_power"], "sharpe": [-0.3, 0.5, -0.1, -0.2]})
    overfit = detect_overfit_features(mdi, mda, sfi)
    assert "bad" in overfit


def test_psr_basic_math():
    psr = probabilistic_sharpe_ratio(2.0, 0.0, 252, 0.0, 3.0)
    assert psr > 0.99
    psr_zero = probabilistic_sharpe_ratio(0.0, 0.0, 252, 0.0, 3.0)
    assert abs(psr_zero - 0.5) < 1e-10


def test_annualized_sharpe_basic():
    returns = np.array([0.011, -0.009] * 126)
    sr = annualized_sharpe(returns, 252)
    assert 0.5 < sr < 3.0

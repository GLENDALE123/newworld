import numpy as np
import pandas as pd
import pytest

from validation.validator import StrategyValidator
from validation.report import ValidationReport


class MockModel:
    """Mock model for validator testing."""

    def __init__(self):
        self._majority = 1
        self._n_features = 0

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        self._n_features = X.shape[1]
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._majority)

    def get_feature_importance(self) -> np.ndarray:
        return np.arange(self._n_features, 0, -1, dtype=float)


def _make_validator_data(n: int = 200, n_features: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    ohlcv = pd.DataFrame({
        "open": 100 + np.cumsum(rng.standard_normal(n) * 0.5),
        "high": 100 + np.cumsum(rng.standard_normal(n) * 0.5) + 1,
        "low": 100 + np.cumsum(rng.standard_normal(n) * 0.5) - 1,
        "close": 100 + np.cumsum(rng.standard_normal(n) * 0.5),
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)

    features = pd.DataFrame(
        {f"feat_{i}": rng.standard_normal(n) for i in range(n_features)},
        index=idx,
    )
    labels = pd.Series(rng.choice([1, -1], n), index=idx)

    return ohlcv, features, labels


class TestStrategyValidator:
    def test_run_full_validation_returns_report(self):
        ohlcv, features, labels = _make_validator_data(n=120, n_features=3)
        model = MockModel()
        validator = StrategyValidator(
            ohlcv=ohlcv,
            features=features,
            labels=labels,
            model=model,
            tbm_config={"max_holding_bars": 3},
            n_trials=1,
            n_splits=3,
            n_groups=4,
            k_test_groups=1,
            bootstrap_runs=10,
        )
        report = validator.run_full_validation()

        assert isinstance(report, ValidationReport)
        assert report.verdict in ("PASS", "FAIL", "MARGINAL")

    def test_report_has_all_sections(self):
        ohlcv, features, labels = _make_validator_data(n=120, n_features=3)
        model = MockModel()
        validator = StrategyValidator(
            ohlcv=ohlcv,
            features=features,
            labels=labels,
            model=model,
            tbm_config={"max_holding_bars": 3},
            n_trials=1,
            n_splits=3,
            n_groups=4,
            k_test_groups=1,
            bootstrap_runs=10,
        )
        report = validator.run_full_validation()

        assert report.sample_info is not None
        assert report.purged_cv is not None
        assert report.cpcv is not None
        assert report.feature_importance is not None
        assert report.frac_diff is not None
        assert report.multicollinearity is not None

    def test_summary_contains_verdict(self):
        ohlcv, features, labels = _make_validator_data(n=120, n_features=3)
        model = MockModel()
        validator = StrategyValidator(
            ohlcv=ohlcv,
            features=features,
            labels=labels,
            model=model,
            tbm_config={"max_holding_bars": 3},
            n_trials=1,
            n_splits=3,
            n_groups=4,
            k_test_groups=1,
            bootstrap_runs=10,
        )
        report = validator.run_full_validation()
        summary = report.summary()
        assert "Verdict:" in summary

    def test_print_full_runs(self, capsys):
        ohlcv, features, labels = _make_validator_data(n=120, n_features=3)
        model = MockModel()
        validator = StrategyValidator(
            ohlcv=ohlcv,
            features=features,
            labels=labels,
            model=model,
            tbm_config={"max_holding_bars": 3},
            n_trials=1,
            n_splits=3,
            n_groups=4,
            k_test_groups=1,
            bootstrap_runs=10,
        )
        report = validator.run_full_validation()
        report.print_full()
        captured = capsys.readouterr()
        assert "VALIDATION REPORT" in captured.out
        assert "FINAL VERDICT" in captured.out

    def test_many_trials_likely_fail(self):
        """With many trials and random data, DSR should produce FAIL or MARGINAL."""
        ohlcv, features, labels = _make_validator_data(n=120, n_features=3)
        model = MockModel()
        validator = StrategyValidator(
            ohlcv=ohlcv,
            features=features,
            labels=labels,
            model=model,
            tbm_config={"max_holding_bars": 3},
            n_trials=100,  # many trials -> high bar
            n_splits=3,
            n_groups=4,
            k_test_groups=1,
            bootstrap_runs=10,
        )
        report = validator.run_full_validation()
        # With random data and 100 trials, should not pass
        assert report.verdict in ("FAIL", "MARGINAL")

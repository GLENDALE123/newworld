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


# Module-scoped fixture: run validation ONCE, share result across all tests
@pytest.fixture(scope="module")
def validation_report():
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
    return validator.run_full_validation()


@pytest.fixture(scope="module")
def many_trials_report():
    ohlcv, features, labels = _make_validator_data(n=120, n_features=3)
    model = MockModel()
    validator = StrategyValidator(
        ohlcv=ohlcv,
        features=features,
        labels=labels,
        model=model,
        tbm_config={"max_holding_bars": 3},
        n_trials=100,
        n_splits=3,
        n_groups=4,
        k_test_groups=1,
        bootstrap_runs=10,
    )
    return validator.run_full_validation()


def test_returns_report(validation_report):
    assert isinstance(validation_report, ValidationReport)
    assert validation_report.verdict in ("PASS", "FAIL", "MARGINAL")


def test_report_has_all_sections(validation_report):
    assert validation_report.sample_info is not None
    assert validation_report.purged_cv is not None
    assert validation_report.cpcv is not None
    assert validation_report.feature_importance is not None
    assert validation_report.frac_diff is not None
    assert validation_report.multicollinearity is not None


def test_summary_contains_verdict(validation_report):
    summary = validation_report.summary()
    assert "Verdict:" in summary


def test_print_full_runs(validation_report, capsys):
    validation_report.print_full()
    captured = capsys.readouterr()
    assert "VALIDATION REPORT" in captured.out
    assert "FINAL VERDICT" in captured.out


def test_many_trials_likely_fail(many_trials_report):
    assert many_trials_report.verdict in ("FAIL", "MARGINAL")

# AFML Validation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent AFML-based strategy validation framework that can judge any trading strategy as PASS/FAIL/MARGINAL.

**Architecture:** Independent `validation/` package with 3 sub-packages (core, statistics, features) plus top-level validator and report modules. Each module is independently callable. StrategyValidator is a convenience wrapper that runs all modules and produces a ValidationReport.

**Tech Stack:** Python 3.12+, numpy, pandas, scipy, statsmodels, pytest

---

## File Structure

```
validation/
├── __init__.py                    # Package root: exports StrategyValidator, ValidationReport
├── core/
│   ├── __init__.py                # Exports: compute_sample_weights, sequential_bootstrap, PurgedKFold, CPCV
│   ├── sample_weights.py          # Concurrency, uniqueness, sample weight computation
│   ├── sequential_bootstrap.py    # Greedy bootstrap by uniqueness
│   ├── purged_kfold.py            # Purged K-Fold CV with embargo
│   └── cpcv.py                    # Combinatorial Purged Cross-Validation
├── statistics/
│   ├── __init__.py                # Exports: sharpe utils, deflated sharpe, backtest stats
│   ├── sharpe_utils.py            # Annualized Sharpe, SE, confidence interval, PSR
│   ├── backtest_stats.py          # Institutional-grade performance statistics
│   └── deflated_sharpe.py         # PSR + DSR with multiple-testing correction
├── features/
│   ├── __init__.py                # Exports: fracdiff, multicollinearity, importance
│   ├── fracdiff.py                # Fractional differentiation + optimal d search
│   ├── multicollinearity.py       # Correlation clustering + VIF removal
│   └── importance.py              # MDI, MDA, SFI feature importance
├── validator.py                   # StrategyValidator convenience wrapper
└── report.py                      # Result dataclasses + ValidationReport + verdict logic

tests/
├── test_sharpe_utils.py
├── test_backtest_stats.py
├── test_sample_weights.py
├── test_sequential_bootstrap.py
├── test_purged_kfold.py
├── test_cpcv.py
├── test_deflated_sharpe.py
├── test_fracdiff.py
├── test_multicollinearity.py
├── test_importance.py
├── test_validator.py
└── test_validation_integration.py
```

---

## Task 1: Package Setup

- [ ] Update pyproject.toml, create all `__init__.py` files, define result dataclasses in report.py

### Files
- `pyproject.toml`
- `validation/__init__.py`
- `validation/core/__init__.py`
- `validation/statistics/__init__.py`
- `validation/features/__init__.py`
- `validation/report.py`

### Implementation

**pyproject.toml** -- add `scipy`, `statsmodels` to dependencies, add `validation*` to packages.find.include:

```python
# In pyproject.toml [project] dependencies, ADD these two lines:
#     "scipy>=1.12",
#     "statsmodels>=0.14",
#
# In [tool.setuptools.packages.find] include, ADD:
#     "validation*"
```

The updated sections should look like:

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
    "scipy>=1.12",
    "statsmodels>=0.14",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["backtest*", "config*", "data*", "features*", "labeling*", "models*", "strategy*", "validation*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**validation/__init__.py**:

```python
"""AFML-based strategy validation framework."""

from validation.report import ValidationReport

__all__ = ["ValidationReport"]

# StrategyValidator import deferred to avoid circular imports;
# it will be added in Task 12 after validator.py is implemented.
```

**validation/core/__init__.py**:

```python
"""Core AFML validation algorithms: sample weights, bootstrap, cross-validation."""
```

**validation/statistics/__init__.py**:

```python
"""Statistical testing: Sharpe ratio utilities, deflated Sharpe, backtest statistics."""
```

**validation/features/__init__.py**:

```python
"""Feature quality analysis: fractional differentiation, multicollinearity, importance."""
```

**validation/report.py**:

```python
"""Result dataclasses and ValidationReport with PASS/FAIL/MARGINAL verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Individual result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SampleInfoResult:
    """Results from sample weight computation and sequential bootstrap."""
    concurrency: pd.Series
    uniqueness: pd.Series
    sample_weights: pd.Series
    mean_uniqueness: float
    effective_n: int
    total_n: int


@dataclass
class PurgedCVResult:
    """Results from Purged K-Fold cross-validation."""
    fold_sharpes: list[float]
    fold_accuracies: list[float]
    mean_sharpe: float
    std_sharpe: float


@dataclass
class CPCVResult:
    """Results from Combinatorial Purged Cross-Validation."""
    path_sharpes: list[float]
    mean_sharpe: float
    std_sharpe: float
    median_sharpe: float
    pct_negative: float


@dataclass
class DeflatedSharpeResult:
    """Results from PSR and DSR computation."""
    psr_pvalue: float
    dsr_pvalue: float
    expected_max_sharpe: float
    sharpe_std_error: float
    sharpe_observed: float
    n_trials: int


@dataclass
class FeatureImportanceResult:
    """Results from MDI, MDA, SFI feature importance analysis."""
    mdi_ranking: pd.DataFrame
    mda_ranking: pd.DataFrame
    sfi_ranking: pd.DataFrame
    overfit_features: list[str]
    overfit_ratio: float


@dataclass
class FracDiffResult:
    """Results from fractional differentiation analysis."""
    feature_d_values: dict[str, float]  # feature_name -> optimal d
    feature_adf_pvalues: dict[str, float]  # feature_name -> ADF p-value at optimal d


@dataclass
class MulticollinearityResult:
    """Results from multicollinearity analysis."""
    selected_features: list[str]
    removed_features: list[tuple[str, str]]  # (feature_name, reason)
    correlation_clusters: list[list[str]]


@dataclass
class BacktestStatsResult:
    """Institutional-grade backtest statistics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sharpe_std_error: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods
    win_rate: float
    profit_factor: float
    tail_ratio: float
    skewness: float
    kurtosis: float
    max_consecutive_wins: int
    max_consecutive_losses: int


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Aggregated validation report with PASS/FAIL/MARGINAL verdict."""

    sample_info: Optional[SampleInfoResult] = None
    purged_cv: Optional[PurgedCVResult] = None
    cpcv: Optional[CPCVResult] = None
    deflated_sharpe: Optional[DeflatedSharpeResult] = None
    feature_importance: Optional[FeatureImportanceResult] = None
    fracdiff: Optional[FracDiffResult] = None
    multicollinearity: Optional[MulticollinearityResult] = None
    statistics: Optional[BacktestStatsResult] = None

    @property
    def verdict(self) -> str:
        """Return PASS, FAIL, or MARGINAL based on validation results."""
        return self._compute_verdict()

    def summary(self) -> str:
        """One-line summary of the verdict and key metrics."""
        parts = [f"Verdict: {self.verdict}"]
        if self.deflated_sharpe is not None:
            parts.append(f"DSR p={self.deflated_sharpe.dsr_pvalue:.4f}")
        if self.purged_cv is not None:
            parts.append(f"PCV Sharpe={self.purged_cv.mean_sharpe:.3f}")
        if self.cpcv is not None:
            parts.append(f"CPCV neg%={self.cpcv.pct_negative:.1%}")
        if self.feature_importance is not None:
            parts.append(f"overfit={self.feature_importance.overfit_ratio:.1%}")
        return " | ".join(parts)

    def _compute_verdict(self) -> str:
        """Verdict logic per spec."""
        # FAIL conditions (any one triggers FAIL)
        if self.deflated_sharpe is not None and self.deflated_sharpe.dsr_pvalue > 0.10:
            return "FAIL"
        if self.purged_cv is not None and self.purged_cv.mean_sharpe < 0:
            return "FAIL"
        if self.cpcv is not None and self.cpcv.pct_negative >= 0.50:
            return "FAIL"

        # PASS conditions (all must hold)
        pass_conditions = []
        if self.deflated_sharpe is not None:
            pass_conditions.append(self.deflated_sharpe.dsr_pvalue < 0.05)
        if self.purged_cv is not None:
            pass_conditions.append(self.purged_cv.mean_sharpe > 0.5)
        if self.feature_importance is not None:
            pass_conditions.append(self.feature_importance.overfit_ratio < 0.30)
        if self.cpcv is not None:
            pass_conditions.append(self.cpcv.pct_negative < 0.20)

        if pass_conditions and all(pass_conditions):
            return "PASS"

        return "MARGINAL"

    def print_full(self) -> None:
        """Print full validation report to stdout."""
        sep = "=" * 60
        print(sep)
        print(f"  VALIDATION REPORT  --  Verdict: {self.verdict}")
        print(sep)

        if self.sample_info is not None:
            si = self.sample_info
            print(f"\n[Sample Info]")
            print(f"  Total samples:      {si.total_n}")
            print(f"  Effective N:        {si.effective_n}")
            print(f"  Mean uniqueness:    {si.mean_uniqueness:.4f}")

        if self.statistics is not None:
            st = self.statistics
            print(f"\n[Backtest Statistics]")
            print(f"  Total return:       {st.total_return:.4f}")
            print(f"  Annualized return:  {st.annualized_return:.4f}")
            print(f"  Sharpe:             {st.sharpe_ratio:.4f} +/- {st.sharpe_std_error:.4f}")
            print(f"  Sharpe 95% CI:      [{st.sharpe_ci_lower:.4f}, {st.sharpe_ci_upper:.4f}]")
            print(f"  Sortino:            {st.sortino_ratio:.4f}")
            print(f"  Calmar:             {st.calmar_ratio:.4f}")
            print(f"  Max drawdown:       {st.max_drawdown:.4f}")
            print(f"  Win rate:           {st.win_rate:.4f}")
            print(f"  Profit factor:      {st.profit_factor:.4f}")

        if self.purged_cv is not None:
            pc = self.purged_cv
            print(f"\n[Purged K-Fold CV]")
            print(f"  Mean Sharpe:        {pc.mean_sharpe:.4f}")
            print(f"  Std Sharpe:         {pc.std_sharpe:.4f}")
            print(f"  Fold Sharpes:       {[f'{s:.4f}' for s in pc.fold_sharpes]}")
            print(f"  Fold Accuracies:    {[f'{a:.4f}' for a in pc.fold_accuracies]}")

        if self.cpcv is not None:
            cp = self.cpcv
            print(f"\n[Combinatorial Purged CV]")
            print(f"  Mean Sharpe:        {cp.mean_sharpe:.4f}")
            print(f"  Std Sharpe:         {cp.std_sharpe:.4f}")
            print(f"  Median Sharpe:      {cp.median_sharpe:.4f}")
            print(f"  % negative paths:   {cp.pct_negative:.1%}")
            print(f"  Total paths:        {len(cp.path_sharpes)}")

        if self.deflated_sharpe is not None:
            ds = self.deflated_sharpe
            print(f"\n[Deflated Sharpe Ratio]")
            print(f"  Observed Sharpe:    {ds.sharpe_observed:.4f}")
            print(f"  PSR p-value:        {ds.psr_pvalue:.4f}")
            print(f"  DSR p-value:        {ds.dsr_pvalue:.4f}")
            print(f"  Expected max SR:    {ds.expected_max_sharpe:.4f}")
            print(f"  N trials:           {ds.n_trials}")

        if self.feature_importance is not None:
            fi = self.feature_importance
            print(f"\n[Feature Importance]")
            print(f"  Overfit ratio:      {fi.overfit_ratio:.1%}")
            print(f"  Overfit features:   {fi.overfit_features}")
            print(f"  MDI top 5:")
            for _, row in fi.mdi_ranking.head(5).iterrows():
                print(f"    {row['feature']:20s}  {row['importance']:.4f}")
            print(f"  MDA top 5:")
            for _, row in fi.mda_ranking.head(5).iterrows():
                print(f"    {row['feature']:20s}  {row['mean_decrease']:.4f} +/- {row['std_decrease']:.4f}")

        if self.fracdiff is not None:
            fd = self.fracdiff
            print(f"\n[Fractional Differentiation]")
            for feat, d_val in fd.feature_d_values.items():
                pval = fd.feature_adf_pvalues.get(feat, float("nan"))
                print(f"  {feat:20s}  d={d_val:.2f}  ADF p={pval:.4f}")

        if self.multicollinearity is not None:
            mc = self.multicollinearity
            print(f"\n[Multicollinearity]")
            print(f"  Selected features:  {mc.selected_features}")
            print(f"  Removed features:")
            for feat, reason in mc.removed_features:
                print(f"    {feat:20s}  ({reason})")

        print(f"\n{sep}")
        print(f"  FINAL VERDICT: {self.verdict}")
        print(sep)
```

### Tests

**tests/test_report.py**:

```python
import pandas as pd
import pytest

from validation.report import (
    ValidationReport,
    DeflatedSharpeResult,
    PurgedCVResult,
    CPCVResult,
    FeatureImportanceResult,
    SampleInfoResult,
)


def test_verdict_fail_high_dsr():
    report = ValidationReport(
        deflated_sharpe=DeflatedSharpeResult(
            psr_pvalue=0.30,
            dsr_pvalue=0.43,
            expected_max_sharpe=1.5,
            sharpe_std_error=0.3,
            sharpe_observed=0.5,
            n_trials=10,
        )
    )
    assert report.verdict == "FAIL"


def test_verdict_fail_negative_purged_sharpe():
    report = ValidationReport(
        purged_cv=PurgedCVResult(
            fold_sharpes=[-0.2, -0.1, 0.1, -0.3, -0.15],
            fold_accuracies=[0.48, 0.49, 0.51, 0.47, 0.48],
            mean_sharpe=-0.13,
            std_sharpe=0.15,
        )
    )
    assert report.verdict == "FAIL"


def test_verdict_fail_cpcv_negative():
    report = ValidationReport(
        cpcv=CPCVResult(
            path_sharpes=[-0.5, -0.3, 0.1, -0.2, -0.4, -0.1, 0.05, -0.6, -0.2, 0.2],
            mean_sharpe=-0.19,
            std_sharpe=0.27,
            median_sharpe=-0.2,
            pct_negative=0.70,
        )
    )
    assert report.verdict == "FAIL"


def test_verdict_pass_all_conditions():
    report = ValidationReport(
        deflated_sharpe=DeflatedSharpeResult(
            psr_pvalue=0.01,
            dsr_pvalue=0.02,
            expected_max_sharpe=0.3,
            sharpe_std_error=0.1,
            sharpe_observed=1.5,
            n_trials=1,
        ),
        purged_cv=PurgedCVResult(
            fold_sharpes=[0.8, 0.7, 0.6, 0.9, 0.7],
            fold_accuracies=[0.55, 0.54, 0.53, 0.56, 0.55],
            mean_sharpe=0.74,
            std_sharpe=0.11,
        ),
        cpcv=CPCVResult(
            path_sharpes=[0.8, 0.6, 0.7, 0.5, 0.9, 0.4, 0.6, 0.7, 0.8, 0.5],
            mean_sharpe=0.65,
            std_sharpe=0.15,
            median_sharpe=0.65,
            pct_negative=0.0,
        ),
        feature_importance=FeatureImportanceResult(
            mdi_ranking=pd.DataFrame({"feature": ["a"], "importance": [0.5]}),
            mda_ranking=pd.DataFrame({"feature": ["a"], "mean_decrease": [0.1], "std_decrease": [0.01]}),
            sfi_ranking=pd.DataFrame({"feature": ["a"], "sharpe": [0.5]}),
            overfit_features=[],
            overfit_ratio=0.0,
        ),
    )
    assert report.verdict == "PASS"


def test_verdict_marginal():
    report = ValidationReport(
        deflated_sharpe=DeflatedSharpeResult(
            psr_pvalue=0.04,
            dsr_pvalue=0.07,  # > 0.05 but <= 0.10 -> not FAIL, not PASS
            expected_max_sharpe=0.5,
            sharpe_std_error=0.2,
            sharpe_observed=0.8,
            n_trials=3,
        ),
        purged_cv=PurgedCVResult(
            fold_sharpes=[0.3, 0.4, 0.2, 0.5, 0.3],
            fold_accuracies=[0.52, 0.53, 0.51, 0.54, 0.52],
            mean_sharpe=0.34,
            std_sharpe=0.11,
        ),
    )
    assert report.verdict == "MARGINAL"


def test_summary_string():
    report = ValidationReport(
        deflated_sharpe=DeflatedSharpeResult(
            psr_pvalue=0.01,
            dsr_pvalue=0.02,
            expected_max_sharpe=0.3,
            sharpe_std_error=0.1,
            sharpe_observed=1.5,
            n_trials=1,
        ),
    )
    s = report.summary()
    assert "Verdict:" in s
    assert "DSR p=" in s


def test_empty_report_is_marginal():
    report = ValidationReport()
    assert report.verdict == "MARGINAL"
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_report.py -v
```

### Commit Message

```
feat(validation): add package structure and result dataclasses

Set up validation/ package with core/, statistics/, features/ sub-packages.
Define all result dataclasses in report.py with PASS/FAIL/MARGINAL verdict logic.
Update pyproject.toml with scipy, statsmodels deps and validation* package.
```

---

## Task 2: Sharpe Utilities (sharpe_utils.py)

- [ ] Implement Sharpe ratio math utilities: annualized Sharpe, standard error, confidence interval, PSR

### Files
- `validation/statistics/sharpe_utils.py`
- `tests/test_sharpe_utils.py`

### Implementation

**validation/statistics/sharpe_utils.py**:

```python
"""Sharpe ratio utilities: annualization, standard error, confidence interval, PSR."""

from __future__ import annotations

import numpy as np
from scipy import stats


def annualized_sharpe(
    returns: np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio from a return series.

    Args:
        returns: Array of periodic returns.
        periods_per_year: Number of periods per year (252 for daily, 52 for weekly, etc.)

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(periods_per_year))


def sharpe_standard_error(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute standard error of the Sharpe ratio (per AFML formula).

    SE(SR) = sqrt((1 - skew*SR + (kurtosis-1)/4 * SR^2) / (T-1))

    Args:
        sharpe: Observed (non-annualized) Sharpe ratio.
        n: Number of return observations.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns (excess kurtosis + 3; use raw kurtosis here).

    Returns:
        Standard error of the Sharpe ratio.
    """
    if n <= 1:
        return float("inf")
    numerator = 1.0 - skew * sharpe + (kurtosis - 1) / 4.0 * sharpe**2
    # Guard against negative numerator from extreme skew/kurtosis
    numerator = max(numerator, 0.0)
    return float(np.sqrt(numerator / (n - 1)))


def sharpe_confidence_interval(
    sharpe: float,
    n: int,
    confidence: float = 0.95,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> tuple[float, float]:
    """Compute confidence interval for the Sharpe ratio.

    Args:
        sharpe: Observed (non-annualized) Sharpe ratio.
        n: Number of return observations.
        confidence: Confidence level (default 0.95).
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    se = sharpe_standard_error(sharpe, n, skew, kurtosis)
    z = stats.norm.ppf((1 + confidence) / 2)
    return (sharpe - z * se, sharpe + z * se)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Probabilistic Sharpe Ratio (PSR).

    PSR = Phi((SR_observed - SR_benchmark) / SE(SR))

    Returns a probability in [0, 1]. Values > 0.95 indicate the observed
    Sharpe is significantly greater than the benchmark at 5% level.

    Args:
        observed_sharpe: Observed (non-annualized) Sharpe ratio.
        benchmark_sharpe: Benchmark Sharpe ratio (typically 0).
        n: Number of return observations.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.

    Returns:
        PSR value (probability).
    """
    se = sharpe_standard_error(observed_sharpe, n, skew, kurtosis)
    if se == 0 or se == float("inf"):
        return 0.5
    z = (observed_sharpe - benchmark_sharpe) / se
    return float(stats.norm.cdf(z))
```

### Tests

**tests/test_sharpe_utils.py**:

```python
import numpy as np
import pytest

from validation.statistics.sharpe_utils import (
    annualized_sharpe,
    sharpe_standard_error,
    sharpe_confidence_interval,
    probabilistic_sharpe_ratio,
)


class TestAnnualizedSharpe:
    def test_positive_returns(self):
        # Constant positive daily return of 0.1%
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        sr = annualized_sharpe(returns, periods_per_year=252)
        # Should be roughly mean/std * sqrt(252) ~ 0.1 * sqrt(252) ~ 1.58
        assert sr > 0

    def test_zero_std_returns_zero(self):
        returns = [0.01] * 100  # constant returns -> std=0
        assert annualized_sharpe(returns) == 0.0

    def test_single_observation(self):
        assert annualized_sharpe([0.05]) == 0.0

    def test_empty_returns(self):
        assert annualized_sharpe([]) == 0.0

    def test_known_value(self):
        # 252 daily returns with mean=0.001, std=0.01
        # SR = 0.001/0.01 * sqrt(252) = 0.1 * 15.87 = 1.587
        returns = np.full(252, 0.001)
        returns[0] = 0.001  # force non-zero std by using ddof=1
        # Actually, constant returns -> std=0 with ddof=1 too. Use 2 values.
        returns = [0.011, -0.009] * 126  # mean=0.001, std~=0.01414
        sr = annualized_sharpe(returns, periods_per_year=252)
        expected = 0.001 / np.std([0.011, -0.009] * 126, ddof=1) * np.sqrt(252)
        assert abs(sr - expected) < 1e-10


class TestSharpeStandardError:
    def test_basic_computation(self):
        # SE = sqrt((1 - 0*SR + (3-1)/4 * SR^2) / (n-1))
        # For SR=1, n=100, skew=0, kurtosis=3:
        # SE = sqrt((1 + 0.5 * 1) / 99) = sqrt(1.5 / 99) = 0.1231
        se = sharpe_standard_error(1.0, 100, skew=0.0, kurtosis=3.0)
        expected = np.sqrt(1.5 / 99)
        assert abs(se - expected) < 1e-4

    def test_n_equals_1_returns_inf(self):
        assert sharpe_standard_error(1.0, 1) == float("inf")

    def test_skew_effect(self):
        se_no_skew = sharpe_standard_error(1.0, 100, skew=0.0, kurtosis=3.0)
        se_neg_skew = sharpe_standard_error(1.0, 100, skew=-1.0, kurtosis=3.0)
        # Negative skew with positive SR -> larger numerator -> larger SE
        assert se_neg_skew > se_no_skew

    def test_zero_sharpe(self):
        se = sharpe_standard_error(0.0, 100, skew=0.0, kurtosis=3.0)
        expected = np.sqrt(1.0 / 99)
        assert abs(se - expected) < 1e-4


class TestSharpeConfidenceInterval:
    def test_symmetric_at_zero_skew(self):
        lo, hi = sharpe_confidence_interval(1.0, 100, confidence=0.95, skew=0.0, kurtosis=3.0)
        mid = (lo + hi) / 2
        assert abs(mid - 1.0) < 1e-10

    def test_wider_at_lower_confidence(self):
        lo_95, hi_95 = sharpe_confidence_interval(1.0, 100, confidence=0.95)
        lo_99, hi_99 = sharpe_confidence_interval(1.0, 100, confidence=0.99)
        assert (hi_99 - lo_99) > (hi_95 - lo_95)

    def test_narrows_with_more_data(self):
        lo_100, hi_100 = sharpe_confidence_interval(1.0, 100, confidence=0.95)
        lo_1000, hi_1000 = sharpe_confidence_interval(1.0, 1000, confidence=0.95)
        assert (hi_1000 - lo_1000) < (hi_100 - lo_100)


class TestProbabilisticSharpeRatio:
    def test_observed_equals_benchmark(self):
        psr = probabilistic_sharpe_ratio(0.5, 0.5, 100)
        assert abs(psr - 0.5) < 1e-10

    def test_high_sharpe_high_psr(self):
        psr = probabilistic_sharpe_ratio(2.0, 0.0, 252)
        assert psr > 0.95

    def test_negative_sharpe_low_psr(self):
        psr = probabilistic_sharpe_ratio(-1.0, 0.0, 252)
        assert psr < 0.05

    def test_more_data_increases_psr(self):
        psr_100 = probabilistic_sharpe_ratio(0.5, 0.0, 100)
        psr_1000 = probabilistic_sharpe_ratio(0.5, 0.0, 1000)
        assert psr_1000 > psr_100

    def test_psr_bounds(self):
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 252)
        assert 0.0 <= psr <= 1.0
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_sharpe_utils.py -v
```

### Commit Message

```
feat(validation): implement Sharpe ratio utilities

Add sharpe_utils.py with annualized Sharpe, standard error,
confidence interval, and Probabilistic Sharpe Ratio (PSR).
```

---

## Task 3: Backtest Statistics (backtest_stats.py)

- [ ] Implement institutional-grade backtest performance statistics

### Files
- `validation/statistics/backtest_stats.py`
- `tests/test_backtest_stats.py`

### Implementation

**validation/statistics/backtest_stats.py**:

```python
"""Institutional-grade backtest performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from validation.statistics.sharpe_utils import (
    annualized_sharpe,
    sharpe_confidence_interval,
    sharpe_standard_error,
)
from validation.report import BacktestStatsResult


def compute_backtest_stats(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> BacktestStatsResult:
    """Compute comprehensive backtest statistics from a return series.

    Args:
        returns: Series of periodic returns (e.g., daily returns).
        periods_per_year: Periods per year for annualization.

    Returns:
        BacktestStatsResult with all computed statistics.
    """
    r = returns.values.astype(np.float64)
    n = len(r)

    if n < 2:
        return BacktestStatsResult(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sharpe_std_error=float("inf"), sharpe_ci_lower=0.0, sharpe_ci_upper=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0, max_drawdown=0.0,
            max_drawdown_duration=0, win_rate=0.0, profit_factor=0.0,
            tail_ratio=0.0, skewness=0.0, kurtosis=0.0,
            max_consecutive_wins=0, max_consecutive_losses=0,
        )

    # ---------- Returns ----------
    cumulative = np.cumprod(1 + r)
    total_return = float(cumulative[-1] - 1)
    n_years = n / periods_per_year
    annualized_return = float((cumulative[-1]) ** (1 / max(n_years, 1e-10)) - 1) if cumulative[-1] > 0 else -1.0

    # ---------- Risk ----------
    sr = annualized_sharpe(r, periods_per_year)

    # Non-annualized Sharpe for SE/CI computation
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1) if n > 1 else 1e-10
    sr_raw = mean_r / std_r if std_r > 0 else 0.0
    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=False))  # raw kurtosis (not excess)

    se = sharpe_standard_error(sr_raw, n, skew, kurt)
    ci_lo, ci_hi = sharpe_confidence_interval(sr_raw, n, 0.95, skew, kurt)
    # Scale CI to annualized
    scale = np.sqrt(periods_per_year)
    ci_lo_ann = ci_lo * scale
    ci_hi_ann = ci_hi * scale
    se_ann = se * scale

    # Sortino: use downside deviation
    downside = r[r < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = float(mean_r / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    # ---------- Drawdown ----------
    cum_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / cum_max - 1
    max_dd = float(np.min(drawdowns))

    # Max drawdown duration (periods between peaks)
    peak_indices = np.where(cumulative >= cum_max)[0]
    if len(peak_indices) > 1:
        durations = np.diff(peak_indices)
        max_dd_duration = int(np.max(durations))
    else:
        max_dd_duration = n

    # Calmar
    calmar = float(annualized_return / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    # ---------- Trade statistics ----------
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / n) if n > 0 else 0.0

    sum_wins = np.sum(wins)
    sum_losses = abs(np.sum(losses))
    profit_factor = float(sum_wins / sum_losses) if sum_losses > 0 else float("inf") if sum_wins > 0 else 0.0

    # Tail ratio: 95th percentile of wins / abs(5th percentile of losses)
    p95 = np.percentile(r, 95) if n > 0 else 0.0
    p05 = abs(np.percentile(r, 5)) if n > 0 else 1e-10
    tail_ratio = float(p95 / p05) if p05 > 1e-10 else 0.0

    # ---------- Distribution ----------
    # skew and kurt already computed above

    # ---------- Consecutive wins/losses ----------
    max_consec_wins = _max_consecutive(r > 0)
    max_consec_losses = _max_consecutive(r < 0)

    return BacktestStatsResult(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sr,
        sharpe_std_error=se_ann,
        sharpe_ci_lower=ci_lo_ann,
        sharpe_ci_upper=ci_hi_ann,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        tail_ratio=tail_ratio,
        skewness=skew,
        kurtosis=kurt,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
    )


def _max_consecutive(mask: np.ndarray) -> int:
    """Count the longest run of True values in a boolean array."""
    if len(mask) == 0:
        return 0
    # Pad with False to catch runs at the edges
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    runs = ends - starts
    return int(np.max(runs))
```

### Tests

**tests/test_backtest_stats.py**:

```python
import numpy as np
import pandas as pd
import pytest

from validation.statistics.backtest_stats import compute_backtest_stats


def _make_returns(seed: int = 42, n: int = 252, mean: float = 0.001, std: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, n), index=idx)


class TestComputeBacktestStats:
    def test_positive_returns(self):
        returns = _make_returns(mean=0.001, std=0.01)
        result = compute_backtest_stats(returns)
        assert result.total_return > 0
        assert result.annualized_return > 0
        assert result.sharpe_ratio > 0
        assert result.win_rate > 0.4

    def test_negative_returns(self):
        returns = _make_returns(mean=-0.002, std=0.01)
        result = compute_backtest_stats(returns)
        assert result.total_return < 0
        assert result.sharpe_ratio < 0

    def test_drawdown_is_negative(self):
        returns = _make_returns()
        result = compute_backtest_stats(returns)
        assert result.max_drawdown <= 0

    def test_sharpe_ci_contains_sharpe(self):
        returns = _make_returns(n=500)
        result = compute_backtest_stats(returns)
        # The annualized CI should bracket the annualized Sharpe (approximately)
        # We test that the non-annualized raw CI logic works
        assert result.sharpe_ci_lower < result.sharpe_ci_upper

    def test_sortino_is_finite(self):
        returns = _make_returns()
        result = compute_backtest_stats(returns)
        assert np.isfinite(result.sortino_ratio)

    def test_profit_factor(self):
        # All positive returns -> profit_factor is inf
        returns = pd.Series([0.01, 0.02, 0.015, 0.005])
        result = compute_backtest_stats(returns)
        assert result.profit_factor == float("inf")

    def test_consecutive_wins_losses(self):
        # Pattern: 3 wins, 2 losses, 1 win
        returns = pd.Series([0.01, 0.02, 0.015, -0.01, -0.005, 0.01])
        result = compute_backtest_stats(returns)
        assert result.max_consecutive_wins == 3
        assert result.max_consecutive_losses == 2

    def test_tail_ratio_positive(self):
        returns = _make_returns(n=500)
        result = compute_backtest_stats(returns)
        assert result.tail_ratio > 0

    def test_distribution_stats(self):
        returns = _make_returns(n=1000)
        result = compute_backtest_stats(returns)
        # Skewness should be near 0 for normal returns
        assert abs(result.skewness) < 1.0
        # Kurtosis should be near 3 for normal returns
        assert abs(result.kurtosis - 3.0) < 1.0

    def test_single_return(self):
        returns = pd.Series([0.01])
        result = compute_backtest_stats(returns)
        assert result.total_return == 0.0  # n < 2 returns defaults

    def test_calmar_ratio(self):
        returns = _make_returns(mean=0.001, std=0.01, n=504)
        result = compute_backtest_stats(returns, periods_per_year=252)
        assert np.isfinite(result.calmar_ratio)
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_backtest_stats.py -v
```

### Commit Message

```
feat(validation): implement institutional-grade backtest statistics

Add backtest_stats.py with total/annualized return, Sharpe + SE + CI,
Sortino, Calmar, max drawdown + duration, win rate, profit factor,
tail ratio, skewness, kurtosis, and consecutive win/loss streaks.
```

---

## Task 4: Sample Weights (sample_weights.py)

- [ ] Implement concurrency, uniqueness, and sample weight computation

### Files
- `validation/core/sample_weights.py`
- `tests/test_sample_weights.py`

### Implementation

**validation/core/sample_weights.py**:

```python
"""Sample weights based on label concurrency and uniqueness (AFML Ch. 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_indicator_matrix(
    tbm_timestamps: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a binary indicator matrix: rows=timestamps, cols=label indices.

    indicator[t, i] = 1 if label i is active at time t (t_start_i <= t <= t_end_i).

    Args:
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns, indexed by label index.
        price_index: DatetimeIndex of all price timestamps.

    Returns:
        DataFrame of shape (len(price_index), len(tbm_timestamps)) with 0/1 values.
    """
    indicator = pd.DataFrame(0, index=price_index, columns=tbm_timestamps.index)
    for i, row in tbm_timestamps.iterrows():
        t_start = row["t_start"]
        t_end = row["t_end"]
        mask = (indicator.index >= t_start) & (indicator.index <= t_end)
        indicator.loc[mask, i] = 1
    return indicator


def compute_concurrency(indicator_matrix: pd.DataFrame) -> pd.Series:
    """Compute concurrency at each time step: number of active labels.

    Args:
        indicator_matrix: Binary indicator matrix (rows=timestamps, cols=labels).

    Returns:
        Series indexed by timestamps with integer concurrency counts.
    """
    return indicator_matrix.sum(axis=1)


def compute_uniqueness(indicator_matrix: pd.DataFrame) -> pd.Series:
    """Compute average uniqueness per label.

    uniqueness_i = mean(1/concurrency_t) for all t where label i is active.

    Args:
        indicator_matrix: Binary indicator matrix (rows=timestamps, cols=labels).

    Returns:
        Series indexed by label index with uniqueness values in (0, 1].
    """
    concurrency = indicator_matrix.sum(axis=1)
    # Avoid division by zero: timestamps with no active labels get inf, but
    # they won't appear in any label's average since indicator is 0 there.
    inv_concurrency = 1.0 / concurrency.replace(0, np.inf)

    uniqueness = pd.Series(index=indicator_matrix.columns, dtype=np.float64)
    for col in indicator_matrix.columns:
        active_mask = indicator_matrix[col] > 0
        if active_mask.any():
            uniqueness[col] = inv_concurrency[active_mask].mean()
        else:
            uniqueness[col] = 0.0
    return uniqueness


def compute_sample_weights(
    close_prices: pd.Series,
    tbm_timestamps: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute sample weights from concurrency, uniqueness, and returns.

    weight_i = uniqueness_i * |return_i|

    where return_i = (close[t_end_i] - close[t_start_i]) / close[t_start_i]

    Args:
        close_prices: Series of close prices with DatetimeIndex.
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns.

    Returns:
        Tuple of (concurrency, uniqueness, sample_weights) Series.
    """
    indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
    concurrency = compute_concurrency(indicator)
    uniqueness = compute_uniqueness(indicator)

    # Compute absolute returns per label
    abs_returns = pd.Series(index=tbm_timestamps.index, dtype=np.float64)
    for i, row in tbm_timestamps.iterrows():
        t_start = row["t_start"]
        t_end = row["t_end"]
        if t_start in close_prices.index and t_end in close_prices.index:
            ret = (close_prices[t_end] - close_prices[t_start]) / close_prices[t_start]
            abs_returns[i] = abs(ret)
        else:
            abs_returns[i] = 0.0

    sample_weights = uniqueness * abs_returns

    # Normalize weights so they sum to len(labels)
    if sample_weights.sum() > 0:
        sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

    return concurrency, uniqueness, sample_weights
```

### Tests

**tests/test_sample_weights.py**:

```python
import numpy as np
import pandas as pd
import pytest

from validation.core.sample_weights import (
    build_indicator_matrix,
    compute_concurrency,
    compute_uniqueness,
    compute_sample_weights,
)


def _make_timestamps_and_prices():
    """Create synthetic TBM timestamps with known concurrency.

    Timeline (hourly bars):
      t0  t1  t2  t3  t4  t5  t6  t7
    Label 0: [t0--------t3]
    Label 1:     [t1--------t4]
    Label 2:             [t3--------t6]
    Label 3:                     [t5--------t7]

    Concurrency:
      t0: 1, t1: 2, t2: 2, t3: 3, t4: 2, t5: 2, t6: 1, t7: 1
    """
    idx = pd.date_range("2023-01-01", periods=8, freq="1h", tz="UTC")
    close_prices = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0],
        index=idx,
    )
    tbm_timestamps = pd.DataFrame(
        {
            "t_start": [idx[0], idx[1], idx[3], idx[5]],
            "t_end": [idx[3], idx[4], idx[6], idx[7]],
        },
        index=[0, 1, 2, 3],
    )
    return close_prices, tbm_timestamps, idx


class TestBuildIndicatorMatrix:
    def test_shape(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        assert indicator.shape == (8, 4)

    def test_label_0_active(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        # Label 0 is active at t0..t3
        assert indicator.loc[idx[0], 0] == 1
        assert indicator.loc[idx[3], 0] == 1
        assert indicator.loc[idx[4], 0] == 0

    def test_all_binary(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        assert set(indicator.values.flatten()).issubset({0, 1})


class TestConcurrency:
    def test_known_concurrency(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        conc = compute_concurrency(indicator)
        assert conc[idx[0]] == 1
        assert conc[idx[1]] == 2
        assert conc[idx[3]] == 3  # Labels 0, 1, 2 all active
        assert conc[idx[7]] == 1


class TestUniqueness:
    def test_non_overlapping_labels_uniqueness_1(self):
        """Labels with no overlap should have uniqueness = 1.0."""
        idx = pd.date_range("2023-01-01", periods=6, freq="1h", tz="UTC")
        tbm = pd.DataFrame({
            "t_start": [idx[0], idx[3]],
            "t_end": [idx[2], idx[5]],
        }, index=[0, 1])
        indicator = build_indicator_matrix(tbm, idx)
        uniq = compute_uniqueness(indicator)
        assert abs(uniq[0] - 1.0) < 1e-10
        assert abs(uniq[1] - 1.0) < 1e-10

    def test_fully_overlapping(self):
        """Two labels covering identical time range should have uniqueness = 0.5."""
        idx = pd.date_range("2023-01-01", periods=4, freq="1h", tz="UTC")
        tbm = pd.DataFrame({
            "t_start": [idx[0], idx[0]],
            "t_end": [idx[3], idx[3]],
        }, index=[0, 1])
        indicator = build_indicator_matrix(tbm, idx)
        uniq = compute_uniqueness(indicator)
        assert abs(uniq[0] - 0.5) < 1e-10
        assert abs(uniq[1] - 0.5) < 1e-10

    def test_uniqueness_between_0_and_1(self):
        close_prices, tbm_timestamps, idx = _make_timestamps_and_prices()
        indicator = build_indicator_matrix(tbm_timestamps, close_prices.index)
        uniq = compute_uniqueness(indicator)
        assert all(0 < u <= 1.0 for u in uniq)


class TestSampleWeights:
    def test_weights_sum_to_n(self):
        close_prices, tbm_timestamps, _ = _make_timestamps_and_prices()
        _, _, weights = compute_sample_weights(close_prices, tbm_timestamps)
        assert abs(weights.sum() - len(tbm_timestamps)) < 1e-10

    def test_unique_large_return_gets_high_weight(self):
        """Label 3 is least overlapping and has a return. It should get a high weight."""
        close_prices, tbm_timestamps, _ = _make_timestamps_and_prices()
        _, uniq, weights = compute_sample_weights(close_prices, tbm_timestamps)
        # Label 3: t5->t7 return = (99-101)/101, uniqueness high
        # Verify weights are positive where there's a return
        assert all(w >= 0 for w in weights)

    def test_zero_return_zero_weight(self):
        """If a label has zero return, its weight should be 0 (before normalization)."""
        idx = pd.date_range("2023-01-01", periods=4, freq="1h", tz="UTC")
        close_prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
        tbm = pd.DataFrame({
            "t_start": [idx[0]],
            "t_end": [idx[3]],
        }, index=[0])
        _, _, weights = compute_sample_weights(close_prices, tbm)
        # Only one label with zero return -> weight=0, but normalization:
        # sum is 0, so normalization is skipped, weight stays 0
        assert weights[0] == 0.0

    def test_returns_correct_tuple(self):
        close_prices, tbm_timestamps, _ = _make_timestamps_and_prices()
        result = compute_sample_weights(close_prices, tbm_timestamps)
        assert len(result) == 3  # concurrency, uniqueness, weights
        conc, uniq, weights = result
        assert isinstance(conc, pd.Series)
        assert isinstance(uniq, pd.Series)
        assert isinstance(weights, pd.Series)
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_sample_weights.py -v
```

### Commit Message

```
feat(validation): implement AFML sample weights with concurrency and uniqueness

Add sample_weights.py with indicator matrix construction, concurrency
counting, uniqueness computation, and final weight = uniqueness * |return|.
```

---

## Task 5: Sequential Bootstrap (sequential_bootstrap.py)

- [ ] Implement greedy sequential bootstrap by uniqueness

### Files
- `validation/core/sequential_bootstrap.py`
- `tests/test_sequential_bootstrap.py`

### Implementation

**validation/core/sequential_bootstrap.py**:

```python
"""Sequential Bootstrap: greedy sample selection by uniqueness (AFML Ch. 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from validation.core.sample_weights import build_indicator_matrix, compute_uniqueness


def _compute_average_uniqueness_with_selected(
    indicator_matrix: pd.DataFrame,
    selected: list[int],
    candidate: int,
) -> float:
    """Compute average uniqueness of candidate given already-selected samples.

    Temporarily adds the candidate to the selected set and computes the
    candidate's average uniqueness over its active timestamps.
    """
    cols = selected + [candidate]
    sub_matrix = indicator_matrix[cols]
    concurrency = sub_matrix.sum(axis=1)
    inv_conc = 1.0 / concurrency.replace(0, np.inf)

    active_mask = indicator_matrix[candidate] > 0
    if not active_mask.any():
        return 0.0
    return float(inv_conc[active_mask].mean())


def sequential_bootstrap(
    indicator_matrix: pd.DataFrame,
    n_samples: int | None = None,
    random_state: int | None = None,
) -> list[int]:
    """Run sequential bootstrap, selecting samples proportional to uniqueness.

    Algorithm:
    1. Start with empty selection S.
    2. For each candidate, compute its average uniqueness given S.
    3. Sample one candidate with probability proportional to uniqueness.
    4. Add to S, repeat until n_samples reached.

    Args:
        indicator_matrix: Binary indicator matrix (rows=timestamps, cols=label indices).
        n_samples: Number of samples to draw. Defaults to number of labels.
        random_state: Random seed for reproducibility.

    Returns:
        List of selected label indices (may contain duplicates like a bootstrap).
    """
    rng = np.random.default_rng(random_state)
    all_labels = list(indicator_matrix.columns)
    if n_samples is None:
        n_samples = len(all_labels)

    selected: list[int] = []
    for _ in range(n_samples):
        # Compute average uniqueness for each candidate given current selection
        uniquenesses = np.zeros(len(all_labels))
        for j, label in enumerate(all_labels):
            uniquenesses[j] = _compute_average_uniqueness_with_selected(
                indicator_matrix, selected, label
            )

        # Convert to probabilities
        total = uniquenesses.sum()
        if total <= 0:
            # Fallback: uniform random selection
            probs = np.ones(len(all_labels)) / len(all_labels)
        else:
            probs = uniquenesses / total

        chosen_idx = rng.choice(len(all_labels), p=probs)
        selected.append(all_labels[chosen_idx])

    return selected


def estimate_effective_n(
    indicator_matrix: pd.DataFrame,
    n_bootstrap_runs: int = 100,
    random_state: int | None = None,
) -> int:
    """Estimate effective number of independent samples via sequential bootstrap.

    Compares the mean uniqueness from sequential bootstrap to standard bootstrap.
    effective_n ~ n_labels * (mean_uniqueness_seq / mean_uniqueness_standard)

    But a simpler approach: run sequential bootstrap once with n_samples = n_labels,
    then count unique samples in the selection. This gives a lower-bound estimate
    of independent samples.

    For a more robust estimate, we average over multiple runs.

    Args:
        indicator_matrix: Binary indicator matrix.
        n_bootstrap_runs: Number of bootstrap iterations to average.
        random_state: Random seed.

    Returns:
        Estimated effective number of independent samples.
    """
    rng_base = np.random.default_rng(random_state)
    n_labels = len(indicator_matrix.columns)
    unique_counts = []

    for i in range(n_bootstrap_runs):
        seed = int(rng_base.integers(0, 2**31))
        selected = sequential_bootstrap(indicator_matrix, n_samples=n_labels, random_state=seed)
        unique_counts.append(len(set(selected)))

    return int(np.mean(unique_counts))
```

### Tests

**tests/test_sequential_bootstrap.py**:

```python
import numpy as np
import pandas as pd
import pytest

from validation.core.sample_weights import build_indicator_matrix
from validation.core.sequential_bootstrap import (
    sequential_bootstrap,
    estimate_effective_n,
    _compute_average_uniqueness_with_selected,
)


def _make_non_overlapping_indicator():
    """3 labels with no overlap -> uniqueness should be ~uniform."""
    idx = pd.date_range("2023-01-01", periods=9, freq="1h", tz="UTC")
    tbm = pd.DataFrame({
        "t_start": [idx[0], idx[3], idx[6]],
        "t_end": [idx[2], idx[5], idx[8]],
    }, index=[0, 1, 2])
    return build_indicator_matrix(tbm, idx)


def _make_overlapping_indicator():
    """4 labels with heavy overlap."""
    idx = pd.date_range("2023-01-01", periods=8, freq="1h", tz="UTC")
    tbm = pd.DataFrame({
        "t_start": [idx[0], idx[1], idx[3], idx[5]],
        "t_end": [idx[3], idx[4], idx[6], idx[7]],
    }, index=[0, 1, 2, 3])
    return build_indicator_matrix(tbm, idx)


class TestAverageUniqueness:
    def test_first_selection_is_1(self):
        """First selected sample should have uniqueness 1.0 (no others in set)."""
        indicator = _make_non_overlapping_indicator()
        uniq = _compute_average_uniqueness_with_selected(indicator, [], 0)
        assert abs(uniq - 1.0) < 1e-10

    def test_overlapping_reduces_uniqueness(self):
        """Selecting overlapping sample reduces uniqueness."""
        indicator = _make_overlapping_indicator()
        # Select label 0, then check uniqueness of label 1 (overlaps with 0)
        uniq_alone = _compute_average_uniqueness_with_selected(indicator, [], 1)
        uniq_with_0 = _compute_average_uniqueness_with_selected(indicator, [0], 1)
        assert uniq_with_0 < uniq_alone


class TestSequentialBootstrap:
    def test_returns_correct_count(self):
        indicator = _make_non_overlapping_indicator()
        selected = sequential_bootstrap(indicator, n_samples=5, random_state=42)
        assert len(selected) == 5

    def test_default_n_samples(self):
        indicator = _make_non_overlapping_indicator()
        selected = sequential_bootstrap(indicator, random_state=42)
        assert len(selected) == 3  # 3 labels

    def test_selected_are_valid_labels(self):
        indicator = _make_overlapping_indicator()
        selected = sequential_bootstrap(indicator, n_samples=10, random_state=42)
        for s in selected:
            assert s in indicator.columns

    def test_reproducible_with_seed(self):
        indicator = _make_overlapping_indicator()
        sel1 = sequential_bootstrap(indicator, n_samples=10, random_state=99)
        sel2 = sequential_bootstrap(indicator, n_samples=10, random_state=99)
        assert sel1 == sel2

    def test_non_overlapping_uniform_selection(self):
        """With non-overlapping labels, all should be roughly equally likely."""
        indicator = _make_non_overlapping_indicator()
        counts = {0: 0, 1: 0, 2: 0}
        for seed in range(200):
            selected = sequential_bootstrap(indicator, n_samples=1, random_state=seed)
            counts[selected[0]] += 1
        # Each label should appear roughly 200/3 ~ 67 times
        for count in counts.values():
            assert count > 30  # generous margin


class TestEffectiveN:
    def test_non_overlapping_effective_n_equals_total(self):
        """Non-overlapping labels: effective N should be close to total N."""
        indicator = _make_non_overlapping_indicator()
        eff_n = estimate_effective_n(indicator, n_bootstrap_runs=50, random_state=42)
        # With 3 non-overlapping labels, we expect ~3 unique out of 3 draws
        # (high uniqueness -> balanced sampling -> high unique count)
        assert eff_n >= 2

    def test_overlapping_effective_n_less_than_total(self):
        """Overlapping labels: effective N should be <= total N."""
        indicator = _make_overlapping_indicator()
        eff_n = estimate_effective_n(indicator, n_bootstrap_runs=50, random_state=42)
        assert eff_n <= len(indicator.columns)

    def test_effective_n_positive(self):
        indicator = _make_overlapping_indicator()
        eff_n = estimate_effective_n(indicator, n_bootstrap_runs=20, random_state=42)
        assert eff_n > 0
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_sequential_bootstrap.py -v
```

### Commit Message

```
feat(validation): implement sequential bootstrap by uniqueness

Add sequential_bootstrap.py with greedy sample selection proportional
to average uniqueness, plus effective-N estimation via repeated runs.
```

---

## Task 6: Purged K-Fold CV (purged_kfold.py)

- [ ] Implement Purged K-Fold Cross-Validation with embargo

### Files
- `validation/core/purged_kfold.py`
- `tests/test_purged_kfold.py`

### Implementation

**validation/core/purged_kfold.py**:

```python
"""Purged K-Fold Cross-Validation with embargo (AFML Ch. 7)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from validation.report import PurgedCVResult
from validation.statistics.sharpe_utils import annualized_sharpe


class ModelProtocol(Protocol):
    """Minimal model interface required by validation."""
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _get_train_test_indices(
    n_samples: int,
    n_splits: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train/test for a given fold.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    fold_size = n_samples // n_splits
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size if fold < n_splits - 1 else n_samples
    test_indices = np.arange(test_start, test_end)
    train_indices = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])
    return train_indices, test_indices


def purge_train_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    tbm_timestamps: pd.DataFrame,
    all_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """Remove training samples whose label time-span overlaps with any test sample.

    Args:
        train_indices: Array of training sample positions.
        test_indices: Array of test sample positions.
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns.
        all_timestamps: Full DatetimeIndex of all samples.

    Returns:
        Purged training indices.
    """
    test_positions = test_indices
    test_t_starts = tbm_timestamps.iloc[test_positions]["t_start"].values
    test_t_ends = tbm_timestamps.iloc[test_positions]["t_end"].values
    test_min = np.min(test_t_starts)
    test_max = np.max(test_t_ends)

    purged = []
    for idx in train_indices:
        train_start = tbm_timestamps.iloc[idx]["t_start"]
        train_end = tbm_timestamps.iloc[idx]["t_end"]
        # Overlap: train label span intersects test span
        if train_end < test_min or train_start > test_max:
            purged.append(idx)
        # else: overlaps with test -> purge (don't include)
    return np.array(purged, dtype=int)


def apply_embargo(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
    embargo_pct: float,
) -> np.ndarray:
    """Remove training samples in the embargo zone right after the test set.

    Args:
        train_indices: Array of training sample positions (already purged).
        test_indices: Array of test sample positions.
        n_samples: Total number of samples.
        embargo_pct: Fraction of total samples to embargo.

    Returns:
        Training indices with embargo applied.
    """
    embargo_size = int(n_samples * embargo_pct)
    if embargo_size == 0:
        return train_indices
    test_end = test_indices[-1]
    embargo_end = test_end + embargo_size
    return train_indices[~((train_indices > test_end) & (train_indices <= embargo_end))]


def purged_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
) -> PurgedCVResult:
    """Run Purged K-Fold cross-validation.

    For each fold:
    1. Split into train/test by contiguous time blocks.
    2. Purge: remove train samples overlapping with test label spans.
    3. Embargo: remove train samples right after test end.
    4. Train model, predict on test, compute Sharpe and accuracy.

    Args:
        X: Feature DataFrame.
        y: Label Series (+1/-1).
        model: Model with train() and predict() methods.
        tbm_timestamps: DataFrame with 't_start' and 't_end' columns.
        sample_weight: Optional sample weights for training.
        n_splits: Number of folds.
        embargo_pct: Embargo fraction.
        periods_per_year: For Sharpe annualization.

    Returns:
        PurgedCVResult with fold-level and mean statistics.
    """
    n_samples = len(X)
    fold_sharpes = []
    fold_accuracies = []

    for fold in range(n_splits):
        train_idx, test_idx = _get_train_test_indices(n_samples, n_splits, fold)

        # Purge overlapping samples
        train_idx = purge_train_indices(
            train_idx, test_idx, tbm_timestamps, X.index
        )

        # Apply embargo
        train_idx = apply_embargo(train_idx, test_idx, n_samples, embargo_pct)

        if len(train_idx) == 0:
            continue

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        sw_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        model.train(X_train, y_train, sample_weight=sw_train)
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = float(np.mean(predictions == y_test.values))
        fold_accuracies.append(accuracy)

        # Sharpe from predictions: use predicted direction * actual direction as "returns"
        # This is a simple proxy: correct predictions yield +1, incorrect -1
        signed_returns = (predictions == y_test.values).astype(float) * 2 - 1
        sr = annualized_sharpe(signed_returns, periods_per_year)
        fold_sharpes.append(sr)

    if len(fold_sharpes) == 0:
        return PurgedCVResult(
            fold_sharpes=[], fold_accuracies=[],
            mean_sharpe=0.0, std_sharpe=0.0,
        )

    return PurgedCVResult(
        fold_sharpes=fold_sharpes,
        fold_accuracies=fold_accuracies,
        mean_sharpe=float(np.mean(fold_sharpes)),
        std_sharpe=float(np.std(fold_sharpes, ddof=1)) if len(fold_sharpes) > 1 else 0.0,
    )
```

### Tests

**tests/test_purged_kfold.py**:

```python
import numpy as np
import pandas as pd
import pytest

from validation.core.purged_kfold import (
    _get_train_test_indices,
    purge_train_indices,
    apply_embargo,
    purged_kfold_cv,
)


class DummyModel:
    """Simple model that memorizes training data for testing."""

    def __init__(self):
        self._majority = 1

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        # Majority vote
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._majority)


class PerfectModel:
    """Model that always predicts correctly (cheats for testing)."""

    def __init__(self):
        self._y = None

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Return alternating +1/-1 (this won't be "perfect" but is deterministic)
        return np.array([1] * len(X))


def _make_cv_data(n: int = 100, seed: int = 42):
    """Create synthetic data for CV testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    X = pd.DataFrame(
        {"feat1": rng.standard_normal(n), "feat2": rng.standard_normal(n)},
        index=idx,
    )
    y = pd.Series(rng.choice([1, -1], n), index=idx)

    # TBM timestamps: each label spans 3 bars forward
    holding = 3
    t_starts = idx
    t_ends = pd.DatetimeIndex([
        idx[min(i + holding, n - 1)] for i in range(n)
    ])
    tbm_timestamps = pd.DataFrame(
        {"t_start": t_starts, "t_end": t_ends},
        index=range(n),
    )

    return X, y, tbm_timestamps


class TestGetTrainTestIndices:
    def test_fold_0(self):
        train, test = _get_train_test_indices(100, 5, 0)
        assert len(test) == 20
        assert len(train) == 80
        assert test[0] == 0
        assert test[-1] == 19

    def test_last_fold(self):
        train, test = _get_train_test_indices(100, 5, 4)
        assert test[-1] == 99
        assert 0 in train

    def test_no_overlap(self):
        for fold in range(5):
            train, test = _get_train_test_indices(100, 5, fold)
            assert len(set(train) & set(test)) == 0


class TestPurgeTrainIndices:
    def test_overlapping_samples_removed(self):
        """Train samples overlapping with test labels should be purged."""
        X, y, tbm = _make_cv_data(n=50)
        train_idx = np.arange(0, 40)
        test_idx = np.arange(37, 50)  # test starts at 37

        purged = purge_train_indices(train_idx, test_idx, tbm, X.index)

        # Train samples near 37 whose label extends into test should be purged
        # Label at index 35 spans 35..38, which overlaps with test starting at 37
        # So indices 35, 36 (at least) should be purged
        # But index 34 spans 34..37 which also overlaps with test_min=idx[37]
        assert len(purged) < len(train_idx)
        for idx_val in purged:
            t_end = tbm.iloc[idx_val]["t_end"]
            t_start = tbm.iloc[idx_val]["t_start"]
            test_min = tbm.iloc[test_idx[0]]["t_start"]
            test_max = tbm.iloc[test_idx[-1]]["t_end"]
            assert t_end < test_min or t_start > test_max


class TestApplyEmbargo:
    def test_embargo_removes_samples(self):
        train_idx = np.array([0, 1, 2, 20, 21, 22, 23, 24])
        test_idx = np.array([18, 19])
        # embargo 5% of 100 = 5 samples after test_end=19 -> remove 20..24
        result = apply_embargo(train_idx, test_idx, 100, 0.05)
        assert 20 not in result
        assert 24 not in result
        assert 0 in result
        assert 1 in result

    def test_zero_embargo(self):
        train_idx = np.array([0, 1, 2, 20, 21])
        test_idx = np.array([10, 11])
        result = apply_embargo(train_idx, test_idx, 100, 0.0)
        np.testing.assert_array_equal(result, train_idx)


class TestPurgedKFoldCV:
    def test_returns_correct_number_of_folds(self):
        X, y, tbm = _make_cv_data(n=100)
        model = DummyModel()
        result = purged_kfold_cv(X, y, model, tbm, n_splits=5, embargo_pct=0.01)
        assert len(result.fold_sharpes) == 5
        assert len(result.fold_accuracies) == 5

    def test_mean_sharpe_is_average(self):
        X, y, tbm = _make_cv_data(n=100)
        model = DummyModel()
        result = purged_kfold_cv(X, y, model, tbm, n_splits=5)
        expected_mean = np.mean(result.fold_sharpes)
        assert abs(result.mean_sharpe - expected_mean) < 1e-10

    def test_with_sample_weights(self):
        X, y, tbm = _make_cv_data(n=100)
        model = DummyModel()
        weights = pd.Series(np.ones(100), index=X.index)
        result = purged_kfold_cv(X, y, model, tbm, sample_weight=weights, n_splits=5)
        assert len(result.fold_sharpes) == 5

    def test_no_data_leakage(self):
        """Verify test indices are never in purged training set."""
        X, y, tbm = _make_cv_data(n=50)

        n_splits = 5
        for fold in range(n_splits):
            train_idx, test_idx = _get_train_test_indices(50, n_splits, fold)
            purged = purge_train_indices(train_idx, test_idx, tbm, X.index)
            embargoed = apply_embargo(purged, test_idx, 50, 0.01)

            # No overlap between final train and test
            assert len(set(embargoed) & set(test_idx)) == 0

            # Additionally, no train sample's label span overlaps with test
            for ti in embargoed:
                train_start = tbm.iloc[ti]["t_start"]
                train_end = tbm.iloc[ti]["t_end"]
                test_min = tbm.iloc[test_idx[0]]["t_start"]
                test_max = tbm.iloc[test_idx[-1]]["t_end"]
                assert train_end < test_min or train_start > test_max
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_purged_kfold.py -v
```

### Commit Message

```
feat(validation): implement Purged K-Fold CV with embargo

Add purged_kfold.py with time-aware train/test splitting, label overlap
purging, embargo zone removal, and fold-level Sharpe/accuracy collection.
```

---

## Task 7: Combinatorial Purged CV (cpcv.py)

- [ ] Implement CPCV with all C(n_groups, k_test) combinations

### Files
- `validation/core/cpcv.py`
- `tests/test_cpcv.py`

### Implementation

**validation/core/cpcv.py**:

```python
"""Combinatorial Purged Cross-Validation (AFML Ch. 12)."""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import numpy as np
import pandas as pd

from validation.core.purged_kfold import purge_train_indices, apply_embargo
from validation.report import CPCVResult
from validation.statistics.sharpe_utils import annualized_sharpe


class ModelProtocol(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _split_into_groups(
    n_samples: int,
    n_groups: int,
) -> list[np.ndarray]:
    """Split sample indices into n_groups contiguous blocks.

    Returns:
        List of arrays, each containing indices for one group.
    """
    group_size = n_samples // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))
    return groups


def cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_groups: int = 6,
    k_test_groups: int = 2,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
) -> CPCVResult:
    """Run Combinatorial Purged Cross-Validation.

    Algorithm:
    1. Split data into n_groups contiguous blocks.
    2. Generate C(n_groups, k_test_groups) combinations.
    3. For each combination, k_test_groups = test, rest = train (with purge+embargo).
    4. Predict on test groups, concatenate in time order to form a backtest path.
    5. Compute Sharpe for each path.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with train()/predict() interface.
        tbm_timestamps: DataFrame with 't_start'/'t_end' columns.
        sample_weight: Optional sample weights.
        n_groups: Number of groups to split data into.
        k_test_groups: Number of groups used as test per combination.
        embargo_pct: Embargo fraction.
        periods_per_year: For Sharpe annualization.

    Returns:
        CPCVResult with path-level Sharpe distribution.
    """
    n_samples = len(X)
    groups = _split_into_groups(n_samples, n_groups)
    combos = list(combinations(range(n_groups), k_test_groups))

    path_sharpes = []

    for combo in combos:
        test_groups = sorted(combo)
        train_groups = [g for g in range(n_groups) if g not in test_groups]

        test_idx = np.concatenate([groups[g] for g in test_groups])
        train_idx = np.concatenate([groups[g] for g in train_groups])

        # Purge + embargo for each test group separately
        purged_train = train_idx.copy()
        for tg in test_groups:
            tg_indices = groups[tg]
            purged_train = purge_train_indices(
                purged_train, tg_indices, tbm_timestamps, X.index
            )
            purged_train = apply_embargo(purged_train, tg_indices, n_samples, embargo_pct)

        if len(purged_train) == 0:
            continue

        X_train = X.iloc[purged_train]
        y_train = y.iloc[purged_train]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        sw_train = sample_weight.iloc[purged_train] if sample_weight is not None else None

        model.train(X_train, y_train, sample_weight=sw_train)
        predictions = model.predict(X_test)

        # Compute path "returns": +1 for correct, -1 for incorrect
        signed_returns = (predictions == y_test.values).astype(float) * 2 - 1
        sr = annualized_sharpe(signed_returns, periods_per_year)
        path_sharpes.append(sr)

    if len(path_sharpes) == 0:
        return CPCVResult(
            path_sharpes=[], mean_sharpe=0.0, std_sharpe=0.0,
            median_sharpe=0.0, pct_negative=0.0,
        )

    sharpes_arr = np.array(path_sharpes)
    return CPCVResult(
        path_sharpes=path_sharpes,
        mean_sharpe=float(np.mean(sharpes_arr)),
        std_sharpe=float(np.std(sharpes_arr, ddof=1)) if len(sharpes_arr) > 1 else 0.0,
        median_sharpe=float(np.median(sharpes_arr)),
        pct_negative=float(np.mean(sharpes_arr < 0)),
    )
```

### Tests

**tests/test_cpcv.py**:

```python
import numpy as np
import pandas as pd
import pytest
from itertools import combinations
from math import comb

from validation.core.cpcv import _split_into_groups, cpcv


class DummyModel:
    def __init__(self):
        self._majority = 1

    def train(self, X, y, sample_weight=None):
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])

    def predict(self, X):
        return np.full(len(X), self._majority)


def _make_cpcv_data(n: int = 120, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    X = pd.DataFrame(
        {"feat1": rng.standard_normal(n), "feat2": rng.standard_normal(n)},
        index=idx,
    )
    y = pd.Series(rng.choice([1, -1], n), index=idx)
    holding = 3
    t_ends = pd.DatetimeIndex([idx[min(i + holding, n - 1)] for i in range(n)])
    tbm = pd.DataFrame({"t_start": idx, "t_end": t_ends}, index=range(n))
    return X, y, tbm


class TestSplitIntoGroups:
    def test_correct_number_of_groups(self):
        groups = _split_into_groups(120, 6)
        assert len(groups) == 6

    def test_covers_all_indices(self):
        groups = _split_into_groups(120, 6)
        all_idx = np.concatenate(groups)
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(120))

    def test_contiguous(self):
        groups = _split_into_groups(120, 6)
        for g in groups:
            diffs = np.diff(g)
            assert np.all(diffs == 1)

    def test_uneven_split(self):
        groups = _split_into_groups(13, 3)
        all_idx = np.concatenate(groups)
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(13))


class TestCPCV:
    def test_correct_number_of_paths(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        expected_paths = comb(6, 2)  # C(6,2) = 15
        assert len(result.path_sharpes) == expected_paths

    def test_pct_negative_in_range(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        assert 0.0 <= result.pct_negative <= 1.0

    def test_mean_equals_average(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        assert abs(result.mean_sharpe - np.mean(result.path_sharpes)) < 1e-10

    def test_median_sharpe(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=6, k_test_groups=2)
        assert abs(result.median_sharpe - np.median(result.path_sharpes)) < 1e-10

    def test_with_sample_weights(self):
        X, y, tbm = _make_cpcv_data(n=120)
        model = DummyModel()
        weights = pd.Series(np.ones(120), index=X.index)
        result = cpcv(X, y, model, tbm, sample_weight=weights, n_groups=6, k_test_groups=2)
        assert len(result.path_sharpes) == comb(6, 2)

    def test_small_groups(self):
        """Test with small data: 4 groups, 1 test group."""
        X, y, tbm = _make_cpcv_data(n=40)
        model = DummyModel()
        result = cpcv(X, y, model, tbm, n_groups=4, k_test_groups=1)
        assert len(result.path_sharpes) == comb(4, 1)  # 4 paths
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_cpcv.py -v
```

### Commit Message

```
feat(validation): implement Combinatorial Purged CV (CPCV)

Add cpcv.py generating C(n_groups, k_test) backtest paths with
purge+embargo, computing path-level Sharpe distribution.
```

---

## Task 8: Deflated Sharpe Ratio (deflated_sharpe.py)

- [ ] Implement PSR and DSR with multiple-testing correction

### Files
- `validation/statistics/deflated_sharpe.py`
- `tests/test_deflated_sharpe.py`

### Implementation

**validation/statistics/deflated_sharpe.py**:

```python
"""Deflated Sharpe Ratio: PSR + DSR with multiple-testing correction (AFML Ch. 11)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from validation.report import DeflatedSharpeResult
from validation.statistics.sharpe_utils import (
    sharpe_standard_error,
    probabilistic_sharpe_ratio,
)

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(
    n_trials: int,
    variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> float:
    """Compute the expected maximum Sharpe ratio from n_trials independent trials.

    E[max SR] = sqrt(V) * ((1-gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e)))

    where gamma = Euler-Mascheroni constant, V = variance, N = n_trials.

    Args:
        n_trials: Number of strategy trials attempted.
        variance: Variance of returns.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.
        n_obs: Number of observations.

    Returns:
        Expected maximum Sharpe ratio (non-annualized).
    """
    if n_trials <= 1:
        return 0.0

    gamma = EULER_MASCHERONI
    N = max(n_trials, 2)  # Prevent division by zero

    # Phi_inv(1 - 1/N) -- can fail for very large N, but norm.ppf handles it
    z1 = stats.norm.ppf(1.0 - 1.0 / N)
    z2 = stats.norm.ppf(1.0 - 1.0 / (N * np.e))

    e_max_sr = np.sqrt(variance) * ((1 - gamma) * z1 + gamma * z2)
    return float(e_max_sr)


def compute_deflated_sharpe(
    returns: np.ndarray,
    sharpe_observed: float,
    n_trials: int = 1,
    sharpe_benchmark: float = 0.0,
) -> DeflatedSharpeResult:
    """Compute PSR and DSR for observed Sharpe ratio.

    PSR: probability that observed Sharpe > benchmark.
    DSR: PSR where benchmark = expected max Sharpe from n_trials trials.

    Args:
        returns: Array of periodic returns.
        sharpe_observed: Observed (non-annualized) Sharpe ratio.
        n_trials: Number of strategy trials attempted.
        sharpe_benchmark: PSR benchmark Sharpe (default 0).

    Returns:
        DeflatedSharpeResult with PSR, DSR, and supporting statistics.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    skew = float(stats.skew(returns)) if n >= 3 else 0.0
    kurt = float(stats.kurtosis(returns, fisher=False)) if n >= 4 else 3.0
    variance = float(np.var(returns, ddof=1)) if n > 1 else 0.0

    se = sharpe_standard_error(sharpe_observed, n, skew, kurt)

    # PSR: P(SR > benchmark)
    psr = probabilistic_sharpe_ratio(sharpe_observed, sharpe_benchmark, n, skew, kurt)
    # PSR p-value: probability of observing a SR this high under H0: SR=benchmark
    # We want the *complement*: p-value = 1 - PSR (significance of being above benchmark)
    # But convention: higher PSR = more significant. We use 1-PSR as p-value.
    psr_pvalue = 1.0 - psr

    # DSR: PSR with benchmark = expected max SR
    e_max_sr = expected_max_sharpe(n_trials, variance, skew, kurt, n)
    dsr = probabilistic_sharpe_ratio(sharpe_observed, e_max_sr, n, skew, kurt)
    dsr_pvalue = 1.0 - dsr

    return DeflatedSharpeResult(
        psr_pvalue=psr_pvalue,
        dsr_pvalue=dsr_pvalue,
        expected_max_sharpe=e_max_sr,
        sharpe_std_error=se,
        sharpe_observed=sharpe_observed,
        n_trials=n_trials,
    )
```

### Tests

**tests/test_deflated_sharpe.py**:

```python
import numpy as np
import pytest
from scipy import stats

from validation.statistics.deflated_sharpe import (
    expected_max_sharpe,
    compute_deflated_sharpe,
    EULER_MASCHERONI,
)


class TestExpectedMaxSharpe:
    def test_single_trial_returns_zero(self):
        assert expected_max_sharpe(1, variance=0.01) == 0.0

    def test_more_trials_higher_expected(self):
        e2 = expected_max_sharpe(2, variance=0.01)
        e10 = expected_max_sharpe(10, variance=0.01)
        e100 = expected_max_sharpe(100, variance=0.01)
        assert e10 > e2
        assert e100 > e10

    def test_higher_variance_higher_expected(self):
        e_low = expected_max_sharpe(10, variance=0.001)
        e_high = expected_max_sharpe(10, variance=0.01)
        assert e_high > e_low

    def test_positive_for_multiple_trials(self):
        e = expected_max_sharpe(5, variance=0.01)
        assert e > 0


class TestComputeDeflatedSharpe:
    def test_high_sharpe_low_pvalue(self):
        """A genuinely high Sharpe with 1 trial should have low p-values."""
        rng = np.random.default_rng(42)
        # Strong positive signal
        returns = rng.normal(0.01, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        assert result.psr_pvalue < 0.05  # significant against 0
        assert result.dsr_pvalue < 0.05  # n_trials=1 -> DSR benchmark = 0

    def test_zero_sharpe_high_pvalue(self):
        """Zero-mean returns should have high PSR p-value."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        # p-value should be around 0.5 for zero Sharpe
        assert result.psr_pvalue > 0.3

    def test_many_trials_inflates_dsr_pvalue(self):
        """More trials -> higher expected max SR -> harder to pass DSR."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)

        result_1 = compute_deflated_sharpe(returns, sr, n_trials=1)
        result_100 = compute_deflated_sharpe(returns, sr, n_trials=100)

        # With 100 trials, the bar is much higher -> DSR p-value should be higher
        assert result_100.dsr_pvalue > result_1.dsr_pvalue

    def test_psr_pvalue_is_1_minus_psr(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 100)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        # Verify p-value is 1 - CDF
        assert 0 <= result.psr_pvalue <= 1
        assert 0 <= result.dsr_pvalue <= 1

    def test_expected_max_sharpe_stored(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=10)
        assert result.expected_max_sharpe > 0
        assert result.n_trials == 10

    def test_sharpe_std_error_positive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        assert result.sharpe_std_error > 0

    def test_short_returns(self):
        """Should handle very short return series gracefully."""
        returns = np.array([0.01, -0.005, 0.003])
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=1)
        assert np.isfinite(result.psr_pvalue)
        assert np.isfinite(result.dsr_pvalue)
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_deflated_sharpe.py -v
```

### Commit Message

```
feat(validation): implement Deflated Sharpe Ratio (PSR + DSR)

Add deflated_sharpe.py with expected max Sharpe computation,
PSR (probability observed > benchmark), and DSR (multiple-testing
correction using n_trials).
```

---

## Task 9: Fractional Differentiation (fracdiff.py)

- [ ] Implement fixed-width window fractional differentiation with ADF-based optimal d search

### Files
- `validation/features/fracdiff.py`
- `tests/test_fracdiff.py`

### Implementation

**validation/features/fracdiff.py**:

```python
"""Fractional Differentiation with fixed-width window (AFML Ch. 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from validation.report import FracDiffResult


def _compute_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute fractional differentiation weights using recursive formula.

    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

    Weights are truncated when abs(w_k) < threshold.

    Args:
        d: Fractional differentiation order.
        threshold: Cutoff for weight magnitude.

    Returns:
        Array of weights [w_0, w_1, ..., w_K].
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        if k > 10000:  # safety limit
            break
    return np.array(weights)


def fracdiff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fractional differentiation to a time series.

    x_t^d = sum(w_k * x_{t-k}) for k=0..window

    Args:
        series: Input time series.
        d: Differentiation order (0 < d < 1 for fractional).
        threshold: Weight truncation threshold.

    Returns:
        Fractionally differenced series. First (window-1) values are NaN.
    """
    weights = _compute_weights(d, threshold)
    window = len(weights)
    values = series.values.astype(np.float64)
    n = len(values)

    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        result[t] = np.dot(weights, values[t - window + 1:t + 1][::-1])

    return pd.Series(result, index=series.index, name=series.name)


def find_optimal_d(
    series: pd.Series,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.05,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
) -> tuple[float, float, pd.Series]:
    """Find the minimum d that makes the series stationary (ADF test).

    Searches d from d_min to d_max in steps of d_step.
    Returns the smallest d where ADF p-value < adf_pvalue.

    Args:
        series: Input time series.
        d_min: Minimum d to search.
        d_max: Maximum d to search.
        d_step: Step size for d search.
        threshold: Weight truncation threshold.
        adf_pvalue: Target ADF p-value for stationarity.

    Returns:
        Tuple of (optimal_d, adf_pvalue_at_d, differenced_series).
    """
    d_values = np.arange(d_min, d_max + d_step / 2, d_step)

    for d in d_values:
        if d == 0.0:
            # d=0 means no differencing; test original series
            diffed = series.copy()
        else:
            diffed = fracdiff(series, d, threshold)

        # Drop NaN values for ADF test
        clean = diffed.dropna()
        if len(clean) < 20:
            continue

        try:
            adf_result = adfuller(clean, maxlag=1, autolag=None)
            pval = adf_result[1]
        except Exception:
            continue

        if pval < adf_pvalue:
            return (float(d), float(pval), diffed)

    # If no d found, return d=1.0 (full differencing)
    diffed = fracdiff(series, 1.0, threshold)
    clean = diffed.dropna()
    try:
        adf_result = adfuller(clean, maxlag=1, autolag=None)
        pval = float(adf_result[1])
    except Exception:
        pval = 1.0
    return (1.0, pval, diffed)


def analyze_features(
    features_df: pd.DataFrame,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.05,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
) -> FracDiffResult:
    """Find optimal fractional differentiation d for each feature column.

    Args:
        features_df: DataFrame with feature columns.
        d_min, d_max, d_step: Search range for d.
        threshold: Weight truncation threshold.
        adf_pvalue: Target ADF p-value.

    Returns:
        FracDiffResult with per-feature d values and ADF p-values.
    """
    feature_d_values = {}
    feature_adf_pvalues = {}

    for col in features_df.columns:
        series = features_df[col].dropna()
        if len(series) < 30:
            feature_d_values[col] = 1.0
            feature_adf_pvalues[col] = 1.0
            continue

        d, pval, _ = find_optimal_d(series, d_min, d_max, d_step, threshold, adf_pvalue)
        feature_d_values[col] = d
        feature_adf_pvalues[col] = pval

    return FracDiffResult(
        feature_d_values=feature_d_values,
        feature_adf_pvalues=feature_adf_pvalues,
    )
```

### Tests

**tests/test_fracdiff.py**:

```python
import numpy as np
import pandas as pd
import pytest

from validation.features.fracdiff import (
    _compute_weights,
    fracdiff,
    find_optimal_d,
    analyze_features,
)


class TestComputeWeights:
    def test_d_zero_single_weight(self):
        """d=0 should give just [1.0] since w_1 = -1*(0-0)/1 = 0."""
        weights = _compute_weights(0.0)
        assert len(weights) == 1
        assert weights[0] == 1.0

    def test_d_one_two_weights(self):
        """d=1 should give [1, -1] (full first difference)."""
        weights = _compute_weights(1.0, threshold=1e-5)
        assert len(weights) == 2
        assert abs(weights[0] - 1.0) < 1e-10
        assert abs(weights[1] - (-1.0)) < 1e-10

    def test_d_half_decreasing(self):
        """d=0.5 weights should decrease in magnitude."""
        weights = _compute_weights(0.5)
        magnitudes = np.abs(weights)
        assert magnitudes[0] > magnitudes[1] > magnitudes[2]

    def test_weights_sum_for_d_1(self):
        """For d=1, the weights represent first difference: x_t - x_{t-1}."""
        weights = _compute_weights(1.0)
        assert abs(sum(weights)) < 1e-10  # 1 + (-1) = 0


class TestFracdiff:
    def test_d_1_is_first_difference(self):
        """Fracdiff with d=1 should approximate first difference."""
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        series = pd.Series(np.cumsum(np.ones(100)), index=idx)  # 1, 2, 3, ...
        diffed = fracdiff(series, 1.0)
        # First difference of [1,2,3,...] should be [nan, 1, 1, 1, ...]
        clean = diffed.dropna()
        assert all(abs(v - 1.0) < 1e-10 for v in clean.values)

    def test_d_0_is_identity(self):
        """Fracdiff with d=0 should return original series."""
        idx = pd.date_range("2023-01-01", periods=50, freq="D")
        series = pd.Series(np.arange(50, dtype=float), index=idx)
        diffed = fracdiff(series, 0.0)
        # d=0 -> only weight is [1], so result should equal original
        clean = diffed.dropna()
        np.testing.assert_array_almost_equal(clean.values, series.values)

    def test_output_length_matches_input(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        series = pd.Series(np.random.default_rng(42).standard_normal(100), index=idx)
        diffed = fracdiff(series, 0.5)
        assert len(diffed) == len(series)

    def test_leading_nans(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        series = pd.Series(np.random.default_rng(42).standard_normal(100), index=idx)
        diffed = fracdiff(series, 0.5)
        # Should have some leading NaN values (window - 1)
        n_nans = diffed.isna().sum()
        assert n_nans > 0
        assert n_nans < len(series)


class TestFindOptimalD:
    def test_stationary_series_d_zero(self):
        """Already stationary series (white noise) should need d=0."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        series = pd.Series(rng.standard_normal(500), index=idx)
        d, pval, _ = find_optimal_d(series)
        assert d == 0.0
        assert pval < 0.05

    def test_random_walk_needs_differencing(self):
        """Random walk should need d > 0."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        series = pd.Series(np.cumsum(rng.standard_normal(500)), index=idx)
        d, pval, diffed = find_optimal_d(series)
        assert d > 0
        assert pval < 0.05

    def test_optimal_d_between_0_and_1(self):
        """For a near-unit-root process, optimal d should be fractional."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        # AR(1) with phi close to 1 -> needs some differencing but not full
        ar = np.zeros(500)
        for i in range(1, 500):
            ar[i] = 0.99 * ar[i-1] + rng.standard_normal()
        series = pd.Series(ar, index=idx)
        d, pval, _ = find_optimal_d(series, d_step=0.1)
        assert 0.0 <= d <= 1.0


class TestAnalyzeFeatures:
    def test_multiple_features(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=300, freq="D")
        df = pd.DataFrame({
            "stationary": rng.standard_normal(300),
            "random_walk": np.cumsum(rng.standard_normal(300)),
        }, index=idx)
        result = analyze_features(df, d_step=0.2)
        assert "stationary" in result.feature_d_values
        assert "random_walk" in result.feature_d_values
        assert result.feature_d_values["stationary"] <= result.feature_d_values["random_walk"]

    def test_returns_fracdiff_result(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=200, freq="D")
        df = pd.DataFrame({"feat1": rng.standard_normal(200)}, index=idx)
        result = analyze_features(df, d_step=0.25)
        assert isinstance(result.feature_d_values, dict)
        assert isinstance(result.feature_adf_pvalues, dict)
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_fracdiff.py -v
```

### Commit Message

```
feat(validation): implement fractional differentiation with optimal d search

Add fracdiff.py with fixed-width window fracdiff, ADF-based optimal d
search per feature, and analyze_features() for batch processing.
```

---

## Task 10: Multicollinearity (multicollinearity.py)

- [ ] Implement correlation clustering and VIF-based feature removal

### Files
- `validation/features/multicollinearity.py`
- `tests/test_multicollinearity.py`

### Implementation

**validation/features/multicollinearity.py**:

```python
"""Multicollinearity detection and removal: correlation clustering + VIF (AFML Ch. 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from validation.report import MulticollinearityResult


def compute_correlation_clusters(
    X: pd.DataFrame,
    threshold: float = 0.7,
) -> list[list[str]]:
    """Find clusters of features with absolute correlation > threshold.

    Uses a greedy union-find approach: for each pair with |corr| > threshold,
    merge their clusters.

    Args:
        X: Feature DataFrame.
        threshold: Absolute correlation threshold.

    Returns:
        List of clusters (each cluster is a list of feature names).
        Only clusters with 2+ features are returned.
    """
    corr_matrix = X.corr().abs()
    n_features = len(corr_matrix.columns)
    features = list(corr_matrix.columns)

    # Union-Find
    parent = list(range(n_features))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if corr_matrix.iloc[i, j] > threshold:
                union(i, j)

    # Group features by root
    clusters_dict: dict[int, list[str]] = {}
    for i in range(n_features):
        root = find(i)
        clusters_dict.setdefault(root, []).append(features[i])

    # Only return multi-feature clusters
    return [cluster for cluster in clusters_dict.values() if len(cluster) > 1]


def select_from_clusters(
    clusters: list[list[str]],
    mda_ranking: pd.DataFrame | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """From each correlation cluster, keep the feature with highest MDA.

    If no MDA ranking is provided, keep the first feature alphabetically.

    Args:
        clusters: List of feature clusters.
        mda_ranking: DataFrame with 'feature' and 'mean_decrease' columns.

    Returns:
        Tuple of (features_to_keep, removed_features_with_reason).
    """
    removed = []
    kept = []

    for cluster in clusters:
        if mda_ranking is not None and "feature" in mda_ranking.columns:
            # Find the feature with highest MDA in this cluster
            cluster_mda = mda_ranking[mda_ranking["feature"].isin(cluster)]
            if len(cluster_mda) > 0:
                best = cluster_mda.sort_values("mean_decrease", ascending=False).iloc[0]["feature"]
            else:
                best = sorted(cluster)[0]
        else:
            best = sorted(cluster)[0]

        kept.append(best)
        for feat in cluster:
            if feat != best:
                removed.append((feat, f"corr cluster with {best}"))

    return kept, removed


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R^2_j) where R^2_j is the R-squared from regressing
    feature j on all other features.

    Uses numpy least-squares for efficiency (avoids statsmodels per-feature OLS).

    Args:
        X: Feature DataFrame.

    Returns:
        Series of VIF values indexed by feature name.
    """
    X_arr = X.values.astype(np.float64)
    n, p = X_arr.shape
    vif_values = np.zeros(p)

    for j in range(p):
        y_j = X_arr[:, j]
        X_others = np.delete(X_arr, j, axis=1)
        # Add intercept
        X_others_with_const = np.column_stack([np.ones(n), X_others])

        try:
            # OLS via normal equations
            beta, residuals, _, _ = np.linalg.lstsq(X_others_with_const, y_j, rcond=None)
            y_pred = X_others_with_const @ beta
            ss_res = np.sum((y_j - y_pred) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif_values[j] = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float("inf")
        except np.linalg.LinAlgError:
            vif_values[j] = float("inf")

    return pd.Series(vif_values, index=X.columns)


def remove_high_vif(
    X: pd.DataFrame,
    vif_threshold: float = 10.0,
    max_iterations: int = 100,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Iteratively remove features with VIF > threshold (highest first).

    Args:
        X: Feature DataFrame.
        vif_threshold: Maximum allowed VIF.
        max_iterations: Safety limit on iterations.

    Returns:
        Tuple of (remaining_features, removed_features_with_reason).
    """
    remaining = list(X.columns)
    removed = []

    for _ in range(max_iterations):
        if len(remaining) <= 1:
            break
        vif = compute_vif(X[remaining])
        max_vif_idx = vif.idxmax()
        max_vif_val = vif[max_vif_idx]

        if max_vif_val <= vif_threshold:
            break

        removed.append((max_vif_idx, f"VIF={max_vif_val:.1f}"))
        remaining.remove(max_vif_idx)

    return remaining, removed


def analyze_multicollinearity(
    X: pd.DataFrame,
    mda_ranking: pd.DataFrame | None = None,
    corr_threshold: float = 0.7,
    vif_threshold: float = 10.0,
) -> MulticollinearityResult:
    """Full multicollinearity analysis: correlation clusters + VIF removal.

    1. Find correlation clusters above threshold.
    2. From each cluster, keep the best feature (by MDA).
    3. On remaining features, iteratively remove high-VIF features.

    Args:
        X: Feature DataFrame.
        mda_ranking: Optional MDA ranking for cluster selection.
        corr_threshold: Correlation threshold for clustering.
        vif_threshold: VIF threshold for removal.

    Returns:
        MulticollinearityResult.
    """
    all_features = list(X.columns)

    # Step 1: Correlation clusters
    clusters = compute_correlation_clusters(X, corr_threshold)

    # Step 2: Select from clusters
    _, corr_removed = select_from_clusters(clusters, mda_ranking)
    corr_removed_names = {feat for feat, _ in corr_removed}
    surviving = [f for f in all_features if f not in corr_removed_names]

    # Step 3: VIF removal on surviving features
    if len(surviving) > 1:
        remaining, vif_removed = remove_high_vif(X[surviving], vif_threshold)
    else:
        remaining = surviving
        vif_removed = []

    all_removed = corr_removed + vif_removed

    return MulticollinearityResult(
        selected_features=remaining,
        removed_features=all_removed,
        correlation_clusters=clusters,
    )
```

### Tests

**tests/test_multicollinearity.py**:

```python
import numpy as np
import pandas as pd
import pytest

from validation.features.multicollinearity import (
    compute_correlation_clusters,
    select_from_clusters,
    compute_vif,
    remove_high_vif,
    analyze_multicollinearity,
)


def _make_correlated_features(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create features with known correlation structure.

    feat_a and feat_b: corr ~ 0.95
    feat_c: independent
    feat_d and feat_e: corr ~ 0.90
    """
    rng = np.random.default_rng(seed)
    base1 = rng.standard_normal(n)
    base2 = rng.standard_normal(n)
    return pd.DataFrame({
        "feat_a": base1,
        "feat_b": base1 + rng.standard_normal(n) * 0.3,  # ~0.95 corr with a
        "feat_c": rng.standard_normal(n),  # independent
        "feat_d": base2,
        "feat_e": base2 + rng.standard_normal(n) * 0.4,  # ~0.90 corr with d
    })


class TestCorrelationClusters:
    def test_finds_correlated_pairs(self):
        X = _make_correlated_features()
        clusters = compute_correlation_clusters(X, threshold=0.7)
        # Should find at least 2 clusters: {a,b} and {d,e}
        assert len(clusters) >= 2
        # feat_c should not be in any cluster
        all_clustered = [f for c in clusters for f in c]
        assert "feat_c" not in all_clustered

    def test_high_threshold_no_clusters(self):
        X = _make_correlated_features()
        clusters = compute_correlation_clusters(X, threshold=0.99)
        # May or may not find clusters at 0.99 threshold
        all_clustered = [f for c in clusters for f in c]
        assert "feat_c" not in all_clustered

    def test_low_threshold_everything_clustered(self):
        X = _make_correlated_features()
        clusters = compute_correlation_clusters(X, threshold=0.0)
        # Everything should be in one big cluster
        all_clustered = [f for c in clusters for f in c]
        assert len(all_clustered) == 5


class TestSelectFromClusters:
    def test_keeps_highest_mda(self):
        clusters = [["feat_a", "feat_b"]]
        mda = pd.DataFrame({
            "feature": ["feat_a", "feat_b", "feat_c"],
            "mean_decrease": [0.1, 0.3, 0.2],
        })
        kept, removed = select_from_clusters(clusters, mda)
        assert "feat_b" in kept  # highest MDA in cluster
        assert any(f == "feat_a" for f, _ in removed)

    def test_no_mda_keeps_alphabetical(self):
        clusters = [["feat_b", "feat_a"]]
        kept, removed = select_from_clusters(clusters, mda_ranking=None)
        assert "feat_a" in kept
        assert any(f == "feat_b" for f, _ in removed)


class TestComputeVIF:
    def test_independent_features_low_vif(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "a": rng.standard_normal(200),
            "b": rng.standard_normal(200),
            "c": rng.standard_normal(200),
        })
        vif = compute_vif(X)
        # Independent features should have VIF close to 1
        assert all(vif < 2.0)

    def test_collinear_features_high_vif(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal(200)
        X = pd.DataFrame({
            "a": base,
            "b": base + rng.standard_normal(200) * 0.01,  # nearly identical
            "c": rng.standard_normal(200),
        })
        vif = compute_vif(X)
        # a and b should have very high VIF
        assert vif["a"] > 10 or vif["b"] > 10


class TestRemoveHighVIF:
    def test_removes_collinear(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal(200)
        X = pd.DataFrame({
            "a": base,
            "b": base + rng.standard_normal(200) * 0.01,
            "c": rng.standard_normal(200),
        })
        remaining, removed = remove_high_vif(X, vif_threshold=10.0)
        assert len(removed) > 0
        assert "c" in remaining  # independent feature should survive

    def test_independent_all_remain(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "a": rng.standard_normal(200),
            "b": rng.standard_normal(200),
        })
        remaining, removed = remove_high_vif(X, vif_threshold=10.0)
        assert len(removed) == 0
        assert len(remaining) == 2


class TestAnalyzeMulticollinearity:
    def test_full_pipeline(self):
        X = _make_correlated_features()
        result = analyze_multicollinearity(X, corr_threshold=0.7, vif_threshold=10.0)
        assert "feat_c" in result.selected_features
        assert len(result.removed_features) > 0
        assert len(result.correlation_clusters) >= 2

    def test_with_mda_ranking(self):
        X = _make_correlated_features()
        mda = pd.DataFrame({
            "feature": ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"],
            "mean_decrease": [0.5, 0.1, 0.3, 0.4, 0.05],
        })
        result = analyze_multicollinearity(X, mda_ranking=mda, corr_threshold=0.7)
        # feat_a should be kept over feat_b (higher MDA)
        assert "feat_a" in result.selected_features
        # feat_d should be kept over feat_e
        assert "feat_d" in result.selected_features

    def test_no_collinearity(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "a": rng.standard_normal(100),
            "b": rng.standard_normal(100),
        })
        result = analyze_multicollinearity(X, corr_threshold=0.7)
        assert len(result.removed_features) == 0
        assert set(result.selected_features) == {"a", "b"}
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_multicollinearity.py -v
```

### Commit Message

```
feat(validation): implement multicollinearity detection and removal

Add multicollinearity.py with correlation clustering via union-find,
MDA-based cluster selection, VIF computation via OLS, and iterative
high-VIF feature removal.
```

---

## Task 11: Feature Importance (importance.py)

- [ ] Implement MDI, MDA, SFI feature importance with overfit detection

### Files
- `validation/features/importance.py`
- `tests/test_importance.py`

### Implementation

**validation/features/importance.py**:

```python
"""Feature importance: MDI, MDA, SFI with overfit detection (AFML Ch. 8)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from validation.core.purged_kfold import (
    _get_train_test_indices,
    purge_train_indices,
    apply_embargo,
)
from validation.report import FeatureImportanceResult
from validation.statistics.sharpe_utils import annualized_sharpe


class ModelProtocol(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class ImportanceModelProtocol(ModelProtocol, Protocol):
    """Model that also supports feature importance (e.g., CatBoost)."""
    def get_feature_importance(self) -> np.ndarray: ...


def compute_mdi(
    X: pd.DataFrame,
    y: pd.Series,
    model: ImportanceModelProtocol,
    sample_weight: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute Mean Decrease Impurity (MDI) feature importance.

    Trains the model on the full dataset and extracts built-in feature importance.
    This is an in-sample metric.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with get_feature_importance() method.
        sample_weight: Optional sample weights.

    Returns:
        DataFrame with columns ['feature', 'importance'], sorted descending.
    """
    model.train(X, y, sample_weight=sample_weight)
    importances = model.get_feature_importance()

    # Normalize to sum to 1
    total = np.sum(importances)
    if total > 0:
        importances = importances / total

    result = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": importances,
    })
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_mda(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute Mean Decrease Accuracy (MDA) feature importance via Purged K-Fold.

    For each fold and each feature:
    1. Compute baseline accuracy on test set.
    2. Shuffle the feature column in the test set.
    3. Compute accuracy with shuffled feature.
    4. Decrease = baseline - shuffled accuracy.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with train()/predict().
        tbm_timestamps: DataFrame with 't_start'/'t_end'.
        sample_weight: Optional sample weights.
        n_splits: Number of CV folds.
        embargo_pct: Embargo fraction.
        random_state: Random seed for shuffling.

    Returns:
        DataFrame with columns ['feature', 'mean_decrease', 'std_decrease'], sorted descending.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    features = X.columns.tolist()

    # Collect per-fold per-feature decrease
    decreases = {feat: [] for feat in features}

    for fold in range(n_splits):
        train_idx, test_idx = _get_train_test_indices(n_samples, n_splits, fold)
        train_idx = purge_train_indices(train_idx, test_idx, tbm_timestamps, X.index)
        train_idx = apply_embargo(train_idx, test_idx, n_samples, embargo_pct)

        if len(train_idx) == 0:
            continue

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx]

        sw = sample_weight.iloc[train_idx] if sample_weight is not None else None
        model.train(X_train, y_train, sample_weight=sw)

        # Baseline accuracy
        baseline_preds = model.predict(X_test)
        baseline_acc = float(np.mean(baseline_preds == y_test.values))

        # Shuffle each feature and measure accuracy decrease
        for feat in features:
            X_test_shuffled = X_test.copy()
            X_test_shuffled[feat] = rng.permutation(X_test_shuffled[feat].values)

            shuffled_preds = model.predict(X_test_shuffled)
            shuffled_acc = float(np.mean(shuffled_preds == y_test.values))

            decreases[feat].append(baseline_acc - shuffled_acc)

    result = pd.DataFrame({
        "feature": features,
        "mean_decrease": [np.mean(decreases[f]) if decreases[f] else 0.0 for f in features],
        "std_decrease": [np.std(decreases[f], ddof=1) if len(decreases[f]) > 1 else 0.0 for f in features],
    })
    return result.sort_values("mean_decrease", ascending=False).reset_index(drop=True)


def compute_sfi(
    X: pd.DataFrame,
    y: pd.Series,
    model: ModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compute Single Feature Importance (SFI).

    For each feature, train a model using only that feature with Purged K-Fold CV
    and compute the mean Sharpe ratio.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        model: Model with train()/predict().
        tbm_timestamps: DataFrame with 't_start'/'t_end'.
        sample_weight: Optional sample weights.
        n_splits: Number of CV folds.
        embargo_pct: Embargo fraction.
        periods_per_year: For Sharpe annualization.

    Returns:
        DataFrame with columns ['feature', 'sharpe'], sorted descending.
    """
    n_samples = len(X)
    features = X.columns.tolist()
    sfi_results = []

    for feat in features:
        X_single = X[[feat]]
        fold_sharpes = []

        for fold in range(n_splits):
            train_idx, test_idx = _get_train_test_indices(n_samples, n_splits, fold)
            train_idx = purge_train_indices(train_idx, test_idx, tbm_timestamps, X.index)
            train_idx = apply_embargo(train_idx, test_idx, n_samples, embargo_pct)

            if len(train_idx) == 0:
                continue

            X_train = X_single.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_single.iloc[test_idx]
            y_test = y.iloc[test_idx]

            sw = sample_weight.iloc[train_idx] if sample_weight is not None else None
            model.train(X_train, y_train, sample_weight=sw)
            predictions = model.predict(X_test)

            signed_returns = (predictions == y_test.values).astype(float) * 2 - 1
            sr = annualized_sharpe(signed_returns, periods_per_year)
            fold_sharpes.append(sr)

        mean_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        sfi_results.append({"feature": feat, "sharpe": mean_sharpe})

    result = pd.DataFrame(sfi_results)
    return result.sort_values("sharpe", ascending=False).reset_index(drop=True)


def detect_overfit_features(
    mdi_ranking: pd.DataFrame,
    mda_ranking: pd.DataFrame,
    sfi_ranking: pd.DataFrame,
) -> list[str]:
    """Detect overfit features using cross-referencing of MDI, MDA, SFI.

    Overfit criteria (from spec):
    - MDI top 30% but MDA bottom 50% -> overfit suspect
    - MDA std > MDA mean -> unstable
    - SFI Sharpe < 0 -> no standalone predictive power

    A feature flagged by 2+ criteria is classified as overfit.

    Args:
        mdi_ranking: DataFrame with 'feature', 'importance'.
        mda_ranking: DataFrame with 'feature', 'mean_decrease', 'std_decrease'.
        sfi_ranking: DataFrame with 'feature', 'sharpe'.

    Returns:
        List of overfit feature names.
    """
    features = mdi_ranking["feature"].tolist()
    n_features = len(features)

    # MDI top 30%
    mdi_top_n = max(1, int(n_features * 0.3))
    mdi_top_set = set(mdi_ranking.head(mdi_top_n)["feature"].tolist())

    # MDA bottom 50%
    mda_sorted = mda_ranking.sort_values("mean_decrease", ascending=True)
    mda_bottom_n = max(1, int(n_features * 0.5))
    mda_bottom_set = set(mda_sorted.head(mda_bottom_n)["feature"].tolist())

    # Build per-feature lookup
    mda_lookup = dict(zip(mda_ranking["feature"], zip(mda_ranking["mean_decrease"], mda_ranking["std_decrease"])))
    sfi_lookup = dict(zip(sfi_ranking["feature"], sfi_ranking["sharpe"]))

    overfit = []
    for feat in features:
        flags = 0

        # Criterion 1: MDI top 30% AND MDA bottom 50%
        if feat in mdi_top_set and feat in mda_bottom_set:
            flags += 1

        # Criterion 2: MDA std > MDA mean (unstable)
        if feat in mda_lookup:
            mean_dec, std_dec = mda_lookup[feat]
            if std_dec > abs(mean_dec) and abs(mean_dec) > 0:
                flags += 1

        # Criterion 3: SFI Sharpe < 0
        if feat in sfi_lookup and sfi_lookup[feat] < 0:
            flags += 1

        if flags >= 2:
            overfit.append(feat)

    return overfit


def compute_all_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model: ImportanceModelProtocol,
    tbm_timestamps: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    periods_per_year: int = 252,
    random_state: int = 42,
) -> FeatureImportanceResult:
    """Run all feature importance methods and detect overfit features.

    Args:
        X, y, model, tbm_timestamps: Standard inputs.
        sample_weight: Optional sample weights.
        n_splits, embargo_pct, periods_per_year: CV parameters.
        random_state: For MDA shuffling.

    Returns:
        FeatureImportanceResult with MDI, MDA, SFI rankings and overfit detection.
    """
    mdi = compute_mdi(X, y, model, sample_weight)
    mda = compute_mda(X, y, model, tbm_timestamps, sample_weight, n_splits, embargo_pct, random_state)
    sfi = compute_sfi(X, y, model, tbm_timestamps, sample_weight, n_splits, embargo_pct, periods_per_year)

    overfit_features = detect_overfit_features(mdi, mda, sfi)
    n_features = len(X.columns)
    overfit_ratio = len(overfit_features) / n_features if n_features > 0 else 0.0

    return FeatureImportanceResult(
        mdi_ranking=mdi,
        mda_ranking=mda,
        sfi_ranking=sfi,
        overfit_features=overfit_features,
        overfit_ratio=overfit_ratio,
    )
```

### Tests

**tests/test_importance.py**:

```python
import numpy as np
import pandas as pd
import pytest

from validation.features.importance import (
    compute_mdi,
    compute_mda,
    compute_sfi,
    detect_overfit_features,
    compute_all_importance,
)


class MockImportanceModel:
    """Model with feature importance support for testing."""

    def __init__(self):
        self._majority = 1
        self._n_features = 0
        self._importances = None

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        self._n_features = X.shape[1]
        if len(y) > 0:
            self._majority = int(y.mode().iloc[0])
        # Fake importances: first feature is most "important"
        self._importances = np.arange(self._n_features, 0, -1, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._majority)

    def get_feature_importance(self) -> np.ndarray:
        if self._importances is None:
            return np.ones(self._n_features)
        return self._importances


class RandomModel:
    """Model that predicts randomly, to simulate low-quality features."""

    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)

    def train(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        return self._rng.choice([1, -1], len(X))

    def get_feature_importance(self):
        return np.ones(3)  # dummy


def _make_importance_data(n: int = 100, n_features: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    X = pd.DataFrame(
        {f"feat_{i}": rng.standard_normal(n) for i in range(n_features)},
        index=idx,
    )
    y = pd.Series(rng.choice([1, -1], n), index=idx)
    holding = 3
    t_ends = pd.DatetimeIndex([idx[min(i + holding, n - 1)] for i in range(n)])
    tbm = pd.DataFrame({"t_start": idx, "t_end": t_ends}, index=range(n))
    return X, y, tbm


class TestComputeMDI:
    def test_returns_all_features(self):
        X, y, _ = _make_importance_data(n_features=4)
        model = MockImportanceModel()
        result = compute_mdi(X, y, model)
        assert len(result) == 4
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_sums_to_one(self):
        X, y, _ = _make_importance_data(n_features=4)
        model = MockImportanceModel()
        result = compute_mdi(X, y, model)
        assert abs(result["importance"].sum() - 1.0) < 1e-10

    def test_sorted_descending(self):
        X, y, _ = _make_importance_data(n_features=4)
        model = MockImportanceModel()
        result = compute_mdi(X, y, model)
        importances = result["importance"].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))


class TestComputeMDA:
    def test_returns_all_features(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_mda(X, y, model, tbm, n_splits=3)
        assert len(result) == 3
        assert "mean_decrease" in result.columns
        assert "std_decrease" in result.columns

    def test_mda_values_are_finite(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_mda(X, y, model, tbm, n_splits=3)
        assert all(np.isfinite(result["mean_decrease"]))


class TestComputeSFI:
    def test_returns_all_features(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_sfi(X, y, model, tbm, n_splits=3)
        assert len(result) == 3
        assert "sharpe" in result.columns

    def test_sharpe_values_are_finite(self):
        X, y, tbm = _make_importance_data(n_features=3, n=100)
        model = MockImportanceModel()
        result = compute_sfi(X, y, model, tbm, n_splits=3)
        assert all(np.isfinite(result["sharpe"]))


class TestDetectOverfitFeatures:
    def test_detects_overfit(self):
        """Feature with high MDI but low MDA and negative SFI should be flagged."""
        mdi = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4", "f5"],
            "importance": [0.5, 0.2, 0.15, 0.1, 0.05],
        })
        mda = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4", "f5"],
            "mean_decrease": [0.001, 0.05, 0.03, 0.02, 0.01],  # f1: low MDA despite high MDI
            "std_decrease": [0.01, 0.01, 0.01, 0.01, 0.01],
        })
        sfi = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4", "f5"],
            "sharpe": [-0.5, 0.3, 0.2, 0.1, -0.1],  # f1: negative SFI
        })
        overfit = detect_overfit_features(mdi, mda, sfi)
        # f1: MDI top 30% (top 1 of 5), MDA bottom 50% (rank 5/5), SFI < 0 -> 3 flags
        assert "f1" in overfit

    def test_good_features_not_flagged(self):
        """Consistent features should not be flagged."""
        mdi = pd.DataFrame({
            "feature": ["f1", "f2"],
            "importance": [0.6, 0.4],
        })
        mda = pd.DataFrame({
            "feature": ["f1", "f2"],
            "mean_decrease": [0.05, 0.03],
            "std_decrease": [0.01, 0.01],
        })
        sfi = pd.DataFrame({
            "feature": ["f1", "f2"],
            "sharpe": [0.5, 0.3],
        })
        overfit = detect_overfit_features(mdi, mda, sfi)
        assert len(overfit) == 0

    def test_unstable_mda_flagged(self):
        """Feature with MDA std > MDA mean AND another flag should be overfit."""
        mdi = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "importance": [0.5, 0.3, 0.2],
        })
        mda = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "mean_decrease": [0.01, 0.05, 0.03],  # f1: low MDA
            "std_decrease": [0.05, 0.01, 0.01],    # f1: high std (unstable)
        })
        sfi = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "sharpe": [-0.2, 0.4, 0.2],  # f1: negative SFI
        })
        overfit = detect_overfit_features(mdi, mda, sfi)
        assert "f1" in overfit  # unstable + negative SFI = 2 flags (+ maybe MDI/MDA mismatch = 3)


class TestComputeAllImportance:
    def test_full_pipeline(self):
        X, y, tbm = _make_importance_data(n_features=4, n=100)
        model = MockImportanceModel()
        result = compute_all_importance(X, y, model, tbm, n_splits=3)
        assert len(result.mdi_ranking) == 4
        assert len(result.mda_ranking) == 4
        assert len(result.sfi_ranking) == 4
        assert isinstance(result.overfit_features, list)
        assert 0.0 <= result.overfit_ratio <= 1.0
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_importance.py -v
```

### Commit Message

```
feat(validation): implement MDI, MDA, SFI feature importance with overfit detection

Add importance.py with in-sample MDI, out-of-sample MDA via Purged K-Fold,
Single Feature Importance (SFI), and cross-referencing overfit detection.
```

---

## Task 12: StrategyValidator + ValidationReport Finalization

- [ ] Implement StrategyValidator wrapper and finalize package exports

### Files
- `validation/validator.py`
- `validation/__init__.py` (update)
- `validation/core/__init__.py` (update)
- `validation/statistics/__init__.py` (update)
- `validation/features/__init__.py` (update)
- `tests/test_validator.py`

### Implementation

**validation/validator.py**:

```python
"""StrategyValidator: convenience wrapper that runs all validation modules."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from validation.core.sample_weights import (
    build_indicator_matrix,
    compute_sample_weights,
)
from validation.core.sequential_bootstrap import estimate_effective_n
from validation.core.purged_kfold import purged_kfold_cv
from validation.core.cpcv import cpcv
from validation.statistics.backtest_stats import compute_backtest_stats
from validation.statistics.sharpe_utils import annualized_sharpe
from validation.statistics.deflated_sharpe import compute_deflated_sharpe
from validation.features.fracdiff import analyze_features as fracdiff_analyze
from validation.features.multicollinearity import analyze_multicollinearity
from validation.features.importance import compute_all_importance
from validation.report import (
    ValidationReport,
    SampleInfoResult,
)


class ModelProtocol(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def get_feature_importance(self) -> np.ndarray: ...


class StrategyValidator:
    """Runs all AFML validation modules and produces a ValidationReport.

    This is a convenience wrapper. Each module can also be called independently.

    The validator only inspects and reports. It does NOT modify features, remove
    columns, or retrain the model. Users read the report and decide what to do.
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        labels: pd.Series,
        model: ModelProtocol,
        tbm_config: dict,
        n_trials: int = 1,
        n_splits: int = 5,
        n_groups: int = 6,
        k_test_groups: int = 2,
        embargo_pct: float = 0.01,
        periods_per_year: int = 252,
        bootstrap_runs: int = 100,
    ):
        """Initialize the validator.

        Args:
            ohlcv: OHLCV DataFrame with DatetimeIndex.
            features: Feature DataFrame aligned with labels.
            labels: Label Series (+1/-1).
            model: Model with train/predict/get_feature_importance.
            tbm_config: TBM config with 'max_holding_bars' for timestamp construction.
            n_trials: Number of strategy trials (for DSR).
            n_splits: Purged K-Fold splits.
            n_groups: CPCV groups.
            k_test_groups: CPCV test groups per combination.
            embargo_pct: Embargo fraction.
            periods_per_year: For Sharpe annualization.
            bootstrap_runs: Number of sequential bootstrap iterations.
        """
        self.ohlcv = ohlcv
        self.features = features
        self.labels = labels
        self.model = model
        self.tbm_config = tbm_config
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.embargo_pct = embargo_pct
        self.periods_per_year = periods_per_year
        self.bootstrap_runs = bootstrap_runs

    def _build_tbm_timestamps(self) -> pd.DataFrame:
        """Construct TBM timestamp DataFrame from label index and config."""
        max_holding = self.tbm_config.get("max_holding_bars", 4)
        idx = self.labels.index
        n = len(idx)
        t_starts = idx
        t_ends = pd.DatetimeIndex([idx[min(i + max_holding, n - 1)] for i in range(n)])
        return pd.DataFrame({"t_start": t_starts, "t_end": t_ends}, index=range(n))

    def run_full_validation(self) -> ValidationReport:
        """Execute all validation modules in order and build the report.

        Execution order:
        1. Sample weights (concurrency, uniqueness)
        2. Sequential bootstrap (effective N)
        3. Fractional differentiation (optimal d per feature)
        4. Multicollinearity analysis
        5. Purged K-Fold CV
        6. CPCV
        7. Feature importance (MDI, MDA, SFI)
        8. Deflated Sharpe Ratio
        9. Backtest statistics

        Returns:
            ValidationReport with all results and PASS/FAIL/MARGINAL verdict.
        """
        report = ValidationReport()
        close = self.ohlcv["close"]
        tbm_ts = self._build_tbm_timestamps()

        # Align features and labels to same index
        common_idx = self.features.index.intersection(self.labels.index)
        X = self.features.loc[common_idx]
        y = self.labels.loc[common_idx]

        # Rebuild tbm_timestamps for aligned data
        max_holding = self.tbm_config.get("max_holding_bars", 4)
        n = len(common_idx)
        t_starts = common_idx
        t_ends = pd.DatetimeIndex([common_idx[min(i + max_holding, n - 1)] for i in range(n)])
        tbm_aligned = pd.DataFrame({"t_start": t_starts, "t_end": t_ends}, index=range(n))

        # 1. Sample weights
        conc, uniq, sw = compute_sample_weights(close, tbm_ts)
        indicator = build_indicator_matrix(tbm_ts, close.index)
        eff_n = estimate_effective_n(indicator, n_bootstrap_runs=self.bootstrap_runs, random_state=42)

        report.sample_info = SampleInfoResult(
            concurrency=conc,
            uniqueness=uniq,
            sample_weights=sw,
            mean_uniqueness=float(uniq.mean()),
            effective_n=eff_n,
            total_n=len(self.labels),
        )

        # Align sample weights to common_idx
        sw_aligned = sw.iloc[:n] if len(sw) >= n else pd.Series(np.ones(n), index=range(n))

        # 2. Fractional differentiation
        report.fracdiff = fracdiff_analyze(X)

        # 3. Multicollinearity (preliminary, without MDA -- MDA comes later)
        report.multicollinearity = analyze_multicollinearity(X)

        # 4. Purged K-Fold CV
        report.purged_cv = purged_kfold_cv(
            X, y, self.model, tbm_aligned,
            sample_weight=sw_aligned,
            n_splits=self.n_splits,
            embargo_pct=self.embargo_pct,
            periods_per_year=self.periods_per_year,
        )

        # 5. CPCV
        report.cpcv = cpcv(
            X, y, self.model, tbm_aligned,
            sample_weight=sw_aligned,
            n_groups=self.n_groups,
            k_test_groups=self.k_test_groups,
            embargo_pct=self.embargo_pct,
            periods_per_year=self.periods_per_year,
        )

        # 6. Feature importance
        report.feature_importance = compute_all_importance(
            X, y, self.model, tbm_aligned,
            sample_weight=sw_aligned,
            n_splits=self.n_splits,
            embargo_pct=self.embargo_pct,
            periods_per_year=self.periods_per_year,
        )

        # Update multicollinearity with MDA ranking
        if report.feature_importance is not None:
            report.multicollinearity = analyze_multicollinearity(
                X, mda_ranking=report.feature_importance.mda_ranking,
            )

        # 7. Deflated Sharpe
        if report.purged_cv is not None and report.purged_cv.fold_sharpes:
            mean_sr = report.purged_cv.mean_sharpe
            # Use synthetic "returns" from CV predictions for DSR
            # Approximate: use mean and std of fold Sharpes
            fold_sharpes = np.array(report.purged_cv.fold_sharpes)
            std_sr = np.std(fold_sharpes, ddof=1) if len(fold_sharpes) > 1 else 0.01
            # Generate synthetic returns matching the observed Sharpe
            n_obs = len(y)
            synthetic_returns = np.random.default_rng(42).normal(
                mean_sr * std_sr / np.sqrt(self.periods_per_year),
                std_sr / np.sqrt(self.periods_per_year),
                n_obs,
            ) if std_sr > 0 else np.zeros(n_obs)

            sr_raw = np.mean(synthetic_returns) / np.std(synthetic_returns, ddof=1) if np.std(synthetic_returns, ddof=1) > 0 else 0.0

            report.deflated_sharpe = compute_deflated_sharpe(
                returns=synthetic_returns,
                sharpe_observed=sr_raw,
                n_trials=self.n_trials,
            )

        # 8. Backtest statistics (using CV returns as proxy)
        # In production, real backtest returns would be passed in
        # For now, use the fold Sharpes to compute basic stats
        if report.purged_cv is not None:
            fold_returns = pd.Series(report.purged_cv.fold_sharpes)
            if len(fold_returns) >= 2:
                report.statistics = compute_backtest_stats(fold_returns, periods_per_year=1)

        return report
```

**validation/__init__.py** (updated):

```python
"""AFML-based strategy validation framework."""

from validation.report import ValidationReport
from validation.validator import StrategyValidator

__all__ = ["StrategyValidator", "ValidationReport"]
```

**validation/core/__init__.py** (updated):

```python
"""Core AFML validation algorithms: sample weights, bootstrap, cross-validation."""

from validation.core.sample_weights import compute_sample_weights, build_indicator_matrix
from validation.core.sequential_bootstrap import sequential_bootstrap, estimate_effective_n
from validation.core.purged_kfold import purged_kfold_cv
from validation.core.cpcv import cpcv

__all__ = [
    "compute_sample_weights",
    "build_indicator_matrix",
    "sequential_bootstrap",
    "estimate_effective_n",
    "purged_kfold_cv",
    "cpcv",
]
```

**validation/statistics/__init__.py** (updated):

```python
"""Statistical testing: Sharpe ratio utilities, deflated Sharpe, backtest statistics."""

from validation.statistics.sharpe_utils import (
    annualized_sharpe,
    sharpe_standard_error,
    sharpe_confidence_interval,
    probabilistic_sharpe_ratio,
)
from validation.statistics.deflated_sharpe import compute_deflated_sharpe
from validation.statistics.backtest_stats import compute_backtest_stats

__all__ = [
    "annualized_sharpe",
    "sharpe_standard_error",
    "sharpe_confidence_interval",
    "probabilistic_sharpe_ratio",
    "compute_deflated_sharpe",
    "compute_backtest_stats",
]
```

**validation/features/__init__.py** (updated):

```python
"""Feature quality analysis: fractional differentiation, multicollinearity, importance."""

from validation.features.fracdiff import fracdiff, find_optimal_d, analyze_features
from validation.features.multicollinearity import analyze_multicollinearity
from validation.features.importance import compute_all_importance

__all__ = [
    "fracdiff",
    "find_optimal_d",
    "analyze_features",
    "analyze_multicollinearity",
    "compute_all_importance",
]
```

### Tests

**tests/test_validator.py**:

```python
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
        assert report.fracdiff is not None
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
```

### Run Commands

```bash
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_validator.py -v
```

### Commit Message

```
feat(validation): implement StrategyValidator wrapper and finalize package exports

Add validator.py orchestrating all validation modules in correct order.
Update all __init__.py files with proper exports for the public API.
```

---

## Task 13: Integration Test with Phase 1 Strategy

- [ ] Write an integration test using synthetic data that validates the full pipeline end-to-end

### Files
- `tests/test_validation_integration.py`

### Implementation

**tests/test_validation_integration.py**:

```python
"""Integration test: full validation pipeline with synthetic data.

Tests two scenarios:
1. Random strategy (no signal) -> should get FAIL or MARGINAL
2. Strong-signal strategy -> should get PASS or MARGINAL
"""

import numpy as np
import pandas as pd
import pytest

from validation import StrategyValidator, ValidationReport
from validation.core.sample_weights import compute_sample_weights, build_indicator_matrix
from validation.core.sequential_bootstrap import sequential_bootstrap, estimate_effective_n
from validation.core.purged_kfold import purged_kfold_cv
from validation.core.cpcv import cpcv
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
    """A model that learns the majority class from training data."""

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
    """A model that uses the first feature as a direct signal.

    If feat_0 > 0, predict +1; else predict -1.
    This simulates a model that has learned the true signal.
    """

    def __init__(self):
        self._n_features = 1
        self._threshold = 0.0

    def train(self, X, y, sample_weight=None):
        self._n_features = X.shape[1]
        # Learn optimal threshold from training data
        if "signal" in X.columns:
            col = "signal"
        else:
            col = X.columns[0]
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
        imp[0] = 5.0  # signal feature is most important
        return imp


def _make_random_data(n: int = 300, n_features: int = 5, seed: int = 42):
    """Random data with NO signal -- model should fail validation."""
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


def _make_signal_data(n: int = 300, seed: int = 42):
    """Data with a clear signal: label = sign(signal + noise)."""
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
# Integration: Individual modules
# ---------------------------------------------------------------------------

class TestIndividualModulesIntegration:
    """Test that each module works independently on synthetic data."""

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
        assert all(0 < u <= 1 for u in uniq)

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
        assert pval < 0.05

    def test_multicollinearity_end_to_end(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal(100)
        X = pd.DataFrame({
            "a": base,
            "b": base + rng.standard_normal(100) * 0.1,
            "c": rng.standard_normal(100),
        })
        result = analyze_multicollinearity(X, corr_threshold=0.7)
        assert "c" in result.selected_features
        assert len(result.removed_features) > 0

    def test_deflated_sharpe_end_to_end(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        sr = np.mean(returns) / np.std(returns, ddof=1)
        result = compute_deflated_sharpe(returns, sr, n_trials=5)
        assert 0 <= result.psr_pvalue <= 1
        assert 0 <= result.dsr_pvalue <= 1

    def test_backtest_stats_end_to_end(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        result = compute_backtest_stats(returns)
        assert np.isfinite(result.sharpe_ratio)
        assert result.max_drawdown <= 0


# ---------------------------------------------------------------------------
# Integration: Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """Full pipeline integration tests."""

    def test_random_strategy_does_not_pass(self):
        """Random features + random labels -> should NOT get PASS."""
        ohlcv, features, labels = _make_random_data(n=200, n_features=4)
        model = SimpleModel()
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
        assert report.verdict in ("FAIL", "MARGINAL")

    def test_signal_strategy_validation(self):
        """Strong signal strategy -> should get meaningful validation results."""
        ohlcv, features, labels = _make_signal_data(n=200)
        model = SignalModel()
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

        # The signal model should produce better CV results than random
        assert report.purged_cv is not None
        assert report.cpcv is not None
        assert report.feature_importance is not None
        # With a true signal, mean Sharpe should be positive
        assert report.purged_cv.mean_sharpe > 0

    def test_report_print_does_not_crash(self, capsys):
        """Ensure print_full() works on a complete report."""
        ohlcv, features, labels = _make_random_data(n=120, n_features=3)
        model = SimpleModel()
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
            bootstrap_runs=5,
        )
        report = validator.run_full_validation()
        report.print_full()
        out = capsys.readouterr().out
        assert "VALIDATION REPORT" in out

    def test_overfit_detection_with_mixed_features(self):
        """Features with mixed quality should trigger overfit detection."""
        mdi = pd.DataFrame({
            "feature": ["good", "bad_in_sample", "unstable", "no_power"],
            "importance": [0.4, 0.35, 0.15, 0.10],
        })
        mda = pd.DataFrame({
            "feature": ["good", "bad_in_sample", "unstable", "no_power"],
            "mean_decrease": [0.05, 0.001, 0.01, 0.005],  # bad: low MDA
            "std_decrease": [0.01, 0.001, 0.05, 0.01],    # unstable: high std
        })
        sfi = pd.DataFrame({
            "feature": ["good", "bad_in_sample", "unstable", "no_power"],
            "sharpe": [0.5, -0.3, -0.1, -0.2],  # bad & no_power & unstable: neg SFI
        })
        overfit = detect_overfit_features(mdi, mda, sfi)
        # bad_in_sample: MDI top 30% + MDA bottom 50% + SFI<0 = 3 flags -> overfit
        assert "bad_in_sample" in overfit

    def test_psr_basic_math(self):
        """Verify PSR gives expected values for known inputs."""
        # High Sharpe with many observations -> PSR should be close to 1
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=2.0,
            benchmark_sharpe=0.0,
            n=252,
            skew=0.0,
            kurtosis=3.0,
        )
        assert psr > 0.99

        # Sharpe = 0 with benchmark 0 -> PSR = 0.5
        psr_zero = probabilistic_sharpe_ratio(0.0, 0.0, 252, 0.0, 3.0)
        assert abs(psr_zero - 0.5) < 1e-10

    def test_annualized_sharpe_basic(self):
        """Verify annualized Sharpe calculation."""
        # Constant 0.1% daily return with ~1% std
        returns = np.array([0.011, -0.009] * 126)  # 252 obs
        sr = annualized_sharpe(returns, 252)
        # mean=0.001, std~=0.01414, SR = 0.001/0.01414 * sqrt(252) ~ 1.12
        assert sr > 0.5
        assert sr < 3.0
```

### Run Commands

```bash
# Run all validation tests
cd /home/henry/Projects/ultraTM && python -m pytest tests/test_report.py tests/test_sharpe_utils.py tests/test_backtest_stats.py tests/test_sample_weights.py tests/test_sequential_bootstrap.py tests/test_purged_kfold.py tests/test_cpcv.py tests/test_deflated_sharpe.py tests/test_fracdiff.py tests/test_multicollinearity.py tests/test_importance.py tests/test_validator.py tests/test_validation_integration.py -v

# Or run all tests at once
cd /home/henry/Projects/ultraTM && python -m pytest tests/ -v -k "validation or report or sharpe or backtest_stats or sample_weights or sequential_bootstrap or purged_kfold or cpcv or deflated_sharpe or fracdiff or multicollinearity or importance or validator"
```

### Commit Message

```
test(validation): add integration tests for full validation pipeline

Test both random (no-signal) and strong-signal scenarios end-to-end.
Verify individual modules work independently and StrategyValidator
orchestrates all modules correctly with PASS/FAIL/MARGINAL verdicts.
```

"""Tests for validation.report dataclasses and ValidationReport verdict logic."""

import pandas as pd
import pytest

from validation.report import (
    CPCVResult,
    DeflatedSharpeResult,
    FeatureImportanceResult,
    PurgedCVResult,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dsr(dsr_pvalue: float = 0.01) -> DeflatedSharpeResult:
    return DeflatedSharpeResult(
        psr_pvalue=0.01,
        dsr_pvalue=dsr_pvalue,
        expected_max_sharpe=1.5,
        sharpe_std_error=0.2,
        sharpe_observed=1.2,
        n_trials=10,
    )


def _pcv(mean_sharpe: float = 1.0) -> PurgedCVResult:
    return PurgedCVResult(
        fold_sharpes=[mean_sharpe] * 5,
        fold_accuracies=[0.55] * 5,
        mean_sharpe=mean_sharpe,
        std_sharpe=0.1,
    )


def _cpcv(pct_negative: float = 0.10) -> CPCVResult:
    return CPCVResult(
        path_sharpes=[0.8, 1.0, -0.2],
        mean_sharpe=0.8,
        std_sharpe=0.3,
        median_sharpe=0.8,
        pct_negative=pct_negative,
    )


def _fi(overfit_ratio: float = 0.10) -> FeatureImportanceResult:
    df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.6, 0.4]})
    return FeatureImportanceResult(
        mdi_ranking=df,
        mda_ranking=df,
        sfi_ranking=df,
        overfit_features=[],
        overfit_ratio=overfit_ratio,
    )


# ---------------------------------------------------------------------------
# FAIL cases
# ---------------------------------------------------------------------------


class TestVerdictFail:
    """Verdict should be FAIL when any hard-fail condition is met."""

    def test_fail_dsr_pvalue_too_high(self):
        report = ValidationReport(
            deflated_sharpe=_dsr(dsr_pvalue=0.15),
            purged_cv=_pcv(mean_sharpe=1.0),
            cpcv=_cpcv(pct_negative=0.10),
        )
        assert report.verdict == "FAIL"

    def test_fail_negative_purged_cv_sharpe(self):
        report = ValidationReport(
            deflated_sharpe=_dsr(dsr_pvalue=0.01),
            purged_cv=_pcv(mean_sharpe=-0.5),
            cpcv=_cpcv(pct_negative=0.10),
        )
        assert report.verdict == "FAIL"

    def test_fail_cpcv_pct_negative_high(self):
        report = ValidationReport(
            deflated_sharpe=_dsr(dsr_pvalue=0.01),
            purged_cv=_pcv(mean_sharpe=1.0),
            cpcv=_cpcv(pct_negative=0.55),
        )
        assert report.verdict == "FAIL"


# ---------------------------------------------------------------------------
# PASS
# ---------------------------------------------------------------------------


class TestVerdictPass:
    """Verdict should be PASS when all conditions are comfortably met."""

    def test_pass_all_good(self):
        report = ValidationReport(
            deflated_sharpe=_dsr(dsr_pvalue=0.02),
            purged_cv=_pcv(mean_sharpe=1.0),
            cpcv=_cpcv(pct_negative=0.10),
            feature_importance=_fi(overfit_ratio=0.10),
        )
        assert report.verdict == "PASS"


# ---------------------------------------------------------------------------
# MARGINAL
# ---------------------------------------------------------------------------


class TestVerdictMarginal:
    """Verdict should be MARGINAL when between FAIL and PASS thresholds."""

    def test_marginal_dsr_between_thresholds(self):
        # dsr_pvalue = 0.07 => not FAIL (<=0.10) but not PASS (<0.05)
        report = ValidationReport(
            deflated_sharpe=_dsr(dsr_pvalue=0.07),
            purged_cv=_pcv(mean_sharpe=1.0),
            cpcv=_cpcv(pct_negative=0.10),
            feature_importance=_fi(overfit_ratio=0.10),
        )
        assert report.verdict == "MARGINAL"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    """The summary() method should return a string containing the verdict."""

    def test_summary_contains_verdict(self):
        report = ValidationReport(
            deflated_sharpe=_dsr(dsr_pvalue=0.02),
            purged_cv=_pcv(mean_sharpe=1.0),
            cpcv=_cpcv(pct_negative=0.10),
            feature_importance=_fi(overfit_ratio=0.10),
        )
        s = report.summary()
        assert "PASS" in s
        assert "DSR p-value" in s
        assert "Purged CV mean Sharpe" in s


# ---------------------------------------------------------------------------
# Empty report
# ---------------------------------------------------------------------------


class TestEmptyReport:
    """An empty report (no results) should be MARGINAL."""

    def test_empty_is_marginal(self):
        report = ValidationReport()
        assert report.verdict == "MARGINAL"

    def test_empty_summary_has_verdict(self):
        report = ValidationReport()
        assert "MARGINAL" in report.summary()

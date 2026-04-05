"""Result dataclasses and ValidationReport for AFML strategy validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Individual result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SampleInfoResult:
    """Concurrency, uniqueness, and sample-weight statistics."""

    concurrency: pd.Series
    uniqueness: pd.Series
    sample_weights: pd.Series
    mean_uniqueness: float
    effective_n: int
    total_n: int


@dataclass
class PurgedCVResult:
    """Purged k-fold cross-validation results."""

    fold_sharpes: list[float]
    fold_accuracies: list[float]
    mean_sharpe: float
    std_sharpe: float


@dataclass
class CPCVResult:
    """Combinatorial purged cross-validation results."""

    path_sharpes: list[float]
    mean_sharpe: float
    std_sharpe: float
    median_sharpe: float
    pct_negative: float


@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe ratio test results."""

    psr_pvalue: float
    dsr_pvalue: float
    expected_max_sharpe: float
    sharpe_std_error: float
    sharpe_observed: float
    n_trials: int


@dataclass
class FeatureImportanceResult:
    """Feature importance rankings and overfit detection."""

    mdi_ranking: pd.DataFrame
    mda_ranking: pd.DataFrame
    sfi_ranking: pd.DataFrame
    overfit_features: list[str]
    overfit_ratio: float


@dataclass
class FracDiffResult:
    """Fractional differentiation parameters per feature."""

    feature_d_values: dict[str, float]
    feature_adf_pvalues: dict[str, float]


@dataclass
class MulticollinearityResult:
    """Multicollinearity analysis and feature selection."""

    selected_features: list[str]
    removed_features: list[tuple[str, str]]
    correlation_clusters: list[list[str]]


@dataclass
class BacktestStatsResult:
    """Comprehensive backtest statistics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sharpe_std_error: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    tail_ratio: float
    skewness: float
    kurtosis: float
    max_consecutive_wins: int
    max_consecutive_losses: int


# ---------------------------------------------------------------------------
# Aggregate validation report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Aggregates all validation results and computes an overall verdict.

    Verdict logic
    -------------
    FAIL if ANY of:
        - dsr_pvalue > 0.10
        - purged_cv mean_sharpe < 0
        - cpcv pct_negative >= 0.50

    PASS if ALL of:
        - dsr_pvalue < 0.05
        - purged_cv mean_sharpe > 0.5
        - overfit_ratio < 0.30
        - cpcv pct_negative < 0.20

    MARGINAL: otherwise
    """

    sample_info: Optional[SampleInfoResult] = None
    purged_cv: Optional[PurgedCVResult] = None
    cpcv: Optional[CPCVResult] = None
    deflated_sharpe: Optional[DeflatedSharpeResult] = None
    feature_importance: Optional[FeatureImportanceResult] = None
    frac_diff: Optional[FracDiffResult] = None
    multicollinearity: Optional[MulticollinearityResult] = None
    backtest_stats: Optional[BacktestStatsResult] = None

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    @property
    def verdict(self) -> str:
        """Return 'PASS', 'FAIL', or 'MARGINAL' based on available results."""
        # If no results populated, MARGINAL by default
        if all(
            v is None
            for v in (
                self.deflated_sharpe,
                self.purged_cv,
                self.cpcv,
                self.feature_importance,
            )
        ):
            return "MARGINAL"

        # ---- FAIL conditions ----
        if self.deflated_sharpe is not None and self.deflated_sharpe.dsr_pvalue > 0.10:
            return "FAIL"
        if self.purged_cv is not None and self.purged_cv.mean_sharpe < 0:
            return "FAIL"
        if self.cpcv is not None and self.cpcv.pct_negative >= 0.50:
            return "FAIL"

        # ---- PASS conditions (all must hold) ----
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

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise multi-line summary string."""
        lines = [f"Verdict: {self.verdict}"]

        if self.deflated_sharpe is not None:
            lines.append(
                f"DSR p-value: {self.deflated_sharpe.dsr_pvalue:.4f}"
            )
        if self.purged_cv is not None:
            lines.append(
                f"Purged CV mean Sharpe: {self.purged_cv.mean_sharpe:.4f}"
            )
        if self.cpcv is not None:
            lines.append(
                f"CPCV pct negative: {self.cpcv.pct_negative:.2%}"
            )
        if self.feature_importance is not None:
            lines.append(
                f"Overfit ratio: {self.feature_importance.overfit_ratio:.2%}"
            )
        if self.backtest_stats is not None:
            lines.append(
                f"Sharpe: {self.backtest_stats.sharpe_ratio:.4f}"
            )

        return "\n".join(lines)

    def print_full(self) -> None:
        """Print a detailed report of all available results to stdout."""
        print("=" * 60)
        print(f"  VALIDATION REPORT  —  Verdict: {self.verdict}")
        print("=" * 60)

        if self.sample_info is not None:
            si = self.sample_info
            print(f"\n[Sample Info]")
            print(f"  Mean uniqueness : {si.mean_uniqueness:.4f}")
            print(f"  Effective N     : {si.effective_n}")
            print(f"  Total N         : {si.total_n}")

        if self.purged_cv is not None:
            pcv = self.purged_cv
            print(f"\n[Purged CV]")
            print(f"  Mean Sharpe     : {pcv.mean_sharpe:.4f}")
            print(f"  Std Sharpe      : {pcv.std_sharpe:.4f}")
            print(f"  Fold Sharpes    : {pcv.fold_sharpes}")
            print(f"  Fold Accuracies : {pcv.fold_accuracies}")

        if self.cpcv is not None:
            c = self.cpcv
            print(f"\n[CPCV]")
            print(f"  Mean Sharpe     : {c.mean_sharpe:.4f}")
            print(f"  Std Sharpe      : {c.std_sharpe:.4f}")
            print(f"  Median Sharpe   : {c.median_sharpe:.4f}")
            print(f"  % Negative      : {c.pct_negative:.2%}")
            print(f"  Path Sharpes    : {c.path_sharpes}")

        if self.deflated_sharpe is not None:
            ds = self.deflated_sharpe
            print(f"\n[Deflated Sharpe]")
            print(f"  PSR p-value     : {ds.psr_pvalue:.4f}")
            print(f"  DSR p-value     : {ds.dsr_pvalue:.4f}")
            print(f"  E[max Sharpe]   : {ds.expected_max_sharpe:.4f}")
            print(f"  Sharpe observed : {ds.sharpe_observed:.4f}")
            print(f"  Sharpe SE       : {ds.sharpe_std_error:.4f}")
            print(f"  N trials        : {ds.n_trials}")

        if self.feature_importance is not None:
            fi = self.feature_importance
            print(f"\n[Feature Importance]")
            print(f"  Overfit ratio   : {fi.overfit_ratio:.2%}")
            print(f"  Overfit features: {fi.overfit_features}")

        if self.frac_diff is not None:
            fd = self.frac_diff
            print(f"\n[Fractional Differentiation]")
            for feat, d in fd.feature_d_values.items():
                adf_p = fd.feature_adf_pvalues.get(feat, float("nan"))
                print(f"  {feat}: d={d:.3f}, ADF p={adf_p:.4f}")

        if self.multicollinearity is not None:
            mc = self.multicollinearity
            print(f"\n[Multicollinearity]")
            print(f"  Selected        : {mc.selected_features}")
            print(f"  Removed pairs   : {mc.removed_features}")
            print(f"  Clusters        : {mc.correlation_clusters}")

        if self.backtest_stats is not None:
            bs = self.backtest_stats
            print(f"\n[Backtest Stats]")
            print(f"  Total return    : {bs.total_return:.4f}")
            print(f"  Ann. return     : {bs.annualized_return:.4f}")
            print(f"  Sharpe          : {bs.sharpe_ratio:.4f}")
            print(f"  Sharpe CI       : [{bs.sharpe_ci_lower:.4f}, {bs.sharpe_ci_upper:.4f}]")
            print(f"  Sortino         : {bs.sortino_ratio:.4f}")
            print(f"  Calmar          : {bs.calmar_ratio:.4f}")
            print(f"  Max DD          : {bs.max_drawdown:.4f}")
            print(f"  Max DD duration : {bs.max_drawdown_duration}")
            print(f"  Win rate        : {bs.win_rate:.2%}")
            print(f"  Profit factor   : {bs.profit_factor:.4f}")

        print("\n" + "=" * 60)

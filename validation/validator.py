"""StrategyValidator: convenience wrapper that runs all validation modules."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from validation.core.sample_weights import (
    compute_sample_weights,
)
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
        skip_sfi: bool = False,
    ):
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
        self.skip_sfi = skip_sfi

    def _build_tbm_timestamps(self) -> pd.DataFrame:
        max_holding = self.tbm_config.get("max_holding_bars", 4)
        idx = self.labels.index
        n = len(idx)
        t_starts = idx
        t_ends = pd.DatetimeIndex([idx[min(i + max_holding, n - 1)] for i in range(n)])
        return pd.DataFrame({"t_start": t_starts, "t_end": t_ends}, index=range(n))

    def run_full_validation(self) -> ValidationReport:
        """Execute all validation modules and build the report."""
        report = ValidationReport()
        close = self.ohlcv["close"]
        tbm_ts = self._build_tbm_timestamps()

        common_idx = self.features.index.intersection(self.labels.index)
        X = self.features.loc[common_idx]
        y = self.labels.loc[common_idx]

        max_holding = self.tbm_config.get("max_holding_bars", 4)
        n = len(common_idx)
        t_starts = common_idx
        t_ends = pd.DatetimeIndex([common_idx[min(i + max_holding, n - 1)] for i in range(n)])
        tbm_aligned = pd.DataFrame({"t_start": t_starts, "t_end": t_ends}, index=range(n))

        # 1. Sample weights
        print("  [1/8] Sample weights...")
        conc, uniq, sw = compute_sample_weights(close, tbm_ts)
        # Estimate effective N from mean uniqueness (avoids building huge indicator matrix)
        # effective_n ~ total_n * mean_uniqueness (AFML approximation)
        mean_uniq = float(uniq.mean())
        eff_n = int(len(self.labels) * mean_uniq)

        report.sample_info = SampleInfoResult(
            concurrency=conc,
            uniqueness=uniq,
            sample_weights=sw,
            mean_uniqueness=mean_uniq,
            effective_n=eff_n,
            total_n=len(self.labels),
        )

        sw_aligned = sw.iloc[:n] if len(sw) >= n else pd.Series(np.ones(n), index=range(n))

        # 2. Fractional differentiation
        print("  [2/8] Fractional differentiation...")
        report.frac_diff = fracdiff_analyze(X)

        # 3. Multicollinearity (preliminary)
        print("  [3/8] Multicollinearity...")
        report.multicollinearity = analyze_multicollinearity(X)

        # 4. Purged K-Fold CV
        print("  [4/8] Purged K-Fold CV...")
        report.purged_cv = purged_kfold_cv(
            X, y, self.model, tbm_aligned,
            sample_weight=sw_aligned,
            n_splits=self.n_splits,
            embargo_pct=self.embargo_pct,
            periods_per_year=self.periods_per_year,
        )

        # 5. CPCV
        print("  [5/8] Combinatorial Purged CV...")
        report.cpcv = cpcv(
            X, y, self.model, tbm_aligned,
            sample_weight=sw_aligned,
            n_groups=self.n_groups,
            k_test_groups=self.k_test_groups,
            embargo_pct=self.embargo_pct,
            periods_per_year=self.periods_per_year,
        )

        # 6. Feature importance
        print("  [6/8] Feature importance (MDI/MDA" + ("/SFI" if not self.skip_sfi else "") + ")...")
        report.feature_importance = compute_all_importance(
            X, y, self.model, tbm_aligned,
            sample_weight=sw_aligned,
            n_splits=self.n_splits,
            embargo_pct=self.embargo_pct,
            periods_per_year=self.periods_per_year,
            skip_sfi=self.skip_sfi,
        )

        # Update multicollinearity with MDA ranking
        if report.feature_importance is not None:
            print("  [7/8] Multicollinearity (with MDA)...")
            report.multicollinearity = analyze_multicollinearity(
                X, mda_ranking=report.feature_importance.mda_ranking,
            )

        # 7. Deflated Sharpe
        print("  [8/8] Deflated Sharpe...")
        if report.purged_cv is not None and report.purged_cv.fold_sharpes:
            mean_sr = report.purged_cv.mean_sharpe
            fold_sharpes = np.array(report.purged_cv.fold_sharpes)
            std_sr = np.std(fold_sharpes, ddof=1) if len(fold_sharpes) > 1 else 0.01
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

        # Backtest statistics
        if report.purged_cv is not None:
            fold_returns = pd.Series(report.purged_cv.fold_sharpes)
            if len(fold_returns) >= 2:
                report.backtest_stats = compute_backtest_stats(fold_returns, periods_per_year=1)

        return report

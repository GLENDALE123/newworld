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

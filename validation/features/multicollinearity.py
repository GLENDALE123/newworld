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

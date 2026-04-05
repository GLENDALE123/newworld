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
    """Compute average uniqueness of candidate given already-selected samples."""
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

    For large datasets (>1000 labels), falls back to fast approximation
    using pre-computed uniqueness scores.
    """
    rng = np.random.default_rng(random_state)
    all_labels = list(indicator_matrix.columns)
    if n_samples is None:
        n_samples = len(all_labels)

    # Fast path for large datasets: sample proportional to pre-computed uniqueness
    if len(all_labels) > 1000:
        return _fast_sequential_bootstrap(indicator_matrix, n_samples, rng)

    selected: list[int] = []
    for _ in range(n_samples):
        uniquenesses = np.zeros(len(all_labels))
        for j, label in enumerate(all_labels):
            uniquenesses[j] = _compute_average_uniqueness_with_selected(
                indicator_matrix, selected, label
            )

        total = uniquenesses.sum()
        if total <= 0:
            probs = np.ones(len(all_labels)) / len(all_labels)
        else:
            probs = uniquenesses / total

        chosen_idx = rng.choice(len(all_labels), p=probs)
        selected.append(all_labels[chosen_idx])

    return selected


def _fast_sequential_bootstrap(
    indicator_matrix: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    """Approximate sequential bootstrap for large datasets.

    Uses pre-computed uniqueness as sampling probabilities instead of
    recomputing after each selection. O(n) instead of O(n^2).
    """
    uniq = compute_uniqueness(indicator_matrix)
    all_labels = list(indicator_matrix.columns)

    probs = uniq.values.astype(float)
    total = probs.sum()
    if total <= 0:
        probs = np.ones(len(all_labels)) / len(all_labels)
    else:
        probs = probs / total

    selected_idx = rng.choice(len(all_labels), size=n_samples, p=probs)
    return [all_labels[i] for i in selected_idx]


def estimate_effective_n(
    indicator_matrix: pd.DataFrame,
    n_bootstrap_runs: int = 100,
    random_state: int | None = None,
) -> int:
    """Estimate effective number of independent samples.

    For large datasets (>1000), uses mean uniqueness directly:
    effective_n = total_n * mean_uniqueness

    For small datasets, uses bootstrap counting.
    """
    n_labels = len(indicator_matrix.columns)

    # Fast path: estimate from uniqueness directly
    if n_labels > 1000:
        uniq = compute_uniqueness(indicator_matrix)
        mean_uniq = float(uniq.mean())
        return max(1, int(n_labels * mean_uniq))

    # Slow path for small datasets: bootstrap counting
    rng_base = np.random.default_rng(random_state)
    unique_counts = []

    for i in range(n_bootstrap_runs):
        seed = int(rng_base.integers(0, 2**31))
        selected = sequential_bootstrap(indicator_matrix, n_samples=n_labels, random_state=seed)
        unique_counts.append(len(set(selected)))

    return int(np.mean(unique_counts))

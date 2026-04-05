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

"""
Multiple testing utilities for simulation comparisons.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def one_sample_t_pvalues(
    groups: Sequence[np.ndarray],
    reference: float = 0.0,
) -> np.ndarray:
    """
    Compute one-sample t-test p-values for each group.
    """

    return np.asarray([
        stats.ttest_1samp(
            group,
            popmean=reference,
        ).pvalue
        for group in groups
    ])


def bonferroni_reject(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Bonferroni family-wise correction.
    """

    reject, _, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method="bonferroni",
    )

    return bool(np.any(reject))


def holm_reject(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Holm step-down family-wise correction.
    """

    reject, _, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method="holm",
    )

    return bool(np.any(reject))


def hochberg_reject(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Hochberg step-up family-wise correction.
    """

    reject, _, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method="simes-hochberg",
    )

    return bool(np.any(reject))


def bh_reject(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Benjamini-Hochberg false discovery rate correction.
    """

    reject, _, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method="fdr_bh",
    )

    return bool(np.any(reject))
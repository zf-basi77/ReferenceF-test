"""
referencef.posthoc
==================

Post-hoc reference contrasts for the Reference F framework.

Provides:
- one-sample reference t-tests for each group
- multiplicity corrections
- contribution summaries

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ReferenceContrast:
    """
    Result of one group versus reference comparison.

    Parameters
    ----------
    group:
        Group identifier.

    mean:
        Sample mean.

    difference:
        Difference from reference.

    statistic:
        One-sample t statistic.

    pvalue:
        Raw p-value.

    adjusted_pvalue:
        Multiplicity-adjusted p-value.

    contribution:
        Contribution to total reference deviation.
    """

    group: int

    mean: float

    difference: float

    statistic: float

    pvalue: float

    adjusted_pvalue: float | None = None

    contribution: float = 0.0


# ---------------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------------


def bonferroni(
    pvalues: Sequence[float],
) -> np.ndarray:
    """
    Bonferroni correction.

    Parameters
    ----------
    pvalues:
        Raw p-values.

    Returns
    -------
    ndarray
        Adjusted p-values.
    """

    p = np.asarray(
        pvalues,
        dtype=float,
    )

    return np.minimum(
        p * len(p),
        1.0,
    )


def holm(
    pvalues: Sequence[float],
) -> np.ndarray:
    """
    Holm step-down correction.
    """

    p = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)

    adjusted = np.empty(m)

    running_max = 0.0

    for rank, index in enumerate(order):

        value = (
            (m - rank)
            *
            p[index]
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[index] = min(
            running_max,
            1.0,
        )

    return adjusted


def benjamini_hochberg(
    pvalues: Sequence[float],
) -> np.ndarray:
    """
    Benjamini-Hochberg false discovery rate correction.
    """

    p = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)

    adjusted = np.empty(m)

    running_min = 1.0

    for rank, index in reversed(
        list(enumerate(order))
    ):

        value = (
            p[index]
            *
            m
            /
            (rank + 1)
        )

        running_min = min(
            running_min,
            value,
        )

        adjusted[index] = min(
            running_min,
            1.0,
        )

    return adjusted


# ---------------------------------------------------------------------
# Reference contrasts
# ---------------------------------------------------------------------


def reference_contrasts(
    groups: Sequence[np.ndarray],
    reference: float,
    *,
    correction: str | None = "holm",
) -> list[ReferenceContrast]:
    """
    Calculate group-specific reference contrasts.

    Parameters
    ----------
    groups:
        Independent samples.

    reference:
        Fixed reference value.

    correction:
        Multiple testing method:
        - None
        - "bonferroni"
        - "holm"
        - "bh"

    Returns
    -------
    list[ReferenceContrast]
    """

    raw_results: list[ReferenceContrast] = []

    contributions = []

    for group in groups:

        x = np.asarray(
            group,
            dtype=float,
        )

        mean = float(
            np.mean(x)
        )

        difference = (
            mean
            -
            reference
        )

        se = (
            np.std(
                x,
                ddof=1,
            )
            /
            np.sqrt(len(x))
        )

        statistic = (
            difference
            /
            se
        )

        pvalue = float(
            2
            *
            stats.t.sf(
                abs(statistic),
                df=len(x)-1,
            )
        )

        contributions.append(
            difference ** 2
        )

        raw_results.append(
            ReferenceContrast(
                group=len(raw_results) + 1,
                mean=mean,
                difference=difference,
                statistic=float(statistic),
                pvalue=pvalue,
            )
        )


    total = np.sum(
        contributions
    )


    for result, contribution in zip(
        raw_results,
        contributions,
    ):

        if total == 0:

            result.contribution = 0.0

        else:

            result.contribution = (
                contribution
                /
                total
            )


    pvalues = [
        result.pvalue
        for result in raw_results
    ]


    if correction == "holm":

        adjusted = holm(pvalues)

    elif correction == "bonferroni":

        adjusted = bonferroni(pvalues)

    elif correction == "bh":

        adjusted = benjamini_hochberg(pvalues)

    elif correction is None:

        adjusted = np.asarray(
            pvalues
        )

    else:

        raise ValueError(
            f"Unknown correction: {correction}"
        )


    for result, value in zip(
        raw_results,
        adjusted,
    ):

        result.adjusted_pvalue = float(
            value
        )


    return raw_results
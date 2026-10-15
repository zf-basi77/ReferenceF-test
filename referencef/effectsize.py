"""
referencef.effectsize
=====================

Effect size calculations for the Reference F framework.

These functions quantify the magnitude and distribution of deviation
from a fixed reference value.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ReferenceFResult


# ---------------------------------------------------------------------
# Effect size result
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ReferenceEffectSize:
    """
    Collection of Reference F effect size measures.

    Parameters
    ----------
    eta_squared:
        Proportion of variation explained by deviation from reference.

    cohens_f:
        Cohen's f effect size.

    rms_reference_deviation:
        Root mean squared deviation of group means from reference.

    mean_reference_deviation:
        Sample-size weighted average deviation from reference.

    standardized_reference_deviation:
        RMS deviation standardized by pooled error SD.
    """

    eta_squared: float

    cohens_f: float

    rms_reference_deviation: float

    mean_reference_deviation: float

    standardized_reference_deviation: float


# ---------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------


def calculate_effect_sizes(
    result: ReferenceFResult,
) -> ReferenceEffectSize:
    """
    Calculate effect sizes from a Reference F result.

    Parameters
    ----------
    result:
        F-test result object.

    Returns
    -------
    ReferenceEffectSize
        Effect size measures.
    """

    eta = result.eta_squared


    # -------------------------------------------------------------
    # Cohen's f
    #
    # f = sqrt(η²/(1-η²))
    # -------------------------------------------------------------

    if eta >= 1:

        cohens_f = float("inf")

    else:

        cohens_f = float(
            np.sqrt(
                eta
                /
                (1 - eta)
            )
        )


    # -------------------------------------------------------------
    # RMS deviation of group means
    #
    # sqrt(
    #     Σni(xbar_i - μ0)^2 / N
    # )
    # -------------------------------------------------------------

    total_n = (
        result.df1
        +
        result.df2
    )

    rms = float(
        np.sqrt(
            result.ss_reference
            /
            total_n
        )
    )


    # -------------------------------------------------------------
    # Weighted mean deviation
    # -------------------------------------------------------------

    weighted_deviation = float(
        np.mean(
            result.group_means
            -
            result.reference
        )
    )


    # -------------------------------------------------------------
    # Standardized deviation
    #
    # RMS deviation / pooled SD
    # -------------------------------------------------------------

    pooled_sd = np.sqrt(
        result.ms_error
    )

    if pooled_sd == 0:

        standardized = float("inf")

    else:

        standardized = float(
            rms
            /
            pooled_sd
        )


    return ReferenceEffectSize(
        eta_squared=float(eta),
        cohens_f=cohens_f,
        rms_reference_deviation=rms,
        mean_reference_deviation=weighted_deviation,
        standardized_reference_deviation=standardized,
    )


# ---------------------------------------------------------------------
# Contribution analysis
# ---------------------------------------------------------------------


def reference_contributions(
    result: ReferenceFResult,
) -> np.ndarray:
    """
    Calculate each group's contribution to the omnibus statistic.

    Contribution_i:

        ni(xbar_i - μ0)^2
        ------------------
        Σ ni(xbar_i - μ0)^2

    Returns
    -------
    ndarray
        Proportion contributed by each group.
    """

    contributions = (
        result.group_means
        -
        result.reference
    ) ** 2


    if np.sum(contributions) == 0:

        return np.zeros_like(
            contributions
        )


    return (
        contributions
        /
        np.sum(contributions)
    )


def reference_contribution_percentages(
    result: ReferenceFResult,
) -> np.ndarray:
    """
    Contribution percentages.

    Returns
    -------
    ndarray
        Percent contribution by group.
    """

    return (
        reference_contributions(result)
        *
        100
    )
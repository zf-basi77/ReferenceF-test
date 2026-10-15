"""
referencef.validation_theorems
==============================

Mathematical validation checks for the Reference F framework.

These functions verify identities described in the Reference F
manuscript:

- Sum of squared reference t statistics equals kF
- One-group case reduces to the one-sample t-test
- Reference F components satisfy ANOVA-style decomposition

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from .model import ReferenceFModel


# ---------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------


@dataclass(slots=True)
class TheoremCheck:
    """
    Result of a mathematical validation check.

    Parameters
    ----------
    passed:
        Whether the identity holds.

    observed:
        Observed numerical value.

    expected:
        Expected numerical value.

    error:
        Absolute difference.

    tolerance:
        Allowed numerical tolerance.
    """

    passed: bool

    observed: float

    expected: float

    error: float

    tolerance: float


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------


def _check_close(
    observed: float,
    expected: float,
    tolerance: float,
) -> TheoremCheck:

    error = abs(
        observed
        -
        expected
    )

    return TheoremCheck(
        passed=error <= tolerance,
        observed=float(observed),
        expected=float(expected),
        error=float(error),
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------
# Identity 1
# Σ t_i² = kF
# ---------------------------------------------------------------------


def validate_sum_t_squared_identity(
    groups: Sequence[np.ndarray],
    reference: float,
    *,
    tolerance: float = 1e-10,
) -> TheoremCheck:
    """
    Validate:

        Σ t_i² = kF

    where each t_i is the individual one-sample
    reference contrast using the pooled error variance.

    Parameters
    ----------
    groups:
        Independent groups.

    reference:
        Fixed reference.

    tolerance:
        Numerical tolerance.

    Returns
    -------
    TheoremCheck
    """

    model = ReferenceFModel(
        reference=reference
    )

    result = model.fit(
        groups
    )


    pooled_sd = np.sqrt(
        result.ms_error
    )


    t_squared_sum = 0.0


    for group, mean in zip(
        groups,
        result.group_means,
    ):

        n = len(group)

        t = (
            mean
            -
            reference
        ) / (
            pooled_sd
            /
            np.sqrt(n)
        )

        t_squared_sum += t**2


    expected = (
        result.k
        *
        result.statistic
    )


    return _check_close(
        t_squared_sum,
        expected,
        tolerance,
    )


# ---------------------------------------------------------------------
# Identity 2
# k = 1 reduces to one sample t-test
# ---------------------------------------------------------------------


def validate_one_group_equivalence(
    group: np.ndarray,
    reference: float,
    *,
    tolerance: float = 1e-10,
) -> TheoremCheck:
    """
    Validate the one-group case.

    The Reference F statistic should equal
    the squared one-sample t statistic.
    """

    model = ReferenceFModel(
        reference=reference
    )

    result = model.fit(
        [group]
    )


    x = np.asarray(
        group,
        dtype=float,
    )

    t = (
        np.mean(x)
        -
        reference
    ) / (
        np.std(
            x,
            ddof=1,
        )
        /
        np.sqrt(len(x))
    )


    return _check_close(
        result.statistic,
        t**2,
        tolerance,
    )


# ---------------------------------------------------------------------
# Identity 3
# Reference mean decomposition
# ---------------------------------------------------------------------


def validate_reference_sum_of_squares(
    groups: Sequence[np.ndarray],
    reference: float,
    *,
    tolerance: float = 1e-10,
) -> TheoremCheck:
    """
    Verify direct computation of SS_reference.

    Checks:

        SS_ref =
        Σ n_i(xbar_i - μ0)^2
    """

    model = ReferenceFModel(
        reference=reference
    )

    result = model.fit(
        groups
    )


    direct = np.sum(
        [
            len(group)
            *
            (
                np.mean(group)
                -
                reference
            ) ** 2
            for group in groups
        ]
    )


    return _check_close(
        result.ss_reference,
        direct,
        tolerance,
    )


# ---------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------


def validate_reference_f(
    groups: Sequence[np.ndarray],
    reference: float,
    *,
    tolerance: float = 1e-10,
) -> dict[str, TheoremCheck]:
    """
    Run all Reference F mathematical checks.

    Returns
    -------
    dict
        Named validation results.
    """

    return {
        "sum_t_squared_identity":
            validate_sum_t_squared_identity(
                groups,
                reference,
                tolerance=tolerance,
            ),

        "one_group_equivalence":
            validate_one_group_equivalence(
                groups[0],
                reference,
                tolerance=tolerance,
            )
            if len(groups) == 1
            else TheoremCheck(
                passed=True,
                observed=float("nan"),
                expected=float("nan"),
                error=0.0,
                tolerance=tolerance,
            ),

        "reference_ss":
            validate_reference_sum_of_squares(
                groups,
                reference,
                tolerance=tolerance,
            ),
    }
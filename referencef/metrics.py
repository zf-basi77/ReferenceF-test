"""
referencef.metrics
==================

Metrics for evaluating Monte Carlo simulation performance.

Includes:
- Type I error estimation
- power estimation
- bias
- mean squared error
- confidence interval coverage
- Monte Carlo standard errors

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def empirical_type1_error(
    rejections: Sequence[bool],
) -> float:
    """
    Estimate empirical Type I error rate.

    Parameters
    ----------
    rejections:
        Boolean rejection indicators from simulations
        performed under the null hypothesis.

    Returns
    -------
    float
        Proportion of rejected null hypotheses.
    """

    values = np.asarray(
        rejections,
        dtype=bool,
    )

    if values.size == 0:
        raise ValueError(
            "rejections cannot be empty."
        )

    return float(
        np.mean(values)
    )


def empirical_power(
    rejections: Sequence[bool],
) -> float:
    """
    Estimate empirical statistical power.

    Parameters
    ----------
    rejections:
        Boolean rejection indicators from simulations
        performed under an alternative hypothesis.

    Returns
    -------
    float
        Probability of rejecting the null hypothesis.
    """

    return empirical_type1_error(
        rejections
    )


def monte_carlo_standard_error(
    probability: float,
    repetitions: int,
) -> float:
    """
    Calculate Monte Carlo standard error for
    an estimated probability.

    Parameters
    ----------
    probability:
        Estimated probability.

    repetitions:
        Number of Monte Carlo repetitions.

    Returns
    -------
    float
        Standard error.

    Notes
    -----
    For a binomial Monte Carlo estimate:

        SE = sqrt(p(1-p)/N)
    """

    if not 0 <= probability <= 1:

        raise ValueError(
            "Probability must be between 0 and 1."
        )

    if repetitions < 1:

        raise ValueError(
            "repetitions must be positive."
        )

    return float(
        np.sqrt(
            probability
            *
            (1 - probability)
            /
            repetitions
        )
    )


def bias(
    estimates: Sequence[float],
    true_value: float,
) -> float:
    """
    Estimate bias.

    Parameters
    ----------
    estimates:
        Simulation estimates.

    true_value:
        True parameter value.

    Returns
    -------
    float
        Estimated bias.
    """

    values = np.asarray(
        estimates,
        dtype=float,
    )

    if values.size == 0:
        raise ValueError(
            "estimates cannot be empty."
        )

    return float(
        np.mean(values)
        -
        true_value
    )


def mean_squared_error(
    estimates: Sequence[float],
    true_value: float,
) -> float:
    """
    Calculate mean squared error.

    Parameters
    ----------
    estimates:
        Simulation estimates.

    true_value:
        True parameter value.

    Returns
    -------
    float
        MSE.
    """

    values = np.asarray(
        estimates,
        dtype=float,
    )

    if values.size == 0:
        raise ValueError(
            "estimates cannot be empty."
        )

    return float(
        np.mean(
            (values - true_value) ** 2
        )
    )


def root_mean_squared_error(
    estimates: Sequence[float],
    true_value: float,
) -> float:
    """
    Calculate root mean squared error.
    """

    return float(
        np.sqrt(
            mean_squared_error(
                estimates,
                true_value,
            )
        )
    )


def coverage_probability(
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    true_value: float,
) -> float:
    """
    Estimate confidence interval coverage.

    Parameters
    ----------
    lower_bounds:
        Lower confidence limits.

    upper_bounds:
        Upper confidence limits.

    true_value:
        True parameter.

    Returns
    -------
    float
        Proportion of intervals containing
        the true value.
    """

    lower = np.asarray(
        lower_bounds,
        dtype=float,
    )

    upper = np.asarray(
        upper_bounds,
        dtype=float,
    )

    if lower.shape != upper.shape:
        raise ValueError(
            "Lower and upper bounds must have "
            "the same shape."
        )

    covered = (
        (lower <= true_value)
        &
        (upper >= true_value)
    )

    return float(
        np.mean(covered)
    )


def rejection_summary(
    rejections: Sequence[bool],
) -> dict[str, float]:
    """
    Produce a compact rejection summary.

    Returns
    -------
    dict
        Rejection probability and Monte Carlo
        standard error.
    """

    values = np.asarray(
        rejections,
        dtype=bool,
    )

    probability = float(
        np.mean(values)
    )

    return {
        "rejection_rate": probability,
        "monte_carlo_se": (
            monte_carlo_standard_error(
                probability,
                len(values),
            )
        ),
    }
"""
referencef.validation
=====================

Validation utilities used throughout the ReferenceF package.

These functions are intentionally small and stateless so they can be
reused by generators, models, and simulation routines.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_non_empty(
    value: Sequence[Any],
    name: str,
) -> None:
    """
    Ensure a sequence contains at least one element.

    Parameters
    ----------
    value:
        Sequence to check.

    name:
        Variable name used in error messages.

    Raises
    ------
    ValueError
        If the sequence is empty.
    """

    if len(value) == 0:
        raise ValueError(
            f"{name} cannot be empty."
        )


def validate_equal_lengths(
    *values: Sequence[Any],
    names: Sequence[str] | None = None,
) -> None:
    """
    Ensure multiple sequences have identical lengths.

    Parameters
    ----------
    values:
        Sequences to compare.

    names:
        Optional names for error messages.

    Raises
    ------
    ValueError
        If lengths differ.
    """

    lengths = [len(value) for value in values]

    if len(set(lengths)) > 1:

        if names is None:
            raise ValueError(
                "All sequences must have equal length."
            )

        details = ", ".join(
            f"{name}={length}"
            for name, length in zip(names, lengths)
        )

        raise ValueError(
            f"Sequence lengths differ: {details}"
        )


def validate_positive(
    values: Sequence[float],
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    """
    Ensure all values are positive.

    Parameters
    ----------
    values:
        Values to check.

    name:
        Variable name.

    allow_zero:
        Whether zero is allowed.

    Raises
    ------
    ValueError
        If invalid values are found.
    """

    array = np.asarray(values)

    if allow_zero:

        invalid = array < 0

    else:

        invalid = array <= 0

    if np.any(invalid):

        raise ValueError(
            f"{name} must contain only "
            f"{'non-negative' if allow_zero else 'positive'} values."
        )


def validate_sample_sizes(
    sample_sizes: Sequence[int],
    *,
    allow_singleton: bool = False,
) -> None:
    """
    Validate group sample sizes.

    Parameters
    ----------
    sample_sizes:
        Number of observations per group.

    allow_singleton:
        Whether groups with n=1 are allowed.

    Raises
    ------
    ValueError
        If sample sizes are invalid.
    """

    validate_non_empty(
        sample_sizes,
        "sample_sizes",
    )

    for n in sample_sizes:

        if not isinstance(n, (int, np.integer)):
            raise TypeError(
                "Sample sizes must be integers."
            )

        if n < 1:
            raise ValueError(
                "Sample sizes must be positive."
            )

        if n == 1 and not allow_singleton:
            raise ValueError(
                "Sample sizes of one are not allowed."
            )


def validate_finite(
    values: np.ndarray,
    name: str,
) -> None:
    """
    Ensure an array contains only finite values.

    Parameters
    ----------
    values:
        Array to check.

    name:
        Variable name.

    Raises
    ------
    ValueError
        If NaN or infinite values are present.
    """

    if not np.all(np.isfinite(values)):

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    """
    Validate a probability parameter.

    Parameters
    ----------
    value:
        Probability value.

    name:
        Variable name.

    Raises
    ------
    ValueError
        If outside [0,1].
    """

    if not 0 <= value <= 1:

        raise ValueError(
            f"{name} must be between 0 and 1."
        )


def validate_degrees_of_freedom(
    df: float,
    *,
    minimum: float = 0,
) -> None:
    """
    Validate degrees of freedom.

    Parameters
    ----------
    df:
        Degrees of freedom.

    minimum:
        Minimum allowed value.

    Raises
    ------
    ValueError
        If invalid.
    """

    if df <= minimum:

        raise ValueError(
            f"Degrees of freedom must exceed {minimum}."
        )
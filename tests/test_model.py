"""
Tests for referencef.model.

Tests verify:
- basic Reference F calculation
- null behavior
- one-sample equivalence
- theorem identities
- edge cases

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from referencef.model import ReferenceFModel
from referencef.validation_theorems import (
    validate_reference_f,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_null_groups(
    seed: int = 123,
) -> list[np.ndarray]:

    rng = np.random.default_rng(
        seed
    )

    return [
        rng.normal(0, 1, 30),
        rng.normal(0, 1, 40),
        rng.normal(0, 1, 50),
    ]


def make_alternative_groups(
    seed: int = 123,
) -> list[np.ndarray]:

    rng = np.random.default_rng(
        seed
    )

    return [
        rng.normal(0, 1, 30),
        rng.normal(1, 1, 40),
        rng.normal(2, 1, 50),
    ]


# ---------------------------------------------------------------------
# Basic model behavior
# ---------------------------------------------------------------------


def test_reference_f_returns_result():

    model = ReferenceFModel(
        reference=0.0
    )

    result = model.fit(
        make_null_groups()
    )

    assert isinstance(
        result.statistic,
        float,
    )

    assert isinstance(
        result.pvalue,
        float,
    )


def test_degrees_of_freedom():

    groups = make_null_groups()

    model = ReferenceFModel(
        reference=0.0
    )

    result = model.fit(groups)

    assert result.df1 == 3

    assert result.df2 == (
        30
        +
        40
        +
        50
        -
        3
    )


def test_statistics_are_nonnegative():

    result = ReferenceFModel(
        reference=0.0
    ).fit(
        make_null_groups()
    )

    assert result.statistic >= 0

    assert 0 <= result.pvalue <= 1


# ---------------------------------------------------------------------
# Detecting alternatives
# ---------------------------------------------------------------------


def test_power_against_large_effect():

    null_result = ReferenceFModel(
        reference=0.0
    ).fit(
        make_null_groups()
    )

    alternative_result = ReferenceFModel(
        reference=0.0
    ).fit(
        make_alternative_groups()
    )

    assert (
        alternative_result.statistic
        >
        null_result.statistic
    )

    assert (
        alternative_result.pvalue
        <
        null_result.pvalue
    )


# ---------------------------------------------------------------------
# One-group equivalence
# ---------------------------------------------------------------------


def test_one_group_equals_squared_t_test():

    rng = np.random.default_rng(
        42
    )

    group = rng.normal(
        2,
        1,
        50,
    )

    reference = 0.0

    result = ReferenceFModel(
        reference=reference
    ).fit(
        [group]
    )


    t = (
        np.mean(group)
        -
        reference
    ) / (
        np.std(
            group,
            ddof=1,
        )
        /
        np.sqrt(len(group))
    )


    np.testing.assert_allclose(
        result.statistic,
        t**2,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------
# Theorem checks
# ---------------------------------------------------------------------


def test_theorem_identity():

    checks = validate_reference_f(
        make_null_groups(),
        reference=0.0,
    )

    for check in checks.values():

        assert check.passed


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------


def test_empty_groups_fail():

    model = ReferenceFModel()

    with pytest.raises(ValueError):

        model.fit([])


def test_single_observation_fails():

    model = ReferenceFModel()

    with pytest.raises(ValueError):

        model.fit(
            [
                np.array([1.0])
            ]
        )


def test_zero_variance_case():

    groups = [
        np.ones(20),
        np.ones(20),
    ]

    result = ReferenceFModel(
        reference=0.0
    ).fit(groups)

    assert np.isinf(
        result.statistic
    ) or result.statistic == 0
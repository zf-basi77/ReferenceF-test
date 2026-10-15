"""
Tests for referencef.effectsize.

Verifies:
- effect size calculations
- contribution percentages
- consistency with Reference F output

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

import numpy as np
import pytest

from referencef.effectsize import (
    calculate_effect_sizes,
    reference_contribution_percentages,
    reference_contributions,
)

from referencef.model import ReferenceFModel


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_result():

    groups = [
        np.array(
            [1, 2, 3, 4, 5],
            dtype=float,
        ),
        np.array(
            [2, 3, 4, 5, 6],
            dtype=float,
        ),
        np.array(
            [0, 1, 2, 3, 4],
            dtype=float,
        ),
    ]

    return ReferenceFModel(
        reference=0.0
    ).fit(groups)


# ---------------------------------------------------------------------
# Basic effect sizes
# ---------------------------------------------------------------------


def test_effect_size_returns_object():

    result = make_result()

    effects = calculate_effect_sizes(
        result
    )

    assert effects.eta_squared >= 0

    assert effects.cohens_f >= 0


def test_eta_squared_range():

    result = make_result()

    effects = calculate_effect_sizes(
        result
    )

    assert (
        0
        <=
        effects.eta_squared
        <=
        1
    )


def test_cohens_f_relationship():

    result = make_result()

    effects = calculate_effect_sizes(
        result
    )

    expected = np.sqrt(
        effects.eta_squared
        /
        (
            1
            -
            effects.eta_squared
        )
    )

    np.testing.assert_allclose(
        effects.cohens_f,
        expected,
    )


# ---------------------------------------------------------------------
# RMS deviation
# ---------------------------------------------------------------------


def test_rms_reference_deviation():

    result = make_result()

    effects = calculate_effect_sizes(
        result
    )

    expected = np.sqrt(
        result.ss_reference
        /
        (
            result.df1
            +
            result.df2
        )
    )

    np.testing.assert_allclose(
        effects.rms_reference_deviation,
        expected,
    )


# ---------------------------------------------------------------------
# Contributions
# ---------------------------------------------------------------------


def test_contributions_sum_to_one():

    result = make_result()

    contributions = (
        reference_contributions(
            result
        )
    )

    np.testing.assert_allclose(
        np.sum(contributions),
        1.0,
    )


def test_percentage_contributions_sum_to_100():

    result = make_result()

    percentages = (
        reference_contribution_percentages(
            result
        )
    )

    np.testing.assert_allclose(
        np.sum(percentages),
        100.0,
    )


def test_zero_deviation_contributions():

    groups = [
        np.ones(20),
        np.ones(20),
    ]

    result = ReferenceFModel(
        reference=1.0
    ).fit(groups)

    contributions = (
        reference_contributions(
            result
        )
    )

    np.testing.assert_allclose(
        contributions,
        np.zeros(2),
    )
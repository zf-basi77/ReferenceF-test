"""
Tests for referencef.posthoc.

Verifies:
- reference contrasts
- multiplicity corrections
- contribution calculations

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

import numpy as np
import pytest

from referencef.posthoc import (
    benjamini_hochberg,
    bonferroni,
    holm,
    reference_contrasts,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_groups():

    return [
        np.array(
            [0, 0, 0, 0, 0],
            dtype=float,
        ),
        np.array(
            [1, 1, 1, 1, 1],
            dtype=float,
        ),
        np.array(
            [-1, -1, -1, -1, -1],
            dtype=float,
        ),
    ]


# ---------------------------------------------------------------------
# Contrast calculation
# ---------------------------------------------------------------------


def test_reference_contrasts_returns_results():

    results = reference_contrasts(
        make_groups(),
        reference=0.0,
    )

    assert len(results) == 3


def test_group_means_are_correct():

    results = reference_contrasts(
        make_groups(),
        reference=0.0,
    )

    assert results[0].mean == 0

    assert results[1].mean == 1

    assert results[2].mean == -1


def test_difference_from_reference():

    results = reference_contrasts(
        make_groups(),
        reference=0.0,
    )

    assert results[1].difference == 1

    assert results[2].difference == -1


# ---------------------------------------------------------------------
# Multiplicity corrections
# ---------------------------------------------------------------------


def test_bonferroni_increases_pvalues():

    p = np.array(
        [
            0.01,
            0.02,
            0.5,
        ]
    )

    adjusted = bonferroni(p)

    assert np.all(
        adjusted >= p
    )


def test_holm_preserves_order():

    p = np.array(
        [
            0.01,
            0.02,
            0.5,
        ]
    )

    adjusted = holm(p)

    assert adjusted[0] <= adjusted[2]


def test_bh_values_between_zero_one():

    p = np.array(
        [
            0.01,
            0.2,
            0.8,
        ]
    )

    adjusted = benjamini_hochberg(
        p
    )

    assert np.all(
        adjusted >= 0
    )

    assert np.all(
        adjusted <= 1
    )


# ---------------------------------------------------------------------
# Contribution
# ---------------------------------------------------------------------


def test_contributions_sum_to_one():

    results = reference_contrasts(
        make_groups(),
        reference=0.0,
    )

    contributions = np.array(
        [
            r.contribution
            for r in results
        ]
    )

    np.testing.assert_allclose(
        np.sum(contributions),
        1.0,
    )


def test_equal_deviation_contributions():

    groups = [
        np.ones(10),
        np.ones(10) * 2,
    ]

    results = reference_contrasts(
        groups,
        reference=0.0,
    )

    contributions = [
        r.contribution
        for r in results
    ]

    np.testing.assert_allclose(
        contributions,
        [
            0.2,
            0.8,
        ],
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_invalid_correction_fails():

    with pytest.raises(ValueError):

        reference_contrasts(
            make_groups(),
            reference=0.0,
            correction="invalid",
        )
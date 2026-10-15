"""
Tests for referencef.generators.

These tests verify:
- reproducibility
- validation behavior
- dataset structure
- registry functionality
- distribution metadata

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

import numpy as np
import pytest

from referencef import (
    GammaGenerator,
    GeneratorConfig,
    GeneratorRegistry,
    LogNormalGenerator,
    NormalGenerator,
    StudentTGenerator,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_config(seed: int = 123) -> GeneratorConfig:
    """
    Standard test configuration.
    """

    return GeneratorConfig(
        sample_sizes=[20, 25, 30],
        seed=seed,
    )


# ---------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------


def test_config_rejects_empty_sample_sizes():

    with pytest.raises(ValueError):

        GeneratorConfig(
            sample_sizes=[],
        )


def test_config_rejects_invalid_sample_sizes():

    with pytest.raises(ValueError):

        GeneratorConfig(
            sample_sizes=[10, 0, 20],
        )


def test_config_allows_singletons_when_requested():

    config = GeneratorConfig(
        sample_sizes=[1, 5],
        allow_singleton=True,
    )

    assert config.sample_sizes == [1, 5]


# ---------------------------------------------------------------------
# Normal generator
# ---------------------------------------------------------------------


def test_normal_generator_shape():

    generator = NormalGenerator(
        config=make_config(),
        means=[0, 1, 2],
        std=[1, 1, 1],
    )

    dataset = generator.generate()

    assert dataset.k == 3
    assert dataset.total_n == 75

    assert [
        len(group)
        for group in dataset.groups
    ] == [
        20,
        25,
        30,
    ]


def test_normal_generator_records_parameters():

    generator = NormalGenerator(
        config=make_config(),
        means=[0, 1, 2],
        std=[1, 2, 3],
    )

    dataset = generator.generate()

    np.testing.assert_array_equal(
        dataset.population_means,
        np.array([0, 1, 2]),
    )

    np.testing.assert_array_equal(
        dataset.population_variances,
        np.array([1, 4, 9]),
    )


def test_normal_generator_reproducible():

    generator1 = NormalGenerator(
        config=make_config(seed=42),
        means=[0, 0, 0],
        std=[1, 1, 1],
    )

    generator2 = NormalGenerator(
        config=make_config(seed=42),
        means=[0, 0, 0],
        std=[1, 1, 1],
    )

    data1 = generator1.generate()
    data2 = generator2.generate()

    for a, b in zip(
        data1.groups,
        data2.groups,
    ):

        np.testing.assert_array_equal(
            a,
            b,
        )


def test_normal_generator_rejects_negative_std():

    with pytest.raises(ValueError):

        NormalGenerator(
            config=make_config(),
            means=[0, 0, 0],
            std=[1, -1, 1],
        )


# ---------------------------------------------------------------------
# Student t generator
# ---------------------------------------------------------------------


def test_student_t_generator():

    generator = StudentTGenerator(
        config=make_config(),
        means=[0, 0, 0],
        std=[1, 1, 1],
        df=10,
    )

    dataset = generator.generate()

    assert dataset.distribution == "Student-t"


def test_student_t_rejects_low_df():

    with pytest.raises(ValueError):

        StudentTGenerator(
            config=make_config(),
            means=[0, 0, 0],
            std=[1, 1, 1],
            df=1,
        )


# ---------------------------------------------------------------------
# Lognormal
# ---------------------------------------------------------------------


def test_lognormal_generator():

    generator = LogNormalGenerator(
        config=make_config(),
        log_means=[0, 0, 0],
        log_sigma=[0.5, 0.5, 0.5],
    )

    dataset = generator.generate()

    assert dataset.distribution == "LogNormal"

    for group in dataset.groups:

        assert np.all(group > 0)


# ---------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------


def test_gamma_generator():

    generator = GammaGenerator(
        config=make_config(),
        shape=[2, 2, 2],
        scale=[1, 1, 1],
    )

    dataset = generator.generate()

    assert dataset.distribution == "Gamma"

    for group in dataset.groups:

        assert np.all(group >= 0)


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


def test_registry_contains_defaults():

    available = GeneratorRegistry.available()

    assert "normal" in available
    assert "student_t" in available
    assert "lognormal" in available
    assert "gamma" in available


def test_registry_create():

    generator = GeneratorRegistry.create(
        "normal",
        config=make_config(),
        means=[0, 0, 0],
        std=[1, 1, 1],
    )

    assert isinstance(
        generator,
        NormalGenerator,
    )
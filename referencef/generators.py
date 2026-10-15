"""
referencef.generators
=====================

Synthetic data generators for Monte Carlo simulation.

Generators create SimulationDataset objects containing simulated
observations and the population parameters used to generate them.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .types import Dataset, FloatArray
from .validation import (
    validate_equal_lengths,
    validate_finite,
    validate_positive,
    validate_sample_sizes,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class GeneratorConfig:
    """
    Common configuration for all generators.

    Parameters
    ----------
    sample_sizes:
        Number of observations in each group.

    seed:
        Random seed.

    dtype:
        Output numerical dtype.

    allow_singleton:
        Whether groups of size one are allowed.

    metadata:
        Optional metadata attached to generated datasets.
    """

    sample_sizes: Sequence[int]

    seed: int | None = None

    dtype: np.dtype = np.dtype(np.float64)

    allow_singleton: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        validate_sample_sizes(
            self.sample_sizes,
            allow_singleton=self.allow_singleton,
        )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


@dataclass(slots=True)
class SimulationDataset:
    """
    A single generated dataset.

    Parameters
    ----------
    groups:
        Observations separated by population.

    population_means:
        True population means.

    population_variances:
        True population variances.

    sample_sizes:
        Number of observations per group.

    distribution:
        Distribution name.

    metadata:
        Generator metadata.
    """

    groups: Dataset

    population_means: FloatArray

    population_variances: FloatArray

    sample_sizes: FloatArray

    distribution: str

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.groups)

    @property
    def total_n(self) -> int:
        return int(np.sum(self.sample_sizes))


# ---------------------------------------------------------------------
# Base generator
# ---------------------------------------------------------------------


class DataGenerator(ABC):
    """
    Abstract generator interface.
    """

    def __init__(
        self,
        config: GeneratorConfig,
    ) -> None:

        self.config = config
        self.rng = np.random.default_rng(
            config.seed
        )

    @abstractmethod
    def generate(self) -> SimulationDataset:
        """
        Generate one dataset.
        """


# ---------------------------------------------------------------------
# Independent generators
# ---------------------------------------------------------------------


class IndependentGenerator(DataGenerator):
    """
    Base class for independent population generators.
    """

    def validate_parameters(
        self,
        *parameters: Sequence[Any],
    ) -> None:

        validate_equal_lengths(
            *parameters
        )

        if len(parameters[0]) != len(
            self.config.sample_sizes
        ):
            raise ValueError(
                "Parameter length must equal "
                "number of groups."
            )


# ---------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------


class NormalGenerator(IndependentGenerator):
    """
    Independent normal populations.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        means: Sequence[float],
        std: Sequence[float],
    ) -> None:

        super().__init__(config)

        self.means = np.asarray(
            means,
            dtype=float,
        )

        self.std = np.asarray(
            std,
            dtype=float,
        )

        self.validate_parameters(
            self.means,
            self.std,
        )

        validate_positive(
            self.std,
            "std",
        )

    def generate(self) -> SimulationDataset:

        groups = [
            self.rng.normal(
                loc=mu,
                scale=sigma,
                size=n,
            ).astype(
                self.config.dtype
            )
            for mu, sigma, n in zip(
                self.means,
                self.std,
                self.config.sample_sizes,
            )
        ]

        return SimulationDataset(
            groups=groups,
            population_means=self.means.copy(),
            population_variances=self.std**2,
            sample_sizes=np.asarray(
                self.config.sample_sizes,
                dtype=float,
            ),
            distribution="Normal",
            metadata=self.config.metadata.copy(),
        )


# ---------------------------------------------------------------------
# Student t
# ---------------------------------------------------------------------


class StudentTGenerator(IndependentGenerator):
    """
    Independent Student t populations.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        means: Sequence[float],
        std: Sequence[float],
        df: float,
    ) -> None:

        super().__init__(config)

        self.means = np.asarray(
            means,
            dtype=float,
        )

        self.std = np.asarray(
            std,
            dtype=float,
        )

        self.df = float(df)

        self.validate_parameters(
            self.means,
            self.std,
        )

        validate_positive(
            self.std,
            "std",
        )

        if self.df <= 2:
            raise ValueError(
                "df must be greater than 2."
            )

    def generate(self) -> SimulationDataset:

        scale_adjustment = np.sqrt(
            (self.df - 2)
            /
            self.df
        )

        groups = []

        for mu, sigma, n in zip(
            self.means,
            self.std,
            self.config.sample_sizes,
        ):

            values = self.rng.standard_t(
                df=self.df,
                size=n,
            )

            values = (
                values
                * sigma
                * scale_adjustment
                + mu
            )

            groups.append(
                values.astype(
                    self.config.dtype
                )
            )

        return SimulationDataset(
            groups=groups,
            population_means=self.means.copy(),
            population_variances=self.std**2,
            sample_sizes=np.asarray(
                self.config.sample_sizes,
                dtype=float,
            ),
            distribution="Student-t",
            metadata={
                **self.config.metadata,
                "df": self.df,
            },
        )


# ---------------------------------------------------------------------
# Lognormal
# ---------------------------------------------------------------------


class LogNormalGenerator(IndependentGenerator):
    """
    Independent lognormal populations.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        log_means: Sequence[float],
        log_sigma: Sequence[float],
        location: Sequence[float] | None = None,
    ) -> None:

        super().__init__(config)

        self.log_means = np.asarray(
            log_means,
            dtype=float,
        )

        self.log_sigma = np.asarray(
            log_sigma,
            dtype=float,
        )
        if location is None:
            location = np.zeros_like(self.log_means)

        self.location = np.asarray(
            location,
            dtype=float,
        )

        self.validate_parameters(
            self.log_means,
            self.log_sigma,
            self.location,
        )

        validate_positive(
            self.log_sigma,
            "log_sigma",
        )

    def generate(self) -> SimulationDataset:

        groups = [
            (
                self.rng.lognormal(
                    mean=mu,
                    sigma=sigma,
                    size=n,
                )
                + loc
            ).astype(self.config.dtype)
            for mu, sigma, loc, n in zip(
                self.log_means,
                self.log_sigma,
                self.location,
                self.config.sample_sizes,
            )
        ]

        means = (
            np.exp(
                self.log_means +
                self.log_sigma**2 / 2
            )
            + self.location
        )

        variances = (
            (
                np.exp(self.log_sigma**2) - 1
            )
            *
            np.exp(
                2*self.log_means +
                self.log_sigma**2
            )
        )

        return SimulationDataset(
            groups=groups,
            population_means=means,
            population_variances=variances,
            sample_sizes=np.asarray(
                self.config.sample_sizes,
                dtype=float,
            ),
            distribution="LogNormal",
            metadata=self.config.metadata.copy(),
        )


# ---------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------


class GammaGenerator(IndependentGenerator):
    """
    Independent gamma populations.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        shape: Sequence[float],
        scale: Sequence[float],
        location: Sequence[float] | None = None,
    ) -> None:

        super().__init__(config)

        self.shape = np.asarray(
            shape,
            dtype=float,
        )

        self.scale = np.asarray(
            scale,
            dtype=float,
        )
        if location is None:
            location = np.zeros_like(self.shape)

        self.location = np.asarray(
            location,
            dtype=float,
        )

        self.validate_parameters(
            self.shape,
            self.scale,
            self.location,
        )

        validate_positive(
            self.shape,
            "shape",
        )

        validate_positive(
            self.scale,
            "scale",
        )

    def generate(self) -> SimulationDataset:

        groups = [
            (
                self.rng.gamma(
                    shape=a,
                    scale=b,
                    size=n,
                )
                + loc
            ).astype(self.config.dtype)
            for a, b, loc, n in zip(
                self.shape,
                self.scale,
                self.location,
                self.config.sample_sizes,
            )
        ]

        means = (
            self.shape *
            self.scale
            +
            self.location
        )

        variances = (
            self.shape
            *
            self.scale**2
        )

        return SimulationDataset(
            groups=groups,
            population_means=np.asarray(means),
            population_variances=np.asarray(
                variances
            ),
            sample_sizes=np.asarray(
                self.config.sample_sizes,
                dtype=float,
            ),
            distribution="Gamma",
            metadata=self.config.metadata.copy(),
        )
from .registry import GeneratorRegistry

GeneratorRegistry.register(
    "normal",
    NormalGenerator,
)
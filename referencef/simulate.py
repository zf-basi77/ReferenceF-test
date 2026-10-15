"""
referencef.simulate
===================

Generic Monte Carlo simulation engine.

The simulation engine is independent of any statistical method.
It accepts any generator producing SimulationDataset objects and any
model implementing a compatible fit() method.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Sequence

import numpy as np

from .generators import DataGenerator, SimulationDataset


# ---------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------


class StatisticalModel(Protocol):
    """
    Interface required by the simulation engine.
    """

    def fit(
        self,
        groups: Sequence[np.ndarray],
    ) -> Any:
        """
        Fit a statistical model.

        Returns
        -------
        Any
            Model result object.
        """


# ---------------------------------------------------------------------
# Single simulation result
# ---------------------------------------------------------------------


@dataclass(slots=True)
class SimulationRecord:
    """
    Stores results from one Monte Carlo repetition.
    """

    statistic: float | None = None

    pvalue: float | None = None

    reject: bool | None = None

    true_means: np.ndarray | None = None

    true_variances: np.ndarray | None = None

    sample_sizes: np.ndarray | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------
# Complete simulation result
# ---------------------------------------------------------------------


@dataclass(slots=True)
class SimulationResult:
    """
    Collection of Monte Carlo simulation results.
    """

    records: list[SimulationRecord]

    repetitions: int

    alpha: float

    generator: str

    model: str

    timestamp: str = field(
        default_factory=lambda:
        datetime.now().isoformat()
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def statistics(self) -> np.ndarray:
        """
        Return test statistics.
        """

        return np.asarray(
            [
                r.statistic
                for r in self.records
                if r.statistic is not None
            ]
        )

    @property
    def pvalues(self) -> np.ndarray:
        """
        Return p-values.
        """

        return np.asarray(
            [
                r.pvalue
                for r in self.records
                if r.pvalue is not None
            ]
        )

    @property
    def rejection_rate(self) -> float:
        """
        Empirical rejection probability.
        """

        rejected = [
            r.reject
            for r in self.records
            if r.reject is not None
        ]

        if len(rejected) == 0:
            return float("nan")

        return float(
            np.mean(rejected)
        )

    def summary(self) -> str:
        """
        Produce text summary.
        """

        lines = [
            "Monte Carlo Simulation Summary",
            "-----------------------------",
            f"Model: {self.model}",
            f"Generator: {self.generator}",
            f"Repetitions: {self.repetitions}",
            f"Alpha: {self.alpha}",
            f"Empirical rejection rate: "
            f"{self.rejection_rate:.5f}",
        ]

        if len(self.statistics) > 0:

            lines.append(
                f"Mean statistic: "
                f"{np.mean(self.statistics):.5f}"
            )

        return "\n".join(lines)
    def to_dataframe(self):
        """
        Convert simulation records to pandas DataFrame.
        """

        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "statistic": r.statistic,
                    "pvalue": r.pvalue,
                    "reject": r.reject,
                    **r.metadata,
                }
                for r in self.records
            ]
        )

# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


class MonteCarloEngine:
    """
    Generic Monte Carlo simulation engine.

    Examples
    --------
    >>> engine = MonteCarloEngine()
    >>> result = engine.run(
    ...     generator,
    ...     model,
    ...     repetitions=10000,
    ... )
    """

    def run(
        self,
        generator: DataGenerator,
        model: StatisticalModel,
        repetitions: int,
        *,
        alpha: float = 0.05,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        generator:
            Dataset generator.

        model:
            Statistical analysis model.

        repetitions:
            Number of simulations.

        alpha:
            Significance level.

        Returns
        -------
        SimulationResult
        """

        if repetitions < 1:

            raise ValueError(
                "repetitions must be positive."
            )

        records: list[SimulationRecord] = []

        for _ in range(repetitions):

            dataset: SimulationDataset = (
                generator.generate()
            )

            result = model.fit(
                dataset.groups
            )

            record = self._extract_result(
                result,
                alpha,
            )
            record.true_means = dataset.population_means
            record.true_variances = dataset.population_variances
            record.sample_sizes = dataset.sample_sizes
            record.metadata.update(
                dataset.metadata
            )
            records.append(record)

        return SimulationResult(
            records=records,
            repetitions=repetitions,
            alpha=alpha,
            generator=generator.__class__.__name__,
            model=model.__class__.__name__,
        )

    @staticmethod
    def _extract_result(
        result: Any,
        alpha: float,
    ) -> SimulationRecord:
        """
        Convert arbitrary model output into a
        SimulationRecord.

        Models should expose:
        - statistic
        - pvalue

        if available.
        """

        statistic = getattr(
            result,
            "statistic",
            None,
        )

        pvalue = getattr(
            result,
            "pvalue",
            None,
        )

        reject = None

        if pvalue is not None:

            reject = bool(
                pvalue < alpha
            )

        return SimulationRecord(
            statistic=statistic,
            pvalue=pvalue,
            reject=reject,
        )
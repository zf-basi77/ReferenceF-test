"""
referencef.model
================

Reference F-test implementation.

The Reference F-test evaluates whether multiple independent population
means are equal to a common externally specified reference value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# Reference contrasts
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceContrast:
    """
    Group-specific deviation from the fixed reference value.
    """

    group: int

    estimate: float

    standard_error: float

    t_statistic: float

    pvalue: float

    ci_lower: float

    ci_upper: float


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

@dataclass(slots=True)
class ReferenceFResult:
    """
    Results from a Reference F-test.
    """

    statistic: float

    pvalue: float

    df1: int

    df2: int

    ss_reference: float

    ss_error: float

    ms_error: float

    group_means: np.ndarray

    reference: float

    reference_contrasts: list[ReferenceContrast] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


    @property
    def k(self) -> int:
        """
        Number of groups.
        """

        return len(self.group_means)


    @property
    def n_total(self) -> int:
        """
        Total sample size.
        """

        return self.df1 + self.df2


    @property
    def eta_squared(self) -> float:
        """
        Reference explained variance.

        η² = SS_reference /
             (SS_reference + SS_error)
        """

        total = (
            self.ss_reference
            +
            self.ss_error
        )

        return self.ss_reference / total


    @property
    def cohens_f(self) -> float:
        """
        Cohen's f derived from eta squared.

        f = sqrt(η² / (1 - η²))
        """

        eta = self.eta_squared

        if eta >= 1:
            raise ValueError(
                "Cohen's f undefined because eta squared >= 1."
            )

        return float(
            np.sqrt(
                eta /
                (1 - eta)
            )
        )
    @property
    def contributions(self) -> np.ndarray:
        """
        Proportion of omnibus statistic attributable
        to each group.

        c_i = t_i^2 / sum(t_i^2)
        """

        t_squared = np.asarray(
            [
                contrast.t_statistic ** 2
                for contrast in self.reference_contrasts
            ]
        )

        return (
            t_squared
            /
            np.sum(t_squared)
        )

    def summary(self) -> str:
        """
        Return formatted statistical summary.
        """

        lines = [
            "Reference F-test",
            "================",
            "",
            f"Reference value: {self.reference}",
            "",
            f"Groups: {self.k}",
            f"Total N: {self.n_total}",
            "",
            "Test statistic",
            "--------------",
            f"F statistic: {self.statistic:.5f}",
            f"df1: {self.df1}",
            f"df2: {self.df2}",
            f"p-value: {self.pvalue:.5g}",
            "",
            "Effect size",
            "-----------",
            f"eta squared: {self.eta_squared:.5f}",
            f"Cohen's f: {self.cohens_f:.5f}",
        ]


        if self.reference_contrasts:

            lines.extend(
                [
                    "",
                    "Reference contrasts",
                    "-------------------",
                ]
            )

            for contrast in self.reference_contrasts:

                lines.append(
                    (
                        f"Group {contrast.group}: "
                        f"estimate={contrast.estimate:.4f}, "
                        f"CI=[{contrast.ci_lower:.4f}, "
                        f"{contrast.ci_upper:.4f}], "
                        f"p={contrast.pvalue:.5g}"
                    )
                )


        return "\n".join(lines)


    def to_dict(self) -> dict:
        """
        Convert result to serializable dictionary.
        """

        output = {
            "statistic": float(self.statistic),
            "pvalue": float(self.pvalue),
            "df1": int(self.df1),
            "df2": int(self.df2),
            "reference": float(self.reference),
            "ss_reference": float(self.ss_reference),
            "ss_error": float(self.ss_error),
            "ms_error": float(self.ms_error),
            "eta_squared": float(self.eta_squared),
            "cohens_f": float(self.cohens_f),
            "group_means": [
                float(x)
                for x in self.group_means
            ],
        }


        output["reference_contrasts"] = [
            {
                "group": c.group,
                "estimate": c.estimate,
                "standard_error": c.standard_error,
                "t_statistic": c.t_statistic,
                "pvalue": c.pvalue,
                "ci_lower": c.ci_lower,
                "ci_upper": c.ci_upper,
            }
            for c in self.reference_contrasts
        ]


        return output

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class ReferenceFModel:
    """
    Reference F-test model.

    Tests:

        H0:
            μ1 = μ2 = ... = μk = reference

        H1:
            At least one population mean differs.

    Assumes independent groups and common within-group variance.
    """


    def __init__(
        self,
        reference: float = 0.0,
    ) -> None:

        self.reference = float(reference)


    def fit(
        self,
        groups: Sequence[np.ndarray],
    ) -> ReferenceFResult:
        """
        Fit the Reference F-test.

        Parameters
        ----------
        groups:
            Sequence of independent samples.

        Returns
        -------
        ReferenceFResult
        """


        if len(groups) < 1:
            raise ValueError(
                "At least one group is required."
            )


        arrays = [
            np.asarray(
                group,
                dtype=float,
            )
            for group in groups
        ]


        k = len(arrays)


        sample_sizes = np.asarray(
            [
                len(x)
                for x in arrays
            ],
            dtype=int,
        )


        if np.any(sample_sizes < 2):
            raise ValueError(
                "Each group requires n >= 2."
            )


        means = np.asarray(
            [
                np.mean(x)
                for x in arrays
            ]
        )


        # -------------------------------------------------------------
        # Reference sum of squares
        #
        # SS_ref = Σ ni(xbar_i - μ0)^2
        # -------------------------------------------------------------

        ss_reference = float(
            np.sum(
                sample_sizes
                *
                (means - self.reference) ** 2
            )
        )


        # -------------------------------------------------------------
        # Error sum of squares
        #
        # SS_error = ΣΣ(xij-xbar_i)^2
        # -------------------------------------------------------------

        ss_error = float(
            np.sum(
                [
                    np.sum(
                        (x - mean) ** 2
                    )
                    for x, mean in zip(
                        arrays,
                        means,
                    )
                ]
            )
        )


        df1 = k

        df2 = int(
            np.sum(sample_sizes)
            -
            k
        )


        ms_error = (
            ss_error / df2
        )


        ms_reference = (
            ss_reference / df1
        )


        statistic = (
            ms_reference
            /
            ms_error
        )


        pvalue = float(
            stats.f.sf(
                statistic,
                df1,
                df2,
            )
        )


        # -------------------------------------------------------------
        # Reference contrasts
        # -------------------------------------------------------------

        contrasts = []

        alpha = 0.05

        tcrit = stats.t.ppf(
            1 - alpha / 2,
            df2,
        )


        for i, (mean, n) in enumerate(
            zip(
                means,
                sample_sizes,
            )
        ):

            estimate = (
                mean
                -
                self.reference
            )


            standard_error = np.sqrt(
                ms_error / n
            )


            t_statistic = (
                estimate
                /
                standard_error
            )


            contrast_pvalue = float(
                2
                *
                stats.t.sf(
                    abs(t_statistic),
                    df2,
                )
            )


            margin = (
                tcrit
                *
                standard_error
            )


            contrasts.append(
                ReferenceContrast(
                    group=i + 1,
                    estimate=float(estimate),
                    standard_error=float(
                        standard_error
                    ),
                    t_statistic=float(
                        t_statistic
                    ),
                    pvalue=contrast_pvalue,
                    ci_lower=float(
                        estimate - margin
                    ),
                    ci_upper=float(
                        estimate + margin
                    ),
                )
            )


        return ReferenceFResult(
            statistic=float(statistic),
            pvalue=pvalue,
            df1=df1,
            df2=df2,
            ss_reference=ss_reference,
            ss_error=ss_error,
            ms_error=ms_error,
            group_means=means,
            reference=self.reference,
            reference_contrasts=contrasts,
        )
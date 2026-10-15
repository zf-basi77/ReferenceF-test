"""
referencef.plotting
===================

Visualization utilities for Reference F analyses
and simulation studies.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy import stats

from .model import ReferenceFResult


# ---------------------------------------------------------------------
# Simulation performance plots
# ---------------------------------------------------------------------


def plot_power_curve(
    deviations: Sequence[float],
    powers: Sequence[float],
    *,
    ax: Axes | None = None,
    label: str = "Reference F",
) -> tuple[Figure, Axes]:
    """
    Plot statistical power versus deviation from reference.

    Parameters
    ----------
    deviations:
        Effect size values.

    powers:
        Estimated rejection probabilities.

    ax:
        Existing matplotlib axis.

    label:
        Curve label.

    Returns
    -------
    Figure, Axes
    """

    if ax is None:

        fig, ax = plt.subplots()

    else:

        fig = ax.figure


    ax.plot(
        deviations,
        powers,
        marker="o",
        label=label,
    )


    ax.set_xlabel(
        "Deviation from reference"
    )

    ax.set_ylabel(
        "Power"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()


    return fig, ax



def plot_method_comparison(
    results: dict[str, Sequence[float]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot multiple method power curves.

    Parameters
    ----------
    results:
        Dictionary mapping method names
        to power values.

    Returns
    -------
    Figure, Axes
    """

    if ax is None:

        fig, ax = plt.subplots()

    else:

        fig = ax.figure


    deviations = results["deviation"]


    for method, values in results.items():

        if method == "deviation":
            continue

        ax.plot(
            deviations,
            values,
            marker="o",
            label=method,
        )


    ax.set_xlabel(
        "Deviation from reference"
    )

    ax.set_ylabel(
        "Rejection probability"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()


    return fig, ax



# ---------------------------------------------------------------------
# Reference F distribution validation
# ---------------------------------------------------------------------


def plot_f_distribution(
    statistics: Sequence[float],
    df1: int,
    df2: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Compare simulated F statistics with theoretical F density.

    Used to validate:

        F_obs ~ F(df1, df2)

    """

    if ax is None:

        fig, ax = plt.subplots()

    else:

        fig = ax.figure


    values = np.asarray(
        statistics
    )


    ax.hist(
        values,
        bins=40,
        density=True,
        alpha=0.5,
        label="Simulation",
    )


    x = np.linspace(
        0,
        np.max(values),
        300,
    )


    ax.plot(
        x,
        stats.f.pdf(
            x,
            df1,
            df2,
        ),
        linewidth=2,
        label="Theoretical F",
    )


    ax.set_xlabel(
        "F statistic"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )


    return fig, ax



# ---------------------------------------------------------------------
# Individual reference contributions
# ---------------------------------------------------------------------


def plot_reference_contributions(
    result: ReferenceFResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot group contributions to the omnibus deviation.

    Contribution:

        n_i(xbar_i - μ0)^2
        -------------------
        total
    """

    if ax is None:

        fig, ax = plt.subplots()

    else:

        fig = ax.figure


    contributions = (
        result.group_means
        -
        result.reference
    ) ** 2


    if np.sum(contributions) > 0:

        contributions = (
            contributions
            /
            np.sum(contributions)
        )


    labels = [
        f"Group {i+1}"
        for i in range(len(contributions))
    ]


    ax.bar(
        labels,
        contributions * 100,
    )


    ax.set_ylabel(
        "Contribution (%)"
    )

    ax.set_title(
        "Reference deviation contributions"
    )


    return fig, ax



# ---------------------------------------------------------------------
# Group means plot
# ---------------------------------------------------------------------


def plot_group_estimates(
    result: ReferenceFResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot observed group means against reference.
    """

    if ax is None:

        fig, ax = plt.subplots()

    else:

        fig = ax.figure


    x = np.arange(
        1,
        len(result.group_means)+1,
    )


    ax.scatter(
        x,
        result.group_means,
        s=80,
        label="Observed means",
    )


    ax.axhline(
        result.reference,
        color="red",
        linestyle="--",
        label="Reference",
    )


    ax.set_xlabel(
        "Group"
    )

    ax.set_ylabel(
        "Mean"
    )


    ax.legend()

    return fig, ax
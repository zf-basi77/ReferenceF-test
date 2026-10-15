"""
Comparison of Reference F with classical procedures.

Methods compared:
    1. Reference F-test
    2. One-way ANOVA
    3. Multiple one-sample t-tests with Holm correction

The simulation evaluates rejection probability when groups
move together away from a fixed reference value.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

import numpy as np

from scipy import stats

from referencef import (
    GeneratorConfig,
    NormalGenerator,
)

from referencef.model import (
    ReferenceFModel,
)

from referencef.posthoc import (
    holm,
)

from referencef.simulate import (
    MonteCarloEngine,
)


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

REPETITIONS = 5000

ALPHA = 0.05

N = 30

DEVIATIONS = np.linspace(
    0,
    2,
    21,
)


# ---------------------------------------------------------------------
# ANOVA helper
# ---------------------------------------------------------------------


def anova_pvalue(
    groups,
):
    """
    Standard one-way ANOVA.
    """

    return float(
        stats.f_oneway(
            *groups
        ).pvalue
    )


# ---------------------------------------------------------------------
# Multiple t-test helper
# ---------------------------------------------------------------------


def multiple_ttest_pvalue(
    groups,
    reference=0.0,
):
    """
    Holm-adjusted minimum p-value
    from individual reference tests.
    """

    pvalues = []

    for group in groups:

        t, p = stats.ttest_1samp(
            group,
            reference,
        )

        pvalues.append(
            p
        )


    adjusted = holm(
        pvalues
    )


    return float(
        np.min(adjusted)
    )


# ---------------------------------------------------------------------
# Single scenario
# ---------------------------------------------------------------------


def run_scenario(
    deviation: float,
):

    rf_rejects = []

    anova_rejects = []

    ttest_rejects = []


    generator = NormalGenerator(
        config=GeneratorConfig(
            sample_sizes=[
                N,
                N,
                N,
            ]
        ),
        means=[
            deviation,
            deviation,
            deviation,
        ],
        std=[
            1,
            1,
            1,
        ],
    )


    rf_model = ReferenceFModel(
        reference=0.0
    )


    engine = MonteCarloEngine()


    for _ in range(REPETITIONS):

        data = generator.generate()

        groups = data.groups


        # Reference F

        rf_result = rf_model.fit(
            groups
        )

        rf_rejects.append(
            rf_result.pvalue < ALPHA
        )


        # ANOVA

        anova_rejects.append(
            anova_pvalue(groups)
            <
            ALPHA
        )


        # Multiple reference tests

        ttest_rejects.append(
            multiple_ttest_pvalue(groups)
            <
            ALPHA
        )


    return {
        "deviation": deviation,

        "reference_f": np.mean(
            rf_rejects
        ),

        "anova": np.mean(
            anova_rejects
        ),

        "multiple_ttest": np.mean(
            ttest_rejects
        ),
    }


# ---------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------


def main():

    results = []


    for deviation in DEVIATIONS:

        result = run_scenario(
            deviation
        )

        results.append(
            result
        )

        print(result)


    with open(
        "method_comparison.csv",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "deviation,"
            "reference_f,"
            "anova,"
            "multiple_ttest\n"
        )

        for row in results:

            file.write(
                f"{row['deviation']},"
                f"{row['reference_f']},"
                f"{row['anova']},"
                f"{row['multiple_ttest']}\n"
            )


if __name__ == "__main__":

    main()
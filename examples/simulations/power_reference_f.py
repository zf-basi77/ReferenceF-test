"""
Power simulation for the Reference F-test.

Evaluates statistical power as a function of deviation
from the fixed reference value.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

import numpy as np

from referencef import (
    GeneratorConfig,
    NormalGenerator,
)

from referencef.model import (
    ReferenceFModel,
)

from referencef.simulate import (
    MonteCarloEngine,
)


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

REPETITIONS = 5000

ALPHA = 0.05

SAMPLE_SIZE = 30

DEVIATIONS = np.linspace(
    0,
    2,
    21,
)


# ---------------------------------------------------------------------
# Simulation function
# ---------------------------------------------------------------------


def estimate_power(
    deviation: float,
) -> float:
    """
    Estimate power for a given deviation.

    All groups are shifted equally away from
    the reference.
    """

    generator = NormalGenerator(
        config=GeneratorConfig(
            sample_sizes=[
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
            ],
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


    model = ReferenceFModel(
        reference=0.0
    )


    engine = MonteCarloEngine()


    result = engine.run(
        generator=generator,
        model=model,
        repetitions=REPETITIONS,
        alpha=ALPHA,
    )


    return result.rejection_rate


# ---------------------------------------------------------------------
# Run power curve
# ---------------------------------------------------------------------


def main():

    powers = []


    for deviation in DEVIATIONS:

        power = estimate_power(
            deviation
        )

        powers.append(
            power
        )

        print(
            f"Deviation={deviation:.2f}, "
            f"Power={power:.4f}"
        )


    results = np.column_stack(
        [
            DEVIATIONS,
            powers,
        ]
    )


    np.savetxt(
        "reference_f_power.csv",
        results,
        delimiter=",",
        header=(
            "deviation,power"
        ),
        comments="",
    )


if __name__ == "__main__":

    main()
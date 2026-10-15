"""
Type I error simulation for the Reference F-test.

This simulation evaluates whether the Reference F-test maintains
nominal false-positive rates under the null hypothesis.

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

from referencef.model import ReferenceFModel

from referencef.simulate import (
    MonteCarloEngine,
)

from referencef.metrics import (
    rejection_summary,
)


# ---------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------

SEED = 123

REPETITIONS = 10000

ALPHA = 0.05


# ---------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------

config = GeneratorConfig(
    sample_sizes=[
        20,
        20,
        20,
    ],
    seed=SEED,
)


generator = NormalGenerator(
    config=config,
    means=[
        0,
        0,
        0,
    ],
    std=[
        1,
        1,
        1,
    ],
)


# ---------------------------------------------------------------------
# Statistical model
# ---------------------------------------------------------------------

model = ReferenceFModel(
    reference=0.0
)


# ---------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------

engine = MonteCarloEngine()


result = engine.run(
    generator=generator,
    model=model,
    repetitions=REPETITIONS,
    alpha=ALPHA,
)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

summary = rejection_summary(
    [
        record.reject
        for record in result.records
    ]
)


print(result.summary())

print()

print("Type I Error Estimate")
print("--------------------")

print(
    f"Empirical alpha: "
    f"{summary['rejection_rate']:.5f}"
)

print(
    f"Monte Carlo SE: "
    f"{summary['monte_carlo_se']:.5f}"
)
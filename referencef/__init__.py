"""
referencef
==========

Reference F statistical simulation framework.

Author
------
ReferenceF Development Team
"""

from .defaults import register_default_generators
from .generators import (
    DataGenerator,
    GammaGenerator,
    GeneratorConfig,
    LogNormalGenerator,
    NormalGenerator,
    SimulationDataset,
    StudentTGenerator,
)
from .registry import GeneratorRegistry
from .model import ReferenceFModel, ReferenceFResult
from .effectsize import calculate_effect_sizes
from .posthoc import reference_contrasts

__version__ = "0.1.0"

__all__ = [
    # Core analysis
    "ReferenceFModel",
    "ReferenceFResult",

    # Effect sizes
    "calculate_effect_sizes",

    # Post-hoc reference contrasts
    "reference_contrasts",

    # Simulation framework
    "DataGenerator",
    "GammaGenerator",
    "GeneratorConfig",
    "GeneratorRegistry",
    "LogNormalGenerator",
    "NormalGenerator",
    "SimulationDataset",
    "StudentTGenerator",

    # Utilities
    "register_default_generators",
]
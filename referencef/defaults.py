"""
referencef.defaults
===================

Default generator registrations.

Importing this module registers the built-in generators with the
GeneratorRegistry.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from .generators import (
    GammaGenerator,
    LogNormalGenerator,
    NormalGenerator,
    StudentTGenerator,
)

from .registry import GeneratorRegistry


def register_default_generators() -> None:
    """
    Register all built-in data generators.

    This function is idempotent. Existing registrations are overwritten
    intentionally so that package initialization always produces a
    consistent registry state.
    """

    GeneratorRegistry.register(
        "normal",
        NormalGenerator,
        overwrite=True,
    )

    GeneratorRegistry.register(
        "student_t",
        StudentTGenerator,
        overwrite=True,
    )

    GeneratorRegistry.register(
        "lognormal",
        LogNormalGenerator,
        overwrite=True,
    )

    GeneratorRegistry.register(
        "gamma",
        GammaGenerator,
        overwrite=True,
    )


register_default_generators()
"""
referencef.types
================

Common type aliases used throughout the ReferenceF package.

Author
------
ReferenceF Development Team
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------
# NumPy arrays
# ---------------------------------------------------------------------

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int_]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# ---------------------------------------------------------------------
# Dataset representation
# ---------------------------------------------------------------------

Group: TypeAlias = FloatArray
Dataset: TypeAlias = list[Group]
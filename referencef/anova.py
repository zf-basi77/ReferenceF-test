"""
referencef.anova

One-way ANOVA model for simulation comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(slots=True)
class ANOVAResult:
    statistic: float
    pvalue: float


class OneWayANOVA:
    """
    Classical one-way ANOVA.

    Tests:

        H0: all group means are equal
    """

    def fit(
        self,
        groups: Sequence[np.ndarray],
    ) -> ANOVAResult:

        statistic, pvalue = stats.f_oneway(
            *groups
        )

        return ANOVAResult(
            statistic=float(statistic),
            pvalue=float(pvalue),
        )
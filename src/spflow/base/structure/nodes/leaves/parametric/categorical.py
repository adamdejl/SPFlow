"""
Created on December 04, 2021

@authors Bennet Wittelsbach
"""

from typing import List, Optional

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
import numpy as np


class Categorical(ParametricLeaf):
    """(Univariate) Categorial distribution.

    PMF(k) = p[k]

    Attributes:
        p:
            Array of size k of probabilities of each category. Each value between 0 and 1. Sums up to 1.
    """

    type = ParametricType.CATEGORICAL

    def __init__(self, scope: List[int], p: Optional[List[float]] = None) -> None:
        if len(scope) != 1:
            raise ValueError(
                f"Scope size for {self.__class__.__name__} should be 1, but was: {len(scope)}"
            )

        super().__init__(scope)

        if p is None:
            k = np.random.randint(
                3, 10
            )  # sample at least 3 categories, less than 3 would be Bernoulli
            p = np.random.uniform(low=0.0, high=1.0, size=k)
            p = p / np.sum(p)  # normalize values to sum up to 1

        self.set_params(p)

    def set_params(self, p: List[float]) -> None:
        if np.any((p <= [0.0]) | (p >= [1.0])):
            raise ValueError(
                f"Values of p for Categorical distribution must be between 0 and 1, but were: {p}"
            )
        if not np.isclose(np.sum(p), 1.0):
            raise ValueError(
                f"Values of p for Categorical distribution must sum up to 1, but were: {np.sum(p)}"
            )

        self.p = p
        self.k = len(p)

    def get_params(self) -> List[int]:
        return self.p

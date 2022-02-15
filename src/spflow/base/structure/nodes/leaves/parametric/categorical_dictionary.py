"""
Created on December 07, 2021

@authors Bennet Wittelsbach
"""

from typing import Dict, List, Optional

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
import numpy as np


class CategoricalDictionary(ParametricLeaf):
    """(Univariate) Categorial distribution.

    PMF(k) = p[k]

    Attributes:
        p:
            Array of size k of probabilities of each category. Each value between 0 and 1. Sums up to 1.
    """

    type = ParametricType.CATEGORICAL

    def __init__(self, scope: List[int], p: Optional[Dict[int, float]] = None) -> None:
        if len(scope) != 1:
            raise ValueError(
                f"Scope size for {self.__class__.__name__} should be 1, but was: {len(scope)}"
            )

        super().__init__(scope)

        if p is None:
            k = np.random.randint(
                3, 10
            )  # sample at least 3 categories, less than 3 would be Bernoulli
            r = np.random.uniform(low=0.0, high=1.0, size=k)
            r = r / np.sum(r)  # normalize values to sum up to 1
            p = dict(zip(list(range(k)), r.tolist()))

        self.set_params(p)

    def set_params(self, p: Dict[int, float]) -> None:
        vals = list(p.values())
        if np.any((vals <= [0.0]) | (vals >= [1.0])):
            raise ValueError(
                f"Values of p for Categorical distribution must be between 0 and 1, but were: {p}"
            )
        if not np.isclose(np.sum(vals), 1.0):
            raise ValueError(
                f"Values of p for Categorical distribution must sum up to 1, but were: {np.sum(p)}"
            )

        self.p = p

    def get_params(self) -> List[int]:
        return self.p

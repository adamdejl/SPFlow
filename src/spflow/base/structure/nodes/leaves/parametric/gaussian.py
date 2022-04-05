"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Optional, Tuple, Dict, List
import numpy as np
from scipy.stats import norm  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Gaussian(ParametricLeaf):
    """(Univariate) Normal distribution.

    PDF(x) =
        1/sqrt(2*pi*sigma^2) * exp(-(x-mu)^2/(2*sigma^2)), where
            - x is an observation
            - mu is the mean
            - sigma is the standard deviation

    Attributes:
        mean:
            mean (mu) of the distribution.
        stdev:
            standard deviation (sigma) of the distribution.
    """

    type = ParametricType.CONTINUOUS

    def __init__(
        self, scope: List[int], mean: Optional[float] = None, stdev: Optional[float] = None
    ) -> None:
        if len(scope) != 1:
            raise ValueError(
                f"Scope size for {self.__class__.__name__} should be 1, but was: {len(scope)}"
            )

        super().__init__(scope)

        if mean is None:
            mean = np.random.uniform(-1.0, 1.0)
        if stdev is None:
            stdev = np.random.uniform(1.0, 3.0)

        self.set_params(mean, stdev)

    def set_params(self, mean: float, stdev: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for Gaussian distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for Gaussian distribution must be greater than 0.0, but was: {stdev}"
            )
        if get_scipy_object(self).pdf(mean, loc=mean, scale=stdev) >= 1.0:
            print(
                f"Warning: 'Degenerated' PDF! Density at node.mean is greater than 1.0! Node: {self}, mean: {mean}, stdev: {stdev}"
            )

        self.mean = mean
        self.stdev = stdev

    def get_params(self) -> Tuple[float, float]:
        return self.mean, self.stdev


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_scipy_object(node: Gaussian) -> rv_continuous:
    return norm


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Gaussian) -> Dict[str, float]:
    if node.mean is None:
        raise InvalidParametersError(f"Parameter 'mean' of {node} must not be None")
    if node.stdev is None:
        raise InvalidParametersError(f"Parameter 'stdev' of {node} must not be None")
    parameters = {"loc": node.mean, "scale": node.stdev}
    return parameters


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_natural_parameters(node: Gaussian) -> np.ndarray:
    natural_parameters = [node.mean / (node.stdev ** 2), -1 / (2 * node.stdev ** 2)]
    return np.array(natural_parameters)


@dispatch(Gaussian, np.ndarray)  # type: ignore[no-redef]
def get_base_measure(node: Gaussian, X: np.ndarray) -> np.ndarray:
    base_measure = [0.398942280401432677939946059934]  # constant: equals 1 / np.sqrt(2*np.pi)
    return np.array(base_measure)


@dispatch(Gaussian, np.ndarray)  # type: ignore[no-redef]
def get_sufficient_statistics(node: Gaussian, X: np.ndarray) -> np.ndarray:
    sufficient_statistics = [[x, x ** 2] for x in X]
    return np.array(sufficient_statistics)


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_log_partition_natural(node: Gaussian) -> float:
    natural_parameters = get_natural_parameters(node)
    log_partition = (-natural_parameters[0] ** 2 / (4 * natural_parameters[1])) - 1 / 2 * np.log(
        -2 * natural_parameters[1]
    )
    return log_partition


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_log_partition_param(node: Gaussian) -> float:
    log_partition = (node.mean ** 2) / (2 * node.stdev ** 2) + np.log(node.stdev)
    return log_partition


@dispatch(Gaussian, np.ndarray, np.ndarray)  # type: ignore[no-redef]
def update_parameters_em(
    node: Gaussian, data: np.ndarray, responsibilities: np.ndarray, hard_em: bool
) -> None:
    responsilbility_factor = 1 / np.sum(responsibilities)
    mean = responsilbility_factor * (responsibilities.T @ data[:, node.scope])
    diff = data[:, node.scope] - mean
    variance = responsilbility_factor * (responsibilities.T @ np.square(diff))
    stdev = np.sqrt(variance)
    node.set_params(mean, stdev)

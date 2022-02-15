"""
Created on July 01, 2021

@authors: Bennet Wittelsbach
"""

import numpy as np # type: ignore
from multipledispatch import dispatch # type: ignore
from spflow.base.structure.nodes.leaves.parametric.exceptions import (
    InvalidParametersError,
    NotViableError,
)
from spflow.base.structure.nodes.leaves.parametric import (
    Gaussian,
    MultivariateGaussian,
    LogNormal,
    Bernoulli,
    Binomial,
    NegativeBinomial,
    Poisson,
    Geometric,
    Hypergeometric,
    Categorical,
    CategoricalDictionary,
    Exponential,
    Gamma,
)
from spflow.base.structure.nodes.node import INode
from scipy.stats import lognorm, gamma  # type: ignore


# TODO: design decision: set mle params directly _in node_ or return them? first approach currently implemented
# TODO: design decision: _numpy arrays_ or default lists?

# TODO: update typing (see when numpy typing became available and if it collides with current requirements)
@dispatch(INode, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: INode, data: np.ndarray) -> None:
    """Compute the parameters of the distribution represented by the node via MLE, if an closed-form estimator is available.

    Arguments:
        node:
            The node which parameters are to be estimated.
        data:
            A 2-dimensional numpy-array holding the observations the parameters are to be estimated from.

    Raises:
        NotViableError:
            There is no (closed-form) maximum-likelihood estimator available for the type of distribution represented by node.
    """
    return NotViableError(f"There is no (closed-form) MLE for {node} implemented or existent")


@dispatch(Gaussian, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Gaussian, data: np.ndarray) -> None:
    data = validate_data(data, 1)
    mean = np.mean(data).item()
    stdev = np.std(data).item()

    if np.isclose(stdev, 0):
        stdev = 1e-8

    node.set_params(mean, stdev)


@dispatch(MultivariateGaussian, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: MultivariateGaussian, data: np.ndarray) -> None:
    data = validate_data(data, data.shape[1])
    if data.shape[1] == 1:
        print(f"Warning: Trying to estimate MultivarateGaussian, but data has shape {data.shape}")

    mean_vector = np.mean(data, axis=0).tolist()
    covariance_matrix = np.cov(data, rowvar=0).tolist()

    # check for univariate degeneracy
    # for i in range(len(node.mean_vector)):
    #   [if pdf(mean) >= 1: print warning]

    node.set_params(mean_vector, covariance_matrix)


@dispatch(LogNormal, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: LogNormal, data: np.ndarray) -> None:
    data = validate_data(data, 1)

    # originally written by Alejandro Molina
    parameters = lognorm.fit(data, floc=0)
    mean = np.log(parameters[2]).item()
    stdev = parameters[0]

    node.set_params(mean, stdev)


@dispatch(Bernoulli, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Bernoulli, data: np.ndarray) -> None:
    data = validate_data(data, 1)
    p = data.sum().item() / len(data)

    node.set_params(p)


@dispatch(Binomial, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Binomial, data: np.ndarray) -> None:
    data = validate_data(data, 1)
    n = len(data)
    p = data.sum().item() / (len(data) ** 2)

    node.set_params(n, p)


@dispatch(NegativeBinomial, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: NegativeBinomial, data: np.ndarray) -> None:
    raise NotViableError(
        "The Negative Binomal distribution parameters 'n, p' cannot be estimated via Maximum-Likelihood Estimation"
    )


@dispatch(Poisson, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Poisson, data: np.ndarray) -> None:
    data = validate_data(data, 1)
    l = np.mean(data).item()
    node.set_params(l)


@dispatch(Geometric, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Geometric, data: np.ndarray) -> None:
    data = validate_data(data, 1)
    p = len(data) / data.sum().item()
    node.set_params(p)


@dispatch(Hypergeometric, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Hypergeometric, data: np.ndarray) -> None:
    raise NotViableError(
        "The Hypergeometric distribution parameters 'M, N, n' cannot be estimated via Maximum-Likelihood Estimation"
    )


@dispatch(Categorical, np.ndarray) # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Categorical, data:np.ndarray) -> None:
    p = [0] * node.k
    for i in range(node.k):
        p[i] = np.sum(data == i)
    p = p / np.sum(p)
    p = p.tolist()
    node.set_params(p)


@dispatch(CategoricalDictionary, np.ndarray) # type: ignore[no-redef]
def maximum_likelihood_estimation(node: CategoricalDictionary, data:np.ndarray) -> None:
    v, c = np.unique(data, return_counts=True)
    r = c / np.sum(c)
    p = dict(zip(v.tolist(), r.tolist()))
    node.set_params(p)


@dispatch(Exponential, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Exponential, data: np.ndarray) -> None:
    data = validate_data(data, 1)
    l = np.mean(data).item()
    node.set_params(l)


@dispatch(Gamma, np.ndarray)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Gamma, data: np.ndarray) -> None:
    data = validate_data(data, 1)

    # default, originally written by Alejandro Molina
    alpha = 1.1
    beta = 1.0
    if np.any(data <= 0):
        # negative data? impossible gamma
        raise InvalidParametersError("All 'data' entries must not be 0")

    # zero variance? adding noise
    if np.isclose(np.std(data), 0):
        alpha = np.mean(data).item()
        print(f"Warning: {node} has 0 variance, adding noise")

    alpha, loc, theta = gamma.fit(data, floc=0)
    beta = 1.0 / theta
    if np.isfinite(alpha):
        node.set_params(alpha, beta)
    else:
        raise InvalidParametersError(f"{node}: 'alpha' is not finite, parameters were NOT set")


def validate_data(
    data: np.ndarray, expected_dimensions: int, remove_nan: bool = True
) -> np.ndarray:
    """Checking the data before using it for maximum-likelihood estimation.

    Arguments:
        data:
            A 2-dimensional numpy-array holding the data used for maximum-likelihood estimation
        expected_dimensions:
            The dimensions of the data and the distribution that is to be estimated. Equals 1 for all
            univariate distributions, else the number of dimensions of a multivariate distribution
        remove_nan:
            Boolean if nan entries (according to numpy.isnan()) shall be removed or kept.

    Returns:
        The validated and possibly cleaned up two-dimensional data numpy-array

    Raises:
        ValueError:
            If the shape of 'data' does not match 'expected' dimensions or is 0.
    """
    if data.shape[0] == 0 or data.shape[1] != expected_dimensions:
        raise ValueError(f"Argument 'data' must have shape of form (>0, {expected_dimensions}).")
    if np.any(np.isnan(data)):
        print("Warning: Argument 'data' contains NaN values that are removed")
        if remove_nan:
            data = data[~np.isnan(data)]

    return data

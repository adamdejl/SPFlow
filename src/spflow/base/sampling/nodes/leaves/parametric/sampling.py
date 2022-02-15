"""
Created on August 09, 2021

@authors: Kevin Huy Nguyen

This file provides the sampling methods for parametric leaves.
"""
from numpy.random import RandomState
from spflow.base.structure.nodes.node import ILeafNode, INode
from multipledispatch import dispatch  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import (
    ParametricLeaf,
    Gaussian,
    Gamma,
    Poisson,
    LogNormal,
    Geometric,
    Exponential,
    Bernoulli,
    Categorical,
    CategoricalDictionary,
    get_scipy_object_parameters,
    get_scipy_object,
)
import numpy as np

# TODO Binomial, NegativeBinomial, Hypergeometric, MultivariateGaussian


@dispatch(INode)  # type: ignore[no-redef]
def sample_parametric_node(node: INode, n_samples, rand_gen) -> None:
    """Sample from the associated scipy object of a parametric leaf node.

    The standard implementation accepts nodes of any type and raises an error, if there is no sampling
    procedure implemented for the given node or if the number of wanted samples is not bigger than zero.

    Arguments:
        node:
            The node which is to be sampled.
        n_samples:
            Number of samples to be generated per node.
        rand_gen:
            Seed for random number generator.


    Returns:
        A scipy object representing the distribution of the given node, or None.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide a scipy object or the node is not a ILeafNode
            and cannot provide a scipy object.

    """
    assert n_samples > 0
    assert isinstance(node, ParametricLeaf)

    if type(node) is ILeafNode:
        raise NotImplementedError(f"{node} does not provide a scipy object")
    else:
        raise NotImplementedError(f"{node} cannot provide scipy objects")


@dispatch(Gaussian)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Gaussian, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Gamma)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Gamma, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(LogNormal)  # type: ignore[no-redef]
def sample_parametric_node(
    node: LogNormal, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Poisson)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Poisson, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Geometric)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Geometric, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Exponential)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Exponential, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Bernoulli)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Bernoulli, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Categorical)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Categorical, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    X = rand_gen.choice(np.arange(node.k), p=node.p, size=n_samples)
    return X


@dispatch(CategoricalDictionary)  # type: ignore[no-redef]
def sample_parametric_node(
    node: CategoricalDictionary, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    vals = []
    ps = []

    for v, p in node.p.items():
        vals.append(v)
        ps.append(p)
    X = rand_gen.choice(vals, p=ps, size=n_samples)
    return X

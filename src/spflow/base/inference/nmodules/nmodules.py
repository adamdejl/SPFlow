"""
Created on July 08, 2021

@authors: Kevin Huy Nguyen

This file provides the inference methods for SPNs.
"""
import numpy as np
from numpy import ndarray
from scipy.special import logsumexp  # type: ignore
from spflow.base.structure.network_type import SPN, NetworkType
from spflow.base.inference.nodes.leaves.parametric import node_log_likelihood
from typing import List, Callable, Type, Dict
from multipledispatch import dispatch  # type: ignore
from spflow.base.memoize import memoize
from spflow.base.structure.nmodule import NLeafNode, NProductNode, NSumNode, NDummy
from spflow.base.structure.nmodules.nmodule import ProductLayer, SumLayer


@dispatch(NSumNode, np.ndarray, SPN, cache=dict)
@memoize(NSumNode)
def log_likelihood(
    module: NSumNode,
    data: np.ndarray,
    network_type: NetworkType,
    cache: Dict = {},
) -> np.ndarray:
    """
    Recursively calculates the log_likelihood for a ISumNode.
    It calls log_likelihood on all it children to calculate its own value by using the fitting evaluation function.

    Args:
        node:
            ISumNode to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for ISumNode.
    """
    inputs = []
    for child in module.children:
        if type(child) is NDummy:
            children = child.ref_module.get_mod_children(module)
            for node_child in children:
                inputs.append(log_likelihood(node_child, data, SPN(), cache=cache))
        else:
            inputs.append(log_likelihood(child, data, SPN(), cache=cache))

    llchildren: ndarray = np.concatenate(inputs, axis=1)
    b: ndarray = module.weights
    sll: ndarray = logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)

    return sll


@dispatch(NProductNode, np.ndarray, SPN, cache=dict)
@memoize(NProductNode)
def log_likelihood(
    module: NProductNode,
    data: np.ndarray,
    network_type: NetworkType,
    cache: Dict = {},
) -> np.ndarray:
    """
    Recursively calculates the log_likelihood for a IProdNode.
    It calls log_likelihood on all it children to calculate its own value by using the fitting evaluation function.

    Args:
        node:
            IProdNode to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for IProdNode.
    """
    inputs = []
    for child in module.children:
        if type(child) is NDummy:
            children = child.ref_module.get_mod_children(module)
            for node_child in children:
                inputs.append(log_likelihood(node_child, data, SPN(), cache=cache))
        else:
            inputs.append(log_likelihood(child, data, SPN(), cache=cache))

    llchildren: ndarray = np.concatenate(inputs, axis=1)
    pll: ndarray = np.sum(llchildren, axis=1).reshape(-1, 1)
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min

    return pll


@dispatch(NLeafNode, np.ndarray, SPN, cache=dict)
@memoize(NLeafNode)
def log_likelihood(
    module: NLeafNode,
    data: np.ndarray,
    network_type: NetworkType,
    cache: Dict = {},
) -> np.ndarray:
    """
    Calculates log_likelihood for a ILeafNode by evaluating the distribution represented by the leaf node type.

    Args:
        node:
            ILeafNode node to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for ILeafNode.
    """
    return node_log_likelihood(module.leaf, data=data)


@dispatch(ProductLayer, np.ndarray, SPN, cache=dict)
@memoize(ProductLayer)
def log_likelihood(
    module: ProductLayer,
    data: np.ndarray,
    network_type: NetworkType,
    cache: Dict = {},
) -> np.ndarray:
    """
    Calculates log_likelihood for a ILeafNode by evaluating the distribution represented by the leaf node type.

    Args:
        node:
            ILeafNode node to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for ILeafNode.
    """
    plls = []
    for prod in module.output_modules:
        plls.append(log_likelihood(prod, data, SPN(), cache=cache))
    llchildren: ndarray = np.concatenate(plls, axis=1)
    return llchildren


@dispatch(SumLayer, np.ndarray, SPN, cache=dict)
@memoize(SumLayer)
def log_likelihood(
    module: SumLayer,
    data: np.ndarray,
    network_type: NetworkType,
    cache: Dict = {},
) -> np.ndarray:
    """
    Calculates log_likelihood for a ILeafNode by evaluating the distribution represented by the leaf node type.

    Args:
        node:
            ILeafNode node to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for ILeafNode.
    """
    plls = []
    for sum in module.output_modules:
        plls.append(log_likelihood(sum, data, SPN(), cache=cache))
    llchildren: ndarray = np.concatenate(plls, axis=1)
    return llchildren

# type: ignore
from typing import Callable, Dict
import numpy as np
from scipy.special import logsumexp

from spflow.base.structure.nodes.node import (
    ILeafNode,
    INode,
    IProductNode,
    ISumNode,
    eval_spn_top_down,
)


def merge_gradients(parent_gradients: np.ndarray) -> float:
    return logsumexp(np.concatenate(parent_gradients).reshape(-1, 1), axis=1)


def leaf_gradient_backward(
    node: ILeafNode,
    parent_result: np.ndarray,
    gradient_result: np.ndarray = None,
    lls_per_node: np.ndarray = None,
) -> None:
    parent_gradients = merge_gradients(parent_result)

    gradients = np.zeros((parent_gradients.shape[0]))
    gradients[:] = parent_gradients

    gradient_result[:, node.id] = gradients


def sum_gradient_backward(
    node: ISumNode,
    parent_result: np.ndarray,
    gradient_result: np.ndarray = None,
    lls_per_node: np.ndarray = None,
) -> Dict[INode, np.ndarray]:
    parent_gradients = merge_gradients(parent_result)

    gradients = np.zeros((parent_gradients.shape[0]))
    gradients[:] = parent_gradients

    gradient_result[:, node.id] = gradients

    messages_to_children = {}
    wlog = np.log(node.weights)

    for i, c in enumerate(node.children):
        children_gradient = gradients + wlog[i]
        children_gradient[np.isinf(children_gradient)] = np.finfo(gradient_result.dtype).min
        messages_to_children[c] = children_gradient

        assert not np.any(np.isnan(children_gradient)), "Nans found in iteration"
        assert not np.any(np.isinf(children_gradient)), "inf found in iteration"

    return messages_to_children


def prod_gradient_backward(
    node: IProductNode,
    parent_result: np.ndarray,
    gradient_result: np.ndarray = None,
    lls_per_node: np.ndarray = None,
) -> Dict[INode, np.ndarray]:
    parent_gradients = merge_gradients(parent_result)

    gradients = np.zeros((parent_gradients.shape[0]))
    gradients[:] = parent_gradients

    gradient_result[:, node.id] = gradients

    messages_to_children = {}

    output_ll = lls_per_node[:, node.id]

    for i, c in enumerate(node.children):
        children_gradient = gradients + output_ll - lls_per_node[:, c.id]
        children_gradient[np.isinf(children_gradient)] = np.finfo(gradient_result.dtype).min
        messages_to_children[c] = children_gradient

        assert not np.any(np.isnan(children_gradient)), "Nans found in iteration"
        assert not np.any(np.isinf(children_gradient)), "inf found in iteration"

    return messages_to_children


_node_gradients = {
    ISumNode: sum_gradient_backward,
    IProductNode: prod_gradient_backward,
    ILeafNode: leaf_gradient_backward,
}


def add_node_gradient(node_type: type, lambda_func: Callable) -> None:
    _node_gradients[node_type] = lambda_func


def gradient_backward(
    spn: INode, lls_per_node: np.ndarray, node_gradients: np.ndarray = _node_gradients
) -> np.ndarray:
    gradient_result = np.zeros_like(lls_per_node)

    eval_spn_top_down(
        spn,
        node_gradients,
        parent_result=np.zeros((lls_per_node.shape[0])),
        gradient_result=gradient_result,
        lls_per_node=lls_per_node,
    )

    return gradient_result

"""
Created on December 07, 2021

@authors Bennet Wittelsbach
"""

from spflow.base.structure.nodes.leaves.parametric import CategoricalDictionary
from multipledispatch import dispatch  # type: ignore
import numpy as np


@dispatch(CategoricalDictionary, data=np.ndarray)  # type: ignore[no-redef]
def node_likelihood(node: CategoricalDictionary, data: np.ndarray) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]

    dict_probs = [node.p.get(val, 0.0) for val in observations]
    probs[~marg_ids] = dict_probs
    return probs


@dispatch(CategoricalDictionary, data=np.ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: CategoricalDictionary, data: np.ndarray) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]

    dict_probs = [np.log(node.p.get(val, 0.0)) for val in observations]
    probs[~marg_ids] = dict_probs
    return probs

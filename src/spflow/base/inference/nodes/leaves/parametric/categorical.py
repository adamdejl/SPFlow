"""
Created on December 04, 2021

@authors Bennet Wittelsbach
"""

from spflow.base.structure.nodes.leaves.parametric import Categorical
from multipledispatch import dispatch # type: ignore
import numpy as np

@dispatch(Categorical, data=np.ndarray) # type: ignore[no-redef]
def node_likelihood(node: Categorical, data: np.ndarray) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]

    categorical_data = observations.astype(np.int64)
    assert np.all(np.equal(np.mod(categorical_data, 1), 0))
    out_domain_ids = categorical_data >= node.k
    idx_out = ~marg_ids
    idx_out[idx_out] = out_domain_ids
    probs[idx_out] = 1

    idx_in = ~marg_ids
    idx_in[idx_in] = ~out_domain_ids
    probs[idx_in] = np.array(node.p)[categorical_data[~out_domain_ids]]
    return probs

@dispatch(Categorical, data=np.ndarray) # type: ignore[no-redef]
def node_log_likelihood(node:Categorical, data: np.ndarray) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]

    categorical_data = observations.astype(np.int64)
    assert np.all(np.equal(np.mod(categorical_data, 1), 0))
    out_domain_ids = categorical_data >= node.k
    idx_out = ~marg_ids
    idx_out[idx_out] = out_domain_ids
    probs[idx_out] = 0

    idx_in = ~marg_ids
    idx_in[idx_in] = ~out_domain_ids
    probs[idx_in] = np.array(np.log(node.p))[categorical_data[~out_domain_ids]]
    return probs
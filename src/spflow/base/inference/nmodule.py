"""
Created on November 27, 2021

@authors: Kevin Huy Nguyen

"""

from spflow.base.memoize import memoize
import numpy as np
from typing import Dict
from multipledispatch import dispatch  # type: ignore
from spflow.base.structure.nmodule import NModule
from spflow.base.inference.nodes.node_modules import log_likelihood
from spflow.base.inference.rat.rat_spn import log_likelihood
from spflow.base.inference.nmodules.nmodules import log_likelihood


@dispatch(NModule, np.ndarray, cache=dict)
@memoize(NModule)
def log_likelihood(module: NModule, data: np.ndarray, cache: Dict = {}) -> np.ndarray:
    return log_likelihood(module, data, module.network_type, cache=cache)


@dispatch(NModule, np.ndarray, cache=dict)
@memoize(NModule)
def likelihood(module: NModule, data: np.ndarray, cache: Dict = {}) -> np.ndarray:
    return np.exp(log_likelihood(module, data, module.network_type, cache=cache))

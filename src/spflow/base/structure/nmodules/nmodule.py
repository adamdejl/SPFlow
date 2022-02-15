"""
Created on October 21, 2021

@authors: Kevin Huy Nguyen
"""
from multipledispatch import dispatch  # type: ignore
from typing import List, Optional
from spflow.base.structure.network_type import NetworkType
from spflow.base.structure.nmodule import (
    _get_node_counts,
    NModule,
    NDummy,
    NNode,
    NSumNode,
    NProductNode,
    NLeafNode,
    _get_child_output_nnodes,
)
import numpy as np


class ProductLayer(NModule):
    def __init__(
        self,
        num_of_nodes: int,
        num_of_children: int,
        scope: List[int],
        children: List["NModule"],
        network_type: Optional[NetworkType] = None,
    ) -> None:
        super().__init__(children=children, network_type=network_type)
        self.scope = scope
        self.num_of_nodes = num_of_nodes
        self.num_of_children = num_of_children

        assert self.children is not None
        assert None not in self.children

        # create the ProdNodes in this layer, all are output nodes, since there is only one layer of nodes
        output_nodes = []
        for i in range(0, num_of_nodes):
            output_nodes.append(NProductNode(scope=scope, children=[NDummy(ref_module=self)]))
        self.output_modules = output_nodes

        # check if the child modules fit
        assert len(_get_child_output_nnodes(self)) == num_of_children * num_of_nodes

    def __len__(self):
        return 1

    def get_mod_children(self, module: NProductNode) -> List[NModule]:
        assert module in self.output_modules
        index = self.output_modules.index(module)
        return _get_child_output_nnodes(self)[
            index * self.num_of_children : (index + 1) * self.num_of_children
        ]


class SumLayer(NModule):
    def __init__(
        self,
        num_of_nodes: int,
        weights: np.ndarray,
        scope: List[int],
        children: List["NModule"],
        network_type: Optional[NetworkType] = None,
    ) -> None:
        super().__init__(children=children, network_type=network_type)
        self.scope = scope
        self.num_of_nodes = num_of_nodes
        self.num_of_children = weights.shape[0]
        self.weights = weights

        assert len(scope) == num_of_nodes
        assert self.children is not None
        assert None not in self.children

        # create the ProdNodes in this layer, all are output nodes, since there is only one layer of nodes
        output_nodes = []
        for i in range(0, num_of_nodes):
            output_nodes.append(
                NSumNode(scope=[scope[i]], weights=weights, children=[NDummy(ref_module=self)])
            )
        self.output_modules = output_nodes

        # check if the child modules fit
        assert len(_get_child_output_nnodes(self)) == self.num_of_children * num_of_nodes

    def __len__(self):
        return 1

    def get_mod_children(self, module: NSumNode) -> List[NModule]:
        assert module in self.output_modules
        index = self.output_modules.index(module)
        return _get_child_output_nnodes(self)[
            index * self.num_of_children : (index + 1) * self.num_of_children
        ]


@dispatch(ProductLayer, cache=list)  # type: ignore[no-redef]
def _get_node_counts(module: ProductLayer, cache: Optional[list] = None) -> np.ndarray:
    """Wrapper for Modules"""
    if not cache:
        cache = []
    if module in cache:
        result = [np.array((0, 0, 0))]
    else:
        cache.append(module)
        result = [np.array((0, module.num_of_nodes, 0))]
    for child in module.children:
        result += _get_node_counts(child, cache=cache)
    return sum(result)


@dispatch(SumLayer, cache=list)  # type: ignore[no-redef]
def _get_node_counts(module: SumLayer, cache: Optional[list] = None) -> np.ndarray:
    """Wrapper for Modules"""
    if not cache:
        cache = []
    if module in cache:
        result = [np.array((0, 0, 0))]
    else:
        cache.append(module)
        result = [np.array((module.num_of_nodes, 0, 0))]
    for child in module.children:
        result += _get_node_counts(child, cache=cache)
    return sum(result)

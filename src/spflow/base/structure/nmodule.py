"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures.
"""
from abc import ABC, abstractmethod
import spflow
from spflow.base.structure.network_type import NetworkType
from typing import List, Optional, cast, Dict
from multipledispatch import dispatch  # type: ignore
import numpy as np
from spflow.base.structure.nodes.node import (
    INode,
)
from spflow.base.learning.context import RandomVariableContext  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import MultivariateGaussian


class NModule(ABC):
    """Abstract module class for building graph structures.

    Attributes:
        children:
            List of child modules to form a graph of modules.
        nodes:
            List of INodes representing all Inodes encapsulated by this module.
        network_type:
            Network Type defining methods to use on this module.
        output_nodes:
            List of INodes representing the root nodes of the module to connect it to other modules.
        scope:
            List of ints representing the scope of all nodes in this module.
    """

    def __init__(
        self,
        children: List["NModule"],
        network_type: Optional[NetworkType] = None,
    ) -> None:

        # Set network type - if none is specified, get default global network type
        if network_type is None:
            self.network_type = spflow.get_network_type()
        else:
            self.network_type = network_type
            self.network_type = cast(NetworkType, self.network_type)

        self.output_modules: List["NModule"] = [self]
        self.children: List["NModule"] = children

    @abstractmethod
    def __len__(self):
        pass


class NDummy(NModule):
    def __init__(self, ref_module: NModule) -> None:
        super().__init__(children=[], network_type=None)

        self.ref_module: NModule = ref_module

    def __len__(self):
        return 0


class NNode(NModule):
    """Base class for all types of nodes modules

    Attributes:
        output_nodes:
            List of one INode as Node modules only encapsulate a single INode, which is at the same time the root.
        nodes:
            List of one INode as Node modules only encapsulate a single INode
    """

    def __init__(
        self,
        scope: List[int],
        children: Optional[List["NModule"]],
        network_type: Optional[NetworkType] = None,
    ) -> None:
        super().__init__(children=children, network_type=network_type)
        self.output_modules: None = None
        self.scope = scope

    def __len__(self):
        return 1


class NSumNode(NNode):
    """SumNode is module encapsulating one ISumNode.

    Args:
        weights:
            A np.array of floats assigning a weight value to each of the encapsulated ISumNode's children.
    """

    def __init__(
        self,
        scope: List[int],
        weights: np.ndarray,
        children: List["NModule"],
        network_type: Optional[NetworkType] = None,
    ) -> None:
        super().__init__(children=children, network_type=network_type, scope=scope)

        # check if all children are Modules and all values in weights are floats
        assert all([issubclass(type(obj), float) for obj in weights])

        # check if weights sum to 1
        assert np.isclose(sum(weights), 1.0)

        # None as this module can not wrap any other module and its output is itself
        self.weights = weights
        self.output_modules: List[NModule] = [self]

        assert self.children is not None
        assert None not in self.children

    def __len__(self):
        return 1


class NProductNode(NNode):
    """ProductNode is module encapsulating one IProductNode."""

    def __init__(
        self,
        scope: List[int],
        children: List["NModule"],
        network_type: Optional[NetworkType] = None,
    ) -> None:
        super().__init__(children=children, network_type=network_type, scope=scope)
        self.output_modules: List[NModule] = [self]
        assert self.children is not None
        assert None not in self.children

    def __len__(self):
        return 1


class NLeafNode(NNode):
    """LeafNode is module encapsulating one ILeafNode.

    Args:
        child:
            Empty list as LeafNodes can not have children.
    """

    def __init__(
        self,
        scope: List[int],
        context: RandomVariableContext,
        network_type: Optional[NetworkType] = None,
    ) -> None:
        super().__init__(children=[], network_type=network_type, scope=scope)
        if len(scope) == 1:
            try:
                leaf = context.parametric_types[scope[0]](scope=scope)
            except IndexError:
                raise IndexError(
                    "Leaf scope outside of scopes specified for parametric types in context."
                )
        else:
            leaf = MultivariateGaussian(
                scope=scope,
                mean_vector=np.zeros(len(scope)),
                covariance_matrix=np.eye(len(scope)),
            )
        self.leaf: INode = leaf
        self.output_modules: List[NModule] = [self]

    def __len__(self):
        return 1


@dispatch(NModule, cache=list)  # type: ignore[no-redef]
def _get_node_counts(module: NModule, cache: Optional[list] = None) -> np.ndarray:
    """Wrapper for Modules"""
    if not cache:
        cache = []
    res_list = [np.array((0, 0, 0))]
    for out_module in module.output_modules:
        res_list += _get_node_counts(
            out_module,
            cache=cache,
        )
    for child in module.children:
        res_list += _get_node_counts(
            child,
            cache=cache,
        )
    return sum(res_list)


@dispatch(NSumNode, cache=list)  # type: ignore[no-redef]
def _get_node_counts(module: NSumNode, cache: Optional[list] = None) -> np.ndarray:
    """Wrapper for Modules"""
    if not cache:
        cache = []
    if module in cache:
        result = [np.array((0, 0, 0))]
    else:
        cache.append(module)
        result = [np.array((1, 0, 0))]

    for child in module.children:
        result += _get_node_counts(
            child,
            cache=cache,
        )
    return sum(result)


@dispatch(NProductNode, cache=list)  # type: ignore[no-redef]
def _get_node_counts(module: NProductNode, cache: Optional[list] = None) -> np.ndarray:
    """Wrapper for Modules"""
    if not cache:
        cache = []
    if module in cache:
        result = [np.array((0, 0, 0))]
    else:
        cache.append(module)
        result = [np.array((0, 1, 0))]
    for child in module.children:
        result += _get_node_counts(
            child,
            cache=cache,
        )
    return sum(result)


@dispatch(NLeafNode, cache=list)  # type: ignore[no-redef]
def _get_node_counts(module: NLeafNode, cache: Optional[list] = None) -> np.ndarray:
    """Wrapper for Modules"""
    if not cache:
        cache = []
    if module in cache:
        result = [np.array((0, 0, 0))]
    else:
        cache.append(module)
        result = [np.array((0, 0, 1))]
    return sum(result)


@dispatch(NModule)  # type: ignore[no-redef]
def _get_child_output_nnodes(module: NModule) -> List[NNode]:
    """Wrapper for Modules"""
    output_nnodes = []
    for child in module.children:
        for output_module in child.output_modules:
            if issubclass(type(output_module), NNode):
                output_nnodes.append(output_module)
            else:
                output_nnodes += _get_child_output_nnodes(output_module)
    return output_nnodes

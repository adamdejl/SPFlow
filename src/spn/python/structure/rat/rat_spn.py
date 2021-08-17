"""
Created on May 24, 2021

@authors: Bennet Wittelsbach
"""
import itertools
import numpy as np
from typing import Dict, List, Union, cast
from spn.python.structure.module import Module
from spn.python.structure.nodes.node import LeafNode, Node, ProductNode, SumNode, _print_node_graph
from spn.python.structure.nodes.leaves.parametric import Gaussian
from .region_graph import (
    Partition,
    Region,
    RegionGraph,
    _print_region_graph,
    random_region_graph,
)


class RatSpn(Module):
    """A RAT-SPN is a randomized SPN, usually built from a RegionGraph.

    Attributes:
        root_node:
            A single SumNode that has a list of SumNodes as children and is the root of the RAT-SPN.
            The root node is the output of SPNs. Usually, SPNs only have one root node,
            but one can also look at its child SumNodes for multiple outputs, e.g. classes.
            When the SPN is constructed from a RegionGraph, the children of the root are the nodes of
            the root_region of the RegionGraph.
    """

    def __init__(self) -> None:
        self.root_node: SumNode
        self.region_graph: RegionGraph
        self.num_nodes_root: int
        self.num_nodes_region: int
        self.num_nodes_leaf: int
        self.rg_nodes: Dict[Union[Region, Partition], List[Node]]

    def __len__(self):
        return 1


def construct_spn(
    region_graph: RegionGraph,
    num_nodes_root: int,
    num_nodes_region: int,
    num_nodes_leaf: int,
) -> RatSpn:
    """Builds a RAT-SPN from a given RegionGraph.

    This algorithm is an implementation of "Algorithm 2" of the original paper. The Regions and
    Partitions in the RegionGraph are equipped with an appropriate number of nodes each, and the
    nodes will be connected afterwards. The resulting RAT-SPN holds a list of the root nodes, which
    in turn hold the whole constructed (graph) SPN. The number of ProductNodes in a Partition is
    determined by the length of the cross product of the children Regions of the respective Partition.

    Args:
        num_nodes_root:
            (C in the paper)
            The number of SumNodes the root_region is equipped with. This will be the length of the children of the
            root_node of the resulting RAT-SPN.
        num_nodes_region:
            (S in the paper)
            The number of SumNodes each region except the root and leaf regions are equipped with.
        num_nodes_leaf:
            (I in the paper)
            The number of LeafNodes each leaf region is equipped with. All LeafNodes of the same region
            are multivariate distributions over the same scope, but possibly differently parametrized.


    Returns:
        A RatSpn with a single SumNode as root. It's children are the SumNodes of the root_region in
        the region_graph. The rest of the SPN consists of alternating Sum- and ProductNodes, providing
        the scope factorizations determined by the region_graph.

    Raises:
        ValueError:
            If any argument is invalid (too less roots to build an SPN).
    """
    if num_nodes_root < 1:
        raise ValueError("num_nodes_root must be at least 1")
    if num_nodes_region < 1:
        raise ValueError("num_nodes_region must be at least 1")
    if num_nodes_leaf < 1:
        raise ValueError("num_nodes_leaf must be at least 1")

    rat_spn = RatSpn()
    rat_spn.region_graph = region_graph
    rat_spn.num_nodes_root = num_nodes_root
    rat_spn.num_nodes_region = num_nodes_region
    rat_spn.num_nodes_leaf = num_nodes_leaf

    rg_nodes: Dict[Union[Region, Partition], List[Node]] = {}

    for region in region_graph.regions:
        # determine the scope of the nodes the Region will be equipped with
        region_scope = list(region.random_variables)
        region_scope.sort()
        if not region.parent:
            # the region is the root_region
            root_nodes: List[Node] = [
                SumNode(children=[], scope=region_scope, weights=np.empty(0))
                for i in range(num_nodes_root)
            ]
            rg_nodes[region] = root_nodes
            rat_spn.root_node = SumNode(
                children=root_nodes,
                scope=region_scope,
                weights=np.full(len(rg_nodes[region]), 1 / len(rg_nodes[region])),
            )
        elif not region.partitions:
            # the region is a leaf
            rg_nodes[region] = [
                Gaussian(scope=region_scope, mean=0.0, stdev=1.0) for i in range(num_nodes_leaf)
            ]
        else:
            # the region is an internal region
            rg_nodes[region] = [
                SumNode(children=[], scope=region_scope, weights=np.empty(0))
                for i in range(num_nodes_region)
            ]

    for partition in region_graph.partitions:
        # determine the number and the scope of the ProductNodes the Partition will be equipped with
        num_nodes_partition = np.prod([len(rg_nodes[region]) for region in partition.regions])

        partition_scope = list(
            itertools.chain(*[region.random_variables for region in partition.regions])
        )
        partition_scope.sort()
        rg_nodes[partition] = [
            ProductNode(children=[], scope=partition_scope) for i in range(num_nodes_partition)
        ]

        # each ProductNode of the Partition points to a unique combination consisting of one Node of each Region
        # that is a child of the partition
        cartesian_product = list(
            itertools.product(*[rg_nodes[region] for region in partition.regions])
        )
        for i in range(len(cartesian_product)):
            rg_nodes[partition][i].children = list(cartesian_product[i])
        # all ProductNodes of the Partition are children of each SumNode in its parent Region
        for parent_node in rg_nodes[partition.parent]:
            # all parent nodes are SumNodes
            parent_node = cast(SumNode, parent_node)
            replicas = len(partition.parent.partitions)
            parent_node.children.extend(rg_nodes[partition])
            # determine the total number of children the parent node might have.
            # this is important for correct weights in the root nodes
            parent_node.weights = np.append(
                parent_node.weights,
                np.full(num_nodes_partition, 1 / (num_nodes_partition * replicas)),
            )

    rat_spn.rg_nodes = rg_nodes

    if not rat_spn.root_node:
        raise ValueError("Constructed RAT-SPN does not have root node")

    return rat_spn


if __name__ == "__main__":
    region_graph = random_region_graph(X=set(range(1, 8)), depth=2, replicas=2)
    _print_region_graph(region_graph)
    rat_spn = construct_spn(region_graph, 3, 2, 2)
    _print_node_graph(rat_spn.root_node)
from multipledispatch import dispatch  # type: ignore
from spflow.base.structure.nmodule import (
    NModule,
    NLeafNode,
    NProductNode,
    NSumNode,
    _get_child_output_nnodes,
)
import numpy as np
from spflow.base.structure.nmodules.nmodule import ProductLayer, SumLayer


@dispatch(NModule)  # type: ignore[no-redef]
def _isvalid_spn(module: NModule) -> None:
    """Wrapper for Modules"""
    for child in module.children:
        _isvalid_spn(child)
    for output_module in module.output_modules:
        _isvalid_spn(output_module)


@dispatch(NSumNode)  # type: ignore[no-redef]
def _isvalid_spn(module: NSumNode) -> None:
    """Wrapper for Modules"""

    assert module.children is not None
    children = _get_child_output_nnodes(module)
    for child in module.children:
        _isvalid_spn(child)

    assert children is not None
    assert None not in children
    assert module.weights is not None
    assert None not in module.weights
    assert np.isclose(sum(module.weights), 1.0)
    assert module.weights.shape == module.weights.shape
    assert np.array(children).shape == module.weights.shape
    for child in children:
        assert child.scope == module.scope


@dispatch(NProductNode)  # type: ignore[no-redef]
def _isvalid_spn(module: NProductNode) -> None:
    """Wrapper for Modules"""

    assert module.children is not None
    assert None not in module.children

    children = _get_child_output_nnodes(module)
    assert len(module.children) >= 1

    assert children is not None
    assert None not in children
    assert module.scope == sorted([scope for child in children for scope in child.scope])
    length = len(children)
    # assert that each child's scope is true subset of IProductNode's scope (set<set = subset)
    for i in range(0, length):
        assert set(children[i].scope) < set(module.scope)
        # assert that all children's scopes are pairwise distinct (set&set = intersection)
        for j in range(i + 1, length):
            assert not set(children[i].scope) & set(children[j].scope)

    for child_module in module.children:
        _isvalid_spn(child_module)


@dispatch(NLeafNode)  # type: ignore[no-redef]
def _isvalid_spn(module: NLeafNode) -> None:
    """Wrapper for Modules"""
    assert module.children == []


@dispatch(ProductLayer)  # type: ignore[no-redef]
def _isvalid_spn(module: ProductLayer) -> None:
    """Wrapper for Modules"""

    assert module.children is not None
    assert None not in module.children

    children = _get_child_output_nnodes(module)
    assert children is not None
    assert None not in children

    num_of_nodes = module.num_of_nodes
    num_of_children = module.num_of_children

    assert len(module.children) == num_of_children * num_of_nodes

    # check for each ProdNode in the layer
    reshaped_children = np.asarray(children).reshape((num_of_nodes, num_of_children))
    for k, output_module in enumerate(module.output_modules):

        assert module.scope == sorted(
            [scope for child in reshaped_children[k] for scope in child.scope]
        )
        length = len(reshaped_children[k])
        # assert that each child's scope is true subset of IProductNode's scope (set<set = subset)
        for i in range(0, length):
            assert set(reshaped_children[k][i].scope) < set(module.scope)
            # assert that all children's scopes are pairwise distinct (set&set = intersection)
            for j in range(i + 1, length):
                assert not set(reshaped_children[k][i].scope) & set(reshaped_children[k][j].scope)

    for child_module in children:
        _isvalid_spn(child_module)


@dispatch(SumLayer)  # type: ignore[no-redef]
def _isvalid_spn(module: SumLayer) -> None:
    """Wrapper for Modules"""

    assert module.children is not None
    assert None not in module.children
    assert module.weights is not None
    assert None not in module.weights
    assert np.isclose(sum(module.weights), 1.0)
    assert module.weights.shape == module.weights.shape

    children = _get_child_output_nnodes(module)
    assert children is not None
    assert None not in children

    num_of_nodes = module.num_of_nodes
    num_of_children = module.num_of_children

    assert len(module.children) == num_of_children * num_of_nodes

    # check for each SumNode in the layer
    reshaped_children = np.asarray(children).reshape((num_of_nodes, num_of_children))
    for k, output_module in enumerate(module.output_modules):

        assert np.array(reshaped_children[k]).shape == output_module.weights.shape
        for child in reshaped_children[k]:
            assert child.scope == output_module.scope

    for child_module in children:
        _isvalid_spn(child_module)

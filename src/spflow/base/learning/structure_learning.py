import os
import multiprocessing
from enum import Enum
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple
import numpy as np

from spflow.base.structure.nodes.node import INode, IProductNode, ISumNode, _print_node_graph
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from spflow.base.structure.nodes.structural_transformations import prune
from spflow.base.learning.context import Context

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="learnSPN_log", level=logging.DEBUG)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time


class Operation(Enum):
    """Operations carried out during the LearnSPN process.

    TODO: Operation as abstract class, elements as classes that inherit from Operation
    TODO: use dispatching to seperate the large learning functions into smaller specific chunks

    """

    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4
    REMOVE_UNINFORMATIVE_FEATURES = 5


def get_next_operation(
    min_instances_slice: int = 100,
    min_features_slice: int = 1,
    multivariate_leaf: bool = False,
    cluster_univariate: bool = False,
) -> Callable:
    """Wrapper for selecting operations during the LearnSPN procedure.

    Arguments:
        min_instances_slice:
            The minimum number of instances used for creating a leaf node.
        min_features_slice:
            The minimum number of features used for creating a leaf node.
        multivariate_leaf:
            Flag, if leafs over multiple random variables/features are permitted.
        cluster_univariate:
            Flag, if TODO

    Returns:
        A function handler for selecting operations during the structure learning procedure.
    """

    def next_operation(
        data: np.ndarray,
        scope: List[int],
        no_clusters: bool = False,
        no_independencies: bool = False,
        is_first: bool = False,
        cluster_first: bool = True,
    ) -> Tuple[Operation, Optional[List[int]]]:
        """Determine the next operation used during LearnSPN depending on the state of the local data and user settings.

        Arguments:
            data:
                A (2-dimensional) numpy array.
            scope:
                The 'data'-indices on which the next operation will be carried out.
            no_clusters:
                Flag, if TODO
            no_independencies:
                Flag, if TODO
            is_first:
                Flag, if this is the first call to this function while learning the current SPN.
            cluster_first:
                Flag, if the SPN's root node shall be a ISumNode instead of a IProductNode.

        Returns:
            An Operation constant used to determine actions during the structure learning procedure, and its parameters, if needed.
        """

        minimalFeatures = len(scope) == min_features_slice
        minimalInstances = data.shape[0] <= min_instances_slice

        if minimalFeatures:
            if minimalInstances or no_clusters:
                return Operation.CREATE_LEAF, None
            else:
                if cluster_univariate:
                    return Operation.SPLIT_ROWS, None
                else:
                    return Operation.CREATE_LEAF, None

        uninformative_features_idx = np.var(data[:, 0 : len(scope)], 0) == 0
        ncols_zero_variance = np.sum(uninformative_features_idx)
        if ncols_zero_variance > 0:
            if ncols_zero_variance == data.shape[1]:
                if multivariate_leaf:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            else:
                return (
                    Operation.REMOVE_UNINFORMATIVE_FEATURES,
                    np.arange(len(scope))[uninformative_features_idx].tolist(),
                )

        if minimalInstances or (no_clusters and no_independencies):
            if multivariate_leaf:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        if no_independencies:
            return Operation.SPLIT_ROWS, None

        if no_clusters:
            return Operation.SPLIT_COLUMNS, None

        if is_first:
            if cluster_first:
                return Operation.SPLIT_ROWS, None
            else:
                return Operation.SPLIT_COLUMNS, None

        return Operation.SPLIT_COLUMNS, None

    return next_operation


def default_slicer(data: np.ndarray, cols: List[int], num_cond_cols: int = None) -> np.ndarray:
    """Slices columns from 2-dimensional data.

    TODO: Elaborate on when and how this function is used

    Arguments:
        data:
            A (2-dimensional) numpy array.
        cols:
            A list of the column indices that shall be sliced from 'data'.
        num_cond_cols:
            A negative index referencing a column in 'data' from which all subsequent columns will be concatenated to the sliced result.
            TODO: better explain what this argument is doing

    Returns:
        A (2-dimensional) numpy array consisisting of columns from 'data'.
    """
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))

        return data[:, cols]
    else:
        # But WHY
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)


def learn_spn_structure(
    dataset: np.ndarray,
    context: Context,
    split_rows: Callable,
    split_cols: Callable,
    create_leaf: Callable,
    next_operation: Callable = get_next_operation(),
    initial_scope: List[int] = None,
    data_slicer: Callable = default_slicer,
    parallel: bool = True,
) -> INode:
    """Learn the structure and initial weights of an SPN from the given dataset, using its context.

    This function implements the LearnSPN procedure. An SPN is learned from data by recursively splitting a 2-dimensional
    dataset along its axes in an alternating way.
    TODO: Reference papers (Domingos, Molina/Vergari?, survey?)
    TODO: Maybe elaborate on LearnSPN

    Arguments:
        dataset:
            A (2-dimensional) numpy array.
        context:
            A Context providing meta-information about the data.
        split_rows:
            A function handler for a row splitting strategy (clustering of instances).
        split_cols:
            A function handler for a column splitting strategy (independency assessments of features/random variables).
        create_leaf:
            A function handler for creating leaves (usually parametric leaves).
        next_operation:
            A function handler for selecting the operations during the LearnSPN procedure.
        initial_scope:
            A list of indexes of which dataset columns/features shall be used to learn the SPN. If it is None (as by default), the SPN will be learned from all features.
        data_slicer:
            A function handler for a data slicing strategy (after splitting columns [or rows]).
        parallel:
            Flag, if steps that can be parallelized will be parallelized.

    Returns:
        The root node of the learned SPN.

    Raises:
        AssertionError:
            If any of the mandatory arguments is None,
            OR if the learned SPN is not valid,
            OR parts of the learned SPN do not have the expected format.
        Exception:
            If the operation to be carried out is unknown.
    """
    assert dataset is not None
    assert context is not None
    assert split_rows is not None
    assert split_cols is not None
    assert create_leaf is not None
    assert next_operation is not None

    if initial_scope is None:
        initial_scope = list(range(dataset.shape[1]))
        num_conditional_cols = None
    elif len(initial_scope) < dataset.shape[1]:
        num_conditional_cols = dataset.shape[1] - len(initial_scope)
    else:
        num_conditional_cols = None
        assert len(initial_scope) > dataset.shape[1], "check initial scope: %s" % initial_scope

    root = IProductNode([], initial_scope)
    root.children.append(None)  # type: ignore
    node: INode

    tasks: Deque = deque()
    tasks.append((dataset, root, 0, initial_scope, False, False))

    while tasks:

        local_data, parent, children_pos, scope, no_clusters, no_independencies = tasks.popleft()

        operation, op_params = next_operation(
            local_data,
            scope,
            no_clusters=no_clusters,
            no_independencies=no_independencies,
            is_first=(parent is root),
        )

        logging.debug("OP: {} on slice {} (remaining tasks {})".format(operation, local_data.shape, len(tasks)))

        if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:
            node = IProductNode([], scope)
            parent.children[children_pos] = node

            rest_scope = set(range(len(scope)))
            for col in op_params:
                rest_scope.remove(col)
                node.children.append(None)  # type: ignore
                tasks.append(
                    (
                        data_slicer(local_data, [col], num_conditional_cols),
                        node,
                        len(node.children) - 1,
                        [scope[col]],
                        True,
                        True,
                    )
                )

            next_final = False

            if len(rest_scope) == 0:
                continue
            elif len(rest_scope) == 1:
                next_final = True

            node.children.append(None)  # type: ignore
            c_pos = len(node.children) - 1

            rest_cols = list(rest_scope)
            rest_scopes = [scope[col] for col in rest_scope]

            tasks.append(
                (
                    data_slicer(local_data, rest_cols, num_conditional_cols),
                    node,
                    c_pos,
                    rest_scopes,
                    next_final,
                    next_final,
                )
            )

            continue

        elif operation == Operation.SPLIT_ROWS:

            split_start_t = perf_counter()
            data_slices = split_rows(local_data, context, scope)
            split_end_t = perf_counter()
            logging.debug(
               "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
            )

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, True, False))
                continue

            node = ISumNode([], scope, np.array([]))
            parent.children[children_pos] = node
            # assert parent.scope == node.scope, (parent, node)

            for data_slice, scope_slice, proportion in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"

                node.children.append(None)  # type: ignore
                node.weights = np.append(
                    node.weights, proportion
                )  # sum_node.weights.append(proportion)
                tasks.append((data_slice, node, len(node.children) - 1, scope, False, False))

            continue

        elif operation == Operation.SPLIT_COLUMNS:
            split_start_t = perf_counter()
            data_slices = split_cols(local_data, context, scope)
            split_end_t = perf_counter()
            logging.debug(
               "\t\tfound {} col clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
            )

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, False, True))
                assert np.shape(data_slices[0][0]) == np.shape(local_data)
                assert data_slices[0][1] == scope
                continue

            node = IProductNode([], scope)
            parent.children[children_pos] = node

            for data_slice, scope_slice, _ in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"

                node.children.append(None)  # type: ignore
                tasks.append((data_slice, node, len(node.children) - 1, scope_slice, False, False))

            continue

        elif operation == Operation.NAIVE_FACTORIZATION:
            # TODO: parallelization if factorizing lot of leafs

            node = IProductNode([], scope)
            parent.children[children_pos] = node

            local_tasks = []
            local_children_params = []
            # split_start_t = perf_counter()
            for col in range(len(scope)):
                node.children.append(None)  # type: ignore
                local_tasks.append(len(node.children) - 1)
                child_data_slice = data_slicer(local_data, [col], num_conditional_cols)
                local_children_params.append((child_data_slice, context, [scope[col]]))

            for (child_params, child_pos) in zip(local_children_params, local_tasks):
                child_node = create_leaf(*child_params)
                node.children[child_pos] = child_node

            # split_end_t = perf_counter()

            logging.debug(
               "\t\tnaive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t)
            )

            continue

        elif operation == Operation.CREATE_LEAF:
            leaf_start_t = perf_counter()
            node = create_leaf(local_data, context, scope)
            parent.children[children_pos] = node
            leaf_end_t = perf_counter()

            logging.debug(
               "\t\t created leaf {} for scope={} (in {:.5f} secs)".format(
                   node.__class__.__name__, scope, leaf_end_t - leaf_start_t
               )
            )

        else:
            raise Exception("Invalid operation: " + operation)

    node = root.children[0]
    _isvalid_spn(node)
    node = prune(node)
    _isvalid_spn(node)

    return node

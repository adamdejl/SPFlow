"""
@author Bennet Wittelsbach, based on code from Alejandro Molina
"""

from typing import Callable, List, Tuple
import numpy as np
from joblib import Memory  # type: ignore
from spflow.base.learning.splitting.clustering import get_split_rows_GMM, get_split_rows_KMeans
from spflow.base.learning.splitting.rdc import get_split_cols_RDC_py, get_split_rows_RDC_py
from spflow.base.learning.structure_learning import get_next_operation, learn_spn_structure
from spflow.base.structure.nodes.leaves.parametric.parametric import ParametricLeaf
from spflow.base.structure.nodes.leaves.parametric.parameter_estimation import (
    maximum_likelihood_estimation,
)
from spflow.base.structure.nodes.node import ILeafNode, INode
from spflow.base.learning.context import Context


def learn_parametric_spn(
    data: np.ndarray,
    context: Context,
    split_cols: str = "rdc",
    split_rows: str = "kmeans",
    min_instances_slice: int = 200,
    min_features_slice: int = 1,
    multivariate_leaf: bool = False,
    cluster_univariate: bool = False,
    threshold: float = 0.3,
    ohe: bool = False,
    leaves: Callable = None,
    memory: Memory = None,
    rand_gen: int = None,
    cpus: int = -1,
) -> INode:
    """Learn the structure and initial weights of an SPN with parametric distributions as leaves.

    The outer function wraps the parametric LearnSPN procedure in a memoization mechanism, if the flag is set.

    TODO: <description, example>

    Arguments:
        data:
            A (2-dimensional) numpy array.
        context:
            A Context providing meta-information about 'data'.
        split_cols:
            A string mapping to a column splitting strategy (independency assessments of features/random variables).
        split_rows:
            A string mapping to a a row splitting strategy (clustering of instances).
        min_instances_slice:
            Threshold of instances needed to create a parametric leaf node.
        min_features_slice:
            Threshold of features needed to create a parametric leaf node.
        multivariate_leaf:
            Flag, if the learning of multivariate leaves is permitted.
        cluster_univariate:
            Flag, if TODO
        threshold:
            Threshold used for the 'rdc' splitting procedure.
        ohe:
            Flag, if One-Hot Encoding shall be applied to the data.
        leaves:
            A function handler for creating parametric leaves.
        memory:
            Flag, if SPNs should be cached.
        rand_gen:
            Random number generator seed.
        cpus:
            Number of CPU's to be used, if the structure learning procedure shall be carried out in parallel (where possible).

    Returns:
        The root node of the learned SPN with parametric distributions as leaves.
    """
    if leaves is None:
        leaves = create_parametric_leaf

    def learn_param(
        data: np.ndarray,
        context: Context,
        split_cols: str,
        split_rows: str,
        min_instances_slice: int,
        threshold: float,
        ohe: bool,
    ):
        """Learn the structure and initial weights of an SPN with parametric distributions as leaves with the given parameters of the outer function.

        TODO: <description, examples>

        Arguments:
            data:
                A (2-dimensional) numpy array.
            context:
                A Context providing meta-information about 'data'.
            split_cols:
                A string used to get a column splitting function handler.
            split:rows:
                A string used to get a row splitting function handler.
            min_instances_slice:
               Threshold of instances needed to create a parametric leaf node.
            min_features_slice:
                Threshold of features needed to create a parametric leaf node.
            threshold:
                Threshold used for the 'rdc' splitting procedure.
            ohe:
                Flag, if One-Hot Encoding shall be applied to the data.

        Returns:
            The root node of the learned SPN with parametric distributions as leaves.
        """
        column_splitting_function, row_splitting_function = get_splitting_functions(split_cols, split_rows, ohe, threshold, rand_gen, cpus)  # type: ignore

        nextop = get_next_operation(
            min_instances_slice, min_features_slice, multivariate_leaf, cluster_univariate
        )

        return learn_spn_structure(data, context, row_splitting_function, column_splitting_function, leaves, nextop)  # type: ignore

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, context, split_cols, split_rows, min_instances_slice, threshold, ohe)


def get_splitting_functions(
    column_strategy: str, row_strategy: str, ohe: bool, threshold: float, rand_gen: int, n_jobs: int
) -> Tuple[Callable, Callable]:
    """Select the splitting functions used for the splitting steps (independence assessment and clustering) in LearnSPN.

    Returns function handlers for the splitting procedures in LearnSPN. The column_strategy is used to test for
    independence between two sets of columns/scopes/random variables. The row_strategy is used to cluster data
    of the same scope.

    Arguments:
        column_strategy:
            Available strategies:
                - "rdc" ("Randomized Dependence Coefficient", see Lopez-Paz et.al., 2013)
        row_strategy:
            Available strategies:
                - "rdc"
                - "KMeans" (standard KMeans procedure, uses the sklearn implementation)
                - "GMM" (Gaussian Mixture Model, uses the sklean implementation)
        ohe:
            Flag for using one-hot encoding in the rdc splitting procedure.
        threshold:
            Threshold between 0 and 1 for the independency assessment in the splitting procedures.
        rand_gen:
            RNG seed used for the splitting procedures.
        n_jobs:
            Multi-threading parameter for the splitting procedures.

    Returns:
        A tuple of two function handlers, the first pointing to the selected column splitting procedure, the second pointing to the row splitting procedure.

    Raises:
        AssertionError:
            If any of the splitting strategies (parameters 'cols', 'rows') are not a str or Callable, or the str representation cannot be matched with implemented strategies.
    """
    # from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py

    if isinstance(column_strategy, str):
        if column_strategy == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, ohe=ohe)
        # elif cols == "poisson": # TODO: is implemented in the Original SPFlow and should also be here
        #    split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError(
                "unknown columns splitting strategy type %s" % str(column_strategy)
            )
    elif isinstance(column_strategy, Callable):
        split_cols = column_strategy
    else:
        raise AssertionError("The column splitting strategy type is neither str nor Callable")

    if isinstance(row_strategy, str):
        if row_strategy == "rdc":
            split_rows = get_split_rows_RDC_py(ohe=ohe, seed=rand_gen)
        elif row_strategy == "kmeans":
            split_rows = get_split_rows_KMeans(seed=rand_gen)
        # elif rows == "tsne": # TODO: should TSNE be used for clustering? still needs some research, but can be fun to play around with
        #    split_rows = get_split_rows_TSNE()
        elif row_strategy == "gmm":
            split_rows = get_split_rows_GMM()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(row_strategy))
    elif isinstance(row_strategy, Callable):
        split_rows = row_strategy
    else:
        raise AssertionError("The row splitting strategy type is neither str nor Callable")
    return split_cols, split_rows


# TODO: while 'scope' is of type list[int], it only allows to build nodes over the first element of the list
# therefore, extend the algorithm to create leafs over multiple scopes
def create_parametric_leaf(data: np.ndarray, context: Context, scope: List[int]) -> ILeafNode:
    """Create a leaf node representing a parametric distribution.

    Create a leaf node of the type given by ds_context.parametric_types[scope] over the given scope
    and estimate its parameters from the given data by applying MLE (if feasible).
    Example: create_parametric_leaf(np.array([[1, 2, 3, 4, 5]]), Context(parametric_types=[Gaussian]), [0])
    will create a Gaussian leaf node with scope=[0], mean=2.5, and stdev=sqrt(2).

        Arguments:
            data:
                A (2-dimensional) numpy array, holding the data from which the node's parameters are to be estimated from.
            context:
                A context with user-specified parametric_types, containing the class of the leaf node of the argument 'scope' (e.g. [Gaussian, Bernoulli]).
            scope:
                A list of integers containing the scopes that will be assigned to the node. WARNING: At the moment, this only uses the first element of the list and creates nodes over exactly one scope.

        Returns:
            A leaf node of the type specified in ds_context.parametric_types and accordingly estimated parameters.

        Raises:
            AssertionError:
                If the ds_context_parametric types is either None or its size is smalle than the given scope.
    """
    assert len(scope) == 1, "scope of univariate parametric for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]

    assert (
        context.parametric_types is not None
    ), "for parametric leaves, the ds_context.parametric_types can't be None"
    assert len(context.parametric_types) > idx, (
        "for parametric leaves, the ds_context.parametric_types must have a parametric type at pos %s "
        % (idx)
    )

    parametric_type: ParametricLeaf = context.parametric_types[idx]

    assert parametric_type is not None

    # TODO: extend to allow parameter estimation also via EM
    node = parametric_type(scope)  # type: ignore
    # if parametric_type == Categorical:
    #    k = int(np.max(ds_context.domains[idx]) + 1)
    #    node = Categorical(p=(np.ones(k) / k).tolist())

    maximum_likelihood_estimation(node, data)

    return node

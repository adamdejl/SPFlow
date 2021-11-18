"""
@author Bennet Wittelsbach, based on code from Alejandro Molina
"""

from typing import List, Optional, Tuple
import numpy as np
from networkx import from_numpy_matrix, connected_components  # type: ignore
from spflow.base.structure.nodes.leaves.parametric.statistical_types import MetaType
from spflow.base.learning.context import Context


def preprocess(
    data: np.ndarray, context: Context, pre_proc: Optional[str], ohe: bool
) -> np.ndarray:
    """Possibly deprecated, need to check to what extent this function is needed.

    Arguments:
        data:
            (2-dim) numpy-array holding the data to be preprocessed
        context:
            indecisive: either a Context as in the other functions that have ds_context as argument (also see getOHE()) OR a str based on the code below.
        pre_proc:
            Flag, if the data shall be preprocessed.
        ohe:
            Flag, if one-hot encoding shall be applied to columns with discrete values.

    Returns:
        The preprocessed 'data'.
    """
    # TODO
    # this code makes no sense atm, all functions that call preprocess pass a Context-type variable, not a str
    # check if this function is needed at all, and adapt it if necessary
    # if pre_proc:
    # f = None
    # if pre_proc == "tf-idf":
    #    f = lambda data: TfidfTransformer().fit_transform(data)
    # el
    # if ds_context == "log+1":
    #    f = lambda data: np.log(data + 1)
    # elif ds_context == "sqrt":
    #    f = lambda data: np.sqrt(data)

    # if f is not None:
    #    data = np.copy(data)
    #    data[:, ds_context.distribution_family == "poisson"] = f(
    #        data[:, ds_context.distribution_family == "poisson"]
    #    )

    if ohe:
        data = getOHE(data, context)

    return data


def getOHE(data: np.ndarray, context: Context) -> np.ndarray:
    """Process discrete values to categorical values via One-Hot Encoding.

    Apply one-hot encoding to data columns which have MetaType.DISCRETE assigned in the given context.

    Arguments:
        data:
            A (2-dimensional) numpy array.
        ds_context:
            A Context providing meta-information about 'data'.
    Returns:
        The processed 'data' (all OHE-processed columns with n unique values each are replaced by n columns respectively).

    Raises:
        AssertionError:
            If the OHE did not work correctly.
    """
    cols = []
    for f in range(data.shape[1]):
        data_col = data[:, f]

        if context.meta_types[f] != MetaType.DISCRETE:
            cols.append(data_col)
            continue

        domain: np.ndarray = context.domains[f]

        dataenc = np.zeros((data_col.shape[0], len(domain)), dtype=data.dtype)

        dataenc[data_col[:, None] == domain[None, :]] = 1

        assert np.all((np.sum(dataenc, axis=1) == 1)), "one hot encoding bug {} {}".format(
            domain, data_col
        )

        cols.append(dataenc)

    return np.column_stack(cols)


def clusters_by_adjacency_matrix(
    adjacency_matrix: np.ndarray, threshold: float, n_features: int
) -> np.ndarray:
    """TODO: find out what exactly is happening here. looks like spectral clustering. run some tests with functions that call this (rdy.py:get_split_cols_RDC())

    <description>

    Arguments:
        adjacency_matrix:

        threshold:
            ...
        n_features:
            ...?

    Returns:

    """
    adjacency_matrix[adjacency_matrix < threshold] = 0

    adjacency_matrix[adjacency_matrix > 0] = 1

    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(adjacency_matrix))):
        result[list(c)] = i + 1

    return result


def split_data_by_clusters(
    data: np.ndarray, clusters: np.ndarray, scope: List[int], rows: bool = True
) -> List[Tuple[np.ndarray, List[int], float]]:
    """Split a dataset into given clusters

    Split a 2-dimensional dataset either horizontal (rows) or vertical (columns), based on the given clusters (usually computed previously by a
    clustering algorithm, e.g. KMeans or GMM, see clustering.py).

    Arguments:
        data:
            A (2-dimensional) numpy array.
        clusters:
            A (1-dimensional) numpy array with the same number of rows as 'data'.
        scope:
            A list of all scopes in the dataset.
        rows:
            Flag if the rows or columns of 'data' are to be splitted.

    Returns:
        A list of tuples containing the partitions of 'data', the scope of the partition, and the relative size of the partition compared to the total size of 'data'.
    """
    unique_clusters = np.unique(clusters)
    result = []

    nscope = np.asarray(scope)

    for uc in unique_clusters:
        if rows:
            local_data = data[clusters == uc, :]
            proportion = local_data.shape[0] / data.shape[0]
            result.append((local_data, scope, proportion))
        else:
            local_data = data[:, clusters == uc].reshape((data.shape[0], -1))
            proportion = local_data.shape[1] / data.shape[1]
            result.append((local_data, nscope[clusters == uc].tolist(), proportion))

    return result

import numpy as np # type: ignore
from multipledispatch import dispatch # type: ignore
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.structure.network_type import SPN
from spflow.base.structure.nodes.leaves.parametric.bernoulli import Bernoulli
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.nodes.leaves.parametric.parametric import get_natural_parameters, get_base_measure, get_sufficient_statistics, get_log_partition_natural, get_log_partition_param, get_scipy_object, get_scipy_object_parameters, update_parameters_em
from spflow.base.structure.nodes.node import ILeafNode, INode, IProductNode, ISumNode, _print_node_graph, eval_spn_bottom_up, get_topological_order


def compute_exponential_family_pdf(node: INode, X: np.ndarray) -> float:
    natural_parameters = get_natural_parameters(node)
    base_measure = get_base_measure(node, X)
    sufficient_statistics = get_sufficient_statistics(node, X)
    log_partition_n = get_log_partition_natural(node)
    log_partition_p = get_log_partition_param(node)
    return base_measure * np.exp(natural_parameters @ sufficient_statistics.T - log_partition_n)


def em(node: INode, data: np.ndarray, iterations:int=10, hard_em:bool=False) -> None:

    ll_pre = np.sum(log_likelihood(SPN(), node, data))

    # TODO: use toplogical order to pass through node via bottom up, ignore leafs
    nodes = get_topological_order(node)
    for n in nodes:
        if isinstance(n, ILeafNode):
            continue
        em_update(n, data, 1, False)

    ll_post = np.sum(log_likelihood(SPN(), node, data))
    print(f"log-L before EM: {ll_pre}")
    print(f"log-L after  EM: {ll_post}")   


@dispatch(ISumNode, np.ndarray, int, bool) # tpye: ignore[no-redef]
def em_update(node: ISumNode, data: np.ndarray, iterations: int, hard_em: bool) -> None: # DICTIONARY SOLUTION: node_updates: dict[INode, Callable], 
    assert len(node.children) == len(node.weights)
    node_data = np.empty(data.shape)
    node_data[:] = np.NaN
    node_data[:, node.scope] = data[:, node.scope]

    for i in range(iterations):
        # E-step
        datapoint_responsibilities = compute_responsibilities(node, node_data, hard_em)
        cluster_responsibilities = np.sum(datapoint_responsibilities, axis=0)
        # M-step
        # here implemented: dispatch solution
        new_weights = cluster_responsibilities / np.sum(cluster_responsibilities)
        node.weights = new_weights

        for i, child_node in enumerate(node.children):
            if isinstance(child_node, ILeafNode):
                update_parameters_em(child_node, data, datapoint_responsibilities[:, i])


@dispatch(IProductNode, np.ndarray, int, bool) # tpye: ignore[no-redef]
def em_update(node: IProductNode, data: np.ndarray, iterations: int, hard_em: bool) -> None:
    node_data = np.empty(data.shape)
    node_data[:] = np.NaN
    node_data[:, node.scope] = data[:, node.scope]

    for i in range(iterations):
        responsibilities = np.ones((len(data), 1))

        for i, child_node in enumerate(node.children):
            if isinstance(child_node, ILeafNode):
                update_parameters_em(child_node, data, responsibilities)
    

def compute_responsibilities(node: ISumNode, data: np.ndarray, hard_em:bool) -> np.ndarray:
    prior_cluster_probs = node.weights

    # dirty prototype
    datapoint_responsibilities = np.zeros(shape=(data.shape[0], len(prior_cluster_probs)))
    for datapoint_index, x in enumerate(data):
        for cluster_index, pi in enumerate(prior_cluster_probs):
            cluster_node = node.children[cluster_index]
            ll = log_likelihood(SPN(), cluster_node, np.array([x]))
            gamma = pi * ll
            datapoint_responsibilities[datapoint_index, cluster_index] = gamma

        # normalize probabilities
        print(datapoint_responsibilities)
        datapoint_responsibilities[datapoint_index, :] /= np.sum(datapoint_responsibilities[datapoint_index, :])

    if hard_em:
        # binarize cluster assignments
        datapoint_responsibilities[:] = np.where(datapoint_responsibilities == datapoint_responsibilities.max(axis=1).reshape(-1, 1), 1, 0)

    return datapoint_responsibilities


if __name__ == "__main__":
    node = Gaussian([0])
    X = np.arange(-3, 3, 0.2)
    
    scipy_param = get_scipy_object_parameters(node)
    scipy_distr = get_scipy_object(node)
    scipy_pdf_results = [scipy_distr.pdf(x.item(), **scipy_param) for x in X]

    em_pdf_results = compute_exponential_family_pdf(node, X)

    from numpy.testing import assert_array_almost_equal

    equal = assert_array_almost_equal(scipy_pdf_results, em_pdf_results)
    if equal is None: # assert_array_almost_equal() returns None if the arrays ARE almost equal, else an exception is raised
        print("Scipy PDF and Exp. Family PDF equality test for Gaussian was successful")


    spn = ISumNode(children=[IProductNode(children=[Gaussian([0], -2.0, 5.0), Bernoulli([1], 0.3)], scope=[0, 1]), 
                            IProductNode(children=[Gaussian([0], 4.5, 2.0), Bernoulli([1], 0.8)], scope=[0, 1])], 
                            scope=[0, 1], weights=[0.1, 0.9])
    top_ord = get_topological_order(spn)
    print(top_ord)

    data = np.array([
        [3.0, 0], 
        [2.5, 0], 
        [3.5, 0], 
        [3.1, 0], 
        [2.8, 0],
        [0.5, 1],
        [0.2, 1],
        [0.7, 1],
        [1.1, 1],
        [0.3, 1] 
    ])
    em(spn, data)
    _print_node_graph(spn)
    print(spn.weights)

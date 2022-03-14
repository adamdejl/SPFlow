import numpy as np  # type: ignore
from scipy.special import logsumexp # type: ignore
from multipledispatch import dispatch  # type: ignore
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.learning.gradient import gradient_backward
from spflow.base.structure.network_type import SPN
from spflow.base.structure.nodes.leaves.parametric.bernoulli import Bernoulli
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.nodes.leaves.parametric.parametric import (
    get_natural_parameters,
    get_base_measure,
    get_sufficient_statistics,
    get_log_partition_natural,
    get_log_partition_param,
    get_scipy_object,
    get_scipy_object_parameters,
    update_parameters_em,
)
from spflow.base.structure.nodes.node import (
    ILeafNode,
    INode,
    IProductNode,
    ISumNode,
    _get_node_counts,
    _print_node_graph,
    get_topological_order,
    set_node_ids,
)


def compute_exponential_family_pdf(node: INode, X: np.ndarray) -> float:
    natural_parameters = get_natural_parameters(node)
    base_measure = get_base_measure(node, X)
    sufficient_statistics = get_sufficient_statistics(node, X)
    log_partition_n = get_log_partition_natural(node)
    log_partition_p = get_log_partition_param(node)
    return base_measure * np.exp(natural_parameters @ sufficient_statistics.T - log_partition_n)    


@dispatch(ISumNode, data=np.ndarray, node_log_likelihood=np.ndarray, node_gradients=np.ndarray, root_log_likelihoods=np.ndarray, all_log_likelihoods=np.ndarray, all_gradients=np.ndarray) # type: ignore[no-redef]
def poon_domingos_em_update(node: ISumNode, data: np.ndarray, node_log_likelihood: np.ndarray = None, node_gradients: np.ndarray = None, root_log_likelihoods: np.ndarray = None, all_log_likelihoods: np.ndarray = None, all_gradients: np.ndarray = None) -> None:
    root_inverse_gradient = node_gradients - root_log_likelihoods

    for i, child in enumerate(node.children):
        new_weight = root_inverse_gradient + (all_log_likelihoods[:, child.id] + np.log(node.weights[i]))
        node.weights[i] = logsumexp(new_weight)

    assert not np.any(np.isnan(node.weights))
    
    node.weights = np.exp(node.weights - logsumexp(node.weights)) + np.exp(-100)
    node.weights = node.weights / np.sum(node.weights)
    assert not np.any(np.isnan(node.weights))
    assert np.isclose(np.sum(node.weights), 1)
    assert not np.any(node.weights < 0)
    assert node.weights.sum() <= 1, "sum: {}, node weights: {}".format(node.weights.sum(), node.weights)


@dispatch(IProductNode, data=np.ndarray, node_log_likelihood=np.ndarray, node_gradients=np.ndarray, root_log_likelihoods=np.ndarray, all_log_likelihoods=np.ndarray, all_gradients=np.ndarray) # type: ignore[no-redef]
def poon_domingos_em_update(node: IProductNode, data: np.ndarray, node_log_likelihood: np.ndarray = None, node_gradients: np.ndarray = None, root_log_likelihoods: np.ndarray = None, all_log_likelihoods: np.ndarray = None, all_gradients: np.ndarray = None) -> None:
    pass


@dispatch(Gaussian, data=np.ndarray, node_log_likelihood=np.ndarray, node_gradients=np.ndarray, root_log_likelihoods=np.ndarray, all_log_likelihoods=np.ndarray, all_gradients=np.ndarray) # type: ignore[no-redef]
def poon_domingos_em_update(node: Gaussian, data: np.ndarray, node_log_likelihood: np.ndarray = None, node_gradients: np.ndarray = None, root_log_likelihoods: np.ndarray = None, all_log_likelihoods: np.ndarray = None, all_gradients: np.ndarray = None) -> None:
    X = data[:, node.scope]
    p = (node_gradients - root_log_likelihoods) + node_log_likelihood
    lse = logsumexp(p)
    w = np.exp(p - lse)

    mean = np.sum(w * X)  
    stdev = np.sqrt(np.sum(w * np.power(X - mean, 2)))
    node.set_params(mean, stdev)


@dispatch(Bernoulli, data=np.ndarray, node_log_likelihood=np.ndarray, node_gradients=np.ndarray, root_log_likelihoods=np.ndarray, all_log_likelihoods=np.ndarray, all_gradients=np.ndarray) # type: ignore[no-redef]
def poon_domingos_em_update(node: Bernoulli, data: np.ndarray, node_log_likelihood: np.ndarray = None, node_gradients: np.ndarray = None, root_log_likelihoods: np.ndarray = None, all_log_likelihoods: np.ndarray = None, all_gradients: np.ndarray = None) -> None:
    X = data[:, node.scope]
    p = (node_gradients - root_log_likelihoods) + node_log_likelihood
    lse = logsumexp(p)
    wl = p - lse
    paramlse = np.exp(logsumexp(wl, b=X))

    assert not np.isnan(paramlse)
    p = min(max(paramlse, 0), 1)
    node.set_params(p)


def poon_domingos_em(spn: INode, data: np.ndarray, iterations: int = 10, hard_em: bool = False) -> None:
    ll_pre = np.sum(log_likelihood(SPN(), spn, data))
    node_log_likelihoods = np.zeros((data.shape[0], np.sum(_get_node_counts(spn))))
    set_node_ids(spn)

    for i in range(iterations):
        results = log_likelihood(SPN(), spn, data, return_all_results=True)

        for node, ll in results.items():
            node_log_likelihoods[:, node.id] = ll[:, 0]

        gradients = gradient_backward(spn, node_log_likelihoods)
        root_log_likelihoods = node_log_likelihoods[:, 0]

        for node in get_topological_order(spn):
            # TODO: use Dict[type, Callable] solution instead of dispatching (yields more flexiblity on this case)
            poon_domingos_em_update(node, data=data, node_log_likelihood=node_log_likelihoods[:, node.id], node_gradients=gradients[:, node.id], root_log_likelihoods=root_log_likelihoods, all_log_likelihoods=node_log_likelihoods, all_gradients=gradients)

    ll_post = np.sum(log_likelihood(SPN(), spn, data))
    print(f"log-L before EM: {ll_pre}")
    print(f"log-L after  EM: {ll_post}")


def local_em(spn: INode, data: np.ndarray, iterations: int = 10, hard_em: bool = False) -> None:
    # note: this does not work in SPNs, as we work at nodes locally, but have only global data, and do not know the assignment of instances to sub-SPNs
    ll_pre = np.sum(log_likelihood(SPN(), spn, data))

    # TODO: use toplogical order to pass through node via bottom up, ignore leafs
    for node in get_topological_order(spn):
        if isinstance(node, ILeafNode):
            continue
        local_em_update(node, data, iterations, hard_em)

    ll_post = np.sum(log_likelihood(SPN(), spn, data))
    print(f"log-L before EM: {ll_pre}")
    print(f"log-L after  EM: {ll_post}")


@dispatch(ISumNode, np.ndarray, int, bool)  # type: ignore[no-redef]
def local_em_update(
    node: ISumNode, data: np.ndarray, iterations: int, hard_em: bool
) -> None:  # DICTIONARY SOLUTION: node_updates: dict[INode, Callable],
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


@dispatch(IProductNode, np.ndarray, int, bool)  # type: ignore[no-redef]
def local_em_update(node: IProductNode, data: np.ndarray, iterations: int, hard_em: bool) -> None:
    for i in range(iterations):
        responsibilities = np.ones((len(data), 1))

        for i, child_node in enumerate(node.children):
            if isinstance(child_node, ILeafNode):
                update_parameters_em(child_node, data, responsibilities)


def compute_responsibilities(node: ISumNode, data: np.ndarray, hard_em: bool) -> np.ndarray:
    prior_cluster_probs = node.weights

    # dirty prototype
    datapoint_responsibilities = np.zeros(shape=(data.shape[0], len(prior_cluster_probs.tolist())))
    for datapoint_index, x in enumerate(data):
        for cluster_index, pi in enumerate(prior_cluster_probs.tolist()):
            cluster_node = node.children[cluster_index]
            ll = log_likelihood(SPN(), cluster_node, np.array([x]))
            gamma = pi * ll
            datapoint_responsibilities[datapoint_index, cluster_index] = gamma

        # normalize probabilities
        print(datapoint_responsibilities)
        datapoint_responsibilities[datapoint_index, :] /= np.sum(
            datapoint_responsibilities[datapoint_index, :]
        )

    if hard_em:
        # binarize cluster assignments
        datapoint_responsibilities[:] = np.where(
            datapoint_responsibilities == datapoint_responsibilities.max(axis=1).reshape(-1, 1),
            1,
            0,
        )

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
    if (
        equal is None
    ):  # assert_array_almost_equal() returns None if the arrays ARE almost equal, else an exception is raised
        print("Scipy PDF and Exp. Family PDF equality test for Gaussian was successful")


    data = np.array(
        [
            [3.0, 0],
            [2.5, 0],
            [3.5, 0],
            [3.1, 0],
            [2.8, 0],
            [0.5, 1],
            [0.2, 1],
            [0.7, 1],
            [1.1, 1],
            [0.3, 1],
        ]
    )

    spn = ISumNode(
        children=[
            IProductNode(children=[Gaussian([0], -2.0, 5.0), Bernoulli([1], 0.3)], scope=[0, 1]),
            IProductNode(children=[Gaussian([0], 4.5, 2.0), Bernoulli([1], 0.8)], scope=[0, 1]),
        ],
        scope=[0, 1],
        weights=np.array([0.1, 0.9]),
    )
    #top_ord = get_topological_order(spn)
    #print(top_ord)
    #local_em(spn, data, iterations=1, hard_em=True)
    poon_domingos_em(spn, data, iterations=100)
    _print_node_graph(spn)
    print(spn.weights)


    spn2 = IProductNode(
        children=[
            ISumNode(children=[Gaussian([0], -2.0, 5.0), Bernoulli([1], 0.3)], scope=[0, 1], weights=np.array([0.1, 0.9])),
            ISumNode(children=[Gaussian([0],  4.5, 2.0), Bernoulli([1], 0.8)], scope=[0, 1], weights=np.array([0.1, 0.9]))],
        scope=[0, 1])
    #top_ord = get_topological_order(spn2)
    #print(top_ord)
    #local_em(spn2, data, iterations=1, hard_em=False)
    poon_domingos_em(spn2, data, iterations=100)
    _print_node_graph(spn)
    print(spn.weights)







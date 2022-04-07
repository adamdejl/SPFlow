import unittest
import numpy as np
from sklearn.datasets import make_blobs
from spflow.base.learning import em
from spflow.base.structure.nodes import ISumNode, IProductNode, Gaussian, Bernoulli
from spflow.base.structure.nodes.node import _print_node_graph
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from copy import deepcopy


class TestEM(unittest.TestCase):


    def em_gaussian_clusters(self, spn, n_samples, centers, centers_std, em_iterations, seed=17) -> bool:
        data_points, cluster_assignments = make_blobs(n_samples=n_samples, n_features=1, centers=centers, cluster_std=centers_std, random_state=17)
        data = np.concatenate((data_points.reshape(n_samples, -1), cluster_assignments.reshape(n_samples, -1)), axis=1)

        time, ll_post, ll_pre = em(spn, data, iterations=em_iterations)

        print(ll_post, ll_pre)
        _print_node_graph(spn)

        return ll_post >= ll_pre


    def test_em_simple_gaussian(self):

        spn = 0.1 * (Gaussian([0], -2.0, 5.0) * Bernoulli([1], 0.3)) \
            + 0.9 * (Gaussian([0], 8.5, 2.0) * Bernoulli([1], 0.8))
        _isvalid_spn(spn)
        

        n_samples = [10, 100, 10000]
        centers = np.array([5.0, -5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = [1, 10, 100]

        for n in n_samples:
            for iter in em_iterations:
                self.assertTrue(self.em_gaussian_clusters(deepcopy(spn), n, centers, centers_std, iter))




if __name__ == "__main__":
    unittest.main()
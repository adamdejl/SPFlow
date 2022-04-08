from typing import Tuple
import unittest
import numpy as np
from sklearn.datasets import make_blobs
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.learning import em
from spflow.base.structure.nodes import ISumNode, IProductNode, Gaussian, Bernoulli
from spflow.base.structure.nodes.leaves.parametric.categorical import Categorical
from spflow.base.structure.nodes.node import _print_node_graph
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from copy import deepcopy
from scipy.stats import poisson, beta, expon, gamma, geom


class TestEM(unittest.TestCase):
    def em_gaussian_clusters(
        self, spn, n_samples, centers, centers_std, em_iterations, seed=17
    ) -> Tuple[float, float, np.ndarray]:
        data_points, cluster_assignments = make_blobs(
            n_samples=n_samples,
            n_features=1,
            centers=centers,
            cluster_std=centers_std,
            random_state=17,
        )
        data = np.concatenate(
            (data_points.reshape(n_samples, -1), cluster_assignments.reshape(n_samples, -1)), axis=1
        )

        time, ll_post, ll_pre = em(spn, data, iterations=em_iterations)
        # print(ll_post, ll_pre)
        # _print_node_graph(spn)
        return ll_post, ll_pre, data

    def test_em_simple_gaussian_likelihood_improvement(self):

        spn = 0.1 * (Gaussian([0], -2.0, 5.0) * Bernoulli([1], 0.3)) + 0.9 * (
            Gaussian([0], 8.5, 3.5) * Bernoulli([1], 0.8)
        )
        _isvalid_spn(spn)

        n_samples = [10, 1000, 100000]
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = [1, 10, 100]

        for n in n_samples:
            for iter in em_iterations:
                ll_post, ll_pre, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                self.assertTrue(ll_post >= ll_pre)

    def test_em_simple_gaussian_random_initialization_likelihood_improvement_1(self):

        spn = 0.1 * (Gaussian([0]) * Bernoulli([1])) + 0.9 * (Gaussian([0]) * Bernoulli([1]))
        _isvalid_spn(spn)

        n_samples = [10, 1000, 100000]
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = [1, 10, 100]

        for n in n_samples:
            for iter in em_iterations:
                ll_post, ll_pre, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                self.assertTrue(ll_post >= ll_pre)

    def test_em_simple_gaussian_random_initialization_likelihood_improvement_2(self):

        spn = 0.1 * (Categorical([1], k=2) * (0.3 * Gaussian([0]) + 0.7 * Gaussian([0]))) + 0.9 * (
            Categorical([1], k=2) * (0.8 * Gaussian([0]) + 0.2 * Gaussian([0]))
        )
        _isvalid_spn(spn)

        n_samples = [10, 1000, 100000]
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = [1, 10, 100]

        for n in n_samples:
            for iter in em_iterations:
                ll_post, ll_pre, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                self.assertTrue(ll_post >= ll_pre)

    def test_em_multiple_gaussian_clusters_likelihood_improvement(self):

        spn = ISumNode(
            children=[
                (Gaussian([0], -100.0, 5.0) * Categorical([1], k=5)),
                (Gaussian([0], -76.0, 25.0) * Categorical([1], k=5)),
                (Gaussian([0], 3.5, 12.0) * Categorical([1], k=5)),
                (Gaussian([0], 25.0, 3.5) * Categorical([1], k=5)),
                (Gaussian([0], 345.0, 50.0) * Categorical([1], k=5)),
            ],
            weights=np.array([0.1, 0.2, 0.3, 0.3, 0.1]),
            scope=[0, 1],
        )
        _isvalid_spn(spn)

        n_samples = [10, 1000, 100000]
        centers = np.array([-100.0, -50.0, 0.0, 50.0, 100.0]).reshape(-1, 1)
        centers_std = 5.0
        em_iterations = [1, 10, 100]

        for n in n_samples:
            for iter in em_iterations:
                ll_post, ll_pre, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                self.assertTrue(ll_post >= ll_pre)

    def test_em_validate_parameters_simple(self):
        spn = 0.1 * (Gaussian([0], -2.0, 5.0) * Bernoulli([1], 0.3)) + 0.9 * (
            Gaussian([0], 8.5, 3.5) * Bernoulli([1], 0.8)
        )
        _isvalid_spn(spn)

        n_samples = 1000
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = 10

        ll_post, ll_pre, data = self.em_gaussian_clusters(
            spn, n_samples, centers, centers_std, em_iterations
        )

        self.assertTrue(ll_post > ll_pre)
        data_cluster_0 = data[data[:, 1] == 0]
        data_cluster_1 = data[data[:, 1] == 1]
        self.assertAlmostEqual(spn.weights[0], len(data_cluster_0) / len(data))
        self.assertAlmostEqual(spn.weights[1], len(data_cluster_1) / len(data))
        cluster_0_gaussian: Gaussian = spn.children[0].children[0]
        cluster_0_bernoulli: Bernoulli = spn.children[0].children[1]
        self.assertAlmostEqual(cluster_0_gaussian.mean, np.mean(data_cluster_0[:, 0]))
        self.assertAlmostEqual(cluster_0_gaussian.stdev, np.std(data_cluster_0[:, 0]))
        self.assertAlmostEqual(cluster_0_bernoulli.p, np.mean(data_cluster_0[:, 1]))
        cluster_1_gaussian: Gaussian = spn.children[1].children[0]
        cluster_1_bernoulli: Bernoulli = spn.children[1].children[1]
        self.assertAlmostEqual(cluster_1_gaussian.mean, np.mean(data_cluster_1[:, 0]))
        self.assertAlmostEqual(cluster_1_gaussian.stdev, np.std(data_cluster_1[:, 0]))
        self.assertAlmostEqual(cluster_1_bernoulli.p, np.mean(data_cluster_1[:, 1]))

    def test_em_mixed_spn(self):
        # test SPNs with Poisson, Beta, Gamma, Expon, Geometric nodes
        pass


if __name__ == "__main__":
    unittest.main()

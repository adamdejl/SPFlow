from typing import List, Tuple
import unittest
import numpy as np
from sklearn.datasets import make_blobs  # type: ignore
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.learning import em
from spflow.base.structure.nodes import (
    ISumNode,
    IProductNode,
    Gaussian,
    Bernoulli,
    Categorical,
    Exponential,
    Gamma,
    Geometric,
    Poisson,
)
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from copy import deepcopy
from scipy.stats import poisson, beta, expon, gamma, geom  # type: ignore


class TestEM(unittest.TestCase):
    def setUp(self):
        np.random.seed(17)

    def em_gaussian_clusters(
        self, spn, n_samples, centers, centers_std, em_iterations, seed=17
    ) -> Tuple[List[float], np.ndarray]:
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

        loll, test = em(spn, data, iterations=em_iterations)
        # print(ll_post, ll_pre)
        # _print_node_graph(spn)
        return loll, data

    def test_em_simple_gaussian_likelihood_improvement(self):

        spn = 0.1 * (Gaussian([0], -2.0, 5.0) * Bernoulli([1], 0.3)) + 0.9 * (
            Gaussian([0], 8.5, 3.5) * Bernoulli([1], 0.8)
        )
        _isvalid_spn(spn)

        n_samples = [10, 100, 10000]
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = [1, 10, -1]

        for n in n_samples:
            for iter in em_iterations:
                loll, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)

    def test_em_simple_gaussian_random_initialization_likelihood_improvement_1(self):

        spn = 0.1 * (Gaussian([0]) * Bernoulli([1])) + 0.9 * (Gaussian([0]) * Bernoulli([1]))
        _isvalid_spn(spn)

        n_samples = [10, 100, 10000]
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = [1, 10, -1]

        for n in n_samples:
            for iter in em_iterations:
                loll, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)

    def test_em_simple_gaussian_random_initialization_likelihood_improvement_2(self):

        spn = 0.1 * (Categorical([1], k=2) * (0.3 * Gaussian([0]) + 0.7 * Gaussian([0]))) + 0.9 * (
            Categorical([1], k=2) * (0.8 * Gaussian([0]) + 0.2 * Gaussian([0]))
        )
        _isvalid_spn(spn)

        n_samples = [10, 100, 10000]
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = [1, 10, -1]

        for n in n_samples:
            for iter in em_iterations:
                loll, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)

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

        n_samples = [10, 100, 10000]
        centers = np.array([-100.0, -50.0, 0.0, 50.0, 100.0]).reshape(-1, 1)
        centers_std = 5.0
        em_iterations = [1, 10, -1]

        for n in n_samples:
            for iter in em_iterations:
                loll, _ = self.em_gaussian_clusters(
                    deepcopy(spn), n, centers, centers_std, iter
                )
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)

    def test_em_validate_parameters_simple(self):
        spn = 0.1 * (Gaussian([0], -2.0, 5.0) * Bernoulli([1], 0.3)) + 0.9 * (
            Gaussian([0], 8.5, 3.5) * Bernoulli([1], 0.8)
        )
        _isvalid_spn(spn)

        n_samples = 1000
        centers = np.array([-5.0, 5.0]).reshape(-1, 1)
        centers_std = 2.0
        em_iterations = 10

        loll, data = self.em_gaussian_clusters(
            spn, n_samples, centers, centers_std, em_iterations
        )

        for i in range (1, len(loll)):
            self.assertTrue(loll[i-1] <= loll[i], msg=loll)
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

    def test_em_pure_gaussian_tree(self):
        n_samples = [10, 100, 10000]
        em_iterations = [1, 10, -1]
        centers = [(5.0, 5.0), (-5.0, -5.0)]
        cluster_std = 3.0

        spn = 0.1 * (Gaussian([0]) * Gaussian([1])) + 0.9 * (Gaussian([0]) * Gaussian([1]))
        _isvalid_spn(spn)

        for n in n_samples:
            data, _ = make_blobs(
                n_samples=n,
                n_features=len(centers),
                centers=centers,
                cluster_std=cluster_std,
                random_state=17,
            )
            for iter in em_iterations:
                loll, test = em(deepcopy(spn), data, iter)
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)

    def test_em_pure_gaussian_graph(self):
        n_samples = [10, 100, 10000]
        em_iterations = [1, 10, -1]
        centers = [(5.0, 5.0), (-5.0, -5.0)]
        cluster_std = 3.0

        gaussian_1 = Gaussian([0])
        gaussian_2 = Gaussian([1])
        product_1 = IProductNode(children=[gaussian_1, gaussian_2], scope=[0, 1])
        product_2 = IProductNode(children=[gaussian_1, gaussian_2], scope=[0, 1])
        spn = ISumNode(children=[product_1, product_2], weights=np.array([0.3, 0.7]), scope=[0, 1])
        _isvalid_spn(spn)

        for n in n_samples:
            data, _ = make_blobs(
                n_samples=n,
                n_features=len(centers),
                centers=centers,
                cluster_std=cluster_std,
                random_state=17,
            )
            for iter in em_iterations:
                loll, test = em(deepcopy(spn), data, iter)
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)

    def test_em_mixed_spn_fully_factorized(self):
        # test SPNs with Poisson, Beta, Gamma, Expon, Geometric nodes
        n_samples = [10, 100, 10000]
        em_iterations = [1, 10, -1]

        distributions = [poisson, gamma, expon, geom]
        distributions_params = [
            {"mu": 10.0},
            {"a": 2.0, "scale": 1.0 / 2.0},
            {"scale": 1.0 / 2.0},
            {"p": 0.5},
        ]

        spn = IProductNode(
            children=[Poisson([0]), Gamma([1]), Exponential([2]), Geometric([3])],
            scope=[0, 1, 2, 3],
        )
        _isvalid_spn(spn)

        for n in n_samples:
            data = np.random.random((n, len(distributions)))
            for i, distr in enumerate(distributions):
                data[:, i] = distr.ppf(data[:, i], **distributions_params[i])

            for iter in em_iterations:
                loll, test = em(deepcopy(spn), data, iter)
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)

    def test_em_deep_mixed_spn(self):
        # test SPNs with Poisson, Beta, Gamma, Expon, Geometric nodes
        n_samples = [10, 100, 10000]
        em_iterations = [1, 10, -1]

        distributions = [poisson, gamma, expon, geom]
        distributions_params_1 = [
            {"mu": 10.0},
            {"a": 2.0, "scale": 1.0 / 2.0},
            {"scale": 1.0 / 2.0},
            {"p": 0.5},
        ]
        distributions_params_2 = [
            {"mu": 3.0},
            {"a": 1.1, "scale": 1.0},
            {"scale": 1.0 / 5.0},
            {"p": 0.7},
        ]

        spn = ((0.2 * (Poisson([0]) * Gamma([1]))) + (0.8 * (Poisson([0]) * Gamma([1])))) * (
            (0.6 * (Exponential([2]) * Geometric([3])))
            + (0.4 * (Exponential([2]) * Geometric([3])))
        )
        _isvalid_spn(spn)

        for n in n_samples:
            data = np.random.random((n, len(distributions)))
            for i, distr in enumerate(distributions):
                data[: len(data), i] = distr.ppf(data[: len(data), i], **distributions_params_1[i])
                data[len(data) :, i] = distr.ppf(data[len(data) :, i], **distributions_params_2[i])

            for iter in em_iterations:
                loll, test = em(deepcopy(spn), data, iter)
                for i in range (1, len(loll)):
                    self.assertTrue(loll[i-1] <= loll[i], msg=loll)


if __name__ == "__main__":
    unittest.main()

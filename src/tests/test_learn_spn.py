import unittest
from spflow.base.inference.nodes import likelihood
from spflow.base.learning.context import Context
from spflow.base.learning.learn_spn import create_parametric_leaf, learn_parametric_spn
from spflow.base.structure.network_type import SPN
from spflow.base.structure.nodes.leaves.parametric import (
    Gaussian,
    LogNormal,
    Bernoulli,
    Categorical,
    CategoricalDictionary,
    Gamma,
    Poisson,
    Geometric,
    Exponential,
    ParametricLeaf,
    get_scipy_object_parameters,
)
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
import numpy as np


class TestLearnSPN(unittest.TestCase):
    def setUp(self):
        self.tested = set()

    def get_2d_gauss_clusters(self, instances=1000, ratio=0.5):
        assert 0.0 <= ratio <= 1.0
        from sklearn.datasets import make_blobs  # type: ignore

        cluster_instances = [round(instances * ratio), round(instances * (1 - ratio))]
        cluster_centers = [(-5, -5), (5, 5)]
        cluster_stdev = [3, 8]
        data, label = make_blobs(
            n_samples=cluster_instances,
            centers=cluster_centers,
            cluster_std=cluster_stdev,
            n_features=2,
            random_state=17,
        )
        data = np.concatenate((data, label.reshape(-1, 1)), axis=1)
        return data

    def assert_learn_parametric_spn_2d_gaussian_clusters(
        self, split_cols, split_rows, min_instances_ratio, threshold
    ):
        instances = (100, 10000)
        ratios = (0.2, 0.5)

        for i in instances:
            for r in ratios:
                data = self.get_2d_gauss_clusters(instances=i, ratio=r)
                context = Context(parametric_types=[Gaussian, Gaussian, Categorical]) # Context(parametric_types=[Gaussian, Gaussian, Bernoulli]) #
                context.add_domains(data)
                min_instances_slice = len(data) * min_instances_ratio
                spn = learn_parametric_spn(
                    data=data,
                    context=context,
                    split_cols=split_cols,
                    split_rows=split_rows,
                    min_instances_slice=min_instances_slice,
                    threshold=threshold,
                )
                _isvalid_spn(spn)

                marginal_inference_cluster0 = likelihood(
                    SPN(), spn, np.array([[np.NaN, np.NaN, 0], [np.NaN, np.NaN, 1]]).reshape(-1, 3)
                )
                np.testing.assert_almost_equal(marginal_inference_cluster0.flatten(), [r, 1 - r])

    # TODO: assert full inference of learned spn and marginal inference w.r.t. ratio (-> bernoulli variable)
    def test_learn_parametric_spn_kmeans_rdc_fully_factorized(self):
        spn_arguments = {
            "split_cols": "rdc",
            "split_rows": "kmeans",
            "min_instances_ratio": 1.0,
            "threshold": 0.3,
        }

        self.assert_learn_parametric_spn_2d_gaussian_clusters(**spn_arguments)

    def test_learn_parametric_spn_kmeans_rdc(self):
        spn_arguments = {
            "split_cols": "rdc",
            "split_rows": "kmeans",
            "min_instances_ratio": 0.1,
            "threshold": 0.3,
        }

        self.assert_learn_parametric_spn_2d_gaussian_clusters(**spn_arguments)

    def test_learn_parametric_spn_gmm_rdc(self):
        spn_arguments = {
            "split_cols": "rdc",
            "split_rows": "gmm",
            "min_instances_ratio": 0.3,
            "threshold": 0.3,
        }

        self.assert_learn_parametric_spn_2d_gaussian_clusters(**spn_arguments)

    def test_learn_parametric_spn_rdc_rdc(self):
        spn_arguments = {
            "split_cols": "rdc",
            "split_rows": "rdc",
            "min_instances_ratio": 0.3,
            "threshold": 0.3,
        }

        self.assert_learn_parametric_spn_2d_gaussian_clusters(**spn_arguments)

    def test_get_splitting_functions(self):
        pass

    def assert_parametric_leaf(self, expected, data):
        self.tested.add(type(expected))

        domains = [[np.min(data), np.max(data)]]

        mle = create_parametric_leaf(
            data, context=Context(parametric_types=[type(expected)], domains=domains), scope=[0]
        )

        exp_param = get_scipy_object_parameters(expected)
        mle_param = get_scipy_object_parameters(mle)

        self.assertEqual(len(exp_param), len(mle_param))
        keys = exp_param.keys()
        for key in keys:
            self.assertAlmostEqual(exp_param[key], mle_param[key])

        return mle

    def test_parametric_leaf(self):
        data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        self.assert_parametric_leaf(
            Gaussian(mean=np.mean(data), stdev=np.std(data), scope=[0]), data
        )

        node = self.assert_parametric_leaf(
            Gamma(alpha=3.701643810008814, beta=1.233881270002938, scope=[0]), data
        )
        self.assertEqual(node.alpha / node.beta, np.mean(data))

        self.assert_parametric_leaf(
            LogNormal(mean=np.log(data).mean(), stdev=np.log(data).std(), scope=[0]), data
        )

        self.assert_parametric_leaf(Poisson(l=np.mean(data), scope=[0]), data)

        self.assert_parametric_leaf(Exponential(l=np.mean(data), scope=[0]), data)

        self.assert_parametric_leaf(Geometric(p=1.0 / np.mean(data), scope=[0]), data)

        data = np.array([0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
        self.assert_parametric_leaf(Bernoulli(p=0.75, scope=[0]), data)

        data = np.array([0, 0, 1, 3, 5, 6, 6, 6, 6, 6]).reshape(-1, 1)
        expected = Categorical(p=[2 / 10, 1 / 10, 0 / 10, 1 / 10, 0 / 10, 1 / 10, 5 / 10], scope=[0])
        context = Context(parametric_types=[Categorical]).add_domains(data)
        mle = create_parametric_leaf(data=data, context=context, scope=[0])
        self.assertListEqual(expected.p, mle.p)
        self.tested.add(Categorical)

        data = np.array([0, 0, 1, 3, 5, 6, 6, 6, 6, 6]).reshape(-1, 1)
        expected = CategoricalDictionary(p={0: 2/10, 1: 1/10, 3: 1/10, 5: 1/10, 6: 5/10}, scope=[0])
        mle = create_parametric_leaf(data=data, context=Context(parametric_types=[CategoricalDictionary]), scope=[0])
        self.assertDictEqual(expected.p, mle.p)
        self.tested.add(CategoricalDictionary)

        for child in ParametricLeaf.__subclasses__():
            if child not in self.tested:
                print("not tested", child)


if __name__ == "__main__":
    unittest.main()

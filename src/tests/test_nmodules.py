import unittest
import numpy as np
from spflow.base.structure.nmodule import NNode, NSumNode, NProductNode, NLeafNode
from spflow.base.structure.nmodules.nmodule import ProductLayer, SumLayer, _get_node_counts
from spflow.base.structure.network_type import SPN, set_network_type
from spflow.base.structure.nmodules.validity_checks import _isvalid_spn
from spflow.base.learning.context import RandomVariableContext  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import Gaussian
from spflow.base.inference.nmodule import likelihood, log_likelihood


class TestNode(unittest.TestCase):
    def test_spn_fail_scope1(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            spn_module: NNode = NProductNode(
                children=[
                    NLeafNode(scope=[1], context=context),
                    NLeafNode(scope=[1], context=context),
                ],
                scope=[0, 1],
            )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn_module)

    def test_spn_fail_scope2(self):
        with self.assertRaises(IndexError):
            context = RandomVariableContext(parametric_types=[Gaussian])

            with set_network_type(SPN()):
                spn_module: NNode = NProductNode(
                    children=[
                        NLeafNode(scope=[0], context=context),
                        NLeafNode(scope=[1], context=context),
                    ],
                    scope=[0],
                )

    def test_spn_fail_scope3(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            spn_module: NNode = NProductNode(
                children=[
                    NLeafNode(scope=[0], context=context),
                    NLeafNode(scope=[1], context=context),
                ],
                scope=[0],
                network_type=SPN(),
            )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn_module)

    def test_spn_fail_weights1(self):
        with self.assertRaises(AssertionError):
            context = RandomVariableContext(parametric_types=[Gaussian])

            with set_network_type(SPN()):
                spn_module: NNode = NSumNode(
                    children=[
                        NLeafNode(scope=[0], context=context),
                        NLeafNode(scope=[0], context=context),
                    ],
                    scope=[0],
                    weights=np.array([0.49, 0.49]),
                )

    def test_spn_fail_weights2(self):

        context = RandomVariableContext(parametric_types=[Gaussian])

        with set_network_type(SPN()):
            spn_module: NNode = NSumNode(
                children=[
                    NLeafNode(scope=[0], context=context),
                    NLeafNode(scope=[0], context=context),
                ],
                scope=[0],
                weights=np.array([1.0]),
            )
        with self.assertRaises(AssertionError):
            _isvalid_spn(spn_module)

    def test_spn_missing_children(self):

        with set_network_type(SPN()):
            with self.assertRaises(AssertionError):
                spn_module: NNode = NProductNode(scope=[0, 1], children=None)

    def test_spn_fail_leaf_with_children(self):
        context = RandomVariableContext(parametric_types=[Gaussian])
        with set_network_type(SPN()):
            spn_module: NNode = NSumNode(
                children=[
                    NLeafNode(scope=[0], context=context),
                    NLeafNode(scope=[0], context=context),
                ],
                scope=[0],
                weights=np.array([0.5, 0.5]),
            )

            # make sure SPN is valid to begin with
            _isvalid_spn(spn_module)

            spn_module.children[0].output_modules[0].children.append(
                NLeafNode(scope=[0], context=context)
            )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn_module)

    def test_spn_fail_none_children(self):
        with self.assertRaises(AssertionError):
            context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

            with set_network_type(SPN()):
                spn_module: NNode = NProductNode(
                    children=[NLeafNode(scope=[0], context=context), None], scope=[1, 0]
                )

    def test_spn_tree_small(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            spn_module: NNode = NProductNode(
                children=[
                    NSumNode(
                        children=[
                            NLeafNode(scope=[0], context=context),
                            NLeafNode(scope=[0], context=context),
                        ],
                        scope=[0],
                        weights=np.array([0.3, 0.7]),
                    ),
                    NLeafNode(scope=[1], network_type=SPN(), context=context),
                ],
                scope=[0, 1],
            )
        _isvalid_spn(spn_module)
        result = _get_node_counts(spn_module)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 1)
        self.assertEqual(leaf_nodes, 3)

    def test_spn_tree_big(self):
        context = RandomVariableContext(
            parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian, Gaussian]
        )

        with set_network_type(SPN()):
            spn_module: NNode = NProductNode(
                children=[
                    NSumNode(
                        children=[
                            NProductNode(
                                children=[
                                    NSumNode(
                                        children=[
                                            NProductNode(
                                                children=[
                                                    NLeafNode(scope=[0], context=context),
                                                    NLeafNode(scope=[1], context=context),
                                                ],
                                                scope=[0, 1],
                                                network_type=SPN(),
                                            ),
                                            NLeafNode(scope=[0, 1], context=context),
                                        ],
                                        scope=[0, 1],
                                        weights=np.array([0.9, 0.1]),
                                    ),
                                    NLeafNode(scope=[2], context=context),
                                ],
                                scope=[0, 1, 2],
                            ),
                            NProductNode(
                                children=[
                                    NLeafNode(scope=[0], context=context),
                                    NSumNode(
                                        children=[
                                            NLeafNode(scope=[1, 2], context=context),
                                            NLeafNode(scope=[1, 2], context=context),
                                        ],
                                        scope=[1, 2],
                                        weights=np.array([0.5, 0.5]),
                                    ),
                                ],
                                scope=[0, 1, 2],
                            ),
                            NLeafNode(scope=[0, 1, 2], context=context),
                        ],
                        scope=[0, 1, 2],
                        weights=np.array([0.4, 0.1, 0.5]),
                    ),
                    NSumNode(
                        children=[
                            NProductNode(
                                children=[
                                    NLeafNode(scope=[3], context=context),
                                    NLeafNode(scope=[4], context=context),
                                ],
                                scope=[3, 4],
                            ),
                            NLeafNode(scope=[3, 4], context=context),
                        ],
                        scope=[3, 4],
                        weights=np.array([0.5, 0.5]),
                    ),
                ],
                scope=[0, 1, 2, 3, 4],
            )

        _isvalid_spn(spn_module)
        result = _get_node_counts(spn_module)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 4)
        self.assertEqual(prod_nodes, 5)
        self.assertEqual(leaf_nodes, 11)

    def test_spn_graph_small(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf1 = NLeafNode(scope=[0], context=context)
            leaf2 = NLeafNode(scope=[1], context=context)
            prod1 = NProductNode(children=[leaf1, leaf2], scope=[0, 1])
            prod2 = NProductNode(children=[leaf1, leaf2], scope=[0, 1])
            sum = NSumNode(children=[prod1, prod2], scope=[0, 1], weights=np.array([0.3, 0.7]))

        _isvalid_spn(sum)
        result = _get_node_counts(sum)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 2)
        self.assertEqual(leaf_nodes, 2)

    def test_spn_graph_medium(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)
            sum_11 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.3, 0.7]))
            sum_12 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.9, 0.1]))
            sum_21 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.4, 0.6]))
            sum_22 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.8, 0.2]))
            prod_11 = NProductNode(children=[sum_11, sum_21], scope=[0, 1])
            prod_12 = NProductNode(children=[sum_11, sum_22], scope=[0, 1])
            prod_13 = NProductNode(children=[sum_12, sum_21], scope=[0, 1])
            prod_14 = NProductNode(children=[sum_12, sum_22], scope=[0, 1])
            sum = NSumNode(
                children=[prod_11, prod_12, prod_13, prod_14],
                scope=[0, 1],
                weights=np.array([0.1, 0.2, 0.3, 0.4]),
            )

        _isvalid_spn(sum)
        result = _get_node_counts(sum)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 5)
        self.assertEqual(prod_nodes, 4)
        self.assertEqual(leaf_nodes, 4)

    def test_product_layer_children(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)

            with self.assertRaises(AssertionError):
                product_layer_invalid_children = ProductLayer(
                    num_of_nodes=2,
                    num_of_children=2,
                    children=[leaf_11, leaf_12, leaf_21],
                    scope=[0, 1],
                )
            with self.assertRaises(AssertionError):
                product_layer_invalid_children2 = ProductLayer(
                    num_of_nodes=2,
                    num_of_children=2,
                    children=[leaf_11, leaf_12, leaf_21, None],
                    scope=[0, 1],
                )
            with self.assertRaises(AssertionError):
                product_layer_invalid_children4 = ProductLayer(
                    num_of_nodes=2,
                    num_of_children=2,
                    children=None,
                    scope=[0, 1],
                )
            with self.assertRaises(AssertionError):
                product_layer_invalid_num_of_nodes = ProductLayer(
                    num_of_nodes=3,
                    num_of_children=2,
                    children=[leaf_11, leaf_12, leaf_21, leaf_22],
                    scope=[0, 1],
                )

            product_layer_invalid_scope = ProductLayer(
                num_of_nodes=2,
                num_of_children=2,
                children=[leaf_11, leaf_12, leaf_21, leaf_22],
                scope=[0, 1, 2],
            )

            with self.assertRaises(AssertionError):
                _isvalid_spn(product_layer_invalid_scope)

            prod_layer1 = ProductLayer(
                num_of_nodes=2,
                num_of_children=2,
                children=[leaf_11, leaf_12, leaf_21, leaf_22],
                scope=[0, 1],
            )
            prod_layer2 = ProductLayer(
                num_of_nodes=2,
                num_of_children=2,
                children=[leaf_11, leaf_21, leaf_12, leaf_22],
                scope=[0, 1],
            )

            with self.assertRaises(AssertionError):
                _isvalid_spn(prod_layer1)

            _isvalid_spn(prod_layer2)

            result = _get_node_counts(prod_layer2)
            sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
            self.assertEqual(sum_nodes, 0)
            self.assertEqual(prod_nodes, 2)
            self.assertEqual(leaf_nodes, 4)

    def test_sum_with_product_layer_child(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)
            prod_layer = ProductLayer(
                num_of_nodes=2,
                num_of_children=2,
                children=[leaf_11, leaf_21, leaf_12, leaf_22],
                scope=[0, 1],
            )
            sum1 = NSumNode(scope=[0, 1], weights=np.array([0.5] * 2), children=[prod_layer])
            sum2 = NSumNode(scope=[0, 1], weights=np.array([0.25] * 4), children=[prod_layer])
            multi_leaf = NLeafNode(scope=[0, 1], context=context)
            sum3 = NSumNode(
                scope=[0, 1], weights=np.array([1 / 3] * 3), children=[multi_leaf, prod_layer]
            )
            sum4 = NSumNode(
                scope=[0, 1],
                weights=np.array([1 / 4] * 4),
                children=[multi_leaf, multi_leaf, prod_layer],
            )

            _isvalid_spn(sum1)
            _isvalid_spn(sum3)
            with self.assertRaises(AssertionError):
                _isvalid_spn(sum2)
            _isvalid_spn(sum4)

        result = _get_node_counts(sum1)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 2)
        self.assertEqual(leaf_nodes, 4)

        result = _get_node_counts(sum3)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 2)
        self.assertEqual(leaf_nodes, 5)

        result = _get_node_counts(sum4)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 2)
        self.assertEqual(leaf_nodes, 5)

    def test_spn_graph_medium_nnodes(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)
            sum_11 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.3, 0.7]))
            sum_12 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.9, 0.1]))
            sum_21 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.4, 0.6]))
            sum_22 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.8, 0.2]))
            prod_11 = NProductNode(children=[sum_11, sum_21], scope=[0, 1])
            prod_12 = NProductNode(children=[sum_11, sum_22], scope=[0, 1])
            prod_13 = NProductNode(children=[sum_12, sum_21], scope=[0, 1])
            prod_14 = NProductNode(children=[sum_12, sum_22], scope=[0, 1])
            sum = NSumNode(
                children=[prod_11, prod_12, prod_13, prod_14],
                scope=[0, 1],
                weights=np.array([0.1, 0.2, 0.3, 0.4]),
            )

        _isvalid_spn(sum)

        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(sum)
        self.assertEqual(sum_nodes, 5)
        self.assertEqual(prod_nodes, 4)
        self.assertEqual(leaf_nodes, 4)

    """def test_inference_log_ll(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian, Gaussian])

        with set_network_type(SPN()):
            spn = NSumNode(
                children=[
                    NProductNode(
                        children=[
                            NLeafNode(scope=[0], context=context),
                            NSumNode(
                                children=[
                                    NProductNode(
                                        children=[
                                            NLeafNode(scope=[1], context=context),
                                            NLeafNode(scope=[2], context=context),
                                        ],
                                        scope=[1, 2],
                                    ),
                                    NProductNode(
                                        children=[
                                            NLeafNode(scope=[1], context=context),
                                            NLeafNode(scope=[2], context=context),
                                        ],
                                        scope=[1, 2],
                                    ),
                                ],
                                scope=[1, 2],
                                weights=np.array([0.3, 0.7]),
                            ),
                        ],
                        scope=[0, 1, 2],
                    ),
                    NProductNode(
                        children=[
                            NProductNode(
                                children=[
                                    NLeafNode(scope=[0], context=context),
                                    NLeafNode(scope=[1], context=context),
                                ],
                                scope=[0, 1],
                            ),
                            NLeafNode(scope=[2], context=context),
                        ],
                        scope=[0, 1, 2],
                    ),
                ],
                scope=[0, 1, 2],
                weights=np.array([0.4, 0.6]),
            )

            _isvalid_spn(spn)

            result = log_likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
            self.assertAlmostEqual(result[0][0], -2.33787707)

            result = likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
            self.assertAlmostEqual(result[0][0], np.exp(-2.33787707))"""

    def test_spn_prod_layer(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)
            sum_11 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.3, 0.7]))
            sum_12 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.9, 0.1]))
            sum_21 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.4, 0.6]))
            sum_22 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.8, 0.2]))
            prod_11 = NProductNode(children=[sum_11, sum_21], scope=[0, 1])
            prod_12 = NProductNode(children=[sum_11, sum_22], scope=[0, 1])
            prod_13 = NProductNode(children=[sum_12, sum_21], scope=[0, 1])
            prod_14 = NProductNode(children=[sum_12, sum_22], scope=[0, 1])
            spn_nodes = NSumNode(
                children=[prod_11, prod_12, prod_13, prod_14],
                scope=[0, 1],
                weights=np.array([0.1, 0.2, 0.3, 0.4]),
            )

            _isvalid_spn(spn_nodes)

            result_nodes_ll = log_likelihood(spn_nodes, np.array([np.nan, 1.0]).reshape(-1, 2))
            result_nodes_l = likelihood(spn_nodes, np.array([np.nan, 1.0]).reshape(-1, 2))

            prod_layer = ProductLayer(
                num_of_nodes=4,
                num_of_children=2,
                scope=[0, 1],
                children=[sum_11, sum_21, sum_11, sum_22, sum_12, sum_21, sum_12, sum_22],
            )
            spn_layer = NSumNode(
                children=[prod_layer],
                scope=[0, 1],
                weights=np.array([0.1, 0.2, 0.3, 0.4]),
            )

            _isvalid_spn(spn_layer)

            result_layer_ll = log_likelihood(spn_layer, np.array([np.nan, 1.0]).reshape(-1, 2))
            result_layer_l = likelihood(spn_layer, np.array([np.nan, 1.0]).reshape(-1, 2))

            self.assertAlmostEqual(result_layer_ll[0][0], result_nodes_ll[0][0])
            self.assertAlmostEqual(result_layer_l[0][0], result_nodes_l[0][0])

    def test_sum_layer_children(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)

            with self.assertRaises(AssertionError):
                sum_layer_invalid_weights = SumLayer(
                    num_of_nodes=2,
                    weights=np.array([0.5, 0.4]),
                    children=[leaf_11, leaf_12, leaf_21, leaf_22],
                    scope=[0, 1],
                )
                sum_layer_invalid_children = SumLayer(
                    num_of_nodes=2,
                    weights=np.array([0.5, 0.5]),
                    children=[leaf_11, leaf_12, leaf_21],
                    scope=[0, 1],
                )
                sum_layer_invalid_children2 = SumLayer(
                    num_of_nodes=2,
                    weights=np.array([0.5, 0.5]),
                    children=[leaf_11, leaf_12, leaf_21, None],
                    scope=[0, 1],
                )
                sum_layer_invalid_children3 = SumLayer(
                    num_of_nodes=2,
                    weights=np.array([0.5, 0.5]),
                    children=None,
                    scope=[0, 1],
                )
                sum_layer_invalid_num_of_nodes = SumLayer(
                    num_of_nodes=3,
                    weights=np.array([0.5, 0.5]),
                    children=[leaf_11, leaf_12, leaf_21, leaf_22],
                    scope=[0, 1],
                )
                sum_layer_invalid_scope = SumLayer(
                    num_of_nodes=2,
                    weights=np.array([0.5, 0.5]),
                    children=[leaf_11, leaf_12, leaf_21, leaf_22],
                    scope=[0, 1, 2],
                )
            sum_layer_invalid_children4 = SumLayer(
                num_of_nodes=2,
                weights=np.array([0.5, 0.5]),
                children=[leaf_11, leaf_21, leaf_21, leaf_22],
                scope=[0, 1],
            )

            with self.assertRaises(AssertionError):
                _isvalid_spn(sum_layer_invalid_children4)

            sum_layer = SumLayer(
                num_of_nodes=2,
                weights=np.array([0.5, 0.5]),
                children=[leaf_11, leaf_12, leaf_21, leaf_22],
                scope=[0, 1],
            )

            _isvalid_spn(sum_layer)

    def test_product_with_sum_layer_child(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)
            leaf_3 = NLeafNode(scope=[2], context=context)
            leaf_4 = NLeafNode(scope=[3], context=context)
            sum_layer = SumLayer(
                num_of_nodes=2,
                weights=np.array([0.5, 0.5]),
                children=[leaf_11, leaf_12, leaf_21, leaf_22],
                scope=[0, 1],
            )
            prod1 = NProductNode(scope=[0, 1], children=[sum_layer])
            prod2 = NProductNode(scope=[0, 1, 2], children=[leaf_3, sum_layer])
            prod3 = NProductNode(scope=[0, 1, 2], children=[sum_layer])
            prod4 = NProductNode(
                scope=[0, 1, 2, 3],
                children=[leaf_3, leaf_4, sum_layer],
            )

            _isvalid_spn(prod1)
            _isvalid_spn(prod2)
            with self.assertRaises(AssertionError):
                _isvalid_spn(prod3)
            _isvalid_spn(prod4)

        result = _get_node_counts(prod1)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 2)
        self.assertEqual(prod_nodes, 1)
        self.assertEqual(leaf_nodes, 4)

        result = _get_node_counts(prod2)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 2)
        self.assertEqual(prod_nodes, 1)
        self.assertEqual(leaf_nodes, 5)

        result = _get_node_counts(prod4)
        sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
        self.assertEqual(sum_nodes, 2)
        self.assertEqual(prod_nodes, 1)
        self.assertEqual(leaf_nodes, 6)

    def test_spn_sum_layer(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian])

        with set_network_type(SPN()):
            leaf_11 = NLeafNode(scope=[0], context=context)
            leaf_12 = NLeafNode(scope=[0], context=context)
            leaf_21 = NLeafNode(scope=[1], context=context)
            leaf_22 = NLeafNode(scope=[1], context=context)
            sum_11 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.3, 0.7]))
            sum_12 = NSumNode(children=[leaf_11, leaf_12], scope=[0], weights=np.array([0.3, 0.7]))
            sum_21 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.3, 0.7]))
            sum_22 = NSumNode(children=[leaf_21, leaf_22], scope=[1], weights=np.array([0.3, 0.7]))
            prod_11 = NProductNode(children=[sum_11, sum_21], scope=[0, 1])
            prod_12 = NProductNode(children=[sum_11, sum_22], scope=[0, 1])
            prod_13 = NProductNode(children=[sum_12, sum_21], scope=[0, 1])
            prod_14 = NProductNode(children=[sum_12, sum_22], scope=[0, 1])
            spn_nodes = NSumNode(
                children=[prod_11, prod_12, prod_13, prod_14],
                scope=[0, 1],
                weights=np.array([0.1, 0.2, 0.3, 0.4]),
            )

            _isvalid_spn(spn_nodes)

            result_nodes_ll = log_likelihood(spn_nodes, np.array([np.nan, 1.0]).reshape(-1, 2))
            result_nodes_l = likelihood(spn_nodes, np.array([np.nan, 1.0]).reshape(-1, 2))

            sum_layer_1 = SumLayer(
                num_of_nodes=2,
                scope=[0, 1],
                weights=np.array([0.3, 0.7]),
                children=[leaf_11, leaf_12, leaf_21, leaf_22],
            )
            sum_layer_2 = SumLayer(
                num_of_nodes=2,
                scope=[0, 1],
                weights=np.array([0.3, 0.7]),
                children=[leaf_11, leaf_12, leaf_21, leaf_22],
            )

            prod_11_sl = NProductNode(children=[sum_layer_1], scope=[0, 1])
            prod_12_sl = NProductNode(children=[sum_layer_1], scope=[0, 1])
            prod_13_sl = NProductNode(children=[sum_layer_2], scope=[0, 1])
            prod_14_sl = NProductNode(children=[sum_layer_2], scope=[0, 1])
            spn_layer = NSumNode(
                children=[prod_11_sl, prod_12_sl, prod_13_sl, prod_14_sl],
                scope=[0, 1],
                weights=np.array([0.1, 0.2, 0.3, 0.4]),
            )

            _isvalid_spn(spn_layer)

            result_layer_ll = log_likelihood(spn_layer, np.array([np.nan, 1.0]).reshape(-1, 2))
            result_layer_l = likelihood(spn_layer, np.array([np.nan, 1.0]).reshape(-1, 2))

            self.assertAlmostEqual(result_layer_ll[0][0], result_nodes_ll[0][0])
            self.assertAlmostEqual(result_layer_l[0][0], result_nodes_l[0][0])

            result = _get_node_counts(spn_nodes)
            sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
            self.assertEqual(sum_nodes, 5)
            self.assertEqual(prod_nodes, 4)
            self.assertEqual(leaf_nodes, 4)

            result = _get_node_counts(spn_layer)
            sum_nodes, prod_nodes, leaf_nodes = result[0], result[1], result[2]
            self.assertEqual(sum_nodes, 5)
            self.assertEqual(prod_nodes, 4)
            self.assertEqual(leaf_nodes, 4)


if __name__ == "__main__":
    unittest.main()

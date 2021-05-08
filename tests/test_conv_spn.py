import tensorflow as tf
from parameterized import parameterized
from tensorflow import test as tftest

import libspn_keras as spnk
from tests.utils import get_discrete_data
from tests.utils import SUM_OPS

tf.config.experimental_run_functions_eagerly(True)


class TestContinuousSPN(tftest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = get_discrete_data().reshape((-1, 2, 2, 1))

    @parameterized.expand(SUM_OPS)
    def test_dense_prod_local_sum(self, sum_op):
        spnk.set_default_sum_op(sum_op)
        spn = tf.keras.Sequential(
            [
                spnk.layers.IndicatorLeaf(num_components=2, input_shape=[2, 2, 1]),
                spnk.layers.Local2DSum(num_sums=2),
                spnk.layers.Conv2DProduct(
                    strides=[1, 1], dilations=[1, 1], kernel_size=[2, 2]
                ),
                spnk.layers.SpatialToRegions(),
                spnk.layers.RootSum(),
            ]
        )
        self.assert_logsum_zero(spn(self.data))

    @parameterized.expand(SUM_OPS)
    def test_dense_prod_conv_sum(self, sum_op):
        spnk.set_default_sum_op(sum_op)
        spn = tf.keras.Sequential(
            [
                spnk.layers.IndicatorLeaf(num_components=2, input_shape=[2, 2, 1]),
                spnk.layers.Conv2DSum(num_sums=2),
                spnk.layers.Conv2DProduct(
                    strides=[1, 1], dilations=[1, 1], kernel_size=[2, 2]
                ),
                spnk.layers.SpatialToRegions(),
                spnk.layers.RootSum(),
            ]
        )
        self.assert_logsum_zero(spn(self.data))

    @parameterized.expand(SUM_OPS)
    def test_depthwise_prod_conv_sum(self, sum_op):
        spnk.set_default_sum_op(sum_op)
        spn = tf.keras.Sequential(
            [
                spnk.layers.IndicatorLeaf(num_components=2, input_shape=[2, 2, 1]),
                spnk.layers.Conv2DSum(num_sums=2),
                spnk.layers.Conv2DProduct(
                    strides=[1, 1], dilations=[1, 1], kernel_size=[2, 2], depthwise=True
                ),
                spnk.layers.SpatialToRegions(),
                spnk.layers.RootSum(),
            ]
        )
        self.assert_logsum_zero(spn(self.data))

    @parameterized.expand(SUM_OPS)
    def test_depthwise_prod_local_sum(self, sum_op):
        spnk.set_default_sum_op(sum_op)
        spn = tf.keras.Sequential(
            [
                spnk.layers.IndicatorLeaf(num_components=2, input_shape=[2, 2, 1]),
                spnk.layers.Local2DSum(num_sums=2),
                spnk.layers.Conv2DProduct(
                    strides=[1, 1], dilations=[1, 1], kernel_size=[2, 2], depthwise=True
                ),
                spnk.layers.SpatialToRegions(),
                spnk.layers.RootSum(),
            ]
        )
        self.assert_logsum_zero(spn(self.data))

    def assert_logsum_zero(self, x):
        self.assertAllClose(0.0, tf.reduce_logsumexp(x))

import tensorflow as tf
from tensorflow import test as tftest

from tests.utils import indicators, product0_out, product1_out, sum0_out, root_out, NUM_VARS, NUM_COMPONENTS, \
    get_discrete_data, get_discrete_model

tf.config.experimental_run_functions_eagerly(True)


class TestDiscreteSPN(tftest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.discrete_spn = get_discrete_model()
        cls.data = get_discrete_data()

    def test_partition_adds_up_to_one(self):
        log_values = self.discrete_spn(self.data)
        self.assertEqual(tf.reduce_logsumexp(log_values), 0.0)

    def _compute_until(self, x, name):
        for layer in self.discrete_spn.layers:
            x = layer(x)
            if layer.name == name:
                return tf.exp(x)

    def test_indicators(self):
        got = self._compute_until(self.data, "indicator_leaf")
        expected = indicators(self.data)
        self.assertAllClose(got, expected)

    def test_prod0(self):
        got = self._compute_until(self.data, "dense_product")
        expected = product0_out(indicators(self.data))
        self.assertAllClose(got, expected)

    def test_sum0(self):
        got = self._compute_until(self.data, "dense_sum")
        expected = sum0_out(product0_out(indicators(self.data)))
        self.assertAllClose(got, expected)

    def test_product1(self):
        got = self._compute_until(self.data, "dense_product_1")
        expected = product1_out(sum0_out(product0_out(indicators(self.data))))
        self.assertAllClose(got, expected)

    def test_root(self):
        got = tf.exp(self.discrete_spn(self.data))
        expected = root_out(product1_out(sum0_out(product0_out(indicators(self.data)))))
        self.assertAllClose(got, expected)


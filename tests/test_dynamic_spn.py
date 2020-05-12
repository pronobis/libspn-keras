import tensorflow as tf
from tensorflow import test as tftest

from tests.utils import NUM_VARS, get_discrete_data, get_dynamic_model

tf.config.experimental_run_functions_eagerly(True)


class TestDynamicSPN(tftest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dynamic_spn = get_dynamic_model()
        cls.data_1_steps = get_discrete_data(num_vars=1 * NUM_VARS).reshape((-1, 1, NUM_VARS))
        cls.data_2_steps = get_discrete_data(num_vars=2 * NUM_VARS).reshape((-1, 2, NUM_VARS))

    def test_partition_1_step_adds_up_to_one(self):
        log_values = self.dynamic_spn([self.data_1_steps, [1] * self.data_1_steps.shape[0]])
        self.assertEqual(tf.reduce_logsumexp(log_values), 0.0)

    def test_partition_2_step_adds_up_to_one(self):
        log_values = self.dynamic_spn([self.data_2_steps, [2] * self.data_2_steps.shape[0]])
        self.assertEqual(tf.reduce_logsumexp(log_values), 0.0)

    def test_partition_2_step_incomplete_does_not_add_up_to_one(self):
        log_values = self.dynamic_spn([self.data_2_steps[:-1], [2] * (self.data_2_steps.shape[0] - 1)])
        self.assertNotEqual(tf.reduce_logsumexp(log_values), 0.0)

    def test_pad_seqs(self):
        data_1_padded = tf.pad(self.data_1_steps, [[0, 0], [2, 0], [0, 0]])
        data_2_padded = tf.pad(self.data_2_steps, [[0, 0], [1, 0], [0, 0]])
        data_concat = tf.concat([data_1_padded, data_2_padded], axis=0)

        log_values_1_padded = self.dynamic_spn([data_1_padded, [1] * self.data_1_steps.shape[0]])
        log_values_2_padded = self.dynamic_spn([data_2_padded, [2] * self.data_2_steps.shape[0]])
        log_values_concat = self.dynamic_spn(
            [data_concat, [1] * self.data_1_steps.shape[0] + [2] * self.data_2_steps.shape[0]])

        self.assertEqual(tf.reduce_logsumexp(log_values_1_padded), 0.0)
        self.assertEqual(tf.reduce_logsumexp(log_values_2_padded), 0.0)
        self.assertAllClose(tf.exp(tf.reduce_logsumexp(log_values_concat)), 2.0)

from collections import namedtuple
import tensorflow as tf

_AccumulatorTuple = namedtuple(
    "AccumulatorTuple", ['first_order_moment_denom_accum', 'first_order_moment_num_accum', 'second_order_moment_denom_accum', 'second_order_moment_num_accum'])


class GenerativeLearningEM:

    def __init__(self, spn, online=True, reset_per_epoch=False, with_labels=False, with_sequence_lens=False):
        """
        Utility class for learning SPNs in generative settings. The inner loop does not apply to (x_i, y_i) pairs,
        but simply to x_i. Will use ``libspn_keras.optimizers.OnlineExpectationMaximization`` as the optimizer.

        Args:
            spn: An instance of ``tf.keras.Model`` representing the SPN to train
        """
        self._spn = spn
        self._trainable_variable_copies = [_copy_variable(v) for v in self._spn.trainable_variables]
        self._trainable_variables_initial_state = [_copy_variable(v) for v in self._spn.trainable_variables]
        self._online = online
        self._reset_per_epoch = reset_per_epoch
        self._with_labels = with_labels
        self._with_sequence_lens = with_sequence_lens

    @tf.function
    def _train_one_step(self, train_batch):
        """
        Trains one step for a ``keras.Model``

        Args:
            x: A batch of samples

        Returns:
            The log marginal likelihood
        """
        with tf.GradientTape() as tape:
            if self._with_labels:
                if self._with_sequence_lens:
                    x, seq_lens, labels = train_batch
                    log_likelihood = self._spn([x, seq_lens])
                else:
                    x, labels = train_batch
                    log_likelihood = self._spn(x)
            elif self._with_sequence_lens:
                x, seq_lens = train_batch
                log_likelihood = self._spn([x, seq_lens])
            else:
                x = train_batch[0]
                log_likelihood = self._spn(x)

        grads = tape.gradient(log_likelihood, self._spn.trainable_variables)

        vars_to_assign = self._spn.trainable_variables if self._online else self._trainable_variable_copies

        for v, g in zip(vars_to_assign, grads):
            v.assign(v + g)

        return log_likelihood

    def fit(self, train_data: tf.data.Dataset, epochs, steps_per_epoch=None):
        """
        Fits the parameters of the SPN

        Args:
            train_data: An instance of ``tf.data.Dataset`` from which we get batches of :math:`x_i`
            steps_per_epoch: Steps per epoch
        """
        for epoch in range(epochs):
            log_probability_x = 0.0
            samples = 0
            step = 0
            for train_batch in train_data:
                log_probability_x += tf.reduce_sum(self._train_one_step(train_batch))
                samples += tf.shape(train_batch[0])[0]
                step += 1
                if steps_per_epoch is not None and step == steps_per_epoch:
                    break
            if not self._online:
                for v, v_copy in zip(self._spn.trainable_variables, self._trainable_variable_copies):
                    v.assign(v_copy)

                if self._reset_per_epoch:
                    for v_initial, v_copy in zip(
                            self._trainable_variables_initial_state, self._trainable_variable_copies):
                        v_copy.assign(v_initial)

            log_probability_x /= tf.cast(samples, tf.float32)
            tf.print('Epoch', epoch, ': mean log(p(X)) =', log_probability_x)

    def evaluate(self, test_dataset):
        log_marginal_likelihood = 0.0
        samples = 0
        for test_batch in test_dataset:
            samples += tf.shape(test_batch[0])[0]
            log_marginal_likelihood += tf.reduce_sum(self._spn(test_batch))
        log_marginal_likelihood /= tf.cast(samples, tf.float32)
        tf.print("Eval: mean log(p(X)) =", log_marginal_likelihood)
        return log_marginal_likelihood


def _copy_variable(v):
    return tf.Variable(
        trainable=v.trainable, name=v.name.rstrip(':0123456789') + "_offline_em_copy", dtype=v.dtype, shape=v.shape,
        initial_value=tf.identity(v)
    )


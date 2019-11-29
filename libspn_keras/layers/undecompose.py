import tensorflow as tf
from tensorflow import keras


class Undecompose(keras.layers.Layer):

    def __init__(self):
        super(Undecompose, self).__init__()
        self._num_decomps = 1
        self._num_scopes = 1
        self._num_nodes = self._num_decomps_in = None

    def build(self, input_shape):
        num_scopes_in, num_decomps_in, _, nodes_in = input_shape
        self._num_nodes = num_decomps_in * nodes_in
        if num_scopes_in != 1:
            raise ValueError("Can only decompose when there is a single scope")

    def call(self, x):
        shape = [self._num_scopes, self._num_decomps, -1, self._num_nodes]
        return tf.reshape(tf.transpose(x, (0, 2, 1, 3)), shape)

    def compute_output_shape(self, input_shape):
        num_scopes_in, num_decomps_in, _, nodes_in = input_shape
        return [1, 1, None, nodes_in * num_decomps_in]


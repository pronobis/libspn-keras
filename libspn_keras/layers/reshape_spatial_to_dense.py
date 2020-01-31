from tensorflow import keras
import tensorflow as tf


class ReshapeSpatialToDense(keras.layers.Layer):

    def __init__(self, **kwargs):
        """
        Reshapes spatial SPN layer to a dense layer. The dense output has leading dimensions for
        scopes and decomps (which will be [1, 1]).

        Args:
            **kwargs: Keyword arguments to pass on the keras.Layer super class
        """

        super(ReshapeSpatialToDense, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reshape(inputs, [1, 1, -1, self._num_nodes_out])

    def build(self, input_shape):
        num_batch, num_cells_vertical, num_cells_horizontal, num_nodes = input_shape
        self._num_nodes_out = num_cells_vertical * num_cells_horizontal * num_nodes

    def compute_output_shape(self, input_shape):
        num_batch, num_cells_vertical, num_cells_horizontal, num_nodes = input_shape
        return [1, 1, num_batch, num_cells_vertical * num_cells_horizontal * num_nodes]
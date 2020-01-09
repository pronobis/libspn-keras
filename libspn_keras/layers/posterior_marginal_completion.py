from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K

from libspn_keras.dimension_permutation import infer_dimension_permutation, DimensionPermutation


class Gradient(keras.layers.Layer):

    def build(self, input_shape):
        _, leaf_shape, _ = input_shape
        if self.dimension_permutation == DimensionPermutation.AUTO:
            self.dimension_permutation = infer_dimension_permutation(leaf_shape)
        super(Gradient, self).build(input_shape)

    def call(self, inputs):
        y, x = inputs
        return tf.gradients(y, x)

    def compute_output_shape(self, input_shape):
        y_shape, x_shape = input_shape
        return y_shape
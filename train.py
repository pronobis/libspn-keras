import tensorflow as tf
from tensorflow import keras
from libspn_keras.layers.across_scope_outer_product import AcrossScopeOuterProduct
from libspn_keras.layers.root_sum import RootSum
from libspn_keras.layers.undecompose import Undecompose
from libspn_keras.models import build_sum_product_network
from libspn_keras.layers.decompose import Decompose
from libspn_keras.layers.normal_leaf import NormalLeaf
from libspn_keras.layers.parallel_scope_sum import ParallelScopeSum
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = (x_train - np.mean(x_train, axis=1, keepdims=True)) / \
          (np.std(x_train, axis=1, keepdims=True) + 1e-4)
x_test = (x_test - np.mean(x_test, axis=1, keepdims=True)) / \
          (np.std(x_test, axis=1, keepdims=True) + 1e-4)

num_vars = x_train.shape[1]

sum_product_stack = []

initializer = tf.initializers.TruncatedNormal(mean=0.5)

# The 'backbone' stack of alternating sums and products
for _ in range(int(np.floor(np.log2(num_vars)))):
    sum_product_stack.extend([
        AcrossScopeOuterProduct(num_factors=2),
        ParallelScopeSum(num_sums=4, logspace_accumulators=True, initializer=initializer),
    ])

# Add another layer for joining two scopes to one remaining, followed by a class-wise root layer
# which is then followed by undecomposing (combining decompositions) and finally followed
# by a root sum. In this case we return
sum_product_stack.extend([
    AcrossScopeOuterProduct(num_factors=2),
    ParallelScopeSum(num_sums=1, logspace_accumulators=True, initializer=initializer),
    Undecompose(),
    RootSum(logspace_accumulators=True, return_weighted_child_logits=True)
])

# Use helper function to build the actual SPN
model = build_sum_product_network(
    num_vars=num_vars,
    decomposer=Decompose(num_decomps=10),
    leaf=NormalLeaf(num_components=4),
    sum_product_stack=keras.models.Sequential(sum_product_stack),
)

# Important to use from_logits=True with the cross-entropy loss
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluate
model.evaluate(x_test,  y_test, verbose=2)

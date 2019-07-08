
# coding: utf-8

# # Generative learning for continuous MNIST data using randomly structured SPNs
# This notebook shows how to build a randomly structured SPN and train it with online hard EM on continuous MNIST data.
# 
# ### Setting up the imports and preparing the data
# We load the data from `tf.keras.datasets`. Preprocessing consists of flattening and scaling of the data.

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import libspn as spn
import tensorflow as tf
import numpy as np
from libspn.examples.utils.dataiterator import DataIterator
import matplotlib.pyplot as plt

# Load
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

def scale(x):
    return x / 255.

def flatten(x):
    return x.reshape(-1, np.prod(x.shape[1:]))

def preprocess(x, y):
    return scale(flatten(x)), np.expand_dims(y, axis=1)

# Preprocess
train_x, train_y = preprocess(train_x, train_y)
test_x, test_y = preprocess(test_x, test_y)


# ### Defining the hyperparameters
# Some hyperparameters for the SPN. 
# - `num_subsets` is used for the `DenseSPNGenerator`. This corresponds to the number of variable subsets joined by product nodes in the SPN.
# - `num_mixtures` is used for the `DenseSPNGenerator`. This corresponds to the number of sum nodes per scope.
# - `num_decomps` is used for the `DenseSPNGenerator`. This corresponds to the number of decompositions generated at each level of products from top-down.
# - `num_vars` corresponds to the number of input variables (the number of pixels in the case of MNIST).
# - `balanced` is used for the `DenseSPNGenerator`. If true, then the generated SPN will have balanced subsets and will consequently be a balanced tree.
# - `input_dist` is the input distribution (the first product/sum layer in the SPN). `spn.DenseSPNGenerator.InputDist.RAW` corresponds to raw indicators being joined (so first layer is a product layer). `spn.DenseSPNGenerator.InputDist.MIXTURE` would correspond to a sums on top of each indicator.
# - `num_leaf_components` is the number of contineuous components in the leaf distribution
# - `inference_type` determines the kind of forward inference where `spn.InferenceType.MARGINAL` corresponds to sum nodes marginalizing their inputs. `spn.InferenceType.MPE` would correspond to having max nodes instead.
# - `num_classes`, `batch_size` and `num_epochs` should be obvious:)

# In[ ]:


# Number of variable subsets that a product joins
num_subsets = 2
# Number of sums per scope
num_mixtures = 4
# Number of variables
num_vars = train_x.shape[1]
# Number of decompositions per product layer
num_decomps = 1
# Generate balanced subsets -> balanced tree
balanced = True
# Input distribution. Raw corresponds to first layer being product that 
# takes raw indicators
input_dist = spn.DenseSPNGenerator.InputDist.RAW
# Number of different values at leaf (binary here, so 2)
num_leaf_components = 2
# Initial value for path count accumulators
initial_accum_value = 0.1
# Inference type (can also be spn.InferenceType.MPE) where 
# sum nodes are turned into max nodes
inference_type = spn.InferenceType.MARGINAL

# Number of classes
num_classes = 10
batch_size = 16
num_epochs = 10


# ### Building the SPN
# Our SPN consists of Gaussian leafs, a dense SPN per class and a root node connecting the 10 class-wise sub-SPNs. We also add an indicator node to the root node to model the latent class variable. Finally, we generate `Weight` nodes for the full SPN by using `spn.generate_weights`.

# In[ ]:


# Reset graph
tf.reset_default_graph()

# Leaf nodes
normal_leafs = spn.NormalLeaf(
    trainable_scale=True,
    trainable_loc=True,
    num_components=num_leaf_components, 
    num_vars=num_vars)

print("Generating random structure")
# Generates densely connected random SPNs
dense_generator = spn.DenseSPNGenerator(
    node_type=spn.DenseSPNGenerator.NodeType.BLOCK,
    num_subsets=num_subsets, num_mixtures=num_mixtures, num_decomps=num_decomps, 
    balanced=balanced, input_dist=input_dist)

# Generate a dense SPN for each class
class_roots = [dense_generator.generate(normal_leafs) for _ in range(num_classes)]

# Connect sub-SPNs to a root
root = spn.Sum(*class_roots, name="RootSum")
root = spn.convert_to_layer_nodes(root)

# Add a IndicatorLeaf node to the root as a latent class variable
class_indicators = root.generate_latent_indicators()

# Generate the weights for the SPN rooted at `root`
spn.generate_weights(root)

print("SPN depth: {}".format(root.get_depth()))
print("Number of products layers: {}".format(root.get_num_nodes(node_type=spn.ProductsLayer)))
print("Number of sums layers: {}".format(root.get_num_nodes(node_type=spn.SumsLayer)))


# ### Defining the TensorFlow graph
# Now that we have defined the SPN graph we can declare the TensorFlow operations needed for training and evaluation. We use the `HardEMLearning` class to help us out. The `MPEState` class can be used to find the MPE state of any node in the graph. In this case we might be interested in generating images or finding the most likely class based on the evidence elsewhere. These correspond to finding the MPE state for `leaf_indicators` and `class_indicators` respectively.

# In[ ]:


# Op for initializing all weights
weight_init_op = spn.initialize_weights(root)
# Op for getting the log probability of the root
root_log_prob = root.get_log_value(inference_type=inference_type)

# Helper for constructing EM learning ops
em_learning = spn.HardEMLearning(
    initial_accum_value=initial_accum_value, root=root, value_inference_type=inference_type,
    sample_prob=0.2, sample_winner=True, use_unweighted=True)

# Accumulate counts and update weights
online_em_update_op = em_learning.accumulate_and_update_weights()

# Op for initializing accumulators
init_accumulators = em_learning.reset_accumulators()

# MPE state generator
mpe_state_generator = spn.MPEState()
# Generate MPE state ops for leaf indicator and class indicator
normal_leaf_mpe, class_indicator_mpe = mpe_state_generator.get_state(root, normal_leafs, class_indicators)


# ### Display TF Graph
# Only works with Chrome browser.

# In[ ]:


spn.display_tf_graph()


# ### Training the SPN
# Here we we train while monitoring the likelihood. Note that we train the SPN generatively, which means that it does not optimize for discriminating between digits. This is why we observe lower accuracies than when e.g. training a discriminative model such as an MLP with cross-entropy loss.

# In[ ]:


# Set up some convenient iterators
train_iterator = DataIterator([train_x, train_y], batch_size=batch_size)
test_iterator = DataIterator([test_x, test_y], batch_size=batch_size)

def fd(x, y):
    return {normal_leafs: x, class_indicators: y}


print("Starting training")
with tf.Session() as sess:
    # Initialize things
    sess.run([weight_init_op, tf.global_variables_initializer(), init_accumulators])
    
    # Do one run for test likelihoods
    log_likelihoods = []
    for batch_x, batch_y in test_iterator.iter_epoch("Testing"):
        batch_llh = sess.run(root_log_prob, fd(batch_x, batch_y))
        log_likelihoods.extend(batch_llh)
        test_iterator.display_progress(LLH="{:.2f}".format(np.mean(batch_llh)))
    mean_test_llh = np.mean(log_likelihoods)
    
    print("Before training test LLH = {:.2f}".format(mean_test_llh))                              
    for epoch in range(num_epochs):
        
        # Train
        log_likelihoods = []
        for batch_x, batch_y in train_iterator.iter_epoch("Training"):
            batch_llh, _ = sess.run(
                [root_log_prob, online_em_update_op], fd(batch_x, batch_y))
            log_likelihoods.extend(batch_llh)
            train_iterator.display_progress(LLH="{:.2f}".format(np.mean(batch_llh)))
        mean_train_llh = np.mean(log_likelihoods)
        
        # Test
        log_likelihoods, matches = [], []
        for batch_x, batch_y in test_iterator.iter_epoch("Testing"):
            batch_llh, batch_class_mpe = sess.run([root_log_prob, class_indicator_mpe], fd(batch_x, -np.ones_like(batch_y, dtype=int)))
            log_likelihoods.extend(batch_llh)
            matches.extend(np.equal(batch_class_mpe, batch_y))
            test_iterator.display_progress(LLH="{:.2f}".format(np.mean(batch_llh)))
        mean_test_llh = np.mean(log_likelihoods)
        mean_test_acc = np.mean(matches)
        
        # Report
        print("Epoch {}, train LLH = {:.2f}, test LLH = {:.2f}, test accuracy = {:.2f}".format(
            epoch, mean_train_llh, mean_test_llh, mean_test_acc))
    
    # Compute MPE state of all digits
    per_class_mpe = sess.run(
        normal_leaf_mpe, 
        fd(
            -np.ones([num_classes, num_vars], dtype=int), 
            np.expand_dims(np.arange(num_classes, dtype=int), 1)
        )
    )
    


# ### Visualize MPE state per class
# We can visualize the MPE state computed at the end of the script above.

# In[ ]:


# for sample in per_class_mpe:
#     _, ax = plt.subplots()
#     ax.imshow(sample.reshape(28, 28).astype(float), cmap='gray')
#     plt.show()
#

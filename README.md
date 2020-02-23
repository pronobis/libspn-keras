# LibSPN Keras
LibSPN Keras is a library for constructing and training Sum-Product Networks. By leveraging the 
Keras framework with a TensorFlow backend, it offers both ease-of-use and scalability. Whereas the 
previously available `libspn` focused on scalability, `libspn-keras` offers scalability **and** 
a straightforward Keras-compatible interface.


## What are SPNs?

Sum-Product Networks (SPNs) are a probabilistic deep architecture with solid theoretical 
foundations, which demonstrated state-of-the-art performance in several domains. Yet, surprisingly, 
there are no mature, general-purpose SPN implementations that would serve as a platform for the 
community of machine learning researchers centered around SPNs. LibSPN Keras is a new 
general-purpose Python library, which aims to become such a platform. The library is designed to 
make it straightforward and effortless to apply various SPN architectures to large-scale datasets 
and problems. The library achieves scalability and efficiency, thanks to a tight coupling with 
TensorFlow and Keras, two frameworks already in use by a large community of researchers and 
developers in multiple domains.

## Dependencies
Currently, LibSPN Keras is tested with `tensorflow==2.0` and `tensorflow-probability==0.8.0`.

## Installation

```
pip install libspn-keras
```

## Note on stability of the repo
Currently, the repo is in an alpha state. Hence, one can expect some sporadic breaking changes.

## Feature Overview
- Gradient based training for generative and discriminative problems
- Hard EM training for generative problems
- Hard EM training with unweighted weights for generative problems
- Soft EM training (experimental) for generative problems
- [Deep Generalized Convolutional Sum-Product Networks](https://arxiv.org/abs/1902.06155)
- SPNs with arbitrary decompositions
- Fully compatible with Keras and TensorFlow 2.0
- Input dropout
- Sum child dropout
- Image completion
- Model saving
- Discrete inputs through an `IndicatorLeaf` node
- Continuous inputs through `NormalLeaf`, `CauchyLeaf` or `LaplaceLeaf`. All of those support both 
univariate as well as *multivariate* inputs.

## Examples
1. [A Deep Generalized Convolutional Sum-Product Network (DGC-SPN) with `libspn-keras` in Colab for
 **image classification**](https://colab.research.google.com/drive/10AXL7oo8LBCTnw7NrJ_zTph9X7J8XRdj)
2. [A Deep Generalized Convolutional Sum-Product Network (DGC-SPN) with `libspn-keras` in Colab for 
**image completion**.](https://colab.research.google.com/drive/1S3JdntlAGYE16QhAKltNYPgyMV8jW4Nv)
3. More to come, and if you would like to see a tutorial on anything in particular 
please raise an issue!

To get a sense of the simplicity, check out the way we can build complex DGC-SPNs in a layer-wise 
fashion:
```python
import libspn_keras as spn
from libspn_keras import layers, initializers
from tensorflow import keras

sum_kwargs = dict(
    accumulator_initializer=keras.initializers.TruncatedNormal(
        stddev=0.5, mean=1.0),
    logspace_accumulators=True
)

sum_product_network = keras.Sequential([
  layers.NormalLeaf(
      input_shape=(28, 28, 1),
      num_components=16, 
      location_trainable=True,
      location_initializer=keras.initializers.TruncatedNormal(
          stddev=1.0, mean=0.0)
  ),
  # Non-overlapping products
  layers.ConvProduct(
      depthwise=True, 
      strides=[2, 2], 
      dilations=[1, 1], 
      kernel_size=[2, 2],
      padding='valid'
  ),
  layers.SpatialLocalSum(num_sums=16, **sum_kwargs),
  # Non-overlapping products
  layers.ConvProduct(
      depthwise=True, 
      strides=[2, 2], 
      dilations=[1, 1], 
      kernel_size=[2, 2],
      padding='valid'
  ),
  layers.SpatialLocalSum(num_sums=32, **sum_kwargs),
  # Overlapping products, starting at dilations [1, 1]
  layers.ConvProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[1, 1], 
      kernel_size=[2, 2],
      padding='full'
  ),
  layers.SpatialLocalSum(num_sums=32, **sum_kwargs),
  # Overlapping products, with dilations [2, 2] and full padding
  layers.ConvProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[2, 2], 
      kernel_size=[2, 2],
      padding='full'
  ),
  layers.SpatialLocalSum(num_sums=64, **sum_kwargs),
  # Overlapping products, with dilations [4, 4] and full padding
  layers.ConvProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[4, 4], 
      kernel_size=[2, 2],
      padding='full'
  ),
  layers.SpatialLocalSum(num_sums=64, **sum_kwargs),
  # Overlapping products, with dilations [8, 8] and 'final' padding to combine 
  # all scopes
  layers.ConvProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[8, 8], 
      kernel_size=[2, 2],
      padding='final'
  ),
  layers.ReshapeSpatialToDense(),
  # Class roots
  layers.DenseSum(num_sums=10, **sum_kwargs),
  layers.RootSum(
      return_weighted_child_logits=True, 
      logspace_accumulators=True, 
      accumulator_initializer=keras.initializers.TruncatedNormal(
          stddev=0.0, mean=1.0)
  )
])
sum_product_network.summary()
```

Which produces:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
normal_leaf (NormalLeaf)     (None, 28, 28, 16)        25088     
_________________________________________________________________
conv_product (ConvProduct)   (None, 14, 14, 16)        4         
_________________________________________________________________
spatial_local_sum (SpatialLo (None, 14, 14, 16)        50176     
_________________________________________________________________
conv_product_1 (ConvProduct) (None, 7, 7, 16)          4         
_________________________________________________________________
spatial_local_sum_1 (Spatial (None, 7, 7, 32)          25088     
_________________________________________________________________
conv_product_2 (ConvProduct) (None, 8, 8, 32)          4         
_________________________________________________________________
spatial_local_sum_2 (Spatial (None, 8, 8, 32)          65536     
_________________________________________________________________
conv_product_3 (ConvProduct) (None, 10, 10, 32)        4         
_________________________________________________________________
spatial_local_sum_3 (Spatial (None, 10, 10, 64)        204800    
_________________________________________________________________
conv_product_4 (ConvProduct) (None, 14, 14, 64)        4         
_________________________________________________________________
spatial_local_sum_4 (Spatial (None, 14, 14, 64)        802816    
_________________________________________________________________
conv_product_5 (ConvProduct) (None, 8, 8, 64)          4         
_________________________________________________________________
reshape_spatial_to_dense (Re (1, 1, None, 4096)        0         
_________________________________________________________________
dense_sum (DenseSum)         (1, 1, None, 10)          40960     
_________________________________________________________________
root_sum (RootSum)           (None, 10)                10        
=================================================================
Total params: 1,214,498
Trainable params: 1,201,930
Non-trainable params: 12,568
_________________________________________________________________
```

## TODOs
- Create docs
- Structure learning
- Advanced regularization e.g. pruning or auxiliary losses on weight accumulators


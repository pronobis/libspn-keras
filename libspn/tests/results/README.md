Performance Test Results
========================

gather_cols
-----------

### Overview

- gather_cols-1-with_gpu_bounds_check - the GPU kernel of custom ops still had bounds check enabled
- gather_cols-2-without_gpu_bounds_check - the GPU kernel now does not have the bounds check
- gather_cols-3-with_compute_capabilities - the version without bounds check is teseted after we added compilation specifically for compute capabilities  ["3.5", "5.2", "6.1"] to setup.py to fix the error on GTX1050.
- gather_cols-4-new_params - version with new parameters of the kernels

### General Observations

- Gather, gather_nd, and custom ops suffer from performance loss for int32 indices. Slicing does not, which might be related to the fact that no index arithmetic happens in case of slice.
- GTX1050 and GTX1080 offer much better GPU performance than Titan X (Pascal) or Tesla P100 on DGX1 (the last one offers good performance for int64 indices)
- i7-7700HQ seems to offer best CPU performance
- adding validate_indices=False to tf.gather() does not affect performance


### 1D 1index

#### CPU

- all methods equivalent run time, with slice having more ops and setup time

#### GPU

- int32 much slower than int64 (particularly for dgx1) for gather and custom, but not slice
- for int64, all methods offer similar performance, with gather and slice being slightly faster
- for int32, slice wins, with gather and custom being comparable to each other, but much slower
- slice makes more ops and setup time

#### Result

- use gather, unless custom is improved a bit for the GPU
- make sure to use int64 indices, otherwise slice might be faster despite generating larger graph

### 1D passthrough

#### CPU

- noop is at least 2x faster than any op and results in smaller graph

#### GPU

- noop is at least 2x faster than any op and results in smaller graph

#### Result

- use noop

### 1D other

#### CPU

- gather and custom offer equivalent performance, with a slight benefit of using gather, unless some possibility of optimization exists in which case, custom might be slightly more optimal
- gather and custom generate graphs of the same size

#### GPU

- gather and custom offer almost equivalent performance, with a small benefit of using gather
- using int64 indices is much faster than int32
- gather and custom generate graphs of the same size

#### Result

- use gather, unless custom is improved a bit for the GPU
- make sure to use int64 indices, otherwise slice might be faster (although generates larger graph)

### 2D 1index

#### CPU

- custom and slice perform the same, much better than gather_nd
- slice and gather_nd build larger graphs than custom

#### GPU

- int32 much slower than int64 (particularly for dgx1) for gather_nd and custom, but not slice
- custom is always much faster than gather_nd and build smaller graphs
- custom is roughly equivalent to slice for int64 indices, but can be 2x slower for int32 indices
- slice makes more ops and setup time

#### Result

- use custom
- make sure to use int64 indices, otherwise slice might be faster despite generating larger graph

### 2D passthrough

#### CPU

- noop is several time faster than any op and results in smaller graph

#### GPU

- noop is several time faster than any op and results in smaller graph

#### Result

- use noop


### 2D other

#### CPU

- custom offers much better performance and smaller graph than gather_nd

#### GPU

- custom offers much better performance and smaller graph than gather_nd
- using int64 indices is much faster than int32

#### Result

- use custom
- make sure to use int64 indices



scatter_cols
------------

### Overview

- scatter_cols-1-with_gpu_bounds_check - original version of scatter_cols making internal gpu/cpu data copying and bounds check. We compile specifically for compute capabilities  ["3.5", "5.2", "6.1"] to fix the error on GTX1050.
- scatter_cols-2-without_gpu_bounds_check - improved version without bounds check and with cleaner parameters of the kernels


### General Observations

- the custom op is very inefficient and shold not be used until fixed

- Gather, gather_nd, and custom ops suffer from performance loss for int32 indices. Slicing does not, which might be related to the fact that no index arithmetic happens in case of slice.
- GTX1050 and GTX1080 offer much better GPU performance than Titan X (Pascal) or Tesla P100 on DGX1 (the last one offers good performance for int64 indices)
- i7-7700HQ seems to offer best CPU performance


### 1D 1index

#### CPU

- pad offers the better performance
- pad offers smallest graph

#### GPU

- pad offers clearly better performance
- pad offers smallest graph

#### Result

- used pad


### 1D passthrough

#### CPU

- noop is much faster than any op and results in smaller graph

#### GPU

- noop is much faster than any op and results in smaller graph

#### Result

- use noop


### 1D other

#### CPU

- gather_1d is more efficient than custom
- custom offers smaller graph

#### GPU

- custom is more efficient than gather_1d
- custom offers smaller graph


#### Result

- use custom, since gpu performance is more important


### 2D 1index

#### CPU

- custom is fastest
- pad offers slightly smaller graph and is 2nd fastest


#### GPU

- pad offers the smallest graph and is clearly the fastest


#### Result

- use pad, since graph size and gpu performance matters


### 2D passthrough

#### CPU

- noop is much faster than any op and results in smaller graph

#### GPU

- noop is much faster than any op and results in smaller graph

#### Result

- use noop


### 2D other

#### CPU

- custom is fastest and offers smallest graph

#### GPU

- custom is fastest and offers smallest graph

#### Result

- use custom

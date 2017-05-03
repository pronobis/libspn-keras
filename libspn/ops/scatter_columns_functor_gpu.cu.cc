#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "scatter_columns_functor_gpu.cu.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPECS_INDEX(T, IndT) \
  template struct functor::ScatterColumnsFunctor<GPUDevice, T, IndT>

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

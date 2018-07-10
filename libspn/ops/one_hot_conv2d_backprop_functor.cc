#if GOOGLE_CUDA

#include "one_hot_conv2d_backprop_functor.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow
{
namespace functor
{
//--Forward declarations of the functor specializations for GPU--//
#define DECLARE_GPU_SPECS_INDEX(T, IndT)                                     \
  template <>                                                                \
  Status OneHotConv2DBackpropFunctor<GPUDevice, T, IndT>::operator()(\
    const GPUDevice& d,\
    const typename TTypes<T, 4>::ConstTensor out_grad,\
    const typename TTypes<IndT, 3>::ConstTensor filter,\
    const int64 dilation_rows, const int64 dilation_cols,\
    const int64 stride_rows, const int64 stride_cols,\
    typename TTypes<T, 4>::Tensor in_grad, bool overlapping_patches);                                   \
  extern template struct OneHotConv2DBackpropFunctor<GPUDevice, T, IndT>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace tensorflow

#else

//#include "tensorflow/core/user_ops/one_hot_conv2d_backprop_functor.h"

#endif  // GOOGLE_CUDA

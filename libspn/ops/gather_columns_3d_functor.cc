#if GOOGLE_CUDA

#include "gather_columns_3d_functor.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow
{
namespace functor
{
//--Forward declarations of the functor specializations for GPU--//
#define DECLARE_GPU_SPECS_INDEX(T, IndT)                                   \
  template <>                                                              \
  Status GatherColumns3dFunctor<GPUDevice, T, IndT>::operator()(           \
      const GPUDevice& dvc, const typename TTypes<T>::ConstMatrix& params, \
      const typename TTypes<IndT>::ConstMatrix& indices,                   \
      typename TTypes<T>::Matrix& output,                                  \
      const bool& padding,                                                 \
      const T pad_elem);                                                   \
  extern template struct GatherColumns3dFunctor<GPUDevice, T, IndT>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace tensorflow

#else

#include "tensorflow/core/user_ops/gather_columns_3d_functor.h"

#endif  // GOOGLE_CUDA

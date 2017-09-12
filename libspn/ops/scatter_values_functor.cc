#if GOOGLE_CUDA

#include "scatter_values_functor.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow
{
namespace functor
{
//--Forward declarations of the functor specializations for GPU--//
#define DECLARE_GPU_SPECS_INDEX(T, IndT)                                     \
  template <>                                                                \
  Status ScatterValuesFunctor<GPUDevice, T, IndT>::operator()(               \
      const GPUDevice& dvc, const typename TTypes<T>::ConstMatrix& params,   \
      const typename TTypes<IndT>::ConstMatrix& indices,                     \
      const IndT& num_out_cols,                                              \
      typename TTypes<T>::Matrix& output);                                   \
  extern template struct ScatterValuesFunctor<GPUDevice, T, IndT>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace tensorflow

#else

#include "tensorflow/core/user_ops/scatter_values_functor.h"

#endif  // GOOGLE_CUDA

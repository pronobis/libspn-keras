#if GOOGLE_CUDA

#include "scatter_columns_functor.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow
{
namespace functor
{
//--Forward declarations of the functor specializations for GPU--//
#define DECLARE_GPU_SPECS_INDEX(T, IndT)                                     \
  template <>                                                                \
  Status ScatterColumnsFunctor<GPUDevice, T, IndT>::operator()(              \
      const GPUDevice& dvc, const typename TTypes<T>::ConstMatrix& params,   \
      const typename TTypes<IndT>::ConstFlat& indices,                       \
      const IndT& num_out_cols, const T* pad_elem, const int64& params_rows, \
      const int64& params_cols, typename TTypes<T>::Matrix& output);         \
  extern template struct ScatterColumnsFunctor<GPUDevice, T, IndT>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace tensorflow

#else

#include "tensorflow/core/user_ops/scatter_columns_functor.h"

#endif  // GOOGLE_CUDA

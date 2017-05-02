#if GOOGLE_CUDA

#include "gather_columns_functor.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow
{
namespace functor
{
//--Forward declarations of the functor specializations for GPU--//
#define DECLARE_GPU_SPECS_INDEX(T, IndT)                            \
  template <>                                                       \
  int64 GatherColumnsFunctor<GPUDevice, T, IndT>::operator()(       \
      const GPUDevice& dvc, typename TTypes<T>::ConstMatrix params, \
      typename TTypes<IndT>::ConstFlat indices, int64 params_rows,  \
      int64 params_cols, typename TTypes<T>::Matrix out);           \
  extern template struct GatherColumnsFunctor<GPUDevice, T, IndT>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace tensorflow

#else

#include "tensorflow/core/user_ops/gather_columns_functor.h"

#endif  // GOOGLE_CUDA

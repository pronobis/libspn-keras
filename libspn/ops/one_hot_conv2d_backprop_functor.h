#ifndef TENSORFLOW_USEROPS_ONE_HOT_CONV2DBACKPROP_FUNCTOR_H_
#define TENSORFLOW_USEROPS_ONE_HOT_CONV2DBACKPROP_FUNCTOR_H_

#include <unordered_set>
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

using namespace std;

namespace tensorflow
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor
{

template <typename Device, typename T, typename IndT>
struct OneHotConv2DBackpropFunctor
{
  Status operator()(const GPUDevice& d,
                    const typename TTypes<T, 4>::ConstTensor out_grad,
                    const typename TTypes<IndT, 3>::ConstTensor filter,
                    const int64 dilation_rows, const int64 dilation_cols,
                    const int64 stride_rows, const int64 stride_cols,
                    typename TTypes<T, 4>::Tensor in_grad, bool overlapping_patches);
};

//template <typename T, typename IndT>
//struct OneHotConv2DFunctor<CPUDevice, T, IndT>
//{
//  Status operator()(const CPUDevice& dvc,
//                    const typename TTypes<T>::ConstMatrix& params,
//                    const typename TTypes<IndT>::ConstMatrix& indices,
//                    const IndT& num_out_cols,
//                    typename TTypes<T>::Matrix& output)
//  {
//    return OneHotConv2DFunctorCPU<T, IndT>()(params, indices,
//                                               num_out_cols, output);
//  }
//};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_ONE_HOT_CONV2DBACKPROP_FUNCTOR_H_

#ifndef TENSORFLOW_USEROPS_SCATTER_VALUES_FUNCTOR_H_
#define TENSORFLOW_USEROPS_SCATTER_VALUES_FUNCTOR_H_

#include <unordered_set>
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

using namespace std;

namespace tensorflow
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor
{
//--Helper method for copying using memcpy()--//
template <typename T, typename IndT>
Status PadAndCopy(const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output)
{
  const int64 params_rows = params.dimension(0);
  const int64 params_cols = params.dimension(1);

  //--Declare padding element as a double, and set it to the default value 0.0--//
  double pad_elem = 0.0;

  //--Debugging flag disabled by default--//
  #if EXEC_TIME_CALC
    clock_t start, end;
    float time_taken;
    start = clock();
  #endif  // EXEC_TIME_CALC

  //--Mem-copy padding element to output tensor--//
  memset(&output(0, 0), pad_elem, ((params_rows * params_cols * num_out_cols) * sizeof(T)));

  for (int row = 0; row < params_rows; row++)
  {
    for (int col = 0, col_next = 1; col < params_cols; col++, col_next++)
    {
      //--Check indices[r][c] ∈ (0, num_out_cols]--//
      if (!FastBoundsCheck(indices(row, col), num_out_cols))
      {
        return errors::InvalidArgument("Indices(", row, ", ", col, "): ", indices(row, col),
                                       " is not in range (0, ", num_out_cols, "].");
      }

      //--If not the final copy--//
      if (col_next < params_cols)
      {
        //--Prefetch the next source (params_matrix) and destination
        //  (output_matrix) memory addresses--//
        port::prefetch<port::PREFETCH_HINT_T0>(
            &output(((row * params_cols) + col_next), indices(row, col_next)));
        port::prefetch<port::PREFETCH_HINT_T0>(
            &params(row, col_next));
      }

      //--Mem-copy a single element from params tensor--//
      memcpy(&output(((row * params_cols) + col), indices(row, col)), &params(row, col), sizeof(T));
    }
  }

  //--Debugging flag disabled by default--//
  #if EXEC_TIME_CALC
    end = clock();
    time_taken =
        (((float)(end - start)) / CLOCKS_PER_SEC) * 1000.0;  //--Milliseconds//
    std::cout << "CPU - Time Taken: " << time_taken << " ms" << endl;
  #endif  // EXEC_TIME_CALC

  return Status::OK();
}

template <typename T, typename IndT>
struct ScatterValuesFunctorCPU
{
  Status operator()(const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output)
  {
    return PadAndCopy<T, IndT>(params, indices,
                                 num_out_cols, output);
  }
};

template <typename Device, typename T, typename IndT>
struct ScatterValuesFunctor
{
  Status operator()(const Device& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output);
};

template <typename T, typename IndT>
struct ScatterValuesFunctor<CPUDevice, T, IndT>
{
  Status operator()(const CPUDevice& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output)
  {
    return ScatterValuesFunctorCPU<T, IndT>()(params, indices,
                                               num_out_cols, output);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_SCATTER_VALUES_FUNCTOR_H_

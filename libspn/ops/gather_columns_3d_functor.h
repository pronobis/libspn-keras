#ifndef TENSORFLOW_USEROPS_GATHER_COLUMNS_3D_FUNCTOR_H_
#define TENSORFLOW_USEROPS_GATHER_COLUMNS_3D_FUNCTOR_H_

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
//--Helper method for copying using memcpy()--//
template <typename T, typename IndT>
Status CopyAndPad(const typename TTypes<T>::ConstMatrix& params,
                  const typename TTypes<IndT>::ConstMatrix& indices,
                  typename TTypes<T>::Matrix& output,
                  const bool& padding)
{
  //--Get rows and cols size of params and indices--//
  const int64 params_rows = params.dimension(0);
  const int64 params_cols = params.dimension(1);
  const int64 indices_rows = indices.dimension(0);
  const int64 indices_cols = indices.dimension(1);

  //--Debugging flag disabled by default--//
  #if EXEC_TIME_CALC
    clock_t start, end;
    float time_taken;
    start = clock();
  #endif  // EXEC_TIME_CALC

  int current_row = 0;

  if(padding)
  {
    //--Declare padding element as a double, and set it to 0.0--//
    double pad_elem = 0.0;

    //--Since output has atlest one padded column,
    //  mem-copy padding element to output tensor--//
    memset(&output(0, 0), pad_elem, ((params_rows * indices_rows * indices_cols) * sizeof(T)));

    for (int row = 0; row < indices_rows; row++)
    {
      for (int slice = 0; slice < params_rows; slice++)
      {
        current_row = (slice * indices_rows) + row;
        for (int col = 0, col_next = 1; col < indices_cols; col++, col_next++)
        {
          //--Check if indices[r][c] ∈ (0, num_out_cols]--//
          if (!FastBoundsCheck(indices(row, col), params_cols))
          {
            //--If so - indicating a padded column - ignore rest of the
            //  current row by breaking--//
            break;
          }

          //--If not the final copy--//
          if (col_next < params_cols)
          {
            //--Prefetch the next source (params_matrix) and destination
            //  (output_matrix) memory addresses--//
            port::prefetch<port::PREFETCH_HINT_T0>(
                  &output(current_row, col_next));
            port::prefetch<port::PREFETCH_HINT_T0>(
                  &params(slice, indices(row, col_next)));
          }
          //--Mem-copy a single element from params tensor--//
          memcpy(&output(current_row, col), &params(slice, indices(row, col)), sizeof(T));
        }
      }
    }
  }
  else //--No Padding needed--//
  {
    for (int row = 0; row < indices_rows; row++)
    {
      for (int slice = 0; slice < params_rows; slice++)
      {
        current_row = (slice * indices_rows) + row;
        for (int col = 0, col_next = 1; col < indices_cols; col++, col_next++)
        {
          //--Check indices[r][c] ∈ (0, num_out_cols]--//
          if (!FastBoundsCheck(indices(row, col), params_cols))
          {
            return errors::InvalidArgument("Indices(", row, ", ", col, "): ", indices(row, col),
                                           " is not in range (0, ", params_cols, "].");
          }

          //--If not the final copy--//
          if (col_next < params_cols)
          {
            //--Prefetch the next source (params_matrix) and destination
            //  (output_matrix) memory addresses--//
            port::prefetch<port::PREFETCH_HINT_T0>(
                  &output(current_row, col_next));
            port::prefetch<port::PREFETCH_HINT_T0>(
                  &params(slice, indices(row, col_next)));
          }
          //--Mem-copy a single element from params tensor--//
          memcpy(&output(current_row, col), &params(slice, indices(row, col)), sizeof(T));
        }
      }
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
struct GatherColumns3dFunctorCPU
{
  Status operator()(const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    typename TTypes<T>::Matrix& output,
                    const bool& padding)
  {
    return CopyAndPad<T, IndT>(params, indices,
                               output, padding);
  }
};

template <typename Device, typename T, typename IndT>
struct GatherColumns3dFunctor
{
  Status operator()(const Device& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    typename TTypes<T>::Matrix& output,
                    const bool& padding);
};

template <typename T, typename IndT>
struct GatherColumns3dFunctor<CPUDevice, T, IndT>
{
  Status operator()(const CPUDevice& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    typename TTypes<T>::Matrix& output,
                    const bool& padding)
  {
    return GatherColumns3dFunctorCPU<T, IndT>()(params, indices,
                                                         output, padding);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_GATHER_COLUMNS_3D_FUNCTOR_H_

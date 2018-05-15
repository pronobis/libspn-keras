#ifndef TENSORFLOW_USEROPS_GATHER_COLUMNS_FUNCTOR_H_
#define TENSORFLOW_USEROPS_GATHER_COLUMNS_FUNCTOR_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

using namespace std;

namespace tensorflow
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor
{
//--Helper method to copy using memcpy()--//
template <typename T, typename IndT>
IndT CountAndCopy(typename TTypes<T>::ConstMatrix params,
                  typename TTypes<IndT>::ConstFlat indices,
                  typename TTypes<T>::Matrix output)
{
  const int64 params_rows = params.dimension(0);
  const int64 params_cols = params.dimension(1);

  if (params_cols == 1 && indices.size() == 1)
  {
    if (indices(0) != 0)
    {
      return 0;
    }
    else
    {
      //--Single column tensor, indices must include it, so just copy params
      // tensor to output tensor and return--//
      memcpy(&output(0, 0), &params(0, 0), (params_rows * sizeof(T)));
      return -1;
    }
  }

  const int64 indices_size = indices.dimension(0);
  std::vector<int> cons_cols_counter(indices_size, 1);

  if (indices_size > 1)
  {
    //--Group consecutive columns together--//
    //--E.g.:     params_size = 10
    //--          indices = [7, 8, 9, 2, 0, 4, 5, 3, 1, 5, 6, 7]
    //--cons_cols_counter = [3, 2, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1]

    int cols;
    for (int c = 0; c < indices_size; c++)
    {
      //--Check indices[i] ∈ (0, params_cols]--//
      if (!FastBoundsCheck(indices(c), params_cols))
      {
        return c;
      }

      cols = 1;
      while (c + cols < indices_size)
      {
        if (indices(c) + cols == indices(c + cols))
          cols++;
        else
          break;
      }

      while (cols > 1)
      {
        cons_cols_counter[c++] = cols--;
      }
    }
  }
  else  //--indices_size == 1--//
  {
    if (!FastBoundsCheck(indices(0), params_cols))
      return 0;
    else
      cons_cols_counter[0] = 1;
  }

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
  clock_t start, end;
  float time_taken;
  start = clock();
#endif  // EXEC_TIME_CALC

  //--Mem-copy columns, bunching consecutive columns together, one row at a
  // time--//
  for (int row = 0; row < params_rows; row++)
  {
    for (int col = 0; col < indices_size;)
    {
      //--If not final iteration--//
      if (col + 1 < indices_size)
      {
        //--Prefetch the next source (params_matrix) and destination
        //(output_matrix) memory addresses--//
        port::prefetch<port::PREFETCH_HINT_T0>(
            &output(row, col + cons_cols_counter[col]));
        port::prefetch<port::PREFETCH_HINT_T0>(
            &params(row, indices(col + cons_cols_counter[col])));
      }

      //--Mem-copy column(s)--//
      memcpy(&output(row, col), &params(row, indices(col)),
             (cons_cols_counter[col] * sizeof(T)));
      col += cons_cols_counter[col];
    }
  }

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
  end = clock();
  time_taken =
      (((float)(end - start)) / CLOCKS_PER_SEC) * 1000.0;  //--Milliseconds//
  std::cout << "CPU - Time Taken: " << time_taken << " ms" << endl;
#endif  // EXEC_TIME_CALC

  return -1;
}

template <typename T, typename IndT>
struct GatherColumnsFunctorCPU
{
  int64 operator()(typename TTypes<T>::ConstMatrix params,
                   typename TTypes<IndT>::ConstFlat indices,
                   typename TTypes<T>::Matrix output)
  {
    return CountAndCopy<T, IndT>(params, indices, output);
  }
};

template <typename Device, typename T, typename IndT>
struct GatherColumnsFunctor
{
  int64 operator()(const Device& dvc, typename TTypes<T>::ConstMatrix params,
                   typename TTypes<IndT>::ConstFlat indices,
                   typename TTypes<T>::Matrix output);
};

template <typename T, typename IndT>
struct GatherColumnsFunctor<CPUDevice, T, IndT>
{
  int64 operator()(const CPUDevice& dvc, typename TTypes<T>::ConstMatrix params,
                   typename TTypes<IndT>::ConstFlat indices, typename TTypes<T>::Matrix output)
  {
    return GatherColumnsFunctorCPU<T, IndT>()(params, indices, output);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_GATHER_COLUMNS_FUNCTOR_H_

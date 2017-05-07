#ifndef TENSORFLOW_USEROPS_SCATTER_COLUMNS_FUNCTOR_H_
#define TENSORFLOW_USEROPS_SCATTER_COLUMNS_FUNCTOR_H_

#include <unordered_set>
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

using namespace std;

namespace tensorflow
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor
{
//--Helper method to count and copy using memcpy()--//
template <typename T, typename IndT>
Status CountAndCopy(const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstFlat& indices,
                    const IndT& num_out_cols, const T* pad_elem,
                    const int64& params_rows, const int64& params_cols,
                    typename TTypes<T>::Matrix& output)
{
  const int64 indices_size = indices.dimension(0);

  unordered_set<IndT> unique_ind(&indices(0), &indices(indices_size));
  if (unique_ind.size() != indices_size)
  {
    return errors::InvalidArgument(
        "Indices cannot contain duplicates.", " Total no. of indices: ",
        indices_size, " != no. of unique indices: ", unique_ind.size(), ".");
  }

  std::vector<IndT> out_indices(
      num_out_cols, -1);  //--Here '-1' refers to padding column(s)--//

  //--Arrange output indices--//
  //--E.g.:  params = [11, 12, 13, 14]
  //-- num_out_cols = 10
  //--     pad_elem = 0
  //--      indices = [7, 4, 2, 3]
  //--       output = [0, 0, 13, 14, 12, 0, 0, 11, 0, 0]
  //--  out_indices = [-1, -1, 2, 3, 1, -1, -1, 0, -1, -1]
  for (IndT i = 0; i < indices_size; i++)
  {
    //--Check indices[i] ∈ (0, num_out_cols]--//
    if (!FastBoundsCheck(indices(i), num_out_cols))
    {
      return errors::InvalidArgument("Indices(", i, "): ", indices(i),
                                     " is not in range (0, ", num_out_cols,
                                     "].");
    }

    out_indices[indices(i)] = i;
  }

  //--Group consecutive padding columns together--//
  //--E.g.:  params = [11, 12, 13, 14]
  //-- num_out_cols = 10
  //--     pad_elem = 0
  //--      indices = [7, 4, 2, 3]
  //--       output = [0, 0, 13, 14, 12, 0, 0, 11, 0, 0]
  //--cons_pad_cols = [2, 1,  0,  0,  0, 2, 1,  0, 2, 1]

  std::vector<int> cons_pad_cols(num_out_cols, 0);
  int pad_cols;
  int max_cons_pad_cols = 0;

  for (int c = 0; c < num_out_cols; c++)
  {
    pad_cols = 0;
    while (out_indices[c + pad_cols] < 0)
    {
      pad_cols++;
      if (c + pad_cols >= num_out_cols)
      {
        break;
      }
    }

    if (pad_cols > max_cons_pad_cols)
    {
      max_cons_pad_cols = pad_cols;
    }

    while (pad_cols > 0)
    {
      cons_pad_cols[c++] = pad_cols--;
    }
  }

  //--Vector containing padding elements. Size of this vector = maximum no. of
  // consecutive padding columns in the output tensor--//
  gtl::InlinedVector<T, 4> pad_elem_vec(max_cons_pad_cols, pad_elem[0]);

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
  clock_t start, end;
  float time_taken;
  start = clock();
#endif  // EXEC_TIME_CALC

  //--Mem-copy columns, bunching consecutive padding columns together, one row
  // at a time--//
  for (int row = 0; row < params_rows; row++)
  {
    for (int col = 0; col < num_out_cols;)
    {
      //--If not the final copy--//
      if (col + 1 < num_out_cols)
      {
        //--Prefetch the next destination (output) memory address--//
        port::prefetch<port::PREFETCH_HINT_T0>(&output(row, (col + 1)));

        //--If the next column is not a padding column--//
        if (out_indices[col + 1] >= 0)
        {
          //--Prefetch the next source (params) memory address--//
          port::prefetch<port::PREFETCH_HINT_T0>(
              &params(row, out_indices[col + 1]));
        }
      }

      if (out_indices[col] >= 0)
      {
        //--Mem-copy a single non-padding element from params tensor--//
        memcpy(&output(row, col), &params(row, out_indices[col]), sizeof(T));
        ++col;
      }
      else
      {
        //--Mem-copy columns of padding elements (per row) from padding element
        // vector--//
        memcpy(&output(row, col), &pad_elem_vec[0],
               (cons_pad_cols[col] * sizeof(T)));
        col += cons_pad_cols[col];
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
struct ScatterColumnsFunctorCPU
{
  Status operator()(const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstFlat& indices,
                    const IndT& num_out_cols, const T* pad_elem,
                    const int64& params_rows, const int64& params_cols,
                    typename TTypes<T>::Matrix& output)
  {
    return CountAndCopy<T, IndT>(params, indices, num_out_cols, pad_elem,
                                 params_rows, params_cols, output);
  }
};

template <typename Device, typename T, typename IndT>
struct ScatterColumnsFunctor
{
  Status operator()(const Device& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstFlat& indices,
                    const IndT& num_out_cols, const T* pad_elem,
                    const int64& params_rows, const int64& params_cols,
                    typename TTypes<T>::Matrix& output);
};

template <typename T, typename IndT>
struct ScatterColumnsFunctor<CPUDevice, T, IndT>
{
  Status operator()(const CPUDevice& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstFlat& indices,
                    const IndT& num_out_cols, const T* pad_elem,
                    const int64& params_rows, const int64& params_cols,
                    typename TTypes<T>::Matrix& output)
  {
    return ScatterColumnsFunctorCPU<T, IndT>()(params, indices, num_out_cols,
                                               pad_elem, params_rows,
                                               params_cols, output);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_SCATTER_COLUMNS_FUNCTOR_H_

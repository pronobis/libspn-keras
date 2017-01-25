#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_COLUMNS_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_COLUMNS_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/user_ops/scatter_columns_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

  typedef Eigen::GpuDevice GPUDevice;

  template <typename T, typename IndT>
  __global__ void ScatterColumnsOpKernel(const T* params, const IndT* out_indices,
                                         const T* pad_elem, const IndT out_num_cols,
                                         const int64 params_cols, const int64 output_size,
                                         T* output)
  {
    CUDA_1D_KERNEL_LOOP(i, output_size)
    {
      int row = i / out_num_cols;
      int col = i % out_num_cols;

      IndT out_ind_col = ldg(out_indices + col);

      if(out_ind_col < 0)
      {
        output[i] = pad_elem[0];
      }
      else
      {
        output[i] = ldg(params + (row * params_cols) + out_ind_col);
      }
    }
  }

  namespace functor {
    template <typename T, typename IndT>
    struct ScatterColumnsFunctor<GPUDevice, T, IndT> {
      Status operator()(const GPUDevice& d, const typename TTypes<T>::ConstMatrix& params,
                        const typename TTypes<IndT>::ConstFlat& indices,
                        const IndT& out_num_cols,
                        const T* pad_elem,
                        const int64& params_rows,
                        const int64& params_cols,
                        typename TTypes<T>::Matrix& output)
      {
        const int64 output_size = output.size();
        const int64 indices_size = indices.size();

        IndT* h_indices = nullptr;
        h_indices = (IndT*)malloc(indices_size * sizeof(IndT));

        std::vector<IndT> out_indices(out_num_cols, -1); //--Here '-1' refers to padding column(s)--//
        if(h_indices)
        {
          cudaMemcpy(h_indices, indices.data(), indices_size * sizeof(IndT), cudaMemcpyDeviceToHost);

          unordered_set<IndT> unique_ind(h_indices, &h_indices[indices_size]);
          if(unique_ind.size() != indices_size)
          {
            return errors::InvalidArgument("Indices cannot contain duplicates.",
                                           " Total no. of indices: ", indices_size,
                                           " != no. of unique indices: ", unique_ind.size(), ".");
          }

          //--Arrange output indices--//
          //--E.g.:  params = [11, 12, 13, 14]
          //-- out_num_cols = 10
          //--     pad_elem = 0
          //--      indices = [7, 4, 2, 3]
          //--       output = [0, 0, 13, 14, 12, 0, 0, 11, 0, 0]
          //--  out_indices = [-1, -1, 2, 3, 1, -1, -1, 0, -1, -1]

          for(IndT i=0; i<indices_size; i++)
          {
            //--Check indices[i] ∈ (0, out_num_cols]--//
            if (!FastBoundsCheck(h_indices[i], out_num_cols))
            {
              return errors::InvalidArgument("Indices(", i, "): ", h_indices[i], " is not in range (0, ", out_num_cols, "].");
            }

            out_indices[h_indices[i]] = i;
          }
        }
        else
        {
          return errors::ResourceExhausted("Could not allocate h_indices on host.");
        }

        IndT* d_out_indices = nullptr;
        cudaMalloc(&d_out_indices, out_num_cols * sizeof(IndT));

        if(d_out_indices)
        {
          cudaMemcpy(d_out_indices, out_indices.data(), out_num_cols * sizeof(IndT), cudaMemcpyHostToDevice);
        }
        else
        {
          return errors::ResourceExhausted("Could not allocate d_out_indices on device.");
        }

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
        float time_taken;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
#endif // EXEC_TIME_CALC

        CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);
        ScatterColumnsOpKernel<T, IndT>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(params.data(), d_out_indices,
                                                                             pad_elem, out_num_cols, params_cols,
                                                                             output_size, output.data());

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_taken,start,stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::cout << "GPU - Time taken: " << time_taken << " ms" << endl;
#endif // EXEC_TIME_CALC

        cudaDeviceSynchronize();
        if(h_indices)
          free(h_indices);

        return Status::OK();
      }
    };
  }  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_COLUMNS_FUNCTOR_GPU_CU_H_

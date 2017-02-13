#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "gather_columns_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

  typedef Eigen::GpuDevice GPUDevice;

  template <typename T, typename IndT>
  __global__ void GatherColumnsOpKernel(const T* params, const IndT* indices,
                                        int64 params_cols, int64 indices_size,
                                        int64 output_size, T* output)
  {
    CUDA_1D_KERNEL_LOOP(i, output_size)
    {
      int row = i / indices_size;
      int col = i % indices_size;

      IndT ind_col = ldg(indices + col);
      output[i] = ldg(params + (row * params_cols) + ind_col);
    }
  }

  namespace functor {
    template <typename T, typename IndT>
    struct GatherColumnsFunctor<GPUDevice, T, IndT> {
      int64 operator()(const GPUDevice& d, typename TTypes<T>::ConstMatrix params,
                       typename TTypes<IndT>::ConstFlat indices,
                       int64 params_rows,
                       int64 params_cols,
                       typename TTypes<T>::Matrix output)
      {
        const int64 output_size = output.size();
        const int64 indices_size = indices.size();

        IndT* h_indices = nullptr;
        h_indices = (IndT*)malloc(indices_size * sizeof(IndT));

        if(h_indices)
        {
          cudaMemcpy(h_indices, indices.data(), indices_size * sizeof(IndT), cudaMemcpyDeviceToHost);

          for(int c=0; c < indices_size; c++)
          {
            //--Check indices[i] ∈ (0, params_cols]--//
            if (!FastBoundsCheck(h_indices[c], params_cols))
            {
              free(h_indices);
              return c;
            }
          }
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
        GatherColumnsOpKernel<T, IndT>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(params.data(), indices.data(),
                                                                             params_cols, indices_size,
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

        if(h_indices)
          free(h_indices);

        return -1;
      }
    };
  }  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_FUNCTOR_GPU_CU_H_

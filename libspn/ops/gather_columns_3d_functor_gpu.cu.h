#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_3D_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_3D_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "gather_columns_3d_functor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename IndT>
__global__ void GatherColumns3dPaddingOpKernel(const T* params,
                                      const int64 params_cols,
                                      const IndT* indices,
                                      const int64 indices_size,
                                      T* output,
                                      const int64 output_size)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    //--Calculate column index of params--//
    IndT col = ldg(indices + (i % indices_size));
    if(col > -1)
    {
      //--If not a negative index (indicating a non-padded column),
      //  then calculate the params row and copy the params elemnt
      //  to output--//
      IndT row = (i / indices_size) * params_cols;
      output[i] = ldg(params + row + col);
    }
  }
}

template <typename T, typename IndT>
__global__ void GatherColumns3dNoPaddingOpKernel(const T* params,
                                      const int64 params_cols,
                                      const IndT* indices,
                                      const int64 indices_size,
                                      T* output,
                                      const int64 output_size)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    //--Calculate row index of params--//
    IndT row = (i / indices_size) * params_cols;
    //--Calculate column index of params--//
    IndT col = ldg(indices + (i % indices_size));

    output[i] = ldg(params + row + col);
  }
}

namespace functor
{
template <typename T, typename IndT>
struct GatherColumns3dFunctor<GPUDevice, T, IndT>
{
  Status operator()(const GPUDevice& d,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    typename TTypes<T>::Matrix& output,
                    const bool& padding)
  {
    //--Get rows and cols size of params and indices--//
    const int64 params_rows = params.dimension(0);
    const int64 params_cols = params.dimension(1);
    const int64 indices_rows = indices.dimension(0);
    const int64 indices_cols = indices.dimension(1);

    //--Get size of params, indices and output--//
    const int64 output_size = output.size();
    const int64 params_size = params.size();
    const int64 indices_size = indices.size();

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
    float time_taken;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif  // EXEC_TIME_CALC
    //--Create a cuda configuration object--//
    CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);
    if(padding)
    {
      //--Declare padding element as a double, and set it to value 0.0--//
      double pad_elem = 0.0;

      //--Since output has atlest one padded column,
      //  initialize output tensor with pad_element--//
      cudaMemset(output.data(), 0, output_size*sizeof(T));

      //--Cuda kernal optimized for output with padded-columns--//
      GatherColumns3dPaddingOpKernel<T, IndT>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          params.data(), params_cols, indices.data(),
          indices_size, output.data(), output_size);
    }
    else //--No Padding needed--//
    {
      //--Cuda kernal optimized for output without padded-columns--//
      GatherColumns3dNoPaddingOpKernel<T, IndT>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          params.data(), params_cols, indices.data(),
          indices_size, output.data(), output_size);
    }

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "GPU - Time taken: " << time_taken << " ms" << endl;
#endif  // EXEC_TIME_CALC

    return Status::OK();
  }
};
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_3D_FUNCTOR_GPU_CU_H_

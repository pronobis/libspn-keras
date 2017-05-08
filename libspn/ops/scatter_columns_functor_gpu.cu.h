#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_COLUMNS_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_COLUMNS_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "scatter_columns_functor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void OutputPadInitKernel(T* output,
                                    const int64 output_size,
                                    const T* pad_elem)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    output[i] = ldg(pad_elem);
  }
}

template <typename T, typename IndT>
__global__ void ScatterColumnsOpKernel(const T* params,
                                       const int64 params_size,
                                       const IndT* indices,
                                       const IndT num_out_cols,
                                       const int64 params_cols,
                                       T* output)
{
  CUDA_1D_KERNEL_LOOP(i, params_size)
  {
    int row = i / params_cols;
    int col = i % params_cols;

    IndT output_ind = (row * num_out_cols) + ldg(indices + col);

    output[output_ind] = ldg(params + i);
  }
}

namespace functor
{
template <typename T, typename IndT>
struct ScatterColumnsFunctor<GPUDevice, T, IndT>
{
  Status operator()(const GPUDevice& d,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstFlat& indices,
                    const IndT& num_out_cols, const T* pad_elem,
                    const int64& params_rows, const int64& params_cols,
                    typename TTypes<T>::Matrix& output)
  {
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

    //--Initialize output tensor with pad_element--//
    CudaLaunchConfig config = GetCudaLaunchConfig(params_size, d);
    OutputPadInitKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        output.data(), output_size, pad_elem);

    config = GetCudaLaunchConfig(params_size, d);
    ScatterColumnsOpKernel<T, IndT>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        params.data(), params_size, indices.data(),
        num_out_cols, params_cols,  output.data());

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

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_COLUMNS_FUNCTOR_GPU_CU_H_

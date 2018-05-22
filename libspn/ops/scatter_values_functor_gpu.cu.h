#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_VALUES_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_VALUES_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "scatter_values_functor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void OutputPadInitKernel(T* output,
                                    const int64 output_size,
                                    const T pad_elem)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    output[i] = pad_elem;
  }
}

template <typename T, typename IndT>
__global__ void ScatterValuesOpKernel(const T* params,
                                      const int64 params_size,
                                      const IndT* indices,
                                      const IndT num_out_cols,
                                      T* output)
{
  CUDA_1D_KERNEL_LOOP(i, params_size)
  {
    IndT output_ind = (i * num_out_cols) + ldg(indices + i);
    output[output_ind] = ldg(params + i);
  }
}

namespace functor
{
template <typename T, typename IndT>
struct ScatterValuesFunctor<GPUDevice, T, IndT>
{
  Status operator()(const GPUDevice& d,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output)
  {
    const int64 output_size = output.size();
    const int64 params_size = params.size();

//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
    float time_taken;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif  // EXEC_TIME_CALC

    //--Declare padding element as a double, and set it to default value 0.0--//
    double pad_elem = 0.0;

    //--Initialize output tensor with pad_element--//
    CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);
    OutputPadInitKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        output.data(), output_size, (T)pad_elem);

    config = GetCudaLaunchConfig(params_size, d);
    ScatterValuesOpKernel<T, IndT>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        params.data(), params_size, indices.data(),
        num_out_cols, output.data());

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

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_VALUES_FUNCTOR_GPU_CU_H_

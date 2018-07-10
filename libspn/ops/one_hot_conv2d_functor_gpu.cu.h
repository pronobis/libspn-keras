#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_ONE_HOT_CONV2D_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_ONE_HOT_CONV2D_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "one_hot_conv2d_functor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename IndT>
__global__ void OneHotConv2DOpKernel(
        const T* input, const IndT* filter, T* output,
        const int64 in_rows, const int64 in_cols,
        const int64 out_rows, const int64 out_cols,
        const int64 filter_rows, const int64 filter_cols,
        const int64 strides_row, const int64 strides_col,
        const int64 dilation_row, const int64 dilation_col,
        const int64 in_depth, const int64 out_depth,
        const int64 out_size)
{
    CUDA_1D_KERNEL_LOOP(i, out_size)
    {
        const int64 batch_ind = i / (out_rows * out_cols * out_depth);
        const int64 out_row = (i % (out_rows * out_cols * out_depth)) / (out_cols * out_depth);
        const int64 out_col = (i % (out_cols * out_depth)) / out_depth;
        const int64 out_channel = i % out_depth;

        const int64 in_row_0 = out_row * strides_row;
        const int64 in_col_0 = out_col * strides_col;

        const int64 in_batch_ind0 = batch_ind * (in_rows * in_cols * in_depth);

        output[i] = static_cast<T>(0);
        for (int filter_row = 0; filter_row < filter_rows; ++filter_row)
        {
            const int64 in_row = in_row_0 + filter_row * dilation_row;
            const int64 in_row_ind0 = in_row * (in_cols * in_depth);
            const int64 filter_row_ind0 = filter_row * (filter_cols * out_depth);
            for (int filter_col = 0; filter_col < filter_cols; ++filter_col)
            {
                const int64 in_col = in_col_0 + filter_col * dilation_col;
                const int64 input_channel = static_cast<int64>(
                    ldg(filter + filter_row_ind0 + filter_col * out_depth + out_channel));
                const int64 in_ind = in_batch_ind0 + in_row_ind0 + in_col * in_depth + input_channel;
                output[i] += ldg(input + in_ind);
            }
        }
    }
}

namespace functor
{
template <typename T, typename IndT>
struct OneHotConv2DFunctor<GPUDevice, T, IndT>
{
  Status operator()(const GPUDevice& d,
                    const typename TTypes<T, 4>::ConstTensor input,
                    const typename TTypes<IndT, 3>::ConstTensor filter,
                    const int64 dilation_rows, const int64 dilation_cols,
                    const int64 stride_rows, const int64 stride_cols,
                    typename TTypes<T, 4>::Tensor output)
  {
    const int64 out_rows = output.dimension(1), out_cols = output.dimension(2);
    const int64 in_rows = input.dimension(1), in_cols = input.dimension(2);
    const int64 in_depth = input.dimension(3), out_depth = output.dimension(3);
    const int64 filter_rows = filter.dimension(0), filter_cols = filter.dimension(1);


//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
    float time_taken;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif  // EXEC_TIME_CALC

    const int64 out_size = output.size();

    //--Initialize output tensor with pad_element--//
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
    OneHotConv2DOpKernel<T, IndT>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        input.data(), filter.data(), output.data(), in_rows, in_cols,
        out_rows, out_cols, filter_rows, filter_cols, stride_rows, stride_cols,
        dilation_rows, dilation_cols, in_depth, out_depth, out_size);

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

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_ONE_HOT_CONV2D_FUNCTOR_GPU_CU_H_

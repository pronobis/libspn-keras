#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_ONE_HOT_CONV2D_BACKPROP_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_ONE_HOT_CONV2D_BACKPROP_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "one_hot_conv2d_backprop_functor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void ZeroInitKernel(T* output, const int64 output_size)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    output[i] = static_cast<T>(0);
  }
}

template <typename T, typename IndT>
__global__ void OneHotConv2DBackpropOpKernel(
        const T* out_grad, const IndT* filter, T* in_grad,
        const int64 in_rows, const int64 in_cols,
        const int64 out_rows, const int64 out_cols,
        const int64 filter_rows, const int64 filter_cols,
        const int64 strides_row, const int64 strides_col,
        const int64 dilation_row, const int64 dilation_col,
        const int64 in_depth, const int64 out_depth,
        const int64 out_grad_size, bool overlapping_patches)
{
    // From the perspective of the out gradient
    CUDA_1D_KERNEL_LOOP(i, out_grad_size)
    {

        // TODO SET TO ZERO FIRST
        const int64 batch_ind = i / (out_rows * out_cols * out_depth);
        const int64 out_row = (i % (out_rows * out_cols * out_depth)) / (out_cols * out_depth);
        const int64 out_col = (i % (out_cols * out_depth)) / out_depth;
        const int64 out_channel = i % out_depth;

        const int64 in_row_0 = out_row * strides_row;
        const int64 in_col_0 = out_col * strides_col;

        const int64 in_batch_ind0 = batch_ind * (in_rows * in_cols * in_depth);


        const T grad_val = ldg(out_grad + i);
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
                // HAS TO BE ATOMIC
//                printf("%d: row      %lld\n"
//                       "%d: col      %lld\n"
//                       "%d: channel  %lld\n"
//                       "%d: in_ind   %lld\n"
//                       "\n"
//                       "%d: out_row  %lld\n"
//                       "%d: out_col  %lld\n",
//                       i, in_row, i, in_col, i, input_channel, i, in_ind, i, out_row, i, out_col);
//                if (overlapping_patches)
                // TODO should be able to exploit the fact that the there is no overlap between
                // patches...
                CudaAtomicAdd(in_grad + in_ind, grad_val);
//                else
//                    in_grad[in_ind] += ldg(out_grad + i);
            }
        }
    }
}


//template <typename T, typename IndT>
//__global__ void OneHotConv2DBackpropOpKernelV2(
//        const T* out_grad, const IndT* filter, T* in_grad,
//        const int64 in_rows, const int64 in_cols,
//        const int64 out_rows, const int64 out_cols,
//        const int64 filter_rows, const int64 filter_cols,
//        const int64 strides_row, const int64 strides_col,
//        const int64 dilation_row, const int64 dilation_col,
//        const int64 in_depth, const int64 out_depth,
//        const int64 out_grad_size)
//{
//    // From the perspective of the out gradient
//    CUDA_1D_KERNEL_LOOP(i, out_grad_size)
//    {

//        // TODO SET TO ZERO FIRST
//        const int64 batch_ind = i / (out_rows * out_cols);
//        const int64 out_row = (i % (out_rows * out_cols)) / out_cols;
//        const int64 out_col = i % out_cols;

//        const int64 in_row_0 = out_row * strides_row;
//        const int64 in_col_0 = out_col * strides_col;

//        const int64 in_batch_ind0 = batch_ind * (in_rows * in_cols * in_depth);

//        in_grad[i] = static_cast<T>(0);
//        for (int filter_row = 0; filter_row < filter_rows; ++filter_row)
//        {
//            const int64 in_row = in_row_0 + filter_row * dilation_row;
//            const int64 in_row_ind0 = in_row * (in_cols * in_depth);
//            const int64 filter_row_ind0 = filter_row * (filter_cols * out_depth);
//            for (int filter_col = 0; filter_col < filter_cols; ++filter_col)
//            {
//                const int64 in_col = in_col_0 + filter_col * dilation_col;
//                const int64 input_channel = static_cast<int64>(
//                    ldg(filter + filter_row_ind0 + filter_col * out_depth + out_channel));
//                const int64 in_ind = in_batch_ind0 + in_row_ind0 + in_col * in_depth + input_channel;
//                // HAS TO BE ATOMIC
//                CudaAtomicAdd(in_grad + in_ind, ldg(out_grad + i));
//            }
//        }
//    }
//}

namespace functor
{
template <typename T, typename IndT>
struct OneHotConv2DBackpropFunctor<GPUDevice, T, IndT>
{
  Status operator()(const GPUDevice& d,
                    const typename TTypes<T, 4>::ConstTensor out_grad,
                    const typename TTypes<IndT, 3>::ConstTensor filter,
                    const int64 dilation_rows, const int64 dilation_cols,
                    const int64 stride_rows, const int64 stride_cols,
                    typename TTypes<T, 4>::Tensor in_grad, bool overlapping_patches)
  {
    const int64 out_rows    = out_grad.dimension(1),    out_cols    = out_grad.dimension(2);
    const int64 in_rows     = in_grad.dimension(1),     in_cols     = in_grad.dimension(2);
    const int64 filter_rows = filter.dimension(0),      filter_cols = filter.dimension(1);
    const int64 in_depth    = in_grad.dimension(3),     out_depth   = out_grad.dimension(3);
//    printf("filter_rows      %lld\n"
//           "filter_cols      %lld\n",
//           filter_rows, filter_cols);


//--Debugging flag disabled by default--//
#if EXEC_TIME_CALC
    float time_taken;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif  // EXEC_TIME_CALC

    int64 num_elements = in_grad.size();

    //--Initialize output tensor with pad_element--//
//    printf("Launching Zero Init\n");
    CudaLaunchConfig config_zero_init = GetCudaLaunchConfig(num_elements, d);
    ZeroInitKernel<T>
        <<<config_zero_init.block_count, config_zero_init.thread_per_block, 0, d.stream()>>>(
        in_grad.data(), num_elements);

    num_elements = out_grad.size();
    CudaLaunchConfig config_backprop = GetCudaLaunchConfig(num_elements, d);

//    printf("Launching Backprop\n");
    OneHotConv2DBackpropOpKernel<T, IndT>
        <<<config_backprop.block_count, config_backprop.thread_per_block, 0, d.stream()>>>(
        out_grad.data(), filter.data(), in_grad.data(), in_rows, in_cols,
        out_rows, out_cols, filter_rows, filter_cols, stride_rows, stride_cols,
        dilation_rows, dilation_cols, in_depth, out_depth, num_elements, overlapping_patches);

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

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_ONE_HOT_CONV2D_BACKPROP_FUNCTOR_GPU_CU_H_

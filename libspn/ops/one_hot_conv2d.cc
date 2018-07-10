#include "one_hot_conv2d_functor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow
{
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;
using shape_inference::DimensionHandle;

REGISTER_OP("OneHotConv2D")
    .Attr("T: type")
    .Attr("IndT: {int32,int64}")
    .Attr("dilations: list(int) = [1, 1]")
    .Attr("strides: list(int)")
    .Input("input: T")
    .Input("filters: IndT")
    .Output("convolved: T")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 4, &input_shape));

      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 3, &filter_shape));

      TensorFormat data_format;
      if (!FormatFromString("NHWC", &data_format)) {
          return errors::InvalidArgument("Invalid data format string: ",
                                         "NHWC");
      }

      std::vector<int32> dilations;
      TF_RETURN_IF_ERROR(ctx->GetAttr("dilations", &dilations));
      if (dilations.size() != 2) {
          return errors::InvalidArgument(
                      "OneHotConv2D requires the dilation attribute to contain 2 values, but got: ",
                      dilations.size());
      }
      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(ctx->GetAttr("strides", &strides));
      if (strides.size() != 2) {
          return errors::InvalidArgument("OneHotConv2D on data format requires "
                                         "the stride attribute to contain"
                                         " 2 values, but got: ",
                                         strides.size());
      }

      DimensionHandle batch_size_dim = ctx->Dim(input_shape, 0);
      DimensionHandle input_depth_dim = ctx->Dim(input_shape, 3);
      DimensionHandle input_rows_dim = ctx->Dim(input_shape, 1);
      DimensionHandle input_cols_dim = ctx->Dim(input_shape, 2);

      const int32 stride_rows = strides[0];
      const int32 stride_cols = strides[1];
      const int32 dilation_rows = dilations[0];
      const int32 dilation_cols = dilations[1];

      DimensionHandle filter_rows_dim = ctx->Dim(filter_shape, 0);
      DimensionHandle filter_cols_dim = ctx->Dim(filter_shape, 1);
      DimensionHandle output_depth_dim = ctx->Dim(filter_shape, 2);

      DimensionHandle output_rows, output_cols;
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
          ctx, input_rows_dim, filter_rows_dim, dilation_rows, stride_rows,
          Padding::VALID, &output_rows));
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
          ctx, input_cols_dim, filter_cols_dim, dilation_cols, stride_cols,
          Padding::VALID, &output_cols));
      std::vector<DimensionHandle> out_shape_dims = {
          batch_size_dim, output_rows, output_cols, output_depth_dim};
      ShapeHandle output_shape = ctx->MakeShape(out_shape_dims);
      ctx->set_output(0, output_shape);
      return Status::OK();
    });

template <typename Device, typename T, typename IndT>
class OneHotConv2DOp : public OpKernel
{
 public:
  explicit OneHotConv2DOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES(ctx, dilations_.size() == 2,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 2 dimensions"));
    OP_REQUIRES(ctx, strides_.size() == 2,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 2 dimensions"));
    const int64 stride_row = strides_[0];
    const int64 stride_col = strides_[1];

    OP_REQUIRES(ctx, stride_row > 0 && stride_col > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));

    const int64 dilation_row = dilations_[0];
    const int64 dilation_col = dilations_[1];


    OP_REQUIRES(
        ctx, dilation_row > 0 && dilation_col > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));
  }

  void Compute(OpKernelContext* ctx) override
  {
    //--Grab the input tensor - params--//
    const Tensor& input = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& filter = ctx->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(ctx, filter.dims() == 3,
                errors::InvalidArgument("filter must be 3-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          ctx,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    // Get input and output depths
    const int64 input_depth = input.dim_size(3);
    const int64 output_depth = filter.dim_size(2);

    // Obtain input dimension sizes
    const int64 batch_size = input.dim_size(0);
    const int64 input_rows = input.dim_size(1);
    const int64 input_cols = input.dim_size(2);

    // Obtain kernel dimension sizes
    const int64 filter_rows = filter.dim_size(0);
    const int64 filter_cols = filter.dim_size(1);

    // TODO get the dilation rates
    const int64 kernel_rows_effective = (filter_rows - 1) * dilations_[0] + 1;
    const int64 kernel_cols_effective = (filter_cols - 1) * dilations_[1] + 1;

    // TODO take care of the rounding and casting, needs something like np.ceil
    const int64 out_rows = (input_rows - kernel_rows_effective) / strides_[0] + 1;
    const int64 out_cols = (input_cols - kernel_cols_effective) / strides_[1] + 1;

    gtl::InlinedVector<int64, 4> dim_sizes = {batch_size, out_rows, out_cols, output_depth};

    TensorShape out_shape(dim_sizes);

    //--Create an output tensor--//
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    // Nothing to compute
    if (out_shape.num_elements() == 0) {
        return;
    }



    functor::OneHotConv2DFunctor<Device, T, IndT> functor;
    functor(ctx->eigen_device<Device>(),
            input.tensor<T, 4>(), filter.tensor<IndT, 3>(),
            dilations_[0], dilations_[1],
            strides_[0], strides_[1],
            output->tensor<T, 4>());
  }

private:
 std::vector<int32> dilations_;
 std::vector<int32> strides_;
};

#define REGISTER_ONEHOTCONV2D_ALL(dev, type, index_type)          \
  REGISTER_KERNEL_BUILDER(Name("OneHotConv2D")                    \
                              .Device(DEVICE_##dev)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<index_type>("IndT"), \
                          OneHotConv2DOp<dev##Device, type, index_type>)

#define REGISTER_ONEHOTCONV2D_ALL_INDICES(dev, type) \
  REGISTER_ONEHOTCONV2D_ALL(dev, type, int32);       \
  REGISTER_ONEHOTCONV2D_ALL(dev, type, int64)

//#define REGISTER_GATHERCOLUMNS_CPU(type) \
//  REGISTER_GATHERCOLUMNS_ALL_INDICES(CPU, type)

//--Registration of the CPU implementations--//
//TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS_CPU);
//TF_CALL_QUANTIZED_TYPES(REGISTER_GATHERCOLUMNS_CPU);

//#undef REGISTER_GATHERCOLUMNS_CPU

#if GOOGLE_CUDA

//--Registration of the GPU implementations--//
#define REGISTER_ONEHOTCONV2D_GPU(type) \
  REGISTER_ONEHOTCONV2D_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ONEHOTCONV2D_GPU);

#undef REGISTER_ONEHOTCONV2D_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_ONEHOTCONV2D_ALL_INDICES
#undef REGISTER_ONEHOTCONV2D_ALL

}  // namespace tensorflow

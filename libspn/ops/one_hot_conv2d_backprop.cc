#include "one_hot_conv2d_backprop_functor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow
{
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

REGISTER_OP("OneHotConv2DBackprop")
    .Attr("T: type")
    .Attr("IndT: {int32,int64}")
    .Attr("dilations: list(int) = [1, 1]")
    .Attr("strides: list(int)")
    .Input("input: T")
    .Input("filters: IndT")
    .Input("grad: T")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* ctx) {
        ShapeHandle filter_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 3, &filter_shape));

        ShapeHandle grad_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 4, &grad_shape));


        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 4, &input_shape));

        // TODO shouldn't be necessary to pass in the input tensor. Should be able to
        // work with just its shape...
//        ShapeHandle out_shape;
//        TF_RETURN_IF_ERROR(ctx->MakeShapeFromShapeTensor(0, &out_shape));
//        TF_RETURN_IF_ERROR(ctx->WithRank(out_shape, 4, &out_shape));
        ctx->set_output(0, input_shape);

        return Status::OK();
    });

template <typename Device, typename T, typename IndT>
class OneHotConv2DBackpropOp : public OpKernel
{
 public:
  explicit OneHotConv2DBackpropOp(OpKernelConstruction* ctx) : OpKernel(ctx)
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

//    printf("Compute OneHotConv2DBackprop started\n");
    const Tensor& input = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& filter = ctx->input(1);

    const Tensor& out_backprop = ctx->input(2);

//    printf("Checking input_sizes\n");
//    OP_REQUIRES(
//        ctx, TensorShapeUtils::IsVector(input_sizes.shape()),
//        errors::InvalidArgument(
//            "OneHotConv2DBackprop: input_sizes input must be 1-dim, not ",
//            input_sizes.dims()));

//    printf("Setting input shape\n");
//    printf("Input sizes %s\n", input_sizes.shape().DebugString().c_str());
//    auto input_sizes_vec = input_sizes.vec<int32>().data();
//    auto input_sizes_flat = input_sizes.flat<int32>().data();
//    printf("Converted to vec\n");
//    printf("Input sizes size %d\n", input_sizes.vec<int32>().size());

//    gtl::InlinedVector<int32, 4> dim_sizes = {batch_size, out_rows, out_cols, output_depth};
//    printf("Input sizes (%d, %d, %d, %d)", input_sizes_flat[0], input_sizes_flat[1], input_sizes_flat[2], input_sizes_flat[3]);
//    gtl::InlinedVector<int32, 4> dim_sizes(input_sizes_vec);
//    TensorShape input_shape;
//    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
//                                input_sizes.vec<int32>(), &input_shape));
//    printf("Input shape %s", input_shape.DebugString().c_str());

//    printf("Input shape set\n");
    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(ctx, out_backprop.dims() == 4,
                errors::InvalidArgument("out grad must be 4-dimensional",
                                        out_backprop.shape().DebugString()));
    OP_REQUIRES(ctx, filter.dims() == 3,
                errors::InvalidArgument("filter must be 3-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          ctx,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }


    auto input_shape = input.shape();
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_shape, &in_backprop));

    // Nothing to compute
    if (input_shape.num_elements() == 0) {
        return;
    }

    const int64 filter_rows = filter.dim_size(0);
    const int64 filter_cols = filter.dim_size(1);
    const int64 kernel_rows_effective = (filter_rows - 1) * dilations_[0] + 1;
    const int64 kernel_cols_effective = (filter_cols - 1) * dilations_[1] + 1;

    bool overlapping_patches = false;
    if (kernel_rows_effective > strides_[0] || kernel_cols_effective > strides_[1])
        overlapping_patches = true;

    functor::OneHotConv2DBackpropFunctor<Device, T, IndT> functor;
    functor(ctx->eigen_device<Device>(),
            out_backprop.tensor<T, 4>(), filter.tensor<IndT, 3>(),
            dilations_[0], dilations_[1],
            strides_[0], strides_[1],
            in_backprop->tensor<T, 4>(), overlapping_patches);
  }

private:
 std::vector<int32> dilations_;
 std::vector<int32> strides_;
};

#define REGISTER_ONEHOTCONV2DBACKPROP_ALL(dev, type, index_type)          \
  REGISTER_KERNEL_BUILDER(Name("OneHotConv2DBackprop")                    \
                              .Device(DEVICE_##dev)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<index_type>("IndT"), \
                          OneHotConv2DBackpropOp<dev##Device, type, index_type>)

#define REGISTER_ONEHOTCONV2DBACKPROP_ALL_INDICES(dev, type) \
  REGISTER_ONEHOTCONV2DBACKPROP_ALL(dev, type, int32);       \
  REGISTER_ONEHOTCONV2DBACKPROP_ALL(dev, type, int64)

//#define REGISTER_GATHERCOLUMNS_CPU(type) \
//  REGISTER_GATHERCOLUMNS_ALL_INDICES(CPU, type)

//--Registration of the CPU implementations--//
//TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS_CPU);
//TF_CALL_QUANTIZED_TYPES(REGISTER_GATHERCOLUMNS_CPU);

//#undef REGISTER_GATHERCOLUMNS_CPU

#if GOOGLE_CUDA

//--Registration of the GPU implementations--//
#define REGISTER_ONEHOTCONV2DBACKPROP_GPU(type) \
  REGISTER_ONEHOTCONV2DBACKPROP_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ONEHOTCONV2DBACKPROP_GPU);

#undef REGISTER_ONEHOTCONV2DBACKPROP_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_ONEHOTCONV2DBACKPROP_ALL_INDICES
#undef REGISTER_ONEHOTCONV2DBACKPROP_ALL

}  // namespace tensorflow

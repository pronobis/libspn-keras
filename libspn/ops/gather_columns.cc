#include "gather_columns_functor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

REGISTER_OP("GatherColumns")
    .Input("params: T")
    .Input("indices: IndT")
    .Output("columns: T")
    .Attr("T: type")
    .Attr("IndT: {int32,int64}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle params_shape;
      TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 1, &params_shape));
      TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 2, &params_shape));

      ShapeHandle indices_shape = ctx->input(1);
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 1, &indices_shape));

      ShapeHandle out_shape;
      TF_RETURN_IF_ERROR(ctx->ReplaceDim(
          params_shape, -1, ctx->Dim(indices_shape, 0), &out_shape));
      ctx->set_output(0, out_shape);

      return Status::OK();
    });

template <typename Device, typename T, typename IndT>
class GatherColumnsOp : public OpKernel
{
 public:
  explicit GatherColumnsOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    const DataType data_t = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<IndT>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({data_t, index_t}, {data_t}));
  }

  void Compute(OpKernelContext* ctx) override
  {
    //--Grab the input tensor - params--//
    const Tensor& params = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& indices = ctx->input(1);

    OP_REQUIRES(ctx, params.TotalBytes() > 0,
                errors::InvalidArgument("Params cannot be empty."));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("Params must be at least a vector."));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(indices.shape()),
        errors::InvalidArgument("Indices must be a vector, but it is a: ",
                                indices.dims(), "D Tensor."));

    const TensorShape& params_shape(params.shape());

    OP_REQUIRES(ctx, params_shape.dims() <= 2,
                errors::InvalidArgument("Params must be 1D or 2D but it is: ",
                                        params_shape.dims(), "D."));

    TensorShape output_shape(params_shape);

    int64 params_rows;
    int64 params_cols;
    const int64 indices_size = indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("Indices cannot be empty."));

    if (params_shape.dims() == 1)
    {
      params_rows = 1;
      params_cols = params.dim_size(0);

      //--Set output tensor dims--//
      output_shape.set_dim(0, indices_size);
    }
    else if (params_shape.dims() == 2)
    {
      params_rows = params.dim_size(0);
      params_cols = params.dim_size(1);

      //--Set output tensor dims--//
      output_shape.set_dim(0, params_rows);
      output_shape.set_dim(1, indices_size);
    }

    //--Create an output tensor--//
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto output_tensor = output->shaped<T, 2>({params_rows, indices_size});
    auto params_tensor = params.shaped<T, 2>({params_rows, params_cols});
    auto indices_flat = indices.flat<IndT>();

    functor::GatherColumnsFunctor<Device, T, IndT> functor;
    int64 bad_i =
        functor(ctx->eigen_device<Device>(), params_tensor,
                indices_flat, output_tensor);

    OP_REQUIRES(ctx, bad_i < 0, errors::InvalidArgument(
                                    "Indices(", bad_i, ") is not in range (0, ",
                                    params_cols, "]."));
  }
};

#define REGISTER_GATHERCOLUMNS_ALL(dev, type, index_type)          \
  REGISTER_KERNEL_BUILDER(Name("GatherColumns")                    \
                              .Device(DEVICE_##dev)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<index_type>("IndT"), \
                          GatherColumnsOp<dev##Device, type, index_type>)

#define REGISTER_GATHERCOLUMNS_ALL_INDICES(dev, type) \
  REGISTER_GATHERCOLUMNS_ALL(dev, type, int32);       \
  REGISTER_GATHERCOLUMNS_ALL(dev, type, int64)

#define REGISTER_GATHERCOLUMNS_CPU(type) \
  REGISTER_GATHERCOLUMNS_ALL_INDICES(CPU, type)

//--Registration of the CPU implementations--//
TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHERCOLUMNS_CPU);

#undef REGISTER_GATHERCOLUMNS_CPU

#if GOOGLE_CUDA

//--Registration of the GPU implementations--//
#define REGISTER_GATHERCOLUMNS_GPU(type) \
  REGISTER_GATHERCOLUMNS_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHERCOLUMNS_GPU);

#undef REGISTER_GATHERCOLUMNS_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHERCOLUMNS_ALL_INDICES
#undef REGISTER_GATHERCOLUMNS_ALL

}  // namespace tensorflow

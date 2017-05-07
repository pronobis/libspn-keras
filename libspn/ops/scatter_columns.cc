#include "scatter_columns_functor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;
using shape_inference::Dimension;

REGISTER_OP("ScatterColumns")
    .Input("params: T")
    .Input("indices: IndT")
    .Input("pad_elem: T")
    .Output("columns: T")
    .Attr("num_out_col: int >= 1")
    .Attr("T: type")
    .Attr("IndT: {int32,int64}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle params_shape;
      TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 1, &params_shape));
      TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 2, &params_shape));

      ShapeHandle unused_shape;
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(1), 1, &unused_shape));  //--indices--//
      TF_RETURN_IF_ERROR(
          ctx->WithRank(ctx->input(2), 0, &unused_shape));  //--pad_elem--//

      int64 num_out_cols;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_out_col", &num_out_cols));

      DimensionHandle out_last_dim;
      out_last_dim = ctx->MakeDim(num_out_cols);

      ShapeHandle out_shape;
      TF_RETURN_IF_ERROR(
          ctx->ReplaceDim(params_shape, -1, out_last_dim, &out_shape));
      ctx->set_output(0, out_shape);

      return Status::OK();
    });

template <typename Device, typename T, typename IndT>
class ScatterColumnsOp : public OpKernel
{
 public:
  explicit ScatterColumnsOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    const DataType data_t = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<IndT>::v();
    OP_REQUIRES_OK(ctx,
                   ctx->MatchSignature({data_t, index_t, data_t}, {data_t}));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_out_col", &num_out_cols));
  }

  void Compute(OpKernelContext* ctx) override
  {
    //--Grab the input tensor - params--//
    const Tensor& params = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<IndT>();

    //--Grab the input - pad_elem--//
    const Tensor& pad_elem_tensor = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("Params must be at least a vector."));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(indices.shape()),
        errors::InvalidArgument("Indices must be a vector, but it is a: ",
                                indices.dims(), "D Tensor."));

    //--Check pad_elem is a scalar--//
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(pad_elem_tensor.shape()),
        errors::InvalidArgument("pad_elem must be a scalar, but it is a: ",
                                pad_elem_tensor.dims(), "D Tensor."));

    int64 indices_size = indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("Indices cannot be empty."));

    const TensorShape& params_shape(params.shape());

    OP_REQUIRES(ctx, params_shape.dims() <= 2,
                errors::InvalidArgument("Params must be 1D or 2D but it is: ",
                                        params_shape.dims(), "D."));

    TensorShape output_shape(params_shape);

    int64 params_rows;
    int64 params_cols;

    if (params_shape.dims() == 1)
    {
      params_rows = 1;
      params_cols = params.dim_size(0);

      //--Set output tensor dims--//
      output_shape.set_dim(0, num_out_cols);
    }
    else if (params_shape.dims() == 2)
    {
      params_rows = params.dim_size(0);
      params_cols = params.dim_size(1);

      //--Set output tensor dims--//
      output_shape.set_dim(0, params_rows);
      output_shape.set_dim(1, num_out_cols);
    }

    OP_REQUIRES(ctx, num_out_cols >= params_cols,
                errors::InvalidArgument(
                    "num_out_cols: ", num_out_cols,
                    " must be >= size of the indexed dimension of params: ",
                    params_cols));

    OP_REQUIRES(
        ctx, indices_size == params_cols,
        errors::InvalidArgument("Size of indices: ", indices_size,
                                " and the indexed dimension of params - ",
                                params_cols, " - must be the same."));

    //--Create an output tensor--//
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto output_tensor = output->shaped<T, 2>({params_rows, num_out_cols});
    auto params_tensor = params.shaped<T, 2>({params_rows, params_cols});
    auto pad_elem = pad_elem_tensor.flat<T>();

    functor::ScatterColumnsFunctor<Device, T, IndT> functor;

    OP_REQUIRES_OK(ctx, functor(ctx->eigen_device<Device>(), params_tensor,
                                indices_flat, num_out_cols, pad_elem.data(),
                                params_rows, params_cols, output_tensor));
  }

 private:
  IndT num_out_cols;
};

#define REGISTER_SCATTERCOLUMNS_ALL(dev, type, index_type)         \
  REGISTER_KERNEL_BUILDER(Name("ScatterColumns")                   \
                              .Device(DEVICE_##dev)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<index_type>("IndT"), \
                          ScatterColumnsOp<dev##Device, type, index_type>)

#define REGISTER_SCATTERCOLUMNS_ALL_INDICES(dev, type) \
  REGISTER_SCATTERCOLUMNS_ALL(dev, type, int32);       \
  REGISTER_SCATTERCOLUMNS_ALL(dev, type, int64)

#define REGISTER_SCATTERCOLUMNS_CPU(type) \
  REGISTER_SCATTERCOLUMNS_ALL_INDICES(CPU, type)

//--Registration of CPU implementations--//
TF_CALL_ALL_TYPES(REGISTER_SCATTERCOLUMNS_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_SCATTERCOLUMNS_CPU);

#undef REGISTER_SCATTERCOLUMNS_CPU

#if GOOGLE_CUDA

//--Registration of the GPU implementations--//
#define REGISTER_SCATTERCOLUMNS_GPU(type) \
  REGISTER_SCATTERCOLUMNS_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTERCOLUMNS_GPU);

#undef REGISTER_SCATTERCOLUMNS_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_SCATTERCOLUMNS_ALL_INDICES
#undef REGISTER_SCATTERCOLUMNS_ALL

}  // namespace tensorflow

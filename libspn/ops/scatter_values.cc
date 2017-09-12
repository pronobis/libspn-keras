#include "scatter_values_functor.h"
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

REGISTER_OP("ScatterValues")
    .Input("params: T")
    .Input("indices: IndT")
    .Output("values: T")
    .Attr("num_out_cols: int >= 1")
    .Attr("T: realnumbertype")
    .Attr("IndT: {int32, int64}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle params_shape;
      TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 1, &params_shape));
      TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 2, &params_shape));

      ShapeHandle indices_shape;
      TF_RETURN_IF_ERROR(
          ctx->WithRankAtLeast(ctx->input(1), 1, &indices_shape));  //--indices--//
      TF_RETURN_IF_ERROR(
          ctx->WithRankAtMost(ctx->input(1), 2, &indices_shape));  //--indices--//

      int64 num_out_cols;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_out_cols", &num_out_cols));

      DimensionHandle out_last_dim;
      out_last_dim = ctx->MakeDim(num_out_cols);

      ShapeHandle out_last_dim_shape;
      out_last_dim_shape = ctx->Vector(out_last_dim);

      ShapeHandle out_shape;
      TF_RETURN_IF_ERROR(
          ctx->Concatenate(params_shape, out_last_dim_shape, &out_shape));
      ctx->set_output(0, out_shape);

      return Status::OK();
    });

template <typename Device, typename T, typename IndT>
class ScatterValuesOp : public OpKernel
{
 public:
  explicit ScatterValuesOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    const DataType data_t = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<IndT>::v();
    OP_REQUIRES_OK(ctx,
                   ctx->MatchSignature({data_t, index_t}, {data_t}));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_out_cols", &num_out_cols));
  }

  void Compute(OpKernelContext* ctx) override
  {
    //--Grab the input tensor - params--//
    const Tensor& params = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& indices = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("Params must be at least a vector."));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(indices.shape()),
                errors::InvalidArgument("Indices must be at least a vector."));

    OP_REQUIRES(ctx, params.IsSameSize(indices),
                errors::InvalidArgument("Params and Indices must be of the same shape."));

    int64 indices_size = indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("Indices cannot be empty."));

    const TensorShape& params_shape(params.shape());

    OP_REQUIRES(ctx, params_shape.dims() <= 2,
                errors::InvalidArgument("Params must be 1D or 2D but it is: ",
                                        params_shape.dims(), "D."));

    TensorShape output_shape(params_shape);

    //--Add a dimension to the end--//
    output_shape.AddDim(num_out_cols);

    int64 params_rows;
    int64 params_cols;

    if (params_shape.dims() == 1)
    {
      params_rows = params.dim_size(0);
      params_cols = 1;
    }
    else if (params_shape.dims() == 2)
    {
      params_rows = params.dim_size(0);
      params_cols = params.dim_size(1);
    }

    //--Create an output tensor--//
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto output_tensor = output->shaped<T, 2>({(params_rows * params_cols),
                                               num_out_cols});
    auto params_tensor = params.shaped<T, 2>({params_rows, params_cols});
    auto indices_tensor = indices.shaped<IndT, 2>({params_rows, params_cols});

    functor::ScatterValuesFunctor<Device, T, IndT> functor;

    OP_REQUIRES_OK(ctx, functor(ctx->eigen_device<Device>(), params_tensor,
                                indices_tensor, num_out_cols, output_tensor));
  }

 private:
  IndT num_out_cols;
};

#define REGISTER_SCATTERVALUES_ALL(dev, type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("ScatterValues")                          \
                              .Device(DEVICE_##dev)                      \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("IndT"),       \
                          ScatterValuesOp<dev##Device, type, index_type>)

#define REGISTER_SCATTERVALUES_ALL_INDICES(dev, type) \
  REGISTER_SCATTERVALUES_ALL(dev, type, int32);       \
  REGISTER_SCATTERVALUES_ALL(dev, type, int64)

#define REGISTER_SCATTERVALUES_CPU(type) \
  REGISTER_SCATTERVALUES_ALL_INDICES(CPU, type)

//--Registration of CPU implementations--//
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTERVALUES_CPU);

#undef REGISTER_SCATTERVALUES_CPU

#if GOOGLE_CUDA

//--Registration of the GPU implementations--//
#define REGISTER_SCATTERVALUES_GPU(type) \
  REGISTER_SCATTERVALUES_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTERVALUES_GPU);

#undef REGISTER_SCATTERVALUES_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_SCATTERVALUES_ALL_INDICES
#undef REGISTER_SCATTERVALUES_ALL

}  // namespace tensorflow

#include "gather_columns_3d_functor.h"
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

REGISTER_OP("GatherColumns3d")
    .Input("params: T")
    .Input("indices: IndT")
    .Output("columns: T")
    .Attr("padding: bool = false")
    .Attr("T: realnumbertype")
    .Attr("IndT: {int32, int64}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle params_shape;
      TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 1, &params_shape));
      TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 2, &params_shape));

      ShapeHandle indices_shape = ctx->input(1);
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 2, &indices_shape));

      ShapeHandle out_shape = ctx->input(1);
      if (ctx->Rank(params_shape) == 2)
      {
        DimensionHandle out_first_dim;
        out_first_dim = ctx->MakeDim(ctx->Dim(params_shape, 0));

        ShapeHandle out_first_dim_shape;
        out_first_dim_shape = ctx->Vector(out_first_dim);

        TF_RETURN_IF_ERROR(
            ctx->Concatenate(out_first_dim_shape, indices_shape, &out_shape));
      }
      ctx->set_output(0, out_shape);

      return Status::OK();
    });

template <typename Device, typename T, typename IndT>
class GatherColumns3dOp : public OpKernel
{
 public:
  explicit GatherColumns3dOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    const DataType data_t = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<IndT>::v();
    OP_REQUIRES_OK(ctx,
                   ctx->MatchSignature({data_t, index_t}, {data_t}));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
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

    //--TODO: Not needed--//
    //OP_REQUIRES(ctx, params.IsSameSize(indices),
    //            errors::InvalidArgument("Params and Indices must be of the same shape."));

    int64 indices_size = indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("Indices cannot be empty."));

    const TensorShape& params_shape(params.shape());
    int params_dims = params_shape.dims();

    OP_REQUIRES(ctx, params_dims <= 2,
                errors::InvalidArgument("Params must be 1D or 2D but it is: ",
                                        params_dims, "D."));

    const TensorShape& indices_shape(indices.shape());

    OP_REQUIRES(ctx, indices_shape.dims() == 2,
                errors::InvalidArgument("Indices must be 2D but it is: ",
                                        indices_shape.dims(), "D."));

    int64 params_rows;
    int64 params_cols;
    int64 indices_rows;
    int64 indices_cols;

    if (params_dims == 1)
    {
      params_rows = 1;
      params_cols = params.dim_size(0);
    }
    else if (params_dims == 2)
    {
      params_rows = params.dim_size(0);
      params_cols = params.dim_size(1);
    }

    indices_rows = indices.dim_size(0);
    indices_cols = indices.dim_size(1);

    TensorShape output_shape(indices_shape);

    if(params_dims > 1)
    {
      //--Add a dimension to the beginning--//
      output_shape.InsertDim(0, params_rows);
    }

    //--Create an output tensor--//
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto output_tensor = output->shaped<T, 2>({(params_rows * indices_rows),
                                               indices_cols});
    auto params_tensor = params.shaped<T, 2>({params_rows, params_cols});
    auto indices_tensor = indices.shaped<IndT, 2>({indices_rows, indices_cols});

    functor::GatherColumns3dFunctor<Device, T, IndT> functor;

    OP_REQUIRES_OK(ctx, functor(ctx->eigen_device<Device>(), params_tensor,
                                indices_tensor, output_tensor, padding));
  }

 private:
  bool padding;
};

#define REGISTER_GATHERCOLUMNS3D_ALL(dev, type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("GatherColumns3d")                          \
                              .Device(DEVICE_##dev)                                 \
                              .TypeConstraint<type>("T")                            \
                              .TypeConstraint<index_type>("IndT"),                  \
                          GatherColumns3dOp<dev##Device, type, index_type>)

#define REGISTER_GATHERCOLUMNS3D_ALL_INDICES(dev, type) \
  REGISTER_GATHERCOLUMNS3D_ALL(dev, type, int32);       \
  REGISTER_GATHERCOLUMNS3D_ALL(dev, type, int64)

#define REGISTER_GATHERCOLUMNS3D_CPU(type) \
  REGISTER_GATHERCOLUMNS3D_ALL_INDICES(CPU, type)

//--Registration of CPU implementations--//
TF_CALL_REAL_NUMBER_TYPES(REGISTER_GATHERCOLUMNS3D_CPU);

#undef REGISTER_GATHERCOLUMNS3D_CPU

#if GOOGLE_CUDA

//--Registration of the GPU implementations--//
#define REGISTER_GATHERCOLUMNS3D_GPU(type) \
  REGISTER_GATHERCOLUMNS3D_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHERCOLUMNS3D_GPU);

#undef REGISTER_GATHERCOLUMNS3D_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHERCOLUMNS3D_ALL_INDICES
#undef REGISTER_GATHERCOLUMNS3D_ALL

}  // namespace tensorflow

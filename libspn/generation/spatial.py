from collections import defaultdict, OrderedDict
from libspn.graph.convprod2d import ConvProd2D, _ConvProdNaive
from libspn.graph.convproddepthwise import ConvProdDepthWise
from libspn.graph.convsum import ConvSum
from libspn.graph.stridedslice import StridedSlice2D
from libspn.graph.localsum import LocalSum
from libspn.graph.sum import Sum
from libspn.graph.concat import Concat
from libspn.graph.sum import Sum
import numpy as np
from libspn.log import get_logger
from libspn.exceptions import StructureError



class ConvSPN:

    __logger = get_logger()

    def __init__(self, convprod_version='v1'):
        self.level_at = 0
        self.nodes_per_level = defaultdict(list)
        self.last_nodes = None
        self.node_level = OrderedDict()
        if convprod_version not in ['v1', 'v2']:
            raise ValueError("Unsupported ConvProd version {}, choose either v1 or v2"
                             .format(convprod_version))
        self._convprod_version = convprod_version
    
    def add_dilate_stride(
            self, *input_nodes, kernel_size=2, strides=(1, 4), dilation_rate=(2, 1), 
            prod_num_channels=512, padding_algorithm='valid', pad_left=None, pad_right=None, 
            pad_top=None, pad_bottom=None, name_prefixes="DilateStride", name_suffixes=("A", "B"), 
            spatial_dims=None, sum_node_type='local', sum_num_channels=None, stack_size=2,
            pad_all=None, prod_node_type='default'):
        return self.add_stack(*input_nodes, kernel_size=kernel_size, strides=strides,
                              dilation_rate=dilation_rate, prod_num_channels=prod_num_channels,
                              padding_algorithm=padding_algorithm, pad_left=pad_left,
                              pad_right=pad_right, pad_top=pad_top, pad_bottom=pad_bottom,
                              name_prefixes=name_prefixes, name_suffixes=name_suffixes,
                              spatial_dims=spatial_dims, sum_node_type=sum_node_type,
                              sum_num_channels=sum_num_channels, stack_size=stack_size,
                              pad_all=pad_all, prod_node_type=prod_node_type)
    
    def add_double_stride(
            self, *input_nodes, kernel_size=2, strides=(2, 2), dilation_rate=(1, 1), 
            prod_num_channels=512, padding_algorithm='valid', pad_left=None, pad_right=None, 
            pad_top=None, pad_bottom=None, name_prefixes="DoubleStride", name_suffixes=("A", "B"), 
            spatial_dims=None, sum_node_type='local', sum_num_channels=None, stack_size=2,
            pad_all=None, prod_node_type='default'):
        return self.add_stack(*input_nodes, kernel_size=kernel_size, strides=strides,
                              dilation_rate=dilation_rate, prod_num_channels=prod_num_channels,
                              padding_algorithm=padding_algorithm, pad_left=pad_left,
                              pad_right=pad_right, pad_top=pad_top, pad_bottom=pad_bottom,
                              name_prefixes=name_prefixes, name_suffixes=name_suffixes,
                              spatial_dims=spatial_dims, sum_node_type=sum_node_type,
                              sum_num_channels=sum_num_channels, stack_size=stack_size,
                              pad_all=pad_all, prod_node_type=prod_node_type)

    def full_wicker(self, *input_nodes, sum_node_type='local',
                    sum_num_channels=16, prod_num_channels=256, spatial_dims=None, kernel_size=2,
                    strides=1, num_channels_top=128, prod_node_type='default'):
        if spatial_dims[0] != spatial_dims[1]:
            raise ValueError("Spatial dimensions must be square.")
        stack_size = int(np.ceil(np.log2(spatial_dims[0])))
        print("Stack size", stack_size)
        if isinstance(strides, int):
            strides = [strides] * stack_size
        wicker_head = self.wicker_stack(
            *input_nodes, strides=strides, stack_size=stack_size, sum_node_type=sum_node_type,
            kernel_size=kernel_size, stack_only=True, spatial_dims=spatial_dims,
            sum_num_channels=sum_num_channels, prod_num_channels=prod_num_channels,
            prod_node_type=prod_node_type)
        out_shape = wicker_head.output_shape_spatial[:2]

        # Optionally pad output to make it a multiple of the dilation rate
        dilation_rate = int((2 ** stack_size) // np.prod(strides))
        pad_bottom = (dilation_rate - (out_shape[0] % dilation_rate)) % dilation_rate
        pad_right = (dilation_rate - (out_shape[1] % dilation_rate)) % dilation_rate

        final_conv = ConvProd2D(
            wicker_head, strides=1, pad_right=pad_right,
            pad_bottom=pad_bottom, grid_dim_sizes=out_shape,
            dilation_rate=int((2 ** stack_size) // np.prod(strides)),
            num_channels=num_channels_top
        )
        root = Sum(final_conv)
        return root

    def wicker_stack(self, *input_nodes, stack_size=2, sum_node_type='local', sum_num_channels=2,
                     pad_top=None, pad_bottom=None, pad_left=None, pad_right=None,
                     spatial_dims=None, kernel_size=2, strides=1, prod_num_channels=16,
                     dense_generator=None, add_root=True, stack_only=False,
                     name_prefix="WickerStack", prod_node_type='default'):
        pad_left_new, pad_right_new, pad_top_new, pad_bottom_new = [], [], [], []

        def none_to_zero(x):
            return 0 if x is None else x

        strides = [strides] * stack_size if isinstance(strides, int) else strides

        strides_cum_prod = np.cumprod(np.concatenate(([1], strides[:-1])))
        for pad_l, pad_r, pad_t, pad_b, s_prod, level in self._ensure_tuples_and_zip(
                pad_left, pad_right, pad_bottom, pad_top, strides_cum_prod.tolist(),
                list(range(stack_size)), size=stack_size):
            pad_at_level = int(2 ** level // s_prod)
            pad_left_new.append(none_to_zero(pad_l) + pad_at_level)
            pad_right_new.append(none_to_zero(pad_r) + pad_at_level)
            pad_top_new.append(none_to_zero(pad_t) + pad_at_level)
            pad_bottom_new.append(none_to_zero(pad_b) + pad_at_level)

        dilation_rates = np.power(2, np.arange(0, stack_size))

        if isinstance(strides, int):
            strides = np.ones(stack_size) * strides
        else:
            strides = np.asarray(strides)

        if np.any(np.greater(strides_cum_prod, dilation_rates)):
            raise ValueError("Given strides exceed dilation rates.")
        dilation_rates_effective = (dilation_rates // strides_cum_prod).astype(int).tolist()
        strides = strides.astype(int).tolist()

        num_slices = int(2 ** stack_size / np.prod(strides))
        
        stack_out = self.add_stack(
            *input_nodes, kernel_size=kernel_size, strides=strides, 
            dilation_rate=dilation_rates_effective, pad_left=pad_left_new, pad_right=pad_right_new,
            pad_top=pad_top_new, pad_bottom=pad_bottom_new, sum_num_channels=sum_num_channels,
            prod_num_channels=prod_num_channels, sum_node_type=sum_node_type, 
            spatial_dims=spatial_dims, name_prefixes=name_prefix, stack_size=stack_size,
            prod_node_type=prod_node_type)

        if stack_only:
            return stack_out

        out_rows, out_cols = stack_out.output_shape_spatial[:2]

        conv_heads = []
        for begin_row in range(num_slices):
            for begin_col in range(num_slices):
                conv_heads.append(StridedSlice2D(
                    stack_out, name="{}Row{}Col{}".format(name_prefix, begin_row, begin_col),
                    begin=(begin_row, begin_col), strides=(num_slices, num_slices),
                    grid_dim_sizes=[out_rows, out_cols]))
        if dense_generator is not None:
            dense_heads = [dense_generator.generate(head) for head in conv_heads]
            if add_root:
                root = Sum(*dense_heads)
                return root
            return dense_heads
        
        return conv_heads
    
    def add_stack(
            self, *input_nodes, kernel_size=2, strides=2, dilation_rate=1, 
            prod_num_channels=512, padding_algorithm='valid', pad_left=None, pad_right=None, 
            pad_top=None, pad_bottom=None, name_prefixes="ConvStack", name_suffixes=None, 
            spatial_dims=None, sum_node_type='local', sum_num_channels=2, stack_size=2,
            pad_all=None, prod_node_type='default'):
        name_suffixes = name_suffixes or list(range(stack_size))
        level, spatial_dims_parsed, input_nodes = self._prepare_inputs(*input_nodes)
        spatial_dims = spatial_dims_parsed or spatial_dims

        if all(p is None for p in [pad_left, pad_right, pad_top, pad_bottom]):
            pad_bottom = pad_top = pad_left = pad_right = pad_all

        for stride, dilation_r, kernel_s, prod_nc, pad_algo, pad_l, pad_r, pad_t, pad_b, sum_nc,\
                name_pref, name_suff, s_node_type, p_node_type in \
                self._ensure_tuples_and_zip(
                    strides, dilation_rate, kernel_size, prod_num_channels, padding_algorithm, 
                    pad_left, pad_right, pad_top, pad_bottom, sum_num_channels, name_prefixes, 
                    name_suffixes, sum_node_type, prod_node_type, size=stack_size):

            if p_node_type == 'default':
                next_node = ConvProd2D(
                    *input_nodes, grid_dim_sizes=spatial_dims, pad_bottom=pad_b, pad_top=pad_t,
                    pad_left=pad_l, pad_right=pad_r, num_channels=prod_nc,
                    name="{}Prod{}".format(name_pref, name_suff), dilation_rate=dilation_r,
                    kernel_size=kernel_s, padding_algorithm=pad_algo, strides=stride)
            elif p_node_type == "depthwise":
                next_node = ConvProdDepthWise(
                    *input_nodes, grid_dim_sizes=spatial_dims, pad_bottom=pad_b, pad_top=pad_t,
                    pad_left=pad_l, pad_right=pad_r, name="{}Prod{}".format(name_pref, name_suff),
                    dilation_rate=dilation_r, kernel_size=kernel_s, padding_algorithm=pad_algo,
                    strides=stride)
                if next_node._num_channels != prod_nc:
                    self.__logger.warn("Built ConvProdDepthWise with {} output channels instead "
                                       "of the user-provided {}.".format(next_node._num_channels,
                                                                         prod_nc))
            else:
                raise ValueError("Unknown spatial product type {}".format(p_node_type))

            spatial_dims = next_node.output_shape_spatial[:2]
            input_nodes = [next_node]
            self.__logger.debug1("Built node {}. ".format(next_node))
            self.__logger.debug1("\tOut shape: {} x {} x {}".format(
                *next_node.output_shape_spatial))
            self.__logger.debug1("\tStrides: {}".format(stride))
            self.__logger.debug1("\tDilations: {}".format(dilation_r))
            self.__logger.debug1("\tKernel size: {}".format(kernel_s))
            self.__logger.debug1("\tPadding: [{},{}] x [{},{}]".format(pad_t, pad_b, pad_l, pad_r))
            self._register_node(next_node, level)
            
            if s_node_type == "conv":
                next_node = ConvSum(*input_nodes, num_channels=sum_nc, grid_dim_sizes=spatial_dims,
                                    name="{}ConvSum{}".format(name_pref, name_suff))
            elif s_node_type == "local":
                next_node = LocalSum(*input_nodes, num_channels=sum_nc, grid_dim_sizes=spatial_dims,
                                     name="{}LocalSum{}".format(name_pref, name_suff))
            elif s_node_type == "skip":
                continue
            else:
                raise ValueError("Unknown sum node type '{}', use either 'conv' or 'local'."
                                 .format(s_node_type))
            input_nodes = [next_node]
            self.__logger.debug1(
                "Built node {}: {} x {} x {}".format(next_node, *next_node.output_shape_spatial))
            self._register_node(next_node, level + 1)
            level += 2
        
        self.last_nodes = input_nodes
        return input_nodes if len(input_nodes) > 1 else input_nodes[0]
        
    def _register_node(self, node, level):
        self.node_level[node] = level
        self.nodes_per_level[level].append(node)
    
    def _ensure_tuples_and_zip(self, *args, size=2):
        return zip(*[self._ensure_tuple(a, size=size) for a in args])
        
    def _prepare_inputs(self, *input_nodes):
        input_nodes = self._assure_inputs(*input_nodes)
        spatial_dims = self._compute_spatial_dims(input_nodes)
        return self._register_level(), spatial_dims, input_nodes

    def _compute_spatial_dims(self, input_nodes):
        if any(not isinstance(n, (ConvProd2D, ConvSum)) for n in input_nodes):
            return None
        spatial_dims = [n.output_shape_spatial[:2] for n in input_nodes]
        if not all(s == spatial_dims[0] for s in spatial_dims):
            raise StructureError("Incompatible spatial dimensions: \n{}".format(
                "\n".join(["{}: {}".format(node.name, s)
                           for node, s in zip(input_nodes, spatial_dims)])))
        return spatial_dims[0]

    def _assure_inputs(self, *input_nodes):
        if self.last_nodes is None and len(input_nodes) == 0:
            raise ValueError("No input was provided. There are also no nodes registered. Please "
                             "provide at least a single input.")
        if len(input_nodes) == 0:
            return self.last_nodes
        return input_nodes
    
    def _register_level(self, *input_nodes):
        node_level = self.level_at
        if any(node in self.node_level for node in input_nodes):
            node_level = max([self.node_level[n] for n in input_nodes])

        for n in input_nodes:
            if n not in self.node_level:
                self.node_level[n] = node_level
        next_level = node_level + 1
        return next_level
    
    def _ensure_tuple(self, x, size=2):
        x = x[0] if isinstance(x, (tuple, list)) and len(x) == 1 else x
        if isinstance(x, list):
            return tuple(x)
        return tuple([x] * size) if not isinstance(x, tuple) else x
    
    @property
    def prod_nodes(self):
        return [n for n in self.node_level.keys() if isinstance(n, (_ConvProdNaive, ConvProd2D))]
    
    @property
    def sum_nodes(self):
        return [n for n in self.node_level.keys() if isinstance(n, (LocalSum, ConvSum))]
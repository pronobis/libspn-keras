from collections import defaultdict
from libspn.graph.convprod2d import ConvProd2D
from libspn.graph.convsum import ConvSum
from libspn.exceptions import StructureError


class ConvSPN:

    def __init__(self):
        self.level_at = 0
        self.nodes_per_level = defaultdict(list)
        self.last_nodes = None
        self.node_level = dict()
    
    def add_dilate_stride(
            self, *input_nodes, kernel_size=2, strides=(1, 4), dilation_rate=(2, 1), 
            prod_num_channels=512, padding_algorithm='valid', pad_left=None, pad_right=None, 
            pad_top=None, pad_bottom=None, name_prefixes="DilateStride", name_suffixes=("A", "B"), 
            spatial_dims=None, use_convsums=True, sum_num_channels=None, stack_size=2, 
            pad_all=None):
        return self.add_stack(*input_nodes, kernel_size=kernel_size, strides=strides,
                              dilation_rate=dilation_rate, prod_num_channels=prod_num_channels,
                              padding_algorithm=padding_algorithm, pad_left=pad_left,
                              pad_right=pad_right, pad_top=pad_top, pad_bottom=pad_bottom,
                              name_prefixes=name_prefixes, name_suffixes=name_suffixes,
                              spatial_dims=spatial_dims, use_convsums=use_convsums, 
                              sum_num_channels=sum_num_channels, stack_size=stack_size,
                              pad_all=pad_all)
    
    def add_double_stride(
            self, *input_nodes, kernel_size=2, strides=(2, 2), dilation_rate=(1, 1), 
            prod_num_channels=512, padding_algorithm='valid', pad_left=None, pad_right=None, 
            pad_top=None, pad_bottom=None, name_prefixes="DoubleStride", name_suffixes=("A", "B"), 
            spatial_dims=None, use_convsums=True, sum_num_channels=None, stack_size=2, 
            pad_all=None):
        return self.add_stack(*input_nodes, kernel_size=kernel_size, strides=strides,
                              dilation_rate=dilation_rate, prod_num_channels=prod_num_channels,
                              padding_algorithm=padding_algorithm, pad_left=pad_left,
                              pad_right=pad_right, pad_top=pad_top, pad_bottom=pad_bottom,
                              name_prefixes=name_prefixes, name_suffixes=name_suffixes,
                              spatial_dims=spatial_dims, use_convsums=use_convsums, 
                              sum_num_channels=sum_num_channels, stack_size=stack_size, 
                              pad_all=pad_all)

    def add_stack(
            self, *input_nodes, kernel_size=2, strides=2, dilation_rate=1, 
            prod_num_channels=512, padding_algorithm='valid', pad_left=None, pad_right=None, 
            pad_top=None, pad_bottom=None, name_prefixes="ConvStack", name_suffixes=None, 
            spatial_dims=None, use_convsums=True, sum_num_channels=None, stack_size=2,
            pad_all=None):
        if use_convsums and sum_num_channels is None:
            raise StructureError("Must provide the number of channels for ConvolutionalSums")
        name_suffixes = name_suffixes or list(range(stack_size))
        level, spatial_dims_parsed, input_nodes = self._prepare_inputs(*input_nodes)
        spatial_dims = spatial_dims_parsed or spatial_dims

        if all(p is None for p in [pad_left, pad_right, pad_top, pad_bottom]):
            pad_bottom = pad_top = pad_left = pad_right = pad_all

        for stride, dilation_r, kernel_s, prod_nc, pad_algo, pad_l, pad_r, pad_t, pad_b, sum_nc,\
                name_pref, name_suff in \
                self._ensure_tuples_and_zip(
                    strides, dilation_rate, kernel_size, prod_num_channels, padding_algorithm, 
                    pad_left, pad_right, pad_top, pad_bottom, sum_num_channels, name_prefixes, 
                    name_suffixes):
            next_node = ConvProd2D(
                *input_nodes, grid_dim_sizes=spatial_dims, pad_bottom=pad_b, pad_top=pad_t,
                pad_left=pad_l, pad_right=pad_r, num_channels=prod_nc, 
                name="{}Prod{}".format(name_pref, name_suff), dilation_rate=dilation_r, 
                kernel_size=kernel_s, padding_algorithm=pad_algo, strides=stride)
            spatial_dims = next_node.output_shape_spatial[:2]
            input_nodes = [next_node]
            print("Built node {}: {} x {} x {}".format(next_node, *next_node.output_shape_spatial))
            self._register_node(next_node, level)
            
            if not use_convsums:
                continue
            
            next_node = ConvSum(*input_nodes, num_channels=sum_nc, grid_dim_sizes=spatial_dims,
                                name="{}Sum{}".format(name_pref, name_suff))
            input_nodes = [next_node]
            print("Built node {}: {} x {} x {}".format(next_node, *next_node.output_shape_spatial))
            self._register_node(next_node, level + 1)
        
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
        if isinstance(x, list):
            return tuple(x)
        return tuple([x] * size) if not isinstance(x, tuple) else x
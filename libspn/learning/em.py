from collections import namedtuple
import tensorflow as tf

from libspn.inference.mpe_path import MPEPath
from libspn.graph.algorithms import traverse_graph
from libspn import conf
from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn import utils


class HardEMLearning:
    """Assembles TF operations performing EM learning of an SPN.

    Args:
        mpe_path (MPEPath): Pre-computed MPE_path.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
    """

    ParamNode = namedtuple("ParamNode", ["node", "name_scope", "accum"])
    LocationScaleLeafNode = namedtuple(
        "LocationScaleLeafNode", ["node", "name_scope", "accum", "sum_data", "sum_data_squared"])

    def __init__(self, root, mpe_path=None, log=True, value_inference_type=None,
                 additive_smoothing=None, initial_accum_value=1.0,
                 use_unweighted=False, sample_winner=False, sample_prob=None,
                 matmul_or_conv=False):
        self._root = root
        self._log = log
        self._additive_smoothing = additive_smoothing
        self._initial_accum_value = initial_accum_value
        self._sample_winner = sample_winner
        # Create internal MPE path generator
        if mpe_path is None:
            self._mpe_path = MPEPath(
                log=log, value_inference_type=value_inference_type,
                use_unweighted=use_unweighted, sample=sample_winner, sample_prob=sample_prob,
                matmul_or_conv=matmul_or_conv)
        else:
            self._mpe_path = mpe_path
        # Create a name scope
        with tf.name_scope("HardEMLearning") as self._name_scope:
            pass
        # Create accumulators
        self._create_accumulators()

    @property
    def mpe_path(self):
        """MPEPath: Computed MPE path."""
        return self._mpe_path

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._mpe_path.value

    @utils.lru_cache
    def reset_accumulators(self):
        with tf.name_scope(self._name_scope):
            return tf.group(*(
                [pn.accum.initializer for pn in self._param_nodes] +
                [dn.accum.initializer for dn in self._loc_scale_leaf_nodes] +
                [dn.sum_data.initializer for dn in self._loc_scale_leaf_nodes] +
                [dn.sum_data_squared.initializer for dn in self._loc_scale_leaf_nodes] +
                [dn.node._total_count_variable.initializer
                 for dn in self._loc_scale_leaf_nodes]),
                            name="reset_accumulators")

    def init_accumulators(self):
        return self.reset_accumulators()

    def accumulate_and_update_weights(self):
        accumulate_updates = self.accumulate_updates()
        with tf.control_dependencies([accumulate_updates]):
            return self.update_spn()

    def accumulate_updates(self):
        # Generate path if not yet generated
        if not self._mpe_path.counts:
            self._mpe_path.get_mpe_path(self._root)

        # Generate all accumulate operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    counts_summed_batch = pn.node._compute_hard_em_update(
                        self._mpe_path.counts[pn.node])
                    assign_ops.append(tf.assign_add(pn.accum, counts_summed_batch))

            for dn in self._loc_scale_leaf_nodes:
                with tf.name_scope(dn.name_scope):
                    counts = self._mpe_path.counts[dn.node]
                    update_value = dn.node._compute_hard_em_update(counts)
                    with tf.control_dependencies(update_value.values()):
                        if dn.node.dimensionality > 1:
                            accum = tf.squeeze(update_value['accum'], axis=-1)
                        else:
                            accum = update_value['accum']
                        assign_ops.extend(
                            [tf.assign_add(dn.accum, accum),
                             tf.assign_add(dn.sum_data, update_value['sum_data']),
                             tf.assign_add(
                                 dn.sum_data_squared, update_value['sum_data_squared'])])

            return tf.group(*assign_ops, name="accumulate_updates")

    def update_spn(self):
        # Generate all update operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    accum = pn.accum
                    if self._additive_smoothing is not None:
                        accum = tf.add(accum, self._additive_smoothing)
                    if pn.node.log:
                        assign_ops.append(pn.node.assign_log(tf.log(accum)))
                    else:
                        assign_ops.append(pn.node.assign(accum))

            for dn in self._loc_scale_leaf_nodes:
                with tf.name_scope(dn.name_scope):
                    assign_ops.extend(dn.node.assign(dn.accum, dn.sum_data, dn.sum_data_squared))

            return tf.group(*assign_ops, name="update_spn")

    def learn(self):
        """Assemble TF operations performing EM learning of the SPN."""
        return None

    def _create_accumulators(self):
        def fun(node):
            if node.is_param:
                with tf.name_scope(node.name) as scope:
                    if self._initial_accum_value is not None:
                        if node.mask and not all(node.mask):
                            accum = tf.Variable(tf.cast(tf.reshape(node.mask,
                                                                   node.variable.shape),
                                                        dtype=conf.dtype) *
                                                self._initial_accum_value,
                                                dtype=conf.dtype)
                        else:
                            accum = tf.Variable(tf.ones_like(node.variable,
                                                             dtype=conf.dtype) *
                                                self._initial_accum_value,
                                                dtype=conf.dtype)
                    else:
                        accum = tf.Variable(tf.zeros_like(node.variable,
                                                          dtype=conf.dtype),
                                            dtype=conf.dtype)
                    param_node = HardEMLearning.ParamNode(node=node, accum=accum,
                                                      name_scope=scope)
                    self._param_nodes.append(param_node)
            if isinstance(node, LocationScaleLeaf) and (node.trainable_scale or node.trainable_loc):
                with tf.name_scope(node.name) as scope:
                    if self._initial_accum_value is not None:
                        accum = tf.Variable(tf.ones_like(node.loc_variable, dtype=conf.dtype) *
                                            self._initial_accum_value,
                                            dtype=conf.dtype)
                        sum_x = tf.Variable(node.loc_variable * self._initial_accum_value,
                                            dtype=conf.dtype)
                        sum_x2 = tf.Variable(tf.square(node.loc_variable) *
                                             self._initial_accum_value,
                                             dtype=conf.dtype)
                    else:
                        accum = tf.Variable(tf.zeros_like(node.loc_variable, dtype=conf.dtype),
                                            dtype=conf.dtype)
                        sum_x = tf.Variable(tf.zeros_like(node.loc_variable), dtype=conf.dtype)
                        sum_x2 = tf.Variable(tf.zeros_like(node.loc_variable), dtype=conf.dtype)
                    loc_scale_node = HardEMLearning.LocationScaleLeafNode(
                        node=node, accum=accum, sum_data=sum_x, sum_data_squared=sum_x2,
                        name_scope=scope)
                    self._loc_scale_leaf_nodes.append(loc_scale_node)

        self._loc_scale_leaf_nodes = []
        self._param_nodes = []
        with tf.name_scope(self._name_scope):
            traverse_graph(self._root, fun=fun)

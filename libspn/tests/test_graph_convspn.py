import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger
from libspn.generation.spatial import ConvSPN

logger = get_logger()


class TestConvSPN(tf.test.TestCase):

    def test_wicker(self):
        rows = 16
        cols = 16
        num_vars = rows * cols
        num_vals = 2
        spatial_dims = [rows, cols]
        ivs = spn.IVs(num_vars=num_vars, num_vals=num_vals)
        dense_generator = spn.DenseSPNGeneratorLayerNodes(
            num_decomps=1, num_mixtures=4, num_subsets=2,
            node_type=spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER)
        convspn = ConvSPN()
        wicker_root = convspn.wicker_stack(
            ivs, stack_size=3, spatial_dims=spatial_dims, dense_generator=dense_generator,
            strides=(1, 2, 2))
        spn.generate_weights(wicker_root)
        init = spn.initialize_weights(wicker_root)

        self.assertTrue(wicker_root.is_valid())
        log_val_op = tf.squeeze(wicker_root.get_log_value())

        with self.test_session() as sess:
            sess.run(init)
            log_val_out = sess.run(
                log_val_op, feed_dict={ivs: -np.ones((1, num_vars), dtype=np.int32)})
        self.assertAllClose(log_val_out, 0.0)


from typing import List
from typing import Optional

import tensorflow as tf


class OnlineExpectationMaximization(tf.keras.optimizers.Optimizer):
    r"""
    Online expectation maximization.

    Requires sum layers to use any of the EM-based :class:`~libspn_keras.SumOpBase` instances, such as
    :class:`~libspn_keras.SumOpEMBackprop` :class:`~libspn_keras.SumOpHardEMBackprop`.

    Args:
        learning_rate: Learning rate for EM. If learning rate is :math:`\eta`, then updates are given by:
            :math:`w \leftarrow (1-\eta)w + \eta \Delta w`
        accumulate_batches: The number of batches to accumulate gradients before applying updates.
        name: Name of the optimizer
        kwargs: Remaining kwargs to pass to :class:`~tensorflow.keras.optimizers.Optimizer` superclass
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate: float = 0.01,
        accumulate_batches: int = 1,
        name: str = "OnlineEM",
        **kwargs
    ):
        super(OnlineExpectationMaximization, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        self._accumulate_batches = accumulate_batches
        self._set_hyper("accumulate_batches", accumulate_batches)

    def _create_slots(self, var_list: List[tf.Variable]) -> None:
        if self._accumulate_batches > 1:
            for var in var_list:
                self.add_slot(var, "multi_batch_accumulates")

    def _resource_apply_dense(
        self, grad: tf.Tensor, var: tf.Variable, apply_state: Optional[dict] = None
    ) -> tf.Tensor:
        updated_var = self._get_multi_batch_update(grad, var, apply_state)
        return var.assign(updated_var)

    def _resource_apply_sparse(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        indices: tf.Tensor,
        apply_state: Optional[dict] = None,
    ) -> tf.Tensor:
        updated_var = self._get_multi_batch_update(grad, var, apply_state)
        return tf.tensor_scatter_nd_update(var, indices, updated_var)

    def _get_multi_batch_update(
        self, grad: tf.Tensor, var: tf.Variable, apply_state: Optional[dict],
    ) -> tf.Tensor:
        if self._accumulate_batches == 1:
            return self._get_updated_var(var, grad, apply_state)

        multi_batch_accumulates = self.get_slot(var, "multi_batch_accumulates")
        multi_batch_updated = multi_batch_accumulates.assign_add(grad)

        must_update = tf.equal(
            _mod_proxy(
                self.iterations + 1, tf.convert_to_tensor(self._accumulate_batches)
            ),
            0,
        )
        updated_var = tf.cond(
            must_update,
            lambda: self._get_updated_var(
                var, multi_batch_updated / self._accumulate_batches, apply_state
            ),
            lambda: var,
        )
        updated_batch_accumulates = tf.cond(
            must_update,
            lambda: tf.zeros_like(multi_batch_accumulates),
            lambda: multi_batch_accumulates,
        )
        multi_batch_accumulates.assign(updated_batch_accumulates)
        return updated_var

    def _get_updated_var_sparse(
        self, var: tf.Variable, grad: tf.Tensor, indices: tf.Tensor, apply_state: dict
    ) -> tf.Tensor:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        updated_var = (1.0 - coefficients["lr_t"]) * tf.gather_nd(
            var, indices
        ) + coefficients["lr_t"] * tf.negative(grad)
        return updated_var

    def _get_updated_var(
        self, var: tf.Variable, grad: tf.Tensor, apply_state: Optional[dict]
    ) -> tf.Tensor:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        updated_var = (1.0 - coefficients["lr_t"]) * var + coefficients[
            "lr_t"
        ] * tf.negative(grad)
        return updated_var

    def get_config(self) -> dict:
        """
        Return the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Returns:
            Python dictionary.
        """
        config = super(OnlineExpectationMaximization, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "accumulate_batches": self._serialize_hyperparameter(
                    "accumulate_batches"
                ),
            }
        )
        return config


def _mod_proxy(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return x - tf.math.floordiv(x, y) * y

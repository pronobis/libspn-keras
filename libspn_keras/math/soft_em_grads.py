from typing import Any, Callable, Tuple

import tensorflow as tf
import tensorflow_probability as tfp


class LocationScaleEMGradWrapper:
    """
    Wraps a location-scale distribution.

    As a consequence gradients that are passed down to the locations and scales result in EM updates.

    Args:
        location_scale_distribution: Distribution to wrap.
        first_order_moment_denom_accum: Denominator accumulator of first order moment.
        first_order_moment_num_accum: Numerator accumulator of first order moment.
        second_order_moment_denom_accum: Denominator accumulator of second order moment.
        second_order_moment_num_accum: Numerator accumulator of first order moment.
    """

    def __init__(
        self,
        location_scale_distribution: tfp.distributions.Distribution,
        first_order_moment_denom_accum: tf.Tensor,
        first_order_moment_num_accum: tf.Tensor,
        second_order_moment_denom_accum: tf.Tensor,
        second_order_moment_num_accum: tf.Tensor,
    ):
        self.location_scale_distribution = location_scale_distribution
        self.first_order_moment_denom_accum = first_order_moment_denom_accum
        self.first_order_moment_num_accum = first_order_moment_num_accum
        self.second_order_moment_denom_accum = second_order_moment_denom_accum
        self.second_order_moment_num_accum = second_order_moment_num_accum

    def __getattr__(self, attr: Any) -> Any:
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recursion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)

        # proxy to the wrapped object
        return getattr(self.location_scale_distribution, attr)

    def log_prob(
        self, x: tf.Tensor
    ) -> Tuple[
        tf.Tensor,
        Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]],
    ]:
        """
        Compute log probability of the distribution at x.

        Args:
            x: The raw input data.

        Returns:
            A Tensor with the log probablity.
        """

        @tf.custom_gradient
        def _inner(
            first_order_moment_denom_accum: tf.Tensor,
            first_order_moment_num_accum: tf.Tensor,
            second_order_moment_denom_accum: tf.Tensor,
            second_order_moment_num_accum: tf.Tensor,
        ) -> Tuple[
            tf.Tensor,
            Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]],
        ]:
            def grad(
                dy: tf.Tensor,
            ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
                denom_grad = tf.reduce_sum(dy, axis=0, keepdims=True)
                first_order_moment_num_grad = tf.reduce_sum(
                    x * dy, axis=0, keepdims=True
                )
                second_order_moment_num_grad = tf.reduce_sum(
                    tf.square(x) * dy, axis=0, keepdims=True
                )

                return (
                    denom_grad,
                    first_order_moment_num_grad,
                    denom_grad,
                    second_order_moment_num_grad,
                )

            out = self.location_scale_distribution.log_prob(x)

            return out, grad

        return _inner(
            self.first_order_moment_denom_accum,
            self.first_order_moment_num_accum,
            self.second_order_moment_denom_accum,
            self.second_order_moment_num_accum,
        )


class LocationEMGradWrapper:
    """
    Wraps a location-scale distribution.

    As a consequence gradients that are passed down to the locations result in EM updates.

    Args:
        location_scale_distribution: Distribution to wrap.
        first_order_moment_denom_accum: Denominator accumulator of first order moment.
        first_order_moment_num_accum: Numerator accumulator of first order moment.
    """

    def __init__(
        self,
        location_scale_distribution: tfp.distributions.Distribution,
        first_order_moment_denom_accum: tf.Tensor,
        first_order_moment_num_accum: tf.Tensor,
    ):
        self.location_scale_distribution = location_scale_distribution
        self.first_order_moment_denom_accum = first_order_moment_denom_accum
        self.first_order_moment_num_accum = first_order_moment_num_accum

    def __getattr__(self, attr: Any) -> Any:
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recursion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)

        # proxy to the wrapped object
        return getattr(self.location_scale_distribution, attr)

    def log_prob(
        self, x: tf.Tensor
    ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
        """
        Compute log probabilty of the distribution at x.

        Args:
            x: The raw input data.

        Returns:
            A Tensor with the log probablity.
        """

        @tf.custom_gradient
        def _inner(
            first_order_moment_denom_accum: tf.Tensor,
            first_order_moment_num_accum: tf.Tensor,
        ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
            def grad(dy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                denom_grad = tf.reduce_sum(dy, axis=0, keepdims=True)
                first_order_moment_num_grad = tf.reduce_sum(
                    x * dy, axis=0, keepdims=True
                )

                return denom_grad, first_order_moment_num_grad

            out = self.location_scale_distribution.log_prob(x)

            return out, grad

        return _inner(
            self.first_order_moment_denom_accum, self.first_order_moment_num_accum
        )

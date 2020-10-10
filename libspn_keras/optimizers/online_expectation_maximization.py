from tensorflow import keras


class OnlineExpectationMaximization(keras.optimizers.SGD):
    """
    Online expectation maximization which requires sum layers to use any of the EM-based SumOpBase instances.

    Internally, this is just an SGD optimizer with unit learning rate.
    """

    def __init__(
        self, learning_rate: float = 0.05, gliding_average: bool = True, **kwargs
    ):
        if gliding_average:
            learning_rate = -learning_rate / (learning_rate - 1.0)
        super().__init__(learning_rate, **kwargs)

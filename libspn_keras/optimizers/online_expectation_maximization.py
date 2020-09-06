from tensorflow import keras


class OnlineExpectationMaximization(keras.optimizers.SGD):
    """
    Online expectation maximization which requires sum layers to use any of the EM-based SumOpBase instances.

    Internally, this is just an SGD optimizer with unit learning rate.
    """

    def __init__(self):
        super(OnlineExpectationMaximization, self).__init__(learning_rate=1.0)

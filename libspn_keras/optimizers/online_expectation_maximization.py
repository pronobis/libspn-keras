from tensorflow import keras


class OnlineExpectationMaximization(keras.optimizers.SGD):
    """
    Online expectation maximization which requires sum layers to have backprop_mode == EM or
    backprop_mode == HARD_EM. Under the hood, it uses SGD with learning_rate == 1.0.
    """

    def __init__(self):
        super(OnlineExpectationMaximization, self).__init__(learning_rate=1.0)

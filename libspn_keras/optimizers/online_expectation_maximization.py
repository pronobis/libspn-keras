from tensorflow import keras


class OnlineExpectationMaximization(keras.optimizers.SGD):
    """
    Online expectation maximization which requires sum layers to have an any of the EM-based
    ``backprop_mode`` s.
    """

    def __init__(self):
        super(OnlineExpectationMaximization, self).__init__()

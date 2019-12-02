from tensorflow import keras


class OnlineExpectationMaximization(keras.optimizers.SGD):

    def __init__(self):
        super(OnlineExpectationMaximization, self).__init__(learning_rate=1.0)

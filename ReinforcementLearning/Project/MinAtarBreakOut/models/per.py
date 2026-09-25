import models.common as common


def build_model(action_size, input_shape=(10, 10, 4)):
    return common.breakout_cnn(action_size, input_shape, "per_dqn")

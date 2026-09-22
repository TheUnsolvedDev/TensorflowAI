import models.common as common


def build_model(action_size, input_shape=(84, 84, 4)):
    return common.pong_cnn(action_size, input_shape, "double_dqn")

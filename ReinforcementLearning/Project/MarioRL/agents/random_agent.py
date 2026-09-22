class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()

    def act_batch(self, states):
        return [self.act(state) for state in states]

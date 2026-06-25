class DQNWrapper:
    def __init__(self, agent):
        self.agent = agent

    def act(self, obs, mask, player):
        return self.agent.act(obs, mask, eps=0.0)
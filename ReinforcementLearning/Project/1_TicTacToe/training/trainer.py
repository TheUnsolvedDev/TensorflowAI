import tqdm

class Trainer:
    def __init__(self, env, agent, opponent, self_play=False):
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.self_play = self_play

    def train(self, episodes=5000):
        pbar = tqdm.tqdm(range(episodes))

        for ep in pbar:
            obs, _ = self.env.reset()
            done = False

            while not done:
                player = self.env.current_player

                if player == 1:
                    acting_agent = self.agent
                else:
                    acting_agent = self.opponent

                action = acting_agent.act(obs, player)
                next_obs, reward, done, _, _ = self.env.step(action)

                if player == 1:
                    self.agent.store_reward(reward)
                    if self.self_play:
                        self.opponent.store_reward(-reward)
                else:
                    if self.self_play:
                        self.opponent.store_reward(reward)
                    self.agent.store_reward(-reward)

                obs = next_obs

            # learning phase
            loss_agent = self.agent.learn()

            if self.self_play:
                loss_opponent = self.opponent.learn()
                pbar.set_postfix(agent=f"{loss_agent:.4f}",
                                 opponent=f"{loss_opponent:.4f}")
            else:
                pbar.set_postfix(agent=f"{loss_agent:.4f}")

        return self.agent
from env.wrappers import training_env


def test_agent(agent, episodes=10, env_factory=None, epsilon=0.0):
    """Run an agent in a visible Atari environment."""
    env = env_factory() if env_factory else training_env(render_mode="human")
    rewards = []
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = agent.act(state, epsilon)
                if hasattr(action, "numpy"):
                    action = action.numpy().item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            print(f"episode={episode + 1} reward={total_reward}")
            rewards.append(total_reward)
    finally:
        env.close()
    average_reward = sum(rewards) / len(rewards)
    print(f"average reward={average_reward}")
    return average_reward

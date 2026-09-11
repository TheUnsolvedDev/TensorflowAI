import gymnasium as gym
from agent import REINFORCEAgent
from config import *
def simulate(weights_path,episodes=5):
    env=gym.make(ENV_NAME,render_mode="human"); agent=REINFORCEAgent(); agent.load(weights_path)
    for ep in range(episodes):
        obs,_=env.reset(); done=False; total=0
        while not done: obs,r,t,tr,_=env.step(agent.get_action(obs)); done=t or tr; total+=r
        print(f"Episode {ep} | Reward {total}")
    env.close()
if __name__=="__main__": simulate("checkpoints/policy.weights.h5")


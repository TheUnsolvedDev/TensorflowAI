import gymnasium as gym
from config import *
class Environment:
    def __init__(self,name=ENV_NAME,seed=42,render_mode=None): self.env=gym.make(name,render_mode=render_mode); self.env.reset(seed=seed)
    def reset(self): return self.env.reset()[0]
    def step(self,action):
        obs,r,t,tr,i=self.env.step(action); return obs,r,t or tr,i
    def render(self): self.env.render()


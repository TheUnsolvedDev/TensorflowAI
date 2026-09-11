import numpy as np
from config import GAMMA
class TrajectoryBuffer:
    def __init__(self): self.obs=[]; self.actions=[]; self.rewards=[]
    def store(self,obs,action,reward): self.obs.append(obs); self.actions.append(action); self.rewards.append(reward)
    def get(self):
        out=[]; value=0
        for r in self.rewards[::-1]: value=r+GAMMA*value; out.append(value)
        return np.asarray(self.obs,np.float32),np.asarray(self.actions,np.int32),np.asarray(out[::-1],np.float32)


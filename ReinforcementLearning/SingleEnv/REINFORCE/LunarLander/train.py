import os,datetime
import tensorflow as tf
import gymnasium as gym
import tqdm
from agent import REINFORCEAgent
from trajectory_buffer import TrajectoryBuffer
from config import *
os.makedirs("logs",exist_ok=True); os.makedirs("videos",exist_ok=True)
def train():
    agent=REINFORCEAgent(); os.makedirs("checkpoints",exist_ok=True); writer=tf.summary.create_file_writer("logs/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")); best=-float("inf")
    for ep in tqdm.tqdm(range(EPOCHS),desc="Training Episodes"):
        env=gym.make(ENV_NAME); obs,_=env.reset(); buf=TrajectoryBuffer(); done=False; total=0
        while not done:
            a=agent.get_action(obs); nxt,r,t,tr,_=env.step(a); done=t or tr; buf.store(obs,a,r); obs=nxt; total+=r
        o,a,g=buf.get(); agent.update(tf.convert_to_tensor(o),tf.convert_to_tensor(a),tf.convert_to_tensor(g))
        if total>best: best=total; agent.save("checkpoints/best.weights.h5")
        with writer.as_default(): tf.summary.scalar("reward",total,ep); tf.summary.scalar("best_reward",best,ep)
        env.close()
if __name__=="__main__": train()


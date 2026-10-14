# Gymnasium 2048

A small [Gymnasium](https://gymnasium.farama.org/) environment for 2048.

## Use

```bash
pip install -r requirements.txt
python -m unittest -v
python main.py --episodes 3 --seed 42
python main.py --episodes 10 --num-envs 8 --seed 42
python main.py --episodes 1 --seed 42 --render --fps 8
```

```python
import gymnasium as gym
import env  # registers Game2048-v0

env = gym.make("Game2048-v0", render_mode="ansi")
observation, info = env.reset(seed=42)

while True:
    action = env.action_space.sample()  # 0=up, 1=right, 2=down, 3=left
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.render())
    if terminated or truncated:
        break
```

The random agent selects only legal actions using the `action_mask` returned in
`info`. Use `--num-envs` to run environments concurrently with Gymnasium's
`AsyncVectorEnv`. Each worker runs the requested number of episodes. Add
`--render` for the colored Pygame visualization and control its speed with
`--fps`; visualization requires `--num-envs 1`.

The observation is a `4 x 4` `int32` board. A move's reward is the sum of the
tiles merged by that move. Invalid moves return zero reward and do not spawn a
new tile. `info` contains the total `score`, current `max_tile`, and whether the
last action `moved` a tile.

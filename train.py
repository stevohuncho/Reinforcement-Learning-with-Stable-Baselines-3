import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3 import DQN
import os
from constants import models_dir

# config
TIMESTEPS = 10000

# frozen lake environoment
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True, render_mode="ansi")
env.reset()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
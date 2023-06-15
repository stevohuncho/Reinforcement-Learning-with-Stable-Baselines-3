import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3 import DQN

# frozen lake environoment
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True, render_mode="ansi")
env.reset()

# model
models_dir = "models/DQN"
model_path = f"{models_dir}/test"
model = DQN.load(model_path, env=env)

# config
episdoes = 5

for ep in range(episdoes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)

env.close()
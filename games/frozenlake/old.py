"""import gymnasium as gym
import sys
import numpy as np
from tabulate import tabulate
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

np.set_printoptions(threshold=sys.maxsize)

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array", max_episode_steps=30, desc=maps[1])

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="logs")
# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e5), progress_bar=True, log_interval=1)
# Save the agent
model.save("DQN_frozen_lake")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("DQN_frozen_lake", env=env, print_system_info=True)
model = DQN.load("DQN_frozen_lake", env=env, tensorboard_log="logs")

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic=True)
print(mean_reward, std_reward)

# Enjoy trained agent

vec_env = model.get_env()
obs = vec_env.reset()
episode = 0
step = 0
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    time.sleep(0.1)

"""
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

def frozen_lake_env(rank: int, map: list[str], seed: int = 0):
    def _init():
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array", desc=map)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def frozen_lake_dummy_vec_env(num_cpu: int, map: list[str]):
    vec_env_list = []
    for i in range(num_cpu):
        vec_env_list.append(frozen_lake_env(i, map))
    return DummyVecEnv(vec_env_list)
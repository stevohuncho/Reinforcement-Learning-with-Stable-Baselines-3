import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from games.frozenlake.env import frozen_lake_env, frozen_lake_dummy_vec_env

class FrozenLake:
    def __init__(self, num_envs: int, map: list[str]):
        self.num_envs: int = num_envs
        self.map: list[str] = map
      
        if self.num_envs > 1:
            self.env = frozen_lake_dummy_vec_env(self.num_envs, self.map)
            self.is_vec_env = True
        else:
            self.env = frozen_lake_env(0, self.map)()
            self.is_vec_env = False

    def get_env(self):
        return self.env

    def eval_callback(self, freq: int = 1e4, eval_eps: int = 5, log_path: str = None) -> EvalCallback:
        return EvalCallback(
            eval_env=self.env,
            eval_freq=freq//self.num_envs,
            n_eval_episodes=eval_eps,
            log_path=log_path,
            best_model_save_path=log_path
        )
        
    def set_model(self, model: DQN):
        self.model = model

    def train(self, steps, eval_callback = None, log_name: str = "DQN"):
        if hasattr(self, "model"):
            self.model: DQN = self.model.learn(
                total_timesteps=steps, 
                callback=eval_callback, 
                log_interval=1,
                progress_bar=True,
                tb_log_name=log_name
            )

    def save_model(self, path: str):
        pass


    def evaluate(self, num_episodes=100):
        if self.is_vec_env:
            pass
        else:
            env = self.model.get_env()
            all_episode_rewards = []
            for i in range(num_episodes):
                episode_rewards = []
                done = False
                obs = env.reset()
                while not done:
                    action, _states = self.model.predict(obs)
                    obs, reward, done, info = env.step(action)
                    episode_rewards.append(reward)
                all_episode_rewards.append(sum(episode_rewards))
            mean_episode_reward = np.mean(all_episode_rewards)
            print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
            return mean_episode_reward

    def print(self, ):

        pass

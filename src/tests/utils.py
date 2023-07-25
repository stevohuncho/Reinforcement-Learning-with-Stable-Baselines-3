from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from typing import Optional, Union
import gymnasium as gym

def visually_test_model(model: DQN , env: VecEnv, steps: int = 1e5):
    obs = env.reset()
    for i in range(int(steps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render("human")

def eval_model(model: DQN , env: VecEnv, eps: int = 100):
    total_rew = 0
    completed_eps = 0

    obs = env.reset()
    while completed_eps < eps:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones[0]:
            steps_taken = info[0]['terminal_observation']
            total_rew += rewards[0]
            completed_eps += 1
            print(f'\r  Episode #{completed_eps} Completed. {steps_taken} Steps Taken. +{rewards[0]} Reward.', end="\r")
        env.render("human")  
    print(f"\n{'{:0.2f}'.format(total_rew/float(eps) * 100)}% Eval Success Rate!")

class CustomEvalCallback():
    def __init__(
        self, 
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ) -> None:
        self.callback_on_new_best = callback_on_new_best
        self.callback_after_eval = callback_after_eval
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.render = render
        self.verbose = verbose
        self.warn = warn

    def create(
        self,
        env: Union[gym.Env, VecEnv],
    ) -> EvalCallback:
        EvalCallback(
            env,
            self.callback_on_new_best,
            self.callback_after_eval,
            self.n_eval_episodes,
            self.eval_freq,
            self.log_path,
            self.best_model_save_path,
            self.deterministic,
            self.render,
            self.verbose,
            self.warn,
        )
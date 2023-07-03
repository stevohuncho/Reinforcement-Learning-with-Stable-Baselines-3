'''
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback


def frozen_lake_dqn_model(env, tlog, learning_rate):
    return DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=tlog, 
        learning_rate=learning_rate,
    )

def frozen_lake_eval_callback(env, freq, eps, log):
    return EvalCallback(
        eval_env=env,
        eval_freq=freq,
        n_eval_episodes=eps,
        log_path=log
    )
'''
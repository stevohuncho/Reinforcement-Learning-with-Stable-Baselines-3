from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

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
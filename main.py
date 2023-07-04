from stable_baselines3 import DQN
from games.frozenlake.frozenlake import FrozenLake
from stable_baselines3.common.monitor import Monitor
from algos.dqn.reward_shaped_dqn import RewardShapedDQN
from algos.dqn.create_dqn_reward_shaper import create_dqn_reward_shaper
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from games.frozenlake.env import frozen_lake_env
from datetime import datetime
from tabulate import tabulate
import gymnasium
import time
import os
import math

def train_frozenlake(model: DQN = None, n_envs = 1, map: list[str] = None, steps = 1e5, name: str = ""):
    log_path_prefix = f'./logs/frozenlake_DQN_{datetime.now().strftime("[%m-%d-%Y]_[%H-%M-%S]")}'
    os.makedirs(log_path_prefix, exist_ok=True)
    os.makedirs(f'{log_path_prefix}/eval', exist_ok=True)
    if map == None:
        map = generate_random_map(4) 
    fl = FrozenLake(num_envs=n_envs, map=map)
    eval_callback = fl.eval_callback(log_path=f'{log_path_prefix}/eval')
    if model == None:
        model = DQN(
            "MlpPolicy", 
            fl.get_env(), 
            verbose=1, 
            tensorboard_log=f'{log_path_prefix}',
        )
    else:
        model.tensorboard_log = f'{log_path_prefix}'
        fl.set_model(model.set_env(fl.get_env()))
    fl.set_model(model)
    fl.train(steps=steps, eval_callback=eval_callback, log_name=name)
    print(map)

'''
def load_frozenlake(fl: FrozenLake):
    # select a model
    print('[Frozenlake Model Loader]')
    frozenlake_model_dirs: dict[str][list[str]] = {'Frozenlake Models (using best_model.zip)': []}
    for file in os.listdir('logs'):
        d = os.path.join('logs', file, 'eval', 'best_model.zip')
        if os.path.isfile(d):
            frozenlake_model_dirs['Frozenlake Models (using best_model.zip)'].append(file)
    print(tabulate(frozenlake_model_dirs, tablefmt='fancy_grid', showindex=True, headers='keys'))
    idx = -1
    while idx < 0 or idx > len(frozenlake_model_dirs):
        idx = int(input('Select A Frozenlake Model By Entering Its Index: '))

    # load selected model
    model = DQN.load(
        path=os.path.join('logs', frozenlake_model_dirs["Frozenlake Models (using best_model.zip)"][idx], 'eval', 'best_model.zip'),
    )

    # make a monitor for the target env
    target_monitor = Monitor(fl.get_env(), 'logs')
    target_model = model.load(os.path.join('logs', frozenlake_model_dirs["Frozenlake Models (using best_model.zip)"][idx], 'eval', 'best_model.zip'))
    target_model.env = frozen_lake_env(0, map=generate_random_map(int(math.sqrt(int(str(model.observation_space).replace('Discrete(', '').replace(')', ''))))))()
    target_model.set_env(target_monitor)
    target_model.learn(1e5)

    reward_shaper = create_dqn_reward_shaper(source_model=model, num_sampling_episodes=50)
    target_reward_reshaping_model = RewardShapedDQN('MlpPolicy', env=target_monitor, verbose=2, reward_shaper=reward_shaper)

    print(f'Selected {frozenlake_model_dirs["Frozenlake Models (using best_model.zip)"][idx]}.')
    time.sleep(3)

    #reshape
'''

def main():
    map: list[str] = ['SFFF', 'FFFH', 'FFFF', 'FHFG']
    '''
    vector env testing
    train_frozenlake(map=map, name="1")
    train_frozenlake(map=map, n_envs=3, name="3")
    train_frozenlake(map=map, n_envs=6, name="6")
    train_frozenlake(map=map, n_envs=9, name="9")
    '''
    '''
    model_1 = DQN.load('./frozenlake_DQN_1/eval/best_model')
    train_frozenlake(map=map,model=model_1,n_envs=1,name='1_trained')
    model_3 = DQN.load('./frozenlake_DQN_3/eval/best_model')
    train_frozenlake(map=map, model=model_3,n_envs=3,name='3_trained')
    '''




if __name__ == "__main__":
    main()
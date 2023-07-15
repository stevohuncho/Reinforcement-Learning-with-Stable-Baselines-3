from stable_baselines3 import DQN
from games.frozenlake.frozenlake import FrozenLake
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from games.frozenlake.env import frozen_lake_dummy_vec_env
from datetime import datetime
from transfer_learning.map_resizer import MapResizer
import time
import os


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
            batch_size=64
        )
    else:
        model.tensorboard_log = f'{log_path_prefix}'
        fl.set_model(model.set_env(fl.get_env()))
    fl.set_model(model)
    fl.train(steps=steps, eval_callback=eval_callback, log_name=name)
    

def main():
    map: list[str] = ['SFFF', 'FFFH', 'FFFF', 'FHFG']
    map8x8: list[str] = ['FFFFHHFF', 'FFHFFHFF', 'HFFFFFFF', 'HHFFFFHH', 'GFFFFFFF', 'HHFFFFFF', 'HHFFFFSH', 'HHFFFFHH']


    env_4x4 = frozen_lake_dummy_vec_env(1, map=map)
    model_4x4 = DQN.load('./frozenlake_DQN_1/eval/best_model', env=env_4x4)
    model_4x4.replay_buffer.add()
    obs = env_4x4.reset()
    done = False
    while not done:
        action, _states = model_4x4.predict(obs)
        obs, reward, done, info = env_4x4.step(action)
        env_4x4.render('human')
        print(obs, reward, done, info)
        time.sleep(2)





if __name__ == "__main__":
    main()
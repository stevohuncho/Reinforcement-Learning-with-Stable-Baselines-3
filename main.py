from stable_baselines3 import DQN
from games.frozenlake.frozenlake import FrozenLake
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from datetime import datetime
import os

def main():
    log_path_prefix = f'./logs/frozenlake_DQN_{datetime.now().strftime("[%m-%d-%Y]_[%H-%M-%S]")}'
    os.makedirs(log_path_prefix, exist_ok=True)
    os.makedirs(f'{log_path_prefix}/eval', exist_ok=True)
    map = generate_random_map(4) 
    fl = FrozenLake(1, map)
    eval_callback = fl.eval_callback(log_path=f'{log_path_prefix}/eval')
    model = DQN(
        "MlpPolicy", 
        fl.get_env(), 
        verbose=1, 
        tensorboard_log=f'{log_path_prefix}',
        
    )
    fl.set_model(model)
    fl.train(1e5, "tb_data", eval_callback=eval_callback)
    print(map)




if __name__ == "__main__":
    main()
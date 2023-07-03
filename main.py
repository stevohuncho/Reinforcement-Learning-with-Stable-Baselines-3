from stable_baselines3 import DQN
from games.frozenlake.frozenlake import FrozenLake
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from datetime import datetime

def main():
    map = generate_random_map(4) 
    fl = FrozenLake(1, map)
    model = DQN(
        "MlpPolicy", 
        fl.get_env(), 
        verbose=1, 
    )
    fl.set_model(model)
    fl.train(1e5, f'frozenlake_DQN_{datetime.now().strftime("[%m-%d-%Y]_[%H-%M-%S]")}')




if __name__ == "__main__":
    main()
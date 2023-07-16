from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3 import DQN
from gymnasium.spaces import Discrete
from games.frozenlake.env import frozen_lake_dummy_vec_env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from transfer_learning.map_resizer import MapResizer
import math, time
import numpy as np
from numpy import ndarray
from tabulate import tabulate

def get_map_from_rgb_array(array: ndarray) -> tuple[list[str], int] | None:
    size = len(array)/64
    if not size.is_integer() or len(array) != len(array[0]):
        return None
    size = int(size)

    map = [''] * size
    tile_set = {
        '[255 194 161]': 'S',
        '[204 230 255]': 'F',
        '[156 247 255]': 'H',
        '[240 181  65]': 'G'
    }

    tiles = ndarray(shape=(size, size, 64, 64, 3), dtype=np.uint8)
    for i, line in enumerate(array):
        for j, pixel in enumerate(line):
           tiles[i//64][j//64][i%64][j%64] = pixel
    for i, row in enumerate(tiles):
        for tile in row:
            map[i] += tile_set[str(tile[32][32])]
    return map, size

def get_size_from_model(predict_model: DQN) -> int:
    return int(math.sqrt([int(i) for i in ''.join(' '.join(str(predict_model.observation_space).split('(')).split(')')).split() if i.isdigit()][0]))

def get_prediction(predict_model: DQN, map: list[str]) -> ndarray:
    env = frozen_lake_dummy_vec_env(1, map)
    obs = env.reset()
    predict_model.env = env
    action, _states = predict_model.predict(obs)
    next_obs, reward, done, info = predict_model.get_env().step(action)
    predict_model.replay_buffer.add(obs, next_obs, action, reward, done, info)
    return action


def frozen_lake_transfer_learn(predict_model: DQN, training_model: DQN, steps: int = 1e5):
    # prepare training model
    training_model_env = training_model.get_env()
    obs = training_model_env.reset()
    training_model_map, training_model_map_size = get_map_from_rgb_array(training_model_env.render())
    print("TRAINING MAP")
    print(f'- {training_model_map_size}x{training_model_map_size} Grid')
    print(tabulate(training_model_map, tablefmt="rounded_grid"))

    # prepare prediction model
    predict_model_size = get_size_from_model(predict_model)
    print("PREDICT MODEL")
    print(f'- {predict_model_size}x{predict_model_size} Grid')

    # prepare map resizer
    map_resizer = MapResizer(training_model_map, predict_model_size)
    map_resizer.set_start(obs)

    
    for i in range(int(steps)):
        #print(tabulate(map_resizer.convert_map(), tablefmt="rounded_grid"))
        action = get_prediction(predict_model, map_resizer.convert_map())
        next_obs, reward, done, info = training_model_env.step(action)
        training_model.replay_buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs
        map_resizer.set_start(obs)
        if int(reward[0]) == 1:
            print("LFG")
        print(i)
        




    
    
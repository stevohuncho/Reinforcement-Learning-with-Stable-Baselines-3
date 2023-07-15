from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3 import DQN
from gymnasium.spaces import Discrete
from PIL import Image
import math, time
import numpy as np
from numpy import ndarray

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


def frozen_lake_transfer_learn(predict_model: DQN, env: GymEnv):
    rendered_env = env.render()
    map, size = get_map_from_rgb_array(rendered_env)

    
    
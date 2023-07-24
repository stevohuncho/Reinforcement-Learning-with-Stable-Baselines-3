from stable_baselines3 import DQN
import math
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
from envs.frozen_lake.frozen_lake import FrozenLakeEnv, generate_random_map
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from transfer_learning.AlternatingTransferDQN.dqn import AlternatingTransferDQN
from transfer_learning.BasicTransferDQN.dqn import BasicTransferDQN
from envs.frozen_lake.frozen_lake import FrozenLakeEnv
from logger import Logger
from constants import NON_TRANSERED_PATH
from tests.utils import CustomEvalCallback
from typing import Literal, List
from matplot import plot_eval_rewards_tl
import os

def tl_vs_no_tl(
    pretrained_model: DQN,
    transfered_dir: str,
    non_transfered_dir: str,
    steps: int = 1e5, 
    pretrain_map_size: int = 4, 
    transfer_map_size: int = 8, 
    map_p: float = 0.8,
    transfer_method: Literal["Basic", "Alternating"] = "Alternating",
    is_slippery: bool = True,
    fps: int = 4,
    reward_range: tuple = (0, 1),
    goal_reward: int = 1,
    frozen_tile_reward: int = 0,
    hole_reward: int = 0,
    map_pretrain: List[str] = None,
    map_transfer: List[str] = None,
):

    # init maps
    if map_pretrain == None:
        map_pretrain = generate_random_map(size=pretrain_map_size, p=map_p)
    if map_transfer == None:
        map_transfer = generate_random_map(size=transfer_map_size, p=map_p)
    

    # train non non-transfer learned model
    non_transfered_env = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_transfer,
        is_slippery = is_slippery,
        fps = fps,
        reward_range = reward_range,
        goal_reward = goal_reward,
        frozen_tile_reward = frozen_tile_reward,
        hole_reward = hole_reward
    ).dummy_vec_env(1)

    non_transfered_eval_callback = CustomEvalCallback(n_eval_episodes=20).create(
        non_transfered_env,
        non_transfered_dir,
        non_transfered_dir
    )

    non_transfered_model = DQN(
        policy="MlpPolicy",
        env=non_transfered_env,
    )

    non_transfered_model = non_transfered_model.learn(
        steps, 
        progress_bar=True, 
        log_interval=1,
        callback=non_transfered_eval_callback
    )

    transfered_env = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_transfer,
        is_slippery = is_slippery,
        fps = fps,
        reward_range = reward_range,
        goal_reward = goal_reward,
        frozen_tile_reward = frozen_tile_reward,
        hole_reward = hole_reward
    ).dummy_vec_env(1)

    if transfer_method == "Basic":
        transfered_model = BasicTransferDQN(
            pretrained_model=pretrained_model,
            policy="MlpPolicy",
            env=transfered_env,
        )
    else:
        transfered_model = AlternatingTransferDQN(
            pretrained_model=pretrained_model,
            policy="MlpPolicy",
            env=transfered_env,
        )

    transfered_eval_callback = CustomEvalCallback(n_eval_episodes=20).create(
        transfered_env,
        transfered_dir,
        transfered_dir
    )

    transfered_model = transfered_model.learn(
        steps, 
        log_interval=1, 
        progress_bar=True,
        callback=transfered_eval_callback,
    )

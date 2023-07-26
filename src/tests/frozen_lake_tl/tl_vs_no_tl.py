from tests.frozen_lake_tl.alternating_transfer_dqn_test import alternating_transfer_dqn_test
from tests.frozen_lake_tl.basic_transfer_dqn_test import basic_transfer_dqn_test
from envs.frozen_lake.frozen_lake import FrozenLakeEnv, generate_random_map
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from envs.frozen_lake.frozen_lake import FrozenLakeEnv
from logger import Logger
from constants import NON_TRANSERED_PATH
from tests.utils import CustomEvalCallback
from typing import Literal, List
from matplot import plot_eval_rewards_tl
import os

def tl_vs_no_tl(
    steps: int = 1e5, 
    pretrain_steps: int = 1e5,
    pretrain_map_size: int = 4, 
    transfer_map_size: int = 8, 
    map_p: float = 0.8,
    transfer_method: Literal["Basic", "Alternating"] = "Basic",
    pretrained_model: DQN = None,
    is_slippery: bool = True,
    fps: int = 4,
    reward_range: tuple = (0, 1),
    goal_reward: int = 1,
    frozen_tile_reward: int = 0,
    hole_reward: int = 0,
    map_pretrain: List[str] = None,
    map_transfer: List[str] = None,
    map_transfer_test: List[str] = None,
    non_transfered_eval_callback: CustomEvalCallback = None,
    pretrained_eval_callback: CustomEvalCallback = None,
    transfered_eval_callback: CustomEvalCallback = None,
    name: str = "",
):
    logger = Logger(name)

    # init maps
    if map_pretrain == None:
        map_pretrain = generate_random_map(size=pretrain_map_size, p=map_p)
    if map_transfer == None:
        map_transfer = generate_random_map(size=transfer_map_size, p=map_p)
    if map_transfer_test == None:
        map_transfer_test = generate_random_map(size=transfer_map_size, p=map_p)
    

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

    non_transfer_tf_log_path = logger.tf_logger("nonTransferedFrozenLakeDQN")
    if non_transfered_eval_callback != None:
        os.makedirs(f'{non_transfer_tf_log_path}/eval', exist_ok=True)
        non_transfered_eval_callback = non_transfered_eval_callback.create(
            non_transfered_env,
            f'{non_transfer_tf_log_path}/eval',
            f'{non_transfer_tf_log_path}/eval'
        )

    non_transfered_model = DQN(
        policy="MlpPolicy",
        env=non_transfered_env,
        tensorboard_log=non_transfer_tf_log_path,
    )

    non_transfered_model = non_transfered_model.learn(
        steps, 
        progress_bar=True, 
        log_interval=1,
        callback=non_transfered_eval_callback
    )

    non_transfered_model.save(NON_TRANSERED_PATH)

    # train using transfer learning
    pretrained_env = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_pretrain,
        is_slippery = is_slippery,
        fps = fps,
        reward_range = reward_range,
        goal_reward = goal_reward,
        frozen_tile_reward = frozen_tile_reward,
        hole_reward = hole_reward
    ).dummy_vec_env(1)

    if pretrained_model == None:
        pretrained_model = DQN(
            policy="MlpPolicy",
            env=pretrained_env,
            tensorboard_log= logger.tf_logger("pretrainedFrozenLakeDQN"),
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

    transfered_model = DQN(
        policy="MlpPolicy",
        env=transfered_env,
        tensorboard_log= logger.tf_logger("transferedFrozenLakeDQN"),
    )

    if transfer_method == "Alternating":
        alternating_transfer_dqn_test(
            pretrained_steps=pretrain_steps, 
            transfered_steps=steps, 
            pretrained_model=pretrained_model,
            transfered_model=transfered_model,
            pretrained_eval_callback=pretrained_eval_callback,
            transfered_eval_callback=transfered_eval_callback,
            logger=logger
        )
    else:
        basic_transfer_dqn_test(
            pretrained_steps=pretrain_steps, 
            transfered_steps=steps, 
            pretrained_model=pretrained_model,
            transfered_model=transfered_model,
            pretrained_eval_callback=pretrained_eval_callback,
            transfered_eval_callback=transfered_eval_callback,
            logger=logger
        )

    plot_eval_rewards_tl(logger.get_dir(), f'{pretrain_map_size}x{pretrain_map_size} to {transfer_map_size}x{transfer_map_size} Mean Eval Rewards {"wo/" if hole_reward == 0 else "w/"} Reward Shaping', 'red' if hole_reward != 0 else 'blue')

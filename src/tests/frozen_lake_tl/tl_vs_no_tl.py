from tests.frozen_lake_tl.alternating_transfer_dqn_test import alternating_transfer_dqn_test
from tests.frozen_lake_tl.basic_transfer_dqn_test import basic_transfer_dqn_test
from envs.frozen_lake.frozen_lake import FrozenLakeEnv, generate_random_map
from stable_baselines3 import DQN
from envs.frozen_lake.frozen_lake import FrozenLakeEnv
from logger import tf_logger
from constants import NON_TRANSERED_PATH, TRANSERED_PATH
from tests.utils import eval_model, CustomEvalCallback
from typing import Literal, List

def tl_vs_no_tl(
    steps: int = 1e5, 
    pretrain_steps: int = 1e5,
    eval_eps: int = 100, 
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
    non_transfer_eval_callback: CustomEvalCallback = None,
    pretrain_eval_callback: CustomEvalCallback = None,
    transfer_callback: CustomEvalCallback = None,
):
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

    non_transfered_model = DQN(
        policy="MlpPolicy",
        env=non_transfered_env,
        tensorboard_log=tf_logger("nonTransferedFrozenLakeDQN"),
    )
    non_transfered_model = non_transfered_model.learn(
        steps, 
        progress_bar=True, 
        log_interval=1,
        eval_callback=non_transfer_eval_callback.create(non_transfered_env)
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
            tensorboard_log=tf_logger("pretrainedFrozenLakeDQN"),
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
        tensorboard_log=tf_logger("transferedFrozenLakeDQN"),
    )

    if transfer_method == "Alternating":
        alternating_transfer_dqn_test(
            pretrained_steps=pretrain_steps, 
            transfered_steps=steps, 
            pretrained_model=pretrained_model,
            transfered_model=transfered_model,
            pretrained_eval_callback=pretrain_eval_callback.create(pretrained_env),
            transfered_eval_callback=transfer_callback.create(transfered_env)
        )
    else:
        basic_transfer_dqn_test(
            pretrained_steps=pretrain_steps, 
            transfered_steps=steps, 
            pretrained_model=pretrained_model,
            transfered_model=transfered_model,
            pretrained_eval_callback=pretrain_eval_callback.create(pretrained_env),
            transfered_eval_callback=transfer_callback.create(transfered_env)
        )

    eval_env = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_transfer_test,
        is_slippery = is_slippery,
        fps = fps,
        reward_range = reward_range,
        goal_reward = goal_reward,
        frozen_tile_reward = frozen_tile_reward,
        hole_reward = hole_reward
    ).dummy_vec_env(1)

    print("Showing non transfered model")
    eval_model(
        DQN.load(NON_TRANSERED_PATH),
        eval_env,
        eval_eps
    )

    print("Showing transfered model")
    eval_model(
        DQN.load(TRANSERED_PATH),
        eval_env,
        eval_eps
    )

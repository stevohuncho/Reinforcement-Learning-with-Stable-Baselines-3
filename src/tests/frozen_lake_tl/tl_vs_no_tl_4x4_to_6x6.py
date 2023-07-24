from tests.frozen_lake_tl.basic_transfer_dqn_test import basic_transfer_dqn_test
from envs.frozen_lake.frozen_lake import FrozenLakeEnv, generate_random_map
from stable_baselines3 import DQN
from envs.frozen_lake.frozen_lake import FrozenLakeEnv
from logger import tf_logger
from constants import PRETRAINED_PATH, NON_TRANSERED_PATH, TRANSERED_PATH
from tests.utils import eval_model

def tl_vs_no_tl_4x4_to_6x6(steps: int, eval_eps: int):
    map_4x4 = generate_random_map(size=4, p=0.9)
    map_6x6 = generate_random_map(size=6, p=0.9)
    map_6x6_test = generate_random_map(size=6, p=0.9)

    # train non non-transfer learned model
    non_transfered_env = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_6x6,
    ).dummy_vec_env(1)

    non_transfered_model = DQN(
        policy="MlpPolicy",
        env=non_transfered_env,
        tensorboard_log=tf_logger("nonTransferedFrozenLakeDQN"),
    )
    non_transfered_model = non_transfered_model.learn(steps, progress_bar=True, log_interval=1)

    non_transfered_model.save(NON_TRANSERED_PATH)

    # train using transfer learning
    pretrained_env = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_4x4,
    ).dummy_vec_env(1)

    pretrained_model = DQN(
        policy="MlpPolicy",
        env=pretrained_env,
        tensorboard_log=tf_logger("pretrainedFrozenLakeDQN"),
    )

    transfered_env = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_6x6,
    ).dummy_vec_env(1)

    transfered_model = DQN(
        policy="MlpPolicy",
        env=transfered_env,
        tensorboard_log=tf_logger("transferedFrozenLakeDQN"),
    )

    basic_transfer_dqn_test(
        pretrained_steps=steps, 
        transfered_steps=steps, 
        pretrained_model=pretrained_model,
        transfered_model=transfered_model,
    )

    print("Showing non transfered model")
    eval_model(
        DQN.load(NON_TRANSERED_PATH),
        non_transfered_env,
        eval_eps
    )

    print("Showing transfered model")
    eval_model(
        DQN.load(TRANSERED_PATH),
        transfered_env,
        eval_eps
    )

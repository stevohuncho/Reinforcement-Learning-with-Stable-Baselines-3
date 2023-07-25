from stable_baselines3 import DQN
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from datetime import datetime
from stable_baselines3.common.callbacks import EvalCallback
from transfer_learning.AlternatingTransferDQN.dqn import AlternatingTransferDQN
from envs.frozen_lake.frozen_lake import FrozenLakeEnv
from logger import tf_logger
import os

MODELS_DIR = "../models"
PRETRAINED_PATH = "../models/pretrainedFrozenLakeDQN"
TRANSERED_PATH = "../models/transferedFrozenLakeDQN"

def alternating_transfer_dqn_test(
    pretrained_steps: int = 1e5,
    transfered_steps: int = 1e5,
    pretrained_model: DQN = None,
    transfered_model: AlternatingTransferDQN = None,
    skip_pretrain_learning_steps: bool = False,
    pretrained_eval_callback: CustomEvalCallback = None,
    transfered_eval_callback: CustomEvalCallback = None,
):
    if pretrained_model is None:
        pretrained_model = DQN(
            "MlpPolicy", 
            FrozenLakeEnv(
                render_mode="rgb_array",
                fps=4,
            ).dummy_vec_env(1), 
            tensorboard_log=tf_logger("pretrainedFrozenLakeDQN"),
            buffer_size=int(1e5)
        )

    if not skip_pretrain_learning_steps:
        pretrained_model = pretrained_model.learn(
            pretrained_steps, 
            log_interval=1, 
            progress_bar=True,
            eval_callback=pretrained_eval_callback.create(pretrained_model.get_env()),
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    pretrained_model.save(PRETRAINED_PATH)

    pretrained_model = DQN.load(PRETRAINED_PATH)

    if transfered_model is None:
        transfered_model = AlternatingTransferDQN(
            "MlpPolicy", 
            FrozenLakeEnv(
                render_mode="rgb_array",
                fps=4,
                desc=generate_random_map(8,0.9),
            ).dummy_vec_env(1),
            pretrained_model=pretrained_model, 
            tensorboard_log=tf_logger("transferedFrozenLakeDQN"),
            buffer_size=int(1e5)
        )

    transfered_model = transfered_model.learn(
        transfered_steps, 
        log_interval=1, 
        progress_bar=True,
        eval_callback=transfered_eval_callback.create(transfered_model.get_env()),
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    transfered_model.save(TRANSERED_PATH)
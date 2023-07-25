from stable_baselines3 import DQN
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from datetime import datetime
from tests.utils import CustomEvalCallback
from transfer_learning.BasicTransferDQN.dqn import BasicTransferDQN
from envs.frozen_lake.frozen_lake import FrozenLakeEnv
from logger import Logger
import os

MODELS_DIR = "../models"
PRETRAINED_PATH = "../models/pretrainedFrozenLakeDQN"
TRANSERED_PATH = "../models/transferedFrozenLakeDQN"

def basic_transfer_dqn_test(
    pretrained_steps: int = 1e5,
    transfered_steps: int = 1e5,
    pretrained_model: DQN = None,
    transfered_model: BasicTransferDQN = None,
    skip_pretrain_learning_steps: bool = False,
    pretrained_eval_callback: CustomEvalCallback = None,
    transfered_eval_callback: CustomEvalCallback = None,
    logger: Logger = None
):
    if logger is None:
        logger = Logger("")

    if pretrained_model is None:
        pretrained_model = DQN(
            "MlpPolicy", 
            FrozenLakeEnv(
                render_mode="rgb_array",
                fps=4,
            ).dummy_vec_env(1), 
            tensorboard_log= logger.tf_logger("pretrainedFrozenLakeDQN"),
            buffer_size=int(1e5)
        )
    
    if pretrained_eval_callback is not None:
        pretrained_eval_callback_log_path = f'{logger.tf_logger("pretrainedFrozenLakeDQN")}/eval'
        os.makedirs(pretrained_eval_callback_log_path, exist_ok=True)
        pretrained_eval_callback = pretrained_eval_callback.create(
            pretrained_model.get_env(),
            pretrained_eval_callback_log_path,
            pretrained_eval_callback_log_path
        )

    if not skip_pretrain_learning_steps:
        pretrained_model = pretrained_model.learn(
            pretrained_steps, 
            log_interval=1, 
            progress_bar=True,
            callback=pretrained_eval_callback,
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    pretrained_model.save(PRETRAINED_PATH)

    pretrained_model = DQN.load(PRETRAINED_PATH)

    if transfered_model is None:
        transfered_model = BasicTransferDQN(
            "MlpPolicy", 
            FrozenLakeEnv(
                render_mode="rgb_array",
                fps=4,
                desc=generate_random_map(8,0.9),
            ).dummy_vec_env(1),
            pretrained_model=pretrained_model, 
            tensorboard_log= logger.tf_logger("transferedFrozenLakeDQN"),
            buffer_size=int(1e5)
        )

    if transfered_eval_callback is not None:
        transfered_eval_callback_log_path = f'{logger.tf_logger("transferedFrozenLakeDQN")}/eval'
        os.makedirs(transfered_eval_callback_log_path, exist_ok=True)
        transfered_eval_callback = transfered_eval_callback.create(
            transfered_model.get_env(),
            transfered_eval_callback_log_path,
            transfered_eval_callback_log_path
        )
    transfered_model = transfered_model.learn(
        transfered_steps, 
        log_interval=1, 
        progress_bar=True,
        callback=transfered_eval_callback,
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    transfered_model.save(TRANSERED_PATH)
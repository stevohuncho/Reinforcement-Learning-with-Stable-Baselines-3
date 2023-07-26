from tests.tl_vs_no_tl import tl_vs_no_tl
from envs.frozen_lake.frozen_lake import generate_random_map, FrozenLakeEnv
from matplot import plot_eval_rewards_tl
from tests.utils import CustomEvalCallback
from stable_baselines3 import DQN
from constants import MODELS_DIR
import os

def main():
    # tests to run:
    #   - pretrained_size(4) -> transfered_size(6-8)
    #   - pretrained_size(6-8) -> transfered_size(4)
    #   - pretrained_size(4) -> transfered_size(6-8) w/ Reward Shaping
    #   - pretrained_size(6-8) -> transfered_size(4) w/ Reward Shaping

    # generate maps if needed
    #map_4x4 = generate_random_map(4, 0.85)
    #map_6x6 = generate_random_map(6, 0.9)
    #map_8x8 = generate_random_map(8, 0.9)
    map_4x4 = ['SFFF', 'FFFF', 'FFFF', 'FFFG']
    map_6x6 = ['SHFFFF', 'FHFFFF', 'FFFFFF', 'FHFFFF', 'FFFFFF', 'FFFFFG']
    map_8x8 = ['SFFFFFFF', 'FFFHFFFH', 'FFFFHFFF', 'FFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFFFF', 'FFFFFFFG']

    # pretrain all differing environments
    TRAINING_TIMESTEPS = 100_000

    # pretrained_4x4_env_w_rs
    pretrained_4x4_env_w_rs = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_4x4,
        hole_reward=-1
    )
    if not os.path.isfile(f'{MODELS_DIR}/pretrained_4x4_env_w_rs/best_model.zip'):
        pretrained_4x4_model_w_rs = DQN(
            policy="MlpPolicy",
            env=pretrained_4x4_env_w_rs
        )
        pretrained_4x4_model_w_rs = pretrained_4x4_model_w_rs.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            progress_bar=True,
            callback=CustomEvalCallback(n_eval_episodes=20).create(
                env=pretrained_4x4_env_w_rs,
                best_model_save_path=f'{MODELS_DIR}/pretrained_4x4_env_w_rs',
                log_path=f'{MODELS_DIR}/pretrained_4x4_env_w_rs'
            )
        )
    else:
        pretrained_4x4_model_w_rs = DQN.load(
            path=f'{MODELS_DIR}/pretrained_4x4_env_w_rs/best_model.zip',
            env=pretrained_4x4_env_w_rs
        )

    # pretrained_4x4_env_wo_rs
    pretrained_4x4_env_wo_rs = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_4x4
    )
    if not os.path.isfile(f'{MODELS_DIR}/pretrained_4x4_env_wo_rs/best_model.zip'):
        pretrained_4x4_model_wo_rs = DQN(
            policy="MlpPolicy",
            env=pretrained_4x4_env_wo_rs
        )
        pretrained_4x4_model_wo_rs = pretrained_4x4_model_wo_rs.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            progress_bar=True,
            callback=CustomEvalCallback(n_eval_episodes=20).create(
                env=pretrained_4x4_env_wo_rs,
                best_model_save_path=f'{MODELS_DIR}/pretrained_4x4_env_wo_rs',
                log_path=f'{MODELS_DIR}/pretrained_4x4_env_wo_rs'
            )
        )
    else:
        pretrained_4x4_model_wo_rs = DQN.load(
            path=f'{MODELS_DIR}/pretrained_4x4_env_wo_rs/best_model.zip',
            env=pretrained_4x4_env_wo_rs
        )

    # pretrained_6x6_env_w_rs
    pretrained_6x6_env_w_rs = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_6x6,
        hole_reward=-1
    )
    if not os.path.isfile(f'{MODELS_DIR}/pretrained_6x6_env_w_rs/best_model.zip'):
        pretrained_6x6_model_w_rs = DQN(
            policy="MlpPolicy",
            env=pretrained_6x6_env_w_rs
        )
        pretrained_6x6_model_w_rs = pretrained_6x6_model_w_rs.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            progress_bar=True,
            callback=CustomEvalCallback(n_eval_episodes=20).create(
                env=pretrained_6x6_env_w_rs,
                best_model_save_path=f'{MODELS_DIR}/pretrained_6x6_env_w_rs',
                log_path=f'{MODELS_DIR}/pretrained_6x6_env_w_rs'
            )
        )
    else:
        pretrained_6x6_model_w_rs = DQN.load(
            path=f'{MODELS_DIR}/pretrained_6x6_env_w_rs/best_model.zip',
            env=pretrained_6x6_env_w_rs
        )

    # pretrained_6x6_env_wo_rs
    pretrained_6x6_env_wo_rs = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_6x6
    )
    if not os.path.isfile(f'{MODELS_DIR}/pretrained_6x6_env_wo_rs/best_model.zip'):
        pretrained_6x6_model_wo_rs = DQN(
            policy="MlpPolicy",
            env=pretrained_6x6_env_wo_rs
        )
        pretrained_6x6_model_wo_rs = pretrained_6x6_model_wo_rs.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            progress_bar=True,
            callback=CustomEvalCallback(n_eval_episodes=20).create(
                env=pretrained_6x6_env_wo_rs,
                best_model_save_path=f'{MODELS_DIR}/pretrained_6x6_env_wo_rs',
                log_path=f'{MODELS_DIR}/pretrained_6x6_env_wo_rs'
            )
        )
    else:
        pretrained_6x6_model_wo_rs = DQN.load(
            path=f'{MODELS_DIR}/pretrained_6x6_env_wo_rs/best_model.zip',
            env=pretrained_6x6_env_wo_rs
        )

    # pretrained_8x8_env_w_rs
    pretrained_8x8_env_w_rs = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_8x8,
        hole_reward=-1
    )
    if not os.path.isfile(f'{MODELS_DIR}/pretrained_8x8_env_w_rs/best_model.zip'):
        pretrained_8x8_model_w_rs = DQN(
            policy="MlpPolicy",
            env=pretrained_8x8_env_w_rs
        )
        pretrained_8x8_model_w_rs = pretrained_8x8_model_w_rs.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            progress_bar=True,
            callback=CustomEvalCallback(n_eval_episodes=20).create(
                env=pretrained_8x8_env_w_rs,
                best_model_save_path=f'{MODELS_DIR}/pretrained_8x8_env_w_rs',
                log_path=f'{MODELS_DIR}/pretrained_8x8_env_w_rs'
            )
        )
    else:
        pretrained_8x8_model_w_rs = DQN.load(
            path=f'{MODELS_DIR}/pretrained_8x8_env_w_rs/best_model.zip',
            env=pretrained_8x8_env_w_rs
        )

    # pretrained_8x8_env_wo_rs
    pretrained_8x8_env_wo_rs = FrozenLakeEnv(
        render_mode="rgb_array",
        desc=map_8x8
    )
    if not os.path.isfile(f'{MODELS_DIR}/pretrained_8x8_env_wo_rs/best_model.zip'):
        pretrained_8x8_model_wo_rs = DQN(
            policy="MlpPolicy",
            env=pretrained_8x8_env_wo_rs
        )
        pretrained_8x8_model_wo_rs = pretrained_8x8_model_wo_rs.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            progress_bar=True,
            callback=CustomEvalCallback(n_eval_episodes=20).create(
                env=pretrained_8x8_env_wo_rs,
                best_model_save_path=f'{MODELS_DIR}/pretrained_8x8_env_wo_rs',
                log_path=f'{MODELS_DIR}/pretrained_8x8_env_wo_rs'
            )
        )
    else:
        pretrained_8x8_model_wo_rs = DQN.load(
            path=f'{MODELS_DIR}/pretrained_8x8_env_wo_rs/best_model.zip',
            env=pretrained_8x8_env_wo_rs
        )

    # conduct transfer learning tests

    # growing map tests
    if not os.path.isfile(f'{MODELS_DIR}/transfered_4x4_to_6x6_env_w_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_4x4_to_6x6_env_w_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_4x4_model_w_rs,
            f'{MODELS_DIR}/transfered_4x4_to_6x6_env_w_rs',
            f'{MODELS_DIR}/non_transfered_4x4_to_6x6_env_w_rs',
            map_pretrain=map_4x4,
            map_transfer=map_6x6,
            hole_reward=-1,
        )

    if not os.path.isfile(f'{MODELS_DIR}/transfered_4x4_to_6x6_env_wo_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_4x4_to_6x6_env_wo_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_4x4_model_wo_rs,
            f'{MODELS_DIR}/transfered_4x4_to_6x6_env_wo_rs',
            f'{MODELS_DIR}/non_transfered_4x4_to_6x6_env_wo_rs',
            map_pretrain=map_4x4,
            map_transfer=map_6x6,
        )

    if not os.path.isfile(f'{MODELS_DIR}/transfered_4x4_to_8x8_env_w_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_4x4_to_8x8_env_w_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_4x4_model_w_rs,
            f'{MODELS_DIR}/transfered_4x4_to_8x8_env_w_rs',
            f'{MODELS_DIR}/non_transfered_4x4_to_8x8_env_w_rs',
            map_pretrain=map_4x4,
            map_transfer=map_8x8,
            hole_reward=-1,
        )

    if not os.path.isfile(f'{MODELS_DIR}/transfered_4x4_to_8x8_env_wo_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_4x4_to_8x8_env_wo_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_4x4_model_wo_rs,
            f'{MODELS_DIR}/transfered_4x4_to_8x8_env_wo_rs',
            f'{MODELS_DIR}/non_transfered_4x4_to_8x8_env_wo_rs',
            map_pretrain=map_4x4,
            map_transfer=map_8x8,
        )

    # shrinking map tests
    if not os.path.isfile(f'{MODELS_DIR}/transfered_8x8_to_4x4_env_w_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_8x8_to_4x4_env_w_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_8x8_model_w_rs,
            f'{MODELS_DIR}/transfered_8x8_to_4x4_env_w_rs',
            f'{MODELS_DIR}/non_transfered_8x8_to_4x4_env_w_rs',
            map_pretrain=map_8x8,
            map_transfer=map_4x4,
            hole_reward=-1,
        )

    if not os.path.isfile(f'{MODELS_DIR}/transfered_8x8_to_4x4_env_wo_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_8x8_to_4x4_env_wo_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_8x8_model_wo_rs,
            f'{MODELS_DIR}/transfered_8x8_to_4x4_env_wo_rs',
            f'{MODELS_DIR}/non_transfered_8x8_to_4x4_env_wo_rs',
            map_pretrain=map_8x8,
            map_transfer=map_4x4,
        )

    if not os.path.isfile(f'{MODELS_DIR}/transfered_8x8_to_6x6_env_w_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_8x8_to_6x6_env_w_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_8x8_model_w_rs,
            f'{MODELS_DIR}/transfered_8x8_to_6x6_env_w_rs',
            f'{MODELS_DIR}/non_transfered_8x8_to_6x6_env_w_rs',
            map_pretrain=map_8x8,
            map_transfer=map_6x6,
            hole_reward=-1,
        )

    if not os.path.isfile(f'{MODELS_DIR}/transfered_8x8_to_6x6_env_wo_rs/best_model.zip') or not os.path.isfile(f'{MODELS_DIR}/non_transfered_8x8_to_6x6_env_wo_rs/best_model.zip'):
        tl_vs_no_tl(
            pretrained_8x8_model_wo_rs,
            f'{MODELS_DIR}/transfered_8x8_to_6x6_env_wo_rs',
            f'{MODELS_DIR}/non_transfered_8x8_to_6x6_env_wo_rs',
            map_pretrain=map_8x8,
            map_transfer=map_6x6,
        )

    # plot data
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_4x4_to_6x6_env_w_rs', f'{MODELS_DIR}/transfered_4x4_to_6x6_env_w_rs', f'{MODELS_DIR}/pretrained_4x4_env_w_rs'],
        "4x4 To 6x6 Mean Evaluations w/ Reward Shaping",
        "red"
    )
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_4x4_to_6x6_env_wo_rs', f'{MODELS_DIR}/transfered_4x4_to_6x6_env_wo_rs', f'{MODELS_DIR}/pretrained_4x4_env_wo_rs'],
        "4x4 To 6x6 Mean Evaluations wo/ Reward Shaping",
        "blue"
    )
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_4x4_to_8x8_env_w_rs', f'{MODELS_DIR}/transfered_4x4_to_8x8_env_w_rs', f'{MODELS_DIR}/pretrained_4x4_env_w_rs'],
        "4x4 To 8x8 Mean Evaluations w/ Reward Shaping",
        "red"
    )
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_4x4_to_8x8_env_wo_rs', f'{MODELS_DIR}/transfered_4x4_to_8x8_env_wo_rs', f'{MODELS_DIR}/pretrained_4x4_env_wo_rs'],
        "4x4 To 8x8 Mean Evaluations wo/ Reward Shaping",
        "blue"
    )
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_8x8_to_6x6_env_w_rs', f'{MODELS_DIR}/transfered_8x8_to_6x6_env_w_rs', f'{MODELS_DIR}/pretrained_8x8_env_w_rs'],
        "8x8 To 6x6 Mean Evaluations w/ Reward Shaping",
        "red"
    )
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_8x8_to_6x6_env_wo_rs', f'{MODELS_DIR}/transfered_8x8_to_6x6_env_wo_rs', f'{MODELS_DIR}/pretrained_8x8_env_wo_rs'],
        "8x8 To 6x6 Mean Evaluations wo/ Reward Shaping",
        "blue"
    )
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_8x8_to_4x4_env_w_rs', f'{MODELS_DIR}/transfered_8x8_to_4x4_env_w_rs', f'{MODELS_DIR}/pretrained_8x8_env_w_rs'],
        "8x8 To 8x8 Mean Evaluations w/ Reward Shaping",
        "red"
    )
    plot_eval_rewards_tl(
        [f'{MODELS_DIR}/non_transfered_8x8_to_4x4_env_wo_rs', f'{MODELS_DIR}/transfered_8x8_to_4x4_env_wo_rs', f'{MODELS_DIR}/pretrained_8x8_env_wo_rs'],
        "8x8 To 4x4 Mean Evaluations wo/ Reward Shaping",
        "blue"
    )

    
if __name__ == "__main__":
    main()
from tests.frozen_lake_tl.tl_vs_no_tl import tl_vs_no_tl
from tests.utils import CustomEvalCallback
from envs.frozen_lake.frozen_lake import generate_random_map

def tl_w_rs_vs_wo(pre_size: int, post_size: int):
    pre_map = generate_random_map(pre_size)
    print(pre_map)
    post_map = generate_random_map(post_size)
    print(post_map)

    tl_vs_no_tl(
        name=f"{pre_size}_to_{post_size}_w_rs",
        steps=2e5,
        pretrain_steps=2e5,
        pretrain_map_size=6,
        transfer_map_size=4,
        non_transfered_eval_callback=CustomEvalCallback(
            eval_freq=5e3,
            n_eval_episodes=10,
        ),
        pretrained_eval_callback=CustomEvalCallback(
            eval_freq=5e3,
            n_eval_episodes=10,
        ),
        transfered_eval_callback=CustomEvalCallback(
            eval_freq=5e3,
            n_eval_episodes=10,
        ),
        hole_reward=-1,
        map_pretrain=post_map,
        map_transfer=pre_map
    )

    tl_vs_no_tl(
        name=f"{pre_size}_to_{post_size}_wo_rs",
        steps=2e5,
        pretrain_steps=2e5,
        pretrain_map_size=6,
        transfer_map_size=4,
        non_transfered_eval_callback=CustomEvalCallback(
            eval_freq=5e3,
            n_eval_episodes=10,
        ),
        pretrained_eval_callback=CustomEvalCallback(
            eval_freq=5e3,
            n_eval_episodes=10,
        ),
        transfered_eval_callback=CustomEvalCallback(
            eval_freq=5e3,
            n_eval_episodes=10,
        ),
        map_pretrain=post_map,
        map_transfer=pre_map
    )
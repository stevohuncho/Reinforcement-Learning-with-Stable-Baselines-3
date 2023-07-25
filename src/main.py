from tests.frozen_lake_tl.tl_vs_no_tl import tl_vs_no_tl
from tests.utils import CustomEvalCallback

def main():
    # tests to run:
    #   - pretrained_size(4) -> transfered_size(5-8)
    #   - pretrained_size(5-8) -> transfered_size(4)
    #   - pretrained_size(4) -> transfered_size(5-8) w/ Reward Shaping
    #   - pretrained_size(5-8) -> transfered_size(4) w/ Reward Shaping
    tl_vs_no_tl(
        steps=1e4,
        pretrain_steps=1e4,
        transfer_map_size=5,
        non_transfered_eval_callback=CustomEvalCallback(
            eval_freq=1e3,
            n_eval_episodes=10,
        ),
        pretrained_eval_callback=CustomEvalCallback(
            eval_freq=1e3,
            n_eval_episodes=10,
        ),
        transfered_eval_callback=CustomEvalCallback(
            eval_freq=1e3,
            n_eval_episodes=10,
        )
    )
    
if __name__ == "__main__":
    main()
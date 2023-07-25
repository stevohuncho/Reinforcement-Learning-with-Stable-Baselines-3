from tests.frozen_lake_tl.tl_vs_no_tl import tl_vs_no_tl
from stable_baselines3.common.callbacks import EvalCallback

def main():
    # tests to run:
    #   - pretrained_size(4) -> transfered_size(5-8)
    #   - pretrained_size(5-8) -> transfered_size(4)
    #   - pretrained_size(4) -> transfered_size(5-8) w/ Reward Shaping
    #   - pretrained_size(5-8) -> transfered_size(4) w/ Reward Shaping
    tl_vs_no_tl(

    )
    
if __name__ == "__main__":
    main()
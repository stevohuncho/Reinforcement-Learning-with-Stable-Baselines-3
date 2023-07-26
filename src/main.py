from tests.tests import tl_w_rs_vs_wo

def main():
    # tests to run:
    #   - pretrained_size(4) -> transfered_size(5-8)
    #   - pretrained_size(5-8) -> transfered_size(4)
    #   - pretrained_size(4) -> transfered_size(5-8) w/ Reward Shaping
    #   - pretrained_size(5-8) -> transfered_size(4) w/ Reward Shaping
    tl_w_rs_vs_wo(4,5)

    
if __name__ == "__main__":
    main()
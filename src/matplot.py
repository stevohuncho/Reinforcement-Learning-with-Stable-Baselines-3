import matplotlib.pyplot as plt
import numpy as np
from typing import List

def _gather_eval_rewards(data: np.ndarray) -> np.ndarray:
    eval_rewards = []
    for i, line in enumerate(data):
        eval_total_reward = 0
        for point in line:
            eval_total_reward += point
        eval_rewards.append(eval_total_reward/len(line))
    return np.array(eval_rewards)

def plot_eval_rewards_tl(paths: List[str], title: str, color: str) -> None:
    non_transfered_data = np.load(f'{paths[0]}/evaluations.npz')
    transfered_data = np.load(f'{paths[1]}/evaluations.npz')
    pretrained_data = np.load(f'{paths[2]}/evaluations.npz')

    if non_transfered_data is None or transfered_data is None or pretrained_data is None:
        return
    
    non_transfered_arr = non_transfered_data['results']
    transfered_arr = transfered_data['results']
    pretrained_arr = pretrained_data['results']

    plt.figure(figsize=(15,5))
    plt.xticks(range(len(non_transfered_data['timesteps'])), non_transfered_data['timesteps'])
    plt.ylim(bottom=0)

    non_transfered_arr[non_transfered_arr < 0.] = 0.
    plt.plot(_gather_eval_rewards(non_transfered_arr), label="Non Transfered Model", linestyle="-.", color=color)
    transfered_arr[transfered_arr < 0.] = 0.
    plt.plot(_gather_eval_rewards(transfered_arr), label="Transfered Model", linestyle="-", color=color)
    pretrained_arr[pretrained_arr < 0.] = 0.
    plt.plot(_gather_eval_rewards(pretrained_arr), label="Pretrained Model", linestyle="--", color=color)

    plt.title(title, fontname='sans-serif', fontsize=14, fontstyle='italic')
    plt.xlabel("Steps", fontname='sans-serif', fontweight="semibold")
    plt.ylabel("Reward", fontname='sans-serif', fontweight="semibold")
    plt.legend()
    plt.savefig(f'../results/{title.replace(" ", "_").replace("/", "")}_mean_eval_rewards.png')
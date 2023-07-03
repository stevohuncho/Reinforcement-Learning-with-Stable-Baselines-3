import os
import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window_size):
    
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    if window_size % 2 == 0:
        raise ValueError("Please pick an odd window size.")
    
    # Create a filter that serves to average the window_size elements about each element.
    averaging_filter = np.repeat(1.0, window_size) / window_size
    
    # Do a convolution, ensuring only to keep the center elements such that |conv| = len(values)
    moving_averages = np.convolve(averaging_filter, values, mode='same')
    
    # Moving averages need to be fixed at the ends as they were not averaged correctly.
    for i in range(window_size // 2 + 1):
        moving_averages[i] = moving_averages[i] * window_size / (window_size // 2 + 1 + i)
        moving_averages[len(moving_averages) - 1 - i] = moving_averages[len(moving_averages) - 1 - i] * window_size / (window_size // 2 + 1 + i)
        
    return moving_averages

def plot_results(log_folder, title='Learning Curve', moving_window=-1):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'episodes')
    y = moving_average(y, window=moving_window)
    # Truncate x
    x = x[len(x) - len(y):]

    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

def plot_multiple_results(log_dir_w_TL, log_dir_w_TL_rs, log_dir_wo_TL, log_dir_w_full_TL_rs, title='Learning Curve', moving_window=-1):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    # With TL
    x_w_TL, y_w_TL = ts2xy(load_results(log_dir_w_TL), 'episodes')
    y_w_TL = moving_average(y_w_TL, window=moving_window)
    x_w_TL = x_w_TL[len(x_w_TL) - len(y_w_TL):]

    # With TL and reward shaping
    x_w_TL_rs, y_w_TL_rs = ts2xy(load_results(log_dir_w_TL_rs), 'episodes')
    y_w_TL_rs = moving_average(y_w_TL_rs, window=moving_window)
    x_w_TL_rs = x_w_TL_rs[len(x_w_TL_rs) - len(y_w_TL_rs):]

    # Without TL
    x_wo_TL, y_wo_TL = ts2xy(load_results(log_dir_wo_TL), 'episodes')
    y_wo_TL = moving_average(y_wo_TL, window=moving_window)
    x_wo_TL = x_wo_TL[len(x_wo_TL) - len(y_wo_TL):]

    # With full TL and reward shaping
    x_w_full_TL_rs, y_w_full_TL_rs = ts2xy(load_results(log_dir_w_full_TL_rs), 'episodes')
    y_w_full_TL_rs = moving_average(y_w_full_TL_rs, window=moving_window)
    x_w_full_TL_rs = x_w_full_TL_rs[len(x_w_full_TL_rs) - len(y_w_full_TL_rs):]


    plt.figure(title)

    plt.plot(x_wo_TL, y_wo_TL, marker='x', markersize=8, linestyle='-', color='b', label='Without TL', linewidth=3)
    plt.plot(x_w_TL, y_w_TL, marker='o', markersize=8, linestyle='-', color='g', label='With TL',  linewidth=3)
    plt.plot(x_w_TL_rs, y_w_TL_rs, marker='s', markersize=8, linestyle='-', color='r', label='With TL rs', linewidth=3)
    plt.plot(x_w_full_TL_rs, y_w_full_TL_rs, marker='*', markersize=8, linestyle='-', color='y', label='With full TL rs', linewidth=3)

    plt.legend(loc='upper left', title='Approaches', fontsize=14)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()


def get_avg_std(list_of_lists):
    # Calculate average list
    avg_list = None
    for a_list in list_of_lists:
        if avg_list is None:
            avg_list = a_list
        else:
            avg_list = [(x + y) for (x, y) in zip(avg_list, a_list)]
    num_lists = len(list_of_lists)
    avg_list = [(x / num_lists) for x in avg_list]

    if num_lists == 1:
        std_list = [0.0 for x in range(len(avg_list))]
        return avg_list, std_list

    # Calculate sample standard dev. list
    std_list = None
    for a_list in list_of_lists:
        to_add_list = [(x - y) * (x - y) for (x, y) in zip(a_list, avg_list)]
        if std_list is None:
            std_list = to_add_list
        else:
            std_list = [(x + y) for (x, y) in zip(to_add_list, std_list)]
    std_list = [math.sqrt(x / (num_lists - 1)) for x in std_list]

    return avg_list, std_list



def loading_all_exp_result_from_directory(dir, running_time_num=1):
    x = []
    y = []
    for i in range(running_time_num):
        new_dir = dir + str(i) + "/"
        xx, yy = ts2xy(load_results(new_dir), 'episodes')
        x.append(xx) # Keep track of all x's; turns out that not all runs have same episode count.
        y.append(yy)

    # Determine run with least episodes. Adjust ALL runs to have only as many as the one with the least episodes.
    episode_count = len(x[0])    
    for i in range(running_time_num):
        if len(x[i]) < episode_count:
            episode_count = len(x[i])
    
    for i in range(running_time_num):
        x[i] = (x[i])[:episode_count]
        y[i] = (y[i])[:episode_count]

    return x[0], y


def extract_xy_for_plotting(dir, running_time_num, moving_window):
    x_plot, y = loading_all_exp_result_from_directory(dir, running_time_num)

    y_moving_averages = []
    for episode_reward_list in y:
        y_moving_averages.append(moving_average(episode_reward_list, window_size=moving_window))

    y_ave, y_std = get_avg_std(y_moving_averages)

    lower_list = [(x - y) for (x, y) in zip(y_ave, y_std)]
    upper_list = [(x + y) for (x, y) in zip(y_ave, y_std)]

    y_plot = y_ave

    return x_plot, y_plot, lower_list, upper_list

def plot_multiple_results_with_multiple_runing_time(log_dir_w_TL,
                                                    log_dir_w_TL_rs,
                                                    log_dir_wo_TL,
                                                    title='Learning Curve', moving_window=-1,
                                                    running_time_num=1,
                                                    figsize=(16,9)):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    # With TL
    x_w_TL,y_w_TL,w_TL_lower_list, w_TL_upper_list = extract_xy_for_plotting(log_dir_w_TL, running_time_num, moving_window)


    # With TL and reward shaping
    x_w_TL_rs, y_w_TL_rs, w_TL_rs_lower_list, w_TL_rs_upper_list = extract_xy_for_plotting(log_dir_w_TL_rs, running_time_num,
                                                                               moving_window)

    # Without TL
    x_wo_TL, y_wo_TL, wo_TL_lower_list, wo_TL_upper_list = extract_xy_for_plotting(log_dir_wo_TL, running_time_num,
                                                                               moving_window)

    plt.figure(title, figsize=figsize)

    plt.plot(x_wo_TL, y_wo_TL, marker='x', markersize=8, linestyle='-', color='b', label='Without TL', linewidth=3)
    plt.fill_between(x_wo_TL, wo_TL_lower_list, wo_TL_upper_list, color='lightblue', alpha=0.4)

    plt.plot(x_w_TL, y_w_TL, marker='o', markersize=8, linestyle='-', color='g', label='With TL',  linewidth=3)
    plt.fill_between(x_w_TL, w_TL_lower_list, w_TL_upper_list, color='lightgreen', alpha=0.4)

    plt.plot(x_w_TL_rs, y_w_TL_rs, marker='s', markersize=8, linestyle='-', color='r', label='With TL rs', linewidth=3)
    plt.fill_between(x_w_TL_rs, w_TL_rs_lower_list, w_TL_rs_upper_list, color='lightcoral', alpha=0.4)

    plt.legend(loc='upper left', title='Approaches', fontsize=14)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()
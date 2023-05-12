import matplotlib.pyplot as plt
import numpy as np
import ast



def plot_multiple_runs():
    # folders = ["n_steps", "5_n_steps", "test_learn_2", "dueling", "double_dqn_2"]
    # folders = ["prioritized", "prioritized_3_n_steps", "3_n_steps_optimized", "test_learn_2", "prioritized_3_n_step_dueling", "double_dqn_2"]
    folders = ["double_dqn_3_5000", "double_dqn_4_5000"]
    data = {}
    for folder in folders:
        with open(f"F:/Coding/breakout/{folder}/rewards.txt", "r") as f:
            data_str = f.read()
            data[folder] = ast.literal_eval(data_str)
    # with open("F:/Coding/breakout/5_n_steps/rewards.txt", "r") as f:
    #     data_str2 = f.read()
    #     data2 = ast.literal_eval(data_str2)
    # fig, ax = plt.subplots()
    fig, axs = plt.subplots(len(folders))
    for folder in folders:
        
        data_array = np.array(data[folder])
        total = 4000
        interval = 100
        # print(data_array[:total].mean())
        reshaped_arr = data_array[:total].reshape(total//interval, interval)
        averaged_arr = reshaped_arr.mean(axis=1)
        print(folder)
        print(averaged_arr)
        print()
        axs[folders.index(folder)].plot(averaged_arr)
        axs[folders.index(folder)].set_title(folder)
        # axs[folders.index(folder)].set_xlabel('Episodes')
        axs[folders.index(folder)].set_ylabel('Rewards')
        axs[folders.index(folder)].set_ylim([0, 40])  # Set y-axis limits
    plt.show()

def plot_rewards():
    # with open("rainbow_models/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/rewards.txt", "r") as f:
    with open("F:/Coding/breakout/prioritized_3_n_step_dueling_20_000/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/prioritized/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/prioritized_exp_replay_retry_loss_2/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/dueling/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/n_steps/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/5_n_steps/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/test_learn_2/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/double_dqn_2/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/control/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/good_rainbow/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/noNormalizationWithSteps_2/rewards.txt", "r") as f:
    # with open("F:/Coding/breakout/normalizedModelSteps/rewards.txt", "r") as f:
    # with open("models_3_actions/rewards.txt", "r") as f:
    # with open("rainbow_models/good_models/rewards_1.txt", "r") as f:
        data_str = f.read()
        data = ast.literal_eval(data_str)

    fig, ax = plt.subplots()
    data_array = np.array(data)
    # print(np.average(data_array[5000:]))
    total = 18000
    interval = 100
    # reshaped_arr = data_array[-2_000:].reshape(40, 50)
    print(data_array[-total:].mean())
    reshaped_arr = data_array[-total:].reshape(total//interval, interval)

    averaged_arr = reshaped_arr.mean(axis=1)
    print(averaged_arr)
    ax.plot(averaged_arr)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_title('Rewards over time')
    plt.show()

plot_multiple_runs()
# plot_rewards()

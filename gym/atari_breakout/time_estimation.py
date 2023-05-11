import numpy as np
import ast
import matplotlib.pyplot as plt

def estimate_remaining_time(total_episodes, current_episode, times):
    # Fit a line to the times
    x = np.arange(current_episode)
    coefficients = np.polyfit(x, times, 2)  # Fit a line (polynomial of degree 1)
    fitted_line = np.poly1d(coefficients)

    # Predict the times of the remaining episodes
    remaining_episodes = np.arange(current_episode, total_episodes)
    predicted_times = fitted_line(remaining_episodes)

    # Sum the predicted times to get the total remaining time
    remaining_time = np.sum(predicted_times)
    
    total_time = np.sum(times) + remaining_time
    print(f"Estimated time remaining: {remaining_time} seconds")
    print(f"Total time: {total_time} seconds")

    # Plot the cumulative times
    plt.plot(x, np.cumsum(times), 'o')

    cum = np.cumsum(predicted_times) + np.sum(times)
    plt.plot(remaining_episodes, cum)
    plt.xlabel("Episode")
    plt.ylabel("Time (seconds)")
    plt.title("Cumulative Time per Episode")
    plt.show()
    
    return remaining_time



def read_times_from_file():
    times = []
    with open("F:/Coding/breakout/double_dqn_2/output.txt", 'r') as file:
        for line in file:
            if "Episode time: " in line:
                words = line.split()
                index = words.index("Episode")
                time_str = words[index + 2]
                # print(time_str)
                try:
                    time = float(time_str[:-2])
                    times.append(time)
                except:
                    continue
                
    return times

times = read_times_from_file()
estimate_remaining_time(len(times), len(times[:-1500]), times[:-1500])

import numpy as np
import ast
import matplotlib.pyplot as plt

def estimate_remaining_time(total_episodes, times):
    # Fit a line to the times
    x = np.arange(len(times))
    coefficients = np.polyfit(x, times, 1)  # Fit a line (polynomial of degree 1)
    fitted_line = np.poly1d(coefficients)

    # Predict the times of the remaining episodes
    remaining_episodes = np.arange(len(times), total_episodes)
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

def estimate_score(total_episodes, scores):
    x = np.arange(len(scores))
    coefficients = np.polyfit(x, scores, 1)
    fitted_line = np.poly1d(coefficients)
    remaining_episodes = np.arange(len(scores), total_episodes)
    predicted_scores = fitted_line(remaining_episodes)
    print(np.mean(predicted_scores[-100:]))
# times = read_times_from_file()
times = [0.5, 1.8469130992889404, 1.4588637351989746, 2.0303847789764404, 2.7521350383758545, 1.7279329299926758, 1.0732841491699219, 1.247460126876831, 1.398866891860962, 1.8133347034454346, 2.0685527324676514]
# estimate_remaining_time(len(times), len(times[:-1500]), times[:-1500])
# estimate_remaining_time(5000, times)
with open("F:/Coding/breakout/double_dqn_3_5000/rewards.txt", "r") as f:
    data_str = f.read()
    scores = ast.literal_eval(data_str)
    estimate_score(2000, scores[2000:3000])

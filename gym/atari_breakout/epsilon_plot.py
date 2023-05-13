import math
import matplotlib.pyplot as plt
import numpy as np

def epsilon_decay(initial_epsilon, decay_rate, episode, num_episodes, minimum):
    epsilon = max(initial_epsilon * (decay_rate ** (episode/num_episodes)), minimum)
    # epsilon = initial_epsilon * (decay_rate ** episode)
    return epsilon

def plot_epsilon():
      
    initial_epsilon = 1.0
    decay_rate = 0.01 # 0.9998 for 20_000 episodes and 0.9996 for 10_000 episodes
    num_episodes = 20_000
    episodes = np.arange(num_episodes+1)

    epsilon_values = [epsilon_decay(initial_epsilon, decay_rate, episode, num_episodes) for episode in episodes]

    for i in range(0, num_episodes+1, num_episodes // 10):
            print((i,round(epsilon_values[i], 6)))
    # Plot the graph
    plt.plot(episodes, epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Function')
    plt.grid(True)
    plt.show()

def plot_learning_rate():
    initial_lr = 0.002
    decay_rate = 0.005 # 0.05 for 5000 episodes, 0.01 for 20_000 episodes
    num_episodes = 20_000
    minimum = 0.0001
    episodes = np.arange(num_episodes+1)

    epsilon_values = [epsilon_decay(initial_lr, decay_rate, episode, num_episodes, minimum) for episode in episodes]

    for i in range(0, num_episodes+1, num_episodes // 10):
            print((i,round(epsilon_values[i], 6)))
    # Plot the graph
    plt.plot(episodes, epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Learning Rate Decay Function')
    plt.grid(True)
    plt.show()

plot_learning_rate()
# plot_epsilon()

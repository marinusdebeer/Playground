import gymnasium as gym
import numpy as np
env = gym.make("Pendulum-v1", render_mode="human")
n_bins = 20  # number of bins to discretize the state space
Q = np.zeros((n_bins, n_bins, n_bins, 1))

alpha = 0.5  # learning rate
gamma = 0.95  # discount factor
epsilon = 0.1  # exploration rate
n_episodes = 1000

def discretize_state(observation):
    print(observation)
    cos_theta, sin_theta, theta_dot = observation
    cos_theta_bin = np.digitize(cos_theta, np.linspace(-1, 1, n_bins-1))
    sin_theta_bin = np.digitize(sin_theta, np.linspace(-1, 1, n_bins-1))
    theta_dot_bin = np.digitize(theta_dot, np.linspace(-8, 8, n_bins-1))
    return cos_theta_bin, sin_theta_bin, theta_dot_bin

for i in range(n_episodes):
    observation = env.reset()
    state = discretize_state(observation)
    done = False
    while not done:
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state[0], state[1], state[2], :])
        next_observation, reward, done, info = env.step([action])
        next_state = discretize_state(next_observation)
        Q[state[0], state[1], state[2], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], next_state[2], :]) - Q[state[0], state[1], state[2], action])
        state = next_state
    print("Episode:", i, "Reward:", reward)

env.close()

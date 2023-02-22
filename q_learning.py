import gym
import numpy as np
import matplotlib.pyplot as plt
env = None
training = False
if training:
    env = gym.make('MountainCar-v0')
else:
    env = gym.make('MountainCar-v0', render_mode="human")


# Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
EPISODES = 1001
STATS_EVERY = 10
FOLDER = 'qtables'

# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# Initialize Q-table
state_space_size = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
state_space_size = np.round(state_space_size, 0).astype(int) + 1
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size[0], state_space_size[1], action_space_size))
# q_table = np.load("chat/25000-qtable.npy")
# Convert continuous state space to discrete state space
def discretize_state(state):
    if(type(state) == tuple):
        state = np.array([state[0][0], state[0][1]])
    discrete_state = (state - env.observation_space.low) * np.array([10, 100])
    return np.round(discrete_state, 0).astype(int)
# Q-learning algorithm
for episode in range(1, EPISODES):
    episode_reward = 0
    state = env.reset()
    discrete_state = discretize_state(state)
    terminated = False
    timesteps = 0

    while not terminated:
        # Choose action
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state[0], discrete_state[1]])
        else:
            action = np.random.randint(0, action_space_size)
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        next_discrete_state = discretize_state(next_state)
        episode_reward += reward
        # Update Q-value
        old_q_value = q_table[discrete_state[0], discrete_state[1], action]
        next_max_q_value = np.max(q_table[next_discrete_state[0], next_discrete_state[1]])
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max_q_value)
        q_table[discrete_state[0], discrete_state[1], action] = new_q_value

        state = next_state
        discrete_state = next_discrete_state
        timesteps += 1

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    ep_rewards.append(episode_reward)
    if (not (episode % STATS_EVERY)) and training:
        # print(ep_rewards, STATS_EVERY)
        if len(ep_rewards) > 0:
            average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
            print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
            np.save(f"{FOLDER}/{episode}-qtable.npy", q_table)
    # Print results
    if episode % 100 == 0:
        print(f'Episode {episode} finished after {timesteps} timesteps. Epsilon: {epsilon:.2f}')

env.close()
if training:
    print(aggr_ep_rewards)
    np.save(f'{FOLDER}/chat_rewards.npy', aggr_ep_rewards)
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
    plt.legend(loc=4)
    plt.show()
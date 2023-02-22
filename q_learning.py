
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


# Define the Q-learning parameters
alpha = 0.1         # learning rate, how much past experience is valued
gamma = 0.9999        # discount factor, importance of future rewards
epsilon = 0.9997    # exploration, how much exploration to do
EPSILON_DECAY=0.9997
SAVE_EVERY = 100
#           0.99    0.993   0.997    0.9997      0.9985  0.9997     0.99985     0.99997
# 50        0.61    0.704                        0.928              0.993
# 100       0.366   0.495                        0.861              0.985
# 300       0.049   0.122   
# 500               0.03    0.223                0.472              0.928
# 1000     0.007    0.0008  0.050                0.223              0.861
# 10_000                              0.050      0.223    0.050     0.223
# 20_000                                         0.050              0.050
# 100_000                                                           0           0.050
rand = 0
choice = 0
# Train the Q-learning agent for a number of episodes
num_episodes = 10_000
training = False
if training:
    env = gym.make("MountainCar-v0")
else:
    env = gym.make("MountainCar-v0", render_mode="human")
Q_TABLES_PATH='q_tables/8600_-132-qtables.npy'
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print("env.action_space.n", env.action_space.n)
# Define the function for selecting an action using epsilon-greedy policy
def select_action(state, q_table):
    if np.random.random() > epsilon or not training:
        return np.argmax(q_table[state])
    else:
        # return np.random.choice(env.action_space.n)
        action = np.random.choice([0, 2])
        
        return action
def get_discrete_state(state):
    # print(type(state))
    # print(type(env.observation_space.low))
    if(type(state) == tuple):
      state = np.array([state[0][0], state[0][1]])
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))
# Define the function to save the Q-table to a file
def save_q_table(q_table, filename):
    np.save(filename, q_table)
# Try to initialize it that action 1 will happen so going straight
# Initialize the Q-table with zeros

if training:
    q_table = np.zeros((20,20,2))
    # q_table[:, :, 0] = -200000
    # q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
else:
    q_table = np.load(Q_TABLES_PATH)
# q_table[:, :, :, :, 1] = 1
episode_rewards = 0
for episode in range(num_episodes+1):
    # Initialize the state
    state = env.reset()
    if not training:
        q_table = np.load(Q_TABLES_PATH)
    episode_reward = 0
    # Reduce the exploration rate over time
    epsilon = pow(EPSILON_DECAY, episode)
    
    # Loop over the time steps in the episode
    while True:
        state = get_discrete_state(state)
        # Select an action using the epsilon-greedy policy
        action = select_action(state, q_table)
        if action == 1:
            action = 2
        # print(action)
        # Take the action and observe the next state and reward
        next_state, reward, done, truncated, info = env.step(action)
        if action == 2:
            action = 1
        # print(next_state)
        episode_reward += reward
        next_discrete_state = get_discrete_state(next_state)
        # print(next_discrete_state)
        # Update the Q-table using the Q-learning rule
        if training:
            q_table[state + (action,)] += alpha * (reward + gamma * np.max(q_table[next_discrete_state]) - q_table[state + (action,)])

        # Update the state
        state = next_state
        # time.sleep(0.2)
        if done:
            # print("rand", rand, " choice", choice, "epsilon", epsilon)
            rand = 0
            choice = 0
            break
    episode_rewards+=episode_reward
    
     # Check if this is the 10th episode and save the Q-table
    if episode % SAVE_EVERY == 0 and training:
        avg = round(episode_rewards/SAVE_EVERY)
        episode_rewards = 0
        np.save(f"q_tables/{episode}_{avg}-qtables.npy", q_table)
    # Print the total reward for the episode
    # print(f"Episode {episode}: Total reward = {episode_reward}")
    epsilon = pow(EPSILON_DECAY, episode)


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

alpha = 0.2         # learning rate, how much past experience is valued
gamma = 0.99        # discount factor, importance of future rewards
epsilon = 1    # exploration, how much exploration to do
initial_epsilon = 1
EPSILON_DECAY= 0.04 # 0.04 # 0.997
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
num_episodes = 5000
training = False
EXISTING_MODEL=False
if training:
    env = gym.make("MountainCar-v0")
else:
    env = gym.make("MountainCar-v0", render_mode="human")
Q_TABLES_PATH='qtables/5000_-135-qtables.npy'
STATES=25
DISCRETE_OS_SIZE = [STATES, STATES]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
def select_action(state, q_table):
    if np.random.random() > epsilon or not training:
        return np.argmax(q_table[state])
    else:
        action = np.random.choice([0, 2])
        
        return action
def get_discrete_state(state):
    if(type(state) == tuple):
      state = np.array([state[0][0], state[0][1]])
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))
def save_q_table(q_table, filename):
    np.save(filename, q_table)

if not training or EXISTING_MODEL:
    q_table = np.load(Q_TABLES_PATH)
else:
    q_table = np.zeros((STATES, STATES ,2))

episode_rewards = 0
for episode in range(num_episodes+1):
    state = env.reset()
    episode_reward = 0
    # epsilon = pow(EPSILON_DECAY, episode)
    epsilon = initial_epsilon * (EPSILON_DECAY ** (episode/num_episodes))
    while True:
        state = get_discrete_state(state)
        action = select_action(state, q_table)
        if action == 1:
            action = 2
        next_state, reward, done, truncated, info = env.step(action)
        reward = -1
        if action == 2:
            action = 1
        episode_reward += reward
        next_discrete_state = get_discrete_state(next_state)
        if training:
            q_table[state + (action,)] = round(q_table[state + (action,)] + alpha * (reward + gamma * np.max(q_table[next_discrete_state]) - q_table[state + (action,)]), 3)

        state = next_state
        if done:
            rand = 0
            choice = 0
            break
    episode_rewards+=episode_reward
    
    if episode % SAVE_EVERY == 0 and training:
        avg = round(episode_rewards/SAVE_EVERY)
        episode_rewards = 0
        np.save(f"qtables/{episode}_{avg}-qtables.npy", q_table)
        print(f"Episode: {episode}, average reward: {avg}, epsilon: {epsilon}")
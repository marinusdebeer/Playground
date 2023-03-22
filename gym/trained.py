import gym
import numpy as np

env = gym.make('MountainCar-v0', render_mode="human")
FOLDER = 'qtables'
# Load Q-table
q_table = np.load(f'{FOLDER}/990-qtable.npy')

# Convert continuous state space to discrete state space
def discretize_state(state):
    if(type(state) == tuple):
        state = np.array([state[0][0], state[0][1]])
    discrete_state = (state - env.observation_space.low) * np.array([10, 100])
    return np.round(discrete_state, 0).astype(int)

# Make predictions
state = env.reset()
discrete_state = discretize_state(state)
terminated = False
timesteps = 0

while not terminated:
    action = np.argmax(q_table[discrete_state[0], discrete_state[1]])
    next_state, reward, terminated, truncated, info = env.step(action)
    print('reward: ',reward)
    next_discrete_state = discretize_state(next_state)
    state = next_state
    discrete_state = next_discrete_state
    timesteps += 1

env.close()

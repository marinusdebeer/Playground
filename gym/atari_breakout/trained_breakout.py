import numpy as np
import gymnasium as gym
import time
import random
from tqdm import tqdm
# from rl_breakout import DQNAgent
from rainbow_breakout import RainbowAgent
# Create a new instance of the DQNAgent class with the same architecture as the trained model
env = gym.make('Breakout-v4', render_mode="human")
# env = gym.make('Breakout-v4')
agent = RainbowAgent()
# Build the model by calling it with an input shape
dummy_input = np.zeros((1, 84, 64, 4))  # Replace with the appropriate input shape
ACTIONS = [0, 2, 3]
_ = agent.q_network(dummy_input)
# Load the saved weights from a specific episode
# agent.q_network.load_weights("rainbow_models/good_models/model_10000_1.h5")
agent.q_network.load_weights("G:/Coding/breakout/testing_prioritized/per_32_5_000_h/models/model_5000.h5")
# agent.model.load_weights("models/model_weights_episode_4380.h5")
# Function to play the game using the trained model
def play_game(agent, num_episodes=5):
    print("game start")
    highscore = 0
    steps = 0
    for episode in tqdm(range(1, num_episodes + 1), ascii=True, unit='episodes'):
        print(f"Episode {episode}")
        lives = 5
        state = env.reset()
        for _ in range(4):
          processed_state, proc_time = agent.preprocess_state(state)
          agent.frame_buffer.append(processed_state)
        state = np.stack(agent.frame_buffer, axis=-1)
        done = False
        episode_reward = 0
        env.step(1)
        while not done:
          # state = np.expand_dims(state, axis=0)
          # print(state.shape)
          q_values = agent.q_network(np.array([state], dtype=np.float32))
          action = ACTIONS[np.argmax(q_values.numpy())]
          # q_values = agent.q_network.predict(state, verbose=0)[0]
          # action = np.argmax(q_values)

          # action += 2
          # print(q_values, action)
          next_state, reward, done, bruh, info = env.step(action)
          # print(info, bruh)
          steps += 1
          # if steps % 100 == 0:
            #  agent.show_frame(agent.preprocess_state(next_state))
          processed_state, proc_time = agent.preprocess_state(next_state)
          agent.frame_buffer.append(processed_state)
          next_state = np.stack(agent.frame_buffer, axis=-1)
          state = next_state

          episode_reward += reward
          if(info["lives"] < lives):
            lives = info["lives"]
            env.step(1)
            if highscore < episode_reward:
              highscore = episode_reward
        
        print(f"Episode {episode}, Reward: {episode_reward}, Highscore: {highscore}")
    
    env.close()

print("play game")
# Play the game using the trained model
play_game(agent, num_episodes=5)

import numpy as np
import gymnasium as gym
import time
import random
from tqdm import tqdm
from rl_breakout import DQNAgent
# Create a new instance of the DQNAgent class with the same architecture as the trained model
env = gym.make('Breakout-v4', render_mode="human")
# env = gym.make('Breakout-v4')
model_truncate = 0
agent = DQNAgent(num_actions=3, model_truncate=model_truncate, trained=True)
# Build the model by calling it with an input shape
dummy_input = np.zeros((1, 84-model_truncate, 84, 4))  # Replace with the appropriate input shape
_ = agent.model(dummy_input)
# Load the saved weights from a specific episode
agent.model.load_weights("models_3_actions/model_weights_episode_1000.h5")
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
          agent.frame_buffer.append(agent.preprocess_state(state))
        state = np.stack(agent.frame_buffer, axis=-1).astype(np.float32) 
        done = False
        episode_reward = 0
        env.step(1)
        while not done:
          state = np.expand_dims(state[model_truncate:], axis=0)
          # print(state.shape)
          q_values = agent.model.predict(state, verbose=0)[0]
          action = np.argmax(q_values)
          if action != 0:
            action += 1
          # action += 2
          # print(q_values, action)
          next_state, reward, done, _, info = env.step(action)
          steps += 1
          # if steps % 100 == 0:
            #  agent.show_frame(agent.preprocess_state(next_state))
          agent.frame_buffer.append(agent.preprocess_state(next_state))
          next_state = np.stack(agent.frame_buffer, axis=-1).astype(np.float32) 
          state = next_state

          episode_reward += reward
          if(info["lives"] < lives):
            lives = info["lives"]
            env.step(1)
            if highscore < episode_reward:
              highscore = episode_reward
        
        print(f"Episode {episode}, Reward: {episode_reward}, Highscore: {highscore}")
    
    env.close()

# Play the game using the trained model
play_game(agent, num_episodes=5)

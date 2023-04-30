import numpy as np
import gymnasium as gym
import time
import random

from rl_breakout import DQNAgent
# Create a new instance of the DQNAgent class with the same architecture as the trained model
env = gym.make('BreakoutDeterministic-v4', render_mode="human", full_action_space=True)
num_actions = 2
agent = DQNAgent()
# Build the model by calling it with an input shape
dummy_input = np.zeros((1, 84, 84, 4))  # Replace with the appropriate input shape
_ = agent.model(dummy_input)
# Load the saved weights from a specific episode
agent.model.load_weights("models/model_weights_episode_670.h5")
# Function to play the game using the trained model
def play_game(agent, num_episodes=5, render=True, sleep_time=0.05):
    fire = True
    lives = 5
    for episode in range(num_episodes):
        state = env.reset()
        for _ in range(4):
            agent.frame_buffer.append(agent.preprocess_state(state))
        state = np.stack(agent.frame_buffer, axis=-1).astype(np.float32) 
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(sleep_time)
            
            # Use the trained model to choose actions instead of exploration
            temp = agent.model.predict(np.expand_dims(state, axis=0))[0]
            action = np.argmax(agent.model.predict(np.expand_dims(state, axis=0))[0]) + 2
            print(temp)
            if fire:
              action = 8
            next_state, reward, done, _, info = env.step(16)
            agent.frame_buffer.append(agent.preprocess_state(next_state))
            next_state = np.stack(agent.frame_buffer, axis=-1).astype(np.float32) 
            state = next_state

            episode_reward += reward
            if(info["lives"] < lives):
              lives = info["lives"]
              fire = True
            else:
              fire = False
        
        print(f"Episode {episode}, Reward: {episode_reward}")
    
    env.close()

# Play the game using the trained model
play_game(agent, num_episodes=5, render=True, sleep_time=0.05)

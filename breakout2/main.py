import numpy as np
import torch
from breakout_env import BreakoutEnv
from dqn_agent import DQNAgent

def train(num_episodes=500, render=False):
    env = BreakoutEnv(render_mode=render)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim=5, action_dim=3, device=device)

    scores = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
        agent.decay_epsilon()
        scores.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    env.close()
    return scores

if __name__ == "__main__":
    train(num_episodes=500, render=False)

import os
import numpy as np
import torch
from breakout_env import BreakoutEnv
from dqn_agent import DQNAgent

# Set mode here: 'train' for training with evaluation statistics,
# or 'demo' to load a checkpoint and watch the agent play.
MODE = 'demo'  # Change to 'demo' for demonstration

# When in demo mode, specify the checkpoint file to load.
DEMO_CHECKPOINT = "checkpoints/breakout_dqn_episode_5000.pt"
MAX_EPISODES = 5000

def train(num_episodes=5000, render=False, save_interval=50):
    env = BreakoutEnv(render_mode=render)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim=5, action_dim=3, device=device, num_episodes=MAX_EPISODES)

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

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
        agent.decay_epsilon(episode=episode)
        scores.append(total_reward)

        # Print summary every 10 episodes (all info in one line)
        if (episode + 1) % 10 == 0:
            last_10 = scores[-10:] if len(scores) >= 10 else scores
            last_50 = scores[-50:] if len(scores) >= 50 else scores
            last_100 = scores[-100:] if len(scores) >= 100 else scores
            avg10 = np.mean(last_10)
            avg50 = np.mean(last_50)
            avg100 = np.mean(last_100)
            total_avg = np.mean(scores)
            print(f"Ep:{episode+1:3d}/{num_episodes} | Reward: {total_reward:6.2f} | Avg10: {avg10:6.2f} | Avg50: {avg50:6.2f} | Avg100: {avg100:6.2f} | TotAvg: {total_avg:6.2f} | Epsilon: {agent.epsilon:5.2f}")

        # Save the model every save_interval episodes in the checkpoints directory
        if (episode + 1) % save_interval == 0:
            save_path = os.path.join("checkpoints", f"breakout_dqn_episode_{episode+1}.pt")
            torch.save(agent.policy_net.state_dict(), save_path)
            print(f" --> Saved model checkpoint to {save_path}")
    env.close()
    return agent, scores

def evaluate_agent(agent, num_episodes=10, render=False):
    env = BreakoutEnv(render_mode=render)
    total_rewards = []
    # Use a greedy policy during evaluation (temporarily set epsilon to 0)
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    env.close()
    agent.epsilon = saved_epsilon
    avg_reward = np.mean(total_rewards)
    return avg_reward, total_rewards

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODE == "train":
        # Train the agent without rendering.
        trained_agent, training_scores = train(num_episodes=MAX_EPISODES, render=False, save_interval=50)
        print(f"\nTotal Average Reward over {len(training_scores)} Training Episodes: {np.mean(training_scores):.2f}")
        # Evaluate the agent over 10 and 50 episodes.
        avg_reward_10, rewards_10 = evaluate_agent(trained_agent, num_episodes=10, render=False)
        avg_reward_50, rewards_50 = evaluate_agent(trained_agent, num_episodes=50, render=False)
        print("\nEvaluation:")
        print(f"  Average Reward over 10 Evaluation Episodes: {avg_reward_10:.2f}")
        print(f"  Average Reward over 50 Evaluation Episodes: {avg_reward_50:.2f}")
    elif MODE == "demo":
        # In demo mode, load a checkpoint of your choosing.
        agent = DQNAgent(state_dim=5, action_dim=3, device=device, num_episodes=MAX_EPISODES)
        if os.path.exists(DEMO_CHECKPOINT):
            checkpoint = torch.load(DEMO_CHECKPOINT, map_location=device)
            agent.policy_net.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {DEMO_CHECKPOINT}")
        else:
            print(f"Checkpoint {DEMO_CHECKPOINT} not found. Exiting.")
            exit(1)
        print("\nRunning demo episodes with rendering enabled...")
        # Run demo episodes continuously.
        while True:
            avg_reward, rewards = evaluate_agent(agent, num_episodes=1, render=True)
            print(f"Demo Episode Reward: {rewards[0]:.2f}")

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create the Atari Breakout environment
env = gym.make("BreakoutNoFrameskip-v4")
env = DummyVecEnv([lambda: env])

# Define the PPO algorithm
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./breakout_tensorboard/",
)

# Train the model
model.learn(total_timesteps=1_000)

# Save the trained model
model.save("breakout_ppo")

# Load the trained model
model = PPO.load("breakout_ppo")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Enjoy the trained agent
# obs = env.reset()
# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, _, done, _ = env.step(action)
#     env.render()

#     if done:
#         obs = env.reset()

env.close()

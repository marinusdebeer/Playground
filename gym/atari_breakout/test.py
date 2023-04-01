import gymnasium as gym
# env = gym.make("ALE/Breakout-v5", render_mode="human")
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
from stable_baselines3 import PPO

model = PPO.load("breakout_ppo")

# Enjoy the trained agent
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    env.render()

    if done:
        obs = env.reset()
# Import the libraries
import os 

import gymnasium as gym
import time
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from huggingface_sb3 import load_from_hub, push_to_hub

# Load the model
checkpoint = load_from_hub("ThomasSimonini/ppo-BreakoutNoFrameskip-v4", "ppo-BreakoutNoFrameskip-v4.zip")

# Because we using 3.7 on Colab and this agent was trained with 3.8 to avoid Pickle errors:
custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

model= PPO.load(checkpoint, custom_objects=custom_objects)

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1)
env = VecFrameStack(env, n_stack=4)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render("human")
    frame = env.render()
    frame = cv2.resize(frame, (800, 600))  # Resize the frame to 800x600
    cv2.imshow('Breakout', frame)
    cv2.waitKey(1)
    time.sleep(0.02)
